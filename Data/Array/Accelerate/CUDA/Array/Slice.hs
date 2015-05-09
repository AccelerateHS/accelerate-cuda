{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE ImpredicativeTypes    #-}

-- |
-- Module      : Data.Array.Accelerate.Array.Slice
--
-- Maintainer  : Frederik Meisner Madsen <fmma@di.ku.dk>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Slice (

  CopyArgs(..),
  Memcpy2Dargs(..),
  Permut(..),
  P(..),

  copyArgs,
  reifyP,
  shapeP,

) where

import GHC.Base                                         ( quotInt )
import Data.Array.Accelerate                            ( Exp, lift1 )
import Data.Array.Accelerate.Array.Sugar                ( Shape, Z(..), (:.)(..), DIM3, DIM5, DIM7, DIM9, shapeToList, listToShape )
import Data.Array.Accelerate.Array.Representation       ( SliceIndex(..), size )

-- Instrument a copy from host to device of all slices in the range
-- 'i' (inclusive) to 'j' (exclusive), the slicing strategy is
-- provided by the given SliceIndex.
copyArgs  :: SliceIndex slix sl co dim -> dim -> Int -> Int -> [CopyArgs]
copyArgs sl dim i j =
  let sls = map normaliseSl (shapeToSl (massageShape sl dim) i j)
      offsets = scanl (+) 0 (map dstSize sls)
  in zipWith f sls offsets
  where
    f sl o = CopyArgs { memcpy2Dargs = copy sl
                      , offset = o
                      , permutation = slicePermut sl
                      }

afDstSize :: (A, F) -> Int
afDstSize (a, _) = a
faDstSize :: (F, A) -> Int
faDstSize (_, a) = a
dstSize :: SlN -> Int
dstSize (ac, r, (afs, as, fas)) = productOdd ac * rangeLen r * product (map afDstSize afs) * as * product (map faDstSize fas)


-- The arguments for one call to memcpy2D, along with an offset into
-- the destination array.
data Memcpy2Dargs = Memcpy2Dargs
        { width
        , height
        , srcPitch
        , srcX
        , srcY
        , dstPitch
        , dstX
        , dstY
        , srcRows        -- Not exported (pitch in the Y-axis)
        , dstRows :: Int -- Not exported
        }
  deriving Show

-- CopyArgs contain the arguments to a list of memcpy2dcalls. Each
-- call should be applied to a region on the device, offset in number
-- of elements by the 'offset' field. To obtain the correct
-- permutation of elements in each region, the 'permutation' field
-- contains shape and permutation information that should be applied
-- after copying. The permutation moves all iteration dimensions to
-- the outermost dimensions, leaving the slice shape as the innermost.
data CopyArgs = CopyArgs
  { memcpy2Dargs :: [Memcpy2Dargs]
  , offset       :: Int
  , permutation  :: Permut
  }

-- A pair, used to denote that "a ranges over 0..b-1".
data In a b = a `In` b

instance (Show a, Show b) => Show (In a b) where
  show (i `In` n) = show i ++ " in " ++ show n

idx :: In a b -> a
idx (i `In` _) = i

dim :: In a b -> b
dim (_ `In` n) = n

-- Dimension transfer types:
type A = Int               -- all elements in dimension
type F = Int        `In` A -- fixed element in dimension
type R = (Int, Int) `In` A -- fixed range of elements in dimension

rangeStart :: R -> Int
rangeStart = fst . idx

rangeEnd :: R -> Int
rangeEnd   = snd . idx

rangeLen :: R -> Int
rangeLen r = rangeEnd r - rangeStart r

-- Some lists:
-- (ab)*
type EvenList a b = [(a, b)]

-- a(ba)*
type OddList a b = (a, EvenList b a)

-- (ab)*a(ba)*
type PivotedOddList a b = (EvenList a b, a, EvenList b a)

-- product defined on OddList.
productOdd :: Num a => OddList a a -> a
productOdd (a0, as) = foldl (\ a (b,c) -> a*b*c) a0 as


shift :: b -> EvenList a b -> a -> EvenList b a
shift a [] b = [(a, b)]
shift a ((b0,a0) : bas) b = (a,b0) : shift a0 bas b

shiftlPivot :: Int -> OddList a b -> PivotedOddList a b
shiftlPivot 0 (a0, bas) = ([], a0, bas)
shiftlPivot n (a0, bas) =
  let (bas0, (b,a):bas1) = (take (n-1) bas, drop (n-1) bas)
  in (shift a0 bas0 b, a, bas1)

indexOfMax :: Ord a => OddList a b -> Int
indexOfMax (a0, bas) = snd $ foldl (\ (a0, i0) ((_, a), i) -> if a > a0 then (a, i) else (a0, i0)) (a0, 0) (bas `zip` [1..])

-- Pivot an odd-length list around the maximum off the odd-positioned
-- elements.
pivotMax :: Ord a => OddList a b -> PivotedOddList a b
pivotMax xs = shiftlPivot (indexOfMax xs) xs

-- Dimension of an array, decorated with slice indexing.
-- Innermost-to-outermost ~ left-to-right.
type Sl = ( OddList A A   -- Innermost contiguous region.
          , R             -- Range.
          , OddList A F)  -- Outermost Index-slice.

-- Same as Sl, but where the outermost dimensions are pivoted around a
-- distinguished A-dimension (FixedAll) that has been chosen as a
-- basis for partitioning, resulting in fewest possible calls to
-- memcpy2D.
type SlN = (OddList A A, R, PivotedOddList A F)

-- Convert a slice index description to a list of arguments for
-- multiple calls to memcpy2D. In most cases, only one call is needed,
-- but in the general case, it is not possible to describe the memory
-- transfer with a single call.
--
-- Number of calls = product of all a's in afs and fas. That is why
-- the largest dimension has been selected as the pivot.
copy :: SlN -> [Memcpy2Dargs]
copy (ac, r, (afs, as, fas)) =
  let
     mem0 = Memcpy2Dargs
        { width    = productOdd ac * rangeLen r
        , height   = as
        , srcPitch = productOdd ac * dim r
        , srcX     = productOdd ac * rangeStart r
        , srcY     = 0
        , dstPitch = productOdd ac * rangeLen r
        , dstX     = 0
        , dstY     = 0
        , srcRows  = as
        , dstRows  = as
        }
  in foldl outerMats (foldl innerMats [mem0] afs) fas
  where
    innerMats :: [Memcpy2Dargs] -> (A, F) -> [Memcpy2Dargs]
    innerMats mems (a, f) =
      [ let
          sp = srcPitch mem
          dp = dstPitch mem
        in
        mem { srcX     = srcX mem + sp * (idx f * a + i)
            , dstX     = dstX mem + dp * i
            , srcPitch = sp * a * dim f
            , dstPitch = dp * a
            }
      | mem <- mems
      , i   <- [0 .. a - 1]
      ]

    outerMats :: [Memcpy2Dargs] -> (F, A) -> [Memcpy2Dargs]
    outerMats mems (f, a) =
      [ let
          sr = srcRows mem
          dr = dstRows mem
        in
        mem { srcY    = srcY mem + sr * (idx f + i * dim f)
            , dstY    = dstY mem + dr * i
            , srcRows = sr * a * dim f
            , dstRows = dr * a
            }
      | mem <- mems
      , i   <- [0 .. a - 1]
      ]

-- Convert a shape to a the form we will be working
-- with. Even-positioned elments represents SliceAll
-- dimensions. Odd-positioned elements represents SliceFixed
-- dimensions. Consecutive SliceAll (SliceFixed) dimensions are
-- collapsed to a single dimension, since finer shape details are
-- irrelevant.
massageShape :: SliceIndex slix sl co dim -> dim -> OddList A A
massageShape = goContiguous 1
  where
    goContiguous :: Int -> SliceIndex slix sl co dim -> dim -> OddList A A
    goContiguous a SliceNil () = (a, [])
    goContiguous a (SliceAll   sl) (sh, sz) = goContiguous (a * sz) sl sh
    goContiguous a (SliceFixed sl) (sh, sz) = (a, goFixed sz sl sh)

    goFixed :: Int -> SliceIndex slix sl co dim -> dim -> EvenList A A
    goFixed i SliceNil () = [(i, 1)]
    goFixed i (SliceAll sl)   (sh, sz)
      | sz == 1   = goFixed sz sl sh
      | otherwise = goAll sz i sl sh
    goFixed i (SliceFixed sl) (sh, sz) = goFixed (i * sz) sl sh

    goAll :: Int -> Int -> SliceIndex slix sl co dim -> dim -> EvenList A A
    goAll a i SliceNil () = [(i, a)]
    goAll a i (SliceAll   sl) (sh, sz) = goAll (a * sz) i sl sh
    goAll a i (SliceFixed sl) (sh, sz)
      | sz == 1   = goAll a i sl sh
      | otherwise = (i, a) : goFixed sz sl sh

-- Compute the slicing that determine a range of slices.
shapeToSl :: OddList A A -> Int -> Int -> [Sl]
shapeToSl (a0, dim) from to = go (reverse xs) from to
  where
    -- Zip the dimension with a scan of the fixed dimensions.
    xs = (init . scanl (*) 1 . map fst $ dim) `zip` dim

    go :: [(Int, (Int, Int))] -> Int -> Int -> [Sl]
    go ((f', (f, a)) : xs) i j
      | i >= j = []
      | otherwise
      = let (i0, i') = deltaForm f' i
            (j0, j') = deltaForm f' j

            -- Prefix the index-slice.
            prefix i = map ((\ (inner, r, (as, ys)) -> (inner, r, (as, ys ++ [(i `In` f, a)]))))
        in if i0 == j0
           then prefix i0 (go xs i' j')
           else
             -- Left dangling region.
             concat [prefix i0 (go xs i' f') | i' > 0] ++

             -- Middle complete region.
             [( contiguous xs               -- Copy everything in remaining dims.
              , (i0 + signum i', j0) `In` f -- Copy the range.
              , (a, []))                    -- Initial index-slice.
             | (i0 + signum i') < j0] ++

             -- Right dangling region.
             concat [prefix j0 (go xs 0  j') | j' > 0]
    go [] _ _ = []

    contiguous :: [(Int, (Int, Int))] -> OddList A A
    contiguous xs = (a0, reverse (map snd xs))

    -- Convert an index x to (x0, x') where x = x0 * d + x'
    deltaForm :: Int -> Int -> (Int, Int)
    deltaForm d x = let x0 = x `quotInt` d
                    in (x0, x - x0 * d)

normaliseSl :: Sl -> SlN
normaliseSl (inner, r, outer) = (inner, r, pivotMax outer) -- pivotMax


-- Once copied, the data needs to be permuted, in order to conform to
-- the specification - Each slice should occupy one contigious region
-- of memory on the device, and the region of a slice should be
-- positioned right after the region of the previous slice. After a
-- copy, the elements are ordered by linear index of the source shape.
-- After permutation, they are ordered by linear index of the
-- iteration shape, followed by the linear index of the slice shape.
data Permut where
  Permut :: Shape sh
         => sh   -- Shape of copied elements
         -> P sh -- Permutation
         -> Permut

instance Show Permut where
  show (Permut sh _) = show sh

data P sh where
  P3 :: P DIM3
  P5 :: P DIM5
  P7 :: P DIM7
  P9 :: P DIM9

-- Reify a permutation
reifyP :: P sh -> (forall x. [x] -> [x], forall x. [x] -> [x])
reifyP p =
  case p of
    P3 ->
      ( \ [as,r,ac] -> [r,as,ac]
      , \ [r,as,ac] -> [as,r,ac])
    P5 ->
      ( \ [as,r,a0,f0,ac] -> [r,f0,as,a0,ac]
      , \ [r,f0,as,a0,ac] -> [as,r,a0,f0,ac])
    P7 ->
      ( \ [as,r,a1,f1,a0,f0,ac] -> [r,f1,f0,as,a1,a0,ac]
      , \ [r,f1,f0,as,a1,a0,ac] -> [as,r,a1,f1,a0,f0,ac])
    P9 ->
      ( \ [as,r,a2,f2,a1,f1,a0,f0,ac] -> [r,f2,f1,f0,as,a2,a1,a0,ac]
      , \ [r,f2,f1,f0,as,a2,a1,a0,ac] -> [as,r,a2,f2,a1,f1,a0,f0,ac])
    {-
    P3 ->
      ( \ [ac, r, as] -> [ac, as, r]
      , \ [ac, as, r] -> [ac, r, as])
    P5 ->
      ( \ [ac, f0, a0, r, as] -> [ac, a0, as, f0, r]
      , \ [ac, a0, as, f0, r] -> [ac, f0, a0, r, as])
    P7 ->
      ( \ [ac, f0, a0, f1, a1, r, as] -> [ac, a0, a1, as, f0, f1, r]
      , \ [ac, a0, a1, as, f0, f1, r] -> [ac, f0, a0, f1, a1, r, as])
    P9 ->
      ( \ [ac, f0, a0, f1, a1, f2, a2, r, as] -> [ac, a0, a1, a2, as, f0, f1, f2, r]
      , \ [ac, a0, a1, a2, as, f0, f1, f2, r] -> [ac, f0, a0, f1, a1, f2, a2, r, as])
-}
slicePermut :: SlN -> Permut
slicePermut ((ac0, ac), r, (afs, as, fas)) =
  let
    sh = Z :. product (map afDstSize afs) * as * product (map faDstSize fas) :. rangeLen r
  in
  case ac of
    [] ->
      Permut
        (sh:.ac0)
        P3
    [(f0,a0)] ->
      Permut
        (sh:.a0:.f0:.ac0)
        P5
    [(f0,a0),(f1,a1)] ->
      Permut
        (sh:.a1:.f1:.a0:.f0:.ac0)
        P7
    [(f0,a0),(f1,a1),(f2,a2)] ->
      Permut
        (sh:.a2:.f2:.a1:.f1:.a0:.f0:.ac0)
        P9
    _ -> error "Too many dimensions"

shapeP :: Shape sh => P sh -> sh -> sh
shapeP p sh =
  let (fw, _) = reifyP p
  in listToShape $ reverse $ fw $ reverse $ shapeToList sh
