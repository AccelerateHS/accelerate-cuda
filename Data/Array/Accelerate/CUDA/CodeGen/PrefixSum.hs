{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ImpredicativeTypes  #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.PrefixSum
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module  Data.Array.Accelerate.CUDA.CodeGen.PrefixSum (

  -- skeletons
  mkScanl, mkScanl1, mkScanl',
  mkScanr, mkScanr1, mkScanr',

) where

import Data.Maybe
import Foreign.CUDA.Analysis
import Language.C.Quote.CUDA
import qualified Language.C.Syntax                      as C

import Data.Array.Accelerate.Array.Sugar                ( Vector, Scalar, Elt, DIM1 )
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base


-- Wrappers
-- --------

mkScanl, mkScanr
    :: Elt e
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> CUExp aenv e
    -> CUDelayedAcc aenv DIM1 e
    -> [CUTranslSkel aenv (Vector e)]
mkScanl dev aenv f z a =
  [ mkScan    L dev aenv f (Just z) a
  , mkScanUp1 L dev aenv f a
  , mkScanUp2 L dev aenv f (Just z) ]

mkScanr dev aenv f z a =
  [ mkScan    R dev aenv f (Just z) a
  , mkScanUp1 R dev aenv f a
  , mkScanUp2 R dev aenv f (Just z) ]

mkScanl1, mkScanr1
    :: Elt e
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> CUDelayedAcc aenv DIM1 e
    -> [CUTranslSkel aenv (Vector e)]
mkScanl1 dev aenv f a =
  [ mkScan    L dev aenv f Nothing a
  , mkScanUp1 L dev aenv f a
  , mkScanUp2 L dev aenv f Nothing ]

mkScanr1 dev aenv f a =
  [ mkScan    R dev aenv f Nothing a
  , mkScanUp1 R dev aenv f a
  , mkScanUp2 R dev aenv f Nothing ]

mkScanl', mkScanr'
    :: Elt e
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> CUExp aenv e
    -> CUDelayedAcc aenv DIM1 e
    -> [CUTranslSkel aenv (Vector e, Scalar e)]
mkScanl' dev aenv f z = map cast . mkScanl dev aenv f z
mkScanr' dev aenv f z = map cast . mkScanr dev aenv f z

cast :: CUTranslSkel aenv a -> CUTranslSkel aenv b
cast (CUTranslSkel entry code) = CUTranslSkel entry code


-- Core implementation
-- -------------------

data Direction = L | R
  deriving Eq

instance Show Direction where
  show L = "l"
  show R = "r"


-- [OVERVIEW]
--
-- Data.List-style exclusive scan, with the additional restriction that the
-- first argument needs to be an /associative/ function to enable efficient
-- parallel implementation. The initial value may be arbitrary.
--
-- scanl :: Elt a
--       => (Exp a -> Exp a -> Exp a)
--       -> Exp a
--       -> Acc (Vector a)
--       -> Acc (Vector a)
--
-- > scanl (+) 10 (use xs)
-- >   where
-- >     xs = fromList (Z:.10) (cycle [1])
-- >
-- > ==> Array (Z:.11) [10,11,12,13,14,15,16,17,18,19,20]
--
-- Data.List-style inclusive scan without an initial value
--
-- scanl1 :: Elt a
--        => (Exp a -> Exp a -> Exp a)
--        -> Acc (Vector a)
--        -> Acc (Vector a)
--
-- > scanl1 (+) (use xs)
-- >   where
-- >     xs = fromList (Z:.10) (cycle [1])
-- >
-- > ==> Array (Z:.10) [1,2,3,4,5,6,7,8,9,10]
--
-- Variant of 'scanl' where the final result is returned separately.
--
-- scanl' :: Elt a
--        => (Exp a -> Exp a -> Exp a)
--        -> Exp a
--        -> Acc (Vector a)
--        -> (Acc (Vector a), Acc (Scalar a))
--
-- Denotationally, we have:
--
-- > scanl' f z xs = (init res, last res)
-- >   where
-- >     res = scanl f z xs
--
--
-- [IMPLEMENTATION]
--
-- This code handles all the above cases, in both left and right-handed
-- variants. This is the _downsweep_ phase to a multi-block scan procedure.
-- We require a work distribution such that there is a _single_ thread block for
-- each interval. For multi-block scans, we have an array of interval sums that
-- are used to determine the carry-in value from the previous interval. Note
-- that 'argBlk' will not be accessed by a single-block scan, so may be null.
--
-- We require some pointer manipulation from the calling code in order to
-- support all types of scans:
--
--   * scanl          : argSum should point to the last position of argOut
--   * scanr          : argSum should be the start of argOut, argOut should be incremented by one
--   * scanl1, scanr1 : no change (argSum is required, even though it will not be used Haskell-side)
--   * scanl', scanr' : no change
--
mkScan :: forall aenv e. Elt e
       => Direction
       -> DeviceProperties
       -> Gamma aenv
       -> CUFun2 aenv (e -> e -> e)
       -> Maybe (CUExp aenv e)
       -> CUDelayedAcc aenv DIM1 e
       -> CUTranslSkel aenv (Vector e)
mkScan dir dev aenv fun@(CUFun2 _ _ combine) mseed (CUDelayed (CUExp shIn) _ (CUFun1 _ get)) =
  CUTranslSkel scan [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    $id:scan
    (
        $params:argIn,
        $params:argOut,
        $params:argBlk,
        $params:(tail argSum)           // just the pointers, no shape information
    )
    {
        $decls:smem
        $decls:declx
        $decls:decly
        $decls:declz
        $items:(sh .=. shIn)

        const int shapeSize     = $exp:(csize sh);
        const int intervalSize  = (shapeSize + gridDim.x - 1) / gridDim.x;

        /*
         * Read in previous result partial sum. We store the carry value in
         * temporary value 'z' and read new values from the input array into
         * 'x', since 'scanBlock' will store its results into 'y' on completion.
         */
        int carryIn = 0;

        if ( threadIdx.x == 0 ) {
            $stm:initialise
        }

        const int start         = blockIdx.x * intervalSize;
        const int end           = min(start + intervalSize, shapeSize);
        const int numElements   = end - start;
              int seg;

        for ( seg = threadIdx.x
            ; seg < numElements
            ; seg += blockDim.x )
        {
            const int ix = $exp:firstIndex;

            /*
             * Generate the next set of values
             */
            $items:(x .=. get ix)

            /*
             * Carry in the result from the privous segment
             */
            if ( $exp:carryIn ) {
                $items:(x .=. combine z x)
            }

            /*
             * Store our input into shared memory and perform a cooperative
             * inclusive left scan.
             */
            $items:(sdata "threadIdx.x" .=. x)
            __syncthreads();

            $items:(scanBlock dev fun x y sdata Nothing)

            /*
             * Exclusive scans write the result of the previous thread to global
             * memory. The first thread must reinstate the carry-in value which
             * is the result of the last thread from the previous interval, or
             * the carry-in/seed value for multi-block scans.
             */
            if ( $exp:(cbool (isJust mseed)) ) {
                if ( threadIdx.x == 0 ) {
                    $items:(x .=. z)
                } else {
                    $items:(x .=. sdata "threadIdx.x - 1")
                }
            }
            $items:(setOut "ix" .=. x)

            /*
             * Carry the final result of this block through the set 'z'. If this
             * is the final interval, this is the value to write out as the
             * reduction result
             */
            if ( threadIdx.x == 0 ) {
                const int last = min(numElements - seg, blockDim.x) - 1;
                $items:(z .=. sdata "last")
            }
            $items:setCarry
        }

        /*
         * Finally, exclusive scans set the overall scan result (reduction value)
         */
        $items:setFinal
    }
  |]
  where
    scan                        = "scan" ++ show dir ++ maybe "1" (const []) mseed
    (texIn, argIn)              = environment dev aenv
    (argOut, _, setOut)         = writeArray "Out" (undefined :: Vector e)
    (argSum, _, totalSum)       = writeArray "Sum" (undefined :: Vector e)
    (argBlk, _, blkSum)         = writeArray "Blk" (undefined :: Vector e)
    (_, x, declx)               = locals "x" (undefined :: e)
    (_, y, decly)               = locals "y" (undefined :: e)
    (_, z, declz)               = locals "z" (undefined :: e)
    (sh, _, _)                  = locals "sh" (undefined :: DIM1)
    (smem, sdata)               = shared (undefined :: e) "sdata" [cexp| blockDim.x |] Nothing
    ix                          = [cvar "ix"]
    setSum                      = totalSum "0"

    -- depending on whether we are inclusive/exclusive scans
    setCarry
      | isNothing mseed         = [[citem| carryIn = 1; |]]
      | otherwise               = []

    setFinal
      | isNothing mseed         = []
      | otherwise               = [[citem| if ( threadIdx.x == 0 && blockIdx.x == $id:lastBlock ) {
                                               $items:(setSum .=. z)
                                           } |]]

    -- accessing neighbouring blocks
    firstBlock          = if dir == L then "0" else "gridDim.x - 1"
    lastBlock           = if dir == R then "0" else "gridDim.x - 1"
    prevBlock           = if dir == L then "blockIdx.x - 1" else "blockIdx.x + 1"

    firstIndex
      | dir == L        = [cexp| start + seg |]
      | otherwise       = [cexp| end - seg - 1 |]

    carryIn
      | isJust mseed    = [cexp| threadIdx.x == 0 |]
      | otherwise       = [cexp| threadIdx.x == 0 && carryIn |]

    -- initialise the first thread with the results of the previous block sweep
    -- or exclusive scan element
    initialise
      | Just (CUExp seed) <- mseed
      = [cstm|  if ( gridDim.x > 1 ) {
                    $items:(z .=. blkSum "blockIdx.x")
                } else {
                    $items:(z .=. seed)
                }
        |]

      | otherwise
      = [cstm|  if ( blockIdx.x != $id:firstBlock ) {
                    $items:(z .=. blkSum prevBlock)
                    carryIn = 1;
                }
        |]


-- This computes the _upsweep_ phase of a multi-block scan. This is much like a
-- regular inclusive scan, except that only the final value for each interval is
-- output, rather than the entire body of the scan. Indeed, if the combination
-- function were commutative, this is equivalent to a parallel tree reduction.
--
mkScanUp1
    :: forall aenv e. Elt e
    => Direction
    -> DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> CUDelayedAcc aenv DIM1 e
    -> CUTranslSkel aenv (Vector e)
mkScanUp1 dir dev aenv fun@(CUFun2 _ _ combine) (CUDelayed (CUExp shIn) _ (CUFun1 _ get)) =
  CUTranslSkel scan [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    $id:scan
    (
        $params:argIn,
        $params:argOut
    )
    {
        $decls:smem
        $decls:declx
        $decls:decly
        $items:(sh .=. shIn)

        const int shapeSize     = $exp:(csize sh);
        const int intervalSize  = (shapeSize + gridDim.x - 1) / gridDim.x;

        const int start         = blockIdx.x * intervalSize;
        const int end           = min(start + intervalSize, shapeSize);
        const int numElements   = end - start;
              int carryIn       = 0;
              int seg;

        for ( seg = threadIdx.x
            ; seg < numElements
            ; seg += blockDim.x )
        {
            const int ix = $exp:firstIndex ;

            /*
             * Read in new values, combine with carry-in
             */
            $items:(x .=. get ix)

            if ( threadIdx.x == 0 && carryIn ) {
                $items:(x .=. combine y x)
            }

            /*
             * Store in shared memory and cooperatively scan
             */
            $items:(sdata "threadIdx.x" .=. x)
            __syncthreads();

            $items:(scanBlock dev fun x y sdata Nothing)

            /*
             * Store the final result of the block to be carried in
             */
            if ( threadIdx.x == 0 ) {
                const int last = min(numElements - seg, blockDim.x) - 1;
                $items:(y .=. sdata "last")
            }
            carryIn = 1;
        }

        /*
         * Finally, the first thread writes the result of this interval
         */
        if ( threadIdx.x == 0 ) {
            $items:(setOut "blockIdx.x" .=. y)
        }
    }
  |]
  where
    scan                        = "scan" ++ show dir ++ "Up"
    (texIn, argIn)              = environment dev aenv
    (argOut, _, setOut)         = writeArray "Out" (undefined :: Vector e)
    (_, x, declx)               = locals "x" (undefined :: e)
    (_, y, decly)               = locals "y" (undefined :: e)
    (sh, _, _)                  = locals "sh" (undefined :: DIM1)
    (smem, sdata)               = shared (undefined :: e) "sdata" [cexp| blockDim.x |] Nothing
    ix                          = [cvar "ix"]

    firstIndex
      | dir == L                = [cexp| start + seg |]
      | otherwise               = [cexp| end - seg - 1 |]


-- Second step of the upsweep phase: scan the interval sums to produce carry-in
-- values for each block of the final downsweep step
--
mkScanUp2
    :: forall aenv e. Elt e
    => Direction
    -> DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> Maybe (CUExp aenv e)
    -> CUTranslSkel aenv (Vector e)
mkScanUp2 dir dev aenv f z
  = let (_, get) = readArray "Blk" (undefined :: Vector e)
    in  mkScan dir dev aenv f z get


-- Block scans
-- ===========

scanBlock
    :: forall aenv e. Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]
    -> (Name -> [C.Exp])
    -> Maybe C.Exp
    -> [C.BlockItem]
scanBlock dev f x0 x1 sdata mlim
  | shflOK dev (undefined :: e) = error "shfl-scan"
  | otherwise                   = scanBlockTree dev f x0 x1 sdata mlim


-- Use a thread block to scan values in shared memory. Each thread must have
-- already stored its initial value into shared memory. The final result for
-- this thread will be stored in x0 as well as the appropriate place in shared
-- memory.
--
scanBlockTree
    :: forall aenv e. Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]               -- temporary variables x0 and x1
    -> (Name -> [C.Exp])                -- index elements from shared memory
    -> Maybe C.Exp                      -- partially full block bounds check?
    -> [C.BlockItem]
scanBlockTree dev (CUFun2 _ _ f) x0 x1 sdata mlim = map (scan . pow2) [ 0 .. maxThreads ]
  where
    pow2 :: Int -> Int
    pow2 x      = 2 ^ x
    maxThreads  = floor (logBase 2 (fromIntegral $ maxThreadsPerBlock dev :: Double))

    inrange n
      | Just m <- mlim  = [cexp| threadIdx.x >= $int:n && threadIdx.x < $exp:m |]
      | otherwise       = [cexp| threadIdx.x >= $int:n |]

    scan n = [citem|
      if ( blockDim.x > $int:n ) {
          if ( $exp:(inrange n) ) {
              $items:(x1 .=. sdata ("threadIdx.x - " ++ show n))
              $items:(x0 .=. f x1 x0)
          }
          __syncthreads();
          $items:(sdata "threadIdx.x" .=. x0)
          __syncthreads();
      }
      |]


-- Shuffle scan
-- ------------

shflOK :: Elt e => DeviceProperties -> e -> Bool
shflOK _dev _ = False
-- shflOk dev dummy
--   = computeCapability dev >= Compute 3 0 && all (`elem` [4,8]) (eltSizeOf dummy)

{--
scanWarpShfl
    :: forall aenv e. Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]               -- temporary variables x0 and x1
    -> Maybe C.Exp                      -- partially full block bounds check
    -> C.Exp                            -- thread identified, usually lane or thread ID
    -> C.Stm
scanWarpShfl _dev (CUFun2 f) x0 x1 mlim tid
  = [cstm|
      for ( int z = 1; z <= warpSize; z *= 2 ) {
          $items:(x0 .=. shfl_up x1)

          if ( $exp:inrange ) {
              $items:(x1 .=. f x1 x0)
          }
      }
    |]
  where
    inrange
      | Just m <- mlim  = [cexp| $exp:tid >= z && $exp:tid < $exp:m |]
      | otherwise       = [cexp| $exp:tid >= z |]

    sizeof      = eltSizeOf (undefined :: e)
    shfl_up     = zipWith (\s x -> ccall (shfl s) [ x, cvar "z" ]) sizeof
      where
        shfl 4  = "shfl_up32"
        shfl 8  = "shfl_up64"
        shfl _  = INTERNAL_ERROR(error) "shfl_up" "I only know about 32- and 64-bit types"
--}

