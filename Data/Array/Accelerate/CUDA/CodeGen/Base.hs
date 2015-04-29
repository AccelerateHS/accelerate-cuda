{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE ImpredicativeTypes    #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverlappingInstances  #-}
{-# LANGUAGE PatternGuards         #-}
{-# LANGUAGE QuasiQuotes           #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TemplateHaskell       #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Base
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Base (

  -- Names and Types
  CUTranslSkel(..), CUDelayedAcc(..), CUExp(..), CUFun1(..), CUFun2(..),
  Eliminate, Instantiate1, Instantiate2,
  Name, namesOfArray, namesOfAvar, groupOfInt,

  -- Declaration generation
  cint, cvar, ccall, cchar, cintegral, cbool, cshape, cslice, csize, cindexHead, cindexTail, ctoIndex, cfromIndex,
  readArray, writeArray, shared,
  indexArray, environment, arrayAsTex, arrayAsArg,
  umul24, gridSize, threadIdx,

  -- Mutable operations
  (.=.), locals, Lvalue(..), Rvalue(..),

) where

-- library
import Prelude                                          hiding ( zipWith, zipWith3 )
import Data.List                                        ( mapAccumR )
import Text.PrettyPrint.Mainland
import Language.C.Quote.CUDA
import qualified Language.C.Syntax                      as C
import qualified Data.HashMap.Strict                    as Map

-- cuda
import Foreign.CUDA.Analysis.Device

-- friends
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Array.Representation       ( SliceIndex(..) )
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.CUDA.CodeGen.Type
import Data.Array.Accelerate.CUDA.AST

-- Names
-- -----

type Name = String

namesOfArray
    :: forall e. Elt e
    => Name             -- name of group: typically "Out" or "InX" for some number 'X'
    -> e                -- dummy
    -> (Name, [Name])   -- shape and array field names
namesOfArray grp _
  = let ty      = eltType (undefined :: e)
        arr x   = "arr" ++ grp ++ '_':show x
        n       = length ty
    in
    ( "sh" ++ grp, map arr [n-1, n-2 .. 0] )


namesOfAvar :: forall aenv sh e. (Shape sh, Elt e) => Gamma aenv -> Idx aenv (Array sh e) -> (Name, [Name])
namesOfAvar gamma ix = namesOfArray (groupOfAvar gamma ix) (undefined::e)

groupOfAvar :: (Shape sh, Elt e) => Gamma aenv -> Idx aenv (Array sh e) -> Name
groupOfAvar (Gamma gamma) = groupOfInt . (gamma Map.!) . Idx_

groupOfInt :: Int -> Name
groupOfInt n = "In" ++ show n


-- Types of compilation units
-- --------------------------

-- A CUDA compilation unit, together with the name of the main __global__ entry
-- function.
--
data CUTranslSkel aenv a = CUTranslSkel Name [C.Definition]

instance Show (CUTranslSkel aenv a) where
  show (CUTranslSkel entry _) = entry

instance Pretty (CUTranslSkel aenv a) where
  ppr  (CUTranslSkel _ code)  = ppr code

-- Scalar expressions, including the environment of local let-bindings to bring
-- into scope before evaluating the body.
--
data CUExp aenv a where
  CUExp  :: ([C.BlockItem], [C.Exp])
         -> CUExp aenv a

-- Scalar functions of particular arity, with local bindings.
--
type Eliminate a        = forall x. [x] -> [(Bool,x)]
type Instantiate1 a b   = forall x. Rvalue x => [x] -> ([C.BlockItem], [C.Exp])
type Instantiate2 a b c = forall x y. (Rvalue x, Rvalue y) => [x] -> [y] -> ([C.BlockItem], [C.Exp])

data CUFun1 aenv f where
  CUFun1 :: (Elt a, Elt b)
         => Eliminate a
         -> Instantiate1 a b
         -> CUFun1 aenv (a -> b)

data CUFun2 aenv f where
  CUFun2 :: (Elt a, Elt b, Elt c)
         => Eliminate a
         -> Eliminate b
         -> Instantiate2 a b c
         -> CUFun2 aenv (a -> b -> c)

-- Delayed arrays
--
data CUDelayedAcc aenv sh e where
  CUDelayed :: CUExp  aenv sh
            -> CUFun1 aenv (sh -> e)
            -> CUFun1 aenv (Int -> e)
            -> CUDelayedAcc aenv sh e


-- Common expression forms
-- -----------------------

cint :: C.Type
cint = typeOf (scalarType :: ScalarType Int)

cvar :: Name -> C.Exp
cvar x = [cexp|$id:x|]

ccall :: Name -> [C.Exp] -> C.Exp
ccall fn args = [cexp|$id:fn ($args:args)|]

cchar :: Char -> C.Exp
cchar c = [cexp|$char:c|]

cintegral :: (Integral a, Show a) => a -> C.Exp
cintegral n = [cexp|$int:n|]

cbool :: Bool -> C.Exp
cbool = cintegral . fromEnum

cslice :: SliceIndex slix sl co dim -> Name -> ([C.Param], [C.Exp], [(C.Type, Name)])
cslice slix sl =
  let xs = cshape' (ncodims slix) sl
      args = [ [cparam| const $ty:cint $id:x |] | x <- xs ]
  in (args, map cvar xs, zip (repeat cint) xs)
  where
    ncodims :: SliceIndex slix sl co dim -> Int
    ncodims SliceNil = 0
    ncodims (SliceAll   s) = ncodims s
    ncodims (SliceFixed s) = ncodims s + 1

-- Generate all the names of a shape given a base name and dimensionality
cshape :: Int -> Name -> [C.Exp]
cshape dim sh = [ cvar x | x <- cshape' dim sh ]

cshape' :: Int -> Name -> [Name]
cshape' dim sh = [ (sh ++ '_':show i) | i <- [dim-1, dim-2 .. 0] ]

-- Get the innermost index of a shape/index
cindexHead :: Rvalue r => [r] -> C.Exp
cindexHead = rvalue . last

-- Get the tail of a shape/index
cindexTail :: Rvalue r => [r] -> [C.Exp]
cindexTail = map rvalue . init

-- generate code that calculates the product of the list of expressions
csize :: Rvalue r => [r] -> C.Exp
csize [] = [cexp| 1 |]
csize ss = foldr1 (\a b -> [cexp| $exp:a * $exp:b |]) (map rvalue ss)

-- Generate code to calculate a linear from a multi-dimensional index (given an array shape).
--
ctoIndex :: (Rvalue sh, Rvalue ix) => [sh] -> [ix] -> C.Exp
ctoIndex extent index
  = toIndex (reverse $ map rvalue extent) (reverse $ map rvalue index)  -- we use a row-major representation
  where
    toIndex []      []     = [cexp| $int:(0::Int) |]
    toIndex [_]     [i]    = i
    toIndex (sz:sh) (i:ix) = [cexp| $exp:(toIndex sh ix) * $exp:sz + $exp:i |]
    toIndex _       _      = $internalError "toIndex" "argument mismatch"

-- Generate code to calculate a multi-dimensional index from a linear index and a given array shape.
-- This version creates temporary values that are reused in the computation.
--
cfromIndex :: (Rvalue sh, Rvalue ix) => [sh] -> ix -> Name -> ([C.BlockItem], [C.Exp])
cfromIndex shName ixName tmpName = fromIndex (map rvalue shName) (rvalue ixName)
  where
    fromIndex [sh]   ix = ([], [[cexp| ({ assert( $exp:ix >= 0 && $exp:ix < $exp:sh ); $exp:ix; }) |]])
    fromIndex extent ix = let ((env, _, _), sh) = mapAccumR go ([], ix, 0) extent
                          in  (reverse env, sh)

    go (tmps,ix,n) d
      = let tmp         = tmpName ++ '_':show (n::Int)
            ix'         = [citem| const $ty:cint $id:tmp = $exp:ix ; |]
        in
        ((ix':tmps, [cexp| $id:tmp / $exp:d |], n+1), [cexp| $id:tmp % $exp:d |])


-- Thread blocks and indices
-- -------------------------

umul24 :: DeviceProperties -> C.Exp -> C.Exp -> C.Exp
umul24 dev x y
  | computeCapability dev < Compute 2 0 = [cexp| __umul24($exp:x, $exp:y) |]
  | otherwise                           = [cexp| $exp:x * $exp:y |]

gridSize :: DeviceProperties -> C.Exp
gridSize dev = umul24 dev [cexp|blockDim.x|] [cexp|gridDim.x|]

threadIdx :: DeviceProperties -> C.Exp
threadIdx dev =
  let block = umul24 dev [cexp|blockDim.x|] [cexp|blockIdx.x|]
  in  [cexp| $exp:block + threadIdx.x |]


-- Generate an array indexing expression. Depending on the hardware class, this
-- will be via direct array indexing or texture references.
--
indexArray
    :: DeviceProperties
    -> C.Type                   -- array element type (Float, Double...)
    -> C.Exp                    -- array variable name (arrInX_Y)
    -> C.Exp                    -- linear index
    -> C.Exp
indexArray dev elt arr ix
  -- use the L2 cache of newer devices
  | computeCapability dev >= Compute 2 0 = [cexp| $exp:arr [ $exp:ix ] |]

  -- use the texture cache of compute 1.x devices
  | [cty|double|] <- elt                 = ccall "indexDArray" [arr, ix]
  | otherwise                            = ccall "indexArray"  [arr, ix]


-- Kernel function parameters
-- --------------------------

-- Generate kernel parameters for an array valued argument, and a function to
-- linearly index this array. Note that dimensional indexing results in error.
--
readArray
    :: forall aenv sh e. (Shape sh, Elt e)
    => Name                             -- group names
    -> Array sh e                       -- dummy to fix types
    -> ( [C.Param]
       , [C.Exp]
       , CUDelayedAcc aenv sh e )
readArray grp dummy
  = let (sh, arrs)      = namesOfArray grp (undefined :: e)
        args            = arrayAsArg dummy grp

        dim             = expDim (undefined :: Exp aenv sh)
        sh'             = cshape dim sh
        get ix          = ([], map (\a -> [cexp| $id:a [ $exp:ix ] |]) arrs)
        manifest        = CUDelayed (CUExp ([], sh'))
                                    ($internalError "readArray" "linear indexing only")
                                    (CUFun1 (zip (repeat True)) (\[i] -> get (rvalue i)))
    in ( args, sh', manifest )


-- Generate function parameters and corresponding variable names for the
-- components of the given output array. The parameter list generated is
-- suitable for marshalling an instance of "Array sh e", consisting of a group
-- name (say "Out") to be welded with a shape name "shOut" followed by the
-- non-parametric array data "arrOut_aX".
--
writeArray
    :: forall sh e. (Shape sh, Elt e)
    => Name                                     -- group names
    -> Array sh e                               -- dummy to fix types
    -> ( [C.Param]                              -- function parameters to marshal the output array
       , [C.Exp]                                -- the shape of the output array
       , forall x. Rvalue x => x -> [C.Exp] )   -- write an element at a given index
writeArray grp _ =
  let (sh, arrs)        = namesOfArray grp (undefined :: e)
      dim               = expDim (undefined :: Exp aenv sh)
      sh'               = cshape' dim sh
      extent            = [ [cparam| const $ty:cint $id:i |] | i <- sh' ]
      adata             = zipWith (\t n -> [cparam| $ty:t * __restrict__ $id:n |]) (eltType (undefined :: e)) arrs
  in
  ( extent ++ adata
  , map cvar sh'
  , \ix -> map (\a -> [cexp| $id:a [ $exp:(rvalue ix) ] |]) arrs
  )


-- All dynamically allocated __shared__ memory will begin at the same base
-- address. If we call this more than once, or the kernel itself declares some
-- shared memory, the first parameter is a pointer to where the new declarations
-- should take as the base address.
--
shared
    :: forall e. Elt e
    => e                                        -- dummy type
    -> Name                                     -- group name
    -> C.Exp                                    -- how much shared memory per type
    -> Maybe C.Exp                              -- (optional) initialise from this base address
    -> ( [C.InitGroup]                          -- shared memory declaration and...
       , forall x. Rvalue x => x -> [C.Exp])    -- ...indexing function
shared _ grp size mprev
  = let e:es                    = eltType (undefined :: e)
        x:xs                    = let k = length es in map (\n -> grp ++ show n) [k, k-1 .. 0]

        sdata t v p             = [cdecl| volatile $ty:t * $id:v = ($ty:t *) & $id:p [ $exp:size ]; |]
        sbase t v
          | Just p <- mprev     = [cdecl| volatile $ty:t * $id:v = ($ty:t *) $exp:p; |]
          | otherwise           = [cdecl| extern volatile __shared__ $ty:t $id:v [] ; |]
    in
    ( sbase e x : zipWith3 sdata es xs (init (x:xs))
    , \ix -> map (\v -> [cexp| $id:v [ $exp:(rvalue ix) ] |]) (x:xs)
    )

-- Array environment references. The method in which arrays are accessed depends
-- on the device architecture (see below). We always include the array shape
-- before the array data terms.
--
--   compute 1.x:
--      texture references of type [Definition]
--
--   compute 2.x and 3.x:
--      function arguments of type [Param]
--
-- NOTE: The environment variables must always be the first argument to the
--       kernel function, as this is where they will be marshaled during the
--       execution phase.
--
environment
    :: forall aenv. DeviceProperties
    -> Gamma aenv
    -> ([C.Definition], [C.Param])
environment dev gamma@(Gamma aenv)
  | computeCapability dev < Compute 2 0
  = Map.foldrWithKey (\(Idx_ v) _ (ds,ps) -> let (d,p) = asTex v in (d++ds, p++ps)) ([],[]) aenv

  | otherwise
  = ([], Map.foldrWithKey (\(Idx_ v) _ vs -> asArg v ++ vs) [] aenv)

  where
    asTex :: forall sh e. (Shape sh, Elt e) => Idx aenv (Array sh e) -> ([C.Definition], [C.Param])
    asTex ix = arrayAsTex (undefined :: Array sh e) (groupOfAvar gamma ix)

    asArg :: forall sh e. (Shape sh, Elt e) => Idx aenv (Array sh e) -> [C.Param]
    asArg ix = arrayAsArg (undefined :: Array sh e) (groupOfAvar gamma ix)


arrayAsTex :: forall sh e. (Shape sh, Elt e) => Array sh e -> Name -> ([C.Definition], [C.Param])
arrayAsTex _ grp =
  let (sh, arrs)        = namesOfArray grp (undefined :: e)
      dim               = expDim (undefined :: Exp aenv sh)
      extent            = [ [cparam| const $ty:cint $id:i |] | i <- cshape' dim sh ]
      adata             = zipWith (\t a -> [cedecl| static $ty:t $id:a; |]) (eltTypeTex (undefined :: e)) arrs
  in
  (adata, extent)

arrayAsArg :: forall sh e. (Shape sh, Elt e) => Array sh e -> Name -> [C.Param]
arrayAsArg _ grp =
  let (sh, arrs)        = namesOfArray grp (undefined :: e)
      dim               = expDim (undefined :: Exp aenv sh)
      extent            = [ [cparam| const $ty:cint $id:i |] | i <- cshape' dim sh ]
      adata             = zipWith (\t n -> [cparam| const $ty:t * __restrict__ $id:n |]) (eltType (undefined :: e)) arrs
  in
  extent ++ adata


-- Mutable operations
-- ------------------

-- Declare some local variables. These can be either const or mutable
-- declarations.
--
locals :: forall e. Elt e
       => Name
       -> e
       -> ( [(C.Type, Name)]            -- const declarations
          , [C.Exp], [C.InitGroup])     -- mutable declaration and names
locals base _
  = let elt             = eltType (undefined :: e)
        n               = length elt
        local t v       = let name = base ++ show v
                          in ( (t, name), cvar name, [cdecl| $ty:t $id:name; |] )
    in
    unzip3 $ zipWith local elt [n-1, n-2 .. 0]

class Lvalue a where
  lvalue :: a -> C.Exp -> C.BlockItem

-- Note: [Mutable l-values]
--
-- Be careful when using mutable l-values that the same variable does not appear
-- on both the left and right hand side. For example, the following will lead to
-- problems (#114, #168):
--
--   $items:(x .=. f x)
--
--   $items:(y .=. combine x y)
--
-- If 'x' and 'y' represent values with tuple types, they will have multiple
-- components. Since the LHS is updated as the new values are calculated, it is
-- possible to get into a situation where computing the new value for some of
-- the components will be using the updated values, not the original values.
--
-- Instead, store the result to some (temporary) variable that does not appear
-- on the RHS, and then update old value using the fully computed result, e.g.:
--
--   $items:(x' .=. f x)
--   $items:(x  .=. x')
--
instance Lvalue C.Exp where
  lvalue x y = [citem| $exp:x = $exp:y; |]

instance Lvalue (C.Type, Name) where
  lvalue (t,x) y = [citem| const $ty:t $id:x = $exp:y; |]


class Rvalue a where
  rvalue :: a -> C.Exp

instance Rvalue C.Exp where
  rvalue = id

instance Rvalue Name where
  rvalue = cvar

instance Rvalue (C.Type, Name) where
  rvalue (_,x) = rvalue x


infixr 0 .=.
(.=.) :: Assign l r => l -> r -> [C.BlockItem]
(.=.) =  assign

class Assign l r where
  assign :: l -> r -> [C.BlockItem]

instance (Lvalue l, Rvalue r) => Assign l r where
  assign lhs rhs = [ lvalue lhs (rvalue rhs) ]

instance Assign l r => Assign (Bool,l) r where
  assign (used,lhs) rhs
    | used      = assign lhs rhs
    | otherwise = []

instance Assign l r => Assign [l] [r] where
  assign []     []     = []
  assign (x:xs) (y:ys) = assign x y ++ assign xs ys
  assign _      _      = $internalError ".=." "argument mismatch"

instance Assign l r => Assign l ([C.BlockItem], r) where
  assign lhs (env, rhs) = env ++ assign lhs rhs


-- Prelude'
-- --------

-- A version of zipWith that requires the lists to be equal length
--
zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith f (x:xs) (y:ys) = f x y : zipWith f xs ys
zipWith _ []     []     = []
zipWith _ _      _      = $internalError "zipWith" "argument mismatch"

zipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
zipWith3 f (x:xs) (y:ys) (z:zs) = f x y z : zipWith3 f xs ys zs
zipWith3 _ []     []     []     = []
zipWith3 _ _      _      _      = $internalError "zipWith3" "argument mismatch"

