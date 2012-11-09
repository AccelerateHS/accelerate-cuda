{-# LANGUAGE CPP                   #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverlappingInstances  #-}
{-# LANGUAGE QuasiQuotes           #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Base
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Base (

  -- Names and Types
  CUTranslSkel(..), CUDelayedAcc(..), CUExp(..), CUFun1(..), CUFun2(..),
  Name, namesOfArray, namesOfAvar,

  -- Declaration generation
  cvar, ccall, cchar, cintegral, cbool, cdim, cshape, getters, setters, shared,
  indexArray, indexHead, shapeSize, environment, arrayAsTex, arrayAsArg,
  umul24, gridSize, threadIdx,

  -- Mutable operations
  (.=.), locals, Lvalue(..), Rvalue(..),

) where

import Text.PrettyPrint.Mainland
import Language.C.Quote.CUDA
import qualified Language.C.Syntax                      as C
import qualified Data.HashSet                           as Set

import Foreign.CUDA.Analysis.Device
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.CUDA.CodeGen.Type
import Data.Array.Accelerate.CUDA.AST

#include "accelerate.h"

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
        arr x   = "arr" ++ grp ++ "_a" ++ show x
        n       = length ty
    in
    ( "sh" ++ grp, map arr [n-1, n-2 .. 0] )


namesOfAvar :: forall aenv sh e. Elt e => Idx aenv (Array sh e) -> (Name, [Name])
namesOfAvar ix = namesOfArray (groupOfAvar ix) (undefined :: e)

groupOfAvar :: Idx aenv (Array sh e) -> Name
groupOfAvar ix = "In" ++ show (idxToInt ix)


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
data CUFun1 aenv f where
  CUFun1 :: (forall x. Rvalue x => [x] -> ([C.BlockItem], [C.Exp]))
         -> CUFun1 aenv (a -> b)

data CUFun2 aenv f where
  CUFun2 :: (forall x y. (Rvalue x, Rvalue y) => [x] -> [y] -> ([C.BlockItem], [C.Exp]))
         -> CUFun2 aenv (a -> b -> c)

-- Delayed arrays
--
data CUDelayedAcc aenv sh e where
  CUDelayed :: CUExp  aenv sh
            -> CUFun1 aenv (sh -> e)
            -> CUFun1 aenv (Int -> e)
            -> CUDelayedAcc aenv sh e


-- Expression and declaration generation
-- -------------------------------------

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

cdim :: Name -> Int -> C.Definition
cdim name n = [cedecl|typedef typename $id:("DIM" ++ show n) $id:name;|]

-- Disassemble a struct-shape into a list of expressions accessing the fields
cshape :: Int -> C.Exp -> [C.Exp]
cshape dim sh
  | dim == 0  = []
  | dim == 1  = [sh]
  | otherwise = map (\i -> [cexp|$exp:sh . $id:('a':show i)|]) [dim-1, dim-2 .. 0]

-- Calculate the size of a shape from its component dimensions
shapeSize :: Rvalue r => [r] -> C.Exp
shapeSize = foldl (\a b -> [cexp| $exp:a * $exp:b |]) [cexp| 1 |]
          . map rvalue

indexHead :: Rvalue r => [r] -> C.Exp
indexHead = rvalue . last


-- Thread blocks and indices
--
umul24 :: DeviceProperties -> C.Exp -> C.Exp -> C.Exp
umul24 dev x y
  | computeCapability dev < Compute 2 0 = [cexp| __umul24($exp:x, $exp:y) |]
  | otherwise                           = [cexp| $exp:x * $exp:y |]

gridSize :: DeviceProperties -> C.Exp
gridSize dev
  | computeCapability dev < Compute 2 0 = [cexp| __umul24(blockDim.x, gridDim.x) |]
  | otherwise                           = [cexp| blockDim.x * gridDim.x |]

threadIdx :: DeviceProperties -> C.Exp
threadIdx dev
  | computeCapability dev < Compute 2 0 = [cexp| __umul24(blockDim.x, blockIdx.x) + threadIdx.x |]
  | otherwise                           = [cexp| blockDim.x * blockIdx.x + threadIdx.x |]


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
  | computeCapability dev >= Compute 2 0                = [cexp| $exp:arr [ $exp:ix ] |]

  -- use the texture cache of compute 1.x devices
  | C.Type (C.DeclSpec _ _ (C.Tdouble _) _) _ _ <- elt  = ccall "indexDArray" [arr, ix]
  | otherwise                                           = ccall "indexArray"  [arr, ix]


-- Generate kernel parameters for an array valued argument, and a function to
-- linearly index this array. Note that dimensional indexing results in error.
--
getters
    :: forall aenv sh e. (Shape sh, Elt e)
    => Name                             -- group names
    -> Array sh e                       -- dummy to fix types
    -> ( [C.Param], CUDelayedAcc aenv sh e )
getters grp dummy
  = let (sh, arrs)      = namesOfArray grp (undefined :: e)
        args            = arrayAsArg dummy grp

        dim             = expDim (undefined :: Exp aenv sh)
        sh'             = cshape dim (cvar sh)
        get ix          = ([], map (\a -> [cexp| $id:a [ $exp:ix ] |]) arrs)
        manifest        = CUDelayed (CUExp ([], sh'))
                                    (INTERNAL_ERROR(error) "getters" "linear indexing only")
                                    (CUFun1 (get . rvalue . head))
    in ( args, manifest )


-- Generate function parameters and corresponding variable names for the
-- components of the given output array. The parameter list generated is
-- suitable for marshalling an instance of "Array sh e", consisting of a group
-- name (say "Out") to be welded with a shape name "shOut" followed by the
-- non-parametric array data "arrOut_aX".
--
setters
    :: forall sh e. (Shape sh, Elt e)
    => Name                             -- group names
    -> Array sh e                       -- dummy to fix types
    -> ([C.Param], Name -> [C.Exp])
setters grp _
  = let (sh, arrs)      = namesOfArray grp (undefined :: e)
        dim             = expDim (undefined :: Exp aenv sh)
        sh'             = [cparam| const typename $id:("DIM" ++ show dim) $id:sh |]
        arrs'           = zipWith (\t n -> [cparam| $ty:t * __restrict__ $id:n |]) (eltType (undefined :: e)) arrs
    in
    ( sh' : arrs'
    , \ix -> map (\a -> [cexp| $id:a [ $id:ix ] |]) arrs
    )


-- All dynamically allocated __shared__ memory will begin at the same base
-- address. If we call this more than once, or the kernel itself declares some
-- shared memory, the first parameter is a pointer to where the new declarations
-- should take as the base address.
--
shared
    :: forall e. Elt e
    => e                                -- dummy type
    -> Name                             -- group name
    -> C.Exp                            -- how much shared memory per type
    -> Maybe C.Exp                      -- (optional) initialise from this base address
    -> ([C.InitGroup], Name -> [C.Exp]) -- shared memory declaration and indexing function
shared _ grp size mprev
  = let e:es                    = eltType (undefined :: e)
        x:xs                    = let k = length es in map (\n -> grp ++ show n) [k, k-1 .. 0]

        sdata t v p             = [cdecl| volatile $ty:t * $id:v = ($ty:t *) & $id:p [ $exp:size ]; |]
        sbase t v
          | Just p <- mprev     = [cdecl| volatile $ty:t * $id:v = ($ty:t *) $exp:p; |]
          | otherwise           = [cdecl| extern volatile __shared__ $ty:t $id:v [] ; |]
    in
    ( sbase e x : zipWith3 sdata es xs (x:xs)
    , \ix -> map (\v -> [cexp| $id:v [ $id:ix ] |]) (x:xs)
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
    :: DeviceProperties
    -> Gamma aenv
    -> ([C.Definition], [C.Param])
environment dev (Gamma aenv)
  | computeCapability dev < Compute 2 0
  = (Set.foldr (\(Idx_ v) vs -> asTex v ++ vs) [] aenv, [])

  | otherwise
  = ([], Set.foldr (\(Idx_ v) vs -> asArg v ++ vs) [] aenv)

  where
    asTex :: forall aenv sh e. (Shape sh, Elt e) => Idx aenv (Array sh e) -> [C.Definition]
    asTex ix = arrayAsTex (undefined :: Array sh e) (groupOfAvar ix)

    asArg :: forall aenv sh e. (Shape sh, Elt e) => Idx aenv (Array sh e) -> [C.Param]
    asArg ix = arrayAsArg (undefined :: Array sh e) (groupOfAvar ix)


arrayAsTex :: forall sh e. (Shape sh, Elt e) => Array sh e -> Name -> [C.Definition]
arrayAsTex _ grp =
  let (sh, arrs)        = namesOfArray grp (undefined :: e)
      dim               = expDim (undefined :: Exp aenv sh)
      sh'               = [cedecl| static __constant__ typename $id:("DIM" ++ show dim) $id:sh; |]
      arrs'             = zipWith (\t a -> [cedecl| static $ty:t $id:a; |]) (eltTypeTex (undefined :: e)) arrs
  in
  sh' : arrs'

arrayAsArg :: forall sh e. (Shape sh, Elt e) => Array sh e -> Name -> [C.Param]
arrayAsArg _ grp =
  let (sh, arrs)        = namesOfArray grp (undefined :: e)
      dim               = expDim (undefined :: Exp aenv sh)
      sh'               = [cparam| const typename $id:("DIM" ++ show dim) $id:sh |]
      arrs'             = zipWith (\t n -> [cparam| const $ty:t * __restrict__ $id:n |]) (eltType (undefined :: e)) arrs
  in
  sh' : arrs'


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

instance Lvalue C.Exp where
  lvalue x y = C.BlockStm  [cstm| $exp:x = $exp:y; |]

instance Lvalue (C.Type, Name) where
  lvalue (t,x) y = C.BlockDecl [cdecl| const $ty:t $id:x = $exp:y; |]


class Rvalue a where
  rvalue :: a -> C.Exp

instance Rvalue C.Exp where
  rvalue = id

instance Rvalue (C.Type, Name) where
  rvalue (_,x) = cvar x


infixr 0 .=.
(.=.) :: Assign l r => l -> r -> [C.BlockItem]
(.=.) =  assign

class Assign l r where
  assign :: l -> r -> [C.BlockItem]

instance (Lvalue l, Rvalue r) => Assign l r where
  assign lhs rhs = return $ lvalue lhs (rvalue rhs)

instance Assign l r => Assign [l] [r] where
  assign []     []     = []
  assign (x:xs) (y:ys) = assign x y ++ assign xs ys
  assign _      _      = INTERNAL_ERROR(error) ".=." "argument mismatch"

instance Assign l r => Assign l ([C.BlockItem], r) where
  assign lhs (env, rhs) = env ++ assign lhs rhs

