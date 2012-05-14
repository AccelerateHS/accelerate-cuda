{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS -fno-warn-incomplete-patterns #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Mapping
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Stencil (

  mkStencil, mkStencil2

) where

import Language.C.Syntax
import Language.C.Quote.CUDA

import Data.Array.Accelerate.Type
import Data.Array.Accelerate.AST                        ( OpenAcc, Fun, Stencil )
import Data.Array.Accelerate.Array.Sugar                ( Array, Elt, shapeToList )
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type

import qualified Data.Array.Accelerate.Analysis.Stencil as Stencil
import qualified Data.Array.IArray                      as IArray


-- Map a stencil over an array.  In contrast to 'map', the domain of a stencil
-- function is an entire /neighbourhood/ of each array element.  Neighbourhoods
-- are sub-arrays centred around a focal point.  They are not necessarily
-- rectangular, but they are symmetric in each dimension and have an extent of
-- at least three in each dimensions — due to the symmetry requirement, the
-- extent is necessarily odd.  The focal point is the array position that is
-- determined by the stencil.
--
-- For those array positions where the neighbourhood extends past the boundaries
-- of the source array, a boundary condition determines the contents of the
-- out-of-bounds neighbourhood positions.
--
-- stencil :: (Shape ix, Elt a, Elt b, Stencil ix a stencil)
--         => (stencil -> Exp b)                 -- stencil function
--         -> Boundary a                         -- boundary condition
--         -> Acc (Array ix a)                   -- source array
--         -> Acc (Array ix b)                   -- destination array
--
-- To improve performance, the input array(s) are read through the texture
-- cache.
--
mkStencil :: forall sh stencil a b. (Stencil sh a stencil, Elt b)
          => Int
          -> CUFun (stencil -> b)
          -> Boundary (CUExp a)
          -> Array sh b                 {- dummy -}
          -> CUTranslSkel
mkStencil dim (CULam use0 (CUBody (CUExp env stencil))) boundary _ =
  CUTranslSkel "stencil" [cunit|
    $edecl:(cdim "Shape" dim)
    $edecls:arrIn0

    extern "C"
    __global__ void
    stencil
    (
        $params:argOut,
        const typename Shape shIn0
    )
    {
        const int shapeSize = size(shIn0);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);
              int i;

        for ( i =  __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; i <  shapeSize
            ; i += gridSize )
        {
            const typename Shape ix = fromIndex(shIn0, i);
            $decls:(getIn0 "ix")
            $decls:env
            $stms:(setOut "i" stencil)
        }
    }
  |]
  where
    tyOut               = eltType    (undefined :: b)
    stencilIn0          = eltTypeTex (undefined :: a)
    (argOut, _, setOut) = setters tyOut
    (arrIn0, getIn0)    = stencilAccess 0 dim stencilIn0 use0 boundary offsets
    --
    offsets             = map shapeToList p0
    p0                  = Stencil.offsets (undefined :: Fun aenv (stencil -> b))
                                          (undefined :: OpenAcc aenv (Array sh a))


-- Map a binary stencil of an array.  The extent of the resulting array is the
-- intersection of the extents of the two source arrays.
--
-- stencil2 :: (Shape ix, Elt a, Elt b, Elt c,
--              Stencil ix a stencil1,
--              Stencil ix b stencil2)
--          => (stencil1 -> stencil2 -> Exp c)  -- binary stencil function
--          -> Boundary a                       -- boundary condition #1
--          -> Acc (Array ix a)                 -- source array #1
--          -> Boundary b                       -- boundary condition #2
--          -> Acc (Array ix b)                 -- source array #2
--          -> Acc (Array ix c)                 -- destination array
--
mkStencil2 :: forall sh stencil1 stencil0 a b c.
              (Stencil sh a stencil1, Stencil sh b stencil0, Elt c)
           => Int
           -> CUFun (stencil1 -> stencil0 -> c)
           -> Boundary (CUExp a)
           -> Boundary (CUExp b)
           -> Array sh c                        {- dummy -}
           -> CUTranslSkel
mkStencil2 dim (CULam use1 (CULam use0 (CUBody (CUExp env stencil)))) boundary1 boundary0 _ =
  CUTranslSkel "stencil2" [cunit|
    $edecl:(cdim "Shape" dim)
    $edecls:arrIn0
    $edecls:arrIn1

    extern "C"
    __global__ void
    stencil2
    (
        $params:argOut,
        const typename Shape shOut,
        const typename Shape shIn1,
        const typename Shape shIn0
    )
    {
        const int shapeSize = size(shOut);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);
              int i;

        for ( i =  __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; i <  shapeSize
            ; i += gridSize )
        {
            const typename Shape ix = fromIndex(shOut, i);
            $decls:(getIn0 "ix")
            $decls:(getIn1 "ix")
            $decls:env
            $stms:(setOut "i" stencil)
        }
    }
  |]
  where
    tyOut               = eltType    (undefined :: c)
    stencilIn0          = eltTypeTex (undefined :: b)
    stencilIn1          = eltTypeTex (undefined :: a)
    (argOut, _, setOut) = setters tyOut
    (arrIn0, getIn0)    = stencilAccess 0 dim stencilIn0 use0 boundary0 offsets0
    (arrIn1, getIn1)    = stencilAccess 1 dim stencilIn1 use1 boundary1 offsets1
    --
    offsets0            = map shapeToList p0
    offsets1            = map shapeToList p1
    (p1, p0)            = Stencil.offsets2 (undefined :: Fun aenv (stencil1 -> stencil0 -> c))
                                           (undefined :: OpenAcc aenv (Array sh a))
                                           (undefined :: OpenAcc aenv (Array sh b))


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

stencilAccess
    :: Int                              -- array de Bruijn index
    -> Int                              -- array dimensionality
    -> [Type]                           -- array type (texture memory)
    -> [(Int, Type, Exp)]               -- the variables used in the scalar expression
    -> Boundary (CUExp a)               -- how to handle boundary array access
    -> [[Int]]                          -- all stencil index offsets, top left to bottom right
    -> ( [Definition]                   -- texture-reference definitions
       , String -> [InitGroup] )        -- array indexing
stencilAccess base dim stencil subs boundary shx =
  ( textures
  , \ix -> concatMap (get ix) subs )
  where
    n           = length stencil
    sh          = "shIn" ++ show  base
    arr x       = "arrIn" ++ shows base "_a" ++ show (x `mod` n)
    textures    = zipWith cglobal stencil (map arr [n-1, n-2 .. 0])
    --
    offsets     :: IArray.Array Int [Int]
    offsets     =  IArray.listArray (0, length shx-1) shx
    --
    get ix (i,t,v) = case boundary of
      Clamp                -> bounded "clamp"
      Mirror               -> bounded "mirror"
      Wrap                 -> bounded "wrap"
      Constant (CUExp _ c) -> inRange c
      where
        j       = 'j':shows base "_a" ++ show i
        k       = 'k':shows base "_a" ++ show i
        --
        bounded f
          = [cdecl| const int $id:j = $exp:ix'; |]
          : [cdecl| const $ty:t $id:(show v) = $exp:(indexArray t (cvar (arr i)) (cvar j)); |]
          : []
          where
            ix'  = case offsets IArray.! div i n of
              ks | all (== 0) ks        -> [cexp| toIndex( $id:sh, ix ) |]
                 | otherwise            -> [cexp| toIndex( $id:sh, $exp:(ccall f [cvar sh, cursor ks]) ) |]
        --
        inRange c = case offsets IArray.! div i n of
          ks | all (== 0) ks    -> let f = indexArray t (cvar (arr i)) (ccall "toIndex" [cvar sh, cvar "ix"])
                                   in  [[cdecl| const $ty:t $id:(show v) = $exp:f; |]]
             | otherwise        -> [cdecl| const typename Shape $id:j = $exp:(cursor ks); |]
                                 : [cdecl| const typename bool  $id:k = inRange( $id:sh, $id:j ); |]
                                 : [cdecl| const $ty:t $id:(show v) = $id:k ? $exp:(indexArray t (cvar (arr i)) (ccall "toIndex" [cvar sh, cvar j]))
                                                                            : $exp:(reverse c !! mod i n); |]
                                 : []
        --
        cursor [c] = [cexp| $id:ix + $int:c |]
        cursor cs  = ccall "shape" $ zipWith (\a c -> [cexp| $id:ix . $id:('a':show a) + $int:c |]) [dim-1,dim-2..0] cs

