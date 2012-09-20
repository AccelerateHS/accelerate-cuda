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

module Data.Array.Accelerate.CUDA.CodeGen.Mapping (

  mkMap, mkZipWith

) where

import Language.C.Quote.CUDA
import Data.Array.Accelerate.Array.Sugar                ( Elt )
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type


-- Apply the given unary function to each element of an array. Each thread
-- processes multiple elements, striding the array by the grid size.
--
-- map :: (Shape sh, Elt a, Elt b)
--     => (Exp a -> Exp b)
--     -> Acc (Array sh a)
--     -> Acc (Array sh b)
--
mkMap :: forall a b. Elt b => CUFun (a -> b) -> CUTranslSkel
mkMap (CULam use0 (CUBody (CUExp env fn))) =
  CUTranslSkel "map" [cunit|
    extern "C"
    __global__ void
    map
    (
        $params:argOut,
        $params:argIn0,
        const typename Ix num_elements
    )
    {
        const int gridSize = __umul24(blockDim.x, gridDim.x);
              int ix;

        for ( ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; ix < num_elements
            ; ix += gridSize)
        {
            $decls:(getIn0 "ix")
            $items:env
            $stms:(setOut "ix" fn)
        }
    }
  |]
  where
    tyIn0                       = eltType (undefined :: a)
    tyOut                       = eltType (undefined :: b)
    (argIn0, _, _, _, getIn0)   = getters 0 tyIn0 use0
    (argOut, _, setOut)         = setters tyOut


-- Apply the given binary function element-wise to the two arrays. The extent of
-- the resulting array is the intersection of the extents of the two source
-- arrays. Each thread processes multiple elements, striding the array by the
-- grid size.
--
-- zipWith :: (Shape ix, Elt a, Elt b, Elt c)
--         => (Exp a -> Exp b -> Exp c)
--         -> Acc (Array ix a)
--         -> Acc (Array ix b)
--         -> Acc (Array ix c)
--
mkZipWith :: forall a b c. Elt c => Int -> CUFun (a -> b -> c) -> CUTranslSkel
mkZipWith dim (CULam use1 (CULam use0 (CUBody (CUExp env fn)))) =
  CUTranslSkel "zipWith" [cunit|
    $edecl:(cdim "DimOut" dim)
    $edecl:(cdim "DimIn0" dim)
    $edecl:(cdim "DimIn1" dim)

    extern "C"
    __global__ void
    zipWith
    (
        $params:argOut,
        $params:argIn1,
        $params:argIn0,
        const typename DimOut shOut,
        const typename DimIn1 shIn1,
        const typename DimIn0 shIn0
    )
    {
        const int shapeSize = size(shOut);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);
              int ix;

        for ( ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; ix < shapeSize
            ; ix += gridSize)
        {
            const int ix1 = toIndex(shIn1, fromIndex(shOut, ix));
            const int ix0 = toIndex(shIn0, fromIndex(shOut, ix));
            $decls:(getIn0 "ix0")
            $decls:(getIn1 "ix1")
            $items:env
            $stms:(setOut "ix" fn)
        }
    }
  |]
  where
    tyIn1                       = eltType (undefined :: a)
    tyIn0                       = eltType (undefined :: b)
    tyOut                       = eltType (undefined :: c)
    (argIn1, _, _, _, getIn1)   = getters 1 tyIn1 use1
    (argIn0, _, _, _, getIn0)   = getters 0 tyIn0 use0
    (argOut, _, setOut)         = setters tyOut

