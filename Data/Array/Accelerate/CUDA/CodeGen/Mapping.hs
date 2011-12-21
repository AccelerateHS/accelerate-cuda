{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.IndexSpace
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Mapping (

  mkMap, mkZipWith

) where

import Data.Loc
import Data.Symbol
import Language.C.Syntax
import Language.C.Quote.CUDA

import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Monad


-- Apply the given unary function to each element of an array. Each thread
-- processes multiple elements, striding the array by the grid size.
--
-- map :: (Shape sh, Elt a, Elt b)
--     => (Exp a -> Exp b)
--     -> Acc (Array sh a)
--     -> Acc (Array sh b)
--
mkMap :: [Type] -> [Type] -> [Exp] -> CGM CUTranslSkel
mkMap tyOut tyIn0 fn = do
  env   <- environment
  return $ CUTranslSkel [cunit|
    extern "C"
    __global__ void
    map
    (
        $params:argOut,
        $params:argIn0,
        const typename Ix shape
    )
    {
              int idx;
        const int gridSize = __umul24(blockDim.x, gridDim.x);

        for ( idx = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; idx < shape
            ; idx += gridSize)
        {
            $decls:(getIn0 "idx")
            $decls:env
            $stms:(zipWith apply fn varOut)
        }
    }
  |]
  where
    (argOut, varOut)    = setters tyOut
    (argIn0, getIn0)    = getters 0 tyIn0
    apply f x           = [cstm| $exp:x [idx] = $exp:f; |]


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
mkZipWith :: Int -> [Type] -> [Type] -> [Type] -> [Exp] -> CGM CUTranslSkel
mkZipWith dim tyOut tyIn1 tyIn0 fn = do
  env   <- environment
  return $ CUTranslSkel [cunit|
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
              int idx;
        const int shapeSize = size(shOut);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);

        for ( idx = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; idx < shapeSize
            ; idx += gridSize)
        {
            const int idx1 = toIndex(shIn1, fromIndex(shOut, idx));
            const int idx0 = toIndex(shIn0, fromIndex(shOut, idx));
            $decls:(getIn0 "idx0")
            $decls:(getIn1 "idx1")
            $decls:env
            $stms:(zipWith apply fn varOut)
        }
    }
  |]
  where
    (argOut, varOut)    = setters tyOut
    (argIn0, getIn0)    = getters 0 tyIn0
    (argIn1, getIn1)    = getters 1 tyIn1
    apply f x           = [cstm| $exp:x [idx] = $exp:f; |]

