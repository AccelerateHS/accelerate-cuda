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

module Data.Array.Accelerate.CUDA.CodeGen.IndexSpace (

  -- Array construction
  mkGenerate,

  -- Permutations
  mkPermute, mkBackpermute

) where

import Data.Loc
import Data.Symbol
import Language.C.Syntax
import Language.C.Quote.CUDA

import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Monad


-- Construct a new array by applying a function to each index. Each thread
-- processes multiple elements, striding the array by the grid size.
--
-- generate :: (Shape ix, Elt e)
--          => Exp ix
--          -> (Exp ix -> Exp a)
--          -> Acc (Array ix a)
--
mkGenerate :: Int -> [Type] -> [Exp] -> CGM CUTranslSkel
mkGenerate dimOut tyOut fn = do
  env   <- environment
  return $ CUTranslSkel [cunit|
    $edecl:(cdim "DimOut" dimOut)

    extern "C"
    __global__ void
    generate
    (
        $params:args,
        const typename DimOut shOut
    )
    {
              int idx;
        const int n        = size(shOut);
        const int gridSize = __umul24(blockDim.x, gridDim.x);

        for ( idx = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; idx < n
            ; idx += gridSize)
        {
              $decls:shape
              $decls:env
              $stms:(zipWith apply fn xs)
        }
    }
  |]
  where
    (args, xs)  = setters tyOut
    apply f x   = [cstm| $exp:x [idx] = $exp:f; |]
    shape       = fromIndex dimOut "DimOut" "shOut" "idx" "x0"


-- Forward permutation specified by an index mapping that determines for each
-- element in the source array where it should go in the target. The resultant
-- array is initialised with the given defaults and any further values that are
-- permuted into the result array are added to the current value using the given
-- combination function.
--
-- The combination function must be associative. Extents that are mapped to the
-- magic value 'ignore' by the permutation function are dropped.
--
-- permute :: (Shape ix, Shape ix', Elt a)
--         => (Exp a -> Exp a -> Exp a)         -- combination function
--         -> Acc (Array ix' a)                 -- array of default values
--         -> (Exp ix -> Exp ix')               -- permutation
--         -> Acc (Array ix  a)                 -- permuted array
--         -> Acc (Array ix' a)
--
mkPermute :: Int -> Int -> [Type] -> [Exp] -> [Exp] -> CGM CUTranslSkel
mkPermute dimOut dimIn0 types combine index = do
  env   <- environment
  return $ CUTranslSkel [cunit|
    $edecl:(cdim "DimOut" dimOut)
    $edecl:(cdim "DimIn0" dimIn0)

    extern "C"
    __global__ void
    permute
    (
        $params:argOut,
        $params:argIn0,
        const typename DimOut shOut,
        const typename DimIn0 shIn0
    )
    {
              int idx;
        const int shapeSize = size(shIn0);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);

        for ( idx = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; idx < shapeSize
            ; idx += gridSize)
        {
            typename DimOut dst;
            $decls:src
            $decls:env
            $stms:project

            if (!ignore(dst))
            {
                const int j = toIndex(shOut, dst);
                $decls:(getIn0 "idx")
                $decls:(getOut "j")
                $stms:(zipWith apply combine varOut)
            }
        }
    }
  |]
  where
    (argOut, varOut)    = setters types
    (argIn0, getIn0)    = getters 0 types
    (_,      getOut)    = getters' "d_out" "x1" types
    src                 = fromIndex dimIn0 "DimIn0" "shIn0" "idx" "x0"
    apply f x           = [cstm| $exp:x [j] = $exp:f; |]
    project
      | [e] <- index    = [[cstm| dst = $exp:e; |]]
      | otherwise       = zipWith (\f c -> [cstm| dst . $id:('a':show c) = $exp:f; |]) index [dimOut-1, dimOut-2 .. 0]

{--
    -- A version of 'apply' using atomicCAS which will correctly combine
    -- elements that write to the same location (e.g. histogram). Requires type
    -- casting, and only works for 32-bit (compute > 1.1) or 64-bit
    -- (compute > 1.2) elements. This operates on each element of a tuple
    -- individually, which could put restrictions on the combining function.
    --
    n                   = length types
    suf x               = map (\c -> x ++ "_a" ++ show c) [n-1, n-2.. 0]
    varOut'             = zipWith (\t v -> [cdecl| $ty:t $id:v; |]) types (suf "v1")
    applyCAS f a x x'   = [cstm| do { $id:x' = $id:x;
                                      $id:x = atomicCAS( & $id:a [j], $id:x', $exp:f);
                                    } while ( $id:x' != $id:x ); |]
--}


-- Backwards permutation (gather) of an array according to a permutation
-- function.
--
-- backpermute :: (Shape ix, Shape ix', Elt a)
--             => Exp ix'                       -- shape of the result array
--             -> (Exp ix' -> Exp ix)           -- permutation
--             -> Acc (Array ix  a)             -- permuted array
--             -> Acc (Array ix' a)
--
mkBackpermute :: Int -> Int -> [Type] -> [Exp] -> CGM CUTranslSkel
mkBackpermute dimOut dimIn0 types index = do
  env   <- environment
  return $ CUTranslSkel [cunit|
    $edecl:(cdim "DimOut" dimOut)
    $edecl:(cdim "DimIn0" dimIn0)

    extern "C"
    __global__ void
    backpermute
    (
        $params:argOut,
        $params:argIn0,
        const typename DimOut shOut,
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
            typename DimIn0 src;
            $decls:dst
            $decls:env
            $stms:project
            {
                const int i = toIndex(shIn0, src);
                $decls:(getIn0 "i")
                $stms:(zipWith permute varOut [n-1,n-2..0])
            }
        }
    }
  |]
  where
    (argOut, varOut)    = setters types
    (argIn0, getIn0)    = getters 0 types
    dst                 = fromIndex dimOut "DimOut" "shOut" "idx" "x0"
    n                   = length types
    permute v c         = [cstm| $exp:v [idx] = $id:("x0_a" ++ show c); |]
    project
      | [e] <- index    = [[cstm| src = $exp:e; |]]
      | otherwise       = zipWith (\f c -> [cstm| src . $id:('a':show c) = $exp:f; |]) index [dimIn0-1, dimIn0-2.. 0]


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- destruct shapes into separate components, since the code generator no
-- longer treats tuples as structs
--
fromIndex :: Int -> String -> String -> String -> String -> [InitGroup]
fromIndex n dim sh ix base
  | n == 1      = [[cdecl| const int $id:(base ++ "_a0") = $id:ix; |]]
  | otherwise   = sh0 : map (unsh . show) [0 .. n-1]
    where
      sh0       = [cdecl| const typename $id:dim $id:base = fromIndex( $id:sh , $id:ix ); |]
      unsh c    = [cdecl| const int $id:(base ++ "_a" ++ c) = $id:base . $id:('a':c); |]

