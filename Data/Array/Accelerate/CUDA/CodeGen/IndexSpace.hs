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
  mkPermute, mkBackpermute,

  -- Multidimensional index and replicate
  mkSlice, mkReplicate

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
  return $ CUTranslSkel "generate" [cunit|
    $edecl:(cdim "DimOut" dimOut)

    extern "C"
    __global__ void
    generate
    (
        $params:args,
        const typename DimOut shOut
    )
    {
              int ix;
        const int n        = size(shOut);
        const int gridSize = __umul24(blockDim.x, gridDim.x);

        for ( ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; ix < n
            ; ix += gridSize)
        {
            $decls:shape
            $decls:env
            $stms:(set "ix" fn)
        }
    }
  |]
  where
    (args, _, set)      = setters tyOut
    shape               = fromIndex dimOut "DimOut" "shOut" "ix" "x0"


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
  return $ CUTranslSkel "permute" [cunit|
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
              int ix;
        const int shapeSize = size(shIn0);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);

        for ( ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; ix < shapeSize
            ; ix += gridSize)
        {
            typename DimOut dst;
            $decls:src
            $decls:env
            $stms:dst

            if (!ignore(dst))
            {
                const int jx = toIndex(shOut, dst);
                $decls:(getIn0 "ix")
                $decls:(getOut "jx")
                $stms:(setOut "jx" combine)
            }
        }
    }
  |]
  where
    (argOut, _, setOut) = setters types
    (argIn0, _, getIn0) = getters 0 types
    (_,      _, getOut) = getters' "d_out" "x1" types
    src                 = fromIndex dimIn0 "DimIn0" "shIn0" "ix" "x0"
    dst                 = project dimOut "dst" index

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
                                      $id:x = atomicCAS( & $id:a [jx], $id:x', $exp:f);
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
  return $ CUTranslSkel "backpermute" [cunit|
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
              int ix;
        const int shapeSize = size(shOut);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);

        for ( ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; ix < shapeSize
            ; ix += gridSize)
        {
            typename DimIn0 src;
            $decls:dst
            $decls:env
            $stms:src
            {
                const int jx = toIndex(shIn0, src);
                $decls:(getIn0 "jx")
                $stms:(setOut "ix" x0)
            }
        }
    }
  |]
  where
    (argOut, _,  setOut)        = setters types
    (argIn0, x0, getIn0)        = getters 0 types
    dst                         = fromIndex dimOut "DimOut" "shOut" "ix" "x0"
    src                         = project dimIn0 "src" index


-- Index an array with a generalised, multidimensional array index. The result
-- is a new array (possibly a singleton) containing all dimensions in their
-- entirety.
--
-- slice :: (Slice slix, Elt e)
--       => Acc (Array (FullShape slix) e)
--       -> Exp slix
--       -> Acc (Array (SliceShape slix) e)
--
mkSlice :: Int -> Int -> Int -> [Type] -> [Exp] -> CGM CUTranslSkel
mkSlice dimSl dimCo dimIn0 types slix = do
  env   <- environment
  return $ CUTranslSkel "slice" [cunit|
    $edecl:(cdim "Slice"    dimSl)
    $edecl:(cdim "CoSlice"  dimCo)
    $edecl:(cdim "SliceDim" dimIn0)

    extern "C"
    __global__ void
    slice
    (
        $params:argOut,
        $params:argIn0,
        const typename Slice    slice,
        const typename CoSlice  co,
        const typename SliceDim sliceDim
    )
    {
              int ix;
        const int shapeSize = size(slice);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);

        for ( ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; ix < shapeSize
            ; ix += gridSize)
        {
            typename Slice    sl  = fromIndex(slice, ix);
            typename SliceDim src;
            $decls:env
            $stms:src
            {
                const int jx = toIndex(sliceDim, src);
                $decls:(getIn0 "jx")
                $stms:(setOut "ix" x0)
            }
        }
    }
  |]
  where
    (argOut, _,  setOut)        = setters types
    (argIn0, x0, getIn0)        = getters 0 types
    src                         = project dimIn0 "sl" slix


-- Replicate an array across one or more dimensions as specified by the
-- generalised array index.
--
-- replicate :: (Slice slix, Elt e)
--           => Exp slix
--           -> Acc (Array (SliceShape slix) e)
--           -> Acc (Array (FullShape  slix) e)
--
mkReplicate :: Int -> Int -> [Type] -> [Exp] -> CGM CUTranslSkel
mkReplicate dimSl dimOut types slix = do
  env   <- environment
  return $ CUTranslSkel "replicate" [cunit|
    $edecl:(cdim "Slice"    dimSl)
    $edecl:(cdim "SliceDim" dimOut)

    extern "C"
    __global__ void
    replicate
    (
        $params:argOut,
        $params:argIn0,
        const typename Slice    slice,
        const typename SliceDim sliceDim
    )
    {
              int ix;
        const int shapeSize = size(sliceDim);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);

        for ( ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; ix < shapeSize
            ; ix += gridSize)
        {
            typename SliceDim dim = fromIndex(sliceDim, ix);
            typename Slice    src;
            $decls:env
            $stms:src
            {
                const int jx = toIndex(slice, src);
                $decls:(getIn0 "jx")
                $stms:(setOut "ix" x0)
            }
        }
    }
  |]
  where
    (argOut, _,  setOut)        = setters types
    (argIn0, x0, getIn0)        = getters 0 types
    src                         = project dimSl "src" slix


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


-- apply expressions to the components of a shape
--
project :: Int -> String -> [Exp] -> [Stm]
project n sh idx
  | [e] <- idx  = [[cstm| $id:sh = $exp:e; |]]
  | otherwise   = zipWith (\i c -> [cstm| $id:sh . $id:('a':show c) = $exp:i; |]) idx [n-1,n-2..0]

