{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS -fno-warn-incomplete-patterns #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.IndexSpace
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
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

import Data.List
import Language.C.Syntax
import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis

import Data.Array.Accelerate.Array.Sugar                ( Array, Elt )
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type


-- Construct a new array by applying a function to each index. Each thread
-- processes multiple elements, striding the array by the grid size.
--
-- generate :: (Shape ix, Elt e)
--          => Exp ix
--          -> (Exp ix -> Exp a)
--          -> Acc (Array ix a)
--
mkGenerate :: forall sh e. Elt e => Int -> CUFun (sh -> e) -> CUTranslSkel
mkGenerate dimOut (CULam _ (CUBody (CUExp env fn))) =
  CUTranslSkel "generate" [cunit|
    $edecl:(cdim "DimOut" dimOut)

    extern "C"
    __global__ void
    generate
    (
        $params:args,
        const typename DimOut shOut
    )
    {
        const int n        = size(shOut);
        const int gridSize = __umul24(blockDim.x, gridDim.x);
              int ix;

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
    tyOut               = eltType (undefined :: e)
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
mkPermute :: forall a ix ix'.
             DeviceProperties
          -> Int                                -- dimensionality ix'
          -> Int                                -- dimensionality ix
          -> CUFun (a -> a -> a)
          -> CUFun (ix -> ix')
          -> CUTranslSkel
mkPermute dev dimOut dimIn0 (CULam useFn (CULam _ (CUBody (CUExp env combine)))) (CULam _ (CUBody (CUExp envIx prj))) =
  CUTranslSkel "permute" [cunit|
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
        const int shapeSize = size(shIn0);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);
              int ix;

        for ( ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; ix < shapeSize
            ; ix += gridSize)
        {
            typename DimOut dst;
            $decls:src
            $decls:envIx
            $stms:dst

            if (!ignore(dst))
            {
                const int jx = toIndex(shOut, dst);
                $decls:decl1
                $decls:temps
                $decls:env
                $stms:(x1 .=. getIn0 "ix")
                $stms:write
            }
        }
    }
  |]
  where
    elt                         = eltType   (undefined :: a)
    sizeof                      = eltSizeOf (undefined :: a)
    (argIn0, _, _, getIn0, _)   = getters 0 elt useFn
    (_, x1, decl1, _, _)        = getters 1 elt useFn
    (argOut, arrOut,  setOut)   = setters elt
    (x0, _)                     = locals "x0" elt
    src                         = fromIndex dimIn0 "DimIn0" "shIn0" "ix" "x0"
    dst                         = project dimOut "dst" prj
    sm                          = computeCapability dev
    unsafe                      = setOut "jx" combine
    (temps, write)              = unzip $ zipWith6 apply unsafe combine elt arrOut x0 sizeof
    --
    -- Apply the combining function between old and new values. If multiple
    -- threads attempt to write to the same location, the hardware
    -- write-combining mechanism will accept one transaction and all other
    -- updates will be lost.
    --
    -- If the hardware supports it, we can use atomicCAS (compare-and-swap) to
    -- work around this. This requires at least compute 1.1 for 32-bit values,
    -- and compute 1.2 for 64-bit values. If hardware support is not available,
    -- write the result as normal and hope for the best.
    --
    -- Each element of a tuple is necessarily written individually, so the tuple
    -- as a whole is not stored atomically.
    --
    apply set f t a z s
      | Just atomicCAS <- reinterpret s
      = let z'        = [cexp| $id:('_':show z) |]
        in
        ( [cdecl| $ty:t $id:(show z), $id:(show z') = $exp:a [ $id:("jx") ]; |]
        , [cstm| do { $exp:z  = $exp:z';
                      $exp:z' = $exp:atomicCAS ( & $exp:a [ $id:("jx") ], $exp:z, $exp:f );
                    } while ( $exp:z != $exp:z' ); |]
        )

      | otherwise
      = ( [cdecl| const $ty:t $id:(show z) = $exp:a [ $id:("jx") ]; |]
        , set
        )
    --
    reinterpret :: Int -> Maybe Exp
    reinterpret 4 | sm >= 1.1   = Just [cexp| $id:("atomicCAS32") |]
    reinterpret 8 | sm >= 1.2   = Just [cexp| $id:("atomicCAS64") |]
    reinterpret _               = Nothing


-- Backwards permutation (gather) of an array according to a permutation
-- function.
--
-- backpermute :: (Shape ix, Shape ix', Elt a)
--             => Exp ix'                       -- shape of the result array
--             -> (Exp ix' -> Exp ix)           -- permutation
--             -> Acc (Array ix  a)             -- permuted array
--             -> Acc (Array ix' a)
--
mkBackpermute :: forall ix ix' a. Elt a
              => Int                            -- dimensionality ix'
              -> Int                            -- dimensionality ix
              -> CUFun (ix' -> ix)
              -> Array ix' a                    -- dummy to fix type variables
              -> CUTranslSkel
mkBackpermute dimOut dimIn0 (CULam _ (CUBody (CUExp env prj))) _ =
  CUTranslSkel "backpermute" [cunit|
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
        const int shapeSize = size(shOut);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);
              int ix;

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
                $stms:(setOut "ix" (reverse x0))
            }
        }
    }
  |]
  where
    elt                         = eltType (undefined :: a)
    (argOut, _, setOut)         = setters elt
    (argIn0, x0, _, _, getIn0)  = getters 0 elt (useAll 0 elt)
    dst                         = fromIndex dimOut "DimOut" "shOut" "ix" "x0"
    src                         = project dimIn0 "src" prj


-- Index an array with a generalised, multidimensional array index. The result
-- is a new array (possibly a singleton) containing all dimensions in their
-- entirety.
--
-- slice :: (Slice slix, Elt e)
--       => Acc (Array (FullShape slix) e)
--       -> Exp slix
--       -> Acc (Array (SliceShape slix) e)
--
mkSlice :: forall sl slix e. Elt e
        => Int                  -- dimensionality sl
        -> Int                  -- dimensionality co
        -> Int                  -- dimensionality sh
        -> CUExp slix
        -> Array sl e           -- dummy
        -> CUTranslSkel
mkSlice dimSl dimCo dimIn0 (CUExp [] slix) _ =
  CUTranslSkel "slice" [cunit|
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
    elt                         = eltType (undefined :: e)
    (argOut, _, setOut)         = setters elt
    (argIn0, x0, _, _, getIn0)  = getters 0 elt (useAll 0 elt)
    src                         = project dimIn0 "src" slix


-- Replicate an array across one or more dimensions as specified by the
-- generalised array index.
--
-- replicate :: (Slice slix, Elt e)
--           => Exp slix
--           -> Acc (Array (SliceShape slix) e)
--           -> Acc (Array (FullShape  slix) e)
--
mkReplicate :: forall sh slix e. Elt e
            => Int              -- dimensionality sl
            -> Int              -- dimensionality sh
            -> CUExp slix
            -> Array sh e       -- dummy
            -> CUTranslSkel
mkReplicate dimSl dimOut (CUExp _ slix) _ =
  CUTranslSkel "replicate" [cunit|
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
    elt                         = eltType (undefined :: e)
    (argOut, _, setOut)         = setters elt
    (argIn0, x0, _, _, getIn0)  = getters 0 elt (useAll 0 elt)
    src                         = project dimSl "src" slix



--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- destruct shapes into separate components, since the code generator no
-- longer treats tuples as structs
--
fromIndex :: Int -> String -> String -> String -> String -> [InitGroup]
fromIndex n dim sh ix base
  | n == 1      = [[cdecl| const $ty:int $id:(base ++ "_a0") = $id:ix; |]]
  | otherwise   = sh0 : map (unsh . show) [0 .. n-1]
    where
#if   SIZEOF_HSINT == 4
      int       = typename "Int32"
#elif SIZEOF_HSINT == 8
      int       = typename "Int64"
#endif
      sh0       = [cdecl| const typename $id:dim $id:base = fromIndex( $id:sh , $id:ix ); |]
      unsh c    = [cdecl| const int $id:(base ++ "_a" ++ c) = $id:base . $id:('a':c); |]


-- apply expressions to the components of a shape
--
project :: Int -> String -> [Exp] -> [Stm]
project n sh idx
  | n   == 0    = [[cstm| $id:sh = 0; |]]
  | [e] <- idx  = [[cstm| $id:sh = $exp:e; |]]
  | otherwise   = zipWith (\i c -> [cstm| $id:sh . $id:('a':show c) = $exp:i; |]) idx [n-1,n-2..0]


-- tell the getters function that we will use all the scalar components
--
useAll :: Int -> [Type] -> [(Int, Type, Exp)]
useAll base elt =
  let n   = length elt
      x i = 'x' : shows base "_a" ++ show i
  in
  zipWith (\i t -> (i,t, cvar (x i))) [n-1, n-2 .. 0] elt

