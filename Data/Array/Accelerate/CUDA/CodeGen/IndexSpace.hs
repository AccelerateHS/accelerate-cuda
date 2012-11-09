{-# LANGUAGE GADTs               #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
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
  mkTransform, mkPermute,

) where

import Data.List
import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis.Device
import qualified Language.C.Syntax                      as C

import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.CUDA.AST                   ( Gamma, Exp )
import Data.Array.Accelerate.CUDA.CodeGen.Type
import Data.Array.Accelerate.CUDA.CodeGen.Base


-- Construct a new array by applying a function to each index. Each thread
-- processes multiple elements, striding the array by the grid size.
--
-- generate :: (Shape ix, Elt e)
--          => Exp ix                           -- dimension of the result
--          -> (Exp ix -> Exp a)                -- function to apply at each index
--          -> Acc (Array ix a)
--
mkGenerate
    :: forall aenv sh e. (Shape sh, Elt e)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun1 aenv (sh -> e)
    -> [CUTranslSkel aenv (Array sh e)]
mkGenerate dev aenv (CUFun1 f)
  = return
  $ CUTranslSkel "generate" [cunit|

    $esc:("#include <accelerate_cuda_extras.h>")
    $edecl:(cdim "DimOut" dim)
    $edecls:texIn

    extern "C" __global__ void
    generate
    (
        $params:argIn,
        $params:argOut
    )
    {
        const int shapeSize     = size(shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            const typename DimOut sh = fromIndex(shOut, ix);

            $items:(setOut "ix" .=. f sh)
        }
    }
  |]
  where
    dim                 = expDim (undefined :: Exp aenv sh)
    sh                  = cshape dim (cvar "sh")
    (texIn, argIn)      = environment dev aenv
    (argOut, setOut)    = setters "Out" (undefined :: Array sh e)


-- A combination map/backpermute, where the index and value transformations have
-- been separated.
--
-- transform :: (Elt a, Elt b, Shape sh, Shape sh')
--           => PreExp     acc aenv sh'                 -- dimension of the result
--           -> PreFun     acc aenv (sh' -> sh)         -- index permutation function
--           -> PreFun     acc aenv (a   -> b)          -- function to apply at each element
--           ->            acc aenv (Array sh  a)       -- source array
--           -> PreOpenAcc acc aenv (Array sh' b)
--
mkTransform
    :: forall aenv sh sh' a b. (Shape sh, Shape sh', Elt a, Elt b)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun1 aenv (sh' -> sh)
    -> CUFun1 aenv (a -> b)
    -> CUDelayedAcc aenv sh a
    -> [CUTranslSkel aenv (Array sh' b)]
mkTransform dev aenv perm fun arr
  | CUFun1 p                    <- perm
  , CUFun1 f                    <- fun
  , CUDelayed _ (CUFun1 get) _  <- arr
  = return
  $ CUTranslSkel "transform" [cunit|

    $esc:("#include <accelerate_cuda_extras.h>")
    $edecl:(cdim "DimOut" dimOut)
    $edecl:(cdim "DimIn"  dimIn)
    $edecls:texIn

    extern "C" __global__ void
    transform
    (
        $params:argIn,
        $params:argOut
    )
    {
        const int shapeSize     = size(shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            const typename DimOut sh_ = fromIndex(shOut, ix);
            $items:(sh          .=. p sh_)
            $items:(x0          .=. get sh)
            $items:(setOut "ix" .=. f x0)
        }
    }
  |]
  where
    dimIn               = expDim (undefined :: Exp aenv sh)
    dimOut              = expDim (undefined :: Exp aenv sh')
    sh_                 = cshape dimOut (cvar "sh_")
    (texIn, argIn)      = environment dev aenv
    (argOut, setOut)    = setters "Out" (undefined :: Array sh' b)
    (x0, _, _)          = locals "x"  (undefined :: a)
    (sh, _, _)          = locals "sh" (undefined :: sh)


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
mkPermute
    :: forall aenv sh sh' e. (Shape sh, Shape sh', Elt e)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> CUFun1 aenv (sh -> sh')
    -> CUDelayedAcc aenv sh e
    -> [CUTranslSkel aenv (Array sh' e)]
mkPermute dev aenv (CUFun2 combine) (CUFun1 prj) arr
  | CUDelayed (CUExp shIn) _ (CUFun1 get) <- arr
  = return
  $ CUTranslSkel "permute" [cunit|

    $esc:("#include <accelerate_cuda_extras.h>")
    $edecl:(cdim "DimOut" dimOut)
    $edecl:(cdim "DimIn"  dimIn)
    $edecls:texIn

    extern "C" __global__ void
    permute
    (
        $params:argIn,
        $params:argOut
    )
    {
        $items:(sh .=. shIn)
        const typename DimIn shIn       = $exp:(ccall "shape" (map rvalue sh));
        const int shapeSize             = $exp:(shapeSize sh);
        const int gridSize              = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            typename DimOut dst;

            const int src = fromIndex( shIn, ix );
            $items:(dst .=. prj src)

            if ( !ignore(dst) )
            {
                $decls:decly
                $decls:decly'

                const int jx = toIndex(shOut, dst);
                $items:(x .=. get ix)
                $items:(y .=. arrOut "jx")

                $items:write
            }
        }
    }
  |]
  where
    dimIn               = expDim (undefined :: Exp aenv sh)
    dimOut              = expDim (undefined :: Exp aenv sh')
    sizeof              = eltSizeOf (undefined::e)
    (texIn, argIn)      = environment dev aenv
    (argOut, arrOut)    = setters "Out" (undefined :: Array sh' e)
    (sh, _, _)          = locals "sh" (undefined :: sh)
    (x, _, _)           = locals "x"  (undefined :: e)
    (_, y, decly)       = locals "y"  (undefined :: e)
    (_, y', decly')     = locals "_y" (undefined :: e)
    ix                  = [cvar "ix"]
    src                 = cshape dimIn  (cvar "src")
    dst                 = cshape dimOut (cvar "dst")
    sm                  = computeCapability dev

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
    write               = env ++ zipWith5 apply sizeof (arrOut "jx") fun y y'
    (env, fun)          = combine x y

    apply size out f x1 x1'
      | Just atomicCAS <- reinterpret size
      = C.BlockStm
        [cstm| do {
                      $exp:x1' = $exp:x1;
                      $exp:x1  = $exp:atomicCAS ( & $exp:out, $exp:x1', $exp:f );

                  } while ( $exp:x1 != $exp:x1' ); |]

      | otherwise
      = C.BlockStm [cstm| $exp:out = $exp:x1; |]

    --
    reinterpret :: Int -> Maybe C.Exp
    reinterpret 4 | sm >= Compute 1 1   = Just [cexp| $id:("atomicCAS32") |]
    reinterpret 8 | sm >= Compute 1 2   = Just [cexp| $id:("atomicCAS64") |]
    reinterpret _                       = Nothing

