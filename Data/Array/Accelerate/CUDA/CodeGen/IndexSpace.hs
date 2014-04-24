{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ImpredicativeTypes  #-}
{-# LANGUAGE PatternGuards       #-}
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

import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, ignore, shapeToList )
import Data.Array.Accelerate.CUDA.AST                   ( Gamma )
import Data.Array.Accelerate.CUDA.CodeGen.Type
import Data.Array.Accelerate.CUDA.CodeGen.Base

#include "accelerate.h"


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
mkGenerate dev aenv (CUFun1 dce f)
  = return
  $ CUTranslSkel "generate" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    generate
    (
        $params:argIn,
        $params:argOut
    )
    {
        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(dce sh      .=. cfromIndex shOut "ix" "tmp")
            $items:(setOut "ix" .=. f sh)
        }
    }
  |]
  where
    (sh, _, _)                  = locals "sh" (undefined :: sh)
    (texIn, argIn)              = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh e)


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
  | CUFun1 dce_p p                      <- perm
  , CUFun1 dce_f f                      <- fun
  , CUDelayed _ (CUFun1 dce_g get) _    <- arr
  = return
  $ CUTranslSkel "transform" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    transform
    (
        $params:argIn,
        $params:argOut
    )
    {
        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(dce_p sh'   .=. cfromIndex shOut "ix" "tmp")
            $items:(dce_g sh    .=. p sh')
            $items:(dce_f x0    .=. get sh)
            $items:(setOut "ix" .=. f x0)
        }
    }
  |]
  where
    (texIn, argIn)              = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh' b)
    (x0, _, _)                  = locals "x"   (undefined :: a)
    (sh, _, _)                  = locals "sh"  (undefined :: sh)
    (sh', _, _)                 = locals "sh_" (undefined :: sh')


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
mkPermute dev aenv (CUFun2 dce_x dce_y combine) (CUFun1 dce_p prj) arr
  | CUDelayed (CUExp shIn) _ (CUFun1 _ get) <- arr
  = return
  $ CUTranslSkel "permute" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    permute
    (
        $params:argIn,
        $params:argOut
    )
    {
        /*
         * The input shape might be a complex expression. Evaluate it first to reuse the result.
         */
        $items:(sh .=. shIn)

        const int shapeSize             = $exp:(csize sh);
        const int gridSize              = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(dce_p src   .=. cfromIndex sh "ix" "srcTmp")
            $items:(dst         .=. prj src)

            if ( ! $exp:(cignore dst) )
            {
                $decls:decly
                $decls:decly'

                $items:(jx      .=. ctoIndex shOut dst)
                $items:(dce_x x .=. get ix)
                $items:(dce_y y .=. arrOut jx)

                $items:write
            }
        }
    }
  |]
  where
    sizeof                      = eltSizeOf (undefined::e)
    (texIn, argIn)              = environment dev aenv
    (argOut, shOut, arrOut)     = writeArray "Out" (undefined :: Array sh' e)
    (x, _, _)                   = locals "x" (undefined :: e)
    (_, y,  decly)              = locals "y" (undefined :: e)
    (_, y', decly')             = locals "_y" (undefined :: e)
    (sh, _, _)                  = locals "shIn" (undefined :: sh)
    (src, _, _)                 = locals "sh" (undefined :: sh)
    (dst, _, _)                 = locals "sh_" (undefined :: sh')
    ([jx], _, _)                = locals "jx" (undefined :: Int)
    ix                          = [cvar "ix"]
    sm                          = computeCapability dev

    -- If the destination index resolves to the magic index "ignore", the result
    -- is dropped from the output array.
    cignore :: Rvalue x => [x] -> C.Exp
    cignore []  = INTERNAL_ERROR(error) "permute" "singleton arrays not supported"
    cignore xs  = foldl1 (\a b -> [cexp| $exp:a && $exp:b |])
                $ zipWith (\a b -> [cexp| $exp:(rvalue a) == $int:b |]) xs
                $ shapeToList (ignore :: sh')

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
    write       = env ++ zipWith6 apply sizeof (arrOut jx) fun x (dce_y y) y'
    (env, fun)  = combine x y

    apply size out f x1 (used,y1) y1'
      | used
      , Just atomicCAS <- reinterpret size
      = [citem| do {
                       $exp:y1' = $exp:y1;
                       $exp:y1  = $exp:atomicCAS ( & $exp:out, $exp:y1', $exp:f );

                   } while ( $exp:y1 != $exp:y1' ); |]

      | otherwise
      = [citem| $exp:out = $exp:(rvalue x1); |]

    --
    reinterpret :: Int -> Maybe C.Exp
    reinterpret 4 | sm >= Compute 1 1   = Just [cexp| $id:("atomicCAS32") |]
    reinterpret 8 | sm >= Compute 1 2   = Just [cexp| $id:("atomicCAS64") |]
    reinterpret _                       = Nothing

