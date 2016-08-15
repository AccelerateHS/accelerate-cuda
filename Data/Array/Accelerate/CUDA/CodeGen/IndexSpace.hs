{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ImpredicativeTypes  #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE ViewPatterns        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.IndexSpace
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
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

import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis.Device
import qualified Language.C.Syntax                      as C

import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, ignore, shapeToList )
import Data.Array.Accelerate.Error                      ( internalError )
import Data.Array.Accelerate.CUDA.AST                   ( Gamma )
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
        $params:argOut,
        typename Int32 * __restrict__ lock
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
                $items:(jx        .=. ctoIndex shOut dst)
                $items:(dce_x x   .=. get ix)

                $items:(atomically jx
                    [ dce_y y   .=. setOut jx
                    , setOut jx .=. combine x y
                    ]
                )
            }
        }
    }
  |]
  where
    (texIn, argIn)              = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh' e)
    (x, _, _)                   = locals "x" (undefined :: e)
    (y, _, _)                   = locals "y" (undefined :: e)
    (sh, _, _)                  = locals "shIn" (undefined :: sh)
    (src, _, _)                 = locals "sh" (undefined :: sh)
    (dst, _, _)                 = locals "sh_" (undefined :: sh')
    ([jx], _, _)                = locals "jx" (undefined :: Int)
    ix                          = [cvar "ix"]
    sm                          = computeCapability dev

    -- If the destination index resolves to the magic index "ignore", the result
    -- is dropped from the output array.
    --
    cignore :: Rvalue x => [x] -> C.Exp
    cignore []  = $internalError "permute" "singleton arrays not supported"
    cignore xs  = foldl1 (\a b -> [cexp| $exp:a && $exp:b |])
                $ zipWith (\a b -> [cexp| $exp:(rvalue a) == $int:b |]) xs
                $ shapeToList (ignore :: sh')

    -- If we can determine that the old values are not used in the combination
    -- function (e.g. filter) then the lock and unlock fragments can be replaced
    -- with a NOP.
    --
    -- If locking is required but the hardware does not support it (compute 1.0)
    -- then we issue a runtime error immediately instead of silently failing.
    --
    mustLock    = or . fst . unzip $ dce_y y

    -- The atomic section is acquired using a spin lock. This requires a
    -- temporary array to represent the lock state for each element of the
    -- output. We use 1 to represent the locked state, and 0 to represent
    -- unlocked elements.
    --
    --   do {
    --     old = atomicExch(&lock[i], 1);       // atomic exchange
    --   } while (old == 1);
    --
    --   /* critical section */
    --
    --   atomicExch(&lock[i], 0);
    --
    -- The initial loop repeatedly attempts to take the lock by writing a 1 into
    -- the slot. Once the 'old' state of the lock returns 0 (unlocked), we have
    -- just acquired the lock, and the atomic section can be computed. Finally,
    -- atomically write a 0 back into the slot to unlock the element.
    --
    -- However, there is a complication with CUDA devices because all threads in
    -- the warp must execute in lockstep (with predicated execution). Once a
    -- thread acquires a lock, then it will be disabled and stop participating
    -- in the first loop, waiting until all other threads in the warp acquire
    -- their locks. If two threads in a warp are attempting to acquire the same
    -- lock, once the lock is acquired by the first thread, it sits idle while
    -- the second thread spins attempting to grab a lock that will never be
    -- released, because the first thread can not progress. DEADLOCK.
    --
    -- So, we need to invert the algorithm so that threads can always make
    -- progress, until each thread in the warp has committed their result.
    --
    --   done = 0;
    --   do {
    --       if (atomicExch(&lock[i], 1) == 0) {
    --
    --           /* critical section */
    --
    --           done = 1;
    --           atomicExch(&lock[i], 0);
    --       }
    --   } while (done == 0)
    --
    atomically :: (C.Type, Name) -> [[C.BlockItem]] -> [C.BlockItem]
    atomically (_,i) (concat -> body)
      | not mustLock            = body
      | sm < Compute 1 1        = $internalError "permute" "Requires at least compute compatibility 1.1"
      | otherwise               =
        [ [citem| typename Int32 done = 0; |]
        , [citem| do {
                      typename Int32 *addr = &lock[ $exp:(cvar i) ];

                      if ( atomicExch( addr, 1 ) == 0 ) {
                          $items:body

                          done = 1;
                          atomicExch( addr, 0 );
                      }
                      __threadfence();
                  } while (done == 0);
                |]
        ]

