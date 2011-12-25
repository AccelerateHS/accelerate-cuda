{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Reduction
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Reduction (

  mkFoldAll

) where

import Data.Loc
import Data.Symbol
import Language.C.Syntax
import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis

import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Monad


-- Reduction of an array of arbitrary rank to a single scalar value. The first
-- argument needs to be an associative function to enable an efficient parallel
-- implementation
--
-- foldAll :: (Shape sh, Elt a)
--         => (Exp a -> Exp a -> Exp a)
--         -> Exp a
--         -> Acc (Array sh a)
--         -> Acc (Scalar a)
--
-- fold1All :: (Shape sh, Elt a)
--          => (Exp a -> Exp a -> Exp a)
--          -> Acc (Array sh a)
--          -> Acc (Scalar a)
--
-- Each thread computes multiple elements sequentially. This reduces the overall
-- cost of the algorithm while keeping the work complexity O(n) and the step
-- complexity O(log n). c.f. Brent's Theorem optimisation.
--
mkFoldAll :: DeviceProperties -> [Type] -> [Exp] -> Maybe [Exp] -> CGM CUTranslSkel
mkFoldAll dev elt combine mseed = do
  env   <- environment
  return $ CUTranslSkel name [cunit|
    extern "C"
    __global__ void
    $id:name
    (
        $params:argOut,
        $params:argIn0,
        const typename Ix len
    )
    {
        const int tid      = threadIdx.x;
        const int gridSize = blockDim.x * gridDim.x;
              int i        = blockIdx.x * blockDim.x + tid;
        $decls:smem
        $decls:locals
        $decls:env

        /*
         * Reduce multiple elements per thread. The number is determined by the
         * number of active thread blocks (via gridDim). More blocks will result in
         * a larger `gridSize', and hence fewer elements per thread
         *
         * The loop stride of `gridSize' is used to maintain coalescing.
         */
        if (i < len)
        {
            $stms:(x1 .=. getIn0 i)

            for (i += gridSize; i < len; i += gridSize)
            {
                $stms:(x0 .=. getIn0 i)
                $stms:(x1 .=. combine)
            }
        }

        /*
         * Each thread puts its local sum into shared memory, then threads
         * cooperatively reduce the shared array to a single value.
         */
        $stms:(sdata tid .=. x1)
        __syncthreads();

        i = min((int) len, blockDim.x);
        $stms:reduceBlock
        $stm:reduceWarp

        /*
         * Write the results of this block back to global memory. If we are the last
         * phase of a recursive multi-block reduction, include the seed element.
         */
        if (tid == 0)
        {
            $stms:(maybe inclusive_fold exclusive_fold mseed)
        }
    }
  |]
  where
    name                = maybe "fold1All" (const "foldAll") mseed
    (argOut, _, setOut) = setters elt
    (argIn0, x0, _)     = getters 0 elt
    (_,      x1, _)     = getters 1 elt
    (svar, smem)        = shared 0 [cexp| blockDim.x |] elt
    locals              = zipWith3 (\t v1 v0 -> [cdecl| $ty:t $id:(show v1), $id:(show v0) ; |]) elt x0 x1
    --
    tid                 = [cexp| $id:("tid") |]
    i                   = [cexp| $id:("i") |]
    sdata ix            = map (\v -> [cexp| $exp:v [ $exp:ix ] |]) svar
    getIn0 ix           = let k = length elt
                          in  map (\s -> [cexp| $id:("d_in0_a"++s) [ $exp:ix ] |]) (map show [k-1,k-2..0])
    --
    inclusive_fold      = setOut "blockIdx.x" x1
    exclusive_fold seed = [[cstm|
      if (len > 0)
      {
          if (gridDim.x == 1)
          {
              $stms:(x0 .=. seed)
              $stms:(x1 .=. combine)
          }
          $stms:(setOut "blockIdx.x" x1)
      }
      else
      {
          $stms:(setOut "blockIdx.x" seed)
      }
    |]]
    --
    -- All threads cooperatively reduce this block's data in shared memory
    --
    reduceBlock = map (reduce . (2^)) [u,u-1..v+1]
      where
        u               = floor (logBase 2 (fromIntegral $ maxThreadsPerBlock dev :: Double)) :: Int
        v               = floor (logBase 2 (fromIntegral $ warpSize dev           :: Double)) :: Int
        reduce n        =
          let m = [cexp| $exp:tid + $int:(n :: Int) |]
          in  [cstm|
                if ( i > $int:n ) {
                    if ( tid < $int:n && $exp:m < i ) {
                        $stms:(x0 .=. sdata m )
                        $stms:(x1 .=. combine)
                        $stms:(sdata tid .=. x1)
                    }
                    __syncthreads();
                }
              |]
    --
    -- Threads of a warp run in lockstep (SIMD) so once we reduce to a single
    -- warp's worth of data we no longer need to __syncthreads().
    --
    reduceWarp =
      [cstm|
          if ( tid < $int:(warpSize dev) ) {
              $stms:(map (reduce . (2^)) [v,v-1..1])
              if ( i > 1) {
                  if ( $exp:tid1 < i ) {
                      $stms:(x0 .=. sdata tid1 )
                      $stms:(x1 .=. combine)
                  }
              }
          }
      |]
      where
        v               = floor (logBase 2 (fromIntegral $ warpSize dev :: Double)) :: Int
        tid1            = [cexp| tid + 1 |]
        reduce n        =
          let m = [cexp| $exp:tid + $int:(n :: Int) |]
          in  [cstm|
                if ( i > $int:n ) {
                    if ( $exp:m < i ) {
                        $stms:(x0 .=. sdata m)
                        $stms:(x1 .=. combine)
                        $stms:(sdata tid .=. x1)
                    }
                }
              |]


