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

  mkFoldAll, mkFold

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
        $decls:decl0
        $decls:decl1
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
            $stms:(x1 .=. getIn0 "i")

            for (i += gridSize; i < len; i += gridSize)
            {
                $stms:(x0 .=. getIn0 "i")
                $stms:(x1 .=. combine)
            }
        }

        /*
         * Each thread puts its local sum into shared memory, then threads
         * cooperatively reduce the shared array to a single value.
         */
        $stms:(sdata "tid" .=. x1)
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
    (argIn0, _, _)      = getters 0 elt
    (svar, smem)        = shared 0 [cexp| blockDim.x |] elt
    (x0, decl0)         = locals "x0" elt
    (x1, decl1)         = locals "x1" elt
    --
    sdata ix            = map (\v -> [cexp| $exp:v [ $id:ix ] |]) svar
    getIn0 ix           = let k = length elt in map (\s -> [cexp| $id:("d_in0_a"++s) [ $id:ix ] |]) (map show [k-1,k-2..0])
    --
    inclusive_fold      = setOut "blockIdx.x" x1
    exclusive_fold seed = [[cstm|
      if (len > 0) {
          if (gridDim.x == 1) {
              $stms:(x0 .=. seed)
              $stms:(x1 .=. combine)
          }
          $stms:(setOut "blockIdx.x" x1)
      }
      else {
          $stms:(setOut "blockIdx.x" seed)
      }
    |]]
    --
    -- All threads cooperatively reduce this block's data in shared memory
    --
    reduceBlock = map (reduce . ((2::Int)^)) [u,u-1..v+1]
      where
        u               = floor (logBase 2 (fromIntegral $ maxThreadsPerBlock dev :: Double)) :: Int
        v               = floor (logBase 2 (fromIntegral $ warpSize dev           :: Double)) :: Int
        reduce n        = [cstm|
          if ( i > $int:n ) {
              if ( tid < $int:n && tid + $int:n < i ) {
                  $stms:(x0 .=. sdata ("tid + " ++ show n))
                  $stms:(x1 .=. combine)
                  $stms:(sdata "tid" .=. x1)
              }
              __syncthreads();
          }
        |]
    --
    -- Threads of a warp run in lockstep (SIMD) so once we reduce to a single
    -- warp's worth of data we no longer need to __syncthreads().
    --
    reduceWarp = [cstm|
      if ( tid < $int:(warpSize dev) ) {
          $stms:(map (reduce . ((2::Int)^)) [v,v-1..0])
      }
    |]
      where
        v               = floor (logBase 2 (fromIntegral $ warpSize dev :: Double)) :: Int
        reduce 1        = [cstm|
          if ( i > 1 ) {
              if ( tid + 1 < i ) {
                  $stms:(x0 .=. sdata "tid + 1")
                  $stms:(x1 .=. combine)
              }
          }
        |]
        reduce n        = [cstm|
          if ( i > $int:n ) {
              if ( tid + $int:n < i ) {
                  $stms:(x0 .=. sdata ("tid + " ++ show n))
                  $stms:(x1 .=. combine)
                  $stms:(sdata "tid" .=. x1)
              }
          }
        |]



-- Reduction of the innermost dimension of an array of arbitrary rank. The first
-- argument needs to be an associative function to enable an efficient parallel
-- implementation
--
-- fold :: (Shape ix, Elt a)
--      => (Exp a -> Exp a -> Exp a)
--      -> Exp a
--      -> Acc (Array (ix :. Int) a)
--      -> Acc (Array ix a)
--
-- fold1 :: (Shape ix, Elt a)
--       => (Exp a -> Exp a -> Exp a)
--       -> Acc (Array (ix :. Int) a)
--       -> Acc (Array ix a)
--
mkFold :: DeviceProperties -> Int -> [Type] -> [Exp] -> Maybe [Exp] -> CGM CUTranslSkel
mkFold dev dim elt combine mseed = do
  env   <- environment
  return $ CUTranslSkel name [cunit|
    $edecl:(cdim "DimOut" dim)
    $edecl:(cdim "DimIn0" (dim+1))

    extern "C"
    __global__ void
    $id:name
    (
        $params:argOut,
        $params:argIn0,
        const typename DimOut shOut,
        const typename DimIn0 shIn0
    )
    {
        const int num_elements = indexHead(shIn0);
        const int num_segments = size(shOut);

        const int num_vectors  = blockDim.x / warpSize * gridDim.x;
        const int thread_id    = blockDim.x * blockIdx.x + threadIdx.x;
        const int vector_id    = thread_id / warpSize;
        const int thread_lane  = threadIdx.x & (warpSize - 1);
        $decls:decl_smem
        $decls:decl_x1
        $decls:decl_x0
        $decls:env

        /*
         * Each warp reduces elements along a projection through an innermost
         * dimension to a single value
         */
        for (int seg = vector_id; seg < num_segments; seg += num_vectors)
        {
            const int start = seg   * num_elements;
            const int end   = start + num_elements;

            if (num_elements > warpSize)
            {
                /*
                 * Ensure aligned access to global memory, and that each thread
                 * initialises its local sum.
                 */
                int i = start - (start & (warpSize - 1)) + thread_lane;
                if (i >= start)
                {
                    $stms:(x1 .=. getIn0 "i")
                }

                if (i + warpSize < end)
                {
                    $decls:(getTmp "i + warpSize")

                    if (i >= start) {
                        $stms:(x1 .=. combine)
                    }
                    else {
                        $stms:(x1 .=. x0)
                    }
                }

                /*
                 * Now, iterate along the inner-most dimension collecting a local sum
                 */
                for (i += 2 * warpSize; i < end; i += warpSize)
                {
                    $stms:(x0 .=. getIn0 "i")
                    $stms:(x1 .=. combine)
                }
            }
            else if (start + thread_lane < end)
            {
                $stms:(x1 .=. getIn0 "start + thread_lane")
            }

            /*
             * Each thread puts its local sum into shared memory, then cooperatively
             * reduce the shared array to a single value.
             */
            const int n = min(num_elements, warpSize);
            $stms:(sdata "threadIdx.x" .=. x1)
            $stms:reduceWarp

            /*
             * Finally, the first thread writes the result for this segment
             */
            if (thread_lane == 0)
            {
                $stms:(maybe inclusive_fold exclusive_fold mseed)
            }
        }
    }
  |]
  where
    name        = maybe "fold1" (const "fold") mseed
    (argOut, _, setOut) = setters elt
    (argIn0, _, getTmp) = getters 0 elt
    (svar, decl_smem)   = shared 0 [cexp| blockDim.x |] elt
    (x0,  decl_x0)      = locals "x0" elt
    (x1,  decl_x1)      = locals "x1" elt
    --
    getIn0 ix           = let k = length elt in map (\s -> [cexp| $id:("d_in0_a"++s) [ $id:ix ] |]) (map show [k-1,k-2..0])
    sdata ix            = map (\v -> [cexp| $exp:v [ $id:ix ] |]) svar
    --
    inclusive_fold      = setOut "seg" x1
    exclusive_fold seed = [cstm|
      if (num_elements > 0) {
          $stms:(x0 .=. seed)
          $stms:(x1 .=. combine)
      } else {
          $stms:(x1 .=. seed)
      }|] :
      setOut "seg" x1
    --
    reduceWarp          = map (reduce . ((2::Int)^)) [v,v-1..0]
      where
        v               = floor (logBase 2 (fromIntegral $ warpSize dev :: Double)) :: Int
        tid             = "threadIdx.x"
        reduce 1        = [cstm|
          if (n > 1 && thread_lane + 1 < n) {
              $stms:(x0 .=. sdata "threadIdx.x + 1")
              $stms:(x1 .=. combine)
          }
        |]
        reduce i        = [cstm|
          if (n > $int:i && thread_lane + $int:i < n) {
              $stms:(x0 .=. sdata (tid ++ "+" ++ show i))
              $stms:(x1 .=. combine)
              $stms:(sdata tid .=. x1)
          }
        |]



