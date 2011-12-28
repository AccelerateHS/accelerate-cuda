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

  mkFold, mkFoldAll, mkFoldSeg

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
    name                        = maybe "fold1All" (const "foldAll") mseed
    (argOut, _,         setOut) = setters elt
    (argIn0, x0, decl0, _)      = getters 0 elt
    (x1,   decl1)               = locals "x1" elt
    (smem, sdata)               = shared 0 [cexp| blockDim.x |] elt
    --
    getIn0 ix =
      let k = length elt
      in  map (\s -> [cexp| $id:("d_in0_a"++s) [ $id:ix ] |]) (map show [k-1,k-2..0])
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
        $decls:smem
        $decls:decl1
        $decls:decl0
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
    name                        = maybe "fold1" (const "fold") mseed
    (argOut, _,         setOut) = setters elt
    (argIn0, x0, decl0, getTmp) = getters 0 elt
    (x1,   decl1)               = locals "x1" elt
    (smem, sdata)               = shared 0 [cexp| blockDim.x |] elt
    --
    getIn0 ix =
      let k = length elt
      in  map (\s -> [cexp| $id:("d_in0_a"++s) [ $id:ix ] |]) (map show [k-1,k-2..0])
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



-- Segmented reduction along the innermost dimension of an array. Performs one
-- individual reduction per segment of the source array. These reductions
-- proceed in parallel.
--
-- foldSeg :: (Shape ix, Elt a)
--         => (Exp a -> Exp a -> Exp a)
--         -> Exp a
--         -> Acc (Array (ix :. Int) a)
--         -> Acc Segments
--         -> Acc (Array (ix :. Int) a)
--
-- fold1Seg :: (Shape ix, Elt a)
--          => (Exp a -> Exp a -> Exp a)
--          -> Acc (Array (ix :. Int) a)
--          -> Acc Segments
--          -> Acc (Array (ix :. Int) a)
--
-- Each segment of the vector is assigned to a warp, which computes the
-- reduction of the i-th section, in parallel.
--
-- This division of work implies that the data arrays are accessed in a
-- contiguous manner (if not necessarily aligned). For devices of compute
-- capability 1.2 and later, these accesses will be coalesced. A single
-- transaction will be issued if all of the addresses for a half-warp happen to
-- fall within a single 128-byte boundary. Extra transactions will be made to
-- cover any spill. The same applies for 2.x devices, except that all widths are
-- doubled since transactions occur on a per-warp basis.
--
-- Since an entire 32-thread warp is assigned for each segment, many threads
-- will remain idle when the segments are very small. This code relies on
-- implicit synchronisation among threads in a warp.
--
-- The offset array contains the starting index for each segment in the input
-- array. The i-th warp reduces values in the input array at indices
-- [d_offset[i], d_offset[i+1]).
--
mkFoldSeg :: DeviceProperties -> Int -> [Type] -> [Exp] -> Maybe [Exp] -> CGM CUTranslSkel
mkFoldSeg dev dim elt combine mseed = do
  env   <- environment
  return $ CUTranslSkel name [cunit|
    $edecl:(cdim "DimOut" dim)
    $edecl:(cdim "DimIn0" dim)

    extern "C"
    __global__ void
    foldSeg
    (
        $params:argOut,
        $params:argIn0,
        const typename Ix*      d_offset,
        const typename DimOut   shOut,
        const typename DimIn0   shIn0
    )
    {
        const int vectors_per_block = blockDim.x / warpSize;
        const int num_vectors       = vectors_per_block * gridDim.x;
        const int thread_id         = blockDim.x * blockIdx.x + threadIdx.x;
        const int vector_id         = thread_id / warpSize;
        const int thread_lane       = threadIdx.x & (warpSize - 1);
        const int vector_lane       = threadIdx.x / warpSize;

        const int num_segments      = indexHead(shOut);
        const int total_segments    = size(shOut);

        $decls:smem
        $decls:decl1
        $decls:decl0
        $decls:env

        volatile int s_ptrs[][2] = (typename Ix**) &s0_a0[blockDim.x];

        for (int seg = vector_id; seg < total_segments; seg += num_vectors)
        {
            const int s    =  seg % num_segments;
            const int base = (seg / num_segments) * indexHead(shIn0);

            /*
             * Use two threads to fetch the indices of the start and end of this
             * segment. This results in single coalesced global read, instead of two
             * separate transactions.
             */
            if (thread_lane < 2)
                s_ptrs[vector_lane][thread_lane] = (int) d_offset[s + thread_lane];

            const int   start        = base + s_ptrs[vector_lane][0];
            const int   end          = base + s_ptrs[vector_lane][1];
            const int   num_elements = end  - start;

            /*
             * Each thread reads in values of this segment, accumulating a local sum
             */
            if (num_elements > warpSize)
            {
                /*
                 * Ensure aligned access to global memory
                 */
                int i = start - (start & (warpSize - 1)) + thread_lane;
                if (i >= start)
                {
                    $stms:(x1 .=. getIn0 "i")
                }

                /*
                 * Subsequent reads to global memory are aligned, but make sure all
                 * threads have initialised their local sum.
                 */
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
             * Store local sums into shared memory and reduce to a single value
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
    name                        = maybe "fold1Seg" (const "foldSeg") mseed
    (argOut, _,         setOut) = setters elt
    (argIn0, x0, decl0, getTmp) = getters 0 elt
    (x1,   decl1)               = locals "x1" elt
    (smem, sdata)               = shared 0 [cexp| blockDim.x |] elt
    --
    getIn0 ix =
      let k = length elt
      in  map (\s -> [cexp| $id:("d_in0_a"++s) [ $id:ix ] |]) (map show [k-1,k-2..0])
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

