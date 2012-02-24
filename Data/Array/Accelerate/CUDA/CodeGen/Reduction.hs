{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Reduction
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Reduction (

  -- skeletons
  mkFold, mkFoldAll, mkFoldSeg,

  -- closets
  reduceWarp, reduceBlock

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
  env                                   <- environment
  (argIn0, x0, decl0, getIn0, _)        <- getters 0 elt
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
        const int gridSize = blockDim.x * gridDim.x;
              int i        = blockIdx.x * blockDim.x + threadIdx.x;
        $decls:smem
        $decls:decl0
        $decls:decl1

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
                $decls:env
                $stms:(x1 .=. combine)
            }
        }

        /*
         * Each thread puts its local sum into shared memory, then threads
         * cooperatively reduce the shared array to a single value.
         */
        $stms:(sdata "threadIdx.x" .=. x1)
        __syncthreads();

        i = min(((int) len) - blockIdx.x * blockDim.x, blockDim.x);
        $stms:(reduceBlock dev elt "i" sdata env combine)

        /*
         * Write the results of this block back to global memory. If we are the last
         * phase of a recursive multi-block reduction, include the seed element.
         */
        if (threadIdx.x == 0)
        {
            $stms:(maybe inclusive_fold (exclusive_fold x0 env) mseed)
        }
    }
  |]
  where
    name                        = maybe "fold1All" (const "foldAll") mseed
    (argOut, _, setOut)         = setters elt
    (x1,   decl1)               = locals "x1" elt
    (smem, sdata)               = shared 0 Nothing [cexp| blockDim.x |] elt
    --
    inclusive_fold              = setOut "blockIdx.x" x1
    exclusive_fold x0 env seed  = [[cstm|
      if (len > 0) {
          if (gridDim.x == 1) {
              $stms:(x0 .=. seed)
              $decls:env
              $stms:(x1 .=. combine)
          }
          $stms:(setOut "blockIdx.x" x1)
      }
      else {
          $stms:(setOut "blockIdx.x" seed)
      }
    |]]


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
mkFold :: DeviceProperties -> [Type] -> [Exp] -> Maybe [Exp] -> CGM CUTranslSkel
mkFold dev elt combine mseed = do
  env                                   <- environment
  (argIn0, x0, decl0, getIn0, getTmp)   <- getters 0 elt
  return $ CUTranslSkel name [cunit|
    extern "C"
    __global__ void
    $id:name
    (
        $params:argOut,
        $params:argIn0,
        const typename Ix interval_size,        // indexHead(shIn0)
        const typename Ix num_intervals,        // size(shOut)
        const typename Ix num_elements          // size(shIn0)
    )
    {
        $decls:smem
        $decls:decl1
        $decls:decl0
        $decls:env

        if (threadIdx.x >= interval_size)
           return;

        /*
         * Threads in a block cooperatively reduce all elements in an interval.
         */
        for (int seg = blockIdx.x; seg < num_intervals; seg += gridDim.x)
        {
            const int start = seg * interval_size;
            const int end   = min(start + interval_size, num_elements);

            /*
             * Ensure aligned access to global memory, and that each thread
             * initialises its local sum
             */
            int i = start - (start & (warpSize - 1));

            if (i == start || interval_size > blockDim.x)
            {
                i += threadIdx.x;

                if (i >= start)
                {
                    $stms:(x1 .=. getIn0 "i")
                }

                if (i + blockDim.x < end)
                {
                    $decls:(getTmp "i + blockDim.x")

                    if (i >= start) {
                        $decls:env
                        $stms:(x1 .=. combine)
                    }
                    else {
                        $stms:(x1 .=. x0)
                    }
                }

                /*
                 * Now, iterate collecting a local sum
                 */
                for (i += 2 * blockDim.x; i < end; i += blockDim.x)
                {
                    $stms:(x0 .=. getIn0 "i")
                    $decls:env
                    $stms:(x1 .=. combine)
                }
            }
            else
            {
                $stms:(x1 .=. getIn0 "start + threadIdx.x")
            }

            /*
             * Each thread puts its local sum into shared memory, and
             * cooperatively reduces this to a single value.
             */
            $stms:(sdata "threadIdx.x" .=. x1)
            __syncthreads();

            const int n = min(interval_size, blockDim.x);
            $stms:(reduceBlock dev elt "n" sdata env combine)

            /*
             * Finally, the first thread writes the result for this segment
             */
            if (threadIdx.x == 0)
            {
                $stms:(maybe inclusive_fold (exclusive_fold x0 env) mseed)
            }
        }
    }
  |]
  where
    name                        = maybe "fold1" (const "fold") mseed
    (argOut, _, setOut)         = setters elt
    (x1,   decl1)               = locals "x1" elt
    (smem, sdata)               = shared 0 Nothing [cexp| blockDim.x |] elt
    --
    inclusive_fold              = setOut "seg" x1
    exclusive_fold x0 env seed  = [cstm|
      if (interval_size > 0) {
          $stms:(x0 .=. seed)
          $decls:env
          $stms:(x1 .=. combine)
      } else {
          $stms:(x1 .=. seed)
      }|] :
      setOut "seg" x1



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
-- reduction of the i-th section, in parallel. Care is taken to ensure that data
-- array access is aligned to a warp boundary.
--
-- Since an entire 32-thread warp is assigned for each segment, many threads
-- will remain idle when the segments are very small. This code relies on
-- implicit synchronisation among threads in a warp.
--
-- The offset array contains the starting index for each segment in the input
-- array. The i-th warp reduces values in the input array at indices
-- [d_offset[i], d_offset[i+1]).
--
mkFoldSeg :: DeviceProperties -> Int -> Type -> [Type] -> [Exp] -> Maybe [Exp] -> CGM CUTranslSkel
mkFoldSeg dev dim seg elt combine mseed = do
  env                                   <- environment
  (argIn0, x0, decl0, getIn0, getTmp)   <- getters 0 elt
  return $ CUTranslSkel name [cunit|
    $edecl:(cdim "DimOut" dim)
    $edecl:(cdim "DimIn0" dim)

    extern "C"
    __global__ void
    foldSeg
    (
        $params:argOut,
        $params:argIn0,
        const $ty:(cptr seg)    d_offset,
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

        extern volatile __shared__ int s_ptrs[][2];

        $decls:smem
        $decls:decl1
        $decls:decl0

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
                        $decls:env
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
                    $decls:env
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
            $stms:(reduceWarp dev elt "n" "thread_lane" sdata env combine)

            /*
             * Finally, the first thread writes the result for this segment
             */
            if (thread_lane == 0)
            {
                $stms:(maybe inclusive_fold (exclusive_fold x0 env) mseed)
            }
        }
    }
  |]
  where
    name                        = maybe "fold1Seg" (const "foldSeg") mseed
    (argOut, _, setOut)         = setters elt
    (x1,   decl1)               = locals "x1" elt
    (smem, sdata)               = shared 0 (Just $ [cexp| &s_ptrs[vectors_per_block][2] |]) [cexp| blockDim.x |] elt
    --
    inclusive_fold              = setOut "seg" x1
    exclusive_fold x0 env seed  = [cstm|
      if (num_elements > 0) {
          $stms:(x0 .=. seed)
          $decls:env
          $stms:(x1 .=. combine)
      } else {
          $stms:(x1 .=. seed)
      }|] :
      setOut "seg" x1


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- Threads of a warp run in lockstep, so there is no need to synchronise. We
-- hijack the standard local variable sets (x0 and x1) for the combination
-- function. The initial values must already be stored in shared memory. The
-- final result is stored in x1.
--
reduceWarp :: DeviceProperties
           -> [Type]
           -> String                    -- number of elements
           -> String                    -- thread identifier: usually the lane or thread id
           -> (String -> [Exp])         -- index shared memory
           -> Environment               -- local binding environment for the..
           -> [Exp]                     -- ..binary associative combination function
           -> [Stm]
reduceWarp dev elt n tid sdata env combine = map (reduce . pow2) [v,v-1..0]
  where
    v           = floor (logBase 2 (fromIntegral $ warpSize dev :: Double)) :: Int
    pow2 x      = (2::Int) ^ x
    (x0, _)     = locals "x0" elt
    (x1, _)     = locals "x1" elt
    --
    reduce 1    = [cstm|
      if ( $id:n > 1 && $id:tid + 1 < $id:n ) {
          $stms:(x0 .=. sdata "threadIdx.x + 1")
          $decls:env
          $stms:(x1 .=. combine)
      }
    |]
    reduce i    = [cstm|
      if ( $id:n > $int:i && $id:tid + $int:i < $id:n ) {
          $stms:(x0 .=. sdata ("threadIdx.x + " ++ show i))
          $decls:env
          $stms:(x1 .=. combine)
          $stms:(sdata "threadIdx.x" .=. x1)
      }
    |]


-- All threads cooperatively reduce this block's data in shared memory. We
-- hijack the standard local variables (x0 and x1) for the combination function.
-- The initial values must already be stored in shared memory.
--
reduceBlock :: DeviceProperties
            -> [Type]
            -> String                   -- number of elements
            -> (String -> [Exp])        -- index shared memory
            -> Environment              -- local binding environment for the..
            -> [Exp]                    -- ..binary associative function
            -> [Stm]
reduceBlock dev elt n sdata env combine = map (reduce . pow2) [u,u-1..v]
  where
    u           = floor (logBase 2 (fromIntegral $ maxThreadsPerBlock dev :: Double)) :: Int
    v           = floor (logBase 2 (fromIntegral $ warpSize dev           :: Double)) :: Int
    pow2 x      = (2::Int) ^ x
    (x0, _)     = locals "x0" elt
    (x1, _)     = locals "x1" elt
    --
    reduce 32   = [cstm|
      if ( threadIdx.x < $int:(warpSize dev) ) {
          $stms:(reduceWarp dev elt n "threadIdx.x" sdata env combine)
      }
    |]
    reduce i    = [cstm|
      if ( $id:n > $int:i ) {
          if ( threadIdx.x < $int:i && threadIdx.x + $int:i < $id:n ) {
              $stms:(x0 .=. sdata ("threadIdx.x + " ++ show i))
              $decls:env
              $stms:(x1 .=. combine)
              $stms:(sdata "threadIdx.x" .=. x1)
          }
          __syncthreads();
      }
    |]


