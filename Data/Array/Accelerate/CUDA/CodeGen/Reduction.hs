{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}
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

  mkFold, mkFold1, mkFoldSeg, mkFold1Seg,

) where

import Foreign.CUDA.Analysis
import Language.C.Quote.CUDA
import qualified Language.C.Syntax                      as C

import Data.Array.Accelerate.Type                       ( IsIntegral )
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, Z(..), (:.)(..) )
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type

#include "accelerate.h"


-- Reduce an array along the innermost dimension. The function must be
-- associative to enable efficient parallel implementation.
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
-- If this is collapsing an array to a single value, we use a multi-pass
-- algorithm that splits the input data over several thread blocks. The first
-- kernel is executed once, and then the second recursively until a single value
-- is produced.
--
mkFold :: forall aenv sh e. (Shape sh, Elt e)
       => DeviceProperties
       -> Gamma aenv
       -> CUFun2 aenv (e -> e -> e)
       -> CUExp aenv e
       -> CUDelayedAcc aenv (sh :. Int) e
       -> [CUTranslSkel aenv (Array sh e)]
mkFold dev aenv f z a
  | expDim (undefined :: Exp aenv sh) > 0 = mkFoldDim dev aenv f (Just z) a
  | otherwise                             = mkFoldAll dev aenv f (Just z) a

mkFold1 :: forall aenv sh e. (Shape sh, Elt e)
        => DeviceProperties
        -> Gamma aenv
        -> CUFun2 aenv (e -> e -> e)
        -> CUDelayedAcc aenv (sh :. Int) e
        -> [ CUTranslSkel aenv (Array sh e) ]
mkFold1 dev aenv f a
  | expDim (undefined :: Exp aenv sh) > 0 = mkFoldDim dev aenv f Nothing a
  | otherwise                             = mkFoldAll dev aenv f Nothing a


-- Reduction of an array of arbitrary rank to a single scalar value. Each thread
-- computes multiple elements sequentially. This reduces the overall cost of the
-- algorithm while keeping the work complexity O(n) and the step complexity
-- O(log n). c.f. Brent's Theorem optimisation.
--
-- Since the reduction occurs over multiple blocks, there are two phases. The
-- first pass incorporates any fused/embedded input arrays, while the second
-- recurses over a manifest array to produce a single value.
--
mkFoldAll
    :: forall aenv sh e. (Shape sh, Elt e)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> Maybe (CUExp aenv e)
    -> CUDelayedAcc aenv (sh :. Int) e
    -> [ CUTranslSkel aenv (Array sh e) ]
mkFoldAll dev aenv f z a
  = let (_, rec) = readArray "Rec" (undefined :: Array (sh:.Int) e)
    in
    [ mkFoldAll' False dev aenv f z a
    , mkFoldAll' True  dev aenv f z rec ]


mkFoldAll'
    :: forall aenv sh e. (Shape sh, Elt e)
    => Bool
    -> DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> Maybe (CUExp aenv e)
    -> CUDelayedAcc aenv (sh :. Int) e
    -> CUTranslSkel aenv (Array sh e)
mkFoldAll' recursive dev aenv fun@(CUFun2 _ _ combine) mseed (CUDelayed (CUExp shIn) _ (CUFun1 _ get))
  = CUTranslSkel foldAll [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    $id:foldAll
    (
        $params:argIn,
        $params:argOut,
        $params:argRec
    )
    {
        $decls:smem
        $decls:declx
        $decls:decly

        $items:(sh .=. shIn)

        const int shapeSize     = $exp:(csize sh);
        const int gridSize      = $exp:(gridSize dev);
              int ix            = $exp:(threadIdx dev);

        /*
         * Reduce multiple elements per thread. The number is determined by the
         * number of active thread blocks (via gridDim). More blocks will result in
         * a larger `gridSize', and hence fewer elements per thread
         *
         * The loop stride of `gridSize' is used to maintain coalescing.
         *
         * Note that we can't simply kill threads that won't participate in the
         * reduction, as exclusive reductions of empty arrays then won't be
         * initialised with their seed element.
         */
        if ( ix < shapeSize )
        {
            /*
             * Initialise the local sum, then ...
             */
            $items:(y .=. get ix)

            /*
             * ... continue striding the array, reading new values into 'x' and
             * combining into the local accumulator 'y'. The non-idiomatic
             * structure of the loop below is because we have already
             * initialised 'y' above.
             */
            for ( ix += gridSize; ix < shapeSize; ix += gridSize )
            {
                $items:(x .=. get ix)
                $items:(y .=. combine x y)
            }
        }

        /*
         * Each thread puts its local sum into shared memory, then threads
         * cooperatively reduce the shared array to a single value.
         */
        $items:(sdata "threadIdx.x" .=. y)

        ix = min(shapeSize - blockIdx.x * blockDim.x, blockDim.x);
        $items:(reduceBlock dev fun x y sdata (cvar "ix"))

        /*
         * Write the results of this block back to global memory. If we are the last
         * phase of a recursive multi-block reduction, include the seed element.
         */
        if ( threadIdx.x == 0 )
        {
            $items:(maybe inclusive_finish exclusive_finish mseed)
        }
    }
  |]
  where
    foldAll                     = maybe "fold1All" (const "foldAll") mseed
    (texIn, argIn)              = environment dev aenv
    (argOut, _, setOut)         = writeArray "Out" (undefined :: Array (sh :. Int) e)
    (argRec, _)
      | recursive               = readArray "Rec" (undefined :: Array (sh :. Int) e)
      | otherwise               = ([], undefined)

    (_, x, declx)               = locals "x" (undefined :: e)
    (_, y, decly)               = locals "y" (undefined :: e)
    (sh, _, _)                  = locals "sh" (undefined :: sh :. Int)
    ix                          = [cvar "ix"]
    (smem, sdata)               = shared (undefined :: e) "sdata" [cexp| blockDim.x |] Nothing
    --
    inclusive_finish                    = setOut "blockIdx.x" .=. y
    exclusive_finish (CUExp seed)       = [[citem|
      if ( shapeSize > 0 ) {
          if ( gridDim.x == 1 ) {
              $items:(x .=. seed)
              $items:(y .=. combine x y)
          }
          $items:(setOut "blockIdx.x" .=. y)
      }
      else {
          $items:(setOut "blockIdx.x" .=. seed)
      }
    |]]


-- Reduction of the innermost dimension of an array of arbitrary rank. Each
-- thread block reduces along one innermost dimension index.
--
mkFoldDim
    :: forall aenv sh e. (Shape sh, Elt e)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> Maybe (CUExp aenv e)
    -> CUDelayedAcc aenv (sh :. Int) e
    -> [ CUTranslSkel aenv (Array sh e) ]
mkFoldDim dev aenv fun@(CUFun2 _ _ combine) mseed (CUDelayed (CUExp shIn) _ (CUFun1 _ get))
  = return
  $ CUTranslSkel fold [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    $id:fold
    (
        $params:argIn,
        $params:argOut
    )
    {
        $decls:smem
        $decls:declx
        $decls:decly

        $items:(sh .=. shIn)

        const int numIntervals  = $exp:(csize shOut);
        const int intervalSize  = $exp:(cindexHead sh);
              int ix;
              int seg;

        /*
         * If the intervals of an exclusive fold are empty, use all threads to
         * map the seed value to the output array and exit.
         */
        $items:(maybe [] mapseed mseed)

        /*
         * Threads in a block cooperatively reduce all elements in an interval.
         */
        for ( seg = blockIdx.x
            ; seg < numIntervals
            ; seg += gridDim.x )
        {
            const int start = seg * intervalSize;
            const int end   = start + intervalSize;
            const int n     = min(end - start, blockDim.x);

            /*
             * Kill threads that will not participate to avoid invalid reads.
             * Take advantage of the fact that the array is rectangular.
             */
            if ( threadIdx.x >= n )
               return;

            /*
             * Ensure aligned access to global memory, and that each thread
             * initialises its local sum
             */
            ix = start - (start & (warpSize - 1));

            if ( ix == start || intervalSize > blockDim.x)
            {
                ix += threadIdx.x;

                if ( ix >= start )
                {
                    $items:(y .=. get ix)
                }

                if ( ix + blockDim.x < end )
                {
                    $items:(x .=. get [cvar "ix + blockDim.x"])

                    if ( ix >= start ) {
                        $items:(y .=. combine x y)
                    }
                    else {
                        $items:(y .=. x)
                    }
                }

                /*
                 * Now, iterate collecting a local sum
                 */
                for ( ix += 2 * blockDim.x; ix < end; ix += blockDim.x )
                {
                    $items:(x .=. get ix)
                    $items:(y .=. combine x y)
                }
            }
            else
            {
                $items:(y .=. get [cvar "start + threadIdx.x"])
            }

            /*
             * Each thread puts its local sum into shared memory, and
             * cooperatively reduces this to a single value.
             */
            $items:(sdata "threadIdx.x" .=. y)
            $items:(reduceBlock dev fun x y sdata (cvar "n"))

            /*
             * Finally, the first thread writes the result for this segment. For
             * exclusive reductions, we also combine with the seed element here.
             */
            if ( threadIdx.x == 0 ) {
                $items:(maybe [] exclusive_finish mseed)
                $items:(setOut "seg" .=. y)
            }
        }
    }
  |]
  where
    fold                        = maybe "fold1" (const "fold") mseed
    (texIn, argIn)              = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh e)
    (_, x, declx)               = locals "x" (undefined :: e)
    (_, y, decly)               = locals "y" (undefined :: e)
    (sh, _, _)                  = locals "sh" (undefined :: sh :. Int)
    ix                          = [cvar "ix"]
    (smem, sdata)               = shared (undefined :: e) "sdata" [cexp| blockDim.x |] Nothing
    --
    mapseed (CUExp seed)
      = [citem|  if ( intervalSize == 0 ) {
                     const int gridSize  = $exp:(gridSize dev);

                     for ( ix = $exp:(threadIdx dev)
                         ; ix < numIntervals
                         ; ix += gridSize )
                     {
                         $items:(setOut "ix" .=. seed)
                     }
                 } |] :[]
    --
    exclusive_finish (CUExp seed)
      = concat [ x .=. seed
               , y .=. combine x y ]


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
mkFoldSeg
    :: (Shape sh, Elt e, Elt i, IsIntegral i)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> CUExp aenv e
    -> CUDelayedAcc aenv (sh :. Int) e
    -> CUDelayedAcc aenv (Z  :. Int) i
    -> [CUTranslSkel aenv (Array (sh :. Int) e)]
mkFoldSeg dev aenv f z a s = [ mkFoldSeg' dev aenv f (Just z) a s ]

mkFold1Seg
    :: (Shape sh, Elt e, Elt i, IsIntegral i)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> CUDelayedAcc aenv (sh :. Int) e
    -> CUDelayedAcc aenv (Z  :. Int) i
    -> [CUTranslSkel aenv (Array (sh :. Int) e)]
mkFold1Seg dev aenv f a s = [ mkFoldSeg' dev aenv f Nothing a s ]


mkFoldSeg'
    :: forall aenv sh e i. (Shape sh, Elt e, Elt i, IsIntegral i)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (e -> e -> e)
    -> Maybe (CUExp aenv e)
    -> CUDelayedAcc aenv (sh :. Int) e
    -> CUDelayedAcc aenv (Z  :. Int) i
    -> CUTranslSkel aenv (Array (sh :. Int) e)
mkFoldSeg' dev aenv fun@(CUFun2 _ _ combine) mseed
  (CUDelayed (CUExp shIn) _ (CUFun1 _ get))
  (CUDelayed _            _ (CUFun1 _ offset))
  = CUTranslSkel foldSeg [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C"
    __global__ void
    $id:foldSeg
    (
        $params:argIn,
        $params:argOut
    )
    {
        const int vectors_per_block     = blockDim.x / warpSize;
        const int num_vectors           = $exp:(umul24 dev vectors_per_block gridDim);
        const int thread_id             = $exp:(threadIdx dev);
        const int vector_id             = thread_id / warpSize;
        const int thread_lane           = threadIdx.x & (warpSize - 1);
        const int vector_lane           = threadIdx.x / warpSize;

        const int num_segments          = $exp:(cindexHead shOut);
        const int total_segments        = $exp:(csize shOut);
              int seg;
              int ix;

        extern volatile __shared__ int s_ptrs[][2];

        $decls:smem
        $decls:declx
        $decls:decly
        $items:(sh .=. shIn)

        /*
         * Threads in a warp cooperatively reduce a segment
         */
        for ( seg = vector_id
            ; seg < total_segments
            ; seg += num_vectors )
        {
            const int s    =  seg % num_segments;
            const int base = (seg / num_segments) * $exp:(cindexHead sh);

            /*
             * Use two threads to fetch the indices of the start and end of this
             * segment. This results in single coalesced global read.
             */
            if ( thread_lane < 2 ) {
                $items:([cvar "s_ptrs[vector_lane][thread_lane]"] .=. offset [cvar "s + thread_lane"])
            }

            const int start             = base + s_ptrs[vector_lane][0];
            const int end               = base + s_ptrs[vector_lane][1];
            const int num_elements      = end  - start;

            /*
             * Each thread reads in values of this segment, accumulating a local sum
             */
            if ( num_elements > warpSize )
            {
                /*
                 * Ensure aligned access to global memory
                 */
                ix = start - (start & (warpSize - 1)) + thread_lane;

                if ( ix >= start )
                {
                    $items:(y .=. get ix)
                }

                /*
                 * Subsequent reads to global memory are aligned, but make sure all
                 * threads have initialised their local sum.
                 */
                if ( ix + warpSize < end )
                {
                    $items:(x .=. get [cvar "ix + warpSize"])

                    if ( ix >= start ) {
                        $items:(y .=. combine x y)
                    }
                    else {
                        $items:(y .=. x)
                    }
                }

                /*
                 * Now, iterate along the inner-most dimension collecting a local sum
                 */
                for ( ix += 2 * warpSize; ix < end; ix += warpSize )
                {
                    $items:(x .=. get ix)
                    $items:(y .=. combine x y)
                }
            }
            else if ( start + thread_lane < end )
            {
                $items:(y .=. get [cvar "start + thread_lane"])
            }

            /*
             * Store local sums into shared memory and reduce to a single value
             */
            ix = min(num_elements, warpSize);
            $items:(sdata "threadIdx.x" .=. y)
            $items:(reduceWarp dev fun x y sdata (cvar "ix") (cvar "thread_lane"))

            /*
             * Finally, the first thread writes the result for this segment
             */
            if ( thread_lane == 0 )
            {
                $items:(maybe [] exclusive_finish mseed)
                $items:(setOut "seg" .=. y)
            }
        }
    }
  |]
  where
    foldSeg                     = maybe "fold1Seg" (const "foldSeg") mseed
    (texIn, argIn)              = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array (sh :. Int) e)
    (_, x, declx)               = locals "x" (undefined :: e)
    (_, y, decly)               = locals "y" (undefined :: e)
    (sh, _, _)                  = locals "sh" (undefined :: sh :. Int)
    (smem, sdata)               = shared (undefined :: e) "sdata" [cexp| blockDim.x |] (Just $ [cexp| &s_ptrs[vectors_per_block][2] |])
    --
    ix                          = [cvar "ix"]
    vectors_per_block           = cvar "vectors_per_block"
    gridDim                     = cvar "gridDim.x"
    --
    exclusive_finish (CUExp seed)
      = [[citem| if ( num_elements > 0 ) {
                     $items:(x .=. seed)
                     $items:(y .=. combine x y)
                 } else {
                     $items:(y .=. seed)
                 } |]]


-- Reducers
-- --------

-- Reductions of values stored in shared memory.
--
-- Two local (mutable) variables are also required to do the reduction. The
-- final result is stored in the second of these.
--
reduceWarp
    :: forall aenv e. Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]               -- temporary variables x0 and x1
    -> (Name -> [C.Exp])                -- index elements from shared memory
    -> C.Exp                            -- number of elements
    -> C.Exp                            -- thread identifier: usually lane or thread ID
    -> [C.BlockItem]
reduceWarp dev fun x0 x1 sdata n tid
  | shflOK dev (undefined :: e) = return
                                $ reduceWarpShfl dev fun x0 x1       n tid
  | otherwise                   = reduceWarpTree dev fun x0 x1 sdata n tid


reduceBlock
    :: forall aenv e. Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]               -- temporary variables x0 and x1
    -> (Name -> [C.Exp])                -- index elements from shared memory
    -> C.Exp                            -- number of elements
    -> [C.BlockItem]
reduceBlock dev fun x0 x1 sdata n
  | shflOK dev (undefined :: e) = reduceBlockShfl dev fun x0 x1 sdata n
  | otherwise                   = reduceBlockTree dev fun x0 x1 sdata n


-- Tree reduction
-- --------------

reduceWarpTree
    :: Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]               -- temporary variables x0 and x1
    -> (Name -> [C.Exp])                -- index elements from shared memory
    -> C.Exp                            -- number of elements
    -> C.Exp                            -- thread identifier: usually lane or thread ID
    -> [C.BlockItem]
reduceWarpTree dev (CUFun2 _ _ f) x0 x1 sdata n tid
  = map (reduce . pow2) [v, v-1 .. 0]
  where
    v = floor (logBase 2 (fromIntegral $ warpSize dev :: Double))

    pow2 :: Int -> Int
    pow2 x = 2 ^ x

    reduce :: Int -> C.BlockItem
    reduce 0
      = [citem| if ( $exp:tid < $exp:n ) {
                    $items:(x0 .=. sdata "threadIdx.x + 1")
                    $items:(x1 .=. f x1 x0)
                } |]
    reduce i
      = [citem| if ( $exp:tid + $int:i < $exp:n ) {
                    $items:(x0 .=. sdata ("threadIdx.x + " ++ show i))
                    $items:(x1 .=. f x1 x0)
                    $items:(sdata "threadIdx.x" .=. x1)
                } |]

reduceBlockTree
    :: Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]               -- temporary variables x0 and x1
    -> (Name -> [C.Exp])                -- index elements from shared memory
    -> C.Exp                            -- number of elements
    -> [C.BlockItem]
reduceBlockTree dev fun@(CUFun2 _ _ f) x0 x1 sdata n
  = flip (foldr1 (.)) []
  $ map (reduce . pow2) [u-1, u-2 .. v]

  where
    u = floor (logBase 2 (fromIntegral $ maxThreadsPerBlock dev :: Double))
    v = floor (logBase 2 (fromIntegral $ warpSize dev           :: Double))

    pow2 :: Int -> Int
    pow2 x = 2 ^ x

    reduce :: Int -> [C.BlockItem] -> [C.BlockItem]
    reduce i rest
      -- Ensure that threads synchronise before reading from or writing to
      -- shared memory. Synchronising after each reduction step is not enough,
      -- because one warp could update the partial results before a different
      -- warp has read in their data for this step.
      --
      -- Additionally, note that all threads of a warp must participate in the
      -- synchronisation. Thus, this must go outside of the test against the
      -- bounds of the array. We do a bit of extra work here, with all threads
      -- writing into shared memory whether they updated their value or not.
      --
      | i > warpSize dev
      = [citem| __syncthreads(); |]
      : [citem| if ( threadIdx.x + $int:i < $exp:n ) {
                    $items:(x0 .=. sdata ("threadIdx.x + " ++ show i))
                    $items:(x1 .=. f x1 x0)
                } |]
      : [citem| __syncthreads(); |]
      : (sdata "threadIdx.x" .=. x1)
      ++ rest

      -- The threads of a warp execute in lockstep, so it is only necessary to
      -- synchronise at the top, to ensure all threads have written their
      -- results into shared memory.
      --
      | otherwise
      = [citem| __syncthreads(); |]
      : [citem| if ( threadIdx.x < $int:(warpSize dev) ) {
                    $items:(reduceWarpTree dev fun x0 x1 sdata n (cvar "threadIdx.x"))
                } |]
      : rest


-- Butterfly reduction
-- -------------------

shflOK :: Elt e => DeviceProperties -> e -> Bool
shflOK _dev _ = False
-- shflOK dev dummy
--   = computeCapability dev >= Compute 3 0 && all (`elem` [4,8]) (eltSizeOf dummy)


-- Reduction using the __shfl_xor() operation for exchanging variables between
-- threads of a without use of shared memory. The exchange occurs simultaneously
-- for all active threads within the wrap, moving 4 bytes of data per thread.
-- 8-byte quantities are broken into two separate transfers.
--
reduceWarpShfl
    :: forall aenv e. Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]
    -> C.Exp
    -> C.Exp
    -> C.BlockItem
reduceWarpShfl _dev (CUFun2 _ _ f) x0 x1 n tid
  = [citem| for ( int z = warpSize/2; z >= 1; z /= 2 ) {
                $items:(x0 .=. shfl_xor x1)

                if ( $exp:tid + z < $exp:n ) {
                    $items:(x1 .=. f x1 x0)
                }
            } |]
  where
    sizeof      = eltSizeOf (undefined :: e)
    shfl_xor    = zipWith (\s x -> ccall (shfl s) [ x, cvar "z" ]) sizeof
      where
        shfl 4  = "shfl_xor32"
        shfl 8  = "shfl_xor64"
        shfl _  = INTERNAL_ERROR(error) "shfl_xor" "I only know about 32- and 64-bit types"


-- Reduce a block of values in butterfly fashion using __shfl_xor(). Each warp
-- calculates a local reduction, and the first thread of a warp writes its
-- result into shared memory. The first warp then reduces these values to the
-- final result.
--
reduceBlockShfl
    :: forall aenv e. Elt e
    => DeviceProperties
    -> CUFun2 aenv (e -> e -> e)
    -> [C.Exp] -> [C.Exp]
    -> (Name -> [C.Exp])
    -> C.Exp
    -> [C.BlockItem]
reduceBlockShfl dev fun x0 x1 sdata n
  = reduceWarpShfl dev fun x0 x1 n (cvar "threadIdx.x")
  : [citem|  if ( (threadIdx.x & warpSize - 1) == 0 ) {
                 $items:(sdata "threadIdx.x / warpSize" .=. x1)
             } |]
  : [citem|  __syncthreads(); |]
  : [citem|  if ( threadIdx.x < warpSize ) {
                 $items:(x1 .=. sdata "threadIdx.x")
                 $exp:n = ($exp:n + warpSize - 1) / warpSize;
                 $item:(reduceWarpShfl dev fun x0 x1 n (cvar "threadIdx.x"))
             } |]
  : []

