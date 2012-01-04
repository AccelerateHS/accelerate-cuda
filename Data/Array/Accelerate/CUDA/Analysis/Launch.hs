{-# LANGUAGE CPP, GADTs #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Analysis.Launch
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Analysis.Launch (

  launchConfig, determineOccupancy

) where

-- friends
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Analysis.Type

-- library
import qualified Foreign.CUDA.Analysis                  as CUDA
import qualified Foreign.CUDA.Driver                    as CUDA

#include "accelerate.h"


-- |
-- Determine kernel launch parameters for the given array computation (as well
-- as compiled function module). This consists of the thread block size, number
-- of blocks, and dynamically allocated shared memory (bytes), respectively.
--
-- For most operations, this selects the minimum block size that gives maximum
-- occupancy, and the grid size limited to the maximum number of physically
-- resident blocks. Hence, kernels may need to process multiple elements per
-- thread. Scan operations select the largest block size of maximum occupancy.
--
launchConfig
    :: OpenAcc aenv a
    -> CUDA.DeviceProperties
    -> CUDA.Occupancy           -- kernel occupancy information
    -> Int                      -- number of elements to configure for
    -> (Int, Int, Int)
launchConfig (OpenAcc acc) dev occ = \n ->
  let cta       = CUDA.activeThreads occ `div` CUDA.activeThreadBlocks occ
      maxGrid   = CUDA.multiProcessorCount dev * CUDA.activeThreadBlocks occ
      smem      = sharedMem dev acc cta
  in
  (cta, maxGrid `min` gridSize dev acc n cta, smem)


-- |
-- Determine maximal occupancy statistics for the given kernel / device
-- combination.
--
determineOccupancy
    :: OpenAcc aenv a
    -> CUDA.DeviceProperties
    -> CUDA.Fun                 -- corresponding __global__ entry function
    -> Int                      -- maximum number of threads per block
    -> IO CUDA.Occupancy
determineOccupancy (OpenAcc acc) dev fn maxBlock = do
  registers     <- CUDA.requires fn CUDA.NumRegs
  static_smem   <- CUDA.requires fn CUDA.SharedSizeBytes        -- static memory only
  return . snd  $  blockSize dev acc maxBlock registers (\threads -> static_smem + dynamic_smem threads)
  where
    dynamic_smem = sharedMem dev acc


-- |
-- Determine an optimal thread block size for a given array computation. Fold
-- requires blocks with a power-of-two number of threads. Scans select the
-- largest size thread block possible, because if only one thread block is
-- needed we can calculate the scan in a single pass.
--
blockSize
    :: CUDA.DeviceProperties
    -> PreOpenAcc OpenAcc aenv a
    -> Int                      -- maximum number of threads per block
    -> Int                      -- number of registers used
    -> (Int -> Int)             -- shared memory as a function of thread block size (bytes)
    -> (Int, CUDA.Occupancy)
blockSize dev acc lim regs smem =
  CUDA.optimalBlockSizeBy dev (filter (<= lim) . strategy) (const regs) smem
  where
    strategy = case acc of
      Fold _ _ _        -> CUDA.decPow2
      Fold1 _ _         -> CUDA.decPow2
      Scanl _ _ _       -> CUDA.incWarp
      Scanl' _ _ _      -> CUDA.incWarp
      Scanl1 _ _        -> CUDA.incWarp
      Scanr _ _ _       -> CUDA.incWarp
      Scanr' _ _ _      -> CUDA.incWarp
      Scanr1 _ _        -> CUDA.incWarp
      _                 -> CUDA.decWarp


-- |
-- Determine the number of blocks of the given size necessary to process the
-- given array expression. This should understand things like #elements per
-- thread for the various kernels.
--
-- foldSeg: 'size' is the number of segments, require one warp per segment
--
gridSize :: CUDA.DeviceProperties -> PreOpenAcc OpenAcc aenv a -> Int -> Int -> Int
gridSize p acc@(FoldSeg _ _ _ _) size cta = split acc (size * CUDA.warpSize p) cta
gridSize p acc@(Fold1Seg _ _ _)  size cta = split acc (size * CUDA.warpSize p) cta
gridSize _ acc                   size cta = split acc size cta

split :: PreOpenAcc OpenAcc aenv a -> Int -> Int -> Int
split acc size cta = (size `between` eltsPerThread acc) `between` cta
  where
    between arr n   = 1 `max` ((n + arr - 1) `div` n)
    eltsPerThread _ = 1


-- |
-- Analyse the given array expression, returning an estimate of dynamic shared
-- memory usage as a function of thread block size. This can be used by the
-- occupancy calculator to optimise kernel launch shape.
--
sharedMem :: CUDA.DeviceProperties -> PreOpenAcc OpenAcc aenv a -> Int -> Int
-- non-computation forms
sharedMem _ (Alet _ _)     _ = INTERNAL_ERROR(error) "sharedMem" "Let"
sharedMem _ (Alet2 _ _)    _ = INTERNAL_ERROR(error) "sharedMem" "Let2"
sharedMem _ (PairArrays _ _) _
                             = INTERNAL_ERROR(error) "sharedMem" "PairArrays"
sharedMem _ (Avar _)      _  = INTERNAL_ERROR(error) "sharedMem" "Avar"
sharedMem _ (Apply _ _)   _  = INTERNAL_ERROR(error) "sharedMem" "Apply"
sharedMem _ (Acond _ _ _) _  = INTERNAL_ERROR(error) "sharedMem" "Acond"
sharedMem _ (Use _)       _  = INTERNAL_ERROR(error) "sharedMem" "Use"
sharedMem _ (Unit _)      _  = INTERNAL_ERROR(error) "sharedMem" "Unit"
sharedMem _ (Reshape _ _) _  = INTERNAL_ERROR(error) "sharedMem" "Reshape"

-- skeleton nodes
sharedMem _ (Generate _ _)       _        = 0
sharedMem _ (Replicate _ _ _)    _        = 0
sharedMem _ (Index _ _ _)        _        = 0
sharedMem _ (Map _ _)            _        = 0
sharedMem _ (ZipWith _ _ _)      _        = 0
sharedMem _ (Permute _ _ _ _)    _        = 0
sharedMem _ (Backpermute _ _ _)  _        = 0
sharedMem _ (Stencil _ _ _)      _        = 0
sharedMem _ (Stencil2 _ _ _ _ _) _        = 0
sharedMem _ (Fold  _ _ a)        blockDim = sizeOf (accType a) * blockDim
sharedMem _ (Fold1 _ a)          blockDim = sizeOf (accType a) * blockDim
sharedMem _ (Scanl _ x _)        blockDim = sizeOf (expType x) * blockDim
sharedMem _ (Scanr _ x _)        blockDim = sizeOf (expType x) * blockDim
sharedMem _ (Scanl' _ x _)       blockDim = sizeOf (expType x) * blockDim
sharedMem _ (Scanr' _ x _)       blockDim = sizeOf (expType x) * blockDim
sharedMem _ (Scanl1 _ a)         blockDim = sizeOf (accType a) * blockDim
sharedMem _ (Scanr1 _ a)         blockDim = sizeOf (accType a) * blockDim
sharedMem p (FoldSeg _ _ a _)    blockDim =
  (blockDim `div` CUDA.warpSize p) * 8 + blockDim * sizeOf (accType a)
sharedMem p (Fold1Seg _ a _) blockDim =
  (blockDim `div` CUDA.warpSize p) * 8 + blockDim * sizeOf (accType a)

