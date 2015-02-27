{-# LANGUAGE GADTs           #-}
{-# LANGUAGE TemplateHaskell #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Analysis.Launch
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Analysis.Launch (

  launchConfig, determineOccupancy

) where

-- friends
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate.Analysis.Type
import Data.Array.Accelerate.Analysis.Shape

-- library
import qualified Foreign.CUDA.Analysis                  as CUDA
import qualified Foreign.CUDA.Driver                    as CUDA


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
    :: DelayedOpenAcc aenv a
    -> CUDA.DeviceProperties    -- the device being executed on
    -> CUDA.Occupancy           -- kernel occupancy information
    -> ( Int                    -- block size
       , Int -> Int             -- number of blocks for input problem size (grid)
       , Int )                  -- shared memory (bytes)
launchConfig Delayed{} _ _ = $internalError "launchConfig" "encountered delayed array"
launchConfig (Manifest acc) dev occ =
  let cta       = CUDA.activeThreads occ `div` CUDA.activeThreadBlocks occ
      maxGrid   = CUDA.multiProcessorCount dev * CUDA.activeThreadBlocks occ
      smem      = sharedMem dev acc cta
  in
  (cta, \n -> maxGrid `min` gridSize dev acc n cta, smem)


-- |
-- Determine maximal occupancy statistics for the given kernel / device
-- combination.
--
determineOccupancy
    :: DelayedOpenAcc aenv a
    -> CUDA.DeviceProperties
    -> CUDA.Fun                 -- corresponding __global__ entry function
    -> Int                      -- maximum number of threads per block
    -> IO CUDA.Occupancy
determineOccupancy Delayed{} _ _ _ = $internalError "determineOccupancy" "encountered delayed array"
determineOccupancy (Manifest acc) dev fn maxBlock = do
  registers     <- CUDA.requires fn CUDA.NumRegs
  static_smem   <- CUDA.requires fn CUDA.SharedSizeBytes        -- static memory only
  return . snd  $  blockSize dev acc maxBlock registers (\threads -> static_smem + dynamic_smem threads)
  where
    dynamic_smem = sharedMem dev acc


-- |
-- Determine an optimal thread block size for a given array computation. Fold
-- requires blocks with a power-of-two number of threads. Scans select the
-- largest size thread block possible, because if only one thread block is
-- needed we can calculate the scan in a single pass, rather than three.
--
blockSize
    :: CUDA.DeviceProperties
    -> PreOpenAcc DelayedOpenAcc aenv a
    -> Int                      -- maximum number of threads per block
    -> Int                      -- number of registers used
    -> (Int -> Int)             -- shared memory as a function of thread block size (bytes)
    -> (Int, CUDA.Occupancy)
blockSize dev acc lim regs smem =
  CUDA.optimalBlockSizeBy dev (filter (<= lim) . strategy) (const regs) smem
  where
    strategy = case acc of
      Fold _ _ _        -> CUDA.incPow2
      Fold1 _ _         -> CUDA.incPow2
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
-- The 'size' parameter is typically the number of elements in the array, except
-- for the following instances:
--
--  * foldSeg: the number of segments; require one warp per segment
--
--  * fold: for multidimensional reductions, this is the size of the shape tail
--          for 1D reductions this is the total number of elements
--
gridSize :: CUDA.DeviceProperties -> PreOpenAcc DelayedOpenAcc aenv a -> Int -> Int -> Int
gridSize p (FoldSeg _ _ _ _) size cta = split (size * CUDA.warpSize p) cta
gridSize p (Fold1Seg _ _ _)  size cta = split (size * CUDA.warpSize p) cta
gridSize _ acc@(Fold _ _ _)  size cta = if preAccDim delayedDim acc == 0 then split size cta else max 1 size
gridSize _ acc@(Fold1 _ _)   size cta = if preAccDim delayedDim acc == 0 then split size cta else max 1 size
gridSize _ _                 size cta = split size cta

split :: Int -> Int -> Int
split size cta = (size `between` eltsPerThread) `between` cta
  where
    between arr n   = 1 `max` ((n + arr - 1) `div` n)
    eltsPerThread   = 1


-- |
-- Analyse the given array expression, returning an estimate of dynamic shared
-- memory usage as a function of thread block size. This can be used by the
-- occupancy calculator to optimise kernel launch shape.
--
sharedMem :: CUDA.DeviceProperties -> PreOpenAcc DelayedOpenAcc aenv a -> Int -> Int
-- non-computation forms
sharedMem _ Alet{}     _ = $internalError "sharedMem" "Let"
sharedMem _ Avar{}     _ = $internalError "sharedMem" "Avar"
sharedMem _ Apply{}    _ = $internalError "sharedMem" "Apply"
sharedMem _ Acond{}    _ = $internalError "sharedMem" "Acond"
sharedMem _ Awhile{}   _ = $internalError "sharedMem" "Awhile"
sharedMem _ Atuple{}   _ = $internalError "sharedMem" "Atuple"
sharedMem _ Aprj{}     _ = $internalError "sharedMem" "Aprj"
sharedMem _ Use{}      _ = $internalError "sharedMem" "Use"
sharedMem _ Unit{}     _ = $internalError "sharedMem" "Unit"
sharedMem _ Reshape{}  _ = $internalError "sharedMem" "Reshape"
sharedMem _ Aforeign{} _ = $internalError "sharedMem" "Aforeign"

-- skeleton nodes
sharedMem _ Generate{}          _        = 0
sharedMem _ Transform{}         _        = 0
sharedMem _ Replicate{}         _        = 0
sharedMem _ Slice{}             _        = 0
sharedMem _ Map{}               _        = 0
sharedMem _ ZipWith{}           _        = 0
sharedMem _ Permute{}           _        = 0
sharedMem _ Backpermute{}       _        = 0
sharedMem _ Stencil{}           _        = 0
sharedMem _ Stencil2{}          _        = 0
sharedMem _ (Fold  _ x _)       blockDim = sizeOf (delayedExpType x) * blockDim
sharedMem _ (Scanl _ x _)       blockDim = sizeOf (delayedExpType x) * blockDim
sharedMem _ (Scanr _ x _)       blockDim = sizeOf (delayedExpType x) * blockDim
sharedMem _ (Scanl' _ x _)      blockDim = sizeOf (delayedExpType x) * blockDim
sharedMem _ (Scanr' _ x _)      blockDim = sizeOf (delayedExpType x) * blockDim
sharedMem _ (Fold1 _ a)         blockDim = sizeOf (delayedAccType a) * blockDim
sharedMem _ (Scanl1 _ a)        blockDim = sizeOf (delayedAccType a) * blockDim
sharedMem _ (Scanr1 _ a)        blockDim = sizeOf (delayedAccType a) * blockDim
sharedMem p (FoldSeg _ x _ _)   blockDim =
  (blockDim `div` CUDA.warpSize p) * 8 + blockDim * sizeOf (delayedExpType x)  -- TLM: why 8? I can't remember...
sharedMem p (Fold1Seg _ a _) blockDim =
  (blockDim `div` CUDA.warpSize p) * 8 + blockDim * sizeOf (delayedAccType a)
sharedMem _ (Collect _)         _        = 0

