-- |
-- Module      : Data.Array.Accelerate.CUDA.Analysis.Device
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Analysis.Device
  where

import Data.Ord
import Data.List
import Data.Function
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Analysis.Device
import qualified Foreign.CUDA.Driver    as CUDA


-- Select the best of the available CUDA capable devices. This prefers devices
-- with higher compute capability, followed by maximum throughput. This does not
-- take into account any other factors, such as whether the device is currently
-- in use by another process.
--
-- Ignore the possibility of emulation-mode devices, as this has been deprecated
-- as of CUDA v3.0 (compute-capability == 9999.9999)
--
selectBestDevice :: IO (Device, DeviceProperties)
selectBestDevice = do
  dev   <- mapM CUDA.device . enumFromTo 0 . subtract 1 =<< CUDA.count
  prop  <- mapM CUDA.props dev
  return . head . sortBy (flip cmp `on` snd) $ zip dev prop
  where
    compute     = computeCapability
    flops d     = multiProcessorCount d * coresPerMultiProcessor d * clockRate d
    cmp x y
      | compute x == compute y  = comparing flops   x y
      | otherwise               = comparing compute x y


-- Number of CUDA cores per streaming multiprocessor for a given architecture
-- revision. This is the number of SIMD arithmetic units per multiprocessor,
-- executing in lockstep in half-warp groupings (16 ALUs).
--
coresPerMultiProcessor :: DeviceProperties -> Int
coresPerMultiProcessor dev =
  let Compute major minor  = computeCapability dev
  in case (major, minor) of
    (1, 0)      -> 8     -- Tesla G80
    (1, 1)      -> 8     -- Tesla G8x
    (1, 2)      -> 8     -- Tesla G9x
    (1, 3)      -> 8     -- Tesla GT200
    (2, 1)      -> 32    -- Fermi GF100
    (2, 2)      -> 48    -- Fermi GF10x
    (3, 0)      -> 192   -- Kepler GK10x
    (3, 5)      -> 192   -- Kepler GK11x

    _           -> -1    -- unknown

