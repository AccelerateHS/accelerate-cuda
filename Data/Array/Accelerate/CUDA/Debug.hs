{-# LANGUAGE CPP #-}
{-# OPTIONS -fno-warn-unused-binds   #-}
{-# OPTIONS -fno-warn-unused-imports #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Debug
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- Hijack some command line arguments to pass runtime debugging options. This
-- might cause problems for users of the library...
--

module Data.Array.Accelerate.CUDA.Debug (

  module Data.Array.Accelerate.Debug,
  module Data.Array.Accelerate.CUDA.Debug,

) where

import Data.Array.Accelerate.Debug

import Control.Concurrent                               ( forkIO )
import Control.Monad                                    ( void )
import Control.Monad.IO.Class                           ( liftIO, MonadIO )
import GHC.Float                                        ( float2Double )
import System.CPUTime
import System.IO.Unsafe

import Foreign.CUDA.Driver.Stream                       ( Stream )
import qualified Foreign.CUDA.Driver.Event              as Event


-- | Execute an action and time the results. The second argument specifies how
-- to format the output string given elapsed GPU and CPU time respectively
--
timed :: Mode -> (Double -> Double -> String) -> Maybe Stream -> IO () -> IO ()
#ifdef ACCELERATE_DEBUG
{-# NOINLINE timed #-}
timed f fmt stream action = do
  enabled <- queryFlag f
  if enabled
    then do
      gpuBegin  <- Event.create []
      gpuEnd    <- Event.create []
      cpuBegin  <- getCPUTime
      Event.record gpuBegin stream
      action
      Event.record gpuEnd stream
      cpuEnd    <- getCPUTime

      -- Wait for the GPU to finish executing then display the timing execution
      -- message. Do this in a separate thread so that the remaining kernels can
      -- be queued asynchronously.
      --
      void . forkIO $ do
        Event.block gpuEnd
        diff    <- Event.elapsedTime gpuBegin gpuEnd
        let gpuTime = float2Double $ diff * 1E-3                         -- milliseconds
            cpuTime = fromIntegral (cpuEnd - cpuBegin) * 1E-12 :: Double -- picoseconds

        Event.destroy gpuBegin
        Event.destroy gpuEnd
        --
        traceIO f (fmt gpuTime cpuTime)

    else
      action
#else
{-# INLINE timed #-}
timed _ _ _ action = action
#endif

{-# INLINE elapsed #-}
elapsed :: Double -> Double -> String
elapsed gpuTime cpuTime
  = "gpu: " ++ showFFloatSIBase (Just 3) 1000 gpuTime "s, " ++
    "cpu: " ++ showFFloatSIBase (Just 3) 1000 cpuTime "s"

