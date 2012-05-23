{-# LANGUAGE CPP             #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TupleSections   #-}
{-# LANGUAGE TypeOperators   #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}   -- Eq CUDA.Context
-- |
-- Module      : Data.Array.Accelerate.CUDA.State
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--
-- This module defines a state monad token which keeps track of the code
-- generator state, including memory transfers and external compilation
-- processes.
--

module Data.Array.Accelerate.CUDA.State (

  -- Evaluating computations
  CIO, evalCUDA,

  -- Querying execution state
  defaultContext, deviceProps, activeContext, kernelTable, memoryTable

) where

-- friends
import Data.Array.Accelerate.CUDA.Debug                 ( message, verbose, dump_gc, showFFloatSIBase )
import Data.Array.Accelerate.CUDA.Persistent            as KT
import Data.Array.Accelerate.CUDA.Array.Table           as MT
import Data.Array.Accelerate.CUDA.Analysis.Device

-- library
import Data.Label
import Control.Exception
import Control.Concurrent                               ( forkIO, threadDelay )
import Control.Concurrent.MVar                          ( MVar, newMVar )
import Control.Monad.State.Strict                       ( StateT(..), evalStateT )
import System.Mem                                       ( performGC )
import System.Mem.Weak                                  ( mkWeakPtr, addFinalizer )
import System.IO.Unsafe                                 ( unsafePerformIO )
import Text.PrettyPrint
import qualified Foreign.CUDA.Driver                    as CUDA hiding ( device )
import qualified Foreign.CUDA.Driver.Context            as CUDA


-- The state token for CUDA accelerated array operations
--
type CIO        = StateT CUDAState IO
data CUDAState  = CUDAState
  {
    _deviceProps        :: !CUDA.DeviceProperties,
    _activeContext      :: {-# UNPACK #-} !Context,
    _kernelTable        :: {-# UNPACK #-} !KernelTable,
    _memoryTable        :: {-# UNPACK #-} !MemoryTable
  }

instance Eq CUDA.Context where
  CUDA.Context p1 == CUDA.Context p2    = p1 == p2

$(mkLabels [''CUDAState])


-- Execution State
-- ---------------

-- |Evaluate a CUDA array computation
--
evalCUDA :: CUDA.Context -> CIO a -> IO a
evalCUDA ctx acc = bracket setup teardown $ evalStateT acc
  where
    teardown _  = CUDA.pop >> performGC
    setup       = do
      CUDA.push ctx
      dev       <- CUDA.device
      prp       <- CUDA.props dev
      weak_ctx  <- mkWeakPtr ctx Nothing
      return $! CUDAState prp (Context ctx weak_ctx) theKernelTable theMemoryTable


-- Top-level mutable state
-- -----------------------
--
-- It is important to keep some information alive for the entire run of the
-- program, not just a single execution. These tokens use unsafePerformIO to
-- ensure they are executed only once, and reused for subsequent invocations.
--

{-# NOINLINE theMemoryTable #-}
theMemoryTable :: MemoryTable
theMemoryTable = unsafePerformIO $ do
  message dump_gc "gc: initialise memory table"
  keepAlive =<< MT.new


{-# NOINLINE theKernelTable #-}
theKernelTable :: KernelTable
theKernelTable = unsafePerformIO $ do
  message dump_gc "gc: initialise kernel table"
  keepAlive =<< KT.new


-- Select and initialise a default CUDA device, and create a new execution
-- context. The device is selected based on compute capability and estimated
-- maximum throughput.
--
{-# NOINLINE defaultContext #-}
defaultContext :: MVar CUDA.Context
defaultContext = unsafePerformIO $ do
  CUDA.initialise []
  (dev,prp)     <- selectBestDevice
  ctx           <- CUDA.create dev [CUDA.SchedAuto] >> CUDA.pop
  ref           <- newMVar ctx
  --
  message dump_gc $ "gc: initialise context"
  message verbose $ deviceInfo dev prp
  --
  addFinalizer ctx $ do
    message dump_gc $ "gc: finalise context"    -- should never happen!
    CUDA.destroy ctx
  --
  keepAlive ref


-- Make sure the GC knows that we want to keep this thing alive past the end of
-- 'evalCUDA'.
--
-- We may want to introduce some way to actually shut this down if, for example,
-- the object has not been accessed in a while, and so let it be collected.
--
keepAlive :: a -> IO a
keepAlive x = forkIO (caffeine x) >> return x
  where
    caffeine hit = do threadDelay 5000000 -- microseconds = 5 seconds
                      caffeine hit


-- Debugging
-- ---------

-- Nicely format a summary of the selected CUDA device, example:
--
-- Device 0: GeForce 9600M GT (compute capability 1.1)
--           4 multiprocessors @ 1.25GHz (32 cores), 512MB global memory
--
deviceInfo :: CUDA.Device -> CUDA.DeviceProperties -> String
deviceInfo dev prp = render $
  devID <> colon <+> vcat [ name <+> parens compute
                          , processors <+> at <+> text clock <+> parens cores <> comma <+> memory
                          ]
  where
    name        = text (CUDA.deviceName prp)
    compute     = text "compute capatability" <+> double (CUDA.computeCapability prp)
    devID       = text "Device" <+> int (fromIntegral $ CUDA.useDevice dev)     -- hax
    processors  = int (CUDA.multiProcessorCount prp)                              <+> text "multiprocessors"
    cores       = int (CUDA.multiProcessorCount prp * coresPerMultiProcessor prp) <+> text "cores"
    memory      = text mem <+> text "global memory"
    --
    clock       = showFFloatSIBase (Just 2) 1000 (fromIntegral $ CUDA.clockRate prp * 1000 :: Double) "Hz"
    mem         = showFFloatSIBase (Just 0) 1024 (fromIntegral $ CUDA.totalGlobalMem prp   :: Double) "B"
    at          = char '@'

