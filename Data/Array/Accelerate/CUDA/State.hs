{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.State
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
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
import Control.Applicative                              ( Applicative )
import Control.Exception                                ( bracket )
import Control.Concurrent                               ( forkIO, threadDelay )
import Control.Monad                                    ( when )
import Control.Monad.Trans                              ( MonadIO )
import Control.Monad.Reader                             ( MonadReader, ReaderT(..), runReaderT )
import Control.Monad.State.Strict                       ( MonadState, StateT(..), evalStateT )
import System.Mem                                       ( performGC )
import System.Mem.Weak                                  ( mkWeakPtr, addFinalizer )
import System.IO.Unsafe                                 ( unsafePerformIO )
import Text.PrettyPrint
import qualified Foreign.CUDA.Driver                    as CUDA hiding ( device )
import qualified Foreign.CUDA.Driver.Context            as CUDA


-- Execution State
-- ---------------

-- The state token for CUDA accelerated array operations. This is a stack of
-- (read only) device properties and context, and mutable state for tracking
-- device memory and kernel object code.
--
data Config = Config {
    deviceProps         :: {-# UNPACK #-} !CUDA.DeviceProperties,       -- information on hardware resources
    activeContext       :: {-# UNPACK #-} !Context                      -- device execution context
  }

data State = State {
    memoryTable         :: {-# UNPACK #-} !MemoryTable,                 -- host/device memory associations
    kernelTable         :: {-# UNPACK #-} !KernelTable                  -- compiled kernel object code
  }

newtype CIO a = CIO {
    runCIO              :: ReaderT Config (StateT State IO) a
  }
  deriving ( Functor, Applicative, Monad, MonadIO
           , MonadReader Config, MonadState State )


-- |Evaluate a CUDA array computation
--
{-# NOINLINE evalCUDA #-}
evalCUDA :: CUDA.Context -> CIO a -> IO a
evalCUDA !ctx !acc
  = bracket setup teardown
  $ \config -> evalStateT (runReaderT (runCIO acc) config) theState
  where
    teardown _  = CUDA.pop >> performGC
    setup       = do
      CUDA.push ctx
      dev       <- CUDA.device
      prp       <- CUDA.props dev
      weak_ctx  <- mkWeakPtr ctx Nothing
      return    $! Config prp (Context ctx weak_ctx)


-- Top-level mutable state
-- -----------------------
--
-- It is important to keep some information alive for the entire run of the
-- program, not just a single execution. These tokens use unsafePerformIO to
-- ensure they are executed only once, and reused for subsequent invocations.
--
{-# NOINLINE theState #-}
theState :: State
theState
  = unsafePerformIO
  $ do  message dump_gc "gc: initialise CUDA state"
        mtb     <- keepAlive =<< MT.new
        ktb     <- keepAlive =<< KT.new
        return  $! State mtb ktb


-- Select and initialise a default CUDA device, and create a new execution
-- context. The device is selected based on compute capability and estimated
-- maximum throughput.
--
{-# NOINLINE defaultContext #-}
defaultContext :: CUDA.Context
defaultContext = unsafePerformIO $ do
  CUDA.initialise []
  (dev,prp)     <- selectBestDevice
  ctx           <- CUDA.create dev [CUDA.SchedAuto]

  -- Generated code does not take particular advantage of shared memory, so
  -- for devices that support it use those banks as an L1 cache instead. Perhaps
  -- make this a command line switch: -fprefer-[l1,shared]
  --
  when (CUDA.computeCapability prp >= CUDA.Compute 2 0)
       (CUDA.setCacheConfig CUDA.PreferL1)

  -- Debugging information
  --
  message dump_gc "gc: initialise context"
  message verbose (deviceInfo dev prp)
  addFinalizer ctx $ do
    message dump_gc "gc: finalise context"      -- should never happen!
    CUDA.destroy ctx

  CUDA.pop >> keepAlive ctx


-- Make sure the GC knows that we want to keep this thing alive past the end of
-- 'evalCUDA'.
--
-- We may want to introduce some way to actually shut this down if, for example,
-- the object has not been accessed in a while, and so let it be collected.
--
-- Broken in ghci-7.6.1 Mac OS X due to bug #7299.
--
keepAlive :: a -> IO a
keepAlive x = forkIO (caffeine x) >> return x
  where
    caffeine hit = do threadDelay (5 * 1000 * 1000) -- microseconds = 5 seconds
                      caffeine hit


-- Debugging
-- ---------

-- Nicely format a summary of the selected CUDA device, example:
--
-- Device 0: GeForce 9600M GT (compute capability 1.1)
--           4 multiprocessors @ 1.25GHz (32 cores), 512MB global memory
--
deviceInfo :: CUDA.Device -> CUDA.DeviceProperties -> String
deviceInfo dev prp = render $ reset <>
  devID <> colon <+> vcat [ name <+> parens compute
                          , processors <+> at <+> text clock <+> parens cores <> comma <+> memory
                          ]
  where
    name        = text (CUDA.deviceName prp)
    compute     = text "compute capatability" <+> text (show $ CUDA.computeCapability prp)
    devID       = text "Device" <+> int (fromIntegral $ CUDA.useDevice dev)     -- hax
    processors  = int (CUDA.multiProcessorCount prp)                              <+> text "multiprocessors"
    cores       = int (CUDA.multiProcessorCount prp * coresPerMultiProcessor prp) <+> text "cores"
    memory      = text mem <+> text "global memory"
    --
    clock       = showFFloatSIBase (Just 2) 1000 (fromIntegral $ CUDA.clockRate prp * 1000 :: Double) "Hz"
    mem         = showFFloatSIBase (Just 0) 1024 (fromIntegral $ CUDA.totalGlobalMem prp   :: Double) "B"
    at          = char '@'
    reset       = zeroWidthText "\r"

