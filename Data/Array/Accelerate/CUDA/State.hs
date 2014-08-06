{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE ScopedTypeVariables        #-}
{-# LANGUAGE TemplateHaskell            #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.State
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
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
  CIO, Context, evalCUDA,

  -- Querying execution state
  defaultContext, deviceProperties, activeContext, kernelTable, memoryTable, streamReservoir,

) where

-- friends
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Debug                 ( message, dump_gc )
import Data.Array.Accelerate.CUDA.Persistent            as KT ( KernelTable, new )
import Data.Array.Accelerate.CUDA.Array.Table           as MT ( MemoryTable, new )
import Data.Array.Accelerate.CUDA.Execute.Stream        as ST ( Reservoir, new )
import Data.Array.Accelerate.CUDA.Analysis.Device

-- library
import Control.Applicative                              ( Applicative )
import Control.Concurrent                               ( runInBoundThread )
import Control.Exception                                ( catch, bracket_ )
import Control.Monad.Trans                              ( MonadIO )
import Control.Monad.Reader                             ( MonadReader, ReaderT(..), runReaderT )
import Control.Monad.State.Strict                       ( MonadState, StateT(..), evalStateT )
import System.Mem                                       ( performGC )
import System.IO.Unsafe                                 ( unsafePerformIO )
import Foreign.CUDA.Driver.Error
import qualified Foreign.CUDA.Driver                    as CUDA


-- Execution State
-- ---------------

-- The state token for CUDA accelerated array operations. This is a stack of
-- (read only) device properties and context, and mutable state for tracking
-- device memory and kernel object code.
--
data State = State {
    memoryTable         :: {-# UNPACK #-} !MemoryTable,                 -- host/device memory associations
    kernelTable         :: {-# UNPACK #-} !KernelTable,                 -- compiled kernel object code
    streamReservoir     :: {-# UNPACK #-} !Reservoir                    -- kernel execution streams
  }

newtype CIO a = CIO {
    runCIO              :: ReaderT Context (StateT State IO) a
  }
  deriving ( Functor, Applicative, Monad, MonadIO
           , MonadReader Context, MonadState State )


-- Extract the active context from the execution state
--
{-# INLINE activeContext #-}
activeContext :: Context -> Context
activeContext = id

-- |Evaluate a CUDA array computation
--
{-# NOINLINE evalCUDA #-}
evalCUDA :: Context -> CIO a -> IO a
evalCUDA !ctx !acc =
  runInBoundThread (bracket_ setup teardown action)
  `catch`
  \e -> $internalError "unhandled" (show (e :: CUDAException))
  where
    setup       = push ctx
    teardown    = pop >> performGC
    action      = evalStateT (runReaderT (runCIO acc) ctx) theState


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
        rsv     <- keepAlive =<< ST.new
        return  $! State mtb ktb rsv


-- Select and initialise a default CUDA device, and create a new execution
-- context. The device is selected based on compute capability and estimated
-- maximum throughput.
--
{-# NOINLINE defaultContext #-}
defaultContext :: Context
defaultContext = unsafePerformIO $ do
  message dump_gc "gc: initialise default context"
  CUDA.initialise []
  (dev,_)       <- selectBestDevice
  create dev [CUDA.SchedAuto]

