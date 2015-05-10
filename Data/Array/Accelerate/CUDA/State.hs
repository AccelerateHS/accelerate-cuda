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
  CIO, Context, evalCUDA, evalCUDAState,

  -- Querying execution state
  defaultContext, deviceProperties, activeContext, kernelTable, memoryTable, streamReservoir,
  eventTable,

) where

-- friends
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Debug                 ( traceIO, dump_gc )
import Data.Array.Accelerate.CUDA.Persistent            as KT ( KernelTable, new )
import Data.Array.Accelerate.CUDA.Array.Cache           as MT ( MemoryTable, new )
import Data.Array.Accelerate.CUDA.Execute.Stream        as ST ( Reservoir, new )
import Data.Array.Accelerate.CUDA.Execute.Event         as ET ( EventTable, new)
import Data.Array.Accelerate.CUDA.Analysis.Device

-- library
import Control.Applicative
import Control.Concurrent                               ( runInBoundThread )
import Control.Exception                                ( catch, bracket_ )
import Control.Monad.Reader                             ( MonadReader, ReaderT(..), runReaderT )
import Control.Monad.State.Strict                       ( MonadState, StateT(..), evalStateT )
import Control.Monad.Trans                              ( MonadIO )
import System.IO.Unsafe                                 ( unsafePerformIO )
import Foreign.CUDA.Driver.Error
import qualified Foreign.CUDA.Driver                    as CUDA
import Prelude


-- Execution State
-- ---------------

-- The state token for CUDA accelerated array operations. This is a stack of
-- (read only) device properties and context, and mutable state for tracking
-- device memory and kernel object code.
--
data State = State {
    memoryTable         :: {-# UNPACK #-} !MemoryTable,                 -- host/device memory associations
    kernelTable         :: {-# UNPACK #-} !KernelTable,                 -- compiled kernel object code
    streamReservoir     :: {-# UNPACK #-} !Reservoir,                   -- kernel execution streams
    eventTable          :: {-# UNPACK #-} !EventTable                   -- CUDA events
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
    teardown    = pop
    action      = evalStateT (runReaderT (runCIO acc) ctx) theState

-- |Evaluate a CUDA array computation with the specific state. Exceptions are
-- not caught.
--
-- RCE: This is unfortunately hacky, but necessary to stop device pointers
-- leaking.
evalCUDAState :: Context -> MemoryTable -> KernelTable -> Reservoir -> EventTable -> CIO a -> IO a
evalCUDAState ctx mt kt rsv etbl acc = evalStateT (runReaderT (runCIO acc) ctx)
                                                  (State mt kt rsv etbl)

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
  $ do  message "initialise CUDA state"
        etbl    <- keepAlive =<< ET.new
        mtb     <- keepAlive =<< MT.new etbl
        ktb     <- keepAlive =<< KT.new
        rsv     <- keepAlive =<< ST.new
        return  $! State mtb ktb rsv etbl


-- Select and initialise a default CUDA device, and create a new execution
-- context. The device is selected based on compute capability and estimated
-- maximum throughput.
--
{-# NOINLINE defaultContext #-}
defaultContext :: Context
defaultContext = unsafePerformIO $ do
  message "initialise default context"
  CUDA.initialise []
  (dev,_)       <- selectBestDevice
  create dev [CUDA.SchedAuto]


-- Debug
-- -----

{-# INLINE message #-}
message :: String -> IO ()
message msg = traceIO dump_gc ("gc: " ++ msg)

