{-# LANGUAGE BangPatterns, CPP, GADTs #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- This module implements the CUDA backend for the embedded array language
-- Accelerate. Expressions are on-line compiled into CUDA code, compiled, and
-- executed in parallel on the GPU.
--

module Data.Array.Accelerate.CUDA (

  Arrays,

  -- * Synchronous execution
  run, run1, stream, runIn, run1In, streamIn,

  -- * Asynchronous execution
  Async, wait, poll, cancel,
  runAsync, run1Async, runAsyncIn, run1AsyncIn

) where

-- standard library
import Prelude                                          hiding ( catch )
import Control.Exception
import Control.Applicative
import Control.Concurrent
import System.IO.Unsafe
import Foreign.CUDA.Driver                              ( Context )
import Foreign.CUDA.Driver.Error

-- friends
import Data.Array.Accelerate.AST                        ( Arrays(..), ArraysR(..) )
import Data.Array.Accelerate.Smart                      ( Acc, convertAcc, convertAccFun1 )
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Compile
import Data.Array.Accelerate.CUDA.Execute

#include "accelerate.h"


-- Accelerate: CUDA
-- ----------------

-- | Compile and run a complete embedded array program using the CUDA backend
--
run :: Arrays a => Acc a -> a
run = runIn defaultContext

runAsync :: Arrays a => Acc a -> Async a
runAsync = runAsyncIn defaultContext

-- | As 'run', but execute using the specified device context rather than
-- creating a new context for an automatically selected device
--
{-# NOINLINE runIn #-}
runIn :: Arrays a => Context -> Acc a -> a
runIn ctx a = unsafePerformIO $ evaluate (runAsyncIn ctx a) >>= wait

{-# NOINLINE runAsyncIn #-}
runAsyncIn :: Arrays a => Context -> Acc a -> Async a
runAsyncIn ctx a = unsafePerformIO $ async execute
  where
    acc     = convertAcc a
    execute = evalCUDA ctx (compileAcc acc >>= executeAcc >>= collect)
              `catch`
              \e -> INTERNAL_ERROR(error) "unhandled" (show (e :: CUDAException))


-- |Prepare and execute an embedded array program of one argument. This function
-- can be used to improve performance in cases where the array program is
-- constant between invocations.
--
run1 :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> a -> b
run1 = run1In defaultContext

run1Async :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> a -> Async b
run1Async = run1AsyncIn defaultContext

{-# NOINLINE run1In #-}
run1In :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> a -> b
run1In ctx f = let go = run1AsyncIn ctx f
               in \a -> unsafePerformIO $ wait (go a)

-- TLM: We need to be very careful with run1 and run1Async to ensure that the
--      returned closure shortcuts directly to the execution phase.
--
{-# NOINLINE run1AsyncIn #-}
run1AsyncIn :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> a -> Async b
run1AsyncIn ctx f = \a -> unsafePerformIO $ async (execute a)
  where
    acc       = convertAccFun1 f
    !afun     = unsafePerformIO $ evalCUDA ctx (compileAfun1 acc)
    execute a = evalCUDA ctx (executeAfun1 afun a >>= collect)
                `catch`
                \e -> INTERNAL_ERROR(error) "unhandled" (show (e :: CUDAException))


-- |Stream a lazily read list of input arrays through the given program,
-- collecting results as we go
--
stream :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> [a] -> [b]
stream f arrs = streamIn defaultContext f arrs

streamIn :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> [a] -> [b]
streamIn ctx f arrs = let go = run1In ctx f
                      in  map go arrs


-- Copy from device to host, and decrement the usage counter. This last step
-- should result in all transient arrays having been removed from the device.
--
collect :: Arrays arrs => arrs -> CIO arrs
collect arrs = collectR arrays arrs
  where
    collectR :: ArraysR arrs -> arrs -> CIO arrs
    collectR ArraysRunit         ()             = return ()
    collectR ArraysRarray        arr            = peekArray arr >> return arr
    collectR (ArraysRpair r1 r2) (arrs1, arrs2) = (,) <$> collectR r1 arrs1
                                                      <*> collectR r2 arrs2


-- Running asynchronously
-- ----------------------
--
-- We need to execute the main thread asynchronously to give finalisers a chance
-- to run. Make sure to catch exceptions to avoid "blocked indefinitely on MVar"
-- errors.
--

data Async a = Async !ThreadId !(MVar (Either SomeException a))

async :: IO a -> IO (Async a)
async action = do
   var <- newEmptyMVar
   tid <- forkIO $ (putMVar var . Right =<< action)
                   `catch`
                   \e -> putMVar var (Left e)
   return (Async tid var)

wait :: Async a -> IO a
wait (Async _ var) = either throwIO return =<< readMVar var

poll :: Async a -> IO (Maybe a)
poll (Async _ var) =
  maybe (return Nothing) (either throwIO (return . Just)) =<< tryTakeMVar var

cancel :: Async a -> IO ()
cancel (Async tid _) = throwTo tid ThreadKilled

