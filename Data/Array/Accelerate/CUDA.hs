{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- This module implements the CUDA backend for the embedded array language
-- Accelerate. Expressions are on-line translated into CUDA code, compiled, and
-- executed in parallel on the GPU.
--
-- The accelerate-cuda library is hosted at: <https://github.com/AccelerateHS/accelerate-cuda>.
-- Comments, bug reports, and patches, are always welcome.
--
--
-- /NOTES:/
--
-- CUDA devices are categorised into different \'compute capabilities\',
-- indicating what operations are supported by the hardware. For example, double
-- precision arithmetic is only supported on devices of compute capability 1.3
-- or higher.
--
-- Devices generally perform best when dealing with (tuples of) 32-bit types, so
-- be cautious when introducing 8-, 16-, or 64-bit elements. Keep in mind the
-- size of 'Int' and 'Data.Word.Word' changes depending on the architecture GHC
-- runs on.
--
-- Additional notes:
--
--  * 'Double' precision requires compute-1.3.
--
--  * 'Bool' is represented internally using 'Data.Word.Word8', 'Char' by
--    'Data.Word.Word32'.
--
--  * If the permutation function to 'Data.Array.Accelerate.permute' resolves to
--    non-unique indices, the combination function requires compute-1.1 to
--    combine 32-bit types, or compute-1.2 for 64-bit types. Tuple components
--    are resolved separately.
--

module Data.Array.Accelerate.CUDA (

  Arrays,

  -- * Synchronous execution
  run, run1, stream, runIn, run1In, streamIn,

  -- * Asynchronous execution
  Async, wait, poll, cancel,
  runAsync, run1Async, runAsyncIn, run1AsyncIn,

  -- * Execution contexts
  Context, create, destroy,

) where

-- standard library
#if !MIN_VERSION_base(4,6,0)
import Prelude                                          hiding ( catch )
#endif
import Control.Exception
import Control.Applicative
import Control.Concurrent
import System.IO.Unsafe
import Foreign.CUDA.Driver.Error

-- friends
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate.Smart                      ( Acc )
import Data.Array.Accelerate.Array.Sugar                ( Arrays(..), ArraysR(..) )
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Compile
import Data.Array.Accelerate.CUDA.Execute

#include "accelerate.h"


-- Accelerate: CUDA
-- ----------------

-- | Compile and run a complete embedded array program using the CUDA backend.
-- This will select the fastest device available on which to execute
-- computations, based on compute capability and estimated maximum GFLOPS.
--
-- /NOTE:/
--   GPUs typically have their own attached memory, which is separate from the
--   computer's main memory. Hence, every 'Data.Array.Accelerate.use' operation
--   implies copying data to the device, and every 'run' operation must copy the
--   results of a computation back to the host. Thus, it is best to keep all
--   computations in the 'Acc' meta-language form and only 'run' the computation
--   once at the end, to avoid transferring (unused) intermediate results.
--
run :: Arrays a => Acc a -> a
run a
  = unsafePerformIO
  $ evaluate (runIn defaultContext a)

-- | As 'run', but allow the computation to continue running in a thread and
-- return immediately without waiting for the result. The status of the
-- computation can be queried using 'wait', 'poll', and 'cancel'.
--
-- Note that a CUDA Context can only be active no one host thread at a time. If
-- you want to execute multiple computations in parallel, use 'runAsyncIn'.
--
runAsync :: Arrays a => Acc a -> Async a
runAsync a
  = unsafePerformIO
  $ evaluate (runAsyncIn defaultContext a)

-- | As 'run', but execute using the specified device context rather than using
-- the default, automatically selected device.
--
-- Contexts passed to this function may all refer to the same device, or to
-- separate devices of differing compute capabilities.
--
-- Note that each thread has a stack of current contexts, and calling
-- 'Foreign.CUDA.Driver.Context.create' pushes the new context on top of the
-- stack and makes it current with the calling thread. You should call
-- 'Foreign.CUDA.Driver.Context.pop' to make the context floating before passing
-- it to 'runIn', which will make it current for the duration of evaluating the
-- expression. See the CUDA C Programming Guide (G.1) for more information.
--
runIn :: Arrays a => Context -> Acc a -> a
runIn ctx a
  = unsafePerformIO
  $ evaluate (runAsyncIn ctx a) >>= wait


-- | As 'runIn', but execute asynchronously. Be sure not to destroy the context,
-- or attempt to attach it to a different host thread, before all outstanding
-- operations have completed.
--
runAsyncIn :: Arrays a => Context -> Acc a -> Async a
runAsyncIn ctx a = unsafePerformIO $ async execute
  where
    !acc    = convertAccWith config a
    execute = evalCUDA ctx (compileAcc acc >>= executeAcc >>= collect)
              `catch`
              \e -> INTERNAL_ERROR(error) "unhandled" (show (e :: CUDAException))


-- | Prepare and execute an embedded array program of one argument.
--
-- This function can be used to improve performance in cases where the array
-- program is constant between invocations, because it allows us to bypass all
-- front-end conversion stages and move directly to the execution phase. If you
-- have a computation applied repeatedly to different input data, use this. If
-- the function is only evaluated once, this is equivalent to 'run'.
--
-- >  let step :: Vector a -> Vector b
-- >      step = run1 f
-- >  in
-- >  simulate step ...
--
-- See the Crystal demo, part of the 'accelerate-examples' package, for an
-- example.
--
run1 :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> a -> b
run1 f
  = unsafePerformIO
  $ evaluate (run1In defaultContext f)


-- | As 'run1', but the computation is executed asynchronously.
--
run1Async :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> a -> Async b
run1Async f
  = unsafePerformIO
  $ evaluate (run1AsyncIn defaultContext f)

-- | As 'run1', but execute in the specified context.
--
run1In :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> a -> b
run1In ctx f = let go = run1AsyncIn ctx f
               in \a -> unsafePerformIO $ wait (go a)

-- | As 'run1In', but execute asynchronously.
--
run1AsyncIn :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> a -> Async b
run1AsyncIn ctx f = \a -> unsafePerformIO $ async (execute a)
  where
    !acc      = convertAccFun1With config f
    !afun     = unsafePerformIO $ evalCUDA ctx (compileAfun acc)
    execute a = evalCUDA ctx (executeAfun1 afun a >>= collect)
                `catch`
                \e -> INTERNAL_ERROR(error) "unhandled" (show (e :: CUDAException))

-- TLM: We need to be very careful with run1* variants, to ensure that the
--      returned closure shortcuts directly to the execution phase.


-- | Stream a lazily read list of input arrays through the given program,
--   collecting results as we go.
--
stream :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> [a] -> [b]
stream f arrs
  = unsafePerformIO
  $ evaluate (streamIn defaultContext f arrs)

-- | As 'stream', but execute in the specified context.
--
streamIn :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> [a] -> [b]
streamIn ctx f arrs
  = let go = run1In ctx f
    in  map go arrs


-- Copy arrays from device to host.
--
collect :: forall arrs. Arrays arrs => arrs -> CIO arrs
collect !arrs = toArr <$> collectR (arrays (undefined :: arrs)) (fromArr arrs)
  where
    collectR :: ArraysR a -> a -> CIO a
    collectR ArraysRunit         ()             = return ()
    collectR ArraysRarray        arr            = peekArray arr >> return arr
    collectR (ArraysRpair r1 r2) (arrs1, arrs2) = (,) <$> collectR r1 arrs1
                                                      <*> collectR r2 arrs2


-- How the Accelerate program should be interpreted.
-- TODO: make sharing/fusion runtime configurable via debug flags or otherwise.
--
config :: Phase
config =  Phase
  { recoverAccSharing      = True
  , recoverExpSharing      = True
  , floatOutAccFromExp     = True
  , enableAccFusion        = True
  , convertOffsetOfSegment = True
  }


-- Running asynchronously
-- ----------------------

-- We need to execute the main thread asynchronously to give finalisers a chance
-- to run. Make sure to catch exceptions to avoid "blocked indefinitely on MVar"
-- errors.
--
data Async a = Async {-# UNPACK #-} !ThreadId
                     {-# UNPACK #-} !(MVar (Either SomeException a))

-- Fork an action to execute asynchronously.
--
-- TLM:
--   CUDA contexts are specific to the processor on which they were created. It
--   may be necessary to take this into account when forking accelerate
--   computations (forkOn or forkOS rather than forkIO), either by always
--   requiring a specific CPU, and/or having the driver API store the processor
--   ordinal when creating contexts.
--
async :: IO a -> IO (Async a)
async action = do
   var <- newEmptyMVar
   tid <- forkOS $ (putMVar var . Right =<< action)
                   `catch`
                   \e -> putMVar var (Left e)
   return (Async tid var)

-- | Block the calling thread until the computation completes, then return the
-- result.
--
{-# INLINE wait #-}
wait :: Async a -> IO a
wait (Async _ var) = either throwIO return =<< readMVar var

-- | Test whether the asynchronous computation has already completed. If so,
-- return the result, else 'Nothing'.
--
{-# INLINE poll #-}
poll :: Async a -> IO (Maybe a)
poll (Async _ var) =
  maybe (return Nothing) (either throwIO (return . Just)) =<< tryTakeMVar var

-- | Cancel a running asynchronous computation.
--
{-# INLINE cancel #-}
cancel :: Async a -> IO ()
cancel (Async tid _) = throwTo tid ThreadKilled

