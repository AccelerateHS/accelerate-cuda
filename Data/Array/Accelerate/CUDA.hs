{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2008..2009] Sean Lee
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- This module implements the CUDA backend for the embedded array language
-- /Accelerate/. Expressions are on-line translated into CUDA code, compiled,
-- and executed in parallel on the GPU.
--
-- The accelerate-cuda library is hosted at: <https://github.com/AccelerateHS/accelerate-cuda>.
-- Comments, bug reports, and patches, are always welcome.
--
--
-- [/Data transfer:/]
--
-- GPUs typically have their own attached memory, which is separate from the
-- computer's main memory. Hence, every 'Data.Array.Accelerate.use' operation
-- implies copying data to the device, and every 'run' operation must copy the
-- results of a computation back to the host.
--
-- Thus, it is best to keep all computations in the 'Acc' meta-language form and
-- only 'run' the computation once at the end, to avoid transferring (unused)
-- intermediate results.
--
-- Note that once an array has been transferred to the GPU, it will remain there
-- for as long as that array remains alive on the host. Any subsequent calls to
-- 'Data.Array.Accelerate.use' will find the array cached on the device and not
-- re-transfer the data.
--
--
-- [/Caching and performance:/]
--
-- When the program runs, the /Accelerate/ library evaluates the expression
-- passed to 'run' to make a series of CUDA kernels. Each kernel takes some
-- arrays as inputs and produces arrays as output. Each kernel is a piece of
-- CUDA code that has to be compiled and loaded onto the GPU; this can take a
-- while, so we remember which kernels we have seen before and try to re-use
-- them.
--
-- The goal is to make kernels that can be re-used. If we don't, the overhead of
-- compiling new kernels can ruin performance.
--
-- For example, consider the following implementation of the function
-- 'Data.Array.Accelerate.drop' for vectors:
--
-- > drop :: Elt e => Exp Int -> Acc (Vector e) -> Acc (Vector e)
-- > drop n arr =
-- >   let n' = the (unit n)
-- >   in  backpermute (ilift1 (subtract n') (shape arr)) (ilift1 (+ n')) arr
--
-- Why did we go to the trouble of converting the @n@ value into a scalar array
-- using 'Data.Array.Accelerate.unit', and then immediately extracting that
-- value using 'Data.Array.Accelerate.the'?
--
-- We can look at the expression /Accelerate/ sees by evaluating the argument to
-- 'run'. Here is what a typical call to 'Data.Array.Accelerate.drop' evaluates
-- to:
--
-- >>> drop (constant 4) (use (fromList (Z:.10) [1..]))
-- let a0 = use (Array (Z :. 10) [1,2,3,4,5,6,7,8,9,10]) in
-- let a1 = unit 4
-- in backpermute
--      (let x0 = Z in x0 :. (indexHead (shape a0)) - (a1!x0))
--      (\x0 -> let x1 = Z in x1 :. (indexHead x0) + (a1!x1))
--      a0
--
-- The important thing to note is the line @let a1 = unit 4@. This corresponds
-- to the scalar array we created for the @n@ argument to
-- 'Data.Array.Accelerate.drop' and it is /outside/ the call to
-- 'Data.Array.Accelerate.backpermute'. The 'Data.Array.Accelerate.backpermute'
-- function is what turns into a CUDA kernel, and to ensure that we get the same
-- kernel each time we need the arguments to it to remain constant.
--
-- Let us see what happens if we change 'Data.Array.Accelerate.drop' to instead
-- use its argument @n@ directly:
--
-- >>> drop (constant 4) (use (fromList (Z:.10) [1..]))
-- let a0 = use (Array (Z :. 10) [1,2,3,4,5,6,7,8,9,10])
-- in backpermute (Z :. -4 + (indexHead (shape a0))) (\x0 -> Z :. 4 + (indexHead x0)) a0
--
-- Instead of @n@ being outside the call to 'Data.Array.Accelerate.backpermute',
-- it is now embedded in it. This will defeat /Accelerate/'s caching of CUDA
-- kernels. Whenever the value of @n@ changes, a new kernel will need to be
-- compiled.
--
-- The rule of thumb is to make sure that any arguments that change are always
-- passed in as arrays, not embedded in the code as constants.
--
-- How can you tell if you got it wrong? One way is to look at the code
-- directly, as in this example. Another is to use the debugging options
-- provided by the library. See debugging options below.
--
--
-- [/Hardware support:/]
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
-- In particular:
--
--  * 'Double' precision requires compute-1.3.
--
--  * 'Bool' is represented internally using 'Data.Word.Word8', 'Char' by
--    'Data.Word.Word32'.
--
--  * If the permutation function to 'Data.Array.Accelerate.permute' resolves to
--    non-unique indices, the combination function requires compute-1.1.
--
--
-- [/Debugging options:/]
--
-- When the library is installed with the @-fdebug@ flag, a few extra debugging
-- options are available, input via the command line arguments. The most useful
-- ones are:
--
--  * @-dverbose:@ Print some information on the type and capabilities of the
--    GPU being used.
--
--  * @-ddump-cc:@ Print information about the CUDA kernels as they are compiled
--    and run. Using this option will indicate whether your program is
--    generating the number of kernels that you were expecting. Note that
--    compiled kernels are cached in your home directory, and the generated code
--    will only be displayed if it was not located in this persistent cache. To
--    clear the cache and always print the generated code, use @-fflush-cache@
--    as well.
--
--  * @-ddump-exec:@ Print each kernel as it is being executed, with timing
--    information.
--
-- See the @accelerate-cuda.cabal@ file for the full list of options.
--
--
-- [/Automatic Graphics Switching on Mac OS X:/]
--
-- Some Apple computers contain two graphics processors: a low-power integrated
-- graphics chipset, as well as a higher-performance NVIDIA GPU. The latter is
-- of course the one we want to use. Usually Mac OS X detects whenever a program
-- attempts to run a CUDA function and switches to the NVIDIA GPU automatically.
--
-- However, sometimes this does not work correctly and the problem can manifest
-- in several ways:
--
--  * The program may report an error such as \"No CUDA-capable device is
--    available\" or \"invalid context handle\".
--
--  * For programs that also use OpenGL, the graphics switching might occur and
--    the Accelerate computation complete as expected, but no OpenGL updates
--    appear on screen.
--
-- There are several solutions:
--
--  * Use a tool such as /gfxCardStatus/ to manually select either the
--    integrated or discrete GPU: <http://gfx.io>
--
--  * Disable automatic graphics switching in the Energy Saver pane of System
--    Preferences. Since this disables use of the low-power integrated GPU, this
--    can decrease battery life.
--
--  * When executing the program, disable the RTS clock by appending @+RTS -V0@
--    to the command line arguments. This disables the RTS clock and all timers
--    that depend on it: the context switch timer and the heap profiling timer.
--    Context switches still happen, but deterministically and at a rate much
--    faster than normal. Automatic graphics switching will work correctly, but
--    this method has the disadvantage of reducing performance of the program.
--

module Data.Array.Accelerate.CUDA (

  Arrays,

  -- * Synchronous execution
  run, run1, runWith, run1With,
  stream, streamOut, streamWith, streamOutWith,

  -- * Asynchronous execution
  Async, wait, poll, cancel,
  runAsync, run1Async, runAsyncWith, run1AsyncWith,

  -- * Execution contexts
  Context, create, destroy,
  unsafeFree, unsafeFreeWith, performGC, performGCWith,

) where

-- standard library
import Control.Exception
import Control.Applicative
import Control.Monad.Trans
import System.IO.Unsafe
import Prelude

-- friends
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate.Smart                      ( Acc, Seq )
import Data.Array.Accelerate                            ( mapSeq, streamIn )
import Data.Array.Accelerate.Array.Sugar                ( Arrays(..), ArraysR(..) )
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.Async
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Compile
import Data.Array.Accelerate.CUDA.Execute

#if ACCELERATE_DEBUG
import Data.Array.Accelerate.Debug
#endif


-- Accelerate: CUDA
-- ----------------

-- | Compile and run a complete embedded array program using the CUDA backend.
-- This will select the fastest device available on which to execute
-- computations, based on compute capability and estimated maximum GFLOPS.
--
-- Note that it is recommended you use 'run1' whenever possible.
--
run :: Arrays a => Acc a -> a
run a
  = unsafePerformIO
  $ evaluate (runWith defaultContext a)

-- | As 'run', but allow the computation to continue running in a thread and
-- return immediately without waiting for the result. The status of the
-- computation can be queried using 'wait', 'poll', and 'cancel'.
--
-- Note that a CUDA Context can be active on only one host thread at a time. If
-- you want to execute multiple computations in parallel, use 'runAsyncWith'.
--
runAsync :: Arrays a => Acc a -> Async a
runAsync a
  = unsafePerformIO
  $ evaluate (runAsyncWith defaultContext a)

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
-- it to 'runWith', which will make it current for the duration of evaluating the
-- expression. See the CUDA C Programming Guide (G.1) for more information.
--
runWith :: Arrays a => Context -> Acc a -> a
runWith ctx a
  = unsafePerformIO
  $ evaluate (runAsyncWith ctx a) >>= wait


-- | As 'runWith', but execute asynchronously. Be sure not to destroy the context,
-- or attempt to attach it to a different host thread, before all outstanding
-- operations have completed.
--
runAsyncWith :: Arrays a => Context -> Acc a -> Async a
runAsyncWith ctx a = unsafePerformIO $ async execute
  where
    !acc    = convertAccWith config a
    execute = evalCUDA ctx (compileAcc acc >>= dumpStats >>= executeAcc >>= collect)


-- | Prepare and execute an embedded array program of one argument.
--
-- This function can be used to improve performance in cases where the array
-- program is constant between invocations, because it allows us to bypass all
-- front-end conversion stages and move directly to the execution phase. If you
-- have a computation applied repeatedly to different input data, use this. If
-- the function is only evaluated once, this is equivalent to 'run'.
--
-- To use 'run1' you must express your program as a function of one argument. If
-- your program takes more than one argument, you can use
-- 'Data.Array.Accelerate.lift' and 'Data.Array.Accelerate.unlift' to tuple up
-- the arguments.
--
-- At an example, once your program is expressed as a function of one argument,
-- instead of the usual:
--
-- > step :: Acc (Vector a) -> Acc (Vector b)
-- > step = ...
-- >
-- > simulate :: Vector a -> Vector b
-- > simulate xs = run $ step (use xs)
--
-- Instead write:
--
-- > simulate xs = run1 step xs
--
-- You can use the debugging options to check whether this is working
-- successfully by, for example, observing no output from the @-ddump-cc@ flag
-- at the second and subsequent invocations.
--
-- See the programs in the 'accelerate-examples' package for examples.
--
run1 :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> a -> b
run1 f
  = unsafePerformIO
  $ evaluate (run1With defaultContext f)


-- | As 'run1', but the computation is executed asynchronously.
--
run1Async :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> a -> Async b
run1Async f
  = unsafePerformIO
  $ evaluate (run1AsyncWith defaultContext f)

-- | As 'run1', but execute in the specified context.
--
run1With :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> a -> b
run1With ctx f = let go = run1AsyncWith ctx f
                 in \a -> unsafePerformIO $ wait (go a)

-- | As 'run1With', but execute asynchronously.
--
run1AsyncWith :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> a -> Async b
run1AsyncWith ctx f = \a -> unsafePerformIO $ async (execute a)
  where
    !acc      = convertAfunWith config f
    !afun     = unsafePerformIO $ evalCUDA ctx (compileAfun acc) >>= dumpStats
    execute a = evalCUDA ctx (executeAfun1 afun a >>= collect)

-- TLM: We need to be very careful with run1* variants, to ensure that the
--      returned closure shortcuts directly to the execution phase.


-- | Stream a lazily read list of input arrays through the given program,
--   collecting results as we go.
--
stream :: (Arrays a, Arrays b) => (Acc a -> Acc b) -> [a] -> [b]
stream f arrs
  = unsafePerformIO
  $ evaluate (streamWith defaultContext f arrs)

-- | As 'stream', but execute in the specified context.
--
streamWith :: (Arrays a, Arrays b) => Context -> (Acc a -> Acc b) -> [a] -> [b]
streamWith ctx f
  = streamOutWith ctx . mapSeq f . streamIn

-- | Generate a lazy list from a sequence computation.
--
streamOut :: Arrays a => Seq [a] -> [a]
streamOut = streamOutWith defaultContext

streamOutWith :: forall a. Arrays a => Context -> Seq [a] -> [a]
streamOutWith ctx = exec . compile . convertSeq
  where
    compile     = unsafePerformIO . evalCUDA ctx . compileSeq
    exec s      = go (streamSeq s)
      where
        go !s' = case step s' of
          Nothing       -> []
          Just (a, s'') -> a : go s''

        step (StreamSeq ss)
          = unsafePerformIO
          $ evalCUDA ctx
          $ do m <- ss
               case m of
                 Nothing      -> return Nothing
                 Just (a, s') -> collect a >> return (Just (a, s'))


-- RCE: Similar to run1* variants, we need to be ultra careful with streamOut*
-- in order to make sure that the entire sequence is not reified at once.
-- The steps of the sequence computation should only be performed as needed
-- when elements of the list are forced.

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
  , recoverSeqSharing      = True
  , floatOutAccFromExp     = True
  , enableAccFusion        = True
  , convertOffsetOfSegment = True
  , vectoriseSequences     = False
  }


dumpStats :: MonadIO m => a -> m a
#if ACCELERATE_DEBUG
dumpStats next = liftIO $ do
  stats <- simplCount
  traceIO dump_simpl_stats (show stats)
  resetSimplCount
  return next
#else
dumpStats next = return next
#endif


-- Device memory management
-- ------------------------
--
-- Temporarily defining here, until we can define the interface for it.


-- Deallocate the device arrays corresponding to the given host side arrays.
-- This is unsafe in the sense that it is possible to call this function while
-- the array is currently in use.
--
unsafeFree :: Arrays arrs => arrs -> IO ()
unsafeFree = unsafeFreeWith defaultContext

unsafeFreeWith :: forall arrs. Arrays arrs => Context -> arrs -> IO ()
unsafeFreeWith !ctx !arrs
  = evalCUDA ctx
  $ freeR (arrays (undefined :: arrs)) (fromArr arrs)
  where
    freeR :: ArraysR a -> a -> CIO ()
    freeR ArraysRunit             ()             = return ()
    freeR ArraysRarray            arr            = freeArray arr
    freeR (ArraysRpair aeR1 aeR2) (arrs1, arrs2) = freeR aeR1 arrs1 >> freeR aeR2 arrs2


-- Release any unused device memory
--
performGC :: IO ()
performGC = performGCWith defaultContext

performGCWith :: Context -> IO ()
performGCWith !ctx = evalCUDA ctx cleanupArrayData

