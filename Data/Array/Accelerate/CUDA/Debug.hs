{-# LANGUAGE CPP             #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeOperators   #-}
{-# OPTIONS -fno-warn-incomplete-patterns #-}
{-# OPTIONS -fno-warn-unused-binds        #-}
{-# OPTIONS -fno-warn-unused-imports      #-}
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

  showFFloatSIBase,

  message, trace, event, when, unless, mode, timed, elapsed,
  verbose, flush_cache,
  dump_gc, dump_cc, debug_cc, dump_exec,

) where

import Numeric
import Data.List
import Data.Label
import Data.IORef
import Debug.Trace                                      ( traceIO, traceEventIO )
import Control.Monad                                    ( void )
import Control.Monad.IO.Class                           ( liftIO, MonadIO )
import Control.Concurrent                               ( forkIO )
import System.CPUTime
import System.IO.Unsafe
import System.Environment
import System.Console.GetOpt
import Foreign.CUDA.Driver.Stream                       ( Stream )
import qualified Foreign.CUDA.Driver.Event              as Event

import GHC.Float


-- -----------------------------------------------------------------------------
-- Pretty-printing

showFFloatSIBase :: RealFloat a => Maybe Int -> a -> a -> ShowS
showFFloatSIBase p b n
  = showString
  $ showFFloat p n' (' ':si_unit)
  where
    n'          = n / (b ^^ pow)
    pow         = (-4) `max` floor (logBase b n) `min` 4        :: Int
    si_unit     = case pow of
                       -4 -> "p"
                       -3 -> "n"
                       -2 -> "µ"
                       -1 -> "m"
                       0  -> ""
                       1  -> "k"
                       2  -> "M"
                       3  -> "G"
                       4  -> "T"


-- -----------------------------------------------------------------------------
-- Internals

data Flags = Flags
  {
    -- debugging
    _dump_gc            :: !Bool        -- garbage collection & memory management
  , _dump_cc            :: !Bool        -- compilation & linking
  , _debug_cc           :: !Bool        -- compile device code with debug symbols
  , _dump_exec          :: !Bool        -- kernel execution
  , _verbose            :: !Bool        -- additional status messages

    -- general options / functionality
  , _flush_cache        :: !Bool        -- delete the persistent cache directory
  , _fast_math          :: !Bool        -- use faster, less accurate maths library operations
  }

$(mkLabels [''Flags])

allFlags :: [OptDescr (Flags -> Flags)]
allFlags =
  [
    -- debugging
    Option [] ["ddump-gc"]      (NoArg (set dump_gc True))      "print device memory management trace"
  , Option [] ["ddump-cc"]      (NoArg (set dump_cc True))      "print generated code and compilation information"
  , Option [] ["ddebug-cc"]     (NoArg (set debug_cc True))     "generate debug information for device code"
  , Option [] ["ddump-exec"]    (NoArg (set dump_exec True))    "print kernel execution trace"
  , Option [] ["dverbose"]      (NoArg (set verbose True))      "print additional information"

    -- functionality / optimisation
  , Option [] ["fflush-cache"]  (NoArg (set flush_cache True))  "delete the persistent cache directory"
  , Option [] ["ffast-math"]    (NoArg (set fast_math True))    "use faster, less accurate maths library operations"
  ]

initialise :: IO Flags
initialise = parse `fmap` getArgs
  where
    defaults      = Flags False False False False False False False
    parse         = foldl parse1 defaults
    parse1 opts x = case filter (\(Option _ [f] _ _) -> x `isPrefixOf` ('-':f)) allFlags of
                      [Option _ _ (NoArg go) _] -> go opts
                      _                         -> opts         -- not specified, or ambiguous

#ifdef ACCELERATE_DEBUG
{-# NOINLINE options #-}
options :: IORef Flags
options = unsafePerformIO $ newIORef =<< initialise
#endif

{-# INLINE mode #-}
mode :: (Flags :-> Bool) -> Bool
#ifdef ACCELERATE_DEBUG
mode f = unsafePerformIO $ get f `fmap` readIORef options
#else
mode _ = False
#endif

{-# INLINE message #-}
message :: MonadIO m => (Flags :-> Bool) -> String -> m ()
#ifdef ACCELERATE_DEBUG
message f str
  = when f . liftIO
  $ do psec     <- getCPUTime
       let sec   = fromIntegral psec * 1E-12 :: Double
       traceIO   $ showFFloat (Just 2) sec (':':str)
#else
message _ _   = return ()
#endif

{-# INLINE event #-}
event :: MonadIO m => (Flags :-> Bool) -> String -> m ()
#ifdef ACCELERATE_DEBUG
event f str = when f (liftIO $ traceEventIO str)
#else
event _ _   = return ()
#endif

{-# INLINE trace #-}
trace :: (Flags :-> Bool) -> String -> a -> a
#ifdef ACCELERATE_DEBUG
trace f str next = unsafePerformIO (message f str) `seq` next
#else
trace _ _   next = next
#endif


{-# INLINE when #-}
when :: MonadIO m => (Flags :-> Bool) -> m () -> m ()
#ifdef ACCELERATE_DEBUG
when f action
  | mode f      = action
  | otherwise   = return ()
#else
when _ _        = return ()
#endif

{-# INLINE unless #-}
unless :: MonadIO m => (Flags :-> Bool) -> m () -> m ()
#ifdef ACCELERATE_DEBUG
unless f action
  | mode f      = return ()
  | otherwise   = action
#else
unless _ action = action
#endif

{-# INLINE timed #-}
timed
    :: MonadIO m
    => (Flags :-> Bool)
    -> (Double -> Double -> String)
    -> Maybe Stream
    -> m ()
    -> m ()
timed _f _str _stream action
#ifdef ACCELERATE_DEBUG
  | mode _f
  = do
      gpuBegin  <- liftIO $ Event.create []
      gpuEnd    <- liftIO $ Event.create []
      cpuBegin  <- liftIO getCPUTime
      liftIO $ Event.record gpuBegin _stream
      action
      liftIO $ Event.record gpuEnd _stream
      cpuEnd    <- liftIO getCPUTime

      -- Wait for the GPU to finish executing then display the timing execution
      -- message. Do this in a separate thread so that the remaining kernels can
      -- be queued asynchronously.
      --
      _         <- liftIO . forkIO $ do
        Event.block gpuEnd
        diff    <- Event.elapsedTime gpuBegin gpuEnd
        let gpuTime = float2Double $ diff * 1E-3                         -- milliseconds
            cpuTime = fromIntegral (cpuEnd - cpuBegin) * 1E-12 :: Double -- picoseconds

        Event.destroy gpuBegin
        Event.destroy gpuEnd
        --
        message _f (_str gpuTime cpuTime)
      --
      return ()

  | otherwise
#endif
  = action

{-# INLINE elapsed #-}
elapsed :: Double -> Double -> String
elapsed gpuTime cpuTime
  = "gpu: " ++ showFFloatSIBase (Just 3) 1000 gpuTime "s, " ++
    "cpu: " ++ showFFloatSIBase (Just 3) 1000 cpuTime "s"

