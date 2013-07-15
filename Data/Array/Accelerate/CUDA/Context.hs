{-# LANGUAGE MagicHash     #-}
{-# LANGUAGE UnboxedTuples #-}
{-# LANGUAGE ViewPatterns  #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Context
-- Copyright   : [2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- This module defines the execution context of an Accelerate computation in the
-- CUDA backend
--

module Data.Array.Accelerate.CUDA.Context (

  -- An execution context
  Context(..), create, push, pop, destroy,
  keepAlive, fromDeviceContext

) where

-- friends
import Data.Array.Accelerate.CUDA.Debug                 ( message, verbose, dump_gc, showFFloatSIBase )
import Data.Array.Accelerate.CUDA.Analysis.Device

-- system
import Data.Function                                    ( on )
import Control.Exception                                ( bracket_ )
import Control.Concurrent                               ( forkIO, threadDelay )
import Control.Monad                                    ( when )
import GHC.Exts                                         ( Ptr(..), mkWeak# )
import GHC.Base                                         ( IO(..) )
import GHC.Weak                                         ( Weak(..) )
import Text.PrettyPrint
import qualified Foreign.CUDA.Driver                    as CUDA hiding ( device )
import qualified Foreign.CUDA.Driver.Context            as CUDA


-- | The execution context
--
data Context = Context {
    deviceProperties    :: {-# UNPACK #-} !CUDA.DeviceProperties,       -- information on hardware resources
    deviceContext       :: {-# UNPACK #-} !CUDA.Context,                -- device execution context
    weakContext         :: {-# UNPACK #-} !(Weak CUDA.Context)          -- weak pointer to the context (for memory management)
  }

instance Eq Context where
  (==) = (==) `on` deviceContext


-- | Create a new CUDA context associated with the calling thread
--
create :: CUDA.Device -> [CUDA.ContextFlag] -> IO Context
create dev flags = do
  ctx                    <- CUDA.create dev flags >> CUDA.pop >>= keepAlive
  actx@(Context prp _ _) <- fromDeviceContext dev ctx

  -- Generated code does not take particular advantage of shared memory, so
  -- for devices that support it use those banks as an L1 cache instead.
  --
  -- TODO: Perhaps make this a command line switch: -fprefer-[l1,shared]
  -- TODO: Make the occupancy calculator aware of adjustable shared memory
  --
  when (CUDA.computeCapability prp >= CUDA.Compute 2 0)
     $ bracket_ (CUDA.push ctx) CUDA.pop (CUDA.setCacheConfig CUDA.PreferL1)

  message verbose (deviceInfo dev prp)
  return actx

-- |Given a device context, construct a new context around it.
fromDeviceContext :: CUDA.Device -> CUDA.Context -> IO Context
fromDeviceContext dev ctx = do
  prp           <- CUDA.props dev
  weak          <- mkWeakContext ctx $ do
    message dump_gc $ "gc: finalise context #" ++ show (CUDA.useContext ctx)
    CUDA.destroy ctx
  message dump_gc $ "gc: initialise context #" ++ show (CUDA.useContext ctx)

  return $! Context prp ctx weak

-- | Destroy the specified context. This will fail if the context is more than
-- single attachment.
--
{-# INLINE destroy #-}
destroy :: Context -> IO ()
destroy (deviceContext -> ctx) = do
  message dump_gc ("gc: destroy context: #" ++ show (CUDA.useContext ctx))
  CUDA.destroy ctx


-- | Push the given context onto the CPU's thread stack of current contexts. The
-- context must be floating (via 'pop'), i.e. not attached to any thread.
--
{-# INLINE push #-}
push :: Context -> IO ()
push (deviceContext -> ctx) = do
  message dump_gc ("gc: push context: #" ++ show (CUDA.useContext ctx))
  CUDA.push ctx


-- | Pop the current context.
--
{-# INLINE pop #-}
pop :: IO ()
pop = do
  ctx <- CUDA.pop
  message dump_gc ("gc: pop context: #" ++ show (CUDA.useContext ctx))


-- Make a weak pointer to a CUDA context. We need to be careful to put the
-- finaliser on the underlying pointer, rather than the box around it as
-- 'mkWeak' will do, because unpacking the context will cause the finaliser to
-- fire prematurely.
--
mkWeakContext :: CUDA.Context -> IO () -> IO (Weak CUDA.Context)
mkWeakContext c@(CUDA.Context (Ptr c#)) f = IO $ \s ->
  case mkWeak# c# c f s of (# s', w #) -> (# s', Weak w #)


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

