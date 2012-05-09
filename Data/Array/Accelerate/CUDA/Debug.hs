{-# LANGUAGE CPP, TemplateHaskell, TypeOperators #-}
{-# OPTIONS -fno-warn-unused-imports #-}
{-# OPTIONS -fno-warn-unused-binds   #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Debug
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--
-- Hijack some command line arguments to pass runtime debugging options. This
-- might cause problems for users of the library...
--

module Data.Array.Accelerate.CUDA.Debug (

  showFFloatSIBase,

  message, event, when, mode,
  verbose, debug,
  dump_gc, dump_cc, dump_exec,

) where

import Numeric
import Data.List
import Data.Label
import Data.IORef
import Control.Monad.IO.Class
import System.IO.Unsafe
import System.Environment
import System.Console.GetOpt

#if MIN_VERSION_base(4,5,0)
import Debug.Trace                              ( traceIO, traceEventIO )
#else
import Debug.Trace                              ( putTraceMsg )

traceIO :: String -> IO ()
traceIO = putTraceMsg

traceEventIO :: String -> IO ()
traceEventIO = traceIO
#endif


-- -----------------------------------------------------------------------------
-- Pretty-printing

showFFloatSIBase :: RealFloat a => Maybe Int -> a -> a -> ShowS
showFFloatSIBase p b n
  = showString
  . nubBy (\x y -> x == ' ' && y == ' ')
  $ showFFloat p n' [ ' ', si_unit ]
  where
    n'          = n / (b ^^ (pow-4))
    pow         = max 0 . min 8 . (+) 4 . floor $ logBase b n
    si_unit     = "pnµm kMGT" !! pow


-- -----------------------------------------------------------------------------
-- Internals

data Flags = Flags
  {
    -- phase control
    _dump_gc    :: !Bool        -- garbage collection & memory management
  , _dump_cc    :: !Bool        -- compilation & linking
  , _dump_exec  :: !Bool        -- kernel execution

    -- general options
  , _verbose    :: !Bool        -- additional status messages
  , _debug      :: !Bool        -- generate device code suitable for debugging
  }

$(mkLabels [''Flags])

flags :: [OptDescr (Flags -> Flags)]
flags =
  [ Option [] ["ddump-gc"]      (NoArg (set dump_gc True))      "print device memory management trace"
  , Option [] ["ddump-cc"]      (NoArg (set dump_cc True))      "print generated code and compilation information"
  , Option [] ["ddump-exec"]    (NoArg (set dump_exec True))    "print kernel execution trace"
  , Option [] ["dverbose"]      (NoArg (set verbose True))      "print additional information"
  , Option [] ["ddebug"]        (NoArg (set debug True))        "generate debug information for device code"
  ]

initialise :: IO Flags
initialise = parse `fmap` getArgs
  where
    defaults      = Flags False False False False False
    parse         = foldl parse1 defaults
    parse1 opts x = case filter (\(Option _ [f] _ _) -> x `isPrefixOf` ('-':f)) flags of
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
message f str = when f (liftIO $ traceIO str)
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

{-# INLINE when #-}
when :: MonadIO m => (Flags :-> Bool) -> m () -> m ()
#ifdef ACCELERATE_DEBUG
when f action
  | mode f      = action
  | otherwise   = return ()
#else
when _ _        = return ()
#endif

