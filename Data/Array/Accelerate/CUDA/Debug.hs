{-# LANGUAGE TemplateHaskell, TypeOperators #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Debug
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--
-- Hijack some command line arguments to pass runtime debugging options. This
-- might cause problems for users of the library...
--

module Data.Array.Accelerate.CUDA.Debug (

  debug, when, mode,
  dump_cuda, dump_gc, dump_exec, verbose

) where

import Data.List
import Data.Label
import Data.IORef
import Debug.Trace                              ( putTraceMsg )
import Control.Monad.IO.Class
import System.IO.Unsafe
import System.Environment
import System.Console.GetOpt


-- -----------------------------------------------------------------------------
-- Internals

data Flags = Flags
  { _dump_cuda  :: Bool
  , _dump_gc    :: Bool
  , _dump_exec  :: Bool
  , _verbose    :: Bool
  }

$(mkLabels [''Flags])

flags :: [OptDescr (Flags -> Flags)]
flags =
  [ Option [] ["ddump-cuda"]    (NoArg (set dump_cuda True))    "print generated CUDA code"
  , Option [] ["ddump-gc"]      (NoArg (set dump_gc True))      "print device memory management trace"
  , Option [] ["ddump-exec"]    (NoArg (set dump_exec True))    "print kernel execution trace"
  , Option [] ["dverbose"]      (NoArg (set verbose True))      "print additional information"
  ]

initialise :: IO Flags
initialise = parse `fmap` getArgs
  where
    defaults      = Flags False False False False
    parse         = foldl parse1 defaults
    parse1 opts x = case filter (\(Option _ [f] _ _) -> x `isPrefixOf` ('-':f)) flags of
                      [Option _ _ (NoArg go) _] -> go opts
                      _                         -> opts         -- not specified, or ambiguous

{-# NOINLINE options #-}
options :: IORef Flags
options = unsafePerformIO $ newIORef =<< initialise

{-# INLINE mode #-}
mode :: (Flags :-> Bool) -> Bool
mode f = unsafePerformIO $ get f `fmap` readIORef options

{-# INLINE debug #-}
debug :: MonadIO m => (Flags :-> Bool) -> String -> m ()
debug f str = when f (liftIO $ putTraceMsg str)

{-# INLINE when #-}
when :: MonadIO m => (Flags :-> Bool) -> m () -> m ()
when f action
  | mode f      = action
  | otherwise   = return ()
