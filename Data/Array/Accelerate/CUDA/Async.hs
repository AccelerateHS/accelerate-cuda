{-# LANGUAGE CPP #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Async
-- Copyright   : [2009..2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Async
  where

#if !MIN_VERSION_base(4,6,0)
import Prelude                                          hiding ( catch )
#endif
import Control.Exception
import Control.Concurrent


-- We need to execute the main thread asynchronously to give finalisers a chance
-- to run. Make sure to catch exceptions to avoid "blocked indefinitely on MVar"
-- errors.
--
data Async a = Async {-# UNPACK #-} !ThreadId
                     {-# UNPACK #-} !(MVar (Either SomeException a))

-- | Fork an action to execute asynchronously.
--
async :: IO a -> IO (Async a)
async action = do
   var <- newEmptyMVar
   tid <- forkIO $ (putMVar var . Right =<< action)
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

