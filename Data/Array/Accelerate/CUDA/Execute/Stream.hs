{-# LANGUAGE BangPatterns  #-}
{-# LANGUAGE MagicHash     #-}
{-# LANGUAGE UnboxedTuples #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Execute.Stream
-- Copyright   : [2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Execute.Stream (

  Stream, Reservoir, new, streaming,

) where

-- friends
import Data.Array.Accelerate.CUDA.Array.Nursery                 ( ) -- hashable CUDA.Context instance
import Data.Array.Accelerate.CUDA.Context                       ( Context(..) )
import Data.Array.Accelerate.CUDA.FullList                      ( FullList(..) )
import Data.Array.Accelerate.CUDA.Execute.Event                 ( Event )
import qualified Data.Array.Accelerate.CUDA.Execute.Event       as Event
import qualified Data.Array.Accelerate.CUDA.FullList            as FL
import qualified Data.Array.Accelerate.CUDA.Debug               as D

-- libraries
import Control.Monad.Trans                                      ( MonadIO, liftIO )
import Control.Exception                                        ( bracket_ )
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, mkWeakMVar )
import System.Mem.Weak                                          ( Weak, deRefWeak )
import Foreign.CUDA.Driver.Stream                               ( Stream(..) )
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Foreign.CUDA.Driver.Stream                     as Stream

import qualified Data.HashTable.IO                              as HT

import GHC.Base
import GHC.Ptr


-- Representation
-- --------------

-- The Reservoir is a place to store CUDA execution streams that are currently
-- inactive. When a new stream is requested one is provided from the reservoir
-- if available, otherwise a fresh execution stream is created.
--
type HashTable key val  = HT.BasicHashTable key val

type RSV                = MVar ( HashTable CUDA.Context (FullList () Stream) )
data Reservoir          = Reservoir {-# UNPACK #-} !RSV
                                    {-# UNPACK #-} !(Weak RSV)


-- Executing operations in streams
-- -------------------------------

-- Execute an operation in a unique execution stream.
--
{-# INLINE streaming #-}
streaming :: MonadIO m => Context -> Reservoir -> (Stream -> m a) -> m (Event, a)
streaming !ctx !rsv !action = do
  stream <- liftIO $ create ctx rsv
  result <- action stream
  end    <- liftIO $ Event.waypoint stream
  return (end, result)


-- Primitive operations
-- --------------------

-- Generate a new empty reservoir. It is not necessary to pre-populate it with
-- any streams because stream creation does not cause a device synchronisation.
--
new :: IO Reservoir
new = do
  tbl    <- HT.new
  ref    <- newMVar tbl
  weak   <- mkWeakMVar ref (flush tbl)
  return $! Reservoir ref weak


-- Create a CUDA execution stream. If an inactive stream is available for use,
-- that is returned, else a fresh stream is created.
--
{-# INLINE create #-}
create :: Context -> Reservoir -> IO Stream
create !ctx (Reservoir !ref !weak_rsv) = withMVar ref $ \tbl -> do
  let key = deviceContext ctx
  --
  old    <- HT.lookup tbl key
  stream <- case old of
    Nothing -> do
      stream <- Stream.create []
      message ("new " ++ showStream stream)
      return stream

    Just (FL () stream rest) -> do
      case rest of
        FL.Nil          -> HT.delete tbl key
        FL.Cons () s ss -> HT.insert tbl key (FL () s ss)
        --
      return stream
  --
  addStreamFinalizer stream $ merge (weakContext ctx) weak_rsv stream
  return stream


-- Merge a stream back into the reservoir. This is done asynchronously, once all
-- pending operations in the stream have completed.
--
{-# INLINE merge #-}
merge :: Weak CUDA.Context -> Weak RSV -> Stream -> IO ()
merge !weak_ctx !weak_rsv !stream = do
  -- wait for all preceding operations submitted to the stream to complete
  Stream.block stream

  -- Now check whether the context and reservoir are still active. Return
  -- the stream back to the reservoir for later reuse if we can, otherwise
  -- destroy it.
  mc <- deRefWeak weak_ctx
  case mc of
    Nothing       -> message ("finalise/dead context " ++ showStream stream)
    Just ctx      -> do
      --
      mr <- deRefWeak weak_rsv
      case mr of
        Nothing   -> trace ("merge/free " ++ showStream stream) $ Stream.destroy stream
        Just ref  -> trace ("merge/save " ++ showStream stream) $ withMVar ref $ \tbl -> do
          --
          ms <- HT.lookup tbl ctx
          case ms of
            Nothing       -> HT.insert tbl ctx (FL.singleton () stream)
            Just ss       -> HT.insert tbl ctx (FL.cons () stream ss)


-- Add a finaliser to an execution stream
--
addStreamFinalizer :: Stream -> IO () -> IO ()
addStreamFinalizer st@(Stream (Ptr st#)) f = IO $ \s ->
  case mkWeak# st# st f s of (# s', _w #) -> (# s', () #)


-- Destroy all streams in the reservoir.
--
flush :: HashTable CUDA.Context (FullList () Stream) -> IO ()
flush !tbl =
  let clean (!ctx,!ss) = do
        bracket_ (CUDA.push ctx) CUDA.pop (FL.mapM_ (const Stream.destroy) ss)
        HT.delete tbl ctx
  in
  message "flush reservoir" >> HT.mapM_ clean tbl


-- Debug
-- -----

{-# INLINE trace #-}
trace :: String -> IO a -> IO a
trace msg next = D.message D.dump_exec ("stream: " ++ msg) >> next

{-# INLINE message #-}
message :: String -> IO ()
message s = s `trace` return ()

{-# INLINE showStream #-}
showStream :: Stream -> String
showStream (Stream.Stream s) = show s

