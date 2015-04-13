{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE CPP               #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns      #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Execute.Stream
-- Copyright   : [2013..2014] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
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
import Data.Array.Accelerate.CUDA.Context                       ( Context(..) )
import Data.Array.Accelerate.CUDA.Execute.Event                 ( Event, EventTable )
import Data.Array.Accelerate.FullList                           ( FullList(..) )
import Data.Array.Accelerate.Lifetime                           ( Lifetime, withLifetime )
import qualified Data.Array.Accelerate.CUDA.Execute.Event       as Event
import qualified Data.Array.Accelerate.CUDA.Debug               as D
import qualified Data.Array.Accelerate.FullList                 as FL

-- libraries
import Control.Monad.Trans                                      ( MonadIO, liftIO )
import Control.Exception                                        ( bracket_ )
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, mkWeakMVar )
import System.Mem.Weak                                          ( Weak, deRefWeak )
import Foreign.CUDA.Driver.Stream                               ( Stream(..) )
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Foreign.CUDA.Driver.Stream                     as Stream

import qualified Data.HashTable.IO                              as HT


-- Representation
-- --------------

-- The Reservoir is a place to store CUDA execution streams that are currently
-- inactive. When a new stream is requested one is provided from the reservoir
-- if available, otherwise a fresh execution stream is created.
--
type HashTable key val  = HT.BasicHashTable key val

type RSV                = MVar ( HashTable (Lifetime CUDA.Context) (FullList () Stream) )
data Reservoir          = Reservoir {-# UNPACK #-} !RSV
                                    {-# UNPACK #-} !(Weak RSV)


-- Executing operations in streams
-- -------------------------------

-- Execute an operation in a unique execution stream. The (asynchronous) result
-- is passed to a second operation together with an event that will be signalled
-- once the operation is complete. The stream and event are released after the
-- second operation completes.
--
{-# INLINE streaming #-}
streaming :: MonadIO m => Context -> Reservoir -> EventTable -> (Stream -> m a) -> (Event -> a -> m b) -> m b
streaming !ctx !rsv@(Reservoir !_ !weak_rsv) !etbl !action !after = do
  stream <- liftIO $ create ctx rsv
  first  <- action stream
  end    <- liftIO $ Event.waypoint ctx etbl stream
  final  <- after end first
  liftIO $! destroy (weakContext ctx) weak_rsv stream
  liftIO $! Event.destroy end
  return final


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
create !ctx (Reservoir !ref !_) = withMVar ref $ \tbl -> do
  --
  let key = deviceContext ctx
  ms    <- HT.lookup tbl key
  case ms of
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


-- Merge a stream back into the reservoir. This must only be done once all
-- pending operations in the stream have completed.
--
{-# INLINE destroy #-}
destroy :: Weak (Lifetime CUDA.Context) -> Weak RSV -> Stream -> IO ()
destroy !weak_ctx !weak_rsv !stream = do
  -- Wait for all preceding operations submitted to the stream to complete. Not
  -- necessary because of the setup of 'streaming'.
  -- Stream.block stream

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
        Nothing   -> trace ("destroy/free " ++ showStream stream) $ Stream.destroy stream
        Just ref  -> trace ("destroy/save " ++ showStream stream) $ withMVar ref $ \tbl -> do
          --
          ms <- HT.lookup tbl ctx
          case ms of
            Nothing       -> HT.insert tbl ctx (FL.singleton () stream)
            Just ss       -> HT.insert tbl ctx (FL.cons () stream ss)


-- Add a finaliser to an execution stream
--
-- addStreamFinalizer :: Stream -> IO () -> IO ()
-- addStreamFinalizer st@(Stream (Ptr st#)) f = IO $ \s ->
--   case mkWeak# st# st f s of (# s', _w #) -> (# s', () #)


-- Destroy all streams in the reservoir.
--
flush :: HashTable (Lifetime CUDA.Context) (FullList () Stream) -> IO ()
flush !tbl =
  let clean (!lctx,!ss) = do
        withLifetime lctx $ \ctx -> bracket_ (CUDA.push ctx) CUDA.pop (FL.mapM_ (const Stream.destroy) ss)
        HT.delete tbl lctx
  in
  message "flush reservoir" >> HT.mapM_ clean tbl


-- Debug
-- -----

{-# INLINE trace #-}
trace :: String -> IO a -> IO a
trace msg next = message msg >> next

{-# INLINE message #-}
message :: String -> IO ()
message msg = D.traceIO D.dump_sched ("stream: " ++ msg)

{-# INLINE showStream #-}
showStream :: Stream -> String
showStream (Stream s) = show s

