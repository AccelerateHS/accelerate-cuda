{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE CPP          #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Execute.Event
-- Copyright   : [2013..2014] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
module Data.Array.Accelerate.CUDA.Execute.Event (

  Event, EventTable, new, create, waypoint, after, block, query, destroy,

) where

-- friends
import Data.Array.Accelerate.FullList                           ( FullList(..) )
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.CUDA.Context                       ( Context(..) )
import qualified Data.Array.Accelerate.CUDA.Debug               as D
import qualified Data.Array.Accelerate.FullList                 as FL

-- libraries
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, mkWeakMVar )
import Control.Exception                                        ( bracket_ )
import Data.Hashable                                            ( Hashable(..) )
import Foreign.CUDA.Driver.Stream                               ( Stream(..) )
import Foreign.Ptr                                              ( ptrToIntPtr )
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Foreign.CUDA.Driver.Event                      as Event
import qualified Data.HashTable.IO                              as HT

type Event              = Lifetime Event.Event

type HashTable key val  = HT.BasicHashTable key val

type EventTable         = MVar ( HashTable (Lifetime CUDA.Context) (FullList () Event.Event) )

instance Hashable (Lifetime CUDA.Context) where
  {-# INLINE hashWithSalt #-}
  hashWithSalt salt (unsafeGetValue -> CUDA.Context ctx)
    = salt `hashWithSalt` (fromIntegral (ptrToIntPtr ctx) :: Int)


-- Generate a new empty event table.
--
new :: IO EventTable
new = do
  tbl    <- HT.new
  ref    <- newMVar tbl
  _      <- mkWeakMVar ref (flush tbl)
  return ref

-- Create a new event. It will be automatically garbage collected, if a recycled
-- event is available, it will be returned, else a new event is created.
--
{-# INLINE create #-}
create :: Context -> EventTable -> IO Event
create ctx ref = withMVar ref $ \tbl -> do
  --
  let key = deviceContext ctx
  me <- HT.lookup tbl key
  e  <- case me of
    Nothing -> do
      e <- Event.create [Event.DisableTiming]
      message ("new " ++ show e)
      return e

    Just (FL () e rest) -> do
      case rest of
        FL.Nil           -> HT.delete tbl key
        FL.Cons () e' es -> HT.insert tbl key (FL () e' es)
        --
      return e
  --
  event <- newLifetime e
  addFinalizer event $ do
    D.traceIO D.dump_gc ("gc: finalise event " ++ showEvent event)
    insert ref (deviceContext ctx) e
  return event

{-# INLINE insert #-}
-- Insert an event into the table.
--
insert :: EventTable -> Lifetime CUDA.Context -> Event.Event -> IO ()
insert ref lctx e = withMVar ref $ \tbl -> do
  me <- HT.lookup tbl lctx
  case me of
    Nothing -> HT.insert tbl lctx (FL.singleton () e)
    Just es -> HT.insert tbl lctx (FL.cons () e es)

-- Create a new event marker that will be filled once execution in the specified
-- stream has completed all previously submitted work.
--
{-# INLINE waypoint #-}
waypoint :: Context -> EventTable -> Stream -> IO Event
waypoint ctx ref stream = do
  event <- create ctx ref
  withLifetime event (`Event.record` Just stream)
  message $ "waypoint " ++ showEvent event ++ " in " ++ showStream stream
  return event

-- Make all future work submitted to the given stream wait until the event
-- reports completion before beginning execution.
--
{-# INLINE after #-}
after :: Event -> Stream -> IO ()
after event stream = do
  message $ "after " ++ showEvent event ++ " in " ++ showStream stream
  withLifetime event $ \e -> Event.wait e (Just stream) []

-- Block the calling thread until the event is recorded
--
{-# INLINE block #-}
block :: Event -> IO ()
block = flip withLifetime Event.block

-- Query the status of the event.
--
{-# INLINE query #-}
query :: Event -> IO Bool
query = flip withLifetime Event.query

-- Explicitly destroy the event.
--
{-# INLINE destroy #-}
destroy :: Event -> IO ()
destroy = finalize

-- Destroy all events in the table.
--
flush :: HashTable (Lifetime CUDA.Context) (FullList () Event.Event) -> IO ()
flush !tbl =
  let clean (!lctx,!es) = do
        withLifetime lctx $ \ctx -> bracket_ (CUDA.push ctx) CUDA.pop (FL.mapM_ (const Event.destroy) es)
        HT.delete tbl lctx
  in
  message "flush reservoir" >> HT.mapM_ clean tbl


-- Debug
-- -----

{-# INLINE message #-}
message :: String -> IO ()
message msg = D.traceIO D.dump_sched ("event: " ++ msg)

{-# INLINE showEvent #-}
showEvent :: Event -> String
showEvent (unsafeGetValue -> Event.Event e) = show e

{-# INLINE showStream #-}
showStream :: Stream -> String
showStream (Stream s) = show s

