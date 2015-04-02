{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE CPP          #-}
{-# LANGUAGE ViewPatterns #-}
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

  Event, create, waypoint, after, block, query, destroy,

) where

-- friends
import Data.Array.Accelerate.Lifetime
import qualified Data.Array.Accelerate.CUDA.Debug               as D

-- libraries
import Foreign.CUDA.Driver.Stream                               ( Stream(..) )
import qualified Foreign.CUDA.Driver.Event                      as CUDA

type Event = Lifetime CUDA.Event

-- Create a new event. It will be automatically garbage collected, but is not
-- suitable for timing purposes.
--
{-# INLINE create #-}
create :: IO Event
create = do
  e <- CUDA.create [CUDA.DisableTiming]
  event <- newLifetime e
  addFinalizer event $
    D.traceIO D.dump_gc ("gc: finalise event " ++ showEvent event) >> CUDA.destroy e
  message ("create " ++ showEvent event)
  return event

-- Create a new event marker that will be filled once execution in the specified
-- stream has completed all previously submitted work.
--
{-# INLINE waypoint #-}
waypoint :: Stream -> IO Event
waypoint stream = do
  event <- create
  withLifetime event (`CUDA.record` Just stream)
  message $ "waypoint " ++ showEvent event ++ " in " ++ showStream stream
  return event

-- Make all future work submitted to the given stream wait until the event
-- reports completion before beginning execution.
--
{-# INLINE after #-}
after :: Event -> Stream -> IO ()
after event stream = do
  message $ "after " ++ showEvent event ++ " in " ++ showStream stream
  withLifetime event $ \e -> CUDA.wait e (Just stream) []

-- Block the calling thread until the event is recorded
--
{-# INLINE block #-}
block :: Event -> IO ()
block = flip withLifetime CUDA.block

-- Query the status of the event.
--
{-# INLINE query #-}
query :: Event -> IO Bool
query = flip withLifetime CUDA.query

-- Explicitly destroy the event.
--
{-# INLINE destroy #-}
destroy :: Event -> IO ()
destroy = finalize


-- Debug
-- -----

{-# INLINE message #-}
message :: String -> IO ()
message msg = D.traceIO D.dump_sched ("event: " ++ msg)

{-# INLINE showEvent #-}
showEvent :: Event -> String
showEvent (unsafeGetValue -> CUDA.Event e) = show e

{-# INLINE showStream #-}
showStream :: Stream -> String
showStream (Stream s) = show s

