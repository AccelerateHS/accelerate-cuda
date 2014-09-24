{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE CPP          #-}
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

  Event, create, waypoint, after, block, Event.destroy,

) where

-- friends
#ifdef ACCELERATE_DEBUG
import qualified Data.Array.Accelerate.CUDA.Debug               as D
#endif

-- libraries
import Foreign.CUDA.Driver.Event                                ( Event(..) )
import Foreign.CUDA.Driver.Stream                               ( Stream(..) )
import qualified Foreign.CUDA.Driver.Event                      as Event


-- Create a new event. It will not be automatically garbage collected, and is
-- not suitable for timing purposes.
--
{-# INLINE create #-}
create :: IO Event
create = do
  event <- Event.create [Event.DisableTiming]
  message ("create " ++ showEvent event)
  return event

-- Create a new event marker that will be filled once execution in the specified
-- stream has completed all previously submitted work.
--
{-# INLINE waypoint #-}
waypoint :: Stream -> IO Event
waypoint stream = do
  event <- create
  Event.record event (Just stream)
  message $ "waypoint " ++ showEvent event ++ " in " ++ showStream stream
  return event

-- Make all future work submitted to the given stream wait until the event
-- reports completion before beginning execution.
--
{-# INLINE after #-}
after :: Event -> Stream -> IO ()
after event stream = do
  message $ "after " ++ showEvent event ++ " in " ++ showStream stream
  Event.wait event (Just stream) []

-- Block the calling thread until the event is recorded
--
{-# INLINE block #-}
block :: Event -> IO ()
block = Event.block


-- Add a finaliser to an event token
--
-- addEventFinalizer :: Event -> IO () -> IO ()
-- addEventFinalizer e@(Event (Ptr e#)) f = IO $ \s ->
--   case mkWeak# e# e f s of (# s', _w #) -> (# s', () #)


-- Debug
-- -----

{-# INLINE trace #-}
trace :: String -> IO a -> IO a
trace _msg next = do
#ifdef ACCELERATE_DEBUG
  D.when D.verbose $ D.message D.dump_exec ("event: " ++ _msg)
#endif
  next

{-# INLINE message #-}
message :: String -> IO ()
message s = s `trace` return ()

{-# INLINE showEvent #-}
showEvent :: Event -> String
showEvent (Event e) = show e

{-# INLINE showStream #-}
showStream :: Stream -> String
showStream (Stream s) = show s

