{-# LANGUAGE BangPatterns  #-}
{-# LANGUAGE MagicHash     #-}
{-# LANGUAGE UnboxedTuples #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Execute.Event
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
module Data.Array.Accelerate.CUDA.Execute.Event (

  Event, create, waypoint, after, block,

) where

-- friends
import qualified Data.Array.Accelerate.CUDA.Debug               as D

-- libraries
import Foreign.CUDA.Driver.Event                                ( Event(..) )
import Foreign.CUDA.Driver.Stream                               ( Stream )
import qualified Foreign.CUDA.Driver.Event                      as Event

import GHC.Base
import GHC.Ptr


-- Create a new event that will be automatically garbage collected. The event is
-- not suitable for timing purposes.
--
{-# INLINE create #-}
create :: IO Event
create = do
  event <- Event.create [Event.DisableTiming]
  addEventFinalizer event $ trace ("destroy " ++ show event) (Event.destroy event)
  return event

-- Create a new event marker that will be filled once execution in the specified
-- stream has completed all previously submitted work.
--
{-# INLINE waypoint #-}
waypoint :: Stream -> IO Event
waypoint stream = do
  event <- create
  Event.record event (Just stream)
--  message $ "waypoint " ++ show event ++ " in " ++ show stream
  return event

-- Make all future work submitted to the given stream wait until the event
-- reports completion before beginning execution.
--
{-# INLINE after #-}
after :: Event -> Stream -> IO ()
after event stream = do
--  message $ "after " ++ show event ++ " in " ++ show stream
  Event.wait event (Just stream) []

-- Block the calling thread until the event is recorded
--
{-# INLINE block #-}
block :: Event -> IO ()
block = Event.block


-- Add a finaliser to an event token
--
addEventFinalizer :: Event -> IO () -> IO ()
addEventFinalizer e@(Event (Ptr e#)) f = IO $ \s ->
  case mkWeak# e# e f s of (# s', _w #) -> (# s', () #)


-- Debug
-- -----

{-# INLINE trace #-}
trace :: String -> IO a -> IO a
trace msg next = D.message D.dump_exec ("event: " ++ msg) >> next

-- {-# INLINE message #-}
-- message :: String -> IO ()
-- message s = s `trace` return ()

