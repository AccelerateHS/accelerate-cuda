{-# LANGUAGE BangPatterns #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Execute.Stream
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Execute.Stream (

  Stream, Reservoir, new, create,

) where

-- friends
import Data.Array.Accelerate.CUDA.Array.Nursery                 ( ) -- hashable CUDA.Context instance
import Data.Array.Accelerate.CUDA.Context                       ( Context(..) )
import Data.Array.Accelerate.CUDA.FullList                      ( FullList(..) )
import qualified Data.Array.Accelerate.CUDA.FullList            as FL
import qualified Data.Array.Accelerate.CUDA.Debug               as D

-- libraries
import Control.Exception                                        ( bracket_ )
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, mkWeakMVar )
import System.Mem.Weak                                          ( Weak, deRefWeak, addFinalizer )
import Foreign.CUDA.Driver.Stream                               ( Stream )
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

type RSV                = MVar ( HashTable CUDA.Context (FullList () Stream) )
data Reservoir          = Reservoir !RSV !(Weak RSV)


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
create !ctx (Reservoir !ref !weak_ref) = withMVar ref $ \tbl -> do
  let key = deviceContext ctx
  --
  ms    <- HT.lookup tbl key
  case ms of
    Nothing -> do
        stream <- Stream.create []
        addFinalizer (Stream.useStream stream) (finalizer (weakContext ctx) weak_ref stream)
        message $ "new " ++ show (Stream.useStream stream)
        return stream

    Just (FL () stream rest) -> do
      case rest of
        FL.Nil          -> HT.delete tbl key
        FL.Cons () s ss -> HT.insert tbl key (FL () s ss)
      --
      return stream


-- Finaliser for CUDA execution streams. Because the finaliser thread might run
-- at any time, we need a weak reference to the context the stream was allocated
-- in. We don't need to make the context current in order to destroy the stream,
-- as we do when deallocating memory, we just need to know whether the context
-- is still alive before attempting to destroy the stream (because if it is not:
-- segfault).
--
finalizer :: Weak CUDA.Context -> Weak RSV -> Stream -> IO ()
finalizer !weak_ctx !weak_ref !stream = do
  mc <- deRefWeak weak_ctx
  case mc of
    Nothing     -> message ("finalise/dead context " ++ showStream stream)
    Just ctx    -> do
      --
      mr <- deRefWeak weak_ref
      case mr of
        Nothing  -> trace ("finalise/free "  ++ showStream stream) $ Stream.destroy stream
        Just ref -> trace ("finalise/stash " ++ showStream stream) $ withMVar ref $ \tbl -> do
          --
          ms <- HT.lookup tbl ctx
          case ms of
            Nothing     -> HT.insert tbl ctx (FL.singleton () stream)
            Just ss     -> HT.insert tbl ctx (FL.cons () stream ss)


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

