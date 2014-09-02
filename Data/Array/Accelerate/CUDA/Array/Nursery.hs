{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Array.Nursery
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Nursery (

  Nursery(..), NRS, new, malloc, stash, flush,

) where

-- friends
import Data.Array.Accelerate.CUDA.FullList                      ( FullList(..) )
import qualified Data.Array.Accelerate.CUDA.FullList            as FL
import qualified Data.Array.Accelerate.CUDA.Debug               as D

-- libraries
import Prelude
import Data.Hashable
import Control.Exception                                        ( bracket_ )
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, mkWeakMVar )
import System.Mem.Weak                                          ( Weak )
import Foreign.Ptr                                              ( ptrToIntPtr )
import Foreign.CUDA.Ptr                                         ( DevicePtr )

import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Data.HashTable.IO                              as HT


-- The nursery is a place to store device memory arrays that are no longer
-- needed. If a new array is requested of a similar size, we might return an
-- array from the nursery instead of calling into the CUDA API to allocate fresh
-- memory.
--
-- Note that pointers are also related to a specific context, so we must include
-- that when looking up the map.
--
-- Note that since there might be many arrays for the same size, each entry in
-- the map keeps a (non-empty) list of device pointers.
--
type HashTable key val  = HT.BasicHashTable key val

type NRS                = MVar ( HashTable (CUDA.Context, Int) (FullList () (DevicePtr ())) )
data Nursery            = Nursery {-# UNPACK #-} !NRS
                                  {-# UNPACK #-} !(Weak NRS)

instance Hashable CUDA.Context where
  {-# INLINE hashWithSalt #-}
  hashWithSalt salt (CUDA.Context ctx)
    = salt `hashWithSalt` (fromIntegral (ptrToIntPtr ctx) :: Int)


-- Generate a fresh nursery
--
new :: IO Nursery
new = do
  tbl    <- HT.new
  ref    <- newMVar tbl
  weak   <- mkWeakMVar ref (flush tbl)
  return $! Nursery ref weak


-- Look for a chunk of memory in the nursery of a given size (or a little bit
-- larger). If found, it is removed from the nursery and a pointer to it
-- returned.
--
{-# INLINE malloc #-}
malloc :: Int -> CUDA.Context -> Nursery -> IO (Maybe (DevicePtr ()))
malloc !n !ctx (Nursery !ref _) = withMVar ref $ \tbl -> do
  let !key = (ctx,n)
  --
  mp  <- HT.lookup tbl key
  case mp of
    Nothing               -> return Nothing
    Just (FL () ptr rest) ->
      case rest of
        FL.Nil          -> HT.delete tbl key              >> return (Just ptr)
        FL.Cons () v xs -> HT.insert tbl key (FL () v xs) >> return (Just ptr)


-- Add a device pointer to the nursery.
--
{-# INLINE stash #-}
stash :: Int -> CUDA.Context -> NRS -> DevicePtr a -> IO ()
stash !n !ctx !ref (CUDA.castDevPtr -> !ptr) = withMVar ref $ \tbl -> do
  let !key = (ctx, n)
  --
  mp  <- HT.lookup tbl key
  case mp of
    Nothing     -> HT.insert tbl key (FL.singleton () ptr)
    Just xs     -> HT.insert tbl key (FL.cons () ptr xs)


-- Delete all entries from the nursery and free all associated device memory.
--
flush :: HashTable (CUDA.Context,Int) (FullList () (CUDA.DevicePtr ())) -> IO ()
flush !tbl =
  let clean (!key@(ctx,_),!val) = do
        bracket_ (CUDA.push ctx) CUDA.pop (FL.mapM_ (const CUDA.free) val)
        HT.delete tbl key
  in
  message "flush nursery" >> HT.mapM_ clean tbl


-- Debug
-- -----

{-# INLINE trace #-}
trace :: String -> IO a -> IO a
trace msg next = D.message D.dump_gc ("gc: " ++ msg) >> next

{-# INLINE message #-}
message :: String -> IO ()
message s = s `trace` return ()

