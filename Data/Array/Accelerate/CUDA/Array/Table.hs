{-# LANGUAGE BangPatterns  #-}
{-# LANGUAGE CPP           #-}
{-# LANGUAGE GADTs         #-}
{-# LANGUAGE PatternGuards #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Array.Table
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Table (

  -- Tables for host/device memory associations
  MemoryTable, Context(..), new, lookup, insert, reclaim

) where

import Prelude                                          hiding ( lookup )
import Data.IORef                                       ( IORef, newIORef, readIORef, mkWeakIORef )
import Data.Maybe                                       ( isJust )
import Data.Hashable                                    ( Hashable(..) )
import Data.Typeable                                    ( Typeable, gcast )
import Control.Monad                                    ( unless )
import Control.Exception                                ( bracket_ )
import Control.Applicative                              ( (<$>) )
import System.Mem                                       ( performGC )
import System.Mem.Weak                                  ( Weak, mkWeak, deRefWeak, finalize )
import System.Mem.StableName                            ( StableName, makeStableName, hashStableName )
import Foreign.Ptr                                      ( ptrToIntPtr )
import Foreign.CUDA.Ptr                                 ( DevicePtr )

import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Data.HashTable.IO                      as HT

import Data.Array.Accelerate.Array.Data                 ( ArrayData )
import qualified Data.Array.Accelerate.CUDA.Debug       as D

#include "accelerate.h"


-- We use an MVar to the hash table, so that several threads may safely access
-- it concurrently. This includes the finalisation threads that remove entries
-- from the table.
--
-- It is important that we can garbage collect old entries from the table when
-- the key is no longer reachable in the heap. Hence the value part of each
-- table entry is a (Weak val), where the stable name 'key' is the key for the
-- memo table, and the 'val' is the value of this table entry. When the key
-- becomes unreachable, a finaliser will fire and remove this entry from the
-- hash buckets, and further attempts to dereference the weak pointer will
-- return Nothing. References from 'val' to the key are ignored (see the
-- semantics of weak pointers in the documentation).
--
type HashTable key val  = HT.BasicHashTable key val
type MT                 = IORef ( HashTable HostArray DeviceArray )
data MemoryTable        = MemoryTable {-# UNPACK #-} !MT
                                      {-# UNPACK #-} !(Weak MT)

-- The currently active context. Finaliser threads need to check if the context
-- is still active before attempting to release their associated memory.
--
data Context = Context {-# UNPACK #-} !CUDA.Context
                       {-# UNPACK #-} !(Weak CUDA.Context)

-- Arrays on the host and device
--
data HostArray where
  HostArray :: Typeable e
            => {-# UNPACK #-} !Int      -- unique ID relating to the parent context
            -> {-# UNPACK #-} !(StableName (ArrayData e))
            -> HostArray

data DeviceArray where
  DeviceArray :: Typeable e
              => {-# UNPACK #-} !(Weak (DevicePtr e))
              -> DeviceArray

instance Eq HostArray where
  HostArray _ a1 == HostArray _ a2
    = maybe False (== a2) (gcast a1)

instance Hashable HostArray where
  hash (HostArray cid sn) = hashWithSalt cid sn

instance Show HostArray where
  show (HostArray _ sn) = "Array #" ++ show (hashStableName sn)


-- Referencing arrays
-- ------------------

-- Create a new hash table from host to device arrays. When the structure is
-- collected it will finalise all entries in the table.
--
new :: IO MemoryTable
new = do
  tbl  <- HT.new
  ref  <- newIORef tbl
  weak <- mkWeakIORef ref (table_finalizer tbl)
  return $! MemoryTable ref weak


-- Look for the device memory corresponding to a given host-side array.
--
lookup :: (Typeable a, Typeable b) => Context -> MemoryTable -> ArrayData a -> IO (Maybe (DevicePtr b))
lookup ctx (MemoryTable ref _) !arr = do
  sa <- makeStableArray ctx arr
  mw <- withIORef ref (`HT.lookup` sa)
  case mw of
    Nothing              -> trace ("lookup/not found: " ++ show sa) $ return Nothing
    Just (DeviceArray w) -> do
      mv <- deRefWeak w
      case mv of
        Just v | Just p <- gcast v -> trace ("lookup/found: " ++ show sa) $ return (Just p)
               | otherwise         -> INTERNAL_ERROR(error) "lookup" $ "type mismatch"
        Nothing                    ->
          makeStableArray ctx arr >>= \x -> INTERNAL_ERROR(error) "lookup" $ "dead weak pair: " ++ show x


-- Record an association between a host-side array and a new device memory area.
-- The device memory will be freed when the host array is garbage collected.
--
insert :: (Typeable a, Typeable b) => Context -> MemoryTable -> ArrayData a -> DevicePtr b -> IO ()
insert ctx@(Context _ weak_ctx) (MemoryTable ref weak_ref) !arr !ptr = do
  key  <- makeStableArray ctx arr
  dev  <- DeviceArray `fmap` mkWeak arr ptr (Just $ finalizer weak_ctx weak_ref key ptr)
  tbl  <- readIORef ref
  message $ "insert: " ++ show key
  HT.insert tbl key dev


-- Removing entries
-- ----------------

-- Initiate garbage collection and finalise any arrays that have been marked as
-- unreachable.
--
reclaim :: MemoryTable -> IO ()
reclaim (MemoryTable _ weak_ref) = do
  (free, total) <- CUDA.getMemInfo
  performGC
  mr <- deRefWeak weak_ref
  case mr of
    Nothing  -> return ()
    Just ref -> withIORef ref $ \tbl ->
      flip HT.mapM_ tbl $ \(_,DeviceArray w) -> do
        alive <- isJust `fmap` deRefWeak w
        unless alive $ finalize w
  --
  D.when D.dump_gc $ do
    (free', _)  <- CUDA.getMemInfo
    message $ "reclaim: freed "   ++ showBytes (fromIntegral (free - free'))
                        ++ ", "   ++ showBytes (fromIntegral free')
                        ++ " of " ++ showBytes (fromIntegral total) ++ " remaining"

-- Because a finaliser might run at any time, we must reinstate the context in
-- which the array was allocated before attempting to release it.
--
-- Note also that finaliser threads will silently terminate if an exception is
-- raised. If the context, and thereby all allocated memory, was destroyed
-- externally before the thread had a chance to run, all we need do is update
-- the hash tables --- but we must do this first before failing to use a dead
-- context.
--
finalizer :: Weak CUDA.Context -> Weak MT -> HostArray -> DevicePtr b -> IO ()
finalizer !weak_ctx !weak_ref !key !ptr = do
  mr <- deRefWeak weak_ref
  case mr of
    Nothing  -> message ("finalise/dead table: " ++ show key)
    Just ref -> trace   ("finalise: "            ++ show key) $ withIORef ref (`HT.delete` key)
  --
  mc <- deRefWeak weak_ctx
  case mc of
    Nothing  -> message ("finalise/dead context: " ++ show key)
    Just ctx -> bracket_ (CUDA.push ctx) CUDA.pop (CUDA.free ptr)


table_finalizer :: HashTable HostArray DeviceArray -> IO ()
table_finalizer !tbl
  = trace "table finaliser"
  $ HT.mapM_ (\(_,DeviceArray w) -> finalize w) tbl


-- Miscellaneous
-- -------------

{-# INLINE makeStableArray #-}
makeStableArray :: Typeable a => Context -> ArrayData a -> IO HostArray
makeStableArray (Context (CUDA.Context !p) !_) !arr =
  let cid = fromIntegral (ptrToIntPtr p)
  in  HostArray cid <$> makeStableName arr

{-# INLINE withIORef #-}
withIORef :: IORef a -> (a -> IO b) -> IO b
withIORef ref f = readIORef ref >>= f


-- Debug
-- -----

{-# INLINE showBytes #-}
showBytes :: Int -> String
showBytes x = D.showFFloatSIBase (Just 0) 1024 (fromIntegral x :: Double) "B"

{-# INLINE trace #-}
trace :: String -> IO a -> IO a
trace msg next = D.message D.dump_gc ("gc: " ++ msg) >> next

{-# INLINE message #-}
message :: String -> IO ()
message s = s `trace` return ()

