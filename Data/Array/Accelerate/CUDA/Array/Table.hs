{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Array.Table
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Table (

  -- Tables for host/device memory associations
  MemoryTable, new, lookup, malloc, free, insert, insertRemote, reclaim

) where

import Control.Applicative
import Control.Concurrent                                       ( yield )
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, mkWeakMVar )
import Control.Exception                                        ( bracket_, catch, throwIO )
import Control.Monad                                            ( unless )
import Data.Hashable                                            ( Hashable(..) )
import Data.Maybe                                               ( isJust )
import Data.Typeable                                            ( Typeable, gcast )
import System.Mem                                               ( performGC )
import System.Mem.StableName                                    ( StableName, makeStableName, hashStableName )
import System.Mem.Weak                                          ( Weak, mkWeak, deRefWeak, finalize )
import Prelude                                                  hiding ( lookup )

import Foreign.Ptr                                              ( ptrToIntPtr )
import Foreign.Storable                                         ( Storable, sizeOf )
import Foreign.CUDA.Ptr                                         ( DevicePtr )
import Foreign.CUDA.Driver.Error
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Data.HashTable.IO                              as HT

import Data.Array.Accelerate.Error                              ( internalError )
import Data.Array.Accelerate.Array.Data                         ( ArrayData )
import Data.Array.Accelerate.CUDA.Context                       ( Context, weakContext, deviceContext )
import Data.Array.Accelerate.CUDA.Array.Nursery                 ( Nursery(..), NRS )
import qualified Data.Array.Accelerate.CUDA.Array.Nursery       as N
import qualified Data.Array.Accelerate.CUDA.Debug               as D


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
type MT                 = MVar ( HashTable HostArray DeviceArray )
data MemoryTable        = MemoryTable {-# UNPACK #-} !MT
                                      {-# UNPACK #-} !(Weak MT)
                                      {-# UNPACK #-} !Nursery

-- Arrays on the host and device
--
type ContextId = Int

data HostArray where
  HostArray :: Typeable e
            => {-# UNPACK #-} !ContextId        -- unique ID relating to the parent context
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
  {-# INLINE hashWithSalt #-}
  hashWithSalt salt (HostArray cid sn)
    = salt `hashWithSalt` cid `hashWithSalt` sn

instance Show HostArray where
  show (HostArray _ sn) = "Array #" ++ show (hashStableName sn)


-- Referencing arrays
-- ------------------

-- Create a new hash table from host to device arrays. When the structure is
-- collected it will finalise all entries in the table.
--
new :: IO MemoryTable
new = do
  message "initialise memory table"
  tbl  <- HT.new
  ref  <- newMVar tbl
  nrs  <- N.new
  weak <- mkWeakMVar ref (table_finalizer tbl)
  return $! MemoryTable ref weak nrs


-- Look for the device memory corresponding to a given host-side array.
--
lookup :: (Typeable a, Typeable b) => Context -> MemoryTable -> ArrayData a -> IO (Maybe (DevicePtr b))
lookup ctx (MemoryTable !ref _ _) !arr = do
  sa <- makeStableArray ctx arr
  mw <- withMVar ref (`HT.lookup` sa)
  case mw of
    Nothing              -> trace ("lookup/not found: " ++ show sa) $ return Nothing
    Just (DeviceArray w) -> do
      mv <- deRefWeak w
      case mv of
        Just v | Just p <- gcast v -> trace ("lookup/found: " ++ show sa) $ return (Just p)
               | otherwise         -> $internalError "lookup" $ "type mismatch"

        -- Note: [Weak pointer weirdness]
        --
        -- After the lookup is successful, there might conceivably be no further
        -- references to 'arr'. If that is so, and a garbage collection
        -- intervenes, the weak pointer might get tombstoned before 'deRefWeak'
        -- gets to it. In that case we throw an error (below). However, because
        -- we have used 'arr' in the continuation, this ensures that 'arr' is
        -- reachable in the continuation of 'deRefWeak' and thus 'deRefWeak'
        -- always succeeds. This sort of weirdness, typical of the world of weak
        -- pointers, is why we can not reuse the stable name 'sa' computed
        -- above in the error message.
        --
        Nothing                    ->
          makeStableArray ctx arr >>= \x -> $internalError "lookup" $ "dead weak pair: " ++ show x


-- Allocate a new device array to be associated with the given host-side array.
-- This will attempt to use an old array from the nursery, but will otherwise
-- allocate fresh data.
--
-- Instead of allocating the exact number of elements requested, we round up to
-- a fixed chunk size; currently set at 128 elements. This means there is a
-- greater chance the nursery will get a hit, and moreover that we can search
-- the nursery for an exact size. TLM: I believe the CUDA API allocates in
-- chunks, of size 4MB.
--
malloc :: forall a b. (Typeable a, Typeable b, Storable b) => Context -> MemoryTable -> ArrayData a -> Int -> IO (DevicePtr b)
malloc !ctx mt@(MemoryTable _ _ !nursery) !ad !n = do
  let -- next highest multiple of f from x
      multiple x f      = floor ((x + (f-1)) / f :: Double)
      chunk             = 1024

      !n'               = chunk * multiple (fromIntegral n) (fromIntegral chunk)
      !bytes            = n' * sizeOf (undefined :: b)
  --
  mp  <- N.malloc bytes (deviceContext ctx) nursery
  ptr <- case mp of
           Just p       -> trace "malloc/nursery" $ return (CUDA.castDevPtr p)
           Nothing      -> trace "malloc/new"     $
             CUDA.mallocArray n' `catch` \(e :: CUDAException) ->
               case e of
                 ExitCode OutOfMemory -> reclaim mt >> CUDA.mallocArray n'
                 _                    -> throwIO e
  insert ctx mt ad ptr bytes
  return ptr


-- Deallocate the device array associated with the given host-side array. This
-- calls the finaliser for that array immediately, regardless of the current (or
-- future) use status of that array.
--
free :: Typeable a => Context -> MemoryTable -> ArrayData a -> IO ()
free !ctx (MemoryTable !ref _ _) !arr = do
  sa <- makeStableArray ctx arr
  mw <- withMVar ref (`HT.lookup` sa)
  case mw of
    Nothing              -> message ("free/not found: " ++ show sa)
    Just (DeviceArray w) -> trace   ("free/evict: " ++ show sa) $ finalize w


-- Record an association between a host-side array and a new device memory area.
-- The device memory will be freed when the host array is garbage collected.
--
insert :: (Typeable a, Typeable b) => Context -> MemoryTable -> ArrayData a -> DevicePtr b -> Int -> IO ()
insert !ctx (MemoryTable !ref !weak_ref (Nursery _ !weak_nrs)) !arr !ptr !bytes = do
  key  <- makeStableArray ctx arr
  dev  <- DeviceArray `fmap` mkWeak arr ptr (Just $ finalizer (weakContext ctx) weak_ref weak_nrs key ptr bytes)
  message      $ "insert: " ++ show key
  withMVar ref $ \tbl -> HT.insert tbl key dev


-- Record an association between a host-side array and a device memory area that was
-- not allocated by accelerate. The device memory will NOT be freed when the host
-- array is garbage collected.
--
insertRemote :: (Typeable a, Typeable b) => Context -> MemoryTable -> ArrayData a -> DevicePtr b -> IO ()
insertRemote !ctx (MemoryTable !ref !weak_ref _) !arr !ptr = do
  key  <- makeStableArray ctx arr
  dev  <- DeviceArray `fmap` mkWeak arr ptr (Just $ remoteFinalizer weak_ref key)
  message      $ "insert/remote: " ++ show key
  withMVar ref $ \tbl -> HT.insert tbl key dev


-- Removing entries
-- ----------------

-- Initiate garbage collection and finalise any arrays that have been marked as
-- unreachable.
--
reclaim :: MemoryTable -> IO ()
reclaim (MemoryTable _ weak_ref (Nursery nrs _)) = do
  (before, total) <- CUDA.getMemInfo
  performGC
  yield
  withMVar nrs N.flush
  mr <- deRefWeak weak_ref
  case mr of
    Nothing  -> return ()
    Just ref -> withMVar ref $ \tbl ->
      flip HT.mapM_ tbl $ \(_,DeviceArray w) -> do
        alive <- isJust `fmap` deRefWeak w
        unless alive $ finalize w
  --
  D.when D.dump_gc $ do
    (after, _) <- CUDA.getMemInfo
    message $ "reclaim: freed "   ++ showBytes (fromIntegral (before - after))
                        ++ ", "   ++ showBytes (fromIntegral after)
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
finalizer :: Weak CUDA.Context -> Weak MT -> Weak NRS -> HostArray -> DevicePtr b -> Int -> IO ()
finalizer !weak_ctx !weak_ref !weak_nrs !key !ptr !bytes = do
  mr <- deRefWeak weak_ref
  case mr of
    Nothing  -> message ("finalise/dead table: " ++ show key)
    Just ref -> withMVar ref (`HT.delete` key)
  --
  mc <- deRefWeak weak_ctx
  case mc of
    Nothing  -> message ("finalise/dead context: " ++ show key)
    Just ctx -> do
      --
      mn <- deRefWeak weak_nrs
      case mn of
        Nothing  -> trace ("finalise/free: "     ++ show key) $ bracket_ (CUDA.push ctx) CUDA.pop (CUDA.free ptr)
        Just nrs -> trace ("finalise/nursery: "  ++ show key) $ N.stash bytes ctx nrs ptr

remoteFinalizer :: Weak MT -> HostArray -> IO ()
remoteFinalizer !weak_ref !key = do
  mr <- deRefWeak weak_ref
  case mr of
    Nothing  -> message ("finalise/dead table: " ++ show key)
    Just ref -> trace   ("finalise: "            ++ show key) $ withMVar ref (`HT.delete` key)

table_finalizer :: HashTable HostArray DeviceArray -> IO ()
table_finalizer !tbl
  = trace "table finaliser"
  $ HT.mapM_ (\(_,DeviceArray w) -> finalize w) tbl


-- Miscellaneous
-- -------------

{-# INLINE makeStableArray #-}
makeStableArray :: Typeable a => Context -> ArrayData a -> IO HostArray
makeStableArray !ctx !arr =
  let CUDA.Context !p   = deviceContext ctx
      !cid              = fromIntegral (ptrToIntPtr p)
  in
  HostArray cid <$> makeStableName arr


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

