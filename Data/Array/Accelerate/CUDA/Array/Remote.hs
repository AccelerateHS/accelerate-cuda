{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE ConstraintKinds     #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeFamilies        #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Array.Remote
-- Copyright   : [2015..2016] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell, Robert Clifton-Everest
-- License     : BSD3
--
-- Maintainer  : Robert Clifton-Everest <robertce@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Remote (

  -- Tables for host/device memory associations
  MemoryTable, R.PrimElt,
  new, malloc, withRemote, free, insertUnmanaged, reclaim

) where

import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, modifyMVar, readMVar )
import Control.Exception
import Control.Monad.IO.Class                                   ( MonadIO, liftIO )
import Control.Monad.Trans.Reader
import Data.Functor
import Data.IntMap.Strict                                       ( IntMap )
import Data.Proxy
import Data.Typeable                                            ( Typeable )
import Foreign.Ptr                                              ( ptrToIntPtr )
import Foreign.Storable
import Prelude                                                  hiding ( lookup )
import qualified Data.IntMap.Strict                             as IM

import Foreign.CUDA.Driver.Error
import qualified Foreign.CUDA.Driver                            as CUDA

import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Lifetime                           ( unsafeGetValue )
import qualified Data.Array.Accelerate.Array.Remote             as R

import Data.Array.Accelerate.CUDA.Context                       ( Context(..), push, pop )
import Data.Array.Accelerate.CUDA.Execute.Event                 ( Event, EventTable, waypoint, query )
import Data.Array.Accelerate.CUDA.Execute.Stream                ( Stream )
import qualified Data.Array.Accelerate.CUDA.Debug               as Debug


-- Instance for basic remote memory management functionality.
--
type CRM = ReaderT (Maybe Stream) IO

instance R.RemoteMemory CRM where
  type RemotePtr CRM = CUDA.DevicePtr
  --
  mallocRemote n       = ReaderT $ \_ ->
    fmap Just (CUDA.mallocArray n)
      `catch` \e -> case e of
                      ExitCode OutOfMemory -> return Nothing
                      _                    -> trace ("malloc failed with error: " ++ show e) (throwIO e)

  pokeRemote n dst ad  = ReaderT $ \mst ->
    transfer "poke" (n * sizeOfPtr dst) $
    CUDA.pokeArrayAsync n (CUDA.HostPtr (ptrsOfArrayData ad)) dst mst

  peekRemote n src ad  = ReaderT $ \mst ->
    transfer "peek" (n * sizeOfPtr src) $
    CUDA.peekArrayAsync n src (CUDA.HostPtr (ptrsOfArrayData ad)) mst

  castRemotePtr _      = CUDA.castDevPtr
  totalRemoteMem       = ReaderT $ \_ -> snd <$> CUDA.getMemInfo
  availableRemoteMem   = ReaderT $ \_ -> fst <$> CUDA.getMemInfo
  remoteAllocationSize = return 1024


-- We leverage the memory cache from the accelerate base package.
--
-- However, we actually need multiple caches because every pointer has an
-- associated CUDA context. We could pair every DevicePtr with its context and
-- just have a single table, but the LRU implementation in the base package
-- assumes that remote pointers can be re-used, something that would not be true
-- for pointers allocated under different contexts.
--
type MT          = IntMap (R.MemoryTable CUDA.DevicePtr Task)
data MemoryTable = MemoryTable {-# UNPACK #-} !EventTable
                               {-# UNPACK #-} !(MVar MT)

type Task = Maybe Event

instance R.Task Task where
  completed Nothing  = return True
  completed (Just e) = query e


-- Create a MemoryTable.
--
new :: EventTable -> IO MemoryTable
new et = do
  message "initialise CUDA memory table"
  MemoryTable et <$> newMVar IM.empty


-- Perform action on the device ptr that matches the given host-side array. Any
-- operations
--
withRemote
    :: forall e a b. R.PrimElt e a
    => Context
    -> MemoryTable
    -> ArrayData e
    -> (CUDA.DevicePtr a -> IO b)
    -> Maybe Stream
    -> IO (Maybe b)
withRemote ctx (MemoryTable et ref) ad run ms = do
  ct <- readMVar ref
  case IM.lookup (contextId ctx) ct of
    Nothing -> $internalError "withRemote" "context not found"
    Just mc -> streaming ms $ R.withRemote mc ad run'
  where
    run' :: R.RemotePtr CRM a -> CRM (Task, b)
    run' p = liftIO $ do
      c  <- run p
      case ms of
        Nothing -> return (Nothing, c)
        Just s  -> do
          e <- waypoint ctx et s
          return (Just e, c)


-- Allocate a new device array to be associated with the given host-side array.
-- Has the same properties as `Data.Array.Accelerate.Array.Remote.LRU.malloc`
malloc :: forall a b. (Typeable a, R.PrimElt a b)
       => Context
       -> MemoryTable
       -> ArrayData a
       -> Bool
       -> Int
       -> IO Bool
malloc !ctx (MemoryTable _ !ref) !ad !frozen !n = do
  mt <- modifyMVar ref $ \ct -> blocking $ do
   case IM.lookup (contextId ctx) ct of
           Nothing -> trace "malloc/context not found" $ insertContext ctx ct
           Just mt -> return (ct, mt)
  blocking $ R.malloc mt ad frozen n


-- Explicitly free an array in the LRU table. Has the same properties as
-- `Data.Array.Accelerate.Array.Remote.LRU.free`
--
free :: R.PrimElt a b
     => Context
     -> MemoryTable
     -> ArrayData a
     -> IO ()
free !ctx (MemoryTable _ !ref) !arr = withMVar ref $ \ct ->
  case IM.lookup (contextId ctx) ct of
    Nothing -> message "free/context not found"
    Just mt -> R.free (Proxy :: Proxy CRM) mt arr


-- Record an association between a host-side array and a device memory area that was
-- not allocated by accelerate. The device memory will NOT be freed by the memory
-- manager.
--
insertUnmanaged
    :: R.PrimElt a b
    => Context
    -> MemoryTable
    -> ArrayData a
    -> CUDA.DevicePtr b
    -> IO ()
insertUnmanaged !ctx (MemoryTable _ !ref) !arr !ptr = do
  mt <- modifyMVar ref $ \ct -> blocking $ do
   case IM.lookup (contextId ctx) ct of
           Nothing -> trace "insertUnmanaged/context not found" $ insertContext ctx ct
           Just mt -> return (ct, mt)
  blocking $ R.insertUnmanaged mt arr ptr

insertContext
    :: Context
    -> MT
    -> CRM ( MT, R.MemoryTable CUDA.DevicePtr Task )
insertContext ctx ct = liftIO $ do
   mt <- R.new (\p -> bracket_ (push ctx) pop (CUDA.free p))
   return (IM.insert (contextId ctx) mt ct, mt)


-- Removing entries
-- ----------------

-- Initiate garbage collection and finalise any arrays that have been marked as
-- unreachable.
--
reclaim :: MemoryTable -> IO ()
reclaim (MemoryTable _ ref) = withMVar ref (blocking . mapM_ R.reclaim . IM.elems)

-- Miscellaneous
-- -------------

{-# INLINE contextId #-}
contextId :: Context -> Int
contextId !ctx =
  let CUDA.Context !p = unsafeGetValue (deviceContext ctx)
  in fromIntegral (ptrToIntPtr p)

{-# INLINE blocking #-}
blocking :: CRM a -> IO a
blocking = flip runReaderT Nothing

{-# INLINE streaming #-}
streaming :: Maybe Stream -> CRM a -> IO a
streaming = flip runReaderT

{-# INLINE sizeOfPtr #-}
sizeOfPtr :: forall a. Storable a => CUDA.DevicePtr a -> Int
sizeOfPtr _ = sizeOf (undefined :: a)

-- Debug
-- -----

{-# INLINE trace #-}
trace :: MonadIO m => String -> m a -> m a
trace msg next = message msg >> next

{-# INLINE message #-}
message :: MonadIO m => String -> m ()
message msg = liftIO $ Debug.traceIO Debug.dump_gc ("gc/lru: " ++ msg)

{-# INLINE showBytes #-}
showBytes :: Int -> String
showBytes x = Debug.showFFloatSIBase (Just 0) 1024 (fromIntegral x :: Double) "B"

{-# INLINE transfer #-}
transfer :: String -> Int -> IO () -> IO ()
transfer name bytes action
  = let showRate x t        = Debug.showFFloatSIBase (Just 3) 1024 (fromIntegral x / t) "B/s"
        msg gpuTime cpuTime = "gc/lru: " ++ name ++ ": "
                                         ++ showBytes bytes ++ " @ " ++ showRate bytes gpuTime ++ ", "
                                         ++ Debug.elapsed gpuTime cpuTime
    in
    Debug.timed Debug.dump_gc msg Nothing action

