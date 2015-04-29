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
-- Module      : Data.Array.Accelerate.CUDA.Array.Cache
-- Copyright   : [2015..2015] Robert Clifton-Everest, Manuel M T Chakravarty,
--                            Gabriele Keller
-- License     : BSD3
--
-- Maintainer  : Robert Clifton-Everest <robertce@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Cache (

  -- Tables for host/device memory associations
  MemoryTable, new, malloc, withRemote, free, insertUnmanaged, reclaim

) where

import Data.Functor
import Data.IntMap.Strict                                       ( IntMap )
import Data.Proxy
import Data.Typeable                                            ( Typeable )
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, modifyMVar, readMVar )
import Control.Exception                                        ( bracket_ )
import Control.Monad.IO.Class                                   ( MonadIO, liftIO )
import Control.Monad.Trans.Reader
import Foreign.CUDA.Ptr                                         ( DevicePtr )
import Prelude                                                  hiding ( lookup )

import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Data.IntMap.Strict                             as IM

import Data.Array.Accelerate.Array.Data                         ( ArrayData )
import Data.Array.Accelerate.Array.Memory                       ( PrimElt )
import Data.Array.Accelerate.CUDA.Array.Table                   ( CRM, contextId )
import Data.Array.Accelerate.CUDA.Context                       ( Context, push, pop )
import Data.Array.Accelerate.CUDA.Execute.Event                 ( Event, EventTable, waypoint, query )
import Data.Array.Accelerate.CUDA.Execute.Stream                ( Stream )
import Data.Array.Accelerate.Error
import qualified Data.Array.Accelerate.CUDA.Debug               as D
import qualified Data.Array.Accelerate.Array.Memory.Cache       as MC

-- We leverage the memory cache from the accelerate base package. However, we
-- actually need multiple caches. This is because every pointer has an
-- associated CUDA context. We could pair every DevicePtr with its context and
-- just have a single table, but the MemoryCache API in the base package assumes
-- that remote pointers can be re-used, something that would not be true for
-- pointers allocated under different contexts.
--
data MemoryTable = MemoryTable EventTable (MVar (IntMap (MC.MemoryCache DevicePtr (Maybe Event))))

instance MC.Task (Maybe Event) where
  isDone Nothing  = return True
  isDone (Just e) = query e

-- Create a MemoryTable.
--
new :: EventTable -> IO MemoryTable
new et = trace "initialise CUDA memory table" $ MemoryTable et <$> newMVar IM.empty

-- Perform action on the device ptr that matches the given host-side array. Any
-- operations
--
withRemote :: PrimElt e a => Context -> MemoryTable -> ArrayData e -> (DevicePtr a -> IO b) -> Maybe Stream -> IO (Maybe b)
withRemote ctx (MemoryTable et ref) ad run ms = do
  ct <- readMVar ref
  case IM.lookup (contextId ctx) ct of
    Nothing -> $internalError "withRemote" "context not found"
    Just mc -> do
      streaming ms $ MC.withRemote mc ad run'
  where
    run' p = do
      c  <- run p
      case ms of
        Nothing -> return (Nothing, c)
        Just s  -> do
          e <- waypoint ctx et s
          return (Just e, c)

-- Allocate a new device array to be associated with the given host-side array.
-- Has the same properties as `Data.Array.Accelerate.Array.Memory.Cache.malloc`
malloc :: forall a b. (Typeable a, PrimElt a b) => Context -> MemoryTable -> ArrayData a -> Bool -> Int -> IO Bool
malloc !ctx (MemoryTable _ !ref) !ad !frozen !n = do
  mt <- modifyMVar ref $ \ct -> blocking $ do
   case IM.lookup (contextId ctx) ct of
           Nothing -> trace "malloc/context not found" $ insertContext ctx ct
           Just mt -> return (ct, mt)
  blocking $ MC.malloc mt ad frozen n

-- Explicitly free an array in the MemoryCache. Has the same properties as
-- `Data.Array.Accelerate.Array.Memory.Cache.free`
free :: PrimElt a b => Context -> MemoryTable -> ArrayData a -> IO ()
free !ctx (MemoryTable _ !ref) !arr = withMVar ref $ \ct ->
  case IM.lookup (contextId ctx) ct of
    Nothing -> message "free/context not found"
    Just mt -> MC.free (Proxy :: Proxy CRM) mt arr


-- Record an association between a host-side array and a device memory area that was
-- not allocated by accelerate. The device memory will NOT be freed by the memory
-- manager.
--
insertUnmanaged :: (PrimElt a b) => Context -> MemoryTable -> ArrayData a -> DevicePtr b -> IO ()
insertUnmanaged !ctx (MemoryTable _ !ref) !arr !ptr = do
  mt <- modifyMVar ref $ \ct -> blocking $ do
   case IM.lookup (contextId ctx) ct of
           Nothing  -> trace "insertUnmanaged/context not found" $ insertContext ctx ct
           Just mt -> return (ct, mt)
  blocking $ MC.insertUnmanaged mt arr ptr

insertContext :: Context -> IntMap (MC.MemoryCache DevicePtr (Maybe Event)) -> CRM (IntMap (MC.MemoryCache DevicePtr (Maybe Event)), MC.MemoryCache DevicePtr (Maybe Event))
insertContext ctx ct = do
   mt <- MC.new (\p -> bracket_ (push ctx) pop (CUDA.free p))
   return (IM.insert (contextId ctx) mt ct, mt)


-- Removing entries
-- ----------------

-- Initiate garbage collection and finalise any arrays that have been marked as
-- unreachable.
--
reclaim :: MemoryTable -> IO ()
reclaim (MemoryTable _ ref) = withMVar ref (blocking . mapM_ MC.reclaim . IM.elems)

-- Miscellaneous
-- -------------

{-# INLINE blocking #-}
blocking :: CRM a -> IO a
blocking = flip runReaderT Nothing

{-# INLINE streaming #-}
streaming :: Maybe Stream -> CRM a -> IO a
streaming = flip runReaderT

-- Debug
-- -----

{-# INLINE trace #-}
trace :: MonadIO m => String -> m a -> m a
trace msg next = message msg >> next

{-# INLINE message #-}
message :: MonadIO m => String -> m ()
message msg = liftIO $ D.traceIO D.dump_gc ("gc: " ++ msg)
