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
  MemoryTable, new, lookup, malloc, free, insertUnmanaged, reclaim,

  CRM, ContextId, contextId

) where

import Data.Functor
import Data.Proxy
import Data.IntMap.Strict                                       ( IntMap )
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, modifyMVar )
import Control.Exception                                        ( catch, throwIO, bracket_ )
import Control.Monad.IO.Class                                   ( MonadIO, liftIO )
import Control.Monad.Trans.Reader
import Foreign.Ptr                                              ( ptrToIntPtr )
import Foreign.Storable                                         ( Storable, sizeOf )
import Foreign.CUDA.Ptr                                         ( DevicePtr )
import Prelude                                                  hiding ( lookup )

import Foreign.CUDA.Driver.Error
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Data.IntMap.Strict                             as IM

import Data.Array.Accelerate.Array.Data                         ( ArrayData, ptrsOfArrayData )
import Data.Array.Accelerate.Lifetime                           ( unsafeGetValue )
import Data.Array.Accelerate.Array.Memory                       ( RemoteMemory, PrimElt )
import Data.Array.Accelerate.CUDA.Context                       ( Context(..), push, pop )
import Data.Array.Accelerate.CUDA.Execute.Stream                ( Stream )
import qualified Data.Array.Accelerate.CUDA.Debug               as D
import qualified Data.Array.Accelerate.Array.Memory             as M
import qualified Data.Array.Accelerate.Array.Memory.Table       as MT

-- We leverage the memory table from the accelerate base package. However, we
-- actually need multiple tables. This is because every pointer has an
-- associated CUDA context. We could have pair every DevicePtr with its
-- context and just have a single table, but the MemoryTable API in the base
-- package assumes that remote pointers can be re-used, something that would
-- not be true for pointers allocated under different contexts.
type MemoryTable = MVar (IntMap (MT.MemoryTable DevicePtr))

-- Contexts
--
type ContextId = Int

-- Referencing arrays
-- ------------------

type CRM = ReaderT (Maybe Stream) IO

instance RemoteMemory CRM where

  type RemotePointer CRM = DevicePtr

  malloc n = ReaderT . const $ fmap Just (CUDA.mallocArray n) `catch` \(e :: CUDAException) ->
             case e of
               ExitCode OutOfMemory -> return Nothing
               _                    -> trace ("malloc failed with unknown error for: " ++ show n)
                                     $ throwIO e

  free = ReaderT . const . trace "free/explicit free" . CUDA.free

  poke n dst ad = ReaderT $ \ms -> transfer "poke" (n * sizeOfPtr dst) $
      CUDA.pokeArrayAsync n (CUDA.HostPtr $ ptrsOfArrayData ad) dst ms

  peek n src ad = ReaderT $ \ms -> transfer "peek" (n * sizeOfPtr src) $
      CUDA.peekArrayAsync n src (CUDA.HostPtr $ ptrsOfArrayData ad) ms

  castPtr _ = CUDA.castDevPtr

  totalMem = ReaderT . const $ snd <$> CUDA.getMemInfo

  availableMem = ReaderT . const $ fst <$> CUDA.getMemInfo

  chunkSize = return 1024

-- Create a MemoryTable.
new :: IO MemoryTable
new = trace "initialise CUDA memory table" $ newMVar IM.empty

-- Look for the device pointer corresponding to a given host-side array.
--
lookup :: PrimElt a b => Context -> MemoryTable -> ArrayData a -> IO (Maybe (DevicePtr b))
lookup !ctx !ref !arr = withMVar ref $ \ct ->
  case IM.lookup (contextId ctx) ct of
    Nothing -> trace "lookup/context not found" $ return Nothing
    Just mt -> MT.lookup mt arr


-- Allocate a new device array to be associated with the given host-side array.
-- Has the same properties as `Data.Array.Accelerate.Array.Memory.Table.malloc`
malloc :: forall a b. PrimElt a b => Context -> MemoryTable -> ArrayData a -> Int -> IO (DevicePtr b)
malloc !ctx !ref !ad !n = do
  mt <- modifyMVar ref $ \ct -> blocking $ do
   case IM.lookup (contextId ctx) ct of
           Nothing  -> trace "malloc/context not found" $ insertContext ctx ct
           Just mt -> return (ct, mt)
  mp <- blocking $ MT.malloc mt ad n :: IO (Maybe (DevicePtr b))
  case mp of
    Nothing -> throwIO (ExitCode OutOfMemory)
    Just p  -> return p

-- Explicitly free an array in the MemoryTable. Has the same properties as
-- `Data.Array.Accelerate.Array.Memory.Table.free`
free :: PrimElt a b => Context -> MemoryTable -> ArrayData a -> IO ()
free !ctx !ref !arr = withMVar ref $ \ct ->
  case IM.lookup (contextId ctx) ct of
    Nothing -> message "free/context not found"
    Just mt -> MT.free (Proxy :: Proxy CRM) mt arr


-- Record an association between a host-side array and a device memory area that was
-- not allocated by accelerate. The device memory will NOT be freed when the host
-- array is garbage collected.
--
insertUnmanaged :: PrimElt a b => Context -> MemoryTable -> ArrayData a -> DevicePtr b -> IO ()
insertUnmanaged !ctx !ref !arr !ptr = do
  mt <- modifyMVar ref $ \ct -> blocking $ do
   case IM.lookup (contextId ctx) ct of
           Nothing  -> trace "insertUnmanaged/context not found" $ insertContext ctx ct
           Just mt -> return (ct, mt)
  blocking $ MT.insertUnmanaged mt arr ptr

insertContext :: Context -> IntMap (MT.MemoryTable DevicePtr) -> CRM (IntMap (MT.MemoryTable DevicePtr), MT.MemoryTable DevicePtr)
insertContext ctx ct = do
   mt <- MT.new (\p -> bracket_ (push ctx) pop (CUDA.free p))
   return (IM.insert (contextId ctx) mt ct, mt)


-- Removing entries
-- ----------------

-- Initiate garbage collection and finalise any arrays that have been marked as
-- unreachable.
--
reclaim :: MemoryTable -> IO ()
reclaim ref = withMVar ref (blocking . mapM_ MT.reclaim . IM.elems)

-- Miscellaneous
-- -------------

{-# INLINE contextId #-}
contextId :: Context -> ContextId
contextId !ctx =
  let CUDA.Context !p = unsafeGetValue (deviceContext ctx)
  in fromIntegral (ptrToIntPtr p)

{-# INLINE sizeOfPtr #-}
sizeOfPtr :: forall a. Storable a => DevicePtr a -> Int
sizeOfPtr _ = sizeOf (undefined :: a)

{-# INLINE blocking #-}
blocking :: CRM a -> IO a
blocking = flip runReaderT Nothing

-- Debug
-- -----

{-# INLINE showBytes #-}
showBytes :: Int -> String
showBytes x = D.showFFloatSIBase (Just 0) 1024 (fromIntegral x :: Double) "B"

{-# INLINE trace #-}
trace :: MonadIO m => String -> m a -> m a
trace msg next = message msg >> next

{-# INLINE message #-}
message :: MonadIO m => String -> m ()
message msg = liftIO $ D.traceIO D.dump_gc ("gc: " ++ msg)

{-# INLINE transfer #-}
transfer :: String -> Int -> IO () -> IO ()
transfer name bytes action
  = let showRate x t        = D.showFFloatSIBase (Just 3) 1024 (fromIntegral x / t) "B/s"
        msg gpuTime cpuTime = "gc: " ++ name ++ ": "
                                     ++ showBytes bytes ++ " @ " ++ showRate bytes gpuTime ++ ", "
                                     ++ D.elapsed gpuTime cpuTime
    in
    D.timed D.dump_gc msg Nothing action
