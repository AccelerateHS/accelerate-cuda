{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE InstanceSigs        #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
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
  MemoryTable, new, lookup, malloc, free, insertUnmanaged, reclaim

) where

import Prelude                                                  hiding ( lookup )
import Data.Functor
import Data.IntMap.Strict                                       ( IntMap )
import Data.Typeable                                            ( Typeable )
import Control.Concurrent.MVar                                  ( MVar, newMVar, withMVar, modifyMVar )
import Control.Exception                                        ( catch, throwIO )
import Foreign.Ptr                                              ( Ptr, ptrToIntPtr )
import Foreign.Storable                                         ( Storable, sizeOf )
import Foreign.CUDA.Ptr                                         ( DevicePtr )

import Foreign.CUDA.Driver.Error
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Data.IntMap.Strict                             as IM

import Data.Array.Accelerate.Array.Data                         ( ArrayData, MutableArrayData, ptrsOfArrayData
                                                                , ArrayPtrs, ArrayElt )
import Data.Array.Accelerate.Array.Memory                       ( RemoteMemory )
import Data.Array.Accelerate.CUDA.Context                       ( Context, deviceContext )
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

instance RemoteMemory DevicePtr where
  malloc n = fmap Just (CUDA.mallocArray n) `catch` \(e :: CUDAException) ->
             case e of
               ExitCode OutOfMemory -> return Nothing
               _                    -> trace ("malloc failed with unknown error for: " ++ show n)
                                     $ throwIO e

  free = trace "free/explicit free" . CUDA.free

  poke :: forall a e. (ArrayElt e, Storable a, ArrayPtrs e ~ Ptr a) => Int -> DevicePtr a -> ArrayData e -> IO ()
  poke n dst ad = transfer "poke" (n * sizeOf (undefined :: a)) $
      CUDA.pokeArray n (ptrsOfArrayData ad) dst

  peek :: forall a e. (ArrayElt e, Storable a, ArrayPtrs e ~ Ptr a) => Int -> DevicePtr a -> MutableArrayData e -> IO ()
  peek n src ad = transfer "peek" (n * sizeOf (undefined :: a)) $
      CUDA.peekArray n src (ptrsOfArrayData ad)

  castPtr = CUDA.castDevPtr

  totalMem _ = fst <$> CUDA.getMemInfo

  availableMem _ = snd <$> CUDA.getMemInfo

  chunkSize _ = 1024

-- Create a MemoryTable.
new :: IO MemoryTable
new = trace "initialise CUDA memory table" $ newMVar IM.empty

-- Look for the device pointer corresponding to a given host-side array.
--
lookup :: (Typeable a, Typeable b) => Context -> MemoryTable -> ArrayData a -> IO (Maybe (DevicePtr b))
lookup !ctx !ref !arr = withMVar ref $ \ct ->
  case IM.lookup (contextId ctx) ct of
    Nothing -> trace "lookup/context not found" $ return Nothing
    Just mt -> MT.lookup mt arr


-- Allocate a new device array to be associated with the given host-side array.
-- Has the same properties as `Data.Array.Accelerate.Array.Memory.Table.malloc`
malloc :: forall a b. (Typeable a, Typeable b, Storable b) => Context -> MemoryTable -> ArrayData a -> Int -> IO (DevicePtr b)
malloc !ctx !ref !ad !n = do
  mt <- modifyMVar ref $ \ct -> do
   case IM.lookup (contextId ctx) ct of
           Nothing  -> trace "malloc/context not found" $ do
             mt <- MT.new
             return (IM.insert (contextId ctx) mt ct, mt)
           Just mt -> return (ct, mt)
  MT.malloc mt ad n

-- Explicitly free an array in the MemoryTable. Has the same properties as
-- `Data.Array.Accelerate.Array.Memory.Table.free`
free :: Typeable a => Context -> MemoryTable -> ArrayData a -> IO ()
free !ctx !ref !arr = withMVar ref $ \ct ->
  case IM.lookup (contextId ctx) ct of
    Nothing -> message "free/context not found"
    Just mt -> MT.free mt arr


-- Record an association between a host-side array and a device memory area that was
-- not allocated by accelerate. The device memory will NOT be freed when the host
-- array is garbage collected.
--
insertUnmanaged :: (Typeable a, Typeable b) => Context -> MemoryTable -> ArrayData a -> DevicePtr b -> IO ()
insertUnmanaged !ctx !ref !arr !ptr = do
  mt <- modifyMVar ref $ \ct -> do
   case IM.lookup (contextId ctx) ct of
           Nothing  -> trace "insertUnmanaged/context not found" $ do
             mt <- MT.new
             return (IM.insert (contextId ctx) mt ct, mt)
           Just mt -> return (ct, mt)
  MT.insertUnmanaged mt arr ptr


-- Removing entries
-- ----------------

-- Initiate garbage collection and finalise any arrays that have been marked as
-- unreachable.
--
reclaim :: MemoryTable -> IO ()
reclaim ref = withMVar ref (mapM_ MT.reclaim . IM.elems)

-- Miscellaneous
-- -------------

{-# INLINE contextId #-}
contextId :: Context -> ContextId
contextId !ctx =
  let CUDA.Context !p   = deviceContext ctx
  in fromIntegral (ptrToIntPtr p)

-- Debug
-- -----

{-# INLINE showBytes #-}
showBytes :: Int -> String
showBytes x = D.showFFloatSIBase (Just 0) 1024 (fromIntegral x :: Double) "B"

{-# INLINE trace #-}
trace :: String -> IO a -> IO a
trace msg next = message msg >> next

{-# INLINE message #-}
message :: String -> IO ()
message msg = D.traceIO D.dump_gc ("gc: " ++ msg)

{-# INLINE transfer #-}
transfer :: String -> Int -> IO () -> IO ()
transfer name bytes action
  = let showRate x t        = D.showFFloatSIBase (Just 3) 1024 (fromIntegral x / t) "B/s"
        msg gpuTime cpuTime = "gc: " ++ name ++ ": "
                                     ++ showBytes bytes ++ " @ " ++ showRate bytes gpuTime ++ ", "
                                     ++ D.elapsed gpuTime cpuTime
    in
    D.timed D.dump_gc msg Nothing action
