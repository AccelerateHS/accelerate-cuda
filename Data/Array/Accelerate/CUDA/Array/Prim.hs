{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Array.Prim
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Prim (

  DevicePtrs, HostPtrs,

  mallocArray, useArray, useArrayAsync, indexArray, copyArray, peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync, marshalDevicePtrs, marshalArrayData, marshalTextureData,
  devicePtrsOfArrayData, advancePtrsOfArrayData

) where

-- libraries
#if MIN_VERSION_base(4,6,0)
import Prelude                                          hiding ( lookup )
#else
import Prelude                                          hiding ( catch, lookup )
#endif
import Data.Int
import Data.Word
import Data.Maybe
import Data.Functor
import Data.Typeable
import Control.Monad
import Control.Exception
import System.Mem.StableName
import Foreign.Ptr
import Foreign.Storable
import Foreign.Marshal.Alloc
import Foreign.CUDA.Driver.Error
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Driver.Stream             as CUDA
import qualified Foreign.CUDA.Driver.Texture            as CUDA

-- friends
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.CUDA.Array.Table
import qualified Data.Array.Accelerate.CUDA.Debug       as D

#include "accelerate.h"


-- Device array representation
-- ---------------------------

type family DevicePtrs e :: *
type family HostPtrs   e :: *

type instance DevicePtrs () = ()
type instance HostPtrs   () = ()

#define primArrayElt(ty)                                                      \
type instance DevicePtrs ty = CUDA.DevicePtr ty ;                             \
type instance HostPtrs   ty = CUDA.HostPtr   ty ;                             \

primArrayElt(Int)
primArrayElt(Int8)
primArrayElt(Int16)
primArrayElt(Int32)
primArrayElt(Int64)

primArrayElt(Word)
primArrayElt(Word8)
primArrayElt(Word16)
primArrayElt(Word32)
primArrayElt(Word64)

-- FIXME:
-- CShort
-- CUShort
-- CInt
-- CUInt
-- CLong
-- CULong
-- CLLong
-- CULLong

primArrayElt(Float)
primArrayElt(Double)

-- FIXME:
-- CFloat
-- CDouble

type instance HostPtrs   Bool = CUDA.HostPtr   Word8
type instance DevicePtrs Bool = CUDA.DevicePtr Word8

primArrayElt(Char)

-- FIXME:
-- CChar
-- CSChar
-- CUChar

type instance DevicePtrs (a,b) = (DevicePtrs a, DevicePtrs b)
type instance HostPtrs   (a,b) = (HostPtrs   a, HostPtrs   b)


-- Texture References
-- ------------------

-- This representation must match the code generator's understanding of how to
-- utilise the texture cache.
--
class TextureData a where
  format :: a -> (CUDA.Format, Int)

instance TextureData Int8   where format _ = (CUDA.Int8,   1)
instance TextureData Int16  where format _ = (CUDA.Int16,  1)
instance TextureData Int32  where format _ = (CUDA.Int32,  1)
instance TextureData Int64  where format _ = (CUDA.Int32,  2)
instance TextureData Word8  where format _ = (CUDA.Word8,  1)
instance TextureData Word16 where format _ = (CUDA.Word16, 1)
instance TextureData Word32 where format _ = (CUDA.Word32, 1)
instance TextureData Word64 where format _ = (CUDA.Word32, 2)
instance TextureData Float  where format _ = (CUDA.Float,  1)
instance TextureData Double where format _ = (CUDA.Int32,  2)
instance TextureData Bool   where format _ = (CUDA.Word8,  1)
#if   SIZEOF_HSINT == 4
instance TextureData Int    where format _ = (CUDA.Int32,  1)
instance TextureData Word   where format _ = (CUDA.Word32, 1)
#elif SIZEOF_HSINT == 8
instance TextureData Int    where format _ = (CUDA.Int32,  2)
instance TextureData Word   where format _ = (CUDA.Word32, 2)
#else
instance TextureData Int    where
  format _ =
    case sizeOf (undefined::Int) of
      4 -> (CUDA.Int32, 1)
      8 -> (CUDA.Int32, 2)
instance TextureData Word   where
  format _ =
    case sizeOf (undefined::Word) of
      4 -> (CUDA.Word32, 1)
      8 -> (CUDA.Word32, 2)
#endif
#if SIZEOF_HSCHAR == 4
instance TextureData Char   where format _ = (CUDA.Word32, 1)
#else
instance TextureData Char   where
  format _ =
    case sizeOf (undefined::Char) of
         4 -> (CUDA.Word32, 1)
#endif


-- Primitive array operations
-- --------------------------

-- Allocate a device-side array associated with the given host array. If the
-- allocation fails due to a lack of memory, run the garbage collector to
-- release any inaccessible arrays and try again.
--
mallocArray
    :: forall e a. (ArrayElt e, DevicePtrs e ~ CUDA.DevicePtr a, Typeable e, Typeable a, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> IO ()
mallocArray !ctx !mt !ad !n0 = do
  let !n = 1 `max` n0
  exists <- isJust <$> (lookup ctx mt ad :: IO (Maybe (CUDA.DevicePtr a)))
  unless exists $ do
    message $ "mallocArray: " ++ showBytes (n * sizeOf (undefined::a))
    ptr <- CUDA.mallocArray n `catch` \(e :: CUDAException) ->
      case e of
        ExitCode OutOfMemory -> reclaim mt >> CUDA.mallocArray n
        _                    -> throwIO e
    insert ctx mt ad (ptr :: CUDA.DevicePtr a)


-- A combination of 'mallocArray' and 'pokeArray' to allocate space on the
-- device and upload an existing array. This is specialised because if the host
-- array is shared on the heap, we do not need to do anything.
--
useArray
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable e, Typeable a, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> IO ()
useArray !ctx !mt !ad !n0 =
  let src = ptrsOfArrayData ad
      !n  = 1 `max` n0
  in do
    exists <- isJust <$> (lookup ctx mt ad :: IO (Maybe (CUDA.DevicePtr a)))
    unless exists $ do
      message $ "useArray/malloc: " ++ showBytes (n * sizeOf (undefined::a))
      dst <- CUDA.mallocArray n `catch` \(e :: CUDAException) ->
        case e of
          ExitCode OutOfMemory -> reclaim mt >> CUDA.mallocArray n
          _                    -> throwIO e
      CUDA.pokeArray n src dst
      insert ctx mt ad dst


useArrayAsync
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable e, Typeable a, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> Maybe CUDA.Stream
    -> IO ()
useArrayAsync !ctx !mt !ad !n0 !ms =
  let src = CUDA.HostPtr (ptrsOfArrayData ad)
      !n  = 1 `max` n0
  in do
    exists <- isJust <$> (lookup ctx mt ad :: IO (Maybe (CUDA.DevicePtr a)))
    unless exists $ do
      message $ "useArrayAsync/malloc: " ++ showBytes (n * sizeOf (undefined::a))
      dst <- CUDA.mallocArray n `catch` \(e :: CUDAException) ->
        case e of
          ExitCode OutOfMemory -> reclaim mt >> CUDA.mallocArray n
          _                    -> throwIO e
      CUDA.pokeArrayAsync n src dst ms
      insert ctx mt ad dst


-- Read a single element from an array at the given row-major index
--
indexArray
    :: forall e a. (ArrayElt e, DevicePtrs e ~ CUDA.DevicePtr a, Typeable e, Typeable a, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> IO a
indexArray !ctx !mt !ad !i =
  alloca                            $ \dst ->
  devicePtrsOfArrayData ctx mt ad >>= \src -> do
    message $ "indexArray: " ++ showBytes (sizeOf (undefined::a))
    CUDA.peekArray 1 (src `CUDA.advanceDevPtr` i) dst
    peek dst


-- Copy data between two device arrays. The operation is asynchronous with
-- respect to the host, but will never overlap kernel execution.
--
copyArray
    :: forall e a b. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr b, Typeable a, Typeable b, Typeable e, Storable b)
    => Context
    -> MemoryTable
    -> ArrayData e              -- source array
    -> ArrayData e              -- destination array
    -> Int                      -- number of array elements
    -> IO ()
copyArray !ctx !mt !from !to !n = do
  message $ "copyArrayAsync: " ++ showBytes (n * sizeOf (undefined :: b))
  src <- devicePtrsOfArrayData ctx mt from
  dst <- devicePtrsOfArrayData ctx mt to
  CUDA.copyArrayAsync n src dst


-- Copy data from the device into the associated Accelerate host-side array
--
peekArray
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable a, Typeable e, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> IO ()
peekArray !ctx !mt !ad !n =
  devicePtrsOfArrayData ctx mt ad >>= \src -> do
    message $ "peekArray: " ++ showBytes (n * sizeOf (undefined :: a))
    CUDA.peekArray n src (ptrsOfArrayData ad)

peekArrayAsync
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable a, Typeable e, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> Maybe CUDA.Stream
    -> IO ()
peekArrayAsync !ctx !mt !ad !n !st =
  devicePtrsOfArrayData ctx mt ad >>= \src -> do
    message $ "peekArrayAsync: " ++ showBytes (n * sizeOf (undefined :: a))
    CUDA.peekArrayAsync n src (CUDA.HostPtr $ ptrsOfArrayData ad) st


-- Copy data from an Accelerate array into the associated device array
--
pokeArray
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable a, Typeable e, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> IO ()
pokeArray !ctx !mt !ad !n =
  devicePtrsOfArrayData ctx mt ad >>= \dst -> do
    message $ "pokeArrayAsync: " ++ showBytes (n * sizeOf (undefined :: a))
    CUDA.pokeArray n (ptrsOfArrayData ad) dst

pokeArrayAsync
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable a, Typeable e, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> Maybe CUDA.Stream
    -> IO ()
pokeArrayAsync !ctx !mt !ad !n !st =
  devicePtrsOfArrayData ctx mt ad >>= \dst -> do
    message $ "pokeArrayAsync: " ++ showBytes (n * sizeOf (undefined :: a))
    CUDA.pokeArrayAsync n (CUDA.HostPtr $ ptrsOfArrayData ad) dst st


-- Marshal device pointers to arguments that can be passed to kernel invocation
--
marshalDevicePtrs
    :: (ArrayElt e, DevicePtrs e ~ CUDA.DevicePtr b)
    => ArrayData e
    -> DevicePtrs e
    -> CUDA.FunParam
marshalDevicePtrs !_ !ptr = CUDA.VArg ptr


-- Wrap a device pointer corresponding corresponding to a host-side array into
-- arguments that can be passed to a kernel upon invocation
--
marshalArrayData
    :: (ArrayElt e, DevicePtrs e ~ CUDA.DevicePtr b, Typeable b, Typeable e)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> IO CUDA.FunParam
marshalArrayData !ctx !mt !ad = marshalDevicePtrs ad <$> devicePtrsOfArrayData ctx mt ad


-- Bind device memory to the given texture reference, setting appropriate type
--
marshalTextureData
    :: forall a e. (ArrayElt e, DevicePtrs e ~ CUDA.DevicePtr a, Typeable a, Typeable e, Storable a, TextureData a)
    => Context
    -> MemoryTable
    -> ArrayData e              -- host array
    -> Int                      -- number of elements
    -> CUDA.Texture             -- texture reference to bind array to
    -> IO ()
marshalTextureData !ctx !mt !ad !n !tex =
  let (fmt, c) = format (undefined :: a)
  in  devicePtrsOfArrayData ctx mt ad >>= \ptr -> do
        CUDA.setFormat tex fmt c
        CUDA.bind tex ptr (fromIntegral $ n * sizeOf (undefined :: a))


-- Lookup the device memory associated with our host array
--
devicePtrsOfArrayData
    :: (ArrayElt e, DevicePtrs e ~ CUDA.DevicePtr b, Typeable e, Typeable b)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> IO (DevicePtrs e)
devicePtrsOfArrayData !ctx !mt !ad = do
  mv <- lookup ctx mt ad
  case mv of
    Just v  -> return v
    Nothing -> do
      sn <- makeStableName ad
      INTERNAL_ERROR(error) "devicePtrsOfArrayData" $ "lost device memory #" ++ show (hashStableName sn)


-- Advance device pointers by a given number of elements
--
advancePtrsOfArrayData
    :: (ArrayElt e, DevicePtrs e ~ CUDA.DevicePtr b, Storable b)
    => Int
    -> ArrayData e
    -> DevicePtrs e
    -> DevicePtrs e
advancePtrsOfArrayData !n !_ !ptr = CUDA.advanceDevPtr ptr n


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

