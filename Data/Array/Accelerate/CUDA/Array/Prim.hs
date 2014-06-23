{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE TypeFamilies        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Array.Prim
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Prim (

  DevicePtrs, HostPtrs,

  mallocArray, indexArray,
  useArray,  useArrayAsync, useArraySlice,
  useDevicePtrs,
  copyArray, copyArrayPeer, copyArrayPeerAsync,
  peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync,
  marshalDevicePtrs, marshalArrayData, marshalTextureData,
  devicePtrsOfArrayData, advancePtrsOfArrayData

) where

-- libraries
import Prelude                                          hiding ( lookup )
import Data.Int
import Data.Word
import Data.Maybe
import Data.Functor
import Data.Typeable
import Control.Monad
import Language.Haskell.TH
import System.Mem.StableName
import Foreign.CUDA.Ptr                                 ( plusDevPtr )
import Foreign.Ptr
import Foreign.C.Types
import Foreign.Storable
import Foreign.Marshal.Alloc                            ( alloca )
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Driver.Stream             as CUDA
import qualified Foreign.CUDA.Driver.Texture            as CUDA

-- friends
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Array.Slice          ( TransferDesc(..), blocksOf )
import Data.Array.Accelerate.CUDA.Array.Table
import qualified Data.Array.Accelerate.CUDA.Debug       as D


-- Device array representation
-- ---------------------------

type family DevicePtrs e :: *
type family HostPtrs   e :: *

type instance DevicePtrs () = ()
type instance HostPtrs   () = ()

#define primArrayEltAs(ty,as)                                                 \
type instance DevicePtrs ty = CUDA.DevicePtr as ;                             \
type instance HostPtrs   ty = CUDA.HostPtr   as ;                             \

#define primArrayElt(ty) primArrayEltAs(ty,ty)

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

primArrayEltAs(CShort,  Int16)
primArrayEltAs(CInt,    Int32)
primArrayEltAs(CLong,   HTYPE_LONG)
primArrayEltAs(CLLong,  Int64)
primArrayEltAs(CUShort, Word16)
primArrayEltAs(CUInt,   Word32)
primArrayEltAs(CULong,  HTYPE_UNSIGNED_LONG)
primArrayEltAs(CULLong, Word64)

primArrayElt(Float)
primArrayElt(Double)
primArrayEltAs(CFloat,  Float)
primArrayEltAs(CDouble, Double)

primArrayElt(Char)
primArrayEltAs(CChar,  Int8)
primArrayEltAs(CSChar, Int8)
primArrayEltAs(CUChar, Word8)

primArrayEltAs(Bool, Word8)

type instance DevicePtrs (a,b) = (DevicePtrs a, DevicePtrs b)
type instance HostPtrs   (a,b) = (HostPtrs   a, HostPtrs   b)


-- Texture References
-- ------------------

-- This representation must match the code generator's understanding of how to
-- utilise the texture cache.
--
class TextureData a where
  format :: a -> (CUDA.Format, Int)

instance TextureData Int8    where format _ = (CUDA.Int8,   1)
instance TextureData Int16   where format _ = (CUDA.Int16,  1)
instance TextureData Int32   where format _ = (CUDA.Int32,  1)
instance TextureData Int64   where format _ = (CUDA.Int32,  2)
instance TextureData Word8   where format _ = (CUDA.Word8,  1)
instance TextureData Word16  where format _ = (CUDA.Word16, 1)
instance TextureData Word32  where format _ = (CUDA.Word32, 1)
instance TextureData Word64  where format _ = (CUDA.Word32, 2)
instance TextureData Float   where format _ = (CUDA.Float,  1)
instance TextureData Double  where format _ = (CUDA.Int32,  2)
instance TextureData Bool    where format _ = (CUDA.Word8,  1)
instance TextureData CShort  where format _ = (CUDA.Int16,  1)
instance TextureData CUShort where format _ = (CUDA.Word16, 1)
instance TextureData CInt    where format _ = (CUDA.Int32,  1)
instance TextureData CUInt   where format _ = (CUDA.Word32, 1)
instance TextureData CLLong  where format _ = (CUDA.Int32,  2)
instance TextureData CULLong where format _ = (CUDA.Word32, 2)
instance TextureData CFloat  where format _ = (CUDA.Float,  1)
instance TextureData CDouble where format _ = (CUDA.Int32,  2)
instance TextureData CChar   where format _ = (CUDA.Int8,   1)
instance TextureData CSChar  where format _ = (CUDA.Int8,   1)
instance TextureData CUChar  where format _ = (CUDA.Word8,  1)
instance TextureData Char    where format _ = (CUDA.Word32, 1)

$( runQ [d| instance TextureData Int    where format _ = (CUDA.Int32,  sizeOf (undefined::Int)    `div` 4) |] )
$( runQ [d| instance TextureData Word   where format _ = (CUDA.Word32, sizeOf (undefined::Word)   `div` 4) |] )
$( runQ [d| instance TextureData CLong  where format _ = (CUDA.Int32,  sizeOf (undefined::CLong)  `div` 4) |] )
$( runQ [d| instance TextureData CULong where format _ = (CUDA.Word32, sizeOf (undefined::CULong) `div` 4) |] )


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
  let !n        = 1 `max` n0
      !bytes    = n * sizeOf (undefined :: a)
  exists <- isJust <$> (lookup ctx mt ad :: IO (Maybe (CUDA.DevicePtr a)))
  unless exists $ do
    message $ "mallocArray: " ++ showBytes bytes
    _ <- malloc ctx mt ad n     :: IO (CUDA.DevicePtr a)
    return ()

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
  let src    = ptrsOfArrayData ad
      !n     = 1 `max` n0
      !bytes = n * sizeOf (undefined :: a)
  in do
    exists <- isJust <$> (lookup ctx mt ad :: IO (Maybe (CUDA.DevicePtr a)))
    unless exists $ do
      dst <- malloc ctx mt ad n
      transfer "useArray/malloc" bytes $ CUDA.pokeArray n src dst

-- A combination of 'mallocArray' and 'pokeArray' to allocate space on the
-- device and upload an existing array. This is specialised because if the host
-- array is shared on the heap, we do not need to do anything.
--
useArraySlice
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable e, Typeable a, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> ArrayData e
    -> TransferDesc
    -> IO ()
useArraySlice !ctx !mt !ad_host !ad_dev !tdesc =
  let src    = ptrsOfArrayData ad_host
      k      = sizeOf (undefined :: a)
  in do
    maybe_dst <- lookup ctx mt ad_dev :: IO (Maybe (CUDA.DevicePtr a))
    case maybe_dst of
      Just dst -> 
        sequence_ 
          [ transfer "useArraySlice/malloc" (k * size) $ CUDA.pokeArray size (plusPtr src (k * src_offset)) (plusDevPtr dst (k * dst_offset))
          | (src_offset, dst_offset, size) <- blocksOf tdesc]
      Nothing -> 
        do dst <- malloc ctx mt ad_dev (k * nblocks tdesc * blocksize tdesc) :: IO (CUDA.DevicePtr a) 
           sequence_ 
             [ transfer "useArraySlice/malloc" (k * size) $ CUDA.pokeArray size (plusPtr src (k * src_offset)) (plusDevPtr dst (k * dst_offset))
             | (src_offset, dst_offset, size) <- blocksOf tdesc]


useArrayAsync
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable e, Typeable a, Storable a)
    => Context
    -> MemoryTable
    -> ArrayData e
    -> Int
    -> Maybe CUDA.Stream
    -> IO ()
useArrayAsync !ctx !mt !ad !n0 !ms =
  let src    = CUDA.HostPtr (ptrsOfArrayData ad)
      !n     = 1 `max` n0
      !bytes = n * sizeOf (undefined :: a)
  in do
    exists <- isJust <$> (lookup ctx mt ad :: IO (Maybe (CUDA.DevicePtr a)))
    unless exists $ do
      dst <- malloc ctx mt ad n
      transfer "useArrayAsync/malloc" bytes $ CUDA.pokeArrayAsync n src dst ms


useDevicePtrs
    :: forall e a. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr a, Typeable e, Typeable a, Storable a)
    => Context
    -> MemoryTable
    -> DevicePtrs e
    -> Int
    -> IO (ArrayData e)
useDevicePtrs !ctx !mt !ptr !n0 =
  let !n         = 1 `max` n0
      !bytes     = n * sizeOf (undefined :: a)
      (adata, _) = runArrayData $ (,undefined) `fmap` newArrayData n
  in do
    message $ "useDevicePtrs: " ++ showBytes bytes
    insertRemote ctx mt adata ptr
    return adata


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
  src <- devicePtrsOfArrayData ctx mt from
  dst <- devicePtrsOfArrayData ctx mt to
  transfer "copyArrayAsync" (n * sizeOf (undefined :: b)) $
    CUDA.copyArrayAsync n src dst


-- Copy data between two device arrays that exist in different contexts and/or
-- devices.
--
copyArrayPeer
    :: forall e a b. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr b, Typeable a, Typeable b, Typeable e, Storable b)
    => MemoryTable
    -> ArrayData e -> Context   -- source array and context
    -> ArrayData e -> Context   -- destination array and context
    -> Int                      -- number of array elements
    -> IO ()
copyArrayPeer !mt !from !ctxSrc !to !ctxDst !n = do
  src <- devicePtrsOfArrayData ctxSrc mt from
  dst <- devicePtrsOfArrayData ctxDst mt to
  transfer "copyArrayPeer" (n * sizeOf (undefined :: b)) $
    CUDA.copyArrayPeer n src (deviceContext ctxSrc) dst (deviceContext ctxDst)

copyArrayPeerAsync
    :: forall e a b. (ArrayElt e, ArrayPtrs e ~ Ptr a, DevicePtrs e ~ CUDA.DevicePtr b, Typeable a, Typeable b, Typeable e, Storable b)
    => MemoryTable
    -> ArrayData e -> Context   -- source array and context
    -> ArrayData e -> Context   -- destination array and context
    -> Int                      -- number of array elements
    -> Maybe CUDA.Stream
    -> IO ()
copyArrayPeerAsync !mt !from !ctxSrc !to !ctxDst !n !st = do
  src <- devicePtrsOfArrayData ctxSrc mt from
  dst <- devicePtrsOfArrayData ctxDst mt to
  transfer "copyArrayPeerAsync" (n * sizeOf (undefined :: b)) $
    CUDA.copyArrayPeerAsync n src (deviceContext ctxSrc) dst (deviceContext ctxDst) st


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
  devicePtrsOfArrayData ctx mt ad >>= \src ->
    transfer "peekArray" (n * sizeOf (undefined :: a)) $
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
  devicePtrsOfArrayData ctx mt ad >>= \src ->
    transfer "peekArrayAsync" (n * sizeOf (undefined :: a)) $
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
  devicePtrsOfArrayData ctx mt ad >>= \dst ->
    transfer "pokeArray: " (n * sizeOf (undefined :: a)) $
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
  devicePtrsOfArrayData ctx mt ad >>= \dst ->
    transfer "pokeArrayAsync: " (n * sizeOf (undefined :: a)) $
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
      $internalError "devicePtrsOfArrayData" $ "lost device memory #" ++ show (hashStableName sn)


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

{-# INLINE transfer #-}
transfer :: String -> Int -> IO () -> IO ()
transfer name bytes action
  = let showRate x t        = D.showFFloatSIBase (Just 3) 1024 (fromIntegral x / t) "B/s"
        msg gpuTime cpuTime = "gc: " ++ name ++ ": "
                                     ++ showBytes bytes ++ " @ " ++ showRate bytes gpuTime ++ ", "
                                     ++ D.elapsed gpuTime cpuTime
    in
    D.timed D.dump_gc msg Nothing action

