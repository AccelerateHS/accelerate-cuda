{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Array.Data
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Data (

  -- * Array operations and representations
  mallocArray, indexArray, copyArray,
  useArray,  useArrayAsync,
  peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync,
  marshalArrayData, marshalTextureData, marshalDevicePtrs,
  devicePtrsOfArrayData, advancePtrsOfArrayData,

  -- * Garbage collection
  cleanupArrayData

) where

-- libraries
import Prelude                                          hiding ( fst, snd )
import Data.Label.PureM
import Control.Applicative
import Control.Monad.Trans

-- friends
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Sugar                ( Array(..), Shape, Elt, fromElt, toElt )
import Data.Array.Accelerate.Array.Representation       ( size, index )
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Array.Table
import qualified Data.Array.Accelerate.CUDA.Array.Prim  as Prim
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Driver.Stream             as CUDA
import qualified Foreign.CUDA.Driver.Texture            as CUDA

#include "accelerate.h"


-- Array Operations
-- ----------------

-- Garbage collection
--
cleanupArrayData :: CIO ()
cleanupArrayData = liftIO . reclaim =<< gets memoryTable

-- Array tuple extraction
--
fst :: ArrayData (a,b) -> ArrayData a
fst = fstArrayData

snd :: ArrayData (a,b) -> ArrayData b
snd = sndArrayData

-- CPP hackery to generate the cases where we dispatch to the worker function handling
-- elementary types.
--
#define mkPrimDispatch(dispatcher,worker)                                   \
; dispatcher ArrayEltRint    = worker                                       \
; dispatcher ArrayEltRint8   = worker                                       \
; dispatcher ArrayEltRint16  = worker                                       \
; dispatcher ArrayEltRint32  = worker                                       \
; dispatcher ArrayEltRint64  = worker                                       \
; dispatcher ArrayEltRword   = worker                                       \
; dispatcher ArrayEltRword8  = worker                                       \
; dispatcher ArrayEltRword16 = worker                                       \
; dispatcher ArrayEltRword32 = worker                                       \
; dispatcher ArrayEltRword64 = worker                                       \
; dispatcher ArrayEltRfloat  = worker                                       \
; dispatcher ArrayEltRdouble = worker                                       \
; dispatcher ArrayEltRbool   = worker                                       \
; dispatcher ArrayEltRchar   = worker                                       \
; dispatcher _               = error "mkPrimDispatcher: not primitive"


-- |Allocate a new device array to accompany the given host-side array.
--
mallocArray :: (Shape dim, Elt e) => Array dim e -> CIO ()
mallocArray (Array sh adata) = doMalloc =<< gets memoryTable
  where
    doMalloc mt = liftIO $ mallocR arrayElt adata
      where
        mallocR :: ArrayEltR e -> ArrayData e -> IO ()
        mallocR ArrayEltRunit             _  = return ()
        mallocR (ArrayEltRpair aeR1 aeR2) ad = mallocR aeR1 (fst ad) >> mallocR aeR2 (snd ad)
        mallocR aer                       ad = mallocPrim aer mt ad (size sh)
        --
        mallocPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(mallocPrim,Prim.mallocArray)


-- |Upload an existing array to the device
--
useArray :: (Shape dim, Elt e) => Array dim e -> CIO ()
useArray (Array sh adata) = doUse =<< gets memoryTable
  where
    doUse mt = liftIO $ useR arrayElt adata
      where
        useR :: ArrayEltR e -> ArrayData e -> IO ()
        useR ArrayEltRunit             _  = return ()
        useR (ArrayEltRpair aeR1 aeR2) ad = useR aeR1 (fst ad) >> useR aeR2 (snd ad)
        useR aer                       ad = usePrim aer mt ad (size sh)
        --
        usePrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(usePrim,Prim.useArray)

useArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> CIO ()
useArrayAsync (Array sh adata) ms = doUse =<< gets memoryTable
  where
    doUse mt = liftIO $ useR arrayElt adata
      where
        useR :: ArrayEltR e -> ArrayData e -> IO ()
        useR ArrayEltRunit             _  = return ()
        useR (ArrayEltRpair aeR1 aeR2) ad = useR aeR1 (fst ad) >> useR aeR2 (snd ad)
        useR aer                       ad = usePrim aer mt ad (size sh) ms
        --
        usePrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Int -> Maybe CUDA.Stream -> IO ()
        mkPrimDispatch(usePrim,Prim.useArrayAsync)


-- |Read a single element from an array at the given row-major index. This is a
-- synchronous operation.
--
indexArray :: (Shape dim, Elt e) => Array dim e -> dim -> CIO e
indexArray (Array sh adata) ix = doIndex =<< gets memoryTable
  where
    i          = index sh (fromElt ix)
    doIndex mt = toElt <$> (liftIO $ indexR arrayElt adata)
      where
        indexR :: ArrayEltR e -> ArrayData e -> IO e
        indexR ArrayEltRunit             _  = return ()
        indexR (ArrayEltRpair aeR1 aeR2) ad = (,) <$> indexR aeR1 (fst ad)
                                                  <*> indexR aeR2 (snd ad)
        --
        indexR ArrayEltRbool             ad = toBool <$> Prim.indexArray mt ad i
          where toBool 0 = False
                toBool _ = True
        --
        indexR ArrayEltRint              ad = Prim.indexArray mt ad i
        indexR ArrayEltRint8             ad = Prim.indexArray mt ad i
        indexR ArrayEltRint16            ad = Prim.indexArray mt ad i
        indexR ArrayEltRint32            ad = Prim.indexArray mt ad i
        indexR ArrayEltRint64            ad = Prim.indexArray mt ad i
        indexR ArrayEltRword             ad = Prim.indexArray mt ad i
        indexR ArrayEltRword8            ad = Prim.indexArray mt ad i
        indexR ArrayEltRword16           ad = Prim.indexArray mt ad i
        indexR ArrayEltRword32           ad = Prim.indexArray mt ad i
        indexR ArrayEltRword64           ad = Prim.indexArray mt ad i
        indexR ArrayEltRfloat            ad = Prim.indexArray mt ad i
        indexR ArrayEltRdouble           ad = Prim.indexArray mt ad i
        indexR ArrayEltRchar             ad = Prim.indexArray mt ad i


-- |Copy data between two device arrays. The operation is asynchronous with
-- respect to the host, but will never overlap kernel execution.
--
copyArray :: (Shape dim, Elt e) => Array dim e -> Array dim e -> CIO ()
copyArray (Array sh1 adata1) (Array sh2 adata2)
  = BOUNDS_CHECK(check) "copyArray" "shape mismatch" (sh1 == sh2)
  $ doCopy =<< gets memoryTable
  where
    doCopy mt = liftIO $ copyR arrayElt adata1 adata2
      where
        copyR :: ArrayEltR e -> ArrayData e -> ArrayData e -> IO ()
        copyR ArrayEltRunit             _   _   = return ()
        copyR (ArrayEltRpair aeR1 aeR2) ad1 ad2 = copyR aeR1 (fst ad1) (fst ad2) >>
                                                  copyR aeR2 (snd ad1) (snd ad2)
        copyR aer                       ad1 ad2 = copyPrim aer mt ad1 ad2 (size sh1)
        --
        copyPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(copyPrim,Prim.copyArray)


-- Copy data from the device into the associated Accelerate host-side array
--
peekArray :: (Shape dim, Elt e) => Array dim e -> CIO ()
peekArray (Array sh adata) = doPeek =<< gets memoryTable
  where
    doPeek mt = liftIO $ peekR arrayElt adata
      where
        peekR :: ArrayEltR e -> ArrayData e -> IO ()
        peekR ArrayEltRunit             _  = return ()
        peekR (ArrayEltRpair aeR1 aeR2) ad = peekR aeR1 (fst ad) >> peekR aeR2 (snd ad)
        peekR aer                       ad = peekPrim aer mt ad (size sh)
        --
        peekPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(peekPrim,Prim.peekArray)

peekArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> CIO ()
peekArrayAsync (Array sh adata) ms = doPeek =<< gets memoryTable
  where
    doPeek mt = liftIO $ peekR arrayElt adata
      where
        peekR :: ArrayEltR e -> ArrayData e -> IO ()
        peekR ArrayEltRunit             _  = return ()
        peekR (ArrayEltRpair aeR1 aeR2) ad = peekR aeR1 (fst ad) >> peekR aeR2 (snd ad)
        peekR aer                       ad = peekPrim aer mt ad (size sh) ms
        --
        peekPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Int -> Maybe CUDA.Stream -> IO ()
        mkPrimDispatch(peekPrim,Prim.peekArrayAsync)


-- Copy data from an Accelerate array into the associated device array
--
pokeArray :: (Shape dim, Elt e) => Array dim e -> CIO ()
pokeArray (Array sh adata) = doPoke =<< gets memoryTable
  where
    doPoke mt = liftIO $ pokeR arrayElt adata
      where
        pokeR :: ArrayEltR e -> ArrayData e -> IO ()
        pokeR ArrayEltRunit             _  = return ()
        pokeR (ArrayEltRpair aeR1 aeR2) ad = pokeR aeR1 (fst ad) >> pokeR aeR2 (snd ad)
        pokeR aer                       ad = pokePrim aer mt ad (size sh)
        --
        pokePrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(pokePrim,Prim.pokeArray)

pokeArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> CIO ()
pokeArrayAsync (Array sh adata) ms = doPoke =<< gets memoryTable
  where
    doPoke mt = liftIO $ pokeR arrayElt adata
      where
        pokeR :: ArrayEltR e -> ArrayData e -> IO ()
        pokeR ArrayEltRunit             _  = return ()
        pokeR (ArrayEltRpair aeR1 aeR2) ad = pokeR aeR1 (fst ad) >> pokeR aeR2 (snd ad)
        pokeR aer                       ad = pokePrim aer mt ad (size sh) ms
        --
        pokePrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Int -> Maybe CUDA.Stream -> IO ()
        mkPrimDispatch(pokePrim,Prim.pokeArrayAsync)


-- |Wrap device pointers into arguments that can be passed to a kernel
-- invocation
--
marshalDevicePtrs :: ArrayElt e => ArrayData e -> Prim.DevicePtrs e -> [CUDA.FunParam]
marshalDevicePtrs adata = marshalR arrayElt adata
  where
    marshalR :: ArrayEltR e -> ArrayData e -> Prim.DevicePtrs e -> [CUDA.FunParam]
    marshalR ArrayEltRunit             _  _       = []
    marshalR (ArrayEltRpair aeR1 aeR2) ad (p1,p2) = marshalR aeR1 (fst ad) p1 ++
                                                    marshalR aeR2 (snd ad) p2
    marshalR aer                       ad ptr     = [marshalPrim aer ad ptr]
    --
    marshalPrim :: ArrayEltR e -> ArrayData e -> Prim.DevicePtrs e -> CUDA.FunParam
    mkPrimDispatch(marshalPrim,Prim.marshalDevicePtrs)


-- |Wrap the device pointers corresponding to a host-side array into arguments
-- that can be passed to a kernel upon invocation.
--
marshalArrayData :: ArrayElt e => ArrayData e -> CIO [CUDA.FunParam]
marshalArrayData adata = doMarshal =<< gets memoryTable
  where
    doMarshal mt = liftIO $ marshalR arrayElt adata
      where
        marshalR :: ArrayEltR e -> ArrayData e -> IO [CUDA.FunParam]
        marshalR ArrayEltRunit             _  = return []
        marshalR (ArrayEltRpair aeR1 aeR2) ad = (++) <$> marshalR aeR1 (fst ad)
                                                     <*> marshalR aeR2 (snd ad)
        marshalR aer                       ad = return <$> marshalPrim aer mt ad
        --
        marshalPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> IO CUDA.FunParam
        mkPrimDispatch(marshalPrim,Prim.marshalArrayData)


-- |Bind the device memory arrays to the given texture reference(s), setting
-- appropriate type. The arrays are bound, and the list of textures thereby
-- consumed, in projection index order --- i.e. right-to-left
--
marshalTextureData :: ArrayElt e => ArrayData e -> Int -> [CUDA.Texture] -> CIO ()
marshalTextureData adata n texs = doMarshal =<< gets memoryTable
  where
    doMarshal mt = liftIO $ marshalR arrayElt adata texs >> return ()
      where
        marshalR :: ArrayEltR e -> ArrayData e -> [CUDA.Texture] -> IO Int
        marshalR ArrayEltRunit             _  _ = return 0
        marshalR (ArrayEltRpair aeR1 aeR2) ad t
          = do r <- marshalR aeR2 (snd ad) t
               l <- marshalR aeR1 (fst ad) (drop r t)
               return (l + r)
        marshalR aer                       ad t
          = do marshalPrim aer mt ad n (head t)
               return 1
        --
        marshalPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Int -> CUDA.Texture -> IO ()
        mkPrimDispatch(marshalPrim,Prim.marshalTextureData)


-- |Raw device pointers associated with a host-side array
--
devicePtrsOfArrayData :: ArrayElt e => ArrayData e -> CIO (Prim.DevicePtrs e)
devicePtrsOfArrayData adata = ptrs =<< gets memoryTable
  where
    ptrs mt = liftIO $ ptrsR arrayElt adata
      where
        ptrsR :: ArrayEltR e -> ArrayData e -> IO (Prim.DevicePtrs e)
        ptrsR ArrayEltRunit             _  = return ()
        ptrsR (ArrayEltRpair aeR1 aeR2) ad = (,) <$> ptrsR aeR1 (fst ad)
                                                 <*> ptrsR aeR2 (snd ad)
        ptrsR aer                       ad = ptrsPrim aer mt ad
        --
        ptrsPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> IO (Prim.DevicePtrs e)
        mkPrimDispatch(ptrsPrim,Prim.devicePtrsOfArrayData)


-- |Advance a set of device pointers by the given number of elements each
--
advancePtrsOfArrayData :: ArrayElt e => ArrayData e -> Int -> Prim.DevicePtrs e -> Prim.DevicePtrs e
advancePtrsOfArrayData adata n = advanceR arrayElt adata
  where
    advanceR :: ArrayEltR e -> ArrayData e -> Prim.DevicePtrs e -> Prim.DevicePtrs e
    advanceR ArrayEltRunit             _  _       = ()
    advanceR (ArrayEltRpair aeR1 aeR2) ad (p1,p2) = (advanceR aeR1 (fst ad) p1
                                                    ,advanceR aeR2 (snd ad) p2)
    advanceR aer                       ad ptr     = advancePrim aer ad ptr
    --
    advancePrim :: ArrayEltR e -> ArrayData e -> Prim.DevicePtrs e -> Prim.DevicePtrs e
    mkPrimDispatch(advancePrim,Prim.advancePtrsOfArrayData n)


