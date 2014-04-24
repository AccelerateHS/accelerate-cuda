{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeFamilies        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Array.Data
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
--               [2013]       Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell, Robert Clifton-Everest
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Array.Data (

  -- * Array operations and representations
  mallocArray, indexArray,
  useArray,  useArrayAsync,
  useDevicePtrs,
  copyArray, copyArrayPeer, copyArrayPeerAsync,
  peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync,
  marshalArrayData, marshalTextureData, marshalDevicePtrs,
  devicePtrsOfArrayData, advancePtrsOfArrayData,
  devicePtrsFromList, devicePtrsToWordPtrs,

  -- * Garbage collection
  cleanupArrayData

) where

-- libraries
import Prelude                                          hiding ( fst, snd )
import qualified Prelude                                as P
import Control.Applicative
import Control.Monad.Reader                             ( asks )
import Control.Monad.State                              ( gets )
import Control.Monad.Trans                              ( liftIO )
import Foreign.C.Types
import Foreign.Ptr

-- friends
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Sugar                ( Array(..), Shape, Elt, toElt, EltRepr )
import Data.Array.Accelerate.Array.Representation       ( size )
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Array.Table
import qualified Data.Array.Accelerate.CUDA.Array.Prim  as Prim
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Driver.Stream             as CUDA
import qualified Foreign.CUDA.Driver.Texture            as CUDA


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

-- Extract the state information to pass along to the primitive data handlers
--
{-# INLINE run #-}
run :: (Context -> MemoryTable -> IO a) -> CIO a
run f = do
  ctx    <- asks activeContext
  mt     <- gets memoryTable
  liftIO $! f ctx mt

-- CPP hackery to generate the cases where we dispatch to the worker function handling
-- elementary types.
--
#define mkPrimDispatch(dispatcher,worker)                                       \
; dispatcher ArrayEltRint     = worker                                          \
; dispatcher ArrayEltRint8    = worker                                          \
; dispatcher ArrayEltRint16   = worker                                          \
; dispatcher ArrayEltRint32   = worker                                          \
; dispatcher ArrayEltRint64   = worker                                          \
; dispatcher ArrayEltRword    = worker                                          \
; dispatcher ArrayEltRword8   = worker                                          \
; dispatcher ArrayEltRword16  = worker                                          \
; dispatcher ArrayEltRword32  = worker                                          \
; dispatcher ArrayEltRword64  = worker                                          \
; dispatcher ArrayEltRfloat   = worker                                          \
; dispatcher ArrayEltRdouble  = worker                                          \
; dispatcher ArrayEltRbool    = worker                                          \
; dispatcher ArrayEltRchar    = worker                                          \
; dispatcher ArrayEltRcshort  = worker                                          \
; dispatcher ArrayEltRcushort = worker                                          \
; dispatcher ArrayEltRcint    = worker                                          \
; dispatcher ArrayEltRcuint   = worker                                          \
; dispatcher ArrayEltRclong   = worker                                          \
; dispatcher ArrayEltRculong  = worker                                          \
; dispatcher ArrayEltRcllong  = worker                                          \
; dispatcher ArrayEltRcullong = worker                                          \
; dispatcher ArrayEltRcfloat  = worker                                          \
; dispatcher ArrayEltRcdouble = worker                                          \
; dispatcher ArrayEltRcchar   = worker                                          \
; dispatcher ArrayEltRcschar  = worker                                          \
; dispatcher ArrayEltRcuchar  = worker                                          \
; dispatcher _                = error "mkPrimDispatcher: not primitive"


-- |Allocate a new device array to accompany the given host-side array.
--
mallocArray :: (Shape dim, Elt e) => Array dim e -> CIO ()
mallocArray (Array !sh !adata) = run doMalloc
  where
    !n                = size sh
    doMalloc !ctx !mt = mallocR arrayElt adata
      where
        mallocR :: ArrayEltR e -> ArrayData e -> IO ()
        mallocR ArrayEltRunit             _  = return ()
        mallocR (ArrayEltRpair aeR1 aeR2) ad = mallocR aeR1 (fst ad) >> mallocR aeR2 (snd ad)
        mallocR aer                       ad = mallocPrim aer ctx mt ad n
        --
        mallocPrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(mallocPrim,Prim.mallocArray)


-- |Upload an existing array to the device
--
useArray :: (Shape dim, Elt e) => Array dim e -> CIO ()
useArray (Array !sh !adata) = run doUse
  where
    !n             = size sh
    doUse !ctx !mt = useR arrayElt adata
      where
        useR :: ArrayEltR e -> ArrayData e -> IO ()
        useR ArrayEltRunit             _  = return ()
        useR (ArrayEltRpair aeR1 aeR2) ad = useR aeR1 (fst ad) >> useR aeR2 (snd ad)
        useR aer                       ad = usePrim aer ctx mt ad n
        --
        usePrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(usePrim,Prim.useArray)

useArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> CIO ()
useArrayAsync (Array !sh !adata) ms = run doUse
  where
    !n             = size sh
    doUse !ctx !mt = useR arrayElt adata
      where
        useR :: ArrayEltR e -> ArrayData e -> IO ()
        useR ArrayEltRunit             _  = return ()
        useR (ArrayEltRpair aeR1 aeR2) ad = useR aeR1 (fst ad) >> useR aeR2 (snd ad)
        useR aer                       ad = usePrim aer ctx mt ad n ms
        --
        usePrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> Int -> Maybe CUDA.Stream -> IO ()
        mkPrimDispatch(usePrim,Prim.useArrayAsync)


useDevicePtrs :: (Shape sh, Elt e) => EltRepr sh -> Prim.DevicePtrs (EltRepr e) -> CIO (Array sh e)
useDevicePtrs sh ptrs = run doUse
  where
    !n             = size sh
    doUse !ctx !mt = Array sh <$> useD arrayElt ptrs
      where
        useD :: ArrayEltR e -> Prim.DevicePtrs e -> IO (ArrayData e)
        useD ArrayEltRunit             _  = return AD_Unit
        useD (ArrayEltRpair aeR1 aeR2) ps = AD_Pair <$> useD aeR1 (P.fst ps)
                                                    <*> useD aeR2 (P.snd ps)
        useD aer                       ps = usePrim aer ctx mt ps n
        --
        usePrim :: ArrayEltR e -> Context -> MemoryTable -> Prim.DevicePtrs e -> Int -> IO (ArrayData e)
        mkPrimDispatch(usePrim,Prim.useDevicePtrs)

devicePtrsFromList :: ArrayEltR e -> [WordPtr] -> Prim.DevicePtrs e
devicePtrsFromList aeR = P.fst . (devP aeR)
  where
    devP :: ArrayEltR e -> [WordPtr] -> (Prim.DevicePtrs e, [WordPtr])
    devP ArrayEltRunit             ps     = ((),ps)
    devP (ArrayEltRpair aeR1 aeR2) ps     = let
        (d1, ps')  = devP aeR1 ps
        (d2, ps'') = devP aeR2 ps'
      in ((d1,d2), ps'')
    devP aer                       (p:ps) = (devPrim aer p, ps)
    devP _                         []     = error "devicePtrsFromList: incorrect number of device pointers for element type"
    --
    devPrim :: ArrayEltR e -> WordPtr -> Prim.DevicePtrs e
    mkPrimDispatch(devPrim,CUDA.wordPtrToDevPtr)


-- |Read a single element from an array at the given row-major index. This is a
-- synchronous operation.
--
indexArray :: (Shape dim, Elt e) => Array dim e -> Int -> CIO e
indexArray (Array _ !adata) i = run doIndex
  where
    doIndex !ctx !mt = toElt <$> indexR arrayElt adata
      where
        indexR :: ArrayEltR e -> ArrayData e -> IO e
        indexR ArrayEltRunit             _  = return ()
        indexR (ArrayEltRpair aeR1 aeR2) ad = (,) <$> indexR aeR1 (fst ad)
                                                  <*> indexR aeR2 (snd ad)
        --
        indexR ArrayEltRint              ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRint8             ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRint16            ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRint32            ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRint64            ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRword             ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRword8            ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRword16           ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRword32           ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRword64           ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRfloat            ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRdouble           ad = Prim.indexArray ctx mt ad i
        indexR ArrayEltRchar             ad = Prim.indexArray ctx mt ad i
        --
        indexR ArrayEltRcshort           ad = CShort  <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcushort          ad = CUShort <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcint             ad = CInt    <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcuint            ad = CUInt   <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRclong            ad = CLong   <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRculong           ad = CULong  <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcllong           ad = CLLong  <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcullong          ad = CULLong <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcchar            ad = CChar   <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcschar           ad = CSChar  <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcuchar           ad = CUChar  <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcfloat           ad = CFloat  <$> Prim.indexArray ctx mt ad i
        indexR ArrayEltRcdouble          ad = CDouble <$> Prim.indexArray ctx mt ad i
        --
        indexR ArrayEltRbool             ad = toBool  <$> Prim.indexArray ctx mt ad i
          where toBool 0 = False
                toBool _ = True


-- |Copy data between two device arrays. The operation is asynchronous with
-- respect to the host, but will never overlap kernel execution.
--
copyArray :: (Shape dim, Elt e) => Array dim e -> Array dim e -> CIO ()
copyArray (Array !sh1 !adata1) (Array !sh2 !adata2)
  = $boundsCheck "copyArray" "shape mismatch" (sh1 == sh2)
  $ run doCopy
  where
    !n              = size sh1
    doCopy !ctx !mt = copyR arrayElt adata1 adata2
      where
        copyR :: ArrayEltR e -> ArrayData e -> ArrayData e -> IO ()
        copyR ArrayEltRunit             _   _   = return ()
        copyR (ArrayEltRpair aeR1 aeR2) ad1 ad2 = copyR aeR1 (fst ad1) (fst ad2) >>
                                                  copyR aeR2 (snd ad1) (snd ad2)
        copyR aer                       ad1 ad2 = copyPrim aer ctx mt ad1 ad2 n
        --
        copyPrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(copyPrim,Prim.copyArray)


-- |Copy data between two device arrays which reside in different contexts. This
-- might entail copying between devices.
--
copyArrayPeer :: (Shape dim, Elt e) => Array dim e -> Context -> Array dim e -> Context -> CIO ()
copyArrayPeer (Array !sh1 !adata1) !ctxSrc (Array !sh2 !adata2) !ctxDst
  = $boundsCheck "copyArrayPeer" "shape mismatch" (sh1 == sh2)
  $ run doCopy
  where
    !n           = size sh1
    doCopy _ !mt = copyR arrayElt adata1 adata2
      where
        copyR :: ArrayEltR e -> ArrayData e -> ArrayData e -> IO ()
        copyR ArrayEltRunit             _   _   = return ()
        copyR (ArrayEltRpair aeR1 aeR2) ad1 ad2 = copyR aeR1 (fst ad1) (fst ad2) >>
                                                  copyR aeR2 (snd ad1) (snd ad2)
        copyR aer                       ad1 ad2 = copyPrim aer mt ad1 ctxSrc ad2 ctxDst n
        --
        copyPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Context -> ArrayData e -> Context -> Int -> IO ()
        mkPrimDispatch(copyPrim,Prim.copyArrayPeer)

copyArrayPeerAsync :: (Shape dim, Elt e) => Array dim e -> Context -> Array dim e -> Context -> Maybe CUDA.Stream -> CIO ()
copyArrayPeerAsync (Array !sh1 !adata1) !ctxSrc (Array !sh2 !adata2) !ctxDst !ms
  = $boundsCheck "copyArrayPeerAsync" "shape mismatch" (sh1 == sh2)
  $ run doCopy
  where
    !n           = size sh1
    doCopy _ !mt = copyR arrayElt adata1 adata2
      where
        copyR :: ArrayEltR e -> ArrayData e -> ArrayData e -> IO ()
        copyR ArrayEltRunit             _   _   = return ()
        copyR (ArrayEltRpair aeR1 aeR2) ad1 ad2 = copyR aeR1 (fst ad1) (fst ad2) >>
                                                  copyR aeR2 (snd ad1) (snd ad2)
        copyR aer                       ad1 ad2 = copyPrim aer mt ad1 ctxSrc ad2 ctxDst n ms
        --
        copyPrim :: ArrayEltR e -> MemoryTable -> ArrayData e -> Context -> ArrayData e -> Context -> Int -> Maybe CUDA.Stream -> IO ()
        mkPrimDispatch(copyPrim,Prim.copyArrayPeerAsync)


-- Copy data from the device into the associated Accelerate host-side array
--
peekArray :: (Shape dim, Elt e) => Array dim e -> CIO ()
peekArray (Array !sh !adata) = run doPeek
  where
    !n              = size sh
    doPeek !ctx !mt = peekR arrayElt adata
      where
        peekR :: ArrayEltR e -> ArrayData e -> IO ()
        peekR ArrayEltRunit             _  = return ()
        peekR (ArrayEltRpair aeR1 aeR2) ad = peekR aeR1 (fst ad) >> peekR aeR2 (snd ad)
        peekR aer                       ad = peekPrim aer ctx mt ad n
        --
        peekPrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(peekPrim,Prim.peekArray)

peekArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> CIO ()
peekArrayAsync (Array !sh !adata) !ms = run doPeek
  where
    !n              = size sh
    doPeek !ctx !mt = peekR arrayElt adata
      where
        peekR :: ArrayEltR e -> ArrayData e -> IO ()
        peekR ArrayEltRunit             _  = return ()
        peekR (ArrayEltRpair aeR1 aeR2) ad = peekR aeR1 (fst ad) >> peekR aeR2 (snd ad)
        peekR aer                       ad = peekPrim aer ctx mt ad n ms
        --
        peekPrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> Int -> Maybe CUDA.Stream -> IO ()
        mkPrimDispatch(peekPrim,Prim.peekArrayAsync)


-- Copy data from an Accelerate array into the associated device array
--
pokeArray :: (Shape dim, Elt e) => Array dim e -> CIO ()
pokeArray (Array !sh !adata) = run doPoke
  where
    !n              = size sh
    doPoke !ctx !mt = pokeR arrayElt adata
      where
        pokeR :: ArrayEltR e -> ArrayData e -> IO ()
        pokeR ArrayEltRunit             _  = return ()
        pokeR (ArrayEltRpair aeR1 aeR2) ad = pokeR aeR1 (fst ad) >> pokeR aeR2 (snd ad)
        pokeR aer                       ad = pokePrim aer ctx mt ad n
        --
        pokePrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> Int -> IO ()
        mkPrimDispatch(pokePrim,Prim.pokeArray)

pokeArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> CIO ()
pokeArrayAsync (Array !sh !adata) !ms = run doPoke
  where
    !n              = size sh
    doPoke !ctx !mt = pokeR arrayElt adata
      where
        pokeR :: ArrayEltR e -> ArrayData e -> IO ()
        pokeR ArrayEltRunit             _  = return ()
        pokeR (ArrayEltRpair aeR1 aeR2) ad = pokeR aeR1 (fst ad) >> pokeR aeR2 (snd ad)
        pokeR aer                       ad = pokePrim aer ctx mt ad n ms
        --
        pokePrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> Int -> Maybe CUDA.Stream -> IO ()
        mkPrimDispatch(pokePrim,Prim.pokeArrayAsync)


-- |Convert the device pointers into a list of word pointers
--
devicePtrsToWordPtrs :: ArrayElt e => ArrayData e -> Prim.DevicePtrs e -> [WordPtr]
devicePtrsToWordPtrs !adata = wordR arrayElt adata
  where
    wordR :: ArrayEltR e -> ArrayData e -> Prim.DevicePtrs e -> [WordPtr]
    wordR ArrayEltRunit             _  _       = []
    wordR (ArrayEltRpair aeR1 aeR2) ad (p1,p2) = wordR aeR1 (fst ad) p1 ++
                                                 wordR aeR2 (snd ad) p2
    wordR aer                       ad ptr     = [wordPrim aer ad ptr]
    --
    wordPrim :: ArrayEltR e -> ArrayData e -> Prim.DevicePtrs e -> WordPtr
    mkPrimDispatch(wordPrim,const CUDA.devPtrToWordPtr)

-- |Wrap device pointers into arguments that can be passed to a kernel
-- invocation
--
marshalDevicePtrs :: ArrayElt e => ArrayData e -> Prim.DevicePtrs e -> [CUDA.FunParam]
marshalDevicePtrs !adata ptrs = map (CUDA.VArg . CUDA.wordPtrToDevPtr) $ devicePtrsToWordPtrs adata ptrs

-- |Wrap the device pointers corresponding to a host-side array into arguments
-- that can be passed to a kernel upon invocation.
--
marshalArrayData :: ArrayElt e => ArrayData e -> CIO [CUDA.FunParam]
marshalArrayData !adata = run doMarshal
  where
    doMarshal !ctx !mt = marshalR arrayElt adata
      where
        marshalR :: ArrayEltR e -> ArrayData e -> IO [CUDA.FunParam]
        marshalR ArrayEltRunit             _  = return []
        marshalR (ArrayEltRpair aeR1 aeR2) ad = (++) <$> marshalR aeR1 (fst ad)
                                                     <*> marshalR aeR2 (snd ad)
        marshalR aer                       ad = return <$> marshalPrim aer ctx mt ad
        --
        marshalPrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> IO CUDA.FunParam
        mkPrimDispatch(marshalPrim,Prim.marshalArrayData)


-- |Bind the device memory arrays to the given texture reference(s), setting
-- appropriate type. The arrays are bound, and the list of textures thereby
-- consumed, in projection index order --- i.e. right-to-left
--
marshalTextureData :: ArrayElt e => ArrayData e -> Int -> [CUDA.Texture] -> CIO ()
marshalTextureData !adata !n !texs = run doMarshal
  where
    doMarshal !ctx !mt = marshalR arrayElt adata texs >> return ()
      where
        marshalR :: ArrayEltR e -> ArrayData e -> [CUDA.Texture] -> IO Int
        marshalR ArrayEltRunit             _  _ = return 0
        marshalR (ArrayEltRpair aeR1 aeR2) ad t
          = do r <- marshalR aeR2 (snd ad) t
               l <- marshalR aeR1 (fst ad) (drop r t)
               return (l + r)
        marshalR aer                       ad t
          = do marshalPrim aer ctx mt ad n (head t)
               return 1
        --
        marshalPrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> Int -> CUDA.Texture -> IO ()
        mkPrimDispatch(marshalPrim,Prim.marshalTextureData)


-- |Raw device pointers associated with a host-side array
--
devicePtrsOfArrayData :: ArrayElt e => ArrayData e -> CIO (Prim.DevicePtrs e)
devicePtrsOfArrayData !adata = run ptrs
  where
    ptrs !ctx !mt = ptrsR arrayElt adata
      where
        ptrsR :: ArrayEltR e -> ArrayData e -> IO (Prim.DevicePtrs e)
        ptrsR ArrayEltRunit             _  = return ()
        ptrsR (ArrayEltRpair aeR1 aeR2) ad = (,) <$> ptrsR aeR1 (fst ad)
                                                 <*> ptrsR aeR2 (snd ad)
        ptrsR aer                       ad = ptrsPrim aer ctx mt ad
        --
        ptrsPrim :: ArrayEltR e -> Context -> MemoryTable -> ArrayData e -> IO (Prim.DevicePtrs e)
        mkPrimDispatch(ptrsPrim,Prim.devicePtrsOfArrayData)


-- |Advance a set of device pointers by the given number of elements each
--
advancePtrsOfArrayData :: ArrayElt e => ArrayData e -> Int -> Prim.DevicePtrs e -> Prim.DevicePtrs e
advancePtrsOfArrayData !adata !n = advanceR arrayElt adata
  where
    advanceR :: ArrayEltR e -> ArrayData e -> Prim.DevicePtrs e -> Prim.DevicePtrs e
    advanceR ArrayEltRunit             _  _       = ()
    advanceR (ArrayEltRpair aeR1 aeR2) ad (p1,p2) = (advanceR aeR1 (fst ad) p1
                                                    ,advanceR aeR2 (snd ad) p2)
    advanceR aer                       ad ptr     = advancePrim aer ad ptr
    --
    advancePrim :: ArrayEltR e -> ArrayData e -> Prim.DevicePtrs e -> Prim.DevicePtrs e
    mkPrimDispatch(advancePrim,Prim.advancePtrsOfArrayData n)

