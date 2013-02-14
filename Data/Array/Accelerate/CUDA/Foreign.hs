{-# LANGUAGE CPP                  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ConstraintKinds      #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Foreign
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Robert Clifton-Everest <robertce@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- This module provides the CUDA backend's implementation of Accelerate's foreign function interface.
-- Also provided are a series of utility functions for transferring arrays from the device to the host
-- (and vice-versa), allocating new arrays, getting the CUDA device pointers of a given array, and
-- executing IO actions within a CUDA context. 
-- 
-- /NOTES:/
--
-- When arrays are passed to the foreign function there is no guarantee that the host side data matches
-- the device side data. If the data is needed host side 'peekArray' or 'peekArrayAsync' must be called.
--
-- Arrays of tuples are represented as tuples of arrays so for example an array of type 
-- 'Array DIM1 (Float, Float)' would have two device pointers associated with it.  

module Data.Array.Accelerate.CUDA.Foreign (
  -- * Backend representation
  BackendRepr(CudaR), cudaFF,
  
  -- * Manipulating arrays
  indexArray, copyArray,
  useArray,  useArrayAsync,
  peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync,
  devicePtrsOfArray,
  allocateArray, newArray,
  DevicePtrsOf,

  -- * Run actions in CUDA context
  inContext, inDefaultContext
) where

import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Array.Sugar           hiding ( allocateArray, newArray, useArray )
import qualified Data.Array.Accelerate.CUDA.Array.Data  as Data
import qualified Data.Array.Accelerate.CUDA.Array.Sugar as Sugar
import qualified Data.Array.Accelerate.CUDA.Array.Prim  as Prim

import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Driver.Stream             as CUDA

import Control.Applicative
import System.IO.Unsafe                                 ( unsafePerformIO )
import System.Mem.StableName

-- CUDA backend representation of foreign functions.
-- ---------------------------------------------------

-- CUDA foreign functions are just native Haskell functions in the IO monad.
newtype CuForeign args results = CuForeign (args -> IO results)

instance ForeignFun CuForeign where
  data BackendRepr args results = CudaR (args -> IO results)

  toBackendRepr (CuForeign ff) = CudaR ff

  -- Using the hash of the stablename in order to uniquely identify the function
  -- when it is pretty printed.
  strForeign ff = "cudaFF<" ++ (show . hashStableName) (unsafePerformIO $ makeStableName ff) ++ ">"


-- Converting between nested and unnested tuples of device pointers.
-- ------------------------------------------------------------------

type family DevicePtrElt d
type instance DevicePtrElt () = ()
type instance DevicePtrElt (CUDA.DevicePtr e) = ((), CUDA.DevicePtr e)
type instance DevicePtrElt (a, b) = (DevicePtrElt a, DevicePtrElt' b)
type instance DevicePtrElt (a, b, c) = (DevicePtrElt (a, b), DevicePtrElt' c)
type instance DevicePtrElt (a, b, c, d) = (DevicePtrElt (a, b, c), DevicePtrElt' d)
type instance DevicePtrElt (a, b, c, d, e) = (DevicePtrElt (a, b, c, d), DevicePtrElt' e)
type instance DevicePtrElt (a, b, c, d, e, f) = (DevicePtrElt (a, b, c, d, e), DevicePtrElt' f)
type instance DevicePtrElt (a, b, c, d, e, f, g) = (DevicePtrElt (a, b, c, d, e, f), DevicePtrElt' g)
type instance DevicePtrElt (a, b, c, d, e, f, g, h) = (DevicePtrElt (a, b, c, d, e, f, g), DevicePtrElt' h)
type instance DevicePtrElt (a, b, c, d, e, f, g, h, i)
  = (DevicePtrElt (a, b, c, d, e, f, g, h), DevicePtrElt' i)

type family DevicePtrElt' d
type instance DevicePtrElt' () = ()
type instance DevicePtrElt' (CUDA.DevicePtr e) = CUDA.DevicePtr e
type instance DevicePtrElt' (a, b) = (DevicePtrElt a, DevicePtrElt' b)
type instance DevicePtrElt' (a, b, c) = (DevicePtrElt (a, b), DevicePtrElt' c)
type instance DevicePtrElt' (a, b, c, d) = (DevicePtrElt (a, b, c), DevicePtrElt' d)
type instance DevicePtrElt' (a, b, c, d, e) = (DevicePtrElt (a, b, c, d), DevicePtrElt' e)
type instance DevicePtrElt' (a, b, c, d, e, f) = (DevicePtrElt (a, b, c, d, e), DevicePtrElt' f)
type instance DevicePtrElt' (a, b, c, d, e, f, g) = (DevicePtrElt (a, b, c, d, e, f), DevicePtrElt' g)
type instance DevicePtrElt' (a, b, c, d, e, f, g, h) = (DevicePtrElt (a, b, c, d, e, f, g), DevicePtrElt' h)
type instance DevicePtrElt' (a, b, c, d, e, f, g, h, i)
  = (DevicePtrElt (a, b, c, d, e, f, g, h), DevicePtrElt' i)

-- |Constraint that implies the tuple of device pointers 'd' matches the element type 'e'.
type DevicePtrsOf e d = (Prim.DevicePtrs (EltRepr e) ~ DevicePtrElt d, Dev d)

class Dev a where
  toDev  :: DevicePtrElt a -> a
  toDev' :: DevicePtrElt' a -> a

instance Dev (CUDA.DevicePtr e) where
  toDev ((),e) = e
  toDev' = id

instance (Dev a, Dev b) => Dev (a,b) where
  toDev  (a,b) = (toDev a, toDev' b)
  toDev' (a,b) = (toDev a, toDev' b)
  
instance (Dev a, Dev b, Dev c) => Dev (a,b,c) where
  toDev  (ab,c) = let (a, b) = toDev ab in (a, b, toDev' c)
  toDev' (ab,c) = let (a, b) = toDev ab in (a, b, toDev' c)

instance (Dev a, Dev b, Dev c, Dev d) => Dev (a,b,c,d) where
  toDev  (abc,d) = let (a, b, c) = toDev abc in (a, b, c, toDev' d)
  toDev' (abc,d) = let (a, b, c) = toDev abc in (a, b, c, toDev' d)

instance (Dev a, Dev b, Dev c, Dev d, Dev e) => Dev (a,b,c,d,e) where
  toDev  (abcd,e) = let (a, b, c, d) = toDev abcd in (a, b, c, d, toDev' e)
  toDev' (abcd,e) = let (a, b, c, d) = toDev abcd in (a, b, c, d, toDev' e)

instance (Dev a, Dev b, Dev c, Dev d, Dev e, Dev f) => Dev (a,b,c,d,e,f) where
  toDev  (abcde,f) = let (a, b, c, d, e) = toDev abcde in (a, b, c, d, e, toDev' f)
  toDev' (abcde,f) = let (a, b, c, d, e) = toDev abcde in (a, b, c, d, e, toDev' f)

instance (Dev a, Dev b, Dev c, Dev d, Dev e, Dev f, Dev g) => Dev (a,b,c,d,e,f,g) where
  toDev  (abcdef,g) = let (a, b, c, d, e, f) = toDev abcdef in (a, b, c, d, e, f, toDev' g)
  toDev' (abcdef,g) = let (a, b, c, d, e, f) = toDev abcdef in (a, b, c, d, e, f, toDev' g)

instance (Dev a, Dev b, Dev c, Dev d, Dev e, Dev f, Dev g, Dev h) => Dev (a,b,c,d,e,f,g,h) where
  toDev  (abcdefg,h) = let (a, b, c, d, e, f, g) = toDev abcdefg in (a, b, c, d, e, f, g, toDev' h)
  toDev' (abcdefg,h) = let (a, b, c, d, e, f, g) = toDev abcdefg in (a, b, c, d, e, f, g, toDev' h)

instance (Dev a, Dev b, Dev c, Dev d, Dev e, Dev f, Dev g, Dev h, Dev i) => Dev (a,b,c,d,e,f,g,h,i) where
  toDev  (abcdefgh,i) = let (a, b, c, d, e, f, g, h) = toDev abcdefgh in (a, b, c, d, e, f, g, h, toDev' i)
  toDev' (abcdefgh,i) = let (a, b, c, d, e, f, g, h) = toDev abcdefgh in (a, b, c, d, e, f, g, h, toDev' i)

-- User facing utility functions
-- -----------------------------

-- |Create a cuda foreign function.
cudaFF :: (Arrays args, Arrays results)
       => (args -> IO results)
       -> CuForeign args results
cudaFF = CuForeign

-- |Upload an existing array to the device
--
useArray :: (Shape dim, Elt e) => Array dim e -> IO ()
useArray = evalCUDA' . Data.useArray

useArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> IO ()
useArrayAsync a = evalCUDA' . Data.useArrayAsync a


-- |Read a single element from an array at the given row-major index. This is a
-- synchronous operation.
--
indexArray :: (Shape dim, Elt e) => Array dim e -> Int -> IO e
indexArray a = evalCUDA' . Data.indexArray a

-- |Copy data between two device arrays. The operation is asynchronous with
-- respect to the host, but will never overlap kernel execution.
--
copyArray :: (Shape dim, Elt e) => Array dim e -> Array dim e -> IO ()
copyArray a = evalCUDA' . Data.copyArray a

-- |Copy data from the device into the associated Accelerate host-side array
--
peekArray :: (Shape dim, Elt e) => Array dim e -> IO ()
peekArray = evalCUDA' . Data.peekArray

peekArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> IO ()
peekArrayAsync a = evalCUDA' . Data.peekArrayAsync a

-- |Copy data from an Accelerate array into the associated device array
--
pokeArray :: (Shape dim, Elt e) => Array dim e -> IO ()
pokeArray = evalCUDA' . Data.pokeArray

pokeArrayAsync :: (Shape dim, Elt e) => Array dim e -> Maybe CUDA.Stream -> IO ()
pokeArrayAsync a = evalCUDA' . Data.pokeArrayAsync a

-- |Get the raw CUDA device pointers associated with an array.
--
devicePtrsOfArray :: (DevicePtrsOf e d)
                  => Array sh e -> IO d
devicePtrsOfArray (Array _ adata) = evalCUDA' $ toDev <$> Data.devicePtrsOfArrayData adata 

-- |Allocate a new unitialised array on both the host and the device and link them together.
--
allocateArray :: (Shape dim, Elt e) => dim -> IO (Array dim e)
allocateArray = evalCUDA' . Sugar.allocateArray

-- |Create an array from its representation function, uploading the result to the
-- device
--
newArray :: (Shape sh, Elt e) => sh -> (sh -> e) -> IO (Array sh e)
newArray sh = newArray sh

inContext :: CUDA.Context -> IO a -> IO a
inContext ctx a = do
  CUDA.push ctx
  r <- a
  _ <- CUDA.pop
  return r

inDefaultContext :: IO a -> IO a
inDefaultContext = inContext defaultContext
