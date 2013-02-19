{-# LANGUAGE CPP                  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE ConstraintKinds      #-}
{-# LANGUAGE DeriveDataTypeable   #-}
{-# LANGUAGE ScopedTypeVariables  #-}
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
  cudaFF, canExecute, CuForeign, CIO,
  liftIO,
  
  -- * Manipulating arrays
  indexArray, copyArray,
  useArray,  useArrayAsync,
  peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync,
  devicePtrsOfArray,
  allocateArray, newArray,
  DevicePtrs,

  -- * Running IO actions in a CUDA context
  inContext, inDefaultContext
) where

import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.Array.Prim            ( DevicePtrs )

import qualified Foreign.CUDA.Driver                    as CUDA

import Control.Applicative
import System.IO.Unsafe                                 ( unsafePerformIO )
import System.Mem.StableName
import Data.Dynamic
import Control.Monad.Trans                              ( liftIO )

-- CUDA backend representation of foreign functions.
-- ---------------------------------------------------

-- CUDA foreign functions are just CIO functions.
newtype CuForeign args results = CuForeign (args -> CIO results) deriving (Typeable)

instance ForeignFun CuForeign where
  -- Using the hash of the stablename in order to uniquely identify the function
  -- when it is pretty printed.
  strForeign ff = "cudaFF<" ++ (show . hashStableName) (unsafePerformIO $ makeStableName ff) ++ ">"

-- |Gives an the executable form of a foreign function if it can be executed by the CUDA backend.
canExecute :: forall ff args results. (ForeignFun ff, Typeable args, Typeable results) 
           => ff args results 
           -> Maybe (args -> CIO results)
canExecute ff =
  let
    df = toDyn ff
    fd = fromDynamic :: Dynamic -> Maybe (CuForeign args results)
  in (\(CuForeign ff') -> ff') <$> fd df 


-- User facing utility functions
-- -----------------------------

-- |Create a cuda foreign function.
cudaFF :: (Arrays args, Arrays results)
       => (args -> CIO results)
       -> CuForeign args results
cudaFF = CuForeign

-- |Get the raw CUDA device pointers associated with an array.
--
devicePtrsOfArray :: Array sh e -> CIO (DevicePtrs (EltRepr e))
devicePtrsOfArray (Array _ adata) = devicePtrsOfArrayData adata 

-- |Run an IO action within the given CUDA context
inContext :: CUDA.Context -> IO a -> IO a
inContext ctx a = do
  CUDA.push ctx
  r <- a
  _ <- CUDA.pop
  return r

-- |Run an IO action in the default CUDA context
inDefaultContext :: IO a -> IO a
inDefaultContext = inContext defaultContext
