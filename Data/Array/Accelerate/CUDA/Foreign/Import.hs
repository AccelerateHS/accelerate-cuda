{-# LANGUAGE CPP                  #-}
{-# LANGUAGE ConstraintKinds      #-}
{-# LANGUAGE DeriveDataTypeable   #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE GADTs                #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Foreign.Import
-- Copyright   : [2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell, Robert Clifton-Everest
-- License     : BSD3
--
-- Maintainer  : Robert Clifton-Everest <robertce@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- This module provides the CUDA backend's implementation of Accelerate's
-- foreign function interface. Also provided are a series of utility functions
-- for transferring arrays from the device to the host (and vice-versa),
-- allocating new arrays, getting the CUDA device pointers of a given array, and
-- executing IO actions within a CUDA context.
--
-- [/NOTE:/]
--
-- When arrays are passed to the foreign function there is no guarantee that the
-- host side data matches the device side data. If the data is needed host side
-- 'peekArray' or 'peekArrayAsync' must be called.
--
-- Arrays of tuples are represented as tuples of arrays so for example an array
-- of type @Array DIM1 (Float, Float)@ would have two device pointers associated
-- with it.
--

module Data.Array.Accelerate.CUDA.Foreign.Import (

  -- * Backend representation
  canExecute, CUDAForeignAcc(..), CUDAForeignExp(..), CIO,
  liftIO, canExecuteExp,

  -- * Manipulating arrays
  DevicePtrs,
  devicePtrsOfArray,
  indexArray, copyArray,
  useArray,  useArrayAsync,
  peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync,
  allocateArray, newArray,

  -- * Running IO actions in an Accelerate context
  inContext, inDefaultContext

) where

import Data.Array.Accelerate.Type
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.Array.Prim            ( DevicePtrs )

import Data.Dynamic
import Control.Applicative
import Control.Exception                                ( bracket_ )
import Control.Monad.Trans                              ( liftIO )
import System.IO.Unsafe                                 ( unsafePerformIO )
import System.Mem.StableName


-- CUDA backend representation of foreign functions
-- ------------------------------------------------

-- |CUDA foreign Acc functions are just CIO functions.
--
newtype CUDAForeignAcc as bs = CUDAForeignAcc (as -> CIO bs)
  deriving (Typeable)

instance Foreign CUDAForeignAcc where
  -- Using the hash of the StableName in order to uniquely identify the function
  -- when it is pretty printed.
  --
  strForeign f =
    let sn = unsafePerformIO $ makeStableName f
    in
    "cudaForeignAcc<" ++ show (hashStableName sn) ++ ">"

-- |Gives the executable form of a foreign function if it can be executed by the
-- CUDA backend.
--
canExecute :: forall f as bs. (Foreign f, Typeable as, Typeable bs)
           => f as bs
           -> Maybe (as -> CIO bs)
canExecute f = (\(CUDAForeignAcc f') -> f') <$> (cast f :: Maybe (CUDAForeignAcc as bs))

-- |CUDA foreign Exp functions consist of a list of C header files necessary to call the function
-- and the name of the function to call.
--
data CUDAForeignExp x y where
  CUDAForeignExp :: IsScalar y => [String] -> String -> CUDAForeignExp x y
  deriving (Typeable)

instance Foreign CUDAForeignExp where
  strForeign (CUDAForeignExp _ n) = "cudaForeignExp<" ++ n ++ ">"

-- |Gives the foreign function name as a string if it is a foreign Exp function
-- for the CUDA backend.
--
canExecuteExp :: forall f x y. (Foreign f, Typeable y, Typeable x)
              => f x y
              -> Maybe ([String], String)
canExecuteExp f = (\(CUDAForeignExp h f') -> (h, f')) <$> (cast f :: Maybe (CUDAForeignExp x y))


-- User facing utility functions
-- -----------------------------

-- |Get the raw CUDA device pointers associated with an array
--
devicePtrsOfArray :: Array sh e -> CIO (DevicePtrs (EltRepr e))
devicePtrsOfArray (Array _ adata) = devicePtrsOfArrayData adata

-- |Run an IO action within the given Acclerate context
--
inContext :: Context -> IO a -> IO a
inContext ctx = bracket_ (push ctx) pop

-- |Run an IO action in the default Acclerate context
--
inDefaultContext :: IO a -> IO a
inDefaultContext = inContext defaultContext

