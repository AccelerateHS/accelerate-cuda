{-# LANGUAGE CPP                  #-}
{-# LANGUAGE ConstraintKinds      #-}
{-# LANGUAGE DeriveDataTypeable   #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Foreign
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

module Data.Array.Accelerate.CUDA.Foreign (

  -- * Backend representation
  cudaAcc, canExecute, CuForeignAcc, CuForeignExp, CIO,
  liftIO, canExecuteExp, cudaExp,

  -- * Manipulating arrays
  DevicePtrs,
  devicePtrsOfArray,
  indexArray, copyArray,
  useArray,  useArrayAsync,
  peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync,
  allocateArray, newArray,

  -- * Running IO actions in a CUDA context
  inContext, inDefaultContext

) where

import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.Array.Prim            ( DevicePtrs )

import qualified Foreign.CUDA.Driver                    as CUDA

import Data.Dynamic
import Control.Applicative
import Control.Exception                                ( bracket_ )
import Control.Monad.Trans                              ( liftIO )
import System.IO.Unsafe                                 ( unsafePerformIO )
import System.Mem.StableName


-- CUDA backend representation of foreign functions
-- ------------------------------------------------

-- CUDA foreign Acc functions are just CIO functions.
--
newtype CuForeignAcc args results = CuForeignAcc (args -> CIO results)
  deriving (Typeable)

instance Foreign CuForeignAcc where
  -- Using the hash of the StableName in order to uniquely identify the function
  -- when it is pretty printed.
  --
  strForeign ff =
    let sn = unsafePerformIO $ makeStableName ff
    in
    "cudaAcc<" ++ (show (hashStableName sn)) ++ ">"

-- |Gives the executable form of a foreign function if it can be executed by the
-- CUDA backend.
--
canExecute :: forall ff args results. (Foreign ff, Typeable args, Typeable results)
           => ff args results
           -> Maybe (args -> CIO results)
canExecute ff =
  let
    df = toDyn ff
    fd = fromDynamic :: Dynamic -> Maybe (CuForeignAcc args results)
  in (\(CuForeignAcc ff') -> ff') <$> fd df

-- CUDA foreign Exp functions are just strings with the header filename and the name of the
-- function separated by a space.
--
newtype CuForeignExp args results = CuForeignExp String
  deriving (Typeable)

instance Foreign CuForeignExp where
  strForeign (CuForeignExp n) = "cudaExp<" ++ n ++ ">"

-- |Gives the foreign function name as a string if it is a foreign Exp function
-- for the CUDA backend.
--
canExecuteExp :: forall ff args results. (Foreign ff, Typeable results, Typeable args)
              => ff args results
              -> Maybe String
canExecuteExp ff =
  let
    df = toDyn ff
    fd = fromDynamic :: Dynamic -> Maybe (CuForeignExp args results)
  in (\(CuForeignExp ff') -> ff') <$> fd df


-- User facing utility functions
-- -----------------------------

-- |Create a CUDA foreign function
--
cudaAcc :: (Arrays args, Arrays results)
        => (args -> CIO results)
        -> CuForeignAcc args results
cudaAcc = CuForeignAcc

-- |Create a CUDA foreign scalar function. The string needs to be formatted in
-- the same way as for the Haskell FFI. That is, the header file name and the
-- name of the function separated by a space. i.e cudaExp "stdlib.h min".
--
cudaExp :: (Elt args, Elt results)
        => String
        -> CuForeignExp args results
cudaExp = CuForeignExp

-- |Get the raw CUDA device pointers associated with an array
--
devicePtrsOfArray :: Array sh e -> CIO (DevicePtrs (EltRepr e))
devicePtrsOfArray (Array _ adata) = devicePtrsOfArrayData adata

-- |Run an IO action within the given CUDA context
--
inContext :: Context -> IO a -> IO a
inContext ctx action =
  bracket_ (push ctx) pop action

-- |Run an IO action in the default CUDA context
--
inDefaultContext :: IO a -> IO a
inDefaultContext = inContext defaultContext

