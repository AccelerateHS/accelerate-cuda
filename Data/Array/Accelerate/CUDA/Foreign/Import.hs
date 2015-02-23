{-# LANGUAGE CPP                  #-}
{-# LANGUAGE ConstraintKinds      #-}
{-# LANGUAGE DeriveDataTypeable   #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE PatternGuards        #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE StandaloneDeriving   #-}
{-# LANGUAGE TypeFamilies         #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Foreign.Import
-- Copyright   : [2013..2014] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell, Robert Clifton-Everest
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
  CUDAForeignAcc(..), canExecuteAcc,
  CUDAForeignExp(..), canExecuteExp,

  -- * Manipulating arrays
  DevicePtrs,
  withDevicePtrs,
  indexArray,
  useArray,  useArrayAsync,
  peekArray, peekArrayAsync,
  pokeArray, pokeArrayAsync,
  copyArray, copyArrayAsync,
  allocateArray, newArray,

  -- * Running IO actions in an Accelerate context
  CIO, Stream, liftIO, inContext, inDefaultContext

) where

import Data.Array.Accelerate.Type
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Array.Data            hiding ( withDevicePtrs )
import qualified Data.Array.Accelerate.CUDA.Array.Data  as D
import Data.Array.Accelerate.CUDA.Array.Prim            ( DevicePtrs )
import Data.Array.Accelerate.CUDA.Execute.Stream        ( Stream )

import Data.Typeable
import Control.Exception                                ( bracket_ )
import Control.Monad.Trans                              ( liftIO )
import qualified Foreign.CUDA.Driver.Stream             as CUDA


-- CUDA backend representation of foreign functions
-- ------------------------------------------------

-- |CUDA foreign Acc functions are just CIO functions.
--
data CUDAForeignAcc as bs where
  CUDAForeignAcc :: String                      -- name of the function
                 -> (Stream -> as -> CIO bs)    -- operation to execute
                 -> CUDAForeignAcc as bs

deriving instance Typeable CUDAForeignAcc

instance Foreign CUDAForeignAcc where
  strForeign (CUDAForeignAcc n _) = n

-- |Gives the executable form of a foreign function if it can be executed by the
-- CUDA backend.
--
canExecuteAcc
    :: (Foreign f, Typeable as, Typeable bs)
    => f as bs
    -> Maybe (Stream -> as -> CIO bs)
canExecuteAcc ff
  | Just (CUDAForeignAcc _ fun) <- cast ff
  = Just fun

  | otherwise
  = Nothing

-- |CUDA foreign Exp functions consist of a list of C header files necessary to call the function
-- and the name of the function to call.
--
data CUDAForeignExp x y where
  CUDAForeignExp :: IsScalar y
                 => [String]                    -- header files to be imported
                 -> String                      -- name of the foreign function
                 -> CUDAForeignExp x y

deriving instance Typeable CUDAForeignExp

instance Foreign CUDAForeignExp where
  strForeign (CUDAForeignExp _ n) = n

-- |Gives the foreign function name as a string if it is a foreign Exp function
-- for the CUDA backend.
--
canExecuteExp
    :: forall f x y. (Foreign f, Typeable y, Typeable x)
    => f x y
    -> Maybe ([String], String)
canExecuteExp ff
  | Just (CUDAForeignExp hdr fun) <- cast ff    :: Maybe (CUDAForeignExp x y)
  = Just (hdr, fun)

  | otherwise
  = Nothing


-- User facing utility functions
-- -----------------------------

-- |Get the raw CUDA device pointers associated with an array and call the given
-- continuation.
--
withDevicePtrs :: Array sh e -> Maybe CUDA.Stream -> (DevicePtrs (EltRepr e) -> CIO b) -> CIO b
withDevicePtrs (Array _ adata) = D.withDevicePtrs adata

-- |Run an IO action within the given Acclerate context
--
inContext :: Context -> IO a -> IO a
inContext ctx = bracket_ (push ctx) pop

-- |Run an IO action in the default Acclerate context
--
inDefaultContext :: IO a -> IO a
inDefaultContext = inContext defaultContext

