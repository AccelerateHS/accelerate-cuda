{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Analysis.Shape
-- Copyright   : [2012..2014] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Analysis.Shape (

  module Data.Array.Accelerate.Analysis.Shape,
  module Data.Array.Accelerate.CUDA.Analysis.Shape,
  (:~:)(..),

) where

import Data.Typeable

import Data.Array.Accelerate.Array.Sugar
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.Analysis.Match

import Data.Array.Accelerate.CUDA.AST


-- | Reify dimensionality of the result type of an array computation
--
execAccDim :: AccDim ExecOpenAcc
execAccDim (ExecAcc _ _ pacc) = preAccDim execAccDim pacc


-- Match reified shape types
--
matchShapeType
    :: forall sh sh'. (Shape sh, Shape sh')
    => sh
    -> sh'
    -> Maybe (sh :~: sh')
matchShapeType _ _
  | Just Refl <- matchTupleType (eltType (undefined::sh)) (eltType (undefined::sh'))
  = gcast Refl

matchShapeType _ _
  = Nothing

