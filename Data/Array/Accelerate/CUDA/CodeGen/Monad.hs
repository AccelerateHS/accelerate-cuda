{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Monad
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Monad (

  CUDA, Gen,
  pushEnv, getEnv, codegen, fresh, bind,

) where

import Prelude                                          hiding ( exp )
import Control.Monad
import Control.Monad.State.Strict
import Control.Applicative
import Language.C.Quote.CUDA
import qualified Language.C                             as C

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.CUDA.CodeGen.Type


-- The state of the code generator monad. The outer monad is used to generate
-- fresh variable names, while the inner is used to collect local environment
-- bindings when generating code.
--
-- This separation is require so that names are unique across all generated code
-- fragments of a skeleton.
--
type CUDA       = State Int
type Gen        = StateT [C.BlockItem] CUDA


-- Create new binding points for the C expressions associated with the given AST
-- term, unless the term is itself a variable.
--
pushEnv :: OpenExp env aenv t -> [C.Exp] -> Gen [C.Exp]
pushEnv exp cs =
  case exp of
    Var _       -> return cs
    _           -> zipWithM bind (expType exp) cs

-- Return the local environment code, consisting of a list of initialisation
-- declarations and statements. During construction, these are introduced to the
-- front of the list, so reverse to get in execution order.
--
getEnv :: Gen [C.BlockItem]
getEnv = reverse <$> get


-- Run the code generator
--
codegen :: CUDA a -> a
codegen = flip evalState 0


-- Generate a fresh variable name
--
fresh :: CUDA String
fresh = do
  n     <- get <* modify (+1)
  return $ 'v' : show n


-- Add an expression of given type to the environment and return the (new,
-- unique) binding name that can be used in place of the thing just bound.
--
bind :: C.Type -> C.Exp -> Gen C.Exp
bind t e = do
  name <- lift fresh
  modify ( C.BlockDecl [cdecl| const $ty:t $id:name = $exp:e;|] : )
  return [cexp|$id:name|]

