{-# LANGUAGE QuasiQuotes #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
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

  CUDA, Gen, GenST(..),
  runCGM, evalCGM, execCGM, pushEnv, getEnv, fresh, bind, use,

) where

import Prelude                                          hiding ( exp )
import Data.HashSet                                     ( HashSet )
import Data.HashMap.Strict                              ( HashMap )
import Data.Hashable
import Control.Monad
import Control.Monad.State.Strict
import Control.Applicative
import Language.C.Quote.CUDA
import qualified Language.C                             as C
import qualified Data.HashSet                           as Set
import qualified Data.HashMap.Strict                    as Map

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.CUDA.CodeGen.Type

instance Hashable C.Exp where
  hash  = hash . show


-- The state of the code generator monad. The outer monad is used to generate
-- fresh variable names, while the inner is used to collect local environment
-- bindings when generating code.
--
-- This separation is require so that names are unique across all generated code
-- fragments of a skeleton.
--
type CUDA       = State Int
type Gen        = StateT GenST CUDA

data GenST = GenST
  { bindings    :: [C.BlockItem]
  , terms       :: HashSet C.Exp
  , letterms    :: HashMap C.Exp C.Exp
  }


-- Run the code generator with a fresh environment, returning the result and
-- final state.
--
runCGM :: Gen a -> CUDA (a, GenST)
runCGM a = runStateT a (GenST [] Set.empty Map.empty)

evalCGM :: Gen a -> CUDA a
evalCGM = fmap fst . runCGM

execCGM :: Gen a -> CUDA GenST
execCGM = fmap snd . runCGM


-- Create new binding points for the C expressions associated with the given AST
-- term, unless the term is itself a variable.
--
-- Additionally, add these new terms to a map from the variable name to original
-- binding expression. This will be used as a reverse lookup when marking terms
-- as used.
--
pushEnv :: OpenExp env aenv t -> [C.Exp] -> Gen [C.Exp]
pushEnv exp cs =
  case exp of
    Var _       -> return cs
    Prj _ _     -> return cs
    _           -> do
      vs <- zipWithM bind (expType exp) cs
      modify (\st -> st { letterms = Map.union (Map.fromList (zip vs cs)) (letterms st) })
      return vs

-- Return the local environment code, consisting of a list of initialisation
-- declarations and statements. During construction, these are introduced to the
-- front of the list, so reverse to get in execution order.
--
getEnv :: Gen [C.BlockItem]
getEnv = reverse <$> gets bindings

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
  modify (\st -> st { bindings = C.BlockDecl [cdecl| const $ty:t $id:name = $exp:e;|] : bindings st })
  return [cexp| $id:name |]

-- Add an expression to the set marking that it will be used to generate the
-- output value(s). If the term exists in the reverse let-map, add that binding
-- instead.
--
use :: C.Exp -> Gen ()
use e = do
  m <- gets letterms
  case Map.lookup e m of
    Nothing     -> modify (\st -> st { terms = Set.insert e (terms st) })
    Just x      -> modify (\st -> st { terms = Set.insert x (terms st) })

