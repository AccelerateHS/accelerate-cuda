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

  CUDA, Gen, AccST(..), ExpST(..),
  runCUDA, runCGM, evalCGM, execCGM, pushEnv, getEnv, fresh, bind, use,

) where

import Prelude                                          hiding ( exp )
import Data.HashSet                                     ( HashSet )
import Data.Hashable
import Control.Monad
import Control.Monad.State.Strict
import Language.C.Quote.CUDA
import qualified Language.C                             as C
import qualified Data.HashSet                           as Set

import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate.CUDA.CodeGen.Type

instance Hashable C.Exp where
  hashWithSalt salt = hashWithSalt salt . show


-- The state of the code generator monad. The outer monad is used to generate
-- fresh variable names and collect any headers required for foreign functions.
-- The inner is used to collect local bindings (for let-bound expressions) and
-- to record when previously bound variables are later used for dead-code
-- analysis.
--
-- This separation is required so that names are unique across all generated
-- code fragments of a skeleton.
--
type CUDA       = State  AccST
type Gen        = StateT ExpST CUDA

data AccST = AccST
  { counter     :: {-# UNPACK #-} !Int
  , headers     :: !(HashSet String)
  }

data ExpST = ExpST
  -- A stack of (typically) declarations that must be evaluated before
  -- computation of the main expression. These are typically introduced by
  -- let-bindings. The list is kept in reverse order, with the last to be
  -- evaluated (newest addition) at the front of the list.
  --
  { localBindings       :: [C.BlockItem]

  -- A set of the Var's that we know have been used in the expression. With this
  -- we can do def-use analysis for simple dead-code elimination.
  --
  , usedTerms           :: !(HashSet C.Exp)
  }


-- Run the code generator with a fresh environment, returning the result and
-- final state.
--
runCUDA :: CUDA a -> (a, AccST)
runCUDA a = runState a (AccST 0 Set.empty)

runCGM :: Gen a -> CUDA (a, ExpST)
runCGM a = runStateT a (ExpST [] Set.empty)

evalCGM :: Gen a -> CUDA a
evalCGM = fmap fst . runCGM

execCGM :: Gen a -> CUDA ExpST
execCGM = fmap snd . runCGM


-- Create new binding points for the C expressions associated with the given AST
-- term, unless the term is itself a variable.
--
-- Additionally, add these new terms to a map from the variable name to original
-- binding expression. This will be used as a reverse lookup when marking terms
-- as used.
--
pushEnv :: DelayedOpenExp env aenv t -> [C.Exp] -> Gen [C.Exp]
pushEnv exp cs =
  let tys               = expType exp
      zipWithM' xs ys f = zipWithM f xs ys
  in
  zipWithM' tys cs $ \ty c ->
    case c of
      C.Var{}   -> return c
      C.Const{} -> return c
      _         -> bind ty c


-- Return the local environment code, consisting of a list of initialisation
-- declarations and statements. During construction, these are introduced to the
-- front of the list, so reverse to get in execution order.
--
getEnv :: Gen [C.BlockItem]
getEnv = reverse `fmap` gets localBindings

-- Generate a fresh variable name
--
fresh :: CUDA String
fresh =
  StateT $ \s -> let n = counter s
                 in  return ('v':show n, s { counter = n+1 })

-- Add an expression of given type to the environment and return the (new,
-- unique) binding name that can be used in place of the thing just bound.
--
bind :: C.Type -> C.Exp -> Gen C.Exp
bind t e =
  case e of
    C.Var{}     -> return e
    _           -> do
      name <- lift fresh
      modify (\st -> st { localBindings = [citem| const $ty:t $id:name = $exp:e;|] : localBindings st })
      return [cexp| $id:name |]


-- Add an expression to the set marking that it will be used to generate the
-- output value(s). If the term exists in the reverse let-map, add that binding
-- instead.
--
use :: C.Exp -> Gen C.Exp
use e@(C.Var{}) = do
  modify (\st -> st { usedTerms = Set.insert e (usedTerms st) })
  return e
--
use e           = return e

