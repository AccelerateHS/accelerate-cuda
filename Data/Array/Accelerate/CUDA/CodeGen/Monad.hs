{-# LANGUAGE BangPatterns, TemplateHaskell, QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Monad
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Monad (

  runCGM, CGM, Environment,
  lam, bind, environment

) where

import Data.Loc
import Data.Symbol
import Data.Label                               ( mkLabels )
import Data.Label.PureM
import Control.Applicative
import Control.Monad.State                      ( State, evalState )
import Language.C
import Language.C.Syntax
import Language.C.Quote.CUDA


type CGM                = State Gamma
type Environment        = [InitGroup]
data Gamma              = Gamma
  {
    _unique     :: !Int,
    _bindings   :: Environment
  }
  deriving Show

$(mkLabels [''Gamma])


runCGM :: CGM a -> a
runCGM = flip evalState (Gamma 0 [])

-- Introduce an expression of a given type and name into the environment. Return
-- an expression that can be used in place of the thing just bound (i.e. the
-- variable name)
--
lam :: String -> Type -> Exp -> CGM Exp
lam name t e = do
  modify bindings ( [cdecl| const $ty:t $id:name = $exp:e;|] : )
  return [cexp|$id:name|]

-- Add an expression of given type to the environment and return the (new,
-- unique) binding name
--
bind :: Type -> Exp -> CGM Exp
bind t e = do
  v     <- fresh
  _     <- lam v t e
  return $ [cexp|$id:v|]

-- Return the environment (list of initialisation declarations). Since we
-- introduce new bindings to the front of the list, need to reverse so they
-- appear in usage order.
--
environment :: CGM Environment
environment = reverse `fmap` gets bindings

-- Generate a fresh variable name
--
fresh :: CGM String
fresh = do
  n     <- gets unique <* modify unique (+1)
  return $ 'v':show n

