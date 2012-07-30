{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE QuasiQuotes     #-}
{-# LANGUAGE TemplateHaskell #-}
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

  runCGM, CGM,
  bind, use, pushEnv, bodycode, subscripts

) where

import Data.Label                               ( mkLabels )
import Data.Label.PureM
import Control.Applicative
import Control.Monad.State                      ( State, evalState )
import Language.C
import Language.C.Quote.CUDA

import Data.IntMap                              ( IntMap )
import Data.Sequence                            ( Seq, (|>) )
import qualified Data.IntMap                    as IM
import qualified Data.Sequence                  as S


type CGM                = State CGEnv
data CGEnv              = CGEnv
  {
    -- Used to generate fresh variable names
    --
    _unique             :: {-# UNPACK #-} !Int

    -- The input variables to a function. Typically these correspond to reading
    -- data from global memory, so we keep track of which inputs are actually
    -- used and only read those values.
    --
  , _freevars           :: Seq (IntMap (Type, Exp))

    -- The body code, consisting of a list of;
    --   a) variable declarations & initialisations
    --   b) C statements
    --
  , _bindings           :: [BlockItem]
  }
  deriving Show

$(mkLabels [''CGEnv])


runCGM :: CGM a -> a
runCGM = flip evalState (CGEnv 0 S.empty [])


-- Add space for another variable
--
pushEnv :: CGM ()
pushEnv = modify freevars (|> IM.empty)

-- Add an expression of given type to the environment and return the (new,
-- unique) binding name that can be used in place of the thing just bound.
--
bind :: Type -> Exp -> CGM Exp
bind t e = do
  name  <- fresh
  modify bindings ( BlockDecl [cdecl| const $ty:t $id:name = $exp:e;|] : )
  return [cexp|$id:name|]

-- Return the body code, consisting of a list of initialisation declarations and
-- statements.
--
-- During construction, these are introduced to the front of the list, so
-- reverse to get in execution order.
--
bodycode :: CGM [BlockItem]
bodycode = reverse `fmap` gets bindings

-- Generate a fresh variable name
--
fresh :: CGM String
fresh = do
  n     <- gets unique <* modify unique (+1)
  return $ 'v':show n

-- Mark a variable at a given base and tuple index as being used.
--
use :: Int -> Int -> Type -> Exp -> CGM ()
use base prj ty var = modify freevars (S.adjust (IM.insert prj (ty,var)) base)

-- Return the tuple components of a given variable that are actually used. These
-- in snoc-list ordering, i.e. with variable zero on the right.
--
subscripts :: Int -> CGM [(Int, Type, Exp)]
subscripts base
  = reverse
  . map swizzle
  . IM.toList
  . flip S.index base <$> gets freevars
  where
    swizzle (i, (t,e)) = (i,t,e)

