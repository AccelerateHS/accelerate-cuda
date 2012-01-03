{-# LANGUAGE GADTs, FlexibleInstances, TypeSynonymInstances #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.AST
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.AST (

  module Data.Array.Accelerate.AST,
  AccKernel(..), AccBinding(..), ExecAcc, ExecAfun, ExecOpenAcc(..),
  List(..), FullList(..)

) where

-- friends
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Pretty
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )
import Data.Array.Accelerate.CUDA.State                 ( CIO )
import Foreign.CUDA.Driver                              ( Module )

-- system
import Text.PrettyPrint


-- A non-empty list of binary objects will be used to execute a kernel
--
data AccKernel a = Kernel String !(CIO Module)

data FullList a  = FL a !(List a)
data List a      = a :> !(List a) | Nil
infixr 5 :>


-- Kernel execution is asynchronous, barriers allow (cross-stream)
-- synchronisation to determine when the operation has completed
--
-- data AccBarrier = AB !Stream !Event

-- Array computations that were embedded within scalar expressions, and will be
-- required to execute the kernel; i.e. bound to texture references or similar.
--
data AccBinding aenv where
  ArrayVar :: (Shape sh, Elt e)
           => Idx aenv (Array sh e)
           -> AccBinding aenv

instance Eq (AccBinding aenv) where
  ArrayVar ix1 == ArrayVar ix2 = idxToInt ix1 == idxToInt ix2


-- Interleave compilation & execution state annotations into an open array
-- computation AST
--
data ExecOpenAcc aenv a where
  ExecAcc :: FullList (AccKernel a)             -- executable binary objects
          -> [AccBinding aenv]                  -- auxiliary arrays from the environment the kernel needs access to
          -> PreOpenAcc ExecOpenAcc aenv a      -- the actual computation
          -> ExecOpenAcc aenv a                 -- the recursive knot

-- An annotated AST suitable for execution in the CUDA environment
--
type ExecAcc  a = ExecOpenAcc () a
type ExecAfun a = PreAfun ExecOpenAcc a

instance Show (ExecOpenAcc aenv a) where
  show = render . prettyExecAcc 0 noParens

instance Show (ExecAfun a) where
  show = render . prettyExecAfun 0


-- Display the annotated AST
--
prettyExecAfun :: Int -> ExecAfun a -> Doc
prettyExecAfun alvl pfun = prettyPreAfun prettyExecAcc alvl pfun

prettyExecAcc :: PrettyAcc ExecOpenAcc
prettyExecAcc alvl wrap (ExecAcc _ fv pacc) =
  let base = prettyPreAcc prettyExecAcc alvl wrap pacc
      ann  = braces (freevars fv)
  in case pacc of
       Avar _         -> base
       Let  _ _       -> base
       Let2 _ _       -> base
       Apply _ _      -> base
       PairArrays _ _ -> base
       Acond _ _ _    -> base
       _              -> ann <+> base
  where
    freevars = (text "fv=" <>) . brackets . hcat . punctuate comma
                               . map (\(ArrayVar ix) -> char 'a' <> int (idxToInt ix))

