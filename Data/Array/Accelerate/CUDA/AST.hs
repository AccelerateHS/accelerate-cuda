{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE TypeSynonymInstances #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.AST
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.AST (

  module Data.Array.Accelerate.AST,
  AccKernel(..), AccBindings(..), ArrayVar(..), ExecAcc, ExecAfun, ExecOpenAcc(..),
  retag

) where

-- friends
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Pretty
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )
import qualified Data.Array.Accelerate.CUDA.FullList    as FL
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Analysis                  as CUDA

-- system
import Text.PrettyPrint
import Data.Hashable
import Data.Monoid                                      ( Monoid(..) )
import qualified Data.HashSet                           as Set


-- A non-empty list of binary objects will be used to execute a kernel. We keep
-- auxiliary information together with the compiled module, such as entry point
-- and execution information.
--
data AccKernel a where
  AccKernel :: !String                          -- __global__ entry function name
            -> {-# UNPACK #-} !CUDA.Fun         -- __global__ function object
            -> {-# UNPACK #-} !CUDA.Module      -- binary module
            -> {-# UNPACK #-} !CUDA.Occupancy   -- occupancy analysis
            -> {-# UNPACK #-} !Int              -- thread block size
            -> {-# UNPACK #-} !Int              -- shared memory per block (bytes)
            -> !(Int -> Int)                    -- number of blocks for input problem size
            -> AccKernel a


-- The kernel lists are monomorphic, so sometimes we need to change the phantom
-- type of the object code.
--
retag :: AccKernel a -> AccKernel b
retag (AccKernel x1 x2 x3 x4 x5 x6 x7) = AccKernel x1 x2 x3 x4 x5 x6 x7


-- Kernel execution is asynchronous, barriers allow (cross-stream)
-- synchronisation to determine when the operation has completed
--
-- data AccBarrier = AB !Stream !Event

-- Array computations that were embedded within scalar expressions, and will be
-- required to execute the kernel; i.e. bound to texture references or similar.
--
newtype AccBindings aenv = AccBindings ( Set.HashSet (ArrayVar aenv) )

instance Monoid (AccBindings aenv) where
  mempty                                = AccBindings ( Set.empty )
  AccBindings x `mappend` AccBindings y = AccBindings ( Set.union x y )

data ArrayVar aenv where
  ArrayVar :: (Shape sh, Elt e)
           => Idx aenv (Array sh e)
           -> ArrayVar aenv

instance Eq (ArrayVar aenv) where
  ArrayVar ix1 == ArrayVar ix2 = idxToInt ix1 == idxToInt ix2

instance Hashable (ArrayVar aenv) where
  hash (ArrayVar ix) = hash (idxToInt ix)


-- Interleave compilation & execution state annotations into an open array
-- computation AST
--
data ExecOpenAcc aenv a where
  ExecAcc :: {-# UNPACK #-} !(FL.FullList () (AccKernel a))     -- executable binary objects
          -> !(AccBindings aenv)                                -- auxiliary arrays from the environment the kernel needs access to
          -> !(PreOpenAcc ExecOpenAcc aenv a)                   -- the actual computation
          -> ExecOpenAcc aenv a                                 -- the recursive knot


-- An annotated AST suitable for execution in the CUDA environment
--
type ExecAcc  a = ExecOpenAcc () a
type ExecAfun a = PreAfun ExecOpenAcc a

instance Show (ExecOpenAcc aenv a) where
  show = render . prettyExecAcc 0 noParens

instance Show (ExecAfun a) where
  show = render . prettyExecAfun 0


-- Display the annotated AST
-- -------------------------

prettyExecAfun :: Int -> ExecAfun a -> Doc
prettyExecAfun alvl pfun = prettyPreAfun prettyExecAcc alvl pfun

prettyExecAcc :: PrettyAcc ExecOpenAcc
prettyExecAcc alvl wrap (ExecAcc _ (AccBindings fv) pacc) =
  let base      = prettyPreAcc prettyExecAcc alvl wrap pacc
      ann       = braces (freevars (Set.toList fv))
      freevars  = (text "fv=" <>) . brackets . hcat . punctuate comma
                                  . map (\(ArrayVar ix) -> char 'a' <> int (idxToInt ix))
  in case pacc of
       Avar _         -> base
       Alet  _ _      -> base
       Apply _ _      -> base
       Acond _ _ _    -> base
       Atuple _       -> base
       Aprj _ _       -> base
       _              -> ann <+> base

