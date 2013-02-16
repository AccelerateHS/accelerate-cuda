{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GADTs                      #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeSynonymInstances       #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.AST
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.AST (

--   module Data.Array.Accelerate.AST,

  AccKernel(..), Gamma(..), 
  ExecAcc, -- ExecAfun, 
  ExecExp, ExecFun, 
  freevar,

) where

-- friends
import Data.Array.Accelerate.BackendKit.IRs.SimpleAcc as S

-- import Data.Array.Accelerate.AST
-- import Data.Array.Accelerate.Pretty
-- import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )

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


-- Kernel execution is asynchronous, barriers allow (cross-stream)
-- synchronisation to determine when the operation has completed
--
-- data AccBarrier = AB !Stream !Event

-- The set of free array variables for array computations that were embedded
-- within scalar expressions. This are required to execute the kernel, bi
-- binding to texture references to similar.
--
newtype Gamma = Gamma ( Set.HashSet (S.Var,S.Type) )
  deriving ( Monoid )

instance Hashable S.Var where
  hashWithSalt salt v = hashWithSalt salt (show v)

instance Hashable S.Type where
  hash ty =
    case ty of
      TTuple (h:t) -> hashWithSalt (hash h) (S.TTuple t)
      TArray n elt -> hashWithSalt (hash n) elt
      TInt8  -> 1; TInt16 -> 2; TInt32 -> 3; TInt64 ->4;
      TWord8 -> 5; TWord16 ->6; TWord32 ->7; TWord64 ->8;
      TInt   -> 9;
      TWord  -> 10
      TCShort -> 11
      TCInt   -> 12
      TCLong  -> 13
      TCLLong -> 14
      TCUShort -> 15
      TCUInt   -> 16
      TCULong  -> 17
      TCULLong -> 18
      TCChar   -> 19; TCSChar -> 20; TCUChar -> 21;
      TFloat  -> 22; TDouble  -> 23;
      TCFloat -> 24; TCDouble -> 25;
      TBool   -> 25; TChar -> 26; 


freevar :: (S.Var, S.Type)-> Gamma 
freevar = Gamma . Set.singleton 


-- An annotated AST suitable for execution in the CUDA environment
--
-- Interleave compilation & execution state annotations.
data ExecAcc a =
  ExecAcc {-# UNPACK #-} !(FL.FullList () (AccKernel a))     -- executable binary objects
          !(Gamma )                                      -- free array variables the kernel needs access to
          !(S.Prog ())                                       -- the actual computation


-- type ExecAfun a         = PreAfun ExecOpenAcc a

-- type ExecOpenExp        = PreOpenExp ExecOpenAcc
-- type ExecOpenFun        = PreOpenFun ExecOpenAcc

type ExecExp = S.Exp 
type ExecFun = S.Fun1 S.Exp 

-- Display the annotated AST
-- -------------------------

-- instance Show (ExecOpenAcc aenv a) where
--   show = render . prettyExecAcc 0 noParens

-- instance Show (ExecAfun a) where
--   show = render . prettyExecAfun 0


-- prettyExecAfun :: Int -> ExecAfun a -> Doc
-- prettyExecAfun alvl pfun = prettyPreAfun prettyExecAcc alvl pfun

-- prettyExecAcc :: PrettyAcc ExecOpenAcc
-- prettyExecAcc alvl wrap (ExecAcc _ (Gamma fv) pacc) =
--   let base      = prettyPreAcc prettyExecAcc alvl wrap pacc
--       ann       = braces (freevars (Set.toList fv))
--       freevars  = (text "fv=" <>) . brackets . hcat . punctuate comma
--                                   . map (\(Idx_ ix) -> char 'a' <> int (idxToInt ix))
--   in case pacc of
--        Avar _         -> base
--        Alet  _ _      -> base
--        Apply _ _      -> base
--        Acond _ _ _    -> base
--        Atuple _       -> base
--        Aprj _ _       -> base
--        _              -> ann <+> base

