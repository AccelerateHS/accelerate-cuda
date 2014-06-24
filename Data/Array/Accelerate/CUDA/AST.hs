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

  module Data.Array.Accelerate.AST,

  AccKernel(..), Free, Gamma(..), Idx_(..),
  ExecAcc, ExecAfun, ExecOpenAfun, ExecOpenAcc(..),
  ExecExp, ExecFun, ExecOpenExp, ExecOpenFun,
  ExecLoop(..), ExecP(..), ExecT(..), ExecC(..),
  freevar, makeEnvMap,

) where

-- friends
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Pretty
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, Arrays, Vector, EltRepr )
import Data.Array.Accelerate.Array.Representation       ( SliceIndex(..) )
import qualified Data.Array.Accelerate.CUDA.FullList    as FL
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Analysis                  as CUDA

-- system
import Text.PrettyPrint
import Data.Hashable
import Data.IORef                                       ( IORef )
import Data.Monoid                                      ( Monoid(..) )
import qualified Data.HashSet                           as Set
import qualified Data.HashMap.Strict                    as Map


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
-- within scalar expressions. These arrays are are required to execute the
-- kernel, by binding to texture references to similar.
--
type Free aenv = Set.HashSet (Idx_ aenv)

freevar :: (Shape sh, Elt e) => Idx aenv (Array sh e) -> Free aenv
freevar = Set.singleton . Idx_


-- A mapping between environment indexes and some token identifying that array
-- in the generated code. This simply compresses the sequence of array indices
-- into a continuous range, rather than directly using the integer equivalent of
-- the de Bruijn index.
--
-- This results in generated code that is (slightly) less sensitive to the
-- placement of let bindings, ultimately leading to a higher hit rate in the
-- compilation cache.
--
newtype Gamma aenv = Gamma ( Map.HashMap (Idx_ aenv) Int )
  deriving ( Monoid )

makeEnvMap :: Free aenv -> Gamma aenv
makeEnvMap indices
  = Gamma
  . Map.fromList
  . flip zip [0..]
--  . sortBy (compare `on` idxType)
  $ Set.toList indices
--  where
--    idxType :: Idx_ aenv -> TypeRep
--    idxType (Idx_ (_ :: Idx aenv (Array sh e))) = typeOf (undefined :: e)


-- Opaque array environment indices
--
data Idx_ aenv where
  Idx_ :: (Shape sh, Elt e) => Idx aenv (Array sh e) -> Idx_ aenv

instance Eq (Idx_ aenv) where
  Idx_ ix1 == Idx_ ix2 = idxToInt ix1 == idxToInt ix2

instance Hashable (Idx_ aenv) where
  hashWithSalt salt (Idx_ ix)
    = salt `hashWithSalt` idxToInt ix


-- Interleave compilation & execution state annotations into an open array
-- computation AST
--
data ExecOpenAcc aenv a where
  ExecAcc   :: {-# UNPACK #-} !(FL.FullList () (AccKernel a))   -- executable binary objects
            -> !(Gamma aenv)                                    -- free array variables the kernel needs access to
            -> !(PreOpenAcc ExecOpenAcc aenv a)                 -- the actual computation
            -> ExecOpenAcc aenv a                               -- the recursive knot

  EmbedAcc  :: (Shape sh, Elt e)
            => !(PreExp ExecOpenAcc aenv sh)                    -- shape of the result array, used by execution
            -> ExecOpenAcc aenv (Array sh e)

  ExecLoop :: Arrays arrs
           => ExecLoop aenv () arrs
           -> ExecOpenAcc aenv arrs


-- An annotated AST suitable for execution in the CUDA environment
--
type ExecAcc  a         = ExecOpenAcc () a
type ExecAfun a         = PreAfun ExecOpenAcc a
type ExecOpenAfun aenv a = PreOpenAfun ExecOpenAcc aenv a

type ExecOpenExp        = PreOpenExp ExecOpenAcc
type ExecOpenFun        = PreOpenFun ExecOpenAcc

type ExecExp            = ExecOpenExp ()
type ExecFun            = ExecOpenFun ()


-- Display the annotated AST
-- -------------------------

instance Show (ExecOpenAcc aenv a) where
  show = render . prettyExecAcc 0 noParens

instance Show (ExecAfun a) where
  show = render . prettyExecAfun 0


prettyLoop :: Int -> ExecLoop aenv lenv arrs -> Doc
prettyLoop alvl l = text "loop" -- TODO

prettyExecAfun :: Int -> ExecAfun a -> Doc
prettyExecAfun alvl pfun = prettyPreAfun prettyExecAcc alvl pfun

prettyExecAcc :: PrettyAcc ExecOpenAcc
prettyExecAcc alvl wrap exec =
  case exec of
    EmbedAcc sh ->
      wrap $ hang (text "Embedded") 2
           $ sep [ prettyPreExp prettyExecAcc 0 alvl parens sh ]

    ExecAcc _ (Gamma fv) pacc ->
      let base      = prettyPreAcc prettyExecAcc alvl wrap pacc
          ann       = braces (freevars (Map.keys fv))
          freevars  = (text "fv=" <>) . brackets . hcat . punctuate comma
                                      . map (\(Idx_ ix) -> char 'a' <> int (idxToInt ix))
      in
      case pacc of
        Avar{}          -> base
        Alet{}          -> base
        Apply{}         -> base
        Acond{}         -> base
        Atuple{}        -> base
        Aprj{}          -> base
        _               -> ann <+> base

    ExecLoop l -> prettyLoop alvl l

data ExecLoop aenv lenv arrs where
  ExecEmpty :: ExecLoop aenv lenv ()
  ExecP :: (Arrays a, Arrays arrs) => ExecP aenv      a -> ExecLoop aenv (lenv, a) arrs -> ExecLoop aenv lenv  arrs
  ExecT :: (Arrays a, Arrays arrs) => ExecT aenv lenv a -> ExecLoop aenv (lenv, a) arrs -> ExecLoop aenv lenv  arrs
  ExecC :: (Arrays a, Arrays arrs) => ExecC aenv lenv a -> ExecLoop aenv  lenv     arrs -> ExecLoop aenv lenv (arrs, a)

data ExecP aenv a where

  ExecToStream :: (Elt slix, Shape sl, Shape sh, Elt e)
               => SliceIndex (EltRepr slix)
                             (EltRepr sl)
                             co
                             (EltRepr sh)
               -> ExecExp aenv slix
               -> ExecOpenAcc aenv (Array sh e)
               -> AccKernel (Array sl e)
               -> !(Gamma aenv)
               -> IORef (Maybe slix, slix, sl)
               -> ExecP aenv (Array sl e)

  ExecUseLazy :: (Elt slix, Shape sl, Shape sh, Elt e)
              => SliceIndex (EltRepr slix)
                            (EltRepr sl)
                            co
                            (EltRepr sh)
              -> ExecExp aenv slix
              -> Array sh e
              -> IORef (Maybe slix, slix, sl)
              -> ExecP aenv (Array sl e)

data ExecT aenv lenv a where
  ExecMap :: (Shape sh, Elt e, Shape sh', Elt e')
          => ExecOpenAfun aenv (Array sh e -> Array sh' e')
          -> Idx lenv (Array sh e)
          -> ExecT aenv lenv (Array sh' e')

  ExecZipWith :: (Shape sh, Elt e, Shape sh'', Elt e'', Shape sh', Elt e')
          => ExecOpenAfun aenv (Array sh e -> Array sh'' e'' -> Array sh' e')
          -> Idx lenv (Array sh e)
          -> Idx lenv (Array sh'' e'')
          -> ExecT aenv lenv (Array sh' e')

  ExecScanStream :: (Shape sh, Elt e)
                 => ExecOpenAfun aenv (Array sh e -> Array sh e -> Array sh e)
                 -> ExecOpenAcc aenv (Array sh e)
                 -> Idx lenv (Array sh e)
                 -> IORef (Array sh e)
                 -> ExecT aenv lenv (Array sh e)

  ExecScanStreamAct :: (Shape sh, Elt e, Shape sh', Elt e')
                    => ExecOpenAfun aenv (Array sh e -> Array sh' e' -> Array sh e)
                    -> ExecOpenAfun aenv (Array sh' e' -> Array sh' e' -> Array sh' e')
                    -> ExecOpenAcc aenv (Array sh e)
                    -> Idx lenv (Array sh' e')
                    -> IORef (Array sh e)
                    -> ExecT aenv lenv (Array sh e)

data ExecC aenv lenv a where
  ExecFoldStream :: (Shape sh, Elt e)
                 => ExecOpenAfun aenv (Array sh e -> Array sh e -> Array sh e)
                 -> ExecOpenAcc aenv (Array sh e)
                 -> Idx lenv (Array sh e)
                 -> IORef (Array sh e)
                 -> ExecC aenv lenv (Array sh e)

  ExecFromStream :: (Shape sh, Elt e)
                 => AccKernel (Vector e)
                 -> Idx lenv (Array sh e)
                 -> IORef ([Array sh e])
                 -> ExecC aenv lenv (Vector sh, Vector e)

  ExecFoldStreamAct :: (Shape sh, Elt e, Shape sh', Elt e')
                    => ExecOpenAfun aenv (Array sh e -> Array sh' e' -> Array sh e)
                    -> ExecOpenAfun aenv (Array sh' e' -> Array sh' e' -> Array sh' e')
                    -> ExecOpenAcc aenv (Array sh e)
                    -> Idx lenv (Array sh' e')
                    -> IORef (Array sh e)
                    -> ExecC aenv lenv (Array sh e)

  ExecFoldStreamFlatten :: (Shape sh, Elt e, Shape sh', Elt e')
                        => ExecOpenAfun aenv (Array sh e -> Vector sh' -> Vector e' -> Array sh e)
                        -> ExecOpenAcc aenv (Array sh e)
                        -> Idx lenv (Array sh' e')
                        -> IORef (Array sh e)
                        -> ExecC aenv lenv (Array sh e)

  ExecCollectStream :: (Shape sh, Elt e)
                    => (Array sh e -> IO ())
                    -> Idx lenv (Array sh e)
                    -> ExecC aenv lenv ()

