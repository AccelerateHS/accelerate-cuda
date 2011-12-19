-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Data
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--
-- Common data types for code generation
--

module Data.Array.Accelerate.CUDA.CodeGen.Data (Macro, CUTranslSkel(..)) where

import Language.C
import Text.PrettyPrint.Mainland

type Macro              = (Id, Maybe Exp)
data CUTranslSkel       = CUTranslSkel [DeclSpec] [Macro] FilePath

instance Pretty CUTranslSkel where
  ppr (CUTranslSkel code defs skel) =
    stack [ include "accelerate_cuda_extras.h"
          , stack (map macro defs)
          , ppr code
          , include skel
          ]


include :: FilePath -> Doc
include hdr = text "#include <" <> text hdr <> text ">"

macro :: Macro -> Doc
macro (d,v) = text "#define" <+> ppr d
                             <+> maybe empty (parens . ppr) v

