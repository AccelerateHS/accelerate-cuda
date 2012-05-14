{-# LANGUAGE CPP         #-}
{-# LANGUAGE GADTs       #-}
{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- Generate types for the reified elements of an array computation
--

module Data.Array.Accelerate.CUDA.CodeGen.Type (

  -- surface types
  accType, accTypeTex, segmentsType, expType,
  eltType, eltTypeTex, eltSizeOf,

  -- primitive bits...
  codegenIntegralType, codegenScalarType

) where

-- friends
import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.CUDA.CodeGen.Base
import qualified Data.Array.Accelerate.Array.Sugar      as Sugar
import qualified Data.Array.Accelerate.Analysis.Type    as Sugar

-- libraries
import Language.C.Quote.CUDA
import qualified Language.C                             as C
import qualified Foreign.Storable                       as F


#include "accelerate.h"


-- Surface element types
-- ---------------------

accType :: OpenAcc aenv (Sugar.Array dim e) -> [C.Type]
accType =  codegenTupleType . Sugar.accType

expType :: OpenExp aenv env t -> [C.Type]
expType =  codegenTupleType . Sugar.expType

segmentsType :: OpenAcc aenv (Sugar.Segments i) -> C.Type
segmentsType seg
  | [s] <- accType seg  = s
  | otherwise           = INTERNAL_ERROR(error) "accType" "non-scalar segment type"


eltType :: Sugar.Elt a => a {- dummy -} -> [C.Type]
eltType =  codegenTupleType . Sugar.eltType

eltTypeTex :: Sugar.Elt a => a {- dummy -} -> [C.Type]
eltTypeTex =  codegenTupleTex . Sugar.eltType

eltSizeOf :: Sugar.Elt a => a {- dummy -} -> [Int]
eltSizeOf =  sizeOf' . Sugar.eltType
  where
    sizeOf' :: TupleType a -> [Int]
    sizeOf' UnitTuple           = []
    sizeOf' x@(SingleTuple _)   = [Sugar.sizeOf x]
    sizeOf' (PairTuple a b)     = sizeOf' a ++ sizeOf' b


-- Implementation
--
codegenTupleType :: TupleType a -> [C.Type]
codegenTupleType UnitTuple         = []
codegenTupleType (SingleTuple  ty) = [codegenScalarType ty]
codegenTupleType (PairTuple t1 t0) = codegenTupleType t1 ++ codegenTupleType t0

codegenScalarType :: ScalarType a -> C.Type
codegenScalarType (NumScalarType    ty) = codegenNumType ty
codegenScalarType (NonNumScalarType ty) = codegenNonNumType ty

codegenNumType :: NumType a -> C.Type
codegenNumType (IntegralNumType ty) = codegenIntegralType ty
codegenNumType (FloatingNumType ty) = codegenFloatingType ty

codegenIntegralType :: IntegralType a -> C.Type
codegenIntegralType (TypeInt8    _) = typename "int8_t"
codegenIntegralType (TypeInt16   _) = typename "int16_t"
codegenIntegralType (TypeInt32   _) = typename "int32_t"
codegenIntegralType (TypeInt64   _) = typename "int64_t"
codegenIntegralType (TypeWord8   _) = typename "uint8_t"
codegenIntegralType (TypeWord16  _) = typename "uint16_t"
codegenIntegralType (TypeWord32  _) = typename "uint32_t"
codegenIntegralType (TypeWord64  _) = typename "uint64_t"
codegenIntegralType (TypeCShort  _) = [cty|short|]
codegenIntegralType (TypeCUShort _) = [cty|unsigned short|]
codegenIntegralType (TypeCInt    _) = [cty|int|]
codegenIntegralType (TypeCUInt   _) = [cty|unsigned int|]
codegenIntegralType (TypeCLong   _) = [cty|long int|]
codegenIntegralType (TypeCULong  _) = [cty|unsigned long int|]
codegenIntegralType (TypeCLLong  _) = [cty|long long int|]
codegenIntegralType (TypeCULLong _) = [cty|unsigned long long int|]
#if SIZEOF_HSINT == 4
codegenIntegralType (TypeInt     _) = typename "int32_t"
#elif SIZEOF_HSINT == 8
codegenIntegralType (TypeInt     _) = typename "int64_t"
#endif
#if SIZEOF_HSINT == 4
codegenIntegralType (TypeWord    _) = typename "uint32_t"
#elif SIZEOF_HSINT == 8
codegenIntegralType (TypeWord    _) = typename "uint64_t"
#endif

codegenFloatingType :: FloatingType a -> C.Type
codegenFloatingType (TypeFloat   _) = [cty|float|]
codegenFloatingType (TypeCFloat  _) = [cty|float|]
codegenFloatingType (TypeDouble  _) = [cty|double|]
codegenFloatingType (TypeCDouble _) = [cty|double|]

codegenNonNumType :: NonNumType a -> C.Type
codegenNonNumType (TypeBool   _) = error "codegenNonNum :: Bool"
codegenNonNumType (TypeChar   _) = error "codegenNonNum :: Char"
codegenNonNumType (TypeCChar  _) = [cty|char|]
codegenNonNumType (TypeCSChar _) = [cty|signed char|]
codegenNonNumType (TypeCUChar _) = [cty|unsigned char|]


-- Texture types
-- -------------

accTypeTex :: OpenAcc aenv (Sugar.Array dim e) -> [C.Type]
accTypeTex = codegenTupleTex . Sugar.accType


-- Implementation
--
codegenTupleTex :: TupleType a -> [C.Type]
codegenTupleTex UnitTuple         = []
codegenTupleTex (SingleTuple t)   = [codegenScalarTex t]
codegenTupleTex (PairTuple t1 t0) = codegenTupleTex t1 ++ codegenTupleTex t0

codegenScalarTex :: ScalarType a -> C.Type
codegenScalarTex (NumScalarType    ty) = codegenNumTex ty
codegenScalarTex (NonNumScalarType ty) = codegenNonNumTex ty;

codegenNumTex :: NumType a -> C.Type
codegenNumTex (IntegralNumType ty) = codegenIntegralTex ty
codegenNumTex (FloatingNumType ty) = codegenFloatingTex ty

codegenIntegralTex :: IntegralType a -> C.Type
codegenIntegralTex (TypeInt8    _) = typename "TexInt8"
codegenIntegralTex (TypeInt16   _) = typename "TexInt16"
codegenIntegralTex (TypeInt32   _) = typename "TexInt32"
codegenIntegralTex (TypeInt64   _) = typename "TexInt64"
codegenIntegralTex (TypeWord8   _) = typename "TexWord8"
codegenIntegralTex (TypeWord16  _) = typename "TexWord16"
codegenIntegralTex (TypeWord32  _) = typename "TexWord32"
codegenIntegralTex (TypeWord64  _) = typename "TexWord64"
codegenIntegralTex (TypeCShort  _) = typename "TexCShort"
codegenIntegralTex (TypeCUShort _) = typename "TexCUShort"
codegenIntegralTex (TypeCInt    _) = typename "TexCInt"
codegenIntegralTex (TypeCUInt   _) = typename "TexCUInt"
codegenIntegralTex (TypeCLong   _) = typename "TexCLong"
codegenIntegralTex (TypeCULong  _) = typename "TexCULong"
codegenIntegralTex (TypeCLLong  _) = typename "TexCLLong"
codegenIntegralTex (TypeCULLong _) = typename "TexCULLong"
#if   SIZEOF_HSINT == 4
codegenIntegralTex (TypeInt     _) = typename "TexInt32"
#elif SIZEOF_HSINT == 8
codegenIntegralTex (TypeInt     _) = typename "TexInt64"
#endif
#if   SIZEOF_HSINT == 4
codegenIntegralTex (TypeWord    _) = typename "TexWord32"
#elif SIZEOF_HSINT == 8
codegenIntegralTex (TypeWord    _) = typename "TexWord64"
#endif


codegenFloatingTex :: FloatingType a -> C.Type
codegenFloatingTex (TypeFloat   _) = typename "TexFloat"
codegenFloatingTex (TypeCFloat  _) = typename "TexCFloat"
codegenFloatingTex (TypeDouble  _) = typename "TexDouble"
codegenFloatingTex (TypeCDouble _) = typename "TexCDouble"


-- TLM 2010-06-29:
--   Bool and Char can be implemented once the array types in
--   Data.Array.Accelerate.[CUDA.]Array.Data are made concrete.
--
codegenNonNumTex :: NonNumType a -> C.Type
codegenNonNumTex (TypeBool   _) = error "codegenNonNumTex :: Bool"
codegenNonNumTex (TypeChar   _) = error "codegenNonNumTex :: Char"
codegenNonNumTex (TypeCChar  _) = typename "TexCChar"
codegenNonNumTex (TypeCSChar _) = typename "TexCSChar"
codegenNonNumTex (TypeCUChar _) = typename "TexCUChar"

