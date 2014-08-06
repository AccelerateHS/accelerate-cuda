{-# LANGUAGE GADTs           #-}
{-# LANGUAGE PatternGuards   #-}
{-# LANGUAGE QuasiQuotes     #-}
{-# LANGUAGE TemplateHaskell #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
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
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Trafo
import qualified Data.Array.Accelerate.Array.Sugar      as Sugar
import qualified Data.Array.Accelerate.Analysis.Type    as Sugar

-- libraries
import Data.Bits
import Data.Typeable
import Language.C.Quote.CUDA
import qualified Language.C                             as C


typename :: String -> C.Type
typename name = [cty| typename $id:name |]

-- Surface element types
-- ---------------------


accType :: DelayedOpenAcc aenv (Sugar.Array dim e) -> [C.Type]
accType = codegenTupleType . Sugar.delayedAccType

expType :: DelayedOpenExp aenv env t -> [C.Type]
expType = codegenTupleType . Sugar.preExpType Sugar.delayedAccType

segmentsType :: DelayedOpenAcc aenv (Sugar.Segments i) -> C.Type
segmentsType seg
  | [s] <- accType seg  = s
  | otherwise           = $internalError "accType" "non-scalar segment type"


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
codegenIntegralType (TypeInt8    _) = typename "Int8"
codegenIntegralType (TypeInt16   _) = typename "Int16"
codegenIntegralType (TypeInt32   _) = typename "Int32"
codegenIntegralType (TypeInt64   _) = typename "Int64"
codegenIntegralType (TypeWord8   _) = typename "Word8"
codegenIntegralType (TypeWord16  _) = typename "Word16"
codegenIntegralType (TypeWord32  _) = typename "Word32"
codegenIntegralType (TypeWord64  _) = typename "Word64"
codegenIntegralType (TypeCShort  _) = [cty|short|]
codegenIntegralType (TypeCUShort _) = [cty|unsigned short|]
codegenIntegralType (TypeCInt    _) = [cty|int|]
codegenIntegralType (TypeCUInt   _) = [cty|unsigned int|]
codegenIntegralType (TypeCLong   _) = [cty|long int|]
codegenIntegralType (TypeCULong  _) = [cty|unsigned long int|]
codegenIntegralType (TypeCLLong  _) = [cty|long long int|]
codegenIntegralType (TypeCULLong _) = [cty|unsigned long long int|]
codegenIntegralType (TypeInt     _) = typename (showsTypeRep (typeOf (undefined::HTYPE_INT))  "")
codegenIntegralType (TypeWord    _) = typename (showsTypeRep (typeOf (undefined::HTYPE_WORD)) "")

codegenFloatingType :: FloatingType a -> C.Type
codegenFloatingType (TypeFloat   _) = [cty|float|]
codegenFloatingType (TypeCFloat  _) = [cty|float|]
codegenFloatingType (TypeDouble  _) = [cty|double|]
codegenFloatingType (TypeCDouble _) = [cty|double|]

codegenNonNumType :: NonNumType a -> C.Type
codegenNonNumType (TypeBool   _) = typename "Word8"
codegenNonNumType (TypeChar   _) = typename "Word32"
codegenNonNumType (TypeCChar  _) = [cty|char|]
codegenNonNumType (TypeCSChar _) = [cty|signed char|]
codegenNonNumType (TypeCUChar _) = [cty|unsigned char|]


-- Texture types
-- -------------

accTypeTex :: DelayedOpenAcc aenv (Sugar.Array dim e) -> [C.Type]
accTypeTex = codegenTupleTex . Sugar.delayedAccType


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
codegenIntegralTex (TypeInt     _) = typename ("TexInt"  ++ show (finiteBitSize (undefined::Int)))
codegenIntegralTex (TypeWord    _) = typename ("TexWord" ++ show (finiteBitSize (undefined::Word)))

codegenFloatingTex :: FloatingType a -> C.Type
codegenFloatingTex (TypeFloat   _) = typename "TexFloat"
codegenFloatingTex (TypeCFloat  _) = typename "TexCFloat"
codegenFloatingTex (TypeDouble  _) = typename "TexDouble"
codegenFloatingTex (TypeCDouble _) = typename "TexCDouble"


codegenNonNumTex :: NonNumType a -> C.Type
codegenNonNumTex (TypeBool   _) = typename "TexWord8"
codegenNonNumTex (TypeChar   _) = typename "TexWord32"
codegenNonNumTex (TypeCChar  _) = typename "TexCChar"
codegenNonNumTex (TypeCSChar _) = typename "TexCSChar"
codegenNonNumTex (TypeCUChar _) = typename "TexCUChar"

