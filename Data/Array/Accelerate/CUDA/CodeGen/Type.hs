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

  -- working with reified dictionaries
  TypeOf(..),
  signedIntegralNum, unsignedIntegralNum,

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
import Language.C.Quote.CUDA
import qualified Data.Typeable                          as T
import qualified Language.C                             as C


class TypeOf a where
  typeOf        :: a -> C.Type
  texTypeOf     :: a -> C.Type

instance TypeOf (ScalarType a) where
  typeOf    = cScalarType
  texTypeOf = cScalarTypeTex

instance TypeOf (NumType a) where
  typeOf    = cNumType
  texTypeOf = cNumTypeTex

instance TypeOf (IntegralType a) where
  typeOf    = cIntegralType
  texTypeOf = cIntegralTypeTex

instance TypeOf (FloatingType a) where
  typeOf    = cFloatingType
  texTypeOf = cFloatingTypeTex

instance TypeOf (NonNumType a) where
  typeOf    = cNonNumType
  texTypeOf = cNonNumTypeTex


-- Surface element types
-- ---------------------

accType :: DelayedOpenAcc aenv (Sugar.Array dim e) -> [C.Type]
accType = cTupleType . Sugar.delayedAccType

expType :: DelayedOpenExp aenv env t -> [C.Type]
expType = cTupleType . Sugar.preExpType Sugar.delayedAccType

segmentsType :: DelayedOpenAcc aenv (Sugar.Segments i) -> C.Type
segmentsType seg
  | [s] <- accType seg  = s
  | otherwise           = $internalError "accType" "non-scalar segment type"


eltType :: Sugar.Elt a => a {- dummy -} -> [C.Type]
eltType =  cTupleType . Sugar.eltType

eltTypeTex :: Sugar.Elt a => a {- dummy -} -> [C.Type]
eltTypeTex =  cTupleTypeTex . Sugar.eltType

eltSizeOf :: Sugar.Elt a => a {- dummy -} -> [Int]
eltSizeOf =  sizeOf' . Sugar.eltType
  where
    sizeOf' :: TupleType a -> [Int]
    sizeOf' UnitTuple           = []
    sizeOf' x@(SingleTuple _)   = [Sugar.sizeOf x]
    sizeOf' (PairTuple a b)     = sizeOf' a ++ sizeOf' b



cTupleType :: TupleType a -> [C.Type]
cTupleType UnitTuple         = []
cTupleType (SingleTuple  ty) = [cScalarType ty]
cTupleType (PairTuple t1 t0) = cTupleType t1 ++ cTupleType t0

cScalarType :: ScalarType a -> C.Type
cScalarType (NumScalarType    ty) = cNumType ty
cScalarType (NonNumScalarType ty) = cNonNumType ty

cNumType :: NumType a -> C.Type
cNumType (IntegralNumType ty) = cIntegralType ty
cNumType (FloatingNumType ty) = cFloatingType ty

cIntegralType :: IntegralType a -> C.Type
cIntegralType (TypeInt8    _) = typename "Int8"
cIntegralType (TypeInt16   _) = typename "Int16"
cIntegralType (TypeInt32   _) = typename "Int32"
cIntegralType (TypeInt64   _) = typename "Int64"
cIntegralType (TypeWord8   _) = typename "Word8"
cIntegralType (TypeWord16  _) = typename "Word16"
cIntegralType (TypeWord32  _) = typename "Word32"
cIntegralType (TypeWord64  _) = typename "Word64"
cIntegralType (TypeCShort  _) = [cty|short|]
cIntegralType (TypeCUShort _) = [cty|unsigned short|]
cIntegralType (TypeCInt    _) = [cty|int|]
cIntegralType (TypeCUInt   _) = [cty|unsigned int|]
cIntegralType (TypeCLong   _) = [cty|long int|]
cIntegralType (TypeCULong  _) = [cty|unsigned long int|]
cIntegralType (TypeCLLong  _) = [cty|long long int|]
cIntegralType (TypeCULLong _) = [cty|unsigned long long int|]
cIntegralType (TypeInt     _) = typename (T.showsTypeRep (T.typeOf (undefined::HTYPE_INT))  "")
cIntegralType (TypeWord    _) = typename (T.showsTypeRep (T.typeOf (undefined::HTYPE_WORD)) "")

cFloatingType :: FloatingType a -> C.Type
cFloatingType (TypeFloat   _) = [cty|float|]
cFloatingType (TypeCFloat  _) = [cty|float|]
cFloatingType (TypeDouble  _) = [cty|double|]
cFloatingType (TypeCDouble _) = [cty|double|]

cNonNumType :: NonNumType a -> C.Type
cNonNumType (TypeBool   _) = typename "Word8"
cNonNumType (TypeChar   _) = typename "Word32"
cNonNumType (TypeCChar  _) = [cty|char|]
cNonNumType (TypeCSChar _) = [cty|signed char|]
cNonNumType (TypeCUChar _) = [cty|unsigned char|]


-- Texture types
-- -------------

accTypeTex :: DelayedOpenAcc aenv (Sugar.Array dim e) -> [C.Type]
accTypeTex = cTupleTypeTex . Sugar.delayedAccType


-- Implementation
--
cTupleTypeTex :: TupleType a -> [C.Type]
cTupleTypeTex UnitTuple         = []
cTupleTypeTex (SingleTuple t)   = [cScalarTypeTex t]
cTupleTypeTex (PairTuple t1 t0) = cTupleTypeTex t1 ++ cTupleTypeTex t0

cScalarTypeTex :: ScalarType a -> C.Type
cScalarTypeTex (NumScalarType    ty) = cNumTypeTex ty
cScalarTypeTex (NonNumScalarType ty) = cNonNumTypeTex ty;

cNumTypeTex :: NumType a -> C.Type
cNumTypeTex (IntegralNumType ty) = cIntegralTypeTex ty
cNumTypeTex (FloatingNumType ty) = cFloatingTypeTex ty

cIntegralTypeTex :: IntegralType a -> C.Type
cIntegralTypeTex (TypeInt8    _) = typename "TexInt8"
cIntegralTypeTex (TypeInt16   _) = typename "TexInt16"
cIntegralTypeTex (TypeInt32   _) = typename "TexInt32"
cIntegralTypeTex (TypeInt64   _) = typename "TexInt64"
cIntegralTypeTex (TypeWord8   _) = typename "TexWord8"
cIntegralTypeTex (TypeWord16  _) = typename "TexWord16"
cIntegralTypeTex (TypeWord32  _) = typename "TexWord32"
cIntegralTypeTex (TypeWord64  _) = typename "TexWord64"
cIntegralTypeTex (TypeCShort  _) = typename "TexCShort"
cIntegralTypeTex (TypeCUShort _) = typename "TexCUShort"
cIntegralTypeTex (TypeCInt    _) = typename "TexCInt"
cIntegralTypeTex (TypeCUInt   _) = typename "TexCUInt"
cIntegralTypeTex (TypeCLong   _) = typename "TexCLong"
cIntegralTypeTex (TypeCULong  _) = typename "TexCULong"
cIntegralTypeTex (TypeCLLong  _) = typename "TexCLLong"
cIntegralTypeTex (TypeCULLong _) = typename "TexCULLong"
cIntegralTypeTex (TypeInt     _) = typename ("TexInt"  ++ show (finiteBitSize (undefined::Int)))
cIntegralTypeTex (TypeWord    _) = typename ("TexWord" ++ show (finiteBitSize (undefined::Word)))

cFloatingTypeTex :: FloatingType a -> C.Type
cFloatingTypeTex (TypeFloat   _) = typename "TexFloat"
cFloatingTypeTex (TypeCFloat  _) = typename "TexCFloat"
cFloatingTypeTex (TypeDouble  _) = typename "TexDouble"
cFloatingTypeTex (TypeCDouble _) = typename "TexCDouble"


cNonNumTypeTex :: NonNumType a -> C.Type
cNonNumTypeTex (TypeBool   _) = typename "TexWord8"
cNonNumTypeTex (TypeChar   _) = typename "TexWord32"
cNonNumTypeTex (TypeCChar  _) = typename "TexCChar"
cNonNumTypeTex (TypeCSChar _) = typename "TexCSChar"
cNonNumTypeTex (TypeCUChar _) = typename "TexCUChar"


-- Utilities
-- ---------

typename :: String -> C.Type
typename name = [cty| typename $id:name |]

signedIntegralNum :: IntegralType a -> Bool
signedIntegralNum t =
  case t of
    TypeInt _    -> True
    TypeInt8 _   -> True
    TypeInt16 _  -> True
    TypeInt32 _  -> True
    TypeInt64 _  -> True
    TypeCShort _ -> True
    TypeCInt _   -> True
    TypeCLong _  -> True
    TypeCLLong _ -> True
    _            -> False

unsignedIntegralNum :: IntegralType a -> Bool
unsignedIntegralNum = not . signedIntegralNum

