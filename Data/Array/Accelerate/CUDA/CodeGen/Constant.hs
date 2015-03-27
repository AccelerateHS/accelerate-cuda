{-# LANGUAGE GADTs           #-}
{-# LANGUAGE PatternGuards   #-}
{-# LANGUAGE QuasiQuotes     #-}
{-# LANGUAGE TemplateHaskell #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Constant
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Constant
  where

-- friends
import Data.Array.Accelerate.AST                        ( PrimConst(..) )
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type

-- libraries
import Data.Loc
import Data.Char
import Language.C
import Language.C.Quote.CUDA


-- | A constant value. Note that this follows the EltRepr representation of the
-- type, meaning that any nested tupling on the surface type is flattened.
--
constant :: TupleType a -> a -> [Exp]
constant UnitTuple           _      = []
constant (SingleTuple ty)    c      = [scalar ty c]
constant (PairTuple ty1 ty0) (cs,c) = constant ty1 cs ++ constant ty0 c

-- | A constant scalar value
--
scalar :: ScalarType a -> a -> Exp
scalar (NumScalarType    ty) = num ty
scalar (NonNumScalarType ty) = nonnum ty

-- | A constant numeric value
--
num :: NumType a -> a -> Exp
num (IntegralNumType ty) = integral ty
num (FloatingNumType ty) = floating ty

-- | A constant integral value
--
integral :: IntegralType a -> a -> Exp
integral ty x | IntegralDict <- integralDict ty = [cexp| ( $ty:(typeOf ty) ) $exp:(cintegral x) |]

-- | A constant floating-point value
--
floating :: FloatingType a -> a -> Exp
floating (TypeFloat   _) x = Const (FloatConst (shows x "f") (toRational x) noLoc) noLoc
floating (TypeCFloat  _) x = Const (FloatConst (shows x "f") (toRational x) noLoc) noLoc
floating (TypeDouble  _) x = Const (DoubleConst (show x) (toRational x) noLoc) noLoc
floating (TypeCDouble _) x = Const (DoubleConst (show x) (toRational x) noLoc) noLoc


-- | A constant non-numeric value
--
nonnum :: NonNumType a -> a -> Exp
nonnum (TypeBool   _) x = cbool x
nonnum (TypeChar   _) x = [cexp|$char:x|]
nonnum (TypeCChar  _) x = [cexp|$char:(chr (fromIntegral x))|]
nonnum (TypeCUChar _) x = [cexp|$char:(chr (fromIntegral x))|]
nonnum (TypeCSChar _) x = [cexp|$char:(chr (fromIntegral x))|]


-- | Primitive constants
--
primConst :: PrimConst t -> Exp
primConst (PrimMinBound t) = primMinBound t
primConst (PrimMaxBound t) = primMaxBound t
primConst (PrimPi t)       = primPi t

primMinBound :: BoundedType a -> Exp
primMinBound (IntegralBoundedType ty) | IntegralDict <- integralDict ty = integral ty minBound
primMinBound (NonNumBoundedType   ty) | NonNumDict   <- nonNumDict   ty = nonnum   ty minBound

primMaxBound :: BoundedType a -> Exp
primMaxBound (IntegralBoundedType ty) | IntegralDict <- integralDict ty = integral ty maxBound
primMaxBound (NonNumBoundedType   ty) | NonNumDict   <- nonNumDict   ty = nonnum   ty maxBound

primPi :: FloatingType a -> Exp
primPi ty | FloatingDict <- floatingDict ty = floating ty pi

