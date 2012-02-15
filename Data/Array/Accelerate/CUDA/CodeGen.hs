{-# LANGUAGE CPP, GADTs, PatternGuards, ScopedTypeVariables, QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen (

  CUTranslSkel, codegenAcc

) where

-- libraries
import Prelude                                                  hiding ( exp )
import Data.Loc
import Data.Char
import Data.Symbol
import Control.Applicative                                      hiding ( Const )
import Text.PrettyPrint.Mainland
import Language.C.Syntax                                        ( Const(..) )
import Language.C.Quote.CUDA
import qualified Data.HashSet                                   as Set
import qualified Language.C                                     as C
import qualified Language.C.Syntax
import qualified Foreign.Storable                               as F
import qualified Foreign.CUDA.Analysis                          as CUDA

-- friends
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Tuple
import Data.Array.Accelerate.Pretty                             ()
import Data.Array.Accelerate.Analysis.Type
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.Analysis.Stencil
import Data.Array.Accelerate.Array.Representation
import qualified Data.Array.Accelerate.Array.Sugar              as Sugar

import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Monad
import Data.Array.Accelerate.CUDA.CodeGen.Mapping
import Data.Array.Accelerate.CUDA.CodeGen.IndexSpace
import Data.Array.Accelerate.CUDA.CodeGen.PrefixSum
import Data.Array.Accelerate.CUDA.CodeGen.Reduction
import Data.Array.Accelerate.CUDA.CodeGen.Stencil

#include "accelerate.h"


-- Array expressions
-- -----------------

-- | Instantiate an array computation with a set of concrete function and type
-- definitions to fix the parameters of an algorithmic skeleton. The generated
-- code can then be pretty-printed to file, and compiled to object code
-- executable on the device.
--
-- The code generator needs to include binding points for array references from
-- scalar code. We require that the only array form allowed within expressions
-- are array variables.
--
-- TODO: include a measure of how much shared memory a kernel requires.
--
codegenAcc :: forall aenv a.
              CUDA.DeviceProperties
           -> OpenAcc aenv a
           -> AccBindings aenv
           -> CUTranslSkel
codegenAcc dev acc (AccBindings vars) =
  let fvars rest                = Set.foldr (\v vs -> liftAcc acc v ++ vs) rest vars
      extras                    = [cedecl| $esc:("#include <accelerate_cuda_extras.h>") |]
      CUTranslSkel entry code   = runCGM $ codegen acc
  in
  CUTranslSkel entry (extras : fvars code)
  where
    codegen :: OpenAcc aenv a -> CGM CUTranslSkel
    codegen (OpenAcc pacc) =
      case pacc of
        -- non-computation forms
        --
        Alet _ _          -> internalError
        Alet2 _ _         -> internalError
        Avar _            -> internalError
        Apply _ _         -> internalError
        Acond _ _ _       -> internalError
        PairArrays _ _    -> internalError
        Use _             -> internalError
        Unit _            -> internalError
        Reshape _ _       -> internalError

        -- computation nodes
        --
        Generate _ f      -> do
          f'    <- codegenFun f
          mkGenerate (accDim acc) (codegenAccType acc) f'

        Replicate sl _ a  ->
          let dimSl  = accDim a
              dimOut = accDim acc
              --
              extend :: SliceIndex slix sl co dim -> Int -> [C.Exp]
              extend (SliceNil)            _ = []
              extend (SliceAll   sliceIdx) n = mkPrj dimOut "dim" n : extend sliceIdx (n+1)
              extend (SliceFixed sliceIdx) n = extend sliceIdx (n+1)
          in
          mkReplicate dimSl dimOut (codegenAccType a) (reverse $ extend sl 0)

        Index sl a slix   ->
          let dimCo  = length (codegenExpType slix)
              dimSl  = accDim acc
              dimIn0 = accDim a
              --
              restrict :: SliceIndex slix sl co dim -> (Int,Int) -> [C.Exp]
              restrict (SliceNil)            _     = []
              restrict (SliceAll   sliceIdx) (m,n) = mkPrj dimSl "sl" n : restrict sliceIdx (m,n+1)
              restrict (SliceFixed sliceIdx) (m,n) = mkPrj dimCo "co" m : restrict sliceIdx (m+1,n)
          in
          mkSlice dimSl dimCo dimIn0 (codegenAccType a) (reverse $ restrict sl (0,0))

        Map f a           -> do
          f'    <- codegenFun f
          mkMap (codegenAccType acc) (codegenAccType a) f'

        ZipWith f a b     -> do
          f'    <- codegenFun f
          mkZipWith (accDim acc) (codegenAccType acc) (codegenAccType a) (codegenAccType b) f'

        Fold f e _        -> do
          e'    <- codegenExp e
          f'    <- codegenFun f
          case accDim acc of
            0   -> mkFoldAll dev (codegenAccType acc) f' (Just e')
            _   -> mkFold    dev (codegenAccType acc) f' (Just e')

        Fold1 f _         -> do
          f'    <- codegenFun f
          case accDim acc of
            0   -> mkFoldAll dev (codegenAccType acc) f' Nothing
            _   -> mkFold    dev (codegenAccType acc) f' Nothing

        FoldSeg f e _ s   -> do
          f'    <- codegenFun f
          e'    <- codegenExp e
          mkFoldSeg dev (accDim acc) (codegenAccSegmentsType s) (codegenAccType acc) f' (Just e')

        Fold1Seg f _ s    -> do
          f'    <- codegenFun f
          mkFoldSeg dev (accDim acc) (codegenAccSegmentsType s) (codegenAccType acc) f' Nothing

        Scanl f e _       -> do
          e'    <- codegenExp e
          f'    <- codegenFun f
          mkScanl dev (codegenExpType e) f' (Just e')

        Scanl' f e _      -> do
          e'    <- codegenExp e
          f'    <- codegenFun f
          mkScanl dev (codegenExpType e) f' (Just e')

        Scanl1 f _        -> do
          f'    <- codegenFun f
          mkScanl dev (codegenAccType acc) f' Nothing

        Scanr f e _       -> do
          e'    <- codegenExp e
          f'    <- codegenFun f
          mkScanr dev (codegenExpType e) f' (Just e')

        Scanr' f e _      -> do
          e'    <- codegenExp e
          f'    <- codegenFun f
          mkScanr dev (codegenExpType e) f' (Just e')

        Scanr1 f _        -> do
          f'    <- codegenFun f
          mkScanr dev (codegenAccType acc) f' Nothing

        Permute f _ g a   -> do
          f'    <- codegenFun f
          g'    <- codegenFun g
          mkPermute dev (accDim acc) (accDim a) (codegenAccType a) (sizeOfAccTypes a) f' g'

        Backpermute _ f a -> do
          f'    <- codegenFun f
          mkBackpermute (accDim acc) (accDim a) (codegenAccType a) f'

        Stencil f b a     -> do
          f'    <- codegenFun f
          mkStencil (accDim acc) (codegenAccType acc) (codegenAccTypeTex a) b0 i0 f'
          where
            b0  = codegenBoundary a b
            i0  = map Sugar.shapeToList (offsets f a)

        Stencil2 f b1 a1 b0 a0 -> do
          f'    <- codegenFun f
          mkStencil2 (accDim acc) (codegenAccType acc)
            (codegenAccTypeTex a1) (codegenBoundary a1 b1) i1
            (codegenAccTypeTex a0) (codegenBoundary a0 b0) i0 f'
          where
            (p1, p0)    = offsets2 f a1 a0
            i0          = map Sugar.shapeToList p0
            i1          = map Sugar.shapeToList p1


    -- Generate binding points (texture references and shapes) for arrays lifted
    -- from scalar expressions
    --
    liftAcc :: OpenAcc aenv a -> ArrayVar aenv -> [C.Definition]
    liftAcc _ (ArrayVar idx) =
      let avar    = OpenAcc (Avar idx)
          idx'    = show $ deBruijnToInt idx
          sh      = cshape ("sh" ++ idx') (accDim avar)
          ty      = codegenAccTypeTex avar
          arr n   = "arr" ++ idx' ++ "_a" ++ show (n::Int)
      in
      sh : zipWith (\t n -> cglobal t (arr n)) (reverse ty) [0..]

    -- Shapes are still represented as C structs, so we need to generate field
    -- indexing code for shapes
    --
    mkPrj :: Int -> String -> Int -> C.Exp
    mkPrj ndim var c
      | ndim <= 1   = cvar var
      | otherwise   = [cexp| $exp:v . $id:field |]
                        where v     = cvar var
                              field = 'a' : show c

    -- caffeine and misery
    --
    internalError =
      let msg = unlines ["unsupported array primitive", pretty 100 (nest 2 doc)]
          pac = show acc
          doc | length pac <= 250 = text pac
              | otherwise         = text (take 250 pac) <+> text "... {truncated}"
      in
      INTERNAL_ERROR(error) "codegenAcc" msg

-- code generation for stencil boundary conditions
--
codegenBoundary :: forall aenv dim e. Sugar.Elt e
                => OpenAcc aenv (Sugar.Array dim e) {- dummy -}
                -> Boundary (Sugar.EltRepr e)
                -> Boundary [C.Exp]
codegenBoundary _ (Constant c) = Constant $ codegenConst (Sugar.eltType (undefined::e)) c
codegenBoundary _ Clamp        = Clamp
codegenBoundary _ Mirror       = Mirror
codegenBoundary _ Wrap         = Wrap


-- Scalar Expressions
-- ------------------

-- Function abstraction
--
-- Although Accelerate includes lambda abstractions, it does not include a
-- general application form. That is, lambda abstractions of scalar expressions
-- are only introduced as arguments to collective operations, so lambdas are
-- always outermost, and can always be translated into plain C functions.
--
codegenFun :: OpenFun env aenv t -> CGM [C.Exp]
codegenFun (Lam  f) = codegenFun f
codegenFun (Body b) = codegenExp b


-- Embedded scalar computations
--
codegenExp :: forall env aenv t. OpenExp env aenv t -> CGM [C.Exp]
codegenExp exp =
  case exp of
    -- local binders and variable indices
    --
    -- NOTE: recording which variables are used is important, because the CUDA
    -- compiler will not eliminate variables that are initialised but never
    -- used. If this is a scalar type mark it as used immediately, otherwise
    -- wait until tuple projection picks out an individual element.
    --
    Let _ _             -> INTERNAL_ERROR(error) "codegenExp" "Let: not implemented yet"
    Var i               -> case ty of
      [t]       -> let x = var 0
                   in  use v 0 t x >> return [x]
      _         -> return $ map var [n-1, n-2 .. 0]
      where
        base    = 'x':shows v "_a"
        var x   = [cexp| $id:(base ++ show x) |]
        ty      = codegenTupleType (Sugar.eltType (undefined::t))
        n       = length ty
        v       = idxToInt i

    -- Constant values
    PrimConst c         -> return [codegenPrimConst c]
    Const c             -> return (codegenConst (Sugar.eltType (undefined::t)) c)

    -- Primitive scalar operations
    PrimApp f arg       -> do
      x                 <- codegenExp arg
      return [codegenPrim f x]

    -- Tuples
    Tuple t             -> codegenTup t
    Prj idx e           -> do
      e'                <- codegenExp e
      case subset (zip e' elt) of
        [(x,t)]         -> uncurry use (unexp x) t x >> return [x]
        xts             -> return $ fst (unzip xts)
      where
        ty      = expType e
        elt     = codegenTupleType ty
        subset  = reverse
                . take (length (codegenExpType exp))
                . drop (prjToInt idx ty)
                . reverse
        --
        --  this is total hax.
        unexp x = case show x of
          ('x':v:'_':'a':n) | [(v',[])] <- reads [v], [(n',[])] <- reads n
                -> (v', n')
          _     -> INTERNAL_ERROR(error) "codegenExp" "expected variable name"

    -- Conditional expression
    Cond p t e          -> do
      t'                <- codegenExp t
      e'                <- codegenExp e
      p'                <- codegenExp p >>= \ps ->
        case ps of
          [x]   -> bind [cty| typename bool |] x
          _     -> INTERNAL_ERROR(error) "codegenExp" "expected conditional predicate"
      --
      return $ zipWith (\a b -> [cexp| $exp:p' ? $exp:a : $exp:b|]) t' e'

    -- Array indices and shapes
    --
    -- NOTE: when deconstructing shape components, we need to check where the
    -- shape data comes from. If referencing a free array variable, we instead
    -- generate struct-indexing code for the appropriate global variable.
    --
    IndexNil            -> return []
    IndexAny            -> return []
    IndexCons sh sz     -> do
      sh'               <- codegenExp sh
      sz'               <- codegenExp sz
      return (sh' ++ sz')

    IndexHead sh@(Shape a)      -> do
      [var]                     <- codegenExp sh
      return $ if accDim a > 1  then [[cexp| $exp:var . $id:("a0") |]]
                                else [var]

    IndexTail sh@(Shape a)      -> do
      [var]                     <- codegenExp sh
      return $ let n = accDim a
               in  map (\c -> [cexp| $exp:var . $id:('a':show c) |]) [n-1, n-2 .. 1]

    IndexHead ix        -> do
      ix'               <- codegenExp ix
      return [last ix']

    IndexTail ix        -> do
      ix'               <- codegenExp ix
      return (init ix')

codegenExp (ShapeSize e)    = return $ ccall "size" (codegenExp e)
    -- Array shape and element indexing
    ShapeSize e         -> do
      sh'               <- codegenExp e
      return [ ccall "size" sh' ]

    Shape arr
      | OpenAcc (Avar a) <- arr -> return [ cvar ("sh" ++ show (idxToInt a)) ]
      | otherwise               -> INTERNAL_ERROR(error) "codegenExp" "expected array variable"

    IndexScalar arr ix
      | OpenAcc (Avar a) <- arr ->
        let avar        = show (idxToInt a)
            sh          = cvar ("sh"  ++ avar)
            array x     = cvar ("arr" ++ avar ++ "_a" ++ show x)
            elt         = codegenAccTypeTex arr
            n           = length elt
        in do
          ix'           <- codegenExp ix
          v             <- bind [cty| int |] (ccall "toIndex" [sh, ccall "shape" ix'])
          return $ zipWith (\t x -> indexArray t (array x) v) elt [n-1, n-2 .. 0]

      | otherwise                -> INTERNAL_ERROR(error) "codegenExp" "expected array variable"


-- Tuples are defined as snoc-lists, so generate code right-to-left
--
codegenTup :: Tuple (OpenExp env aenv) t -> CGM [C.Exp]
codegenTup NilTup          = return []
codegenTup (t `SnocTup` e) = (++) <$> codegenTup t <*> codegenExp e


-- Convert a tuple index into the corresponding integer. Since the internal
-- representation is flat, be sure to walk over all sub components when indexing
-- past nested tuples.
--
prjToInt :: TupleIdx t e -> TupleType a -> Int
prjToInt ZeroTupIdx     _                 = 0
prjToInt (SuccTupIdx i) (b `PairTuple` a) = length (codegenTupleType a) + prjToInt i b
prjToInt _ _ =
  INTERNAL_ERROR(error) "prjToInt" "inconsistent valuation"


-- Types
-- -----

-- Generate types for the reified elements of an array computation
--
codegenAccType :: OpenAcc aenv (Sugar.Array dim e) -> [C.Type]
codegenAccType =  codegenTupleType . accType

codegenExpType :: OpenExp aenv env t -> [C.Type]
codegenExpType =  codegenTupleType . expType

codegenAccSegmentsType :: OpenAcc aenv (Sugar.Segments i) -> C.Type
codegenAccSegmentsType seg
  | [s] <- codegenAccType seg   = s
  | otherwise                   = INTERNAL_ERROR(error) "codegenAcc" "non-scalar segment type"

sizeOfAccTypes :: OpenAcc aenv (Sugar.Array dim e) -> [Int]
sizeOfAccTypes = sizeOf' . accType
  where
    sizeOf' :: TupleType a -> [Int]
    sizeOf' UnitTuple           = []
    sizeOf' x@(SingleTuple _)   = [sizeOf x]
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

codegenIntegralType (TypeInt     _) =
  case F.sizeOf (undefined::Int) of
       4 -> typename "int32_t"
       8 -> typename "int64_t"
       _ -> error "we can never get here"

codegenIntegralType (TypeWord    _) =
  case F.sizeOf (undefined::Int) of
       4 -> typename "uint32_t"
       8 -> typename "uint64_t"
       _ -> error "we can never get here"


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
--
codegenAccTypeTex :: OpenAcc aenv (Sugar.Array dim e) -> [C.Type]
codegenAccTypeTex = codegenTupleTex . accType

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

codegenIntegralTex (TypeInt     _) =
  case F.sizeOf (undefined::Int) of
       4 -> typename "TexInt32"
       8 -> typename "TexInt64"
       _ -> error "we can never get here"

codegenIntegralTex (TypeWord    _) =
  case F.sizeOf (undefined::Word) of
       4 -> typename "TexWord32"
       8 -> typename "TexWord64"
       _ -> error "we can never get here"


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


-- Scalar Primitives
-- -----------------

codegenPrimConst :: PrimConst a -> C.Exp
codegenPrimConst (PrimMinBound ty) = codegenMinBound ty
codegenPrimConst (PrimMaxBound ty) = codegenMaxBound ty
codegenPrimConst (PrimPi       ty) = codegenPi ty


codegenPrim :: PrimFun p -> [C.Exp] -> C.Exp
codegenPrim (PrimAdd              _) [a,b] = [cexp|$exp:a + $exp:b|]
codegenPrim (PrimSub              _) [a,b] = [cexp|$exp:a - $exp:b|]
codegenPrim (PrimMul              _) [a,b] = [cexp|$exp:a * $exp:b|]
codegenPrim (PrimNeg              _) [a]   = [cexp| - $exp:a|]
codegenPrim (PrimAbs             ty) [a]   = codegenAbs ty a
codegenPrim (PrimSig             ty) [a]   = codegenSig ty a
codegenPrim (PrimQuot             _) [a,b] = [cexp|$exp:a / $exp:b|]
codegenPrim (PrimRem              _) [a,b] = [cexp|$exp:a % $exp:b|]
codegenPrim (PrimIDiv             _) [a,b] = ccall "idiv" [a,b]
codegenPrim (PrimMod              _) [a,b] = ccall "mod"  [a,b]
codegenPrim (PrimBAnd             _) [a,b] = [cexp|$exp:a & $exp:b|]
codegenPrim (PrimBOr              _) [a,b] = [cexp|$exp:a | $exp:b|]
codegenPrim (PrimBXor             _) [a,b] = [cexp|$exp:a ^ $exp:b|]
codegenPrim (PrimBNot             _) [a]   = [cexp|~ $exp:a|]
codegenPrim (PrimBShiftL          _) [a,b] = [cexp|$exp:a << $exp:b|]
codegenPrim (PrimBShiftR          _) [a,b] = [cexp|$exp:a >> $exp:b|]
codegenPrim (PrimBRotateL         _) [a,b] = ccall "rotateL" [a,b]
codegenPrim (PrimBRotateR         _) [a,b] = ccall "rotateR" [a,b]
codegenPrim (PrimFDiv             _) [a,b] = [cexp|$exp:a / $exp:b|]
codegenPrim (PrimRecip           ty) [a]   = codegenRecip ty a
codegenPrim (PrimSin             ty) [a]   = ccall (FloatingNumType ty `postfix` "sin")   [a]
codegenPrim (PrimCos             ty) [a]   = ccall (FloatingNumType ty `postfix` "cos")   [a]
codegenPrim (PrimTan             ty) [a]   = ccall (FloatingNumType ty `postfix` "tan")   [a]
codegenPrim (PrimAsin            ty) [a]   = ccall (FloatingNumType ty `postfix` "asin")  [a]
codegenPrim (PrimAcos            ty) [a]   = ccall (FloatingNumType ty `postfix` "acos")  [a]
codegenPrim (PrimAtan            ty) [a]   = ccall (FloatingNumType ty `postfix` "atan")  [a]
codegenPrim (PrimAsinh           ty) [a]   = ccall (FloatingNumType ty `postfix` "asinh") [a]
codegenPrim (PrimAcosh           ty) [a]   = ccall (FloatingNumType ty `postfix` "acosh") [a]
codegenPrim (PrimAtanh           ty) [a]   = ccall (FloatingNumType ty `postfix` "atanh") [a]
codegenPrim (PrimExpFloating     ty) [a]   = ccall (FloatingNumType ty `postfix` "exp")   [a]
codegenPrim (PrimSqrt            ty) [a]   = ccall (FloatingNumType ty `postfix` "sqrt")  [a]
codegenPrim (PrimLog             ty) [a]   = ccall (FloatingNumType ty `postfix` "log")   [a]
codegenPrim (PrimFPow            ty) [a,b] = ccall (FloatingNumType ty `postfix` "pow")   [a,b]
codegenPrim (PrimLogBase         ty) [a,b] = codegenLogBase ty a b
codegenPrim (PrimTruncate     ta tb) [a]   = codegenTruncate ta tb a
codegenPrim (PrimRound        ta tb) [a]   = codegenRound ta tb a
codegenPrim (PrimFloor        ta tb) [a]   = codegenFloor ta tb a
codegenPrim (PrimCeiling      ta tb) [a]   = codegenCeiling ta tb a
codegenPrim (PrimAtan2           ty) [a,b] = ccall (FloatingNumType ty `postfix` "atan2") [a,b]
codegenPrim (PrimLt               _) [a,b] = [cexp|$exp:a < $exp:b|]
codegenPrim (PrimGt               _) [a,b] = [cexp|$exp:a > $exp:b|]
codegenPrim (PrimLtEq             _) [a,b] = [cexp|$exp:a <= $exp:b|]
codegenPrim (PrimGtEq             _) [a,b] = [cexp|$exp:a >= $exp:b|]
codegenPrim (PrimEq               _) [a,b] = [cexp|$exp:a == $exp:b|]
codegenPrim (PrimNEq              _) [a,b] = [cexp|$exp:a != $exp:b|]
codegenPrim (PrimMax             ty) [a,b] = codegenMax ty a b
codegenPrim (PrimMin             ty) [a,b] = codegenMin ty a b
codegenPrim PrimLAnd                 [a,b] = [cexp|$exp:a && $exp:b|]
codegenPrim PrimLOr                  [a,b] = [cexp|$exp:a || $exp:b|]
codegenPrim PrimLNot                 [a]   = [cexp| ! $exp:a|]
codegenPrim PrimOrd                  [a]   = codegenOrd a
codegenPrim PrimChr                  [a]   = codegenChr a
codegenPrim PrimBoolToInt            [a]   = codegenBoolToInt a
codegenPrim (PrimFromIntegral ta tb) [a]   = codegenFromIntegral ta tb a

-- If the argument lists are not the correct length
codegenPrim _ _ =
  INTERNAL_ERROR(error) "codegenPrim" "inconsistent valuation"

-- Implementation of scalar primitives
--
codegenConst :: TupleType a -> a -> [C.Exp]
codegenConst UnitTuple           _      = []
codegenConst (SingleTuple ty)    c      = [codegenScalar ty c]
codegenConst (PairTuple ty1 ty0) (cs,c) = codegenConst ty1 cs ++ codegenConst ty0 c


-- Scalar constants
--
codegenScalar :: ScalarType a -> a -> C.Exp
codegenScalar (NumScalarType    ty) = codegenNumScalar ty
codegenScalar (NonNumScalarType ty) = codegenNonNumScalar ty

codegenNumScalar :: NumType a -> a -> C.Exp
codegenNumScalar (IntegralNumType ty) = codegenIntegralScalar ty
codegenNumScalar (FloatingNumType ty) = codegenFloatingScalar ty

codegenIntegralScalar :: IntegralType a -> a -> C.Exp
codegenIntegralScalar ty | IntegralDict <- integralDict ty = cintegral

codegenFloatingScalar :: FloatingType a -> a -> C.Exp
codegenFloatingScalar (TypeFloat   _) x = C.Const (FloatConst (shows x "f") (toRational x) noSrcLoc) noSrcLoc
codegenFloatingScalar (TypeCFloat  _) x = C.Const (FloatConst (shows x "f") (toRational x) noSrcLoc) noSrcLoc
codegenFloatingScalar (TypeDouble  _) x = C.Const (DoubleConst (show x) (toRational x) noSrcLoc) noSrcLoc
codegenFloatingScalar (TypeCDouble _) x = C.Const (DoubleConst (show x) (toRational x) noSrcLoc) noSrcLoc

codegenNonNumScalar :: NonNumType a -> a -> C.Exp
codegenNonNumScalar (TypeBool   _) x = cbool x
codegenNonNumScalar (TypeChar   _) x = [cexp|$char:x|]
codegenNonNumScalar (TypeCChar  _) x = [cexp|$char:(chr (fromIntegral x))|]
codegenNonNumScalar (TypeCUChar _) x = [cexp|$char:(chr (fromIntegral x))|]
codegenNonNumScalar (TypeCSChar _) x = [cexp|$char:(chr (fromIntegral x))|]


-- Constant methods of floating
--
codegenPi :: FloatingType a -> C.Exp
codegenPi ty | FloatingDict <- floatingDict ty = codegenFloatingScalar ty pi


-- Constant methods of bounded
--
codegenMinBound :: BoundedType a -> C.Exp
codegenMinBound (IntegralBoundedType ty) | IntegralDict <- integralDict ty = codegenIntegralScalar ty minBound
codegenMinBound (NonNumBoundedType   ty) | NonNumDict   <- nonNumDict   ty = codegenNonNumScalar   ty minBound


codegenMaxBound :: BoundedType a -> C.Exp
codegenMaxBound (IntegralBoundedType ty) | IntegralDict <- integralDict ty = codegenIntegralScalar ty maxBound
codegenMaxBound (NonNumBoundedType   ty) | NonNumDict   <- nonNumDict   ty = codegenNonNumScalar   ty maxBound


-- Methods from Num, Floating, Fractional and RealFrac
--
codegenAbs :: NumType a -> C.Exp -> C.Exp
codegenAbs ty@(IntegralNumType _) x = ccall (ty `postfix` "abs")  [x]
codegenAbs ty@(FloatingNumType _) x = ccall (ty `postfix` "fabs") [x]


codegenSig :: NumType a -> C.Exp -> C.Exp
codegenSig (IntegralNumType ty) = codegenIntegralSig ty
codegenSig (FloatingNumType ty) = codegenFloatingSig ty

codegenIntegralSig :: IntegralType a -> C.Exp -> C.Exp
codegenIntegralSig ty x = [cexp|$exp:x == $exp:zero ? $exp:zero : $exp:(ccall "copysign" [one,x]) |]
  where
    zero | IntegralDict <- integralDict ty = codegenIntegralScalar ty 0
    one  | IntegralDict <- integralDict ty = codegenIntegralScalar ty 1

codegenFloatingSig :: FloatingType a -> C.Exp -> C.Exp
codegenFloatingSig ty x = [cexp|$exp:x == $exp:zero ? $exp:zero : $exp:(ccall (FloatingNumType ty `postfix` "copysign") [one,x]) |]
  where
    zero | FloatingDict <- floatingDict ty = codegenFloatingScalar ty 0
    one  | FloatingDict <- floatingDict ty = codegenFloatingScalar ty 1


codegenRecip :: FloatingType a -> C.Exp -> C.Exp
codegenRecip ty x | FloatingDict <- floatingDict ty = [cexp|$exp:(codegenFloatingScalar ty 1) / $exp:x|]


codegenLogBase :: FloatingType a -> C.Exp -> C.Exp -> C.Exp
codegenLogBase ty x y = let a = ccall (FloatingNumType ty `postfix` "log") [x]
                            b = ccall (FloatingNumType ty `postfix` "log") [y]
                        in
                        [cexp|$exp:b / $exp:a|]


codegenMin :: ScalarType a -> C.Exp -> C.Exp -> C.Exp
codegenMin (NumScalarType ty@(IntegralNumType _)) a b = ccall (ty `postfix` "min")  [a,b]
codegenMin (NumScalarType ty@(FloatingNumType _)) a b = ccall (ty `postfix` "fmin") [a,b]
codegenMin (NonNumScalarType _)                   a b =
  let ty = scalarType :: ScalarType Int32
  in  codegenMin ty (ccast ty a) (ccast ty b)


codegenMax :: ScalarType a -> C.Exp -> C.Exp -> C.Exp
codegenMax (NumScalarType ty@(IntegralNumType _)) a b = ccall (ty `postfix` "max")  [a,b]
codegenMax (NumScalarType ty@(FloatingNumType _)) a b = ccall (ty `postfix` "fmax") [a,b]
codegenMax (NonNumScalarType _)                   a b =
  let ty = scalarType :: ScalarType Int32
  in  codegenMax ty (ccast ty a) (ccast ty b)


-- Type coercions
--
codegenOrd :: C.Exp -> C.Exp
codegenOrd = ccast (scalarType :: ScalarType Int)

codegenChr :: C.Exp -> C.Exp
codegenChr = ccast (scalarType :: ScalarType Char)

codegenBoolToInt :: C.Exp -> C.Exp
codegenBoolToInt = ccast (scalarType :: ScalarType Int)

codegenFromIntegral :: IntegralType a -> NumType b -> C.Exp -> C.Exp
codegenFromIntegral _ ty = ccast (NumScalarType ty)

codegenTruncate :: FloatingType a -> IntegralType b -> C.Exp -> C.Exp
codegenTruncate ta tb x
  = ccast (NumScalarType (IntegralNumType tb))
  $ ccall (FloatingNumType ta `postfix` "trunc") [x]

codegenRound :: FloatingType a -> IntegralType b -> C.Exp -> C.Exp
codegenRound ta tb x
  = ccast (NumScalarType (IntegralNumType tb))
  $ ccall (FloatingNumType ta `postfix` "round") [x]

codegenFloor :: FloatingType a -> IntegralType b -> C.Exp -> C.Exp
codegenFloor ta tb x
  = ccast (NumScalarType (IntegralNumType tb))
  $ ccall (FloatingNumType ta `postfix` "floor") [x]

codegenCeiling :: FloatingType a -> IntegralType b -> C.Exp -> C.Exp
codegenCeiling ta tb x
  = ccast (NumScalarType (IntegralNumType tb))
  $ ccall (FloatingNumType ta `postfix` "ceil") [x]


-- Auxiliary Functions
-- -------------------

ccast :: ScalarType a -> C.Exp -> C.Exp
ccast ty x = [cexp|($ty:(codegenScalarType ty)) $exp:x|]

postfix :: NumType a -> String -> String
postfix (FloatingNumType (TypeFloat  _)) = (++ "f")
postfix (FloatingNumType (TypeCFloat _)) = (++ "f")
postfix _                                = id

