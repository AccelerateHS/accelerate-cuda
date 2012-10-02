{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
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

module Data.Array.Accelerate.CUDA.CodeGen (

  CUTranslSkel, codegenAcc,
  codegenScanlIntervals, codegenScanrIntervals,

) where

-- libraries
import Prelude                                                  hiding ( exp )
import Data.Loc
import Data.Char
import Data.Label.PureM
import Control.Monad
import Control.Applicative                                      hiding ( Const )
import Language.C.Syntax                                        ( Const(..) )
import Language.C.Quote.CUDA
import Text.PrettyPrint.Mainland
import qualified Data.HashSet                                   as Set
import qualified Language.C                                     as C
import qualified Foreign.CUDA.Analysis                          as CUDA

-- friends
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Tuple
import Data.Array.Accelerate.Pretty                             ()
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.Array.Representation               hiding ( sliceIndex )
import qualified Data.Array.Accelerate.Array.Sugar              as Sugar
import qualified Data.Array.Accelerate.Analysis.Type            as Sugar

import Data.Array.Accelerate.CUDA.AST                           hiding ( Val(..), prj )
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type
import Data.Array.Accelerate.CUDA.CodeGen.Monad
import Data.Array.Accelerate.CUDA.CodeGen.Mapping
import Data.Array.Accelerate.CUDA.CodeGen.IndexSpace
import Data.Array.Accelerate.CUDA.CodeGen.PrefixSum
import Data.Array.Accelerate.CUDA.CodeGen.Reduction
import Data.Array.Accelerate.CUDA.CodeGen.Stencil

#include "accelerate.h"


data Val env where
  Empty ::                       Val ()
  Push  :: Val env -> [C.Exp] -> Val (env, s)

prj :: Idx env t -> Val env -> [C.Exp]
prj ZeroIdx      (Push _   v) = v
prj (SuccIdx ix) (Push val _) = prj ix val
prj _            _            = INTERNAL_ERROR(error) "prj" "inconsistent valuation"

sizeEnv :: Val env -> Int
sizeEnv Empty        = 0
sizeEnv (Push env _) = 1 + sizeEnv env


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
codegenAcc :: CUDA.DeviceProperties -> OpenAcc aenv a -> AccBindings aenv -> CUTranslSkel
codegenAcc dev acc var
  = codegenBindEnv var
  $ codegenOpenAcc dev acc


-- Add the include statement for the standard Accelerate headers, and any
-- binding points required for environment arrays accessed by scalar functions.
--
codegenBindEnv :: AccBindings aenv -> CUTranslSkel -> CUTranslSkel
codegenBindEnv (AccBindings vars) (CUTranslSkel entry code) =
  CUTranslSkel entry (headers : fvars code)
  where
    headers     = [cedecl| $esc:("#include <accelerate_cuda_extras.h>") |]
    fvars rest  = Set.foldr (\v vs -> liftAcc v ++ vs) rest vars

    -- Generate binding points (texture references and shapes) for arrays lifted
    -- from scalar expressions
    --
    liftAcc :: ArrayVar aenv -> [C.Definition]
    liftAcc (ArrayVar idx) =
      let avar    = OpenAcc (Avar idx)
          idx'    = show $ idxToInt idx
          sh      = cshape ("sh" ++ idx') (accDim avar)
          ty      = accTypeTex avar
          arr n   = "avar" ++ idx' ++ "_a" ++ show (n::Int)
      in
      sh : zipWith (\t n -> cglobal t (arr n)) (reverse ty) [0..]


-- Generate code for a single __device__ entry function.
--
codegenOpenAcc :: forall aenv a. CUDA.DeviceProperties -> OpenAcc aenv a -> CUTranslSkel
codegenOpenAcc dev acc@(OpenAcc pacc) = case pacc of
  --
  -- Non-computation forms -> sadness.
  --
  Alet _ _      -> internalError
  Avar _        -> internalError
  Apply _ _     -> internalError
  Acond _ _ _   -> internalError
  Atuple _      -> internalError
  Aprj _ _      -> internalError
  Use _         -> internalError
  Unit _        -> internalError
  Reshape _ _   -> internalError

  --
  -- Skeleton nodes
  --
  Generate _ f ->
    mkGenerate (accDim acc) (codegenFun f)

  Transform _ p f a ->
    mkTransform (accDim acc) (accDim a) (codegenFun p) (codegenFun f)

  Replicate sl _ a ->
    mkReplicate dimSl dimOut (extend sl) (undefined :: a)
    where
      dimSl  = accDim a
      dimOut = accDim acc
      --
      extend :: SliceIndex slix sl co dim -> CUExp dim
      extend = CUExp [] . reverse . extend' 0

      extend' :: Int -> SliceIndex slix sl co dim -> [C.Exp]
      extend' _ (SliceNil)            = []
      extend' n (SliceAll   sliceIdx) = mkPrj dimOut "dim" n : extend' (n+1) sliceIdx
      extend' n (SliceFixed sliceIdx) =                        extend' (n+1) sliceIdx

  Slice sl a slix ->
    mkSlice dimSl dimCo dimIn0 (restrict sl) (undefined :: a)
    where
      dimCo  = length (expType slix)
      dimSl  = accDim acc
      dimIn0 = accDim a
      --
      restrict :: SliceIndex slix sl co dim -> CUExp slix
      restrict = CUExp [] . reverse . restrict' (0,0)

      restrict' :: (Int,Int) -> SliceIndex slix sl co dim -> [C.Exp]
      restrict' _     (SliceNil)            = []
      restrict' (m,n) (SliceAll   sliceIdx) = mkPrj dimSl "sl" n : restrict' (m,n+1) sliceIdx
      restrict' (m,n) (SliceFixed sliceIdx) = mkPrj dimCo "co" m : restrict' (m+1,n) sliceIdx

  Map f _ ->
    mkMap (codegenFun f)

  ZipWith f _ _ ->
    mkZipWith (accDim acc) (codegenFun f)

  Fold f e _ ->
    if accDim acc == 0
       then mkFoldAll dev (codegenFun f) (Just (codegenExp e))
       else mkFold    dev (codegenFun f) (Just (codegenExp e))

  Fold1 f _ ->
    if accDim acc == 0
       then mkFoldAll dev (codegenFun f) Nothing
       else mkFold    dev (codegenFun f) Nothing

  FoldSeg f e _ s ->
    mkFoldSeg dev (accDim acc) (segmentsType s) (codegenFun f) (Just (codegenExp e))

  Fold1Seg f _ s ->
    mkFoldSeg dev (accDim acc) (segmentsType s) (codegenFun f) Nothing

  Scanl f e _ ->
    mkScanl dev (codegenFun f) (Just (codegenExp e))

  Scanl' f e _ ->
    mkScanl dev (codegenFun f) (Just (codegenExp e))

  Scanl1 f _ ->
    mkScanl dev (codegenFun f) Nothing

  Scanr f e _ ->
    mkScanr dev (codegenFun f) (Just (codegenExp e))

  Scanr' f e _ ->
    mkScanr dev (codegenFun f) (Just (codegenExp e))

  Scanr1 f _ ->
    mkScanr dev (codegenFun f) Nothing

  Permute f _ ix a ->
    mkPermute dev (accDim acc) (accDim a) (codegenFun f) (codegenFun ix)

  Backpermute _ f a ->
    mkBackpermute (accDim acc) (accDim a) (codegenFun f) (undefined :: a)

  Stencil  f b0 a0 ->
    mkStencil  (accDim acc) (codegenFun f) (codegenBoundary a0 b0) (undefined :: a)

  Stencil2 f b1 a1 b0 a0 ->
    mkStencil2 (accDim acc) (codegenFun f) (codegenBoundary a1 b1)
                                           (codegenBoundary a0 b0) (undefined :: a)
  where
    -- caffeine and misery
    --
    internalError =
      let msg = unlines ["unsupported array primitive", pretty 100 (nest 2 doc)]
          pac = show acc
          doc | length pac <= 250 = text pac
              | otherwise         = text (take 250 pac) <+> text "... {truncated}"
      in
      INTERNAL_ERROR(error) "codegenAcc" msg

    -- Shapes are still represented as C structs, so we need to generate field
    -- indexing code for shapes
    --
    mkPrj :: Int -> String -> Int -> C.Exp
    mkPrj ndim var c
      | ndim <= 1   = cvar var
      | otherwise   = [cexp| $exp:(cvar var) . $id:('a':show c) |]

    -- code generation for stencil boundary conditions
    --
    codegenBoundary :: forall dim e. Sugar.Elt e
                    => OpenAcc aenv (Sugar.Array dim e) {- dummy -}
                    -> Boundary (Sugar.EltRepr e)
                    -> Boundary (CUExp e)
    codegenBoundary _ Clamp        = Clamp
    codegenBoundary _ Mirror       = Mirror
    codegenBoundary _ Wrap         = Wrap
    codegenBoundary _ (Constant c)
      = Constant . CUExp []
      $ codegenConst (Sugar.eltType (undefined::e)) c


-- Exceptional cases
-- -----------------

-- Scan is a multi-pass algorithm that requires a few additional operations not
-- in the standard set. These helper functions generate the necessary code.
--
codegenScanlIntervals, codegenScanrIntervals
    :: CUDA.DeviceProperties
    -> Fun aenv (a -> a -> a)
    -> AccBindings aenv
    -> CUTranslSkel
codegenScanlIntervals dev f avar
  = codegenBindEnv avar
  $ mkScanlIntervals dev (codegenFun f)

codegenScanrIntervals dev f avar
  = codegenBindEnv avar
  $ mkScanrIntervals dev (codegenFun f)


-- Scalar Expressions
-- ------------------

-- Function abstraction
--
-- Although Accelerate includes lambda abstractions, it does not include a
-- general application form. That is, lambda abstractions of scalar expressions
-- are only introduced as arguments to collective operations, so lambdas are
-- always outermost, and can always be translated into plain C functions.
--
codegenFun :: Fun aenv t -> CUFun t
codegenFun fun = runCGM $ codegenOpenFun (arity fun) fun Empty
  where
    arity :: OpenFun env aenv t -> Int
    arity (Body _) = -1
    arity (Lam f)  =  1 + arity f

codegenOpenFun :: Int -> OpenFun env aenv t -> Val env -> CGM (CUFun t)
codegenOpenFun _lvl (Body e) env = do
  e'    <- codegenOpenExp e env
  env'  <- bodycode
  zipWithM_ addVar (expType e) e'
  return $ CUBody (CUExp env' e')

codegenOpenFun lvl (Lam (f :: OpenFun (env,a) aenv b)) env = do
  let ty    = eltType (undefined::a)
      n     = length ty
      vars  = map (\i -> cvar ('x':shows lvl "_a" ++ show i)) [n-1,n-2..0]
  pushEnv
  f'    <- codegenOpenFun (lvl-1) f (env `Push` vars)
  vars' <- subscripts lvl
  return $ CULam vars' f'


-- Embedded scalar computations
--
codegenExp :: Exp aenv t -> CUExp t
codegenExp exp = runCGM $ do
  e'    <- codegenOpenExp exp Empty
  env'  <- bodycode
  return $ CUExp env' e'


codegenOpenExp :: forall env aenv t. OpenExp env aenv t -> Val env -> CGM [C.Exp]
codegenOpenExp exp env =
  case exp of
    -- local binders and variable indices
    --
    -- NOTE: recording which variables are used is important, because the CUDA
    -- compiler will not eliminate variables that are initialised but never
    -- used. If this is a scalar type mark it as used immediately, otherwise
    -- wait until tuple projection picks out an individual element.
    --
    Let a b -> do
      a'        <- codegenOpenExp a env
      vars      <- case a of
                     -- Const _    -> return a'
                     Var _      -> return a'
                     _          -> zipWithM bindVars (expType a) a'
      codegenOpenExp b (env `Push` vars)
      where
        -- FIXME: if we are let-binding an input argument (read from global
        --   array) mark that as used and return the variable name directly,
        --   otherwise create a fresh binding point.
        --
        bindVars t x = do
          p     <- addVar t x
          if p then return x
               else bind t x

    Var ix
      | [t] <- ty, [v] <- var   -> addVar t v >> return var
      | otherwise               -> return var
      where
        var     = prj ix env
        ty      = eltType (undefined :: t)

    -- Constant values
    --
    PrimConst c         -> return [codegenPrimConst c]
    Const c             -> return (codegenConst (Sugar.eltType (undefined::t)) c)

    -- Primitive scalar operations
    --
    PrimApp f arg       -> do
      x                 <- codegenOpenExp arg env
      return [codegenPrim f x]

    -- Tuples
    --
    Tuple t             -> codegenTup t env
    Prj idx e           -> do
      e'                <- codegenOpenExp e env
      case subset (zip e' elt) of
        [(x,t)]         -> addVar t x >> return [x]
        xts             -> return $ fst (unzip xts)
      where
        elt     = expType e
        subset  = reverse
                . take (length (expType exp))
                . drop (prjToInt idx (Sugar.expType e))
                . reverse

    -- Conditional expression
    --
    Cond p t e          -> do
      t'                <- codegenOpenExp t env
      e'                <- codegenOpenExp e env
      p'                <- codegenOpenExp p env >>= \ps ->
        case ps of
          [x]   -> bind [cty| typename bool |] x
          _     -> INTERNAL_ERROR(error) "codegenOpenExp" "expected conditional predicate"
      --
      let cond ty a b   = addVar ty a >> addVar ty b >>
                          return [cexp| $exp:p' ? $exp:a : $exp:b|]
      sequence $ zipWith3 cond (expType t) t' e'

    -- Value recursion
    --
    -- TLM: Foolishly, TLM introduced lambda abstractions into the scalar
    --      language, where previously these were always outermost. He now pays
    --      for that crime.
    --
    Iterate n (Lam f) x -> do
      -- Generate the initial values used to seed the loop. Bind these initial
      -- values to fresh variables that the loop will accumulate into.
      --
      x'        <- codegenOpenExp x env
      xn        <- replicateM (length x') fresh
      let seed   = zipWith3 (\t a v -> [cdecl| $ty:t $id:a = $exp:v; |]) (expType x) xn x'
          acc    = map cvar xn

      -- Generate the loop in an almost clean environment. Specifically, we
      -- don't want any previous code pieces to be included inside the body.
      --
      outer     <- gets bindings
      bindings  =: []

      -- Generate the loop body. The environment consists of the surrounding
      -- environment plus the new fresh accumulator variables.
      --
      pushEnv
      CUBody (CUExp f' v') <- codegenOpenFun (sizeEnv env - 1) f (env `Push` acc)
      i                    <- fresh

      let loop = [cstm| for (int $id:i = 0; $id:i < $int:n; ++ $id:i) {
                            $items:f'
                            $stms:(acc .=. v')
                        } |]

      -- Restore the binding state
      --
      bindings  =: outer
      modify bindings ( map C.BlockDecl (reverse seed) ++ )
      modify bindings ( C.BlockStm loop : )
      return acc

    Iterate _ _ _
      -> INTERNAL_ERROR(error) "codegenOpenExp" "expected unary function"

    -- Array indices and shapes
    --
    IndexNil            -> return []
    IndexAny            -> return []
    IndexCons sh sz     -> do
      sh'               <- codegenOpenExp sh env
      sz'               <- codegenOpenExp sz env
      return (sh' ++ sz')

    IndexHead ix        -> do
      ix'               <- last <$> codegenOpenExp ix env
      _                 <- addVar (last (expType ix)) ix'
      return [ix']

    IndexTail ix        -> do
      ix'               <- codegenOpenExp ix env
      return (init ix')

    IndexSlice sliceIndex slix sh -> do
      slix'             <- codegenOpenExp slix env
      sh'               <- codegenOpenExp sh env

      return . reverse $ restrict sliceIndex (reverse slix') (reverse sh')
      where
        restrict :: SliceIndex slix sl co sh -> [C.Exp] -> [C.Exp] -> [C.Exp]
        restrict SliceNil               _   _       = []
        restrict (SliceAll sliceIdx)    slx (sz:sl)     -- elide Any from the head of slx
          = let sl' = restrict sliceIdx slx sl
            in sz : sl'
        restrict (SliceFixed sliceIdx) (_:slx) (_:sl)
          = restrict sliceIdx slx sl
        restrict _ _ _
          = INTERNAL_ERROR(error) "IndexSlice" "unexpected shapes"

    IndexFull sliceIndex slix sh -> do
      slix'             <- codegenOpenExp slix env
      sh'               <- codegenOpenExp sh env
      return . reverse $ extend sliceIndex (reverse slix') (reverse sh')
      where
        extend :: SliceIndex slix sl co sh -> [C.Exp] -> [C.Exp] -> [C.Exp]
        extend SliceNil               _   _       = []
        extend (SliceAll sliceIdx)    slx (sz:sl)       -- elide Any from head of slx
          = let sh' = extend sliceIdx slx sl
            in sz : sh'
        extend (SliceFixed sliceIdx) (sz:slx) sl
          = let sh' = extend sliceIdx slx sl
            in sz : sh'
        extend _ _ _
          = INTERNAL_ERROR(error) "IndexFull" "unexpected shapes"

    ToIndex sh ix       -> do
      sh'               <- codegenOpenExp sh env
      ix'               <- codegenOpenExp ix env
      return [ ccall "toIndex" [ccall "shape" sh', ccall "shape" ix']]

    FromIndex sh ix     -> do
      sh'               <- codegenOpenExp sh env
      ix'               <- codegenOpenExp ix env
      return [ ccall "fromIndex" (ccall "shape" sh' : ix') ]

    -- Array shape and element indexing
    --
    ShapeSize sh        -> do
      sh'               <- codegenOpenExp sh env
      return [ ccall "size" [ccall "shape" sh'] ]

    Shape arr
      | OpenAcc (Avar a) <- arr ->
          let ndim      = accDim arr
              sh        = cvar ("sh" ++ show (idxToInt a))
          in return $ if ndim <= 1
                then [sh]
                else map (\c -> [cexp| $exp:sh . $id:('a':show c) |] ) [ndim-1, ndim-2 .. 0]

      | otherwise               -> INTERNAL_ERROR(error) "codegenOpenExp" "expected array variable"

    Index arr ix
      | OpenAcc (Avar a) <- arr ->
        let avar        = show (idxToInt a)
            sh          = cvar ("sh"   ++ avar)
            array x     = cvar ("avar" ++ avar ++ "_a" ++ show x)
            elt         = accTypeTex arr
            n           = length elt
        in do
          ix'           <- codegenOpenExp ix env
          v             <- bind [cty| int |] (ccall "toIndex" [sh, ccall "shape" ix'])
          return $ zipWith (\t x -> indexArray t (array x) v) elt [n-1, n-2 .. 0]

      | otherwise                -> INTERNAL_ERROR(error) "codegenOpenExp" "expected array variable"

    LinearIndex arr ix
      | OpenAcc (Avar a) <- arr ->
        let v           = show (idxToInt a)
            array x     = cvar ("avar" ++ v ++ "_a" ++ show x)
            elt         = accTypeTex arr
            n           = length elt
        in do
          [i]   <- codegenOpenExp ix env
          return $ zipWith (\t x -> indexArray t (array x) i) elt [n-1, n-2 .. 0]

      | otherwise
      -> INTERNAL_ERROR(error) "codegenOpenExp" "expected array variable"

    Intersect sh1 sh2   -> do
      sh1'              <- codegenOpenExp sh1 env
      sh2'              <- codegenOpenExp sh2 env
      return $ zipWith (\a b -> ccall "min" [a,b]) sh1' sh2'


-- Tuples are defined as snoc-lists, so generate code right-to-left
--
codegenTup :: Tuple (OpenExp env aenv) t -> Val env -> CGM [C.Exp]
codegenTup tup env = case tup of
  NilTup        -> return []
  SnocTup t e   -> (++) <$> codegenTup t env <*> codegenOpenExp e env


-- Convert a tuple index into the corresponding integer. Since the internal
-- representation is flat, be sure to walk over all sub components when indexing
-- past nested tuples.
--
prjToInt :: TupleIdx t e -> TupleType a -> Int
prjToInt ZeroTupIdx     _                 = 0
prjToInt (SuccTupIdx i) (b `PairTuple` a) = sizeTupleType a + prjToInt i b
prjToInt _ _ =
  INTERNAL_ERROR(error) "prjToInt" "inconsistent valuation"

sizeTupleType :: TupleType a -> Int
sizeTupleType UnitTuple         = 0
sizeTupleType (SingleTuple _)   = 1
sizeTupleType (PairTuple a b)   = sizeTupleType a + sizeTupleType b


-- Recording which variables of a computation are actually used is important,
-- particularly for stencils and arrays of tuples, because the CUDA compiler
-- will not eliminate variables that are initialised but never used.
--
-- FIXME: This dubious hack is used to inspect the expression and mark as used
--   if it refers to an array input.
--
addVar :: C.Type -> C.Exp -> CGM Bool
addVar ty exp = case show exp of
  ('x':v:'_':'a':n) | [(v',[])] <- reads [v], [(n',[])] <- reads n
        -> use v' n' ty exp >> return True
  ('v':n) | [(_ :: Int,[])] <- reads n
        ->                     return True
  _     ->                     return False


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
codegenIntegralScalar ty x | IntegralDict <- integralDict ty = [cexp| ( $ty:(codegenIntegralType ty) ) $exp:(cintegral x) |]

codegenFloatingScalar :: FloatingType a -> a -> C.Exp
codegenFloatingScalar (TypeFloat   _) x = C.Const (FloatConst (shows x "f") (toRational x) noLoc) noLoc
codegenFloatingScalar (TypeCFloat  _) x = C.Const (FloatConst (shows x "f") (toRational x) noLoc) noLoc
codegenFloatingScalar (TypeDouble  _) x = C.Const (DoubleConst (show x) (toRational x) noLoc) noLoc
codegenFloatingScalar (TypeCDouble _) x = C.Const (DoubleConst (show x) (toRational x) noLoc) noLoc

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
codegenAbs (FloatingNumType ty) x = ccall (FloatingNumType ty `postfix` "fabs") [x]
codegenAbs (IntegralNumType ty) x =
  case ty of
    TypeWord _          -> x
    TypeWord8 _         -> x
    TypeWord16 _        -> x
    TypeWord32 _        -> x
    TypeWord64 _        -> x
    TypeCUShort _       -> x
    TypeCUInt _         -> x
    TypeCULong _        -> x
    TypeCULLong _       -> x
    _                   -> ccall "abs" [x]


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

