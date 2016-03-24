{-# LANGUAGE GADTs               #-}
{-# LANGUAGE NoImplicitPrelude   #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE ViewPatterns        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Arithmetic
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Arithmetic
  where

-- friends
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Constant
import Data.Array.Accelerate.CUDA.CodeGen.Monad
import Data.Array.Accelerate.CUDA.CodeGen.Type

-- libraries
import Prelude                                          ( String, Char, ($), (++), (-), undefined, otherwise )
import Data.Bits                                        ( finiteBitSize )
import Control.Monad.State.Strict
import Language.C
import Language.C.Quote.CUDA
import Foreign.Storable                                 ( sizeOf )


-- Operations from Num
-- ===================

add :: Exp -> Exp -> Exp
add x y = [cexp| $exp:x + $exp:y |]

sub :: Exp -> Exp -> Exp
sub x y = [cexp| $exp:x - $exp:y |]

mul :: Exp -> Exp -> Exp
mul x y = [cexp| $exp:x * $exp:y |]

negate :: Exp -> Exp
negate x = [cexp| - $exp:x |]

abs :: forall a. NumType a -> Exp -> Exp
abs (FloatingNumType t) x
  = mathf t "fabs" [x]

abs (IntegralNumType t) x
  | signedIntegralNum t
  , IntegralDict <- integralDict t
  = case sizeOf (undefined::a) of
      8 -> ccall "llabs" [x]
      _ -> ccall "abs" [x]

  | otherwise
  = x

signum :: NumType a -> Exp -> Gen Exp
signum (IntegralNumType t) x
  | IntegralDict <- integralDict t
  , unsignedIntegralNum t
  = return [cexp| $exp:x > $exp:(integral t 0) |]

  | IntegralDict <- integralDict t
  = do x' <- bind (typeOf t) x
       return [cexp| ($exp:x' > $exp:(integral t 0)) - ($exp:x' < $exp:(integral t 0)) |]

signum (FloatingNumType t) x
  | FloatingDict <- floatingDict t
  = do x' <- bind (typeOf t) x
       return [cexp| $exp:x' == $exp:(floating t 0)
                       ? $exp:(floating t 0)
                       : $exp:(mathf t "copysign" [floating t 1, x']) |]


-- Operators from Integral & Bits
-- ==============================

quot :: Exp -> Exp -> Exp
quot x y = [cexp| $exp:x / $exp:y |]

rem :: Exp -> Exp -> Exp
rem x y = [cexp| $exp:x % $exp:y |]

quotRem :: IntegralType a -> Exp -> Exp -> Gen (Exp,Exp)
quotRem (typeOf -> t') x y = do
  x' <- bind t' x
  y' <- bind t' y
  q  <- bind t' (x' `quot` y')
  r  <- bind t' (x' `sub` (y' `mul` q))
  return (q, r)

idiv :: IntegralType a -> Exp -> Exp -> Gen Exp
idiv t x y
  | unsignedIntegralNum t
  = return (x `quot` y)

  | IntegralDict <- integralDict t
  , zero         <- integral t 0
  , one          <- integral t 1
  = do
      x' <- bind (typeOf t) x
      y' <- bind (typeOf t) y
      return $
        cases [ ((x' `gt` zero) `land` (y' `lt` zero), ((x' `sub` one) `quot` y') `sub` one)
              , ((x' `lt` zero) `land` (y' `gt` zero), ((x' `add` one) `quot` y') `sub` one)
              ]
              (x' `quot` y')

mod :: IntegralType a -> Exp -> Exp -> Gen Exp
mod t x y
  | unsignedIntegralNum t
  = return (x `rem` y)

  | IntegralDict <- integralDict t
  , zero         <- integral t 0
  = do
       x' <- bind (typeOf t) x
       y' <- bind (typeOf t) y
       r  <- bind (typeOf t) (x' `rem` y')
       return $
         ((((x' `gt` zero) `land` (y' `lt` zero)) `lor` ((x' `lt` zero) `land` (y' `gt` zero)))
          ?: ( r `neq` zero ?: ( r `add` y', zero )
             , r ))

divMod :: IntegralType a -> Exp -> Exp -> Gen (Exp, Exp)
divMod t x y | IntegralDict <- integralDict t = do
  x'    <- bind (typeOf t) x
  y'    <- bind (typeOf t) y
  (q,r) <- quotRem t x' y'

  sr    <- signum (IntegralNumType t) r
  sy'   <- signum (IntegralNumType t) y'

  -- Somewhat awful way to inject an ifThenElse statement
  vd    <- lift fresh
  vm    <- lift fresh
  modify (\st -> st { localBindings = [citem| $ty:(typeOf t) $id:vd, $id:vm; |] : localBindings st })
  modify (\st -> st { localBindings = [citem| if ( $exp:(sr `eq` negate sy') ) {
                                                  $id:vd = $exp:(q `sub` integral t 1);
                                                  $id:vm = $exp:(r `add` y') ;
                                              } else {
                                                  $id:vd = $exp:q;
                                                  $id:vm = $exp:r;
                                              } |] : localBindings st })
  return ( cvar vd, cvar vm )


band :: Exp -> Exp -> Exp
band x y = [cexp| $exp:x & $exp:y |]

bor :: Exp -> Exp -> Exp
bor x y = [cexp| $exp:x | $exp:y |]

xor :: Exp -> Exp -> Exp
xor x y = [cexp| $exp:x ^ $exp:y |]

bnot :: Exp -> Exp
bnot x = [cexp| ~ $exp:x |]

shiftL :: Exp -> Exp -> Exp
shiftL x i = [cexp| $exp:x << $exp:i |]

-- Arithmetic right shift (unchecked)
--
shiftRA :: Exp -> Exp -> Exp
shiftRA x i = [cexp| $exp:x >> $exp:i |]

-- Logical right shift (unchecked)
--
shiftRL :: IntegralType a -> Exp -> Exp -> Exp
shiftRL ty x i =
  let int  = typeOf (integralType :: IntegralType Int)
      word = typeOf (integralType :: IntegralType Word)
  in
  case ty of
    TypeInt{}    -> [cexp| ($ty:int)        (($ty:word)           $exp:x >> $exp:i) |]
    TypeInt8{}   -> [cexp| (typename Int8)  ((typename Word8)     $exp:x >> $exp:i) |]
    TypeInt16{}  -> [cexp| (typename Int16) ((typename Word16)    $exp:x >> $exp:i) |]
    TypeInt32{}  -> [cexp| (typename Int32) ((typename Word32)    $exp:x >> $exp:i) |]
    TypeInt64{}  -> [cexp| (typename Int64) ((typename Word64)    $exp:x >> $exp:i) |]
    TypeCShort{} -> [cexp| (short)          ((unsigned short)     $exp:x >> $exp:i) |]
    TypeCInt{}   -> [cexp| (int)            ((unsigned int)       $exp:x >> $exp:i) |]
    TypeCLong{}  -> [cexp| (long)           ((unsigned long)      $exp:x >> $exp:i) |]
    TypeCLLong{} -> [cexp| (long long)      ((unsigned long long) $exp:x >> $exp:i) |]

    -- unsigned types use arithmetic shift
    _            -> $internalCheck "shiftRL" "unhandled signed type" (unsignedIntegralNum ty) (shiftRA x i)


rotateL :: forall a. IntegralType a -> Exp -> Exp -> Gen Exp
rotateL t x i | IntegralDict <- integralDict t = do
  let int  = integralType :: IntegralType Int
      wsib = finiteBitSize (undefined::a)
  --
  x' <- bind (typeOf t)    x
  i' <- bind (typeOf int) (i `band` integral int (wsib - 1))
  return $ (x' `shiftL` i') `bor` (shiftRL t x' (integral int wsib `sub` i'))

rotateR :: IntegralType a -> Exp -> Exp -> Gen Exp
rotateR t x i = rotateL t x (negate i)


-- Operators from Fractional & Floating
-- ====================================

fdiv :: Exp -> Exp -> Exp
fdiv x y = [cexp| $exp:x / $exp:y |]

recip :: FloatingType a -> Exp -> Exp
recip t x | FloatingDict <- floatingDict t = fdiv (floating t 1) x

sin :: FloatingType a -> Exp -> Exp
sin t x = mathf t "sin" [x]

cos :: FloatingType a -> Exp -> Exp
cos t x = mathf t "cos" [x]

tan :: FloatingType a -> Exp -> Exp
tan t x = mathf t "tan" [x]

asin :: FloatingType a -> Exp -> Exp
asin t x = mathf t "asin" [x]

acos :: FloatingType a -> Exp -> Exp
acos t x = mathf t "acos" [x]

atan :: FloatingType a -> Exp -> Exp
atan t x = mathf t "atan" [x]

sinh :: FloatingType a -> Exp -> Exp
sinh t x = mathf t "sinh" [x]

cosh :: FloatingType a -> Exp -> Exp
cosh t x = mathf t "cosh" [x]

tanh :: FloatingType a -> Exp -> Exp
tanh t x = mathf t "tanh" [x]

asinh :: FloatingType a -> Exp -> Exp
asinh t x = mathf t "asinh" [x]

acosh :: FloatingType a -> Exp -> Exp
acosh t x = mathf t "acosh" [x]

atanh :: FloatingType a -> Exp -> Exp
atanh t x = mathf t "atanh" [x]

exp :: FloatingType a -> Exp -> Exp
exp t x = mathf t "exp" [x]

sqrt :: FloatingType a -> Exp -> Exp
sqrt t x = mathf t "sqrt" [x]

pow :: FloatingType a -> Exp -> Exp -> Exp
pow t x y = mathf t "pow" [x,y]

log :: FloatingType a -> Exp -> Exp
log t x = mathf t "log" [x]

logBase :: FloatingType a -> Exp -> Exp -> Exp
logBase t x y = log t y `fdiv` log t x


-- Operators from RealFrac
-- =======================

trunc :: FloatingType a -> IntegralType b -> Exp -> Exp
trunc ta tb x = cast tb $ mathf ta "trunc" [x]

round :: FloatingType a -> IntegralType b -> Exp -> Exp
round ta tb x = cast tb $ mathf ta "round" [x]

floor :: FloatingType a -> IntegralType b -> Exp -> Exp
floor ta tb x = cast tb $ mathf ta "floor" [x]

ceiling :: FloatingType a -> IntegralType b -> Exp -> Exp
ceiling ta tb x = cast tb $ mathf ta "ceil" [x]


-- Operators from RealFloat
-- ========================

atan2 :: FloatingType a -> Exp -> Exp -> Exp
atan2 t x y = mathf t "atan2" [x, y]

isNaN :: Exp -> Exp
isNaN x = ccall "isnan" [x]


-- Relational and equality operators
-- =================================

lt :: Exp -> Exp -> Exp
lt x y = [cexp| $exp:x < $exp:y |]

gt :: Exp -> Exp -> Exp
gt x y = [cexp| $exp:x > $exp:y |]

leq  :: Exp -> Exp -> Exp
leq x y = [cexp| $exp:x <= $exp:y |]

geq :: Exp -> Exp -> Exp
geq x y = [cexp| $exp:x >= $exp:y |]

eq :: Exp -> Exp -> Exp
eq x y = [cexp| $exp:x == $exp:y |]

neq :: Exp -> Exp -> Exp
neq x y = [cexp| $exp:x != $exp:y |]

max :: ScalarType a -> Exp -> Exp -> Exp
max (NonNumScalarType _) x y =
  let t = scalarType :: ScalarType Int32
  in  max t (cast t x) (cast t y)

max (NumScalarType (IntegralNumType _)) x y = ccall   "max"  [x,y]
max (NumScalarType (FloatingNumType t)) x y = mathf t "fmax" [x,y]

min :: ScalarType a -> Exp -> Exp -> Exp
min (NonNumScalarType _) x y =
  let t = scalarType :: ScalarType Int32
  in  min t (cast t x) (cast t y)

min (NumScalarType (IntegralNumType _)) x y = ccall   "min"  [x,y]
min (NumScalarType (FloatingNumType t)) x y = mathf t "fmin" [x,y]


-- Logical operators
-- =================

land :: Exp -> Exp -> Exp
land x y = [cexp| $exp:x && $exp:y |]

lor :: Exp -> Exp -> Exp
lor x y = [cexp| $exp:x || $exp:y |]

lnot :: Exp -> Exp
lnot x = [cexp| ! $exp:x |]


-- Type Conversions
-- ================

ord :: Exp -> Exp
ord = cast (scalarType :: ScalarType Int)

chr :: Exp -> Exp
chr = cast (scalarType :: ScalarType Char)

boolToInt :: Exp -> Exp
boolToInt = cast (scalarType :: ScalarType Int)

fromIntegral :: IntegralType a -> NumType b -> Exp -> Exp
fromIntegral _ tb = cast tb


-- Helpers
-- =======

cast :: TypeOf a => a -> Exp -> Exp
cast t x = [cexp| ($ty:(typeOf t)) $exp:x |]

mathf :: forall t. FloatingType t -> String -> [Exp] -> Exp
mathf ty f args | FloatingDict <- floatingDict ty =
  let
      fun = f ++ case sizeOf (undefined :: t) of
                   4  -> "f"
                   8  -> []
                   16 -> "l"        -- long double
                   _  -> $internalError "mathf" "unsupported floating point size"
  in
  ccall fun args


infix 0 ?:
(?:) :: Exp -> (Exp, Exp) -> Exp
(?:) p (t,e) = [cexp| $exp:p ? $exp:t : $exp:e |]

cases :: [(Exp, Exp)] -> Exp -> Exp
cases []           def = def
cases ((p,b):rest) def = p ?: (b, cases rest def)

