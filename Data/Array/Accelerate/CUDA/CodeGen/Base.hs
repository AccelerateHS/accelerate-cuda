{-# LANGUAGE GADTs       #-}
{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Base
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Base (

  -- Types
  CUTranslSkel(..), CUFun(..), CUExp(..),

  -- Declaration generation
  typename, cptr, cvar, ccall, cchar, cintegral, cbool, cdim, cglobal, cshape,
  setters, getters, shared, indexArray,

  -- Mutable operations
  (.=.), locals

) where

import Data.Loc
import Data.Char
import Data.List
import Language.C.Syntax
import Language.C.Quote.CUDA
import Text.PrettyPrint.Mainland                ( Pretty(..) )

import Data.Array.Accelerate.Array.Sugar        ( Elt )


-- Compilation units
-- -----------------

-- A CUDA compilation unit, together with the name of the main __global__ entry
-- function.
--
data CUTranslSkel = CUTranslSkel String [Definition]

instance Show CUTranslSkel where
  show (CUTranslSkel entry _) = entry

instance Pretty CUTranslSkel where
  ppr  (CUTranslSkel _ code)  = ppr code


-- Scalar functions and expressions, including the environment of local
-- let-bindings and array data elements.
--
data CUFun a where
  CUBody ::                                CUExp a -> CUFun a
  CULam  :: Elt a => [(Int, Type, Exp)] -> CUFun t -> CUFun (a -> t)

data CUExp e where
  CUExp  :: [InitGroup] -> [Exp] -> CUExp e


-- Expression and Declaration generation
-- -------------------------------------

cvar :: String -> Exp
cvar x = [cexp|$id:x|]

ccall :: String -> [Exp] -> Exp
ccall fn args = [cexp|$id:fn ($args:args)|]

typename :: String -> Type
typename name = Type (DeclSpec [] [] (Tnamed (Id name noSrcLoc) noSrcLoc) noSrcLoc) (DeclRoot noSrcLoc) noSrcLoc

cchar :: Char -> Exp
cchar c = [cexp|$char:c|]

cintegral :: (Integral a, Show a) => a -> Exp
cintegral n = [cexp|$int:n|]

cbool :: Bool -> Exp
cbool = cintegral . fromEnum

cdim :: String -> Int -> Definition
cdim name n = [cedecl|typedef typename $id:("DIM" ++ show n) $id:name;|]


cglobal :: Type -> String -> Definition
cglobal ty name = [cedecl|static $ty:ty $id:name;|]

cshape :: String -> Int -> Definition
cshape name n = [cedecl| static __constant__ typename $id:("DIM" ++ show n) $id:name;|]

indexArray :: Type -> Exp -> Exp -> Exp
indexArray ty arr ix
  | "double" `isSuffixOf` map toLower (show ty) = ccall "indexDArray" [arr, ix]
  | otherwise                                   = ccall "indexArray"  [arr, ix]


-- Generate a list of variable bindings and declarations to read from the input
-- arrays.
--
-- In the case where the input array is an array of tuples, the function
-- parameters naturally include all components, but the scalar declarations
-- include only those indices that are used.
--
getters
    :: Int                              -- base de Bruijn index
    -> [Type]                           -- the array element type
    -> [(Int, Type, Exp)]               -- the variables used in the scalar expression
    -> ( [Param]                        -- function parameters for array(s) input
       , [Exp]                          -- variable names
       , [InitGroup]                    -- non-const variable declarations
       , String -> [Exp]                -- index global array
       , String -> [InitGroup] )        -- const declarations and initialisation from index
getters base arrElt expElt =
  let n                 = length arrElt
      arr x             = "d_in" ++ shows base "_a" ++ show x
      arrParams         = zipWith (\t x -> [cparam| const $ty:(cptr t) $id:(arr x) |]) arrElt [n-1, n-2 .. 0]
      expVars           = map (\(_,_,v) -> v) expElt
      expDecls          = map (\(_,t,v) -> [cdecl| $ty:t $id:(show v) ; |]) expElt
  in
  ( arrParams
  , expVars
  , expDecls
  , \ix -> map (\(i,_,_) -> [cexp| $id:(arr i) [ $id:ix ] |] ) expElt
  , \ix -> map (\(i,t,v) -> [cdecl| const $ty:t $id:(show v) = $id:(arr i) [$id:ix] ; |]) expElt
  )


-- Generate function parameters and corresponding variable names for the
-- components of the given output array.
--
setters
    :: [Type]                           -- element type
    -> ( [Param]                        -- function parameter declarations
       , [Exp]                          -- variable name
       , String -> [Exp] -> [Stm])      -- store a value to the given index
setters arrElt =
  let n                 = length arrElt
      arrVars           = map (\x -> "d_out_a" ++ show x) [n-1, n-2 .. 0]
      arrParams         = zipWith (\t x -> [cparam| $ty:(cptr t) $id:x |]) arrElt arrVars
      set ix a x        = [cstm| $id:a [$id:ix] = $exp:x; |]
  in
  ( arrParams
  , map cvar arrVars
  , \ix e -> zipWith (set ix) arrVars e
  )


-- shared memory declaration. All dynamically allocated __shared__ memory will
-- begin at the same base address. If we call this more than once, or the kernel
-- itself declares some shared memory, the first parameter is a pointer to where
-- the new declarations should start from.
--
shared
    :: Int                              -- shared memory shadowing which input array
    -> Maybe Exp                        -- (optional) initialise from this base address
    -> Exp                              -- how much shared memory per type
    -> [Type]                           -- element types
    -> ( [InitGroup]                    -- shared memory declaration
    , String -> [Exp] )                 -- index shared memory
shared base = shared' ('s':shows base "_a")

shared' :: String -> Maybe Exp -> Exp -> [Type] -> ([InitGroup], String -> [Exp])
shared' base mprev ix elt =
  ( sdecl (head elt) (head vars) : zipWith3 sdata (tail elt) (tail vars) vars
  , \i -> map (\v -> [cexp| $id:v [ $id:i ] |]) vars )
  where
    vars                = let k = length elt in map (\n -> base ++ show n) [k-1,k-2..0]
    sdecl t v
      | Just p <- mprev = [cdecl| volatile $ty:(cptr t) $id:v = ( $ty:(cptr t) ) $exp:p; |]
      | otherwise       = [cdecl| extern volatile __shared__ $ty:t $id:v []; |]
    sdata t v p         = [cdecl| volatile $ty:(cptr t) $id:v = ( $ty:(cptr t) ) & $id:p [ $exp:ix ]; |]


-- Turn a plain type into a ptr type
--
cptr :: Type -> Type
cptr t | Type d@(DeclSpec _ _ _ _) r@(DeclRoot _) lb <- t = Type d (Ptr [] r noSrcLoc) lb
       | otherwise                                        = t


-- Mutable operations
-- ------------------

-- Variable assignment
--
(.=.) :: [Exp] -> [Exp] -> [Stm]
(.=.) = zipWith (\v e -> [cstm| $exp:v = $exp:e; |])

locals :: String -> [Type] -> ([Exp], [InitGroup])
locals base elt = unzip (zipWith local elt names)
  where
    suf         = let n = length elt in map show [n-1,n-2..0]
    names       = map (\n -> base ++ "_a" ++ n) suf
    local t n   = ( cvar n, [cdecl| $ty:t $id:n; |] )

