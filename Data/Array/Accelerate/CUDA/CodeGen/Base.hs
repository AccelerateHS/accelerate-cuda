{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Base
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Base (

  -- Types
  CUTranslSkel(..),

  -- Declaration generation
  typename, cvar, ccall, cchar, cintegral, cbool, cdim, cglobal, cshape,
  getters, setters

) where

import Data.Loc
import Data.Symbol
import Language.C.Syntax
import Language.C.Quote.CUDA
import Text.PrettyPrint.Mainland                ( Pretty(..), Doc, stack, text, (<>) )


-- Compilation unit
-- ----------------

newtype CUTranslSkel = CUTranslSkel [Definition]

instance Pretty CUTranslSkel where
  ppr (CUTranslSkel code) =
    stack ( include "accelerate_cuda_extras.h"
          : map ppr code
          )

include :: FilePath -> Doc
include hdr = text "#include <" <> text hdr <> text ">"


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

cintegral :: Integral a => a -> Exp
cintegral n = [cexp|$int:n|]

cbool :: Bool -> Exp
cbool = cintegral . fromEnum

cdim :: String -> Int -> Definition
cdim name n = [cedecl|typedef typename $id:("DIM" ++ show n) $id:name;|]


cglobal :: String -> Type -> Definition
cglobal name ty = [cedecl|static $ty:ty $id:name;|]

cshape :: String -> Int -> Definition
cshape name n = [cedecl| static __constant__ typename $id:("DIM" ++ show n) $id:name;|]


-- Generate a list of function parameters and variable initialisers that read
-- elements from the global input arrays at the given index.
--
getters :: Int -> [Type] -> ([Param], String -> [InitGroup])
getters base ts = (zipWith param ts arrs, \idx -> zipWith3 (get idx) ts arrs xs)
  where
    n                   = length ts
    suffixes            = map (\x -> shows base "_a" ++ show x) [n-1, n-2.. 0]
    arrs                = map ("d_in" ++) suffixes
    xs                  = map ('x':)      suffixes
    param t arr         = [cparam| const $ty:(ptr t) $id:arr |]
    get idx t arr x     = [cdecl| const $ty:t $id:x = $id:arr [$id:idx]; |]

-- Generate function parameters and corresponding variable names for the
-- components of the given output array.
--
setters :: [Type] -> ([Param], [Exp])
setters ts = (zipWith param ts names, map cvar names)
  where
    n           = length ts
    names       = map (\x -> "d_out_a" ++ show x) [n-1, n-2 .. 0]
    param t x   = [cparam| $ty:(ptr t) $id:x |]

-- Turn a plain type into a ptr type
--
ptr :: Type -> Type
ptr t | Type d@(DeclSpec _ _ _ _) r@(DeclRoot _) lb <- t = Type d (Ptr [] r noSrcLoc) lb
      | otherwise                                        = t


{--
cstruct :: Bool -> String -> [Type] -> Definition
cstruct volatile name types =
  [cedecl| typedef struct $id:name {
             $sdecls:(zipWith field names types)
           } $id:name;
  |]
  where
    n           = length types
    names       = ['a':show v | v <- [n-1,n-2.. 0]]
    field v ty  = [csdecl| $ty:(unstable ty) $id:v; |]
    --
    unstable ty
      | volatile, Type (DeclSpec s q t la) r lb <- ty   = Type (DeclSpec s (Tvolatile noSrcLoc:q) t la) r lb
      | otherwise                                       = ty

ctypedef :: Bool -> String -> Type -> Definition
ctypedef volatile name typ
  | volatile    = [cedecl| typedef volatile $ty:typ $id:name; |]
  | otherwise   = [cedecl| typedef          $ty:typ $id:name; |]
--}

{--
-- Common device functions
-- -----------------------

data Direction = Forward | Backward


mkIdentity :: [CExpr] -> CExtDecl
mkIdentity = mkDeviceFun "identity" (typename "TyOut") []

mkApply :: Int -> [CExpr] -> CExtDecl
mkApply argc
  = mkDeviceFun "apply" (typename "TyOut")
  $ map (\n -> (typename ("TyIn"++ show n), 'x':show n)) [argc-1,argc-2..0]

mkProject :: Direction -> [CExpr] -> CExtDecl
mkProject Forward  = mkDeviceFun "project" (typename "DimOut") [(typename "DimIn0","x0")]
mkProject Backward = mkDeviceFun "project" (typename "DimIn0") [(typename "DimOut","x0")]

mkSliceIndex :: [CExpr] -> CExtDecl
mkSliceIndex =
  mkDeviceFun "sliceIndex" (typename "SliceDim") [(typename "Slice","sl"), (typename "CoSlice","co")]

mkSliceReplicate :: [CExpr] -> CExtDecl
mkSliceReplicate =
  mkDeviceFun "sliceIndex" (typename "Slice") [(typename "SliceDim","dim")]
--}

{--
mkDim :: String -> Int -> CExtDecl
mkDim name n =
  mkTypedef name False False [CTypeDef (internalIdent ("DIM" ++ show n)) internalNode]

mkTypedef :: String -> Bool -> Bool -> CType -> CExtDecl
mkTypedef var volatile ptr ty =
  CDeclExt $ CDecl
    (CStorageSpec (CTypedef internalNode) : [CTypeQual (CVolatQual internalNode) | volatile] ++ map CTypeSpec ty)
    [(Just (CDeclr (Just (internalIdent var)) [CPtrDeclr [] internalNode | ptr] Nothing [] internalNode), Nothing, Nothing)]
    internalNode

mkShape :: Int -> String -> CExtDecl
mkShape d n = mkGlobal [constant,dimension] n
  where
    constant  = CTypeQual (CAttrQual (CAttr (internalIdent "constant") [] internalNode))
    dimension = CTypeSpec (CTypeDef (internalIdent ("DIM" ++ show d)) internalNode)

mkGlobal :: [CDeclSpec] -> String -> CExtDecl
mkGlobal spec name =
  CDeclExt (CDecl (CStorageSpec (CStatic internalNode) : spec)
           [(Just (CDeclr (Just (internalIdent name)) [] Nothing [] internalNode),Nothing,Nothing)] internalNode)

mkInitList :: [CExpr] -> CInit
mkInitList []  = CInitExpr (CConst (CIntConst (cInteger 0) internalNode)) internalNode
mkInitList [x] = CInitExpr x internalNode
mkInitList xs  = CInitList (map (\e -> ([],CInitExpr e internalNode)) xs) internalNode


-- typedef struct {
--   ... (volatile?) ty1 (*?) a1; (volatile?) ty0 (*?) a0;
-- } var;
--
-- NOTE: The Accelerate language uses snoc based tuple projection, so the last
--       field of the structure is named 'a' instead of the first.
--
mkStruct :: String -> Bool -> Bool -> [CType] -> CExtDecl
mkStruct name volatile ptr types =
  CDeclExt $ CDecl
    [CStorageSpec (CTypedef internalNode) , CTypeSpec (CSUType (CStruct CStructTag Nothing (Just (zipWith field names types)) [] internalNode) internalNode)]
    [(Just (CDeclr (Just (internalIdent name)) [] Nothing [] internalNode),Nothing,Nothing)]
    internalNode
  where
    names      = reverse . take (length types) $ (enumFrom 0 :: [Int])
    field v ty = CDecl ([CTypeQual (CVolatQual internalNode) | volatile] ++ map CTypeSpec ty)
                       [(Just (CDeclr (Just (internalIdent ('a':show v))) [CPtrDeclr [] internalNode | ptr] Nothing [] internalNode), Nothing, Nothing)]
                       internalNode


-- typedef struct __attribute__((aligned(n * sizeof(ty)))) {
--     ty [x, y, z, w];
-- } var;
--
mkTyVector :: String -> Int -> CType -> CExtDecl
mkTyVector var n ty =
  CDeclExt $ CDecl
    [CStorageSpec (CTypedef internalNode), CTypeSpec (CSUType (CStruct CStructTag Nothing (Just [CDecl (map CTypeSpec ty) fields internalNode]) [CAttr (internalIdent "aligned") [CBinary CMulOp (CConst (CIntConst (cInteger (toInteger n)) internalNode)) (CSizeofType (CDecl (map CTypeSpec ty) [] internalNode) internalNode) internalNode] internalNode] internalNode) internalNode)]
    [(Just (CDeclr (Just (internalIdent var)) [] Nothing [] internalNode), Nothing, Nothing)]
    internalNode
  where
    fields = take n . flip map "xyzw" $ \f ->
      (Just (CDeclr (Just (internalIdent [f])) [] Nothing [] internalNode), Nothing, Nothing)


-- static inline __attribute__((device)) tyout name(args)
-- {
--   tyout r = { expr };
--   return r;
-- }
--
mkDeviceFun :: String -> CType -> [(CType,String)] -> [CExpr] -> CExtDecl
mkDeviceFun name tyout args expr =
  let body = [ CBlockDecl (CDecl (map CTypeSpec tyout) [(Just (CDeclr (Just (internalIdent "r")) [] Nothing [] internalNode), Just (mkInitList expr), Nothing)] internalNode)
             , CBlockStmt (CReturn (Just (CVar (internalIdent "r") internalNode)) internalNode)]
  in
  mkDeviceFun' name tyout args body


-- static inline __attribute__((device)) tyout name(args)
-- {
--   body
-- }
--
mkDeviceFun' :: String -> CType -> [(CType, String)] -> [CBlockItem] -> CExtDecl
mkDeviceFun' name tyout args body =
  CFDefExt $ CFunDef
    ([CStorageSpec (CStatic internalNode), CTypeQual (CInlineQual internalNode), CTypeQual (CAttrQual (CAttr (builtinIdent "device") [] internalNode))] ++ map CTypeSpec tyout)
    (CDeclr (Just (internalIdent name)) [CFunDeclr (Right (argv,False)) [] internalNode] Nothing [] internalNode)
    []
    (CCompound [] body internalNode)
    internalNode
    where
      argv = flip map args $ \(ty,var) ->
        CDecl (CTypeQual (CConstQual internalNode) : map CTypeSpec ty)
              [(Just (CDeclr (Just (internalIdent var)) [] Nothing [] internalNode), Nothing, Nothing)]
              internalNode
--}
