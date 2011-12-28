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
  typename, cptr, cvar, ccall, cchar, cintegral, cbool, cdim, cglobal, cshape,
  setters, setters', getters, getters', shared,

  -- Mutable operations
  (.=.), locals

) where

import Data.Loc
import Data.Symbol
import Language.C.Syntax
import Language.C.Quote.CUDA
import Text.PrettyPrint.Mainland                ( Pretty(..) )


-- Compilation unit
-- ----------------

-- A CUDA compilation unit, together with the name of the main __global__ entry
-- function.
--
data CUTranslSkel = CUTranslSkel String [Definition]

instance Show CUTranslSkel where
  show (CUTranslSkel entry _) = entry

instance Pretty CUTranslSkel where
  ppr  (CUTranslSkel _ code)  = ppr code


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


-- Generate a set of variable bindings and declarations to read from an input
-- array (as function parameter)
--
getters :: Int                          -- input array number (c.f. de Bruijn index)
        -> [Type]                       -- the element type
        -> ( [Param]                    -- function parameters
           , [Exp]                      -- variable names
           , [InitGroup]                -- non-const variable declarations
           , String -> [InitGroup])     -- const declaration and initialisation with data at index
getters base = let b = show base
               in  getters' ("d_in"++b) ('x':b)

getters' :: String -> String -> [Type] -> ([Param], [Exp], [InitGroup], String -> [InitGroup])
getters' arr_ x_ ts =
  ( zipWith param ts arrs
  , map cvar xs
  , zipWith decls ts xs
  , \idx -> zipWith3 (get idx) ts arrs xs
  )
  where
    n           = length ts
    suffixes    = map (\c -> "_a" ++ show c) [n-1, n-2.. 0]
    arrs        = map (arr_++) suffixes
    xs          = map (x_  ++) suffixes
    param t a   = [cparam| const $ty:(cptr t) $id:a |]
    decls t v   = [cdecl| $ty:t $id:v; |]
    get i t a v = [cdecl| const $ty:t $id:v = $id:a [$id:i]; |]


-- Generate function parameters and corresponding variable names for the
-- components of the given output array.
--
setters :: [Type]                       -- element type
        -> ( [Param]                    -- function parameter declarations
           , [Exp]                      -- variable name
           , String -> [Exp] -> [Stm])  -- store a value to the given index
setters = setters' "d_out_a"

setters' :: String -> [Type] -> ([Param], [Exp], String -> [Exp] -> [Stm])
setters' base ts =
  ( zipWith param ts arrs
  , map cvar arrs
  , \ix e -> zipWith (set ix) arrs e )
  where
    n           = length ts
    arrs        = map (\x -> base ++ show x) [n-1, n-2 .. 0]
    param t x   = [cparam| $ty:(cptr t) $id:x |]
    set ix a x  = [cstm| $id:a [$id:ix] = $exp:x; |]

-- shared memory declaration: this can only be issued once since all __shared__
-- declarations begin at the same base pointer and must be offset manually
--
shared :: Int                           -- shared memory shadowing which input array
       -> Exp                           -- how much shared memory per type
       -> [Type]                        -- element types
       -> ( [InitGroup]                 -- shared memory declaration
          , String -> [Exp] )           -- index shared memory
shared base ix elt =
  ( sdata' : zipWith3 sdata (tail elt) (tail vars) vars
  , \i -> map (\v -> [cexp| $id:v [ $id:i ] |]) vars )
  where
    n           = length elt
    vars        = let k = ('s':shows base "_a") in map ((k++) . show) [n-1,n-2..0]
    sdata'      = [cdecl| extern volatile __shared__ $ty:(head elt) $id:(head vars) []; |]
    sdata t v p = [cdecl| volatile $ty:(cptr t) $id:v = ( $ty:(cptr t) ) & $id:p [ $exp:ix ]; |]

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
