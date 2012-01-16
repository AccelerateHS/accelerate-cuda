{-# LANGUAGE BangPatterns, QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Mapping
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Stencil (

  mkStencil, mkStencil2

) where

import Data.Loc
import Data.List
import Data.Symbol
import Language.C.Syntax
import Language.C.Quote.CUDA

import Data.Array.IArray                                ( Array, listArray, (!) )
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Monad


-- Map a stencil over an array.  In contrast to 'map', the domain of a stencil
-- function is an entire /neighbourhood/ of each array element.  Neighbourhoods
-- are sub-arrays centred around a focal point.  They are not necessarily
-- rectangular, but they are symmetric in each dimension and have an extent of
-- at least three in each dimensions — due to the symmetry requirement, the
-- extent is necessarily odd.  The focal point is the array position that is
-- determined by the stencil.
--
-- For those array positions where the neighbourhood extends past the boundaries
-- of the source array, a boundary condition determines the contents of the
-- out-of-bounds neighbourhood positions.
--
-- stencil :: (Shape ix, Elt a, Elt b, Stencil ix a stencil)
--         => (stencil -> Exp b)                 -- stencil function
--         -> Boundary a                         -- boundary condition
--         -> Acc (Array ix a)                   -- source array
--         -> Acc (Array ix b)                   -- destination array
--
-- To improve performance, the input array(s) are read through the texture
-- cache.
--
mkStencil :: Int -> [Type] -> [Type] -> Boundary [Exp] -> [[Int]] -> [Exp] -> CGM CUTranslSkel
mkStencil dim tyOut stencilIn0 boundary offsets combine = do
  env                   <- environment
  (arrIn0, getIn0)      <- stencilAccess 0 dim stencilIn0 boundary offsets
  return $ CUTranslSkel "stencil" [cunit|
    $edecl:(cdim "Shape" dim)
    $edecls:arrIn0

    extern "C"
    __global__ void
    stencil
    (
        $params:argOut,
        const typename Shape shIn0
    )
    {
        const int shapeSize = size(shIn0);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);
              int i;

        for ( i =  __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; i <  shapeSize
            ; i += gridSize )
        {
            const typename Shape ix = fromIndex(shIn0, i);
            $decls:(getIn0 "ix")
            $decls:env
            $stms:(setOut "i" combine)
        }
    }
  |]
  where
    (argOut, _, setOut) = setters tyOut



-- Map a binary stencil of an array.  The extent of the resulting array is the
-- intersection of the extents of the two source arrays.
--
-- stencil2 :: (Shape ix, Elt a, Elt b, Elt c,
--              Stencil ix a stencil1,
--              Stencil ix b stencil2)
--          => (stencil1 -> stencil2 -> Exp c)  -- binary stencil function
--          -> Boundary a                       -- boundary condition #1
--          -> Acc (Array ix a)                 -- source array #1
--          -> Boundary b                       -- boundary condition #2
--          -> Acc (Array ix b)                 -- source array #2
--          -> Acc (Array ix c)                 -- destination array
--
mkStencil2 :: Int -> [Type] -> [Type] -> Boundary [Exp] -> [[Int]]
                            -> [Type] -> Boundary [Exp] -> [[Int]] -> [Exp] -> CGM CUTranslSkel
mkStencil2 dim tyOut stencilIn1 boundary1 offsets1 stencilIn0 boundary0 offsets0 combine = do
  env   <- environment
  (arrIn0, getIn0)      <- stencilAccess 0 dim stencilIn0 boundary0 offsets0
  (arrIn1, getIn1)      <- stencilAccess 1 dim stencilIn1 boundary1 offsets1
  return $ CUTranslSkel "stencil2" [cunit|
    $edecl:(cdim "Shape" dim)
    $edecls:arrIn0
    $edecls:arrIn1

    extern "C"
    __global__ void
    stencil2
    (
        $params:argOut,
        const typename Shape shOut,
        const typename Shape shIn1,
        const typename Shape shIn0
    )
    {
        const int shapeSize = size(shOut);
        const int gridSize  = __umul24(blockDim.x, gridDim.x);
              int i;

        for ( i =  __umul24(blockDim.x, blockIdx.x) + threadIdx.x
            ; i <  shapeSize
            ; i += gridSize )
        {
            const typename Shape ix = fromIndex(shOut, i);
            $decls:(getIn0 "ix")
            $decls:(getIn1 "ix")
            $decls:env
            $stms:(setOut "i" combine)
        }
    }
  |]
  where
    (argOut, _, setOut) = setters tyOut


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

stencilAccess
    :: Int                              -- array de Bruijn index
    -> Int                              -- array dimensionality
    -> [Type]                           -- array type (texture memory)
    -> Boundary [Exp]                   -- how to handle boundary array access
    -> [[Int]]                          -- all stencil index offsets, top left to bottom right
    -> CGM ( [Definition]               -- texture-reference definitions
           , String -> [InitGroup] )    -- array indexing
stencilAccess base dim stencil boundary offsets' = do
  subs          <- subscripts base
  return ( textures
         , \ix -> concatMap (get ix) subs )
  where
    n           = length stencil
    sh          = "shIn"    ++ show  base
    arr x       = "stencil" ++ shows base "_a" ++ show (x `mod` n)
    textures    = zipWith cglobal stencil (map arr [n-1, n-2 .. 0])
    --
    offsets     :: Array Int [Int]
    offsets     = let (i,xs)            = rev 0 offsets' []
                      rev !k []     a   = (k, a)
                      rev !k (l:ls) a   = rev (k+1) ls (l:a)
                  in  listArray (0, i-1) xs
    --
    get ix (i,t,v) = case boundary of
      Clamp             -> bounded "clamp"
      Mirror            -> bounded "mirror"
      Wrap              -> bounded "wrap"
      Constant c        -> inRange c
      where
        j       = 'j':shows base "_a" ++ show i
        k       = 'k':shows base "_a" ++ show i
        --
        bounded f
          = [cdecl| const int $id:j = $exp:ix'; |]
          : [cdecl| const $ty:t $id:(show v) = $exp:(indexArray t (cvar (arr i)) (cvar j)); |]
          : []
          where
            ix'  = case offsets ! div i n of
              ks | all (== 0) ks        -> [cexp| i |]
                 | otherwise            ->
                    let iz = ccall "shape" $ zipWith (\a o -> [cexp| $id:ix . $id:('a':show a) + $int:o |]) [dim-1,dim-2..0] ks
                    in  [cexp| toIndex( $id:sh, $exp:(ccall f [cvar sh, iz]) ) |]
        --
        inRange c = case offsets ! div i n of
          ks | all (== 0) ks    -> [[cdecl| const $ty:t $id:(show v) = $exp:(indexArray t (cvar (arr i)) (cvar "i")); |]]
             | otherwise        -> [cdecl| const typename Shape $id:j = $exp:sh'; |]
                                 : [cdecl| const typename bool  $id:k = $exp:ok ; |]
                                 : [cdecl| const $ty:t $id:(show v) = $id:k ? $exp:(indexArray t (cvar (arr i)) (ccall "toIndex" [cvar sh, cvar j]))
                                                                            : $exp:(reverse c !! mod i n); |]
                                 : []
             where
               sh' = ccall "shape" $ zipWith (\a o -> [cexp| $id:ix . $id:('a':show a) + $int:o |]) [dim-1,dim-2..0] ks
               ok  = [cexp| inRange( $id:sh, $id:j ) |]


{--
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Tuple
-- Copyright   : [2010..2011] Ben Lever
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Stencil (
  mkStencilType, mkStencilGet, mkStencilGather, mkStencilApply
)
where

import Language.C
import Data.Array.Accelerate.CUDA.CodeGen.Data
import Data.Array.Accelerate.CUDA.CodeGen.Util

import Data.Array.Accelerate.Type


-- Getter function for a single element of a stencil array. These arrays are
-- read via texture memory, and additionally we need to specify the boundary
-- condition handler.
--
mkStencilGet :: Int -> Boundary [CExpr] -> [CType] -> [CExtDecl]
mkStencilGet base bndy ty =
  case bndy of
    Constant e -> [mkConstant e, mkFun [constant]]
    Clamp      -> [mkFun (boundary "clamp")]
    Mirror     -> [mkFun (boundary "mirror")]
    Wrap       -> [mkFun (boundary "wrap")]
  where
    dim   = typename (subscript "DimIn")
    mkFun = mkDeviceFun' (subscript "get") (typename (subscript "TyIn")) [(dim, "sh"), (dim, "ix")]

    mkConstant = mkDeviceFun (subscript "constant") (typename (subscript "TyIn")) []

    constant   = CBlockStmt $
      CIf (ccall "inRange" [cvar "sh", cvar "ix"])
      (CCompound [] [ CBlockDecl (CDecl [CTypeQual (CConstQual internalNode), CTypeSpec (CTypeDef (internalIdent "int") internalNode)] [(Just (CDeclr (Just (internalIdent "i")) [] Nothing [] internalNode),Just (CInitExpr (ccall "toIndex" [cvar "sh", cvar "ix"]) internalNode),Nothing)] internalNode)
                    , initA
                    , CBlockStmt (CReturn (Just (cvar "r")) internalNode) ]
                    internalNode)
      (Just (CCompound [] [CBlockStmt (CReturn (Just (ccall (subscript "constant") [])) internalNode)] internalNode))
      internalNode

    boundary f =
      [ CBlockDecl (CDecl [CTypeQual (CConstQual internalNode), CTypeSpec (CTypeDef (internalIdent "int") internalNode)] [(Just (CDeclr (Just (internalIdent "i")) [] Nothing [] internalNode),Just (CInitExpr (ccall "toIndex" [cvar "sh", ccall f [cvar "sh", cvar "ix"]]) internalNode),Nothing)] internalNode)
      , initA
      , CBlockStmt (CReturn (Just (CVar (internalIdent "r") internalNode)) internalNode)
      ]

    subscript = (++ show base)
    ix        = cvar "i"
    arr c     = cvar (subscript "stencil" ++ "_a" ++ show c)

    initA = CBlockDecl
      (CDecl [CTypeSpec (CTypeDef (internalIdent (subscript "TyIn")) internalNode)]
             [( Just (CDeclr (Just (internalIdent "r")) [] Nothing [] internalNode)
              , Just . mkInitList . reverse $ zipWith indexA (reverse ty) (enumFrom 0 :: [Int])
              , Nothing)]
             internalNode)

    indexA [CDoubleType _] c = ccall "indexDArray" [arr c, ix]
    indexA _               c = ccall "indexArray"  [arr c, ix]


-- A structure to hold all components of a stencil, mimicking our nested-tuple
-- representation for neighbouring elements.
--
mkStencilType :: Int -> Int -> [CType] -> CExtDecl
mkStencilType subscript size
  = mkStruct ("Stencil" ++ show subscript) False False
  . concat . replicate size


-- Gather all neighbouring array elements for our stencil
--
mkStencilGather :: Int -> Int -> [CType] -> [[Int]] -> CExtDecl
mkStencilGather base dim ty ixs =
  mkDeviceFun' (subscript "gather") (typename (subscript "Stencil")) [(dimIn, "sh"), (dimIn, "ix")] body
  where
    dimIn     = typename (subscript "DimIn")
    subscript = (++ show base)

    plus a b  = CBinary CAddOp a b internalNode
    cint c    = CConst $ CIntConst (cInteger (toInteger c)) internalNode
    offset is
      | dim == 1  = [cvar "ix" `plus` cint (head is)]
      | otherwise = zipWith (\c i -> CMember (cvar "ix") (internalIdent ('a':show c)) False internalNode `plus` cint i) [dim-1, dim-2 ..] is

    initX x is = CBlockDecl
      (CDecl [CTypeQual (CConstQual internalNode), CTypeSpec (CTypeDef (internalIdent (subscript "TyIn")) internalNode)]
             [( Just (CDeclr (Just (internalIdent ('x':show x))) [] Nothing [] internalNode)
              , Just (CInitExpr (ccall (subscript "get") [cvar "sh", ccall "shape" (offset is)]) internalNode)
              , Nothing)]
             internalNode)

    initS =
      let xs    = let l = length ixs in [l-1, l-2 .. 0]
          names = case length ty of
            1 -> [ cvar ('x':show x) | x <- xs]
            n -> [ CMember (cvar ('x':show x)) (internalIdent ('a':show c)) False internalNode | x <- xs , c <- [n-1,n-2..0]]
      in
      CBlockDecl
      (CDecl [CTypeSpec (CTypeDef (internalIdent (subscript "Stencil")) internalNode)]
             [( Just (CDeclr (Just (internalIdent "r")) [] Nothing [] internalNode)
              , Just (mkInitList names)
              , Nothing)]
             internalNode)

    body =
      zipWith initX [0::Int ..] (reverse ixs) ++
      [ initS
      , CBlockStmt (CReturn (Just (CVar (internalIdent "r") internalNode)) internalNode) ]


mkStencilApply :: Int -> [CExpr] -> CExtDecl
mkStencilApply argc
  = mkDeviceFun "apply" (typename "TyOut")
  $ map (\n -> (typename ("Stencil" ++ show n), 'x':show n)) [argc-1, argc-2 .. 0]
--}
