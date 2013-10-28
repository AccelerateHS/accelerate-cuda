{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Stencil
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Stencil (

  mkStencil, mkStencil2

) where

import Control.Applicative
import Control.Monad.State.Strict
import Foreign.CUDA.Analysis
import Language.C.Quote.CUDA
import qualified Language.C.Syntax                      as C

import Data.Array.Accelerate.Type                       ( Boundary(..) )
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, shapeToList )
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.Analysis.Stencil
import Data.Array.Accelerate.CUDA.AST                   hiding ( stencil, stencilAccess )
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type


-- Map a stencil over an array.  In contrast to 'map', the domain of a stencil
-- function is an entire /neighbourhood/ of each array element.  Neighbourhoods
-- are sub-arrays centred around a focal point.  They are not necessarily
-- rectangular, but they are symmetric and have an extent of at least three in
-- each dimensions. Due to this symmetry requirement, the extent is necessarily
-- odd.  The focal point is the array position that determines the single output
-- element for each application of the stencil.
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
-- To improve performance on older (1.x series) devices, the input array(s) are
-- read through the texture cache.
--
mkStencil
    :: forall aenv sh stencil a b. (Stencil sh a stencil, Elt b)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun1 aenv (stencil -> b)
    -> Boundary (CUExp aenv a)
    -> [CUTranslSkel aenv (Array sh b)]
mkStencil dev aenv (CUFun1 dce f) boundary
  = return
  $ CUTranslSkel "stencil" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecl:(cdim "Shape" dim)
    $edecls:texIn
    $edecls:texStencil

    extern "C" __global__ void
    stencil
    (
        $params:argIn,
        $params:argOut,
        $params:argStencil
    )
    {
        const int shapeSize     = size(shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            const typename Shape sh = fromIndex( shOut, ix );
            $items:(dce xs      .=. stencil sh)
            $items:(setOut "ix" .=. f xs)
        }
    }
  |]
  where
    dim                 = expDim (undefined :: Exp aenv sh)
    (texIn,  argIn)     = environment dev aenv
    (argOut, setOut)    = setters "Out" (undefined :: Array sh b)
    ix                  = cvar "ix"
    sh                  = "sh"
    (xs,_,_)            = locals "x" (undefined :: stencil)
    dx                  = offsets (undefined :: Fun aenv (stencil -> b)) (undefined :: OpenAcc aenv (Array sh a))

    (texStencil, argStencil, stencil) = stencilAccess True "Stencil" "w" dev dx ix boundary dce


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
mkStencil2
    :: forall aenv sh stencil1 stencil2 a b c.
       (Stencil sh a stencil1, Stencil sh b stencil2, Elt c)
    => DeviceProperties
    -> Gamma aenv
    -> CUFun2 aenv (stencil1 -> stencil2 -> c)
    -> Boundary (CUExp aenv a)
    -> Boundary (CUExp aenv b)
    -> [CUTranslSkel aenv (Array sh c)]
mkStencil2 dev aenv (CUFun2 dce1 dce2 f) boundary1 boundary2
  = return
  $ CUTranslSkel "stencil2" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecl:(cdim "Shape" dim)
    $edecls:texIn
    $edecls:texS1
    $edecls:texS2

    extern "C" __global__ void
    stencil2
    (
        $params:argIn,
        $params:argOut,
        $params:argS1,
        $params:argS2
    )
    {
        const int shapeSize     = size(shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            const typename Shape sh = fromIndex( shOut, ix );

            $items:(dce1 xs     .=. stencil1 sh)
            $items:(dce2 ys     .=. stencil2 sh)
            $items:(setOut "ix" .=. f xs ys)
        }
    }
  |]
  where
    dim                 = expDim (undefined :: Exp aenv sh)
    (texIn,  argIn)     = environment dev aenv
    (argOut, setOut)    = setters "Out" (undefined :: Array sh c)
    ix                  = cvar "ix"
    sh                  = "sh"
    (xs,_,_)            = locals "x" (undefined :: stencil1)
    (ys,_,_)            = locals "y" (undefined :: stencil2)

    (dx1, dx2)          = offsets2 (undefined :: Fun aenv (stencil1 -> stencil2 -> c))
                                   (undefined :: OpenAcc aenv (Array sh a))
                                   (undefined :: OpenAcc aenv (Array sh b))

    (texS1, argS1, stencil1) = stencilAccess False "Stencil1" "w" dev dx1 ix boundary1 dce1
    (texS2, argS2, stencil2) = stencilAccess False "Stencil2" "z" dev dx2 ix boundary2 dce2


-- Generate declarations for reading in a stencil pattern surrounding a given
-- focal point. The first parameter determines whether it is safe to use linear
-- indexing at the centroid position. This is true for:
--
--  * stencil1
--  * stencil2 if both input stencil have the same dimensionality
--
stencilAccess
    :: forall aenv sh e. (Shape sh, Elt e)
    => Bool                                     -- linear indexing at centroid?
    -> Name                                     -- array group name
    -> Name                                     -- secondary group name, for fresh variables
    -> DeviceProperties                         -- properties of currently executing device
    -> [sh]                                     -- list of offset indices
    -> C.Exp                                    -- linear index of the centroid
    -> Boundary (CUExp aenv e)                  -- stencil boundary condition
    -> ([C.Exp] -> [(Bool,C.Exp)])              -- dead code elimination flags for this var
    -> ( [C.Definition]                         -- input arrays as texture references; or
       , [C.Param]                              -- function arguments
       , (Name -> ([C.BlockItem], [C.Exp])) )   -- read data at a given shape centroid
stencilAccess linear grp grp' dev shx centroid boundary dce
  = (texStencil, argStencil, stencil)
  where
    stencil ix = flip evalState 0 $ do
      (envs, xs) <- mapAndUnzipM (access ix . shapeToList) shx

      let (envs', xs') = unzip
                       $ eliminate
                       $ zip envs
                       $ unconcat (map length xs)
                       $ dce (concat xs)

      return ( concat envs', concat xs' )

    -- Filter unused components of the stencil. Environment bindings are shared
    -- between tuple components of each cursor position, so filter these out
    -- only if all elements of that position are unused.
    --
    unconcat :: [Int] -> [a] -> [[a]]
    unconcat []     _  = []
    unconcat (n:ns) xs = let (h,t) = splitAt n xs in h : unconcat ns t

    eliminate :: [ ([C.BlockItem], [(Bool, C.Exp)]) ] -> [ ([C.BlockItem], [C.Exp]) ]
    eliminate []         = []
    eliminate ((e,v):xs) = (e', x) : eliminate xs
      where
        (flags, x)      = unzip v
        e' | or flags   = e
           | otherwise  = []

    -- Generate the entire stencil, including any local environment bindings
    --
    access :: Name -> [Int] -> State Int ([C.BlockItem], [C.Exp])
    access ix dx = case boundary of
      Clamp                     -> bounded "clamp"
      Mirror                    -> bounded "mirror"
      Wrap                      -> bounded "wrap"
      Constant (CUExp (_,c))    -> inrange c            -- constant value: no environment possible

      where
        focus                   = all (==0) dx
        dim                     = expDim (undefined :: Exp aenv sh)
        cursor
          | all (==0) dx        = cvar ix
          | otherwise           = ccall "shape"
                                $ zipWith (\a b -> [cexp| $exp:a + $int:b |]) (cshape dim ix) (reverse dx)

        bounded f
          | focus && linear     = return $ ( [], getStencil centroid )
          | otherwise           = do
              j <- fresh
              return ( if focus then [C.BlockDecl [cdecl| const int $id:j = toIndex( $id:shIn, $id:ix ); |]]
                                else [C.BlockDecl [cdecl| const int $id:j = toIndex( $id:shIn, $exp:(ccall f [cvar shIn, cursor]) ); |]]
                     , getStencil (cvar j) )

        inrange cs
          | focus && linear     = return ( [], getStencil centroid )
          | focus               = do
              j <- fresh
              return ( [C.BlockDecl [cdecl| const int $id:j = toIndex( $id:shIn, $id:ix ); |]]
                     , getStencil (cvar j) )

          | otherwise           = do
              j     <- fresh
              i     <- fresh
              p     <- fresh
              return $ ( [ C.BlockDecl [cdecl| const typename Shape $id:j = $exp:cursor; |]
                         , C.BlockDecl [cdecl| const typename bool  $id:p = inRange( $id:shIn, $id:j ); |]
                         , C.BlockDecl [cdecl| const int            $id:i = toIndex( $id:shIn, $id:j ); |] ]
                       , zipWith (\a c -> [cexp| $id:p ? $exp:a : $exp:c |]) (getStencil (cvar i)) cs )

    -- Extra parameters for accessing the stencil data. We are doing things a
    -- little out of the ordinary, so don't get this "for free". sadface.
    --
    getStencil ix       = zipWith (\t a -> indexArray dev t a ix) (eltType (undefined :: e)) (map cvar stencilIn)
    (shIn, stencilIn)   = namesOfArray grp (undefined :: e)
    (texStencil, argStencil)
      | computeCapability dev < Compute 2 0 = let (d,p) = arrayAsTex (undefined :: Array sh e) grp in (d,[p])
      | otherwise                           = ([], arrayAsArg (undefined :: Array sh e) grp)

    -- Generate a fresh variable name
    --
    fresh :: State Int Name
    fresh = do
      n <- get <* modify (+1)
      return $ grp' ++ show n

