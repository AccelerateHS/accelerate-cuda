{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ImpredicativeTypes  #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ViewPatterns        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Stencil.Extra
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Stencil.Extra (

  stencilAccess,

  cinRange, cclamp, cmirror, cwrap, insideRegion, borderRegion,

) where

-- standard library
import Prelude                                          hiding ( and, zipWith, zipWith3 )
import Data.List                                        ( transpose )
import Control.Monad
import Control.Monad.State.Strict                       ( State, StateT(..), evalState )

-- language-c
import Language.C.Quote.CUDA
import qualified Language.C.Syntax                      as C

-- friends
import Data.Array.Accelerate.Type                       ( Boundary(..) )
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, shapeToList )
import Data.Array.Accelerate.Analysis.Shape

import Foreign.CUDA.Analysis
import Data.Array.Accelerate.CUDA.AST                   hiding ( stencil, stencilAccess )
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Type

#include "accelerate.h"


-- Stencil Access
-- --------------

-- Generate declarations for reading in a stencil pattern surrounding a given
-- focal point.
--
stencilAccess
    :: forall aenv sh e. (Shape sh, Elt e)
    => DeviceProperties
    -> Bool                                     -- can we use linear indexing?
    -> Bool                                     -- do we need to do bounds checking?
    -> Name                                     -- array group name
    -> Name                                     -- seed name for temporary variables
    -> Name                                     -- linear index at the focus
    -> [sh]                                     -- list of offset indices
    -> Boundary (CUExp aenv e)                  -- stencil boundary condition
    -> (forall x. [x] -> [(Bool,x)])            -- dead code elimination flags
    -> ( [C.Definition]                                                 -- kernel texture reference definitions
       , [C.Param]                                                      -- kernel function arguments
       , forall x. Rvalue x => [x] -> ([C.BlockItem], [C.Exp]))         -- access stencil at given multidimensional index
stencilAccess dev doLinearIndexing doBoundsChecks grp tmp centroid shx boundary dce =
  ( decls, params, stencil )
  where
    (decls, params, shIn, getIn)        = readStencil dev grp (undefined :: Array sh e)

    getInAt ix = do
      j <- fresh
      let (env, arr) = getIn j
      return ( [citem| const $ty:cint $id:j = $exp:ix; |] : env, arr )

    -- Generate the entire stencil, reading elements from those positions of the
    -- pattern that are used and eliminating reads from those that are not.
    --
    stencil ix = withNameGen tmp $ do
      (envs, xs) <- mapAndUnzipM (access ix . shapeToList) shx

      let (envs', xs') = unzip
                       $ eliminate
                       $ zipWith (,) envs               -- our version of zipwith that checks lengths
                       $ unconcat (map length xs)
                       $ dce (concat xs)

      return ( concat envs', concat xs' )

    -- Read the stencil component at the given offset (second argument). This
    -- may generate additional environment terms, such as for the index
    -- calculations.
    --
    access :: Rvalue x => [x] -> [Int] -> Gen ([C.BlockItem], [C.Exp])
    access (map rvalue -> ix) dx
      | doBoundsChecks          = safeAccess
      | otherwise               = unsafeAccess
      where
        focus                   = all (==0) dx

        -- The current stencil position into the array, as a multidimensional index
        cursor | focus          = ix
               | otherwise      = zipWith (\i d -> [cexp| $exp:i + $int:d |]) ix (reverse dx)

        -- Read the array position without any bounds checks
        unsafeAccess
          | doLinearIndexing && focus   = return $ getIn centroid
          | otherwise                   = getInAt (ctoIndex shIn cursor)

        -- Read the array, applying appropriate bounds checks
        safeAccess = case boundary of
          Clamp                  -> bounded cclamp
          Mirror                 -> bounded cmirror
          Wrap                   -> bounded cwrap
          Constant (CUExp (_,c)) -> inrange c

        bounded f
          | focus               = unsafeAccess
          | otherwise           = getInAt (ctoIndex shIn (f shIn cursor))

        inrange cs
          | focus               = unsafeAccess
          | otherwise           = do
              (env, as) <- unsafeAccess
              p         <- fresh
              return ( [citem| const int $id:p = $exp:(cinRange shIn cursor); |] : env
                     , zipWith (\a c -> [cexp| $id:p ? $exp:a : $exp:c |]) as cs )



-- Filter unused components of the stencil. Environment bindings are shared
-- between tuple components of each cursor position, so filter these out only if
-- all elements of that position are unused.
--
eliminate :: [ ([a], [(Bool,b)]) ] -> [ ([a],[b]) ]
eliminate []         = []
eliminate ((e,v):xs) = (e', x) : eliminate xs
  where
    (flags, x)          = unzip v
    e' | or flags       = e
       | otherwise      = []


-- A simple fresh name supply
--
type Gen = State (Name,Int)

withNameGen :: Name -> Gen a -> a
withNameGen base f = evalState f (base,0)

fresh :: Gen Name
fresh = StateT $ \(base,n) -> return (base ++ show n, (base,n+1))


-- Boundary conditions
-- -------------------

-- Test whether the given multidimensional index lies in the inside region of
-- the stencil.
--
insideRegion
    :: [C.Exp]                  -- The shape of the array
    -> [Int]                    -- The width of the stencil in each direction
    -> [C.Exp]                  -- The index in question
    -> C.Exp
insideRegion shape border index = foldl1 and (zipWith3 inside shape border index)
  where
    inside sz dx i      = [cexp| $exp:i >= $int:dx && $exp:i < $exp:sz - $int:dx |]
    and x y             = [cexp| $exp:x && $exp:y |]


-- Given a list of stencil offset positions, calculate the size of the border
-- region along each dimension.
--
-- Note that this does not consider any positions of the stencil that are not
-- actually used. We assume the user is sensible and uses the minimally sized
-- stencil for their application, but this can still be problematic for
-- non-symmetric stencils. For example, a large stencil that uses elements from
-- only one quadrant.
--
borderRegion :: Shape sh => [sh] -> [Int]
borderRegion
  = reverse
  . map maximum
  . transpose
  . map shapeToList


-- Test whether an index lies within the boundaries of a shape (first argument)
--
cinRange :: [C.Exp] -> [C.Exp] -> C.Exp
cinRange []    []    = INTERNAL_ERROR(error) "inRange" "singleton index"
cinRange shape index = foldl1 and (zipWith inside shape index)
  where
    inside sz i = [cexp| ({ const $ty:cint _i = $exp:i; _i >= 0 && _i < $exp:sz; }) |]
    and x y     = [cexp| $exp:x && $exp:y |]

-- Clamp an index to the boundary of the shape (first argument)
--
cclamp :: [C.Exp] -> [C.Exp] -> [C.Exp]
cclamp = zipWith f
  where
    f sz i = [cexp| max(($ty:cint) 0, min( $exp:i, $exp:sz - 1 )) |]

-- Indices out of bounds of the shape are mirrored back in range. Assumes that
-- the array is at least as large as the stencil.
--
cmirror :: [C.Exp] -> [C.Exp] -> [C.Exp]
cmirror = zipWith f
  where
    f sz i = [cexp| ({ const $ty:cint _i  = $exp:i;
                       const $ty:cint _sz = $exp:sz;
                      _i < 0    ? -_i
                    : _i >= _sz ?  _sz - (_i - _sz + 2)
                    : _i; }) |]

-- Indices out of bounds are wrapped to the opposite edge of the shape
--
cwrap :: [C.Exp] -> [C.Exp] -> [C.Exp]
cwrap = zipWith f
  where
    f sz i = [cexp| ({ const $ty:cint _i  = $exp:i;
                       const $ty:cint _sz = $exp:sz;
                    _i < 0    ? _sz + _i
                  : _i >= _sz ? _i  - _sz
                  : _i; }) |]


-- Kernel parameters
-- -----------------

-- Generate kernel parameters for input arrays. This is similar to 'readArray',
-- but we force compute 1.x devices to read through the texture cache as well.
-- Note that using a multidimensional index result in error.
--
readStencil
    :: forall sh e. (Shape sh, Elt e)
    => DeviceProperties
    -> Name                             -- group names
    -> Array sh e                       -- dummy to fix the types
    -> ( [C.Definition]                 -- global definitions for stencils read via texture references (compute < 2.0)
       , [C.Param]                      -- function arguments for stencils read as arrays (compute >= 2.0)
       , [C.Exp]                        -- shape of the array
       , forall x. Rvalue x => x -> ([C.BlockItem], [C.Exp])    -- read elements from linear index
       )
readStencil dev grp dummy
  = let (sh, arrs)      = namesOfArray grp (undefined :: e)
        (decl, args)
          | computeCapability dev < Compute 2 0 = arrayAsTex dummy grp
          | otherwise                           = ([], arrayAsArg dummy grp)

        dim             = expDim (undefined :: Exp aenv sh)
        sh'             = cshape dim sh
        get ix          = ([], zipWith (\t a -> indexArray dev t (cvar a) (rvalue ix)) (eltType (undefined :: e)) arrs)
    in
    ( decl, args, sh', get )


-- Prelude'
-- --------

-- A version of 'zipWith' that requires the lists to be equal length
--
zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith f (x:xs) (y:ys) = f x y : zipWith f xs ys
zipWith _ []     []     = []
zipWith _ _      _      = INTERNAL_ERROR(error) "zipWith" "argument mismatch"

zipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
zipWith3 f (x:xs) (y:ys) (z:zs) = f x y z : zipWith3 f xs ys zs
zipWith3 _ []     []     []     = []
zipWith3 _ _      _      _      = INTERNAL_ERROR(error) "zipWith3" "argument mismatch"


-- Split a list into segments of given length
--
unconcat :: [Int] -> [a] -> [[a]]
unconcat []     _  = []
unconcat (n:ns) xs = let (h,t) = splitAt n xs in h : unconcat ns t

