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

import Foreign.CUDA.Analysis
import Language.C.Quote.CUDA

import Data.Array.Accelerate.Type                       ( Boundary(..) )
import Data.Array.Accelerate.Array.Sugar                ( Array, Elt )
import Data.Array.Accelerate.Analysis.Stencil
import Data.Array.Accelerate.CUDA.AST                   hiding ( stencil, stencilAccess )
import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Stencil.Extra


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
        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(sh .=. cfromIndex shOut "ix" "tmp")
            $items:stencilBody
        }
    }
  |]
  where
    (texIn,  argIn)             = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh b)
    (sh, _, _)                  = locals "sh" (undefined :: sh)
    (xs,_,_)                    = locals "x" (undefined :: stencil)

    dx  = offsets (undefined :: Fun aenv (stencil -> b))
                  (undefined :: OpenAcc aenv (Array sh a))

    (texStencil, argStencil, safeIndex)   = stencilAccess dev True True  "Stencil" "Stencil" "w" "ix" dx boundary dce
    (_,          _,          unsafeIndex) = stencilAccess dev True False "Stencil" "Stencil" "w" "ix" dx boundary dce

    stencilBody
      | computeCapability dev < Compute 1 2     = with safeIndex
      | otherwise                               =
          [[citem| if ( __all( $exp:(insideRegion shOut (borderRegion dx) (map rvalue sh)) ) ) {
                       $items:(with unsafeIndex)
                   } else {
                       $items:(with safeIndex)
                   } |]]

      where
        with stencil = (dce xs      .=. stencil sh) ++
                       (setOut "ix" .=. f xs)



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
mkStencil2 dev aenv stencil boundary1 boundary2
  = [ mkStencil2' dev False aenv stencil boundary1 boundary2
    , mkStencil2' dev True  aenv stencil boundary1 boundary2
    ]

mkStencil2'
    :: forall aenv sh stencil1 stencil2 a b c.
       (Stencil sh a stencil1, Stencil sh b stencil2, Elt c)
    => DeviceProperties
    -> Bool                                     -- are the source arrays the same extent?
    -> Gamma aenv
    -> CUFun2 aenv (stencil1 -> stencil2 -> c)
    -> Boundary (CUExp aenv a)
    -> Boundary (CUExp aenv b)
    -> CUTranslSkel aenv (Array sh c)
mkStencil2' dev sameExtent aenv (CUFun2 dce1 dce2 f) boundary1 boundary2
  = CUTranslSkel "stencil2" [cunit|

    $esc:("#include <accelerate_cuda.h>")
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
        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(sh        .=. cfromIndex shOut "ix" "tmp")
            $items:stencilBody
        }
    }
  |]
  where
    (texIn,  argIn)             = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh c)
    (sh, _, _)                  = locals "sh" (undefined :: sh)
    (xs,_,_)                    = locals "x" (undefined :: stencil1)
    (ys,_,_)                    = locals "y" (undefined :: stencil2)
    grp1                        = "Stencil1"
    grp2                        = "Stencil2"

    -- If the source arrays are the same extent, twiddle the names a bit so that
    -- code generation refers to the same set of shape variables. Then, if there
    -- are any duplicate calculations, hope that the CUDA compiler is smart
    -- enough and spots this.
    --
    sh1                         = grp1
    sh2 | sameExtent            = sh1
        | otherwise             = grp2

    (dx1, dx2)  = offsets2 (undefined :: Fun aenv (stencil1 -> stencil2 -> c))
                           (undefined :: OpenAcc aenv (Array sh a))
                           (undefined :: OpenAcc aenv (Array sh b))

    border      = zipWith max (borderRegion dx1) (borderRegion dx2)

    (texS1, argS1, safeIndex1)   = stencilAccess dev sameExtent True  grp1 sh1 "w" "ix" dx1 boundary1 dce1
    (_, _,         unsafeIndex1) = stencilAccess dev sameExtent False grp1 sh1 "w" "ix" dx1 boundary1 dce1

    (texS2, argS2, safeIndex2)   = stencilAccess dev sameExtent True  grp2 sh2 "z" "ix" dx2 boundary2 dce2
    (_, _,         unsafeIndex2) = stencilAccess dev sameExtent False grp2 sh2 "z" "ix" dx2 boundary2 dce2

    stencilBody
      | computeCapability dev < Compute 1 2     = with safeIndex1 safeIndex2
      | otherwise                               =
          [[citem| if ( __all( $exp:(insideRegion shOut border (map rvalue sh)) ) ) {
                       $items:(with unsafeIndex1 unsafeIndex2)
                   } else {
                       $items:(with safeIndex1 safeIndex2)
                   } |]]

      where
        with stencil1 stencil2 =
          (dce1 xs     .=. stencil1 sh) ++
          (dce2 ys     .=. stencil2 sh) ++
          (setOut "ix" .=. f xs ys)

