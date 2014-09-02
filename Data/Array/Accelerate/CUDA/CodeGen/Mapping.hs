{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ImpredicativeTypes  #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Mapping
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Mapping (

  mkMap,

) where

import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis.Device

import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base


-- Apply the given unary function to each element of an array. Each thread
-- processes multiple elements, striding the array by the grid size.
--
-- map :: (Shape sh, Elt a, Elt b)
--     => (Exp a -> Exp b)
--     -> Acc (Array sh a)
--     -> Acc (Array sh b)
--
mkMap :: forall aenv sh a b. (Shape sh, Elt a, Elt b)
      => DeviceProperties
      -> Gamma aenv
      -> CUFun1 aenv (a -> b)
      -> CUDelayedAcc aenv sh a
      -> [CUTranslSkel aenv (Array sh b)]
mkMap dev aenv fun arr
  | CUFun1 dce f                 <- fun
  , CUDelayed _ _ (CUFun1 _ get) <- arr
  = return
  $ CUTranslSkel "map" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    map
    (
        $params:argIn,
        $params:argOut
    )
    {
        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(dce x       .=. get ix)
            $items:(setOut "ix" .=. f x)
        }
    }
  |]
  where
    (texIn, argIn)              = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh b)
    (x, _, _)                   = locals "x" (undefined :: a)
    ix                          = [cvar "ix"]

