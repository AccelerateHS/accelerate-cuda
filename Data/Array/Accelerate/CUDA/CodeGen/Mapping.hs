{-# LANGUAGE GADTs               #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Mapping
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
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

import qualified Data.Array.Accelerate.BackendKit.IRs.SimpleAcc as S
-- import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt )
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
mkMap :: S.Type -> S.Type 
      -> DeviceProperties -> Gamma -> CUFun1 -> CUDelayedAcc  -> [CUTranslSkel]
mkMap tyIn tyOut dev aenv fun arr
  | CUFun1 dce f                 <- fun
  , CUDelayed _ _ (CUFun1 _ get) <- arr
  = return
  $ CUTranslSkel "map" [cunit|

    $esc:("#include <accelerate_cuda_extras.h>")
    $edecls:texIn

    extern "C" __global__ void
    map
    (
        $params:argIn,
        $params:argOut
    )
    {
        const int shapeSize     = size(shOut);
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
    (texIn, argIn)      = environment  dev aenv
    (argOut, setOut)    = setters tyOut "Out" 
    (x, _, _)           = locals tyIn "x" 
    ix                  = [cvar "ix"]

