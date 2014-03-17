{-# LANGUAGE GADTs               #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Streaming
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Frederik M. Madsen <fmma@diku.dk>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Streaming (

  mkToStream, mkFromStream

) where

import Language.C.Quote.CUDA
import qualified Language.C.Syntax                      as C
import Foreign.CUDA.Analysis.Device

import Data.Array.Accelerate.Array.Sugar                ( (:.), Array, Shape, Elt, Vector, Scalar )
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base

streamIdParam :: C.Param
streamIdParam = [cparam| $ty:cint $id:sid |] where sid = "_sid"

mkToStream
    :: forall aenv sh a. (Shape sh, Elt a)
    => DeviceProperties
    -> Gamma aenv
    -> CUDelayedAcc aenv (sh :. Int) a
    -> [CUTranslSkel aenv [Array sh a]]
mkToStream dev aenv arr
  | CUDelayed _ (CUFun1 _ get) _ <- arr
  = return
  $ CUTranslSkel "tostream" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    tostream
    (
        $params:([streamIdParam]),
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
            $items:(i           .=. cfromIndex shOut "ix" "tmp")
            $items:({-j-} sh    .=. {-ctoIndex shIn-} (i ++ [(cint, "_sid")]))
            $items:(setOut "ix" .=. get {-[j]-} sh)
        }
    }
  |]
  where
    (texIn, argIn)              = environment dev aenv
    (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sh a)
--    (argArr, shIn, arr)         = readArray "In" (undefined :: Array (sh :. Int) a)
    (i, _, _)                   = locals "i_" (undefined :: sh)
    (sh, _, _)                  = locals "sh_" (undefined :: sh :. Int)
--    ([j], _, _)                 = locals "j" (undefined :: Int)


mkFromStream
    :: forall aenv a. (Elt a)
    => DeviceProperties
    -> Gamma aenv
--    -> CUDelayedAcc aenv (sh :. Int) a
    -> [CUTranslSkel aenv (Vector a)]
mkFromStream dev aenv
  | CUDelayed _ _ (CUFun1 _ get) <- arr
  = return
  $ CUTranslSkel "fromstream" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    fromstream
    (
        $params:([streamIdParam]),
        $params:argIn,
        $params:argOut,
        $params:argArr
    )
    {
        const int shapeSize     = $exp:(csize shIn);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(setOut "_sid" .=. get ([cintegral (0 :: Int)] :: [C.Exp]))
        }
    }
  |]
  where
    (texIn, argIn)        = environment dev aenv
    (argOut, _, setOut)   = writeArray "Out" (undefined :: Vector a)
    (argArr, shIn, arr)   = readArray "In" (undefined :: Scalar a)
