{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ImpredicativeTypes  #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}

-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Streaming
--
-- Maintainer  : Frederik Meisner Madsen <fmma@diku.dk>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Streaming (

  mkToSeq

) where

import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis.Device

import Data.Array.Accelerate.Error                      ( internalError )
import Data.Array.Accelerate.Array.Representation       ( SliceIndex(..) )
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, EltRepr )
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base

mkToSeq
    :: forall slix aenv sh co sl a. (Shape sl, Shape sh, Elt a)
    => SliceIndex slix
                  (EltRepr sl)
                  co
                  (EltRepr sh)
    -> DeviceProperties
    -> Gamma aenv
    -> CUDelayedAcc aenv sh a
    -> CUTranslSkel aenv (Array sl a)
mkToSeq slix dev aenv arr
  | CUDelayed (CUExp shIn) (CUFun1 _ get) _ <- arr
  = CUTranslSkel "tostream" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    tostream
    (
        $params:argIn,
        $params:slIn,
        $params:argOut
    )
    {
        $items:(sh .=. shIn)

        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(src  .=. cfromIndex shOut "ix" "tmp")
            $items:(setOut "ix" .=. get fullsrc)
        }
    }
  |]
    where
      (slIn, _, cosrc)            = cslice slix "cosrc"
      (sh, _, _)                  = locals "shIn" (undefined :: sh)
      (src, _, _)                 = locals "src" (undefined :: sl)
      fullsrc                     = reverse (combine slix (reverse src) (reverse cosrc))
      (texIn, argIn)              = environment dev aenv
      (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sl a)

combine :: SliceIndex slix sl co dim -> [a] -> [a] -> [a]
combine SliceNil [] [] = []
combine (SliceAll   sl) (x:xs) ys = x:(combine sl xs ys)
combine (SliceFixed sl) xs (y:ys) = y:(combine sl xs ys)
combine _ _ _ = $internalError "mkToSeq" "Something went wrong with the slice index."

