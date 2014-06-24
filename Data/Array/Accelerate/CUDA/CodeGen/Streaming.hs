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

  mkToStream, mkFromStream

) where

import Language.C.Quote.CUDA
import qualified Language.C.Syntax                      as C
import Foreign.CUDA.Analysis.Device

import Data.Array.Accelerate.Error                      ( internalError )
import Data.Array.Accelerate.Array.Representation       ( SliceIndex(..) )
import Data.Array.Accelerate.Array.Sugar                ( Array, Shape, Elt, Vector, EltRepr )
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base

import Data.Monoid                                      ( mempty )

streamIdParam :: C.Param
streamIdParam = [cparam| $ty:cint $id:sid |] where sid = "_sid"

mkToStream
    :: forall slix aenv sh co sl a. (Shape sl, Shape sh, Elt a)
    => SliceIndex slix
                  (EltRepr sl)
                  co
                  (EltRepr sh)
    -> DeviceProperties
    -> Gamma aenv
    -> CUDelayedAcc aenv sh a
    -> CUTranslSkel aenv (Array sl a)
mkToStream slix dev aenv arr
  | CUDelayed (CUExp shIn) _ (CUFun1 _ get) <- arr
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
            $items:(isrc .=. ctoIndex sh fullsrc)
            $items:(setOut "ix" .=. get [isrc])
        }
    }
  |]
    where
      (slIn, _, cosrc)            = cslice slix "cosrc"
      (sh, _, _)                  = locals "shIn" (undefined :: sh)
      (src, _, _)                 = locals "src" (undefined :: sl)
      fullsrc                     = combine slix src cosrc
      (texIn, argIn)              = environment dev aenv
      (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array sl a)
      ([isrc], _, _)              = locals "isrc" (undefined :: Int)

combine :: SliceIndex slix sl co dim -> [a] -> [a] -> [a]
combine SliceNil [] [] = []
combine (SliceAll   sl) (x:xs) ys = x:(combine sl xs ys)
combine (SliceFixed sl) xs (y:ys) = y:(combine sl xs ys)
combine _ _ _ = $internalError "mkToStream" "Something went wrong with the slice index."

mkFromStream
    :: forall aenv sh a. (Shape sh, Elt a)
    => sh
    -> DeviceProperties
    -> CUTranslSkel aenv (Vector a)
mkFromStream _ dev
  | CUDelayed _ _ (CUFun1 _ get) <- arr
  = CUTranslSkel "fromstream" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    fromstream
    (
        $params:argIn,
        $params:([streamIdParam]),
        $params:argOut,
        $params:argArr
    )
    {
        const int shapeSize     = $exp:(csize shIn);
        const int gridSize      = $exp:(gridSize dev);
              int ix, jx;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            jx = ix + _sid;
            $items:(setOut "jx" .=. get [cvar "ix"])
        }
    }
  |]
  where
    (texIn, argIn)        = environment dev mempty
    (argOut, _, setOut)   = writeArray "Out" (undefined :: Vector a)
    (argArr, shIn, arr)   = readArray "In" (undefined :: Array sh a)

