{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ImpredicativeTypes  #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE QuasiQuotes         #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeOperators       #-}

-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.Streaming
--
-- Maintainer  : Frederik Meisner Madsen <fmma@diku.dk>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.CodeGen.Streaming (

  mkToSeq, mkInplaceUpdate,

) where

import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis.Device

import Data.Array.Accelerate.Error                      ( internalError )
import Data.Array.Accelerate.Array.Representation       ( SliceIndex(..) )
import Data.Array.Accelerate.Array.Sugar                ( Array, Vector, Shape, Elt, EltRepr, (:.) )
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.CodeGen.Base


{- 

The mkToSeq kernel. A kernel to backpermute a sub-sequence (chunk) of
slices from a CUDelayedAcc.

Example:
Let the iteration space be defined by sl as

sl = SliceFixed $ SliceAll $ SliceFixed SliceNil

meaning that the elements of the sequence are vectors, and the
sequence is obtained by slicing a 3-dimensional array on the outer
dimension and the inner dimension.

Assuming unbounded chunk size, the toSeq kernel is a permutation
that brings the iteration space to the outer dimension:

perm :: DIM3 -> DIM2
perm (Z :. f0 :. a :. f1) = Z :. f0 * f1 :. a

(For backpermutation, we need the inverse of this which requires a
call to "fromIndex")

If the chunk size is smaller than the iteration space, we just
manifest this array in chunks by packpermute.

chunkShape i k = Z :. k :. <elemShape>
chunkPerm i k (Z :. j :. <elemIx>) = Z :. i + j :. <elemIx>

yielding the desired combined backward permutation function
  p :: DIM2 -> DIM3
  p = perm^(-1) . chunkPerm

-}
mkToSeq
    :: forall slix senv aenv sh co sl a. (Shape sl, Shape sh, Elt a)
    => SliceIndex slix
                  (EltRepr sl)
                  co
                  (EltRepr sh)
    -> DeviceProperties
    -> Gamma aenv
    -> CUDelayedAcc aenv sh a
    -> CUTranslSkel aenv (Array (sl :. Int) a)
mkToSeq slix dev aenv arr
  | CUDelayed (CUExp shIn) (CUFun1 dce get) _ <- arr
  = CUTranslSkel "toSeq" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    toSeq
    (
        $params:argIn,
        const $ty:cint j,
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
            $items:(sl .=. offsetOuter $ cfromIndex shOut "ix" "tmpsl")
            $items:(co .=. cfromIndex (coShape slix sh) (head sl) "tmpco")
            $items:(setOut "ix" .=. get (combine slix (tail sl) co))
        }
    }
  |]
    where
      (sh, _, _) = locals "shIn" (undefined :: sh)
      (sl, _, _) = locals "sl" (undefined :: (sl :. Int))
      (co, _, _) = localShape "co" (length sh - (length sl - 1))
      (texIn, argIn)              = environment dev aenv
      (argOut, shOut, setOut)     = writeArray "Out" (undefined :: Array (sl :. Int) a)
--      offsetOuter :: ([C.BlockItem], [C.Exp]) -> ([C.BlockItem], [C.Exp])
      offsetOuter (bs, es) = (bs, [cexp|$exp:(head es) + j|] : tail es)

coShape :: SliceIndex slix sl co dim
        -> [a] -- dim
        -> [a] -- co
coShape slix dim = reverse (go slix (reverse dim)) -- blasted c-exp lists are backwards!
  where
    go :: SliceIndex slix sl co dim -> [a] -> [a]
    go SliceNil [] = []
    go (SliceAll   sl) (_:xs) = go sl xs
    go (SliceFixed sl) (x:xs) = x : go sl xs
    go _ _ = $internalError "coShape" "Something went wrong with the slice index."

combine :: SliceIndex slix sl co dim 
        -> [a] -- sl
        -> [a] -- co
        -> [a] -- dim
combine slix sl co = reverse (go slix (reverse sl) (reverse co))
  where
    go :: SliceIndex slix sl co dim -> [a] -> [a] -> [a]
    go SliceNil [] [] = []
    go (SliceAll   sl) (x:xs) ys = x:(go sl xs ys)
    go (SliceFixed sl) xs (y:ys) = y:(go sl xs ys)
    go _ _ _ = $internalError "combine" "Something went wrong with the slice index."

mkInplaceUpdate :: forall aenv e. (Elt e)
      => DeviceProperties
      -> Gamma aenv
      -> CUFun2 aenv (e -> e -> e)
      -> CUTranslSkel aenv (Vector e)
mkInplaceUpdate dev aenv fun
  | CUFun2 dcea dceb f            <- fun
  , CUDelayed _ _ (CUFun1 _ get)  <- arr
  , CUDelayed _ _ (CUFun1 _ getd) <- arrd
  = CUTranslSkel "inplaceUpdate" [cunit|

    $esc:("#include <accelerate_cuda.h>")
    $edecls:texIn

    extern "C" __global__ void
    inplaceUpdate
    (
        $params:argIn,
        $params:argOut,
        $params:argDelta
    )
    {
        const int shapeSize     = $exp:(csize shOut);
        const int gridSize      = $exp:(gridSize dev);
              int ix;

        for ( ix =  $exp:(threadIdx dev)
            ; ix <  shapeSize
            ; ix += gridSize )
        {
            $items:(dcea x      .=. getd ix)
            $items:(dceb y      .=. get ix)
            $items:(setOut "ix" .=. f x y)
        }
    }
  |]
  where
    (texIn, argIn)          = environment dev aenv
    (argOut, shOut, setOut) = writeArray "Out" (undefined :: Vector e)
    (_, _, arr)             = readArray  "Out" (undefined :: Vector e)
    (argDelta, _, arrd)     = readArray  "Delta" (undefined :: Vector e)
    (x, _, _)               = locals "x" (undefined :: e)
    (y, _, _)               = locals "y" (undefined :: e)
    ix                      = [cvar "ix"]
