{-# LANGUAGE UndecidableInstances, OverlappingInstances, IncoherentInstances #-}
{-# LANGUAGE BangPatterns, CPP, GADTs, ScopedTypeVariables, FlexibleInstances #-}
{-# LANGUAGE RankNTypes, TupleSections, TypeOperators, TypeSynonymInstances #-}
{-# OPTIONS -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Execute
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Execute (

  -- * Execute a computation under a CUDA environment
  executeAcc, executeAfun1

) where


-- friends
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Tuple
import Data.Array.Accelerate.Array.Representation               hiding (Shape, sliceIndex)
import qualified Data.Array.Accelerate.Interpreter              as I
import qualified Data.Array.Accelerate.Array.Data               as AD
import qualified Data.Array.Accelerate.Array.Representation     as R

import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.FullList                      ( FullList(..), List(..) )
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.Array.Sugar                   hiding
   (dim, size, index, shapeToList, sliceIndex)
import qualified Data.Array.Accelerate.CUDA.Array.Sugar         as Sugar
import qualified Data.Array.Accelerate.CUDA.Debug               as D ( message, dump_exec )


-- libraries
import Prelude                                                  hiding ( sum, exp )
import Control.Applicative                                      hiding ( Const )
import Control.Monad
import Control.Monad.Trans
import System.IO.Unsafe
import qualified Data.HashSet                                   as Set

import Foreign                                                  ( Ptr, Storable )
import qualified Foreign                                        as F
import qualified Foreign.CUDA.Driver                            as CUDA

#include "accelerate.h"


-- Array expression evaluation
-- ---------------------------

-- Computations are evaluated by traversing the AST bottom-up, and for each node
-- distinguishing between three cases:
--
-- 1. If it is a Use node, return a reference to the device memory holding the
--    array data
--
-- 2. If it is a non-skeleton node, such as a let-binding or shape conversion,
--    this is executed directly by updating the environment or similar
--
-- 3. If it is a skeleton node, the associated binary object is retrieved,
--    memory allocated for the result, and the kernel(s) that implement the
--    skeleton are invoked
--

-- Evaluate a closed array expression
--
executeAcc :: Arrays a => ExecAcc a -> CIO a
executeAcc acc = executeOpenAcc acc Empty

-- Evaluate an expression with free array variables
--
executeAfun1 :: (Arrays a, Arrays b) => ExecAfun (a -> b) -> a -> CIO b
executeAfun1 (Alam (Abody f)) arrs = do
  applyArraysR useArray arrays arrs
  executeOpenAcc f (Empty `Push` arrs)

executeAfun1 _ _                   =
  error "the sword comes out after you swallow it, right?"


-- Evaluate an open array expression
--
executeOpenAcc :: ExecOpenAcc aenv a -> Val aenv -> CIO a
executeOpenAcc (ExecAcc kernelList@(FL _ kernel _) bindings acc) aenv =
  case acc of
    --
    -- (1) Array introduction
    --
    Use arr -> return arr

    --
    -- (2) Environment manipulation
    --
    Avar ix  -> return (prj ix aenv)

    Alet a b -> do
      a0 <- executeOpenAcc a aenv
      executeOpenAcc b (aenv `Push` a0)

    Alet2 a b -> do
      (a1, a0) <- executeOpenAcc a aenv
      executeOpenAcc b (aenv `Push` a1 `Push` a0)

    PairArrays a b ->
      (,) <$> executeOpenAcc a aenv
          <*> executeOpenAcc b aenv

    Apply (Alam (Abody f)) a -> do
      a0 <- executeOpenAcc a aenv
      executeOpenAcc f (Empty `Push` a0)
    Apply _ _   -> error "Awww... the sky is crying"

    Acond p t e -> do
      cond <- executeExp p aenv
      if cond then executeOpenAcc t aenv
              else executeOpenAcc e aenv

    Reshape e a -> do
      ix <- executeExp e aenv
      a0 <- executeOpenAcc a aenv
      reshapeOp ix a0

    Unit e ->
      unitOp =<< executeExp e aenv

    --
    -- (3) Array computations
    --
    Generate e _        ->
      generateOp kernel bindings aenv =<< executeExp e aenv

    Replicate sliceIndex e a -> do
      slix <- executeExp e aenv
      a0   <- executeOpenAcc a aenv
      replicateOp kernel bindings aenv sliceIndex slix a0

    Index sliceIndex a e -> do
      slix <- executeExp e aenv
      a0   <- executeOpenAcc a aenv
      indexOp kernel bindings aenv sliceIndex a0 slix

    Map _ a             -> do
      a0 <- executeOpenAcc a aenv
      mapOp kernel bindings aenv a0

    ZipWith _ a b       -> do
      a1 <- executeOpenAcc a aenv
      a0 <- executeOpenAcc b aenv
      zipWithOp kernel bindings aenv a1 a0

    Fold _ _ a          -> do
      a0 <- executeOpenAcc a aenv
      foldOp kernel bindings aenv a0

    Fold1 _ a           -> do
      a0 <- executeOpenAcc a aenv
      fold1Op kernel bindings aenv a0

    FoldSeg _ _ a s     -> do
      a0 <- executeOpenAcc a aenv
      s0 <- executeOpenAcc s aenv
      foldSegOp kernel bindings aenv a0 s0

    Fold1Seg _ a s      -> do
      a0 <- executeOpenAcc a aenv
      s0 <- executeOpenAcc s aenv
      fold1SegOp kernel bindings aenv a0 s0

    Scanl _ _ a         -> do
      a0 <- executeOpenAcc a aenv
      scanOp L kernelList bindings aenv a0

    Scanl' _ _ a        -> do
      a0 <- executeOpenAcc a aenv
      scan'Op kernelList bindings aenv a0

    Scanl1 _ a          -> do
      a0 <- executeOpenAcc a aenv
      scan1Op kernelList bindings aenv a0

    Scanr _ _ a         -> do
      a0 <- executeOpenAcc a aenv
      scanOp R kernelList bindings aenv a0

    Scanr' _ _ a        -> do
      a0 <- executeOpenAcc a aenv
      scan'Op kernelList bindings aenv a0

    Scanr1 _ a          -> do
      a0 <- executeOpenAcc a aenv
      scan1Op kernelList bindings aenv a0

    Permute _ a _ b     -> do
      a0 <- executeOpenAcc a aenv
      a1 <- executeOpenAcc b aenv
      permuteOp kernel bindings aenv a0 a1

    Backpermute e _ a   -> do
      sh <- executeExp e aenv
      a0 <- executeOpenAcc a aenv
      backpermuteOp kernel bindings aenv sh a0

    Stencil _ _ a       -> do
      a0 <- executeOpenAcc a aenv
      stencilOp kernel bindings aenv a0

    Stencil2 _ _ a _ b  -> do
      a1 <- executeOpenAcc a aenv
      a0 <- executeOpenAcc b aenv
      stencil2Op kernel bindings aenv a1 a0


-- Implementation of primitive array operations
-- --------------------------------------------

reshapeOp
    :: Shape dim
    => dim
    -> Array dim' e
    -> CIO (Array dim e)
reshapeOp newShape (Array oldShape adata)
  = BOUNDS_CHECK(check) "reshape" "shape mismatch" (Sugar.size newShape == size oldShape)
  $ return $ Array (fromElt newShape) adata


unitOp
    :: Elt e
    => e
    -> CIO (Scalar e)
unitOp v = newArray Z (const v)


generateOp
    :: (Shape dim, Elt e)
    => AccKernel (Array dim e)
    -> AccBindings aenv
    -> Val aenv
    -> dim
    -> CIO (Array dim e)
generateOp kernel bindings aenv sh = do
  res@(Array _ out) <- allocateArray sh
  execute kernel bindings aenv (Sugar.size sh)
    (((), out)
        , sh)
  return res


replicateOp
    :: forall aenv e dim sl co slix. (Shape dim, Elt slix)
    => AccKernel (Array dim e)
    -> AccBindings aenv
    -> Val aenv
    -> SliceIndex (EltRepr slix) (EltRepr sl) co (EltRepr dim)
    -> slix
    -> Array sl e
    -> CIO (Array dim e)
replicateOp kernel bindings aenv sliceIndex slix (Array sh0 in0) = do
  let sh                = toElt $ extend sliceIndex (fromElt slix) sh0
      sl                = toElt sh0 :: sl
  res@(Array _ out)     <- allocateArray sh
  execute kernel bindings aenv (Sugar.size sh)
    (((((), out)
          , in0)
          , sl)
          , sh)
  return res
  where
    extend :: SliceIndex slix' sl' co' dim' -> slix' -> sl' -> dim'
    extend (SliceNil)            ()       ()      = ()
    extend (SliceAll sliceIdx)   (slx,()) (sl,sz) = (extend sliceIdx slx sl, sz)
    extend (SliceFixed sliceIdx) (slx,sz) sl      = (extend sliceIdx slx sl, sz)


indexOp
    :: forall sl co slix aenv dim e. (Shape sl, Elt slix)
    => AccKernel (Array sl e)
    -> AccBindings aenv
    -> Val aenv
    -> SliceIndex (EltRepr slix) (EltRepr sl) co (EltRepr dim)
    -> Array dim e
    -> slix
    -> CIO (Array sl e)
indexOp kernel bindings aenv sliceIndex (Array sh0 in0) slix = do
  let sz                = toElt sh0                                                 :: dim
      sh                = toElt $ restrict sliceIndex (fromElt slix) sh0            :: sl
      sl                = Sugar.listToShape $ convertSlix sliceIndex (fromElt slix) :: sl
  res@(Array _ out)     <- allocateArray sh
  execute kernel bindings aenv (Sugar.size sh)
    ((((((), out)
           , in0)
           , sh)
           , sl)
           , sz)
  return res
  where
    restrict :: SliceIndex slix' sl' co' dim' -> slix' -> dim' -> sl'
    restrict (SliceNil)            ()       ()      = ()
    restrict (SliceAll sliceIdx)   (slx,()) (sh,sz) = (restrict sliceIdx slx sh, sz)
    restrict (SliceFixed sliceIdx) (slx,i)  (sh,sz)
      = BOUNDS_CHECK(checkIndex) "slice" i sz $ restrict sliceIdx slx sh
    --
    convertSlix :: SliceIndex slix' sl' co' dim' -> slix' -> [Int]
    convertSlix (SliceNil)            ()     = []
    convertSlix (SliceAll   sliceIdx) (s,()) = convertSlix sliceIdx s
    convertSlix (SliceFixed sliceIdx) (s,i)  = i : convertSlix sliceIdx s


mapOp
    :: Elt e
    => AccKernel (Array dim e)
    -> AccBindings aenv
    -> Val aenv
    -> Array dim e'
    -> CIO (Array dim e)
mapOp kernel bindings aenv (Array sh0 in0) = do
  res@(Array _ out) <- allocateArray (toElt sh0)
  execute kernel bindings aenv (size sh0)
    ((((), out)
         , in0)
         , convertIx (size sh0))
  return res

zipWithOp
    :: forall aenv dim a b c. Elt c
    => AccKernel (Array dim c)
    -> AccBindings aenv
    -> Val aenv
    -> Array dim a
    -> Array dim b
    -> CIO (Array dim c)
zipWithOp kernel bindings aenv (Array sh1 in1) (Array sh0 in0) = do
  res@(Array sh out) <- allocateArray $ toElt (sh1 `intersect` sh0)
  execute kernel bindings aenv (size sh)
    (((((((), out)
            , in1)
            , in0)
            , toElt sh  :: dim)
            , toElt sh1 :: dim)
            , toElt sh0 :: dim)
  return res

foldOp, fold1Op
    :: forall dim e aenv. Shape dim
    => AccKernel (Array dim e)
    -> AccBindings aenv
    -> Val aenv
    -> Array (dim:.Int) e
    -> CIO (Array dim e)
fold1Op kernel bindings aenv in0@(Array (_,sz) _)
  = BOUNDS_CHECK(check) "fold1" "empty array" (sz > 0)
  $ foldOp kernel bindings aenv in0

foldOp kernel bindings aenv (Array sh0 in0)
  -- A recursive multi-block reduction when collapsing to a single value
  --
  | dim sh0 == 1 = do
      let numElements           = size sh0
          (_,numBlocks,_)       = configure kernel (size sh0)
      res@(Array _ out)         <- allocateArray (toElt (fst sh0,numBlocks)) :: CIO (Array (dim:.Int) e)
      execute kernel bindings aenv numElements
        ((((), out)
             , in0)
             , convertIx numElements)
      if numBlocks > 1 then foldOp kernel bindings aenv res
                       else return (Array (fst sh0) out)
  --
  -- Reduction over the innermost dimension of an array (single pass operation)
  --
  | otherwise    = do
      let (sh, sz)              = sh0
          interval_size         = sz
          num_intervals         = size sh
          num_elements          = size sh0
      res@(Array _ out)         <- allocateArray $ toElt sh
      execute kernel bindings aenv (num_intervals `max` 1)
        ((((((), out)
               , in0)
               , convertIx interval_size)
               , convertIx num_intervals)
               , convertIx num_elements)
      return res

foldSegOp, fold1SegOp
    :: forall aenv dim e i. Shape dim
    => AccKernel (Array (dim:.Int) e)
    -> AccBindings aenv
    -> Val aenv
    -> Array (dim:.Int) e
    -> Segments i
    -> CIO (Array (dim:.Int) e)
fold1SegOp kernel bindings aenv in0 seg =
  foldSegOp kernel bindings aenv in0 seg

foldSegOp kernel bindings aenv (Array sh0 in0) (Array shs seg) = do
  res@(Array sh out) <- allocateArray $ toElt (fst sh0, size shs-1)
  --
  message $ "foldSeg: shOut = (" ++ showShape (toElt sh  :: dim :. Int) ++ ")"
                ++ ", shIn0 = (" ++ showShape (toElt sh0 :: dim :. Int) ++ ")"
  execute kernel bindings aenv (size sh)
    ((((((), out)
           , in0)
           , seg)
           , toElt sh  :: dim :. Int)
           , toElt sh0 :: dim :. Int)
  return res


data ScanDirection = L | R

scanOp
    :: forall aenv e. Elt e
    => ScanDirection
    -> FullList () (AccKernel (Vector e))
    -> AccBindings aenv
    -> Val aenv
    -> Vector e
    -> CIO (Vector e)
scanOp dir (FL _ kfold1' (Cons _ kscan Nil)) bindings aenv (Array sh0 in0) = do
  let (_,num_intervals,_)       =  configure kscan num_elements
  a_out@(Array _ out)           <- allocateArray (Z :. num_elements + 1)
  (Array _ blk)                 <- allocateArray (Z :. num_intervals) :: CIO (Vector e)
  d_out                         <- devicePtrsOfArrayData out
  --
  -- depending on whether we are a left or right scan, we need to manipulate the
  -- pointers that specify the final element and main scan body
  --
  let interval_size             = (num_elements + num_intervals - 1) `div` num_intervals
      body                      = marshalDevicePtrs out d_body
      sum                       = marshalDevicePtrs out d_sum
      (d_body, d_sum)           =
        case dir of
          L -> (d_out, advancePtrsOfArrayData out num_elements d_out)
          R -> (advancePtrsOfArrayData out 1 d_out, d_out)
  --
  -- If the array is sized such that there is only a single interval, the first
  -- phase of calculating a per-interval carry-in value can be skipped
  --
  when (num_intervals > 1) $ do
    -- Compute the interval sum. Since we use associative operations, this can
    -- be done as a reduction instead of requiring a full left-/right-scan.
    message $ "scan phase 1: interval_size = " ++ shows interval_size
                ", num_intervals = " ++ shows num_intervals
                ", num_elements = " ++ show num_elements
    execute kfold1 bindings aenv num_elements
      ((((((), blk)
             , in0)
             , convertIx interval_size)
             , convertIx num_intervals)
             , convertIx num_elements)

    --
    -- Inclusive scan of the per-interval results to compute each segment's
    -- carry-in value.
    execute kscan bindings aenv 1
      (((((((), blk)
              , sum)
              , blk)
              , blk)    -- not used, just need the right number of arguments
              , convertIx num_intervals)
              , convertIx num_intervals)

  --
  -- Prefix-sum of the input array, using interval carry-in values.
  message $ "scan phase 2: interval_size = " ++ shows interval_size
              ", num_elements = " ++ show num_elements
  execute kscan bindings aenv num_elements
    (((((((), body)
            , sum)
            , in0)
            , blk)
            , convertIx interval_size)
            , convertIx num_elements)
  return a_out
  where
    num_elements                = size sh0
    kfold1                      = retag kfold1' :: AccKernel (Vector e)
--    kscan1                      = retag kscan1' :: AccKernel (Vector e)

scanOp _ _ _ _ _ = error "I'll just pretend to hug you until you get here."


scan'Op
    :: forall aenv e. Elt e
    => FullList () (AccKernel (Vector e, Scalar e))
    -> AccBindings aenv
    -> Val aenv
    -> Vector e
    -> CIO (Vector e, Scalar e)
scan'Op (FL _ kfold1' (Cons _ kscan Nil)) bindings aenv (Array sh0 in0) = do
  let (_,num_intervals,_)       =  configure kscan num_elements
  (Array _ blk)                 <- allocateArray (Z :. num_intervals) :: CIO (Vector e)
  a_out@(Array _ out)           <- allocateArray (Z :. num_elements)
  a_sum@(Array _ sum)           <- allocateArray Z
  let interval_size             = (num_elements + num_intervals - 1) `div` num_intervals
  --
  -- see comments in 'scanOp'
  when (num_intervals > 1) $ do
    message $ "scan phase 1: interval_size = " ++ shows interval_size
                ", num_intervals = " ++ shows num_intervals
                ", num_elements = " ++ show num_elements
    execute kfold1 bindings aenv num_elements
      ((((((), blk)
             , in0)
             , convertIx interval_size)
             , convertIx num_intervals)
             , convertIx num_elements)
    --
    execute kscan bindings aenv 1
      (((((((), blk)
              , sum)
              , blk)
              , blk)    -- not used
              , convertIx num_intervals)
              , convertIx num_intervals)
  --
  message $ "scan phase 2: interval_size = " ++ shows interval_size
              ", num_elements = " ++ show num_elements
  execute kscan bindings aenv num_elements
    (((((((), out)
            , sum)
            , in0)
            , blk)
            , convertIx interval_size)
            , convertIx num_elements)
  return (a_out, a_sum)
  where
    num_elements        = size sh0
    kfold1              = retag kfold1' :: AccKernel (Vector e)

scan'Op _ _ _ _ = error "If I promise not to kill you, can I have a hug?"


scan1Op
    :: forall aenv e. Elt e
    => FullList () (AccKernel (Vector e))
    -> AccBindings aenv
    -> Val aenv
    -> Vector e
    -> CIO (Vector e)
scan1Op (FL _ kfold1' (Cons _ kscan1 Nil)) bindings aenv (Array sh0 in0) = do
  let (_,num_intervals,_)       =  configure kscan1 num_elements
  (Array _ sum)                 <- allocateArray Z                      :: CIO (Scalar e)
  (Array _ blk)                 <- allocateArray (Z :. num_intervals)   :: CIO (Vector e)
  a_out@(Array _ out)           <- allocateArray (Z :. num_elements)
  let interval_size             = (num_elements + num_intervals - 1) `div` num_intervals
  --
  -- see comments in 'scanOp'
  when (num_intervals > 1) $ do
    message $ "scan phase 1: interval_size = " ++ shows interval_size
                ", num_intervals = " ++ shows num_intervals
                ", num_elements = " ++ show num_elements
    execute kfold1 bindings aenv num_elements
      ((((((), blk)
             , in0)
             , convertIx interval_size)
             , convertIx num_intervals)
             , convertIx num_elements)
    --
    execute kscan1 bindings aenv 1
      (((((((), blk)
              , sum)
              , blk)
              , blk)   -- not used
              , convertIx num_intervals)
              , convertIx num_intervals)
  --
  message $ "scan phase 2: interval_size = " ++ shows interval_size
              ", num_elements = " ++ show num_elements
  execute kscan1 bindings aenv num_elements
    (((((((), out)
            , sum)
            , in0)
            , blk)
            , convertIx interval_size)
            , convertIx num_elements)
  return a_out
  where
    num_elements        = size sh0
    kfold1              = retag kfold1' :: AccKernel (Vector e)

scan1Op _ _ _ _ = error "If you get wet, you'll get sick."


permuteOp
    :: forall aenv dim dim' e. Elt e
    => AccKernel (Array dim' e)
    -> AccBindings aenv
    -> Val aenv
    -> Array dim' e             -- default values
    -> Array dim e              -- permuted array
    -> CIO (Array dim' e)
permuteOp kernel bindings aenv in0@(Array sh0 _) (Array sh1 in1) = do
  res@(Array _ out) <- allocateArray (toElt sh0)
  copyArray in0 res
  execute kernel bindings aenv (size sh0)
    (((((), out)
          , in1)
          , toElt sh0 :: dim')
          , toElt sh1 :: dim)
  return res

backpermuteOp
    :: forall aenv dim dim' e. (Shape dim', Elt e)
    => AccKernel (Array dim' e)
    -> AccBindings aenv
    -> Val aenv
    -> dim'
    -> Array dim e
    -> CIO (Array dim' e)
backpermuteOp kernel bindings aenv dim' (Array sh0 in0) = do
  res@(Array sh out) <- allocateArray dim'
  execute kernel bindings aenv (size sh)
    (((((), out)
          , in0)
          , toElt sh  :: dim')
          , toElt sh0 :: dim)
  return res

stencilOp
    :: forall aenv dim a b. Elt b
    => AccKernel (Array dim b)
    -> AccBindings aenv
    -> Val aenv
    -> Array dim a
    -> CIO (Array dim b)
stencilOp kernel@(Kernel _ mdl _ _ _) bindings aenv in0@(Array sh0 _) = do
  res@(Array _ out)     <- allocateArray (toElt sh0)
  bindStencil 0 mdl in0
  execute kernel bindings aenv (size sh0)
    (((), out)
        , toElt sh0 :: dim)
  return res

stencil2Op
    :: forall aenv dim a b c. Elt c
    => AccKernel (Array dim c)
    -> AccBindings aenv
    -> Val aenv
    -> Array dim a
    -> Array dim b
    -> CIO (Array dim c)
stencil2Op kernel@(Kernel _ mdl _ _ _) bindings aenv in0@(Array sh0 _) in1@(Array sh1 _) = do
  res@(Array sh out)    <- allocateArray $ toElt (sh1 `intersect` sh0)
  bindStencil 1 mdl in1
  bindStencil 0 mdl in0
  execute kernel bindings aenv (size sh)
    (((((), out)
          , toElt sh  :: dim)
          , toElt sh1 :: dim)
          , toElt sh0 :: dim)
  return res


-- Expression evaluation
-- ---------------------

-- Evaluate an open expression
--
executeOpenExp :: PreOpenExp ExecOpenAcc env aenv t -> Val env -> Val aenv -> CIO t
executeOpenExp exp env aenv = do
  case exp of
    -- Local binders and variable indices, ranging over tuples and scalars
    Var ix              -> return $! prj ix env
    Let x e             -> do
      x'                <- executeOpenExp x env aenv
      executeOpenExp e (env `Push` x') aenv

    -- Constant values
    Const c             -> return $! toElt c
    PrimConst c         -> return $! I.evalPrimConst c

    -- Primitive scalar operations
    PrimApp fun arg     -> do
      x                 <- executeOpenExp arg env aenv
      return            $! I.evalPrim fun x

    -- Tuples
    Tuple tup           -> do
      t                 <- executeTuple tup env aenv
      return            $! toTuple t

    Prj ix e            -> do
      t                 <- executeOpenExp e env aenv
      return            $! I.evalPrj ix (fromTuple t)

    -- Conditional expression
    Cond p t e          -> do
      p'                <- executeOpenExp p env aenv
      case p' of
        True            -> executeOpenExp t env aenv
        False           -> executeOpenExp e env aenv

    -- Array indices and shapes
    IndexAny            -> return Sugar.Any
    IndexNil            -> return Z
    IndexCons sh sz     -> do
      sh'               <- executeOpenExp sh env aenv
      sz'               <- executeOpenExp sz env aenv
      return            $! sh' :. sz'

    IndexHead sh        -> do
      (_ :. ix)         <- executeOpenExp sh env aenv
      return            $! ix

    IndexTail sh        -> do
      (ix :. _)         <- executeOpenExp sh env aenv
      return            $! ix

    -- Array shape and element indexing
    IndexScalar acc ix  -> do
      arr'              <- executeOpenAcc acc aenv
      ix'               <- executeOpenExp ix env aenv
      indexArray arr' ix'

    Shape acc           -> do
      (Array sh _)      <- executeOpenAcc acc aenv
      return            $! toElt sh

    ShapeSize e         -> do
      sh                <- executeOpenExp e env aenv
      return            $! size (fromElt sh)


-- Evaluate a closed expression
--
executeExp :: PreExp ExecOpenAcc aenv t -> Val aenv -> CIO t
executeExp e = executeOpenExp e Empty


-- Tuple evaluation
--
executeTuple :: Tuple (PreOpenExp ExecOpenAcc env aenv) t -> Val env -> Val aenv -> CIO t
executeTuple NilTup          _   _    = return ()
executeTuple (t `SnocTup` e) env aenv = (,) <$> executeTuple   t env aenv
                                            <*> executeOpenExp e env aenv


-- Array references in scalar code
-- -------------------------------

bindLifted :: CUDA.Module -> Val aenv -> AccBindings aenv -> CIO ()
bindLifted mdl aenv (AccBindings vars) = mapM_ (bindAcc mdl aenv) (Set.toList vars)


bindAcc
    :: CUDA.Module
    -> Val aenv
    -> ArrayVar aenv
    -> CIO ()
bindAcc mdl aenv (ArrayVar idx) =
  let idx'        = show $ idxToInt idx
      Array sh ad = prj idx aenv
      --
      bindDim = liftIO $
        CUDA.getPtr mdl ("sh" ++ idx') >>=
        CUDA.pokeListArray (convertSh sh) . fst
      --
      arr n   = "arr" ++ idx' ++ "_a" ++ show (n::Int)
      tex     = CUDA.getTex mdl . arr
      bindTex = marshalTextureData ad (size sh) =<< liftIO (sequence' $ map tex [0..])
  in
  bindDim >> bindTex


bindStencil
    :: Int
    -> CUDA.Module
    -> Array dim e
    -> CIO ()
bindStencil s mdl (Array sh ad) =
  let sten n = "stencil" ++ show s ++ "_a" ++ show (n::Int)
      tex    = CUDA.getTex mdl . sten
  in
  marshalTextureData ad (size sh) =<< liftIO (sequence' $ map tex [0..])


-- Kernel execution
-- ----------------

-- Data which can be marshalled as arguments to a kernel invocation.
--
class Marshalable a where
  marshal :: a -> CIO [CUDA.FunParam]

instance Marshalable () where
  marshal _ = return []

#define primMarshalable(ty)                                                    \
instance Marshalable (ty) where {                                              \
  marshal x = return [CUDA.VArg x] }

primMarshalable(Int)
primMarshalable(Int8)
primMarshalable(Int16)
primMarshalable(Int32)
primMarshalable(Int64)
primMarshalable(Word)
primMarshalable(Word8)
primMarshalable(Word16)
primMarshalable(Word32)
primMarshalable(Word64)
primMarshalable(Float)
primMarshalable(Double)
primMarshalable(Ptr a)
primMarshalable(CUDA.DevicePtr a)

instance Marshalable CUDA.FunParam where
  marshal x = return [x]

instance AD.ArrayElt e => Marshalable (AD.ArrayData e) where
  marshal = marshalArrayData

instance Marshalable a => Marshalable [a] where
  marshal = concatMapM marshal

instance (Marshalable a, Marshalable b) => Marshalable (a,b) where
  marshal (a,b) = (++) <$> marshal a <*> marshal b


-- This requires incoherent instances \=
--
instance Shape sh => Storable sh where
  sizeOf sh     = F.sizeOf    (undefined::Int32) * (1 `max` Sugar.dim sh)
  alignment _   = F.alignment (undefined::Int32)
  poke p sh     = F.pokeArray (F.castPtr p) (convertSh (fromElt sh))

instance Shape sh => Marshalable sh where
  marshal sh = return [CUDA.VArg sh]


-- What launch parameters should we use to execute the kernel with a number of
-- array elements?
--
configure :: AccKernel a -> Int -> (Int, Int, Int)
configure (Kernel _ !_ !_ !_ !launchConfig) !n = launchConfig n


-- Link the binary object implementing the computation, configure the kernel
-- launch parameters, and initiate the computation. This also handles lifting
-- and binding of array references from scalar expressions.
--
execute :: Marshalable args
        => AccKernel a                  -- The binary module implementing this kernel
        -> AccBindings aenv             -- Array variables embedded in scalar expressions
        -> Val aenv
        -> Int
        -> args
        -> CIO ()
execute kernel@(Kernel _ !mdl !_ !_ !_) !bindings !aenv !n !args = do
  bindLifted mdl aenv bindings
  launch kernel (configure kernel n) args


-- Execute a device function, with the given thread configuration and function
-- parameters. The tuple contains (threads per block, grid size, shared memory)
--
launch :: Marshalable args => AccKernel a -> (Int,Int,Int) -> args -> CIO ()
launch (Kernel entry _ !fn _ _) !(cta, grid, smem) !a = do
  message $ entry ++ " <<< " ++ shows grid ", " ++ shows cta ", " ++ shows smem " >>>"
  --
  args  <- marshal a
  liftIO $ CUDA.launchKernel fn (grid,1,1) (cta,1,1) smem Nothing args


-- Auxiliary functions
-- -------------------

-- Generalise concatMap for teh monadz
--
concatMapM :: Monad m => (a -> m [b]) -> [a] -> m [b]
concatMapM f xs = concat `liftM` mapM f xs

-- A lazier version of 'Control.Monad.sequence'
--
sequence' :: [IO a] -> IO [a]
sequence' = foldr k (return [])
  where k m ms = do { x <- m; xs <- unsafeInterleaveIO ms; return (x:xs) }

-- Extract shape dimensions as a list of integers. Singleton dimensions are
-- considered to be of unit size.
--
-- Internally, Accelerate uses snoc-based tuple projection, while the data
-- itself is stored in reading order. Ensure we match the behaviour of regular
-- tuples and code generation thereof.
--
convertSh :: R.Shape sh => sh -> [Int32]
convertSh = post . shapeToList
  where
    post [] = [1]
    post xs = reverse (map convertIx xs)

convertIx :: Int -> Int32
convertIx ix   = INTERNAL_ASSERT "convertIx" (ix <= intmax) (fromIntegral ix)
  where intmax = fromIntegral (maxBound :: Int32)


-- Apply a function to all components of an Arrays structure
--
applyArraysR
    :: (forall sh e. (Shape sh, Elt e) => Array sh e -> CIO ())
    -> ArraysR arrs
    -> arrs
    -> CIO ()
applyArraysR _ ArraysRunit         ()       = return ()
applyArraysR f (ArraysRpair r1 r0) (a1, a0) = applyArraysR f r1 a1 >> applyArraysR f r0 a0
applyArraysR f ArraysRarray        arr      = f arr


-- Debug
-- -----

{-# INLINE trace #-}
trace :: String -> CIO a -> CIO a
trace msg next = D.message D.dump_exec ("exec: " ++ msg) >> next

{-# INLINE message #-}
message :: String -> CIO ()
message s = s `trace` return ()

