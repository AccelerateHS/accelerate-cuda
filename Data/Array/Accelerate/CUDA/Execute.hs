{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE CPP                        #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GADTs                      #-}
{-# LANGUAGE IncoherentInstances        #-}
{-# LANGUAGE MultiParamTypeClasses      #-}
{-# LANGUAGE NoForeignFunctionInterface #-}
{-# LANGUAGE PatternGuards              #-}
{-# LANGUAGE ScopedTypeVariables        #-}
{-# LANGUAGE TemplateHaskell            #-}
{-# LANGUAGE TupleSections              #-}
{-# LANGUAGE TypeOperators              #-}
{-# LANGUAGE TypeSynonymInstances       #-}
{-# LANGUAGE UndecidableInstances       #-}
{-# LANGUAGE ScopedTypeVariables        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Execute
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Execute (

  -- * Execute a computation under a CUDA environment
  executeAcc, executeAfun1,

  -- * Executing a sequence computation and streaming its output.
  StreamSeq(..), streamSeq,

) where

-- friends
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Array.Data
import Data.Array.Accelerate.CUDA.Array.Slice
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Foreign.Import                ( canExecuteAcc )
import Data.Array.Accelerate.CUDA.CodeGen.Base                  ( Name, namesOfArray, groupOfInt )
import Data.Array.Accelerate.CUDA.Execute.Event                 ( Event )
import Data.Array.Accelerate.CUDA.Execute.Stream                ( Stream )
import qualified Data.Array.Accelerate.CUDA.Array.Prim          as Prim
import qualified Data.Array.Accelerate.CUDA.Debug               as D
import qualified Data.Array.Accelerate.CUDA.Execute.Event       as Event
import qualified Data.Array.Accelerate.CUDA.Execute.Stream      as Stream

import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Interpreter                        ( evalPrim, evalPrimConst, evalPrj )
import Data.Array.Accelerate.Array.Data                         ( ArrayElt, ArrayData )
import Data.Array.Accelerate.Array.Lifted
import Data.Array.Accelerate.Array.Representation               ( SliceIndex(..) )
import Data.Array.Accelerate.Analysis.Shape
import Data.Array.Accelerate.FullList                           ( FullList(..), List(..) )
import Data.Array.Accelerate.Lifetime                           ( withLifetime )
import Data.Array.Accelerate.Trafo                              ( Extend(..), DelayedOpenAcc )
import qualified Data.Array.Accelerate.Array.Representation     as R


-- standard library
import Control.Applicative                                      hiding ( Const )
import Control.Monad                                            ( join, when, liftM )
import Control.Monad.Reader                                     ( asks )
import Control.Monad.State                                      ( gets )
import Control.Monad.Trans                                      ( MonadIO, liftIO, lift )
import Control.Monad.Trans.Cont                                 ( ContT(..) )
import Control.Monad.Trans.Maybe                                ( MaybeT(..), runMaybeT )
import System.IO.Unsafe                                         ( unsafeInterleaveIO, unsafePerformIO )
import Data.Int
import Data.Monoid                                              ( mempty )
import Data.Word
import Prelude                                                  hiding ( exp, sum, iterate )

import Foreign.CUDA.Analysis.Device                             ( computeCapability, Compute(..) )
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Data.HashMap.Strict                            as Map


-- Asynchronous kernel execution
-- -----------------------------

-- Arrays with an associated CUDA Event that will be signalled once the
-- computation has completed.
--
data Async a = Async {-# UNPACK #-} !Event !a

-- Valuation for an environment of asynchronous array computations
--
data Aval env where
  Aempty :: Aval ()
  Apush  :: Aval env -> Async t -> Aval (env, t)
  ApushNoSync  :: Aval env -> t -> Aval (env, t)

-- A suspended sequence computation.
newtype StreamSeq a = StreamSeq (CIO (Maybe ([a], StreamSeq a)))

-- Projection of a value from a valuation using a de Bruijn index.
--
aprj :: Idx env t -> Aval env -> Either t (Async t)
aprj ZeroIdx       (Apush _         x) = Right x
aprj ZeroIdx       (ApushNoSync _   x) = Left  x
aprj (SuccIdx idx) (Apush val _)       = aprj idx val
aprj (SuccIdx idx) (ApushNoSync val _) = aprj idx val
aprj _             _             = $internalError "aprj" "inconsistent valuation"


-- All work submitted to the given stream will occur after the asynchronous
-- event for the given array has been fulfilled. Synchronisation is performed
-- efficiently on the device. This function returns immediately.
--
after :: MonadIO m => Stream -> Either a (Async a) -> m a
after stream (Right (Async event arr)) = liftIO $ Event.after event stream >> return arr
after _      (Left a)                  = return a


-- Block the calling thread until the event for the given array computation
-- is recorded.
--
wait :: MonadIO m => Async a -> m a
wait (Async e x) = liftIO $ Event.block e >> return x


-- Execute the given computation in a unique execution stream.
--
streaming :: (Stream -> CIO a) -> (Async a -> CIO b) -> CIO b
streaming first second = do
  context   <- asks activeContext
  reservoir <- gets streamReservoir
  table     <- gets eventTable
  Stream.streaming context reservoir table first (\e a -> second (Async e a))


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

executeAcc :: Arrays a => ExecAcc a -> CIO a
executeAcc !acc = streaming (executeOpenAcc True acc Aempty) wait

executeAfun1 :: (Arrays a, Arrays b) => ExecAfun (a -> b) -> a -> CIO b
executeAfun1 !afun !arrs = do
  streaming (useArrays (arrays arrs) (fromArr arrs))
            (\(Async event ()) -> executeOpenAfun1 afun Aempty (Async event arrs))
  where
    useArrays :: ArraysR arrs -> arrs -> Stream -> CIO ()
    useArrays ArraysRunit         ()       _  = return ()
    useArrays (ArraysRpair r1 r0) (a1, a0) st = useArrays r1 a1 st >> useArrays r0 a0 st
    useArrays ArraysRarray        arr      st = useArrayAsync arr (Just st)


executeOpenAfun1 :: PreOpenAfun ExecOpenAcc aenv (a -> b) -> Aval aenv -> Async a -> CIO b
executeOpenAfun1 (Alam (Abody f)) aenv x = streaming (executeOpenAcc True f (aenv `Apush` x)) wait
executeOpenAfun1 _                _    _ = error "the sword comes out after you swallow it, right?"

-- Evaluate an open array computation
--
executeOpenAcc
    :: forall aenv arrs.
       Bool -- Spawn new CUDA streams on let-bindings?
    -> ExecOpenAcc aenv arrs
    -> Aval aenv
    -> Stream
    -> CIO arrs
executeOpenAcc _ EmbedAcc{} _ _
  = $internalError "execute" "unexpected delayed array"
executeOpenAcc cudaStreams (ExecSeq !dsequ !sequ) !aenv !stream
  = do (pd, s) <- initialiseSeq defaultSeqConfig dsequ sequ aenv stream
       streaming (executeSequence s pd) wait
executeOpenAcc cudaStreams (ExecAcc (FL () kernel more) !gamma !pacc) !aenv !stream
  = case pacc of

      -- Array introduction
      Use arr                   -> return (toArr arr)
      Unit x                    -> newArray Z . const =<< travE x

      -- Environment manipulation
      Avar ix                   -> after stream (aprj ix aenv)
      Alet bnd body             ->
        if cudaStreams
           then streaming (executeOpenAcc cudaStreams bnd aenv) (\x -> executeOpenAcc cudaStreams body (aenv `Apush`       x) stream)
           else executeOpenAcc cudaStreams bnd aenv stream >>=  (\x -> executeOpenAcc cudaStreams body (aenv `ApushNoSync` x) stream)
      Apply f a                 -> streaming (executeOpenAcc cudaStreams a aenv)   (executeOpenAfun1 f aenv)
      Atuple tup                -> toAtuple <$> travT tup
      Aprj ix tup               -> evalPrj ix . fromAtuple <$> travA tup
      Acond p t e               -> travE p >>= \x -> if x then travA t else travA e
      Awhile p f a              -> awhile p f =<< travA a

      -- Foreign
      Aforeign ff afun a        -> aforeign ff afun =<< travA a

      -- Producers
      Map _ a                   -> executeOp =<< extent a
      Generate sh _             -> executeOp =<< travE sh
      Transform sh _ _ _        -> executeOp =<< travE sh
      Backpermute sh _ _        -> executeOp =<< travE sh
      Reshape sh a              -> reshapeOp <$> travE sh <*> travA a

      -- Consumers
      Fold _ _ a                -> foldOp  =<< extent a
      Fold1 _ a                 -> fold1Op =<< extent a
      FoldSeg _ _ a s           -> join $ foldSegOp <$> extent a <*> extent s
      Fold1Seg _ a s            -> join $ foldSegOp <$> extent a <*> extent s
      Scanl1 _ a                -> scan1Op =<< extent a
      Scanr1 _ a                -> scan1Op =<< extent a
      Scanl' _ _ a              -> scan'Op =<< extent a
      Scanr' _ _ a              -> scan'Op =<< extent a
      Scanl _ _ a               -> scanOp True  =<< extent a
      Scanr _ _ a               -> scanOp False =<< extent a
      Permute _ d _ a           -> join $ permuteOp <$> extent a <*> travA d
      Stencil _ _ a             -> stencilOp =<< travA a
      Stencil2 _ _ a1 _ a2      -> join $ stencil2Op <$> travA a1 <*> travA a2

      -- AST nodes that should be inaccessible at this point
      Replicate{}               -> fusionError
      Slice{}                   -> fusionError
      ZipWith{}                 -> fusionError
      Collect{}                 -> streamingError

  where
    fusionError    = $internalError "executeOpenAcc" "unexpected fusible matter"
    streamingError = $internalError "executeOpenAcc" "unexpected sequence computation"

    -- term traversals
    travA :: ExecOpenAcc aenv a -> CIO a
    travA !acc = executeOpenAcc cudaStreams acc aenv stream

    travE :: ExecExp aenv t -> CIO t
    travE !exp = executeExp exp aenv stream

    travT :: Atuple (ExecOpenAcc aenv) t -> CIO t
    travT NilAtup          = return ()
    travT (SnocAtup !t !a) = (,) <$> travT t <*> travA a

    awhile :: PreOpenAfun ExecOpenAcc aenv (a -> Scalar Bool) -> PreOpenAfun ExecOpenAcc aenv (a -> a) -> a -> CIO a
    awhile p f a = do
      tbl <- gets eventTable
      ctx <- asks activeContext
      nop <- liftIO $ Event.create ctx tbl      -- record event never call, so this is a functional no-op
      r   <- executeOpenAfun1 p aenv (Async nop a)
      ok  <- indexArray r 0                     -- TLM TODO: memory manager should remember what is already on the host
      if ok then awhile p f =<< executeOpenAfun1 f aenv (Async nop a)
            else return a

    aforeign :: (Arrays as, Arrays bs, Foreign f) => f as bs -> PreAfun ExecOpenAcc (as -> bs) -> as -> CIO bs
    aforeign ff pureFun a =
      case canExecuteAcc ff of
        Just cudaFun -> cudaFun stream a
        Nothing      -> executeAfun1 pureFun a

    -- get the extent of an embedded array
    extent :: Shape sh => ExecOpenAcc aenv (Array sh e) -> CIO sh
    extent ExecAcc{}     = $internalError "executeOpenAcc" "expected delayed array"
    extent ExecSeq{}     = $internalError "executeOpenAcc" "expected delayed array"
    extent (EmbedAcc sh) = travE sh

    -- Skeleton implementation
    -- -----------------------

    -- Execute a skeleton that has no special requirements: thread decomposition
    -- is based on the given shape.
    --
    executeOp :: (Shape sh, Elt e) => sh -> CIO (Array sh e)
    executeOp !sh = do
      out       <- allocateArray sh
      execute kernel gamma aenv (size sh) out stream
      return out

    -- Executing fold operations depend on whether we are recursively collapsing
    -- to a single value using multiple thread blocks, or a multidimensional
    -- single-pass reduction where there is one block per inner dimension.
    --
    fold1Op :: (Shape sh, Elt e) => (sh :. Int) -> CIO (Array sh e)
    fold1Op !sh@(_ :. sz)
      = $boundsCheck "fold1" "empty array" (sz > 0)
      $ foldCore sh

    foldOp :: (Shape sh, Elt e) => (sh :. Int) -> CIO (Array sh e)
    foldOp !(!sh :. sz)
      = foldCore ((listToShape . map (max 1) . shapeToList $ sh) :. sz)

    foldCore :: (Shape sh, Elt e) => (sh :. Int) -> CIO (Array sh e)
    foldCore !(!sh :. sz)
      | dim sh > 0              = executeOp sh
      | otherwise
      = let !numElements        = size sh * sz
            (_,!numBlocks,_)    = configure kernel numElements
        in do
          out   <- allocateArray (sh :. numBlocks)
          execute kernel gamma aenv numElements out stream
          foldRec out

    -- Recursive step(s) of a multi-block reduction
    --
    foldRec :: (Shape sh, Elt e) => Array (sh:.Int) e -> CIO (Array sh e)
    foldRec arr@(Array _ !adata)
      | Cons _ rec _ <- more
      = let sh :. sz            = shape arr
            !numElements        = size sh * sz
            (_,!numBlocks,_)    = configure rec numElements
        in if sz <= 1
              then return $ Array (fromElt sh) adata
              else do
                out     <- allocateArray (sh :. numBlocks)
                execute rec gamma aenv numElements (out, arr) stream
                foldRec out

      | otherwise
      = $internalError "foldRec" "missing phase-2 kernel module"

    -- Segmented reduction. Subtract one from the size of the segments vector as
    -- this is the result of an exclusive scan to calculate segment offsets.
    --
    foldSegOp :: (Shape sh, Elt e) => (sh :. Int) -> (Z :. Int) -> CIO (Array (sh :. Int) e)
    foldSegOp (!sh :. _) !(Z :. sz) = executeOp (sh :. sz - 1)

    -- Scans, all variations on a theme.
    --
    scanOp :: Elt e => Bool -> (Z :. Int) -> CIO (Vector e)
    scanOp !left !(Z :. numElements) = do
      arr@(Array _ adata)       <- allocateArray (Z :. numElements + 1)
      withDevicePtrs adata (Just stream) $ \out -> do
        let (!body, !sum)
              | left      = (out, advancePtrsOfArrayData adata numElements out)
              | otherwise = (advancePtrsOfArrayData adata 1 out, out)
        --
        scanCore numElements arr body sum
        return arr

    scan1Op :: forall e. Elt e => (Z :. Int) -> CIO (Vector e)
    scan1Op !(Z :. numElements) = do
      arr@(Array _ adata)       <- allocateArray (Z :. numElements + 1) :: CIO (Vector e)
      withDevicePtrs adata (Just stream) $ \body -> do
        let sum {- to fix type -} =  advancePtrsOfArrayData adata numElements body
        --
        scanCore numElements arr body sum
        return (Array ((),numElements) adata)

    scan'Op :: forall e. Elt e => (Z :. Int) -> CIO (Vector e, Scalar e)
    scan'Op !(Z :. numElements) = do
      vec@(Array _ ad_vec)      <- allocateArray (Z :. numElements) :: CIO (Vector e)
      sum@(Array _ ad_sum)      <- allocateArray Z                  :: CIO (Scalar e)
      withDevicePtrs ad_vec (Just stream) $ \d_vec ->
        withDevicePtrs ad_sum (Just stream) $ \d_sum -> do
          --
          scanCore numElements vec d_vec d_sum
          return (vec, sum)

    scanCore
        :: forall e. Elt e
        => Int
        -> Vector e                     -- to fix Elt vs. EltRepr
        -> Prim.DevicePtrs (EltRepr e)
        -> Prim.DevicePtrs (EltRepr e)
        -> CIO ()
    scanCore !numElements (Array _ !adata) !body !sum
      | Cons _ !upsweep1 (Cons _ !upsweep2 _) <- more
      = let (_,!numIntervals,_) = configure kernel numElements
            !d_body             = marshalDevicePtrs adata body
            !d_sum              = marshalDevicePtrs adata sum
        in do
          blk   <- allocateArray (Z :. numIntervals) :: CIO (Vector e)

          -- Phase 1: Split the array over multiple thread blocks and calculate
          --          the final scan result from each interval.
          --
          when (numIntervals > 1) $ do
            execute upsweep1 gamma aenv numElements blk stream
            execute upsweep2 gamma aenv numIntervals (blk, blk, d_sum) stream

          -- Phase 2: Re-scan the input using the carry-in value from each
          --          interval sum calculated in phase 1.
          --
          execute kernel gamma aenv numElements (numElements, d_body, blk, d_sum) stream

      | otherwise
      = $internalError "scanOp" "missing multi-block kernel module(s)"

    -- Forward permutation
    --
    permuteOp :: forall sh sh' e. (Shape sh, Shape sh', Elt e) => sh -> Array sh' e -> CIO (Array sh' e)
    permuteOp !sh !dfs = do
      let sh'   = shape dfs
          n'    = size sh'

      out               <- allocateArray sh'
      Array _ locks     <- allocateArray sh'            :: CIO (Array sh' Int32)
      withDevicePtrs locks (Just stream) $ \d_locks -> do

        liftIO $ CUDA.memsetAsync d_locks n' 0 (Just stream)      -- TLM: overlap these two operations?
        copyArrayAsync dfs out (Just stream)
        execute kernel gamma aenv (size sh) (out, d_locks) stream
        return out

    -- Stencil operations. NOTE: the arguments to 'namesOfArray' must be the
    -- same as those given in the function 'mkStencil[2]'.
    --
    stencilOp :: forall sh a b. (Shape sh, Elt a, Elt b) => Array sh a -> CIO (Array sh b)
    stencilOp !arr = do
      let sh    =  shape arr
      out       <- allocateArray sh
      dev       <- asks deviceProperties

      if computeCapability dev < Compute 2 0
         then marshalAccTex (namesOfArray "Stencil" (undefined :: a)) kernel arr (Just stream) $
                execute kernel gamma aenv (size sh) (out, sh) stream
         else execute kernel gamma aenv (size sh) (out, arr) stream
      execute kernel gamma aenv (size sh) (out, arr) stream
      --
      return out

    stencil2Op :: forall sh a b c. (Shape sh, Elt a, Elt b, Elt c)
               => Array sh a -> Array sh b -> CIO (Array sh c)
    stencil2Op !arr1 !arr2
      | Cons _ spec _ <- more
      = let sh1         =  shape arr1
            sh2         =  shape arr2
            (sh, op)
              | fromElt sh1 == fromElt sh2      = (sh1,                 spec)
              | otherwise                       = (sh1 `intersect` sh2, kernel)
        in do
          out   <- allocateArray sh
          dev   <- asks deviceProperties

          if computeCapability dev < Compute 2 0
             then marshalAccTex (namesOfArray "Stencil1" (undefined :: a)) op arr1 (Just stream) $
                  marshalAccTex (namesOfArray "Stencil2" (undefined :: b)) op arr2 (Just stream) $
                  execute op gamma aenv (size sh) (out, sh1,  sh2) stream
             else execute op gamma aenv (size sh) (out, arr1, arr2) stream
          execute op gamma aenv (size sh) (out, arr1, arr2) stream
          --
          return out

      | otherwise
      = $internalError "stencil2Op" "missing stencil specialisation kernel"


-- Change the shape of an array without altering its contents. This does not
-- execute any kernel programs.
--
reshapeOp :: Shape sh => sh -> Array sh' e -> Array sh e
reshapeOp sh (Array sh' adata)
  = $boundsCheck "reshape" "shape mismatch" (size sh == R.size sh')
    $ Array (fromElt sh) adata

-- Configuration for sequence evaluation.
--
data SeqConfig = SeqConfig
  { chunkSize :: !Int -- Allocation limit for a sequence in
                      -- words. Actual runtime allocation should be the
                      -- maximum of this size and the size of the
                      -- largest element in the sequence.
  }

-- Default sequence evaluation configuration for testing purposes.
--
-- Default sequence evaluation configuration for testing purposes.
--
defaultSeqConfig :: SeqConfig
defaultSeqConfig = SeqConfig { chunkSize = case unsafePerformIO (D.queryFlag D.chunk_size) of Nothing -> 128; Just n -> n }

-- An executable stream DAG for executing sequence expressions in a
-- streaming fashion.
--
data StreamDAG senv arrs where
  StreamProducer :: !(StreamProducer senv a) -> !(StreamDAG (senv, a) arrs) -> StreamDAG senv arrs
  StreamConsumer :: !(StreamConsumer senv a)                                -> StreamDAG senv a
  StreamReify    :: !(Val senv -> Int -> Stream -> CIO [a]) -> StreamDAG senv [a]

-- An executable producer.
--
data StreamProducer senv a where
  StreamStreamIn :: [a]
                 -> StreamProducer senv a

  StreamMap :: !(Val senv -> Stream -> CIO a)
            -> StreamProducer senv a

  StreamMapFin :: !(Int, Int)
               -> !(Val senv -> Int -> Stream -> CIO a)
               -> StreamProducer senv a

  -- Stream scan skeleton.
  StreamScan :: !(Val senv -> s -> Stream -> CIO (a, s)) -- Chunk scanner.
             -> !s                                       -- Accumulator (internal state).
             -> StreamProducer senv a

-- An executable consumer.
--
data StreamConsumer senv a where

  -- Stream reduction skeleton.
  StreamFold :: !(Val senv -> s -> Stream -> CIO s) -- Chunk consumer function.
             -> !(s -> Stream -> CIO r)             -- Finalizer function.
             -> !s                                  -- Accumulator (internal state).
             -> StreamConsumer senv r

  StreamStuple :: IsAtuple a
               => !(Atuple (StreamConsumer senv) (TupleRepr a))
               -> StreamConsumer senv a

type Chunk a = Regular a

-- Get all the shapes of a chunk of arrays. O(1).
--
chunkShape :: Shape sh => Chunk (Array sh a) -> sh
chunkShape !c = shape' c

-- Get all the elements of a chunk of arrays. O(1).
--
chunkElems :: Chunk (Array sh a) -> Vector a
chunkElems !c = elements' c

-- Convert a vector to a chunk of scalars.
--
vec2Chunk :: Elt e => Vector e -> Chunk (Scalar e)
vec2Chunk !v = vec2Regular v

-- Type conversion from ordinary sequence contexts to sequence
-- contexts of chunks.
data ChunkContext senv senv' where
  ChunkCtxEmpty :: ChunkContext () ()
  ChunkCtxCons  :: ChunkContext senv senv' -> ChunkContext (senv, a) (senv', Chunk a)

initialiseSeq :: SeqConfig
              -> PreOpenSeq DelayedOpenAcc aenv () arrs
              -> ExecOpenSeq               aenv () arrs
              -> Aval aenv
              -> Stream
              -> CIO (Int, StreamDAG () arrs)
initialiseSeq !conf !dseq !topSeq !aenv !stream =
  let !maxElemSize = shapeTreeMaxSize <$> seqShapes dseq (avalToValPartial aenv)
      !pd = maxStepSize (chunkSize conf) maxElemSize
  in
  if isVect dseq && pd > 1
    then liftIO (D.traceIO D.verbose $ "chunking with parallel degree " ++ show pd ++ "..") >>
         (pd,) <$> initialiseSeqChunked aenv topSeq ChunkCtxEmpty pd stream
    else liftIO (D.traceIO D.verbose "no chunking..")
         >> (1,)  <$> initialiseSeqLoop aenv topSeq stream
  where
    maxStepSize :: Int -> Maybe Int -> Int
    maxStepSize _             Nothing          = 1
    maxStepSize !maxChunkSize (Just !elemSize) =
      let (!a,!b) = maxChunkSize `quotRem` (elemSize `max` 1)
      in a + signum b

    -- Avoid synchronization and copying data from device to host
    -- by only considering the shapes of aenv for shape
    -- analysis. This means that the analysis is less total.
    avalToValPartial :: Aval aenv' -> ValPartial aenv'
    avalToValPartial !Aempty = EmptyPartial
    avalToValPartial (Apush       !aenv0 (Async _ !a)) = avalToValPartial aenv0 `PushTotalShapesOnly` a
    avalToValPartial (ApushNoSync !aenv0 !a)           = avalToValPartial aenv0 `PushTotalShapesOnly` a

    isJust !(Just _) = True
    isJust _         = False

    -- Is sequence amenable for vectorization?
    isVect :: PreOpenSeq acc aenv senv a -> Bool
    isVect !s =
      case s of
        Producer !p !s0 -> isVectP p && isVect s0
        Consumer !c    -> isVectC c
        Reify !f _     -> isJust f

    isVectP :: Producer acc aenv senv a -> Bool
    isVectP !p =
      case p of
        StreamIn _          -> True
        ToSeq !f _ _ _      -> isJust f
        MapSeq _ !f _       -> isJust f
        ZipWithSeq _ !f _ _ -> isJust f
        ScanSeq _ _ _       -> True

    isVectC :: Consumer acc aenv senv a -> Bool
    isVectC !c =
      case c of
        FoldSeq !f _ _ _ -> isJust f
        FoldSeqFlatten _ _ _ _ -> True
        Stuple !stup ->
          let isVectT :: Atuple (Consumer acc aenv senv) t -> Bool
              isVectT !NilAtup         = True
              isVectT !(SnocAtup t c0) = isVectT t && isVectC c0
          in isVectT stup

initialiseSeqChunked :: forall aenv senv senv' arrs.
                        Aval aenv
                     -> ExecOpenSeq aenv senv arrs
                     -> ChunkContext senv senv'
                     -> Int
                     -> Stream
                     -> CIO (StreamDAG senv' arrs)
initialiseSeqChunked !aenv !s !cctx !pd !spineStream =
      case s of
        ExecP !p !s0 -> StreamProducer <$> initProducer p <*> initialiseSeqChunked aenv s0 (ChunkCtxCons cctx) pd spineStream
        ExecC !c     -> StreamConsumer <$> initConsumer c
        ExecR !f !x  -> return $ initReify f x
      where
        cvtIdx :: Idx senv a -> Idx senv' (Chunk a)
        cvtIdx !x = go cctx x
          where
            go :: ChunkContext senv0 senv0' -> Idx senv0 a -> Idx senv0' (Chunk a)
            go !(ChunkCtxCons _  ) !ZeroIdx      = ZeroIdx
            go (ChunkCtxCons !ctx) (SuccIdx !x0) = SuccIdx (go ctx x0)
            go _ _ = error "unreachable"

        initReify :: Maybe (ExecOpenAfun aenv (Regular a -> Scalar Int -> a))
                  -> Idx senv a
                  -> StreamDAG senv' [a]
        initReify (Just !f) !x = StreamReify $ \ !senv !k !stream ->
              let c = prj (cvtIdx x) senv
                  g !i =
                    do
                      !i' <- newArrayAsync Z (const i) stream
                      evalAF2 f c i' stream
              in mapM g [0..k-1]
        initReify _ _ = error "unreachable"

        initProducer :: ExecP aenv senv a
                     -> CIO (StreamProducer senv' (Chunk a))
        initProducer !p =
          case p of
            ExecStreamIn _as -> error "ExecStreamIn is not supported with chunking"
            ExecToSeq (Just !f) !slix _ !arg -> do
              !sh <- case arg of
                Right (!shExp, _, _ )    -> evalE shExp
                Left  (!arr, _, _, _, _) -> return $ shape arr
              let
                !sl = sliceShape slix sh
                !n = coShapeSize slix (fromElt sh)
              return $ StreamMapFin (0, n) $ \ !_senv !i !stream -> do
                let !k = (pd `min` ((n - i) `max` 0))
                !out <- case arg of
                  Right (_, !kernel, !gamma)          -> toSeqOp kernel gamma aenv          sl i k stream
                  Left (!arr, !kp3, !kp5, !kp7, !kp9) -> useLazyOp kp3 kp5 kp7 kp9 arr slix sl i k stream
                -- Convert result to chunk
                evalAF1 f out stream
            ExecToSeq !Nothing _ _ _ -> error "unreachable"
            ExecMap _ !f' !x         -> return $ initMapSeq f' x
            ExecZipWith _ !f' !x !y  -> return $ initZipWithSeq f' x y
            ExecScanSeq !e _ !f !x   -> StreamScan scanner <$> (newArray Z . const =<< evalE e)
              where
                scanner !senv !a !stream = do
                  let !c = prj (cvtIdx x) senv
                  (!v, !accum) <- evalAF2 f a (chunkElems c) stream
                  return (vec2Chunk v, accum)

        initMapSeq :: Maybe (ExecOpenAfun aenv (Regular a -> Regular b))
                   -> Idx senv a
                   -> StreamProducer senv' (Chunk b)
        initMapSeq (Just !f') !x  = StreamMap (\ senv -> evalAF1 f' (prj (cvtIdx x) senv))
        initMapSeq _ _ = error "unreachable"

        initZipWithSeq :: Maybe (ExecOpenAfun aenv (Regular a -> Regular b -> Regular c))
                       -> Idx senv a
                       -> Idx senv b
                       -> StreamProducer senv' (Chunk c)
        initZipWithSeq (Just !f') !x !y = StreamMap (\ senv -> evalAF2 f' (prj (cvtIdx x) senv) (prj (cvtIdx y) senv))
        initZipWithSeq _ _ _ = error "unreachable"


        initConsumer :: ExecC aenv senv a
                     -> CIO (StreamConsumer senv' a)
        initConsumer !c =
          case c of
            ExecFoldSeq (Just !zipfun) !foldfun !e _ !x -> do
              let consumer !senv !v !stream =
                    let !arr = prj (cvtIdx x) senv
                    in evalAF2 zipfun v (chunkElems arr) stream
                  finalizer !v !stream = do
                    evalAF1 foldfun v stream
              !e' <- evalE e
              !a0 <- newArray (Z :. pd) (const e')
              return $ StreamFold consumer finalizer a0
            ExecFoldSeq _ _ _ _ _ -> error "unreachable"
            ExecFoldSeqFlatten (Just !f') _ !acc !x -> do
              let consumer !senv !a !stream =
                    let !arr = prj (cvtIdx x) senv
                    in evalAF2 f' a arr stream
              !a0 <- executeOpenAcc True acc aenv spineStream
              return $ StreamFold consumer (\ accum _ -> return accum) a0
            ExecFoldSeqFlatten _ _ _ _ -> error "unreachable"
            ExecStuple t ->
              let initTup :: Atuple (ExecC aenv senv) t -> CIO (Atuple (StreamConsumer senv') t)
                  initTup NilAtup            = return $ NilAtup
                  initTup (SnocAtup !t0 !c0) = SnocAtup <$> initTup t0 <*> initConsumer c0
              in StreamStuple <$> initTup t

        evalAF1 :: ExecOpenAfun aenv (a -> b) -> a -> Stream -> CIO b
        evalAF1 (Alam (Abody !f)) !x !stream = do
          executeOpenAcc False f (aenv `ApushNoSync` x) stream
        evalAF1 _ _ _ = error "error AF1"

        evalAF2 :: ExecOpenAfun aenv (a -> b -> c) -> a -> b -> Stream -> CIO c
        evalAF2 (Alam (Alam (Abody !f))) !x !y !stream = do
          executeOpenAcc False f (aenv `ApushNoSync` x `ApushNoSync` y) stream
        evalAF2 _ _ _ _ = error "error AF2"

        evalE :: ExecExp aenv t -> CIO t
        evalE !exp = executeExp exp aenv spineStream


initialiseSeqLoop :: forall aenv senv arrs.
                     Aval aenv
                  -> ExecOpenSeq aenv senv arrs
                  -> Stream
                  -> CIO (StreamDAG senv arrs)
initialiseSeqLoop !aenv !s !spineStream =
      case s of
        ExecP !p !s0 -> StreamProducer <$> initProducer p <*> initialiseSeqLoop aenv s0 spineStream
        ExecC !c     -> StreamConsumer <$> initConsumer c
        ExecR _ !x   -> return $ StreamReify (\ senv _ _ -> return [prj x senv])
      where
        initProducer :: ExecP aenv senv a
                     -> CIO (StreamProducer senv a)
        initProducer !p =
          case p of
            ExecStreamIn arrs -> return (StreamStreamIn arrs)
            ExecToSeq _ !slix _ !arg -> do
              !sh <- case arg of
                Right (!shExp, _, _)     -> evalE shExp
                Left  (!arr, _, _, _, _) -> return $ shape arr
              let
                !sl = sliceShape slix sh
                !n = coShapeSize slix (fromElt sh)
              return $ StreamMapFin (0, n) $ \ _senv i stream -> do
                !out <- case arg of
                  Right (_, !kernel, !gamma)           -> toSeqOp kernel gamma aenv          sl i 1 stream
                  Left  (!arr, !kp3, !kp5, !kp7, !kp9) -> useLazyOp kp3 kp5 kp7 kp9 arr slix sl i 1 stream
                -- Convert result to chunk
                return (reshapeOp sl out)
            ExecMap !f _ !x        -> return $ StreamMap $ \ senv -> evalAF1 f (prj x senv)
            ExecZipWith !f _ !x !y -> return $ StreamMap $ \ senv -> evalAF2 f (prj x senv) (prj y senv)
            ExecScanSeq !e !f _ !x -> StreamScan scanner <$> (newArray Z . const =<< evalE e)
              where
                scanner !senv !a !stream = do
                  let !c = prj x senv
                  !v <- evalAF2 f a c stream
                  return (v, v)

        initConsumer :: ExecC aenv senv a
                     -> CIO (StreamConsumer senv a)
        initConsumer !c =
          case c of
            ExecFoldSeq _ _ !e !f !x -> do
              let consumer !senv !a !stream = do
                    let !arr = prj x senv
                    evalAF2 f a arr stream
              StreamFold consumer (\ !accum _ -> return accum) <$> (newArray Z . const =<< evalE e)
            ExecFoldSeqFlatten _ !f !acc !x -> do
              let consumer !senv !a !stream =
                    let !v = prj x senv
                    in do
                      !sh <- newArrayAsync (Z :. 1) (const (shape v)) stream
                      evalAF3 f a sh  (reshapeOp (Z :. size (shape v)) v) stream
              !a0 <- executeOpenAcc True acc aenv spineStream
              return $ StreamFold consumer (\ accum _ -> return accum) a0
            ExecStuple t ->
              let initTup :: Atuple (ExecC aenv senv) t -> CIO (Atuple (StreamConsumer senv) t)
                  initTup !NilAtup           = return $ NilAtup
                  initTup (SnocAtup !t0 !c0) = SnocAtup <$> initTup t0 <*> initConsumer c0
              in StreamStuple <$> initTup t

        evalAF1 :: ExecOpenAfun aenv (a -> b) -> a -> Stream -> CIO b
        evalAF1 (Alam (Abody !f)) !x !stream = do
          executeOpenAcc False f (aenv `ApushNoSync` x) stream
        evalAF1 _ _ _ = error "error AF1"

        evalAF2 :: ExecOpenAfun aenv (a -> b -> c) -> a -> b -> Stream -> CIO c
        evalAF2 (Alam (Alam (Abody !f))) !x !y !stream =  do
          executeOpenAcc False f (aenv `ApushNoSync` x `ApushNoSync` y) stream
        evalAF2 _ _ _ _ = error "error AF2"

        evalAF3 :: ExecOpenAfun aenv (a -> b -> c -> d) -> a -> b -> c -> Stream -> CIO d
        evalAF3 (Alam (Alam (Alam (Abody !f)))) !x !y !z !stream =  do
          executeOpenAcc False f (aenv `ApushNoSync` x `ApushNoSync` y `ApushNoSync` z) stream
        evalAF3 _ _ _ _ _ = error "error AF3"

        evalE :: ExecExp aenv t -> CIO t
        evalE !exp = executeExp exp aenv spineStream


coShapeSize :: SliceIndex slix sl co sh -> sh -> Int
coShapeSize SliceNil            ()          = 1
coShapeSize (SliceAll   !slix0) (!sh0, _  ) = coShapeSize slix0 sh0
coShapeSize (SliceFixed !slix0) (!sh0, !sz) = coShapeSize slix0 sh0 * sz

(.:) :: Shape sl => Int -> sl -> (sl :. Int)
(.:) !sz !sh = listToShape (shapeToList sh ++ [sz])

toSeqOp :: (Shape sl, Elt e)
        => AccKernel (Array (sl :. Int) e)
        -> Gamma aenv
        -> Aval aenv
        -> sl
        -> Int
        -> Int
        -> Stream
        -> CIO (Array (sl :. Int) e)
toSeqOp !kernel !gamma !aenv !sl !i !k !stream = do
  let !sh = k .: sl
  !out <- allocateArray sh -- ###
  execute kernel gamma aenv (size sh) (i, out) stream
  return out

useLazyOp :: (Shape sh, Shape sl, Elt e)
          => AccKernel (Array DIM3 e)
          -> AccKernel (Array DIM5 e)
          -> AccKernel (Array DIM7 e)
          -> AccKernel (Array DIM9 e)
          -> Array sh e
          -> SliceIndex slix (EltRepr sl) co (EltRepr sh)
          -> sl
          -> Int
          -> Int
          -> Stream
          -> CIO (Array (sl :. Int) e)
useLazyOp !kp3 !kp5 !kp7 !kp9 !arr !slix !sl !i !k !stream = do
  let
    !sh = (k .: sl)
    !args = copyArgs slix (fromElt (shape arr)) i (i + k)

    -- specialize mapM_. Otherwise get 'untouchable' type error.
    map' :: (a -> CIO ()) -> [a] -> CIO ()
    map' = mapM_

  -- tmp holds the copied array before permutation.
  tmpArr@(Array _ tmp) <- allocateArray sh -- ###
  -- out holds the end result.
  outArr@(Array _ out) <- allocateArray sh -- ###
  -- Poke 2D regions from host to device:
  mapM_ (\ !x -> pokeCopyArgs x arr tmpArr) args

  -- Permute each poked region to conform with slicing. TODO test
  -- whether permutation is needed at all before doing this.
  withDevicePtrs tmp (Just stream) $ \ !dtmp ->
    withDevicePtrs out (Just stream) $ \ !dout ->
      map' (\ !x ->
             let !dtmp' = advancePtrsOfArrayData tmp (offset x) dtmp
                 !dout' = advancePtrsOfArrayData out (offset x) dout
                 !mdtmp = marshalDevicePtrs tmp dtmp'
                 !mdout = marshalDevicePtrs out dout'
             in
             case permutation x of
               Permut sh0 p ->
                 case p of
                   P3 -> execute kp3 mempty Aempty (size sh0) (sh0, mdtmp, shapeP P3 sh0, mdout) stream
                   P5 -> execute kp5 mempty Aempty (size sh0) (sh0, mdtmp, shapeP P5 sh0, mdout) stream
                   P7 -> execute kp7 mempty Aempty (size sh0) (sh0, mdtmp, shapeP P7 sh0, mdout) stream
                   P9 -> execute kp9 mempty Aempty (size sh0) (sh0, mdtmp, shapeP P9 sh0, mdout) stream
           ) args
  return outArr


streamSeq :: ExecSeq [a] -> StreamSeq a
streamSeq (ExecS !binds !dsequ !sequ) = StreamSeq $ do
  !aenv <- executeExtend binds Aempty
  streaming
    (\ !stream ->
      do (!pd, !s) <- initialiseSeq defaultSeqConfig dsequ sequ aenv stream
         return $ Just ([], streamOutSequence s pd stream)
    ) wait

streamOutSequence :: StreamDAG () [arrs]
                  -> Int
                  -> Stream
                  -> StreamSeq arrs
streamOutSequence !topSeq !pd !stream
  = loop pd topSeq
  where
    loop :: Int
         -> StreamDAG () [arrs]
         -> StreamSeq arrs
    loop !n !s = StreamSeq $
      let !k = stepSize n s
      in if k == 0
            then return Nothing
            else do
              (!s', !arrs0) <- stepSequence s Empty k stream
              return $ Just (arrs0, loop n s')

executeSequence :: StreamDAG () arrs
                -> Int
                -> Stream
                -> CIO arrs
executeSequence !topSeq !pd !stream
  = loop pd topSeq
  where
    loop :: Int
         -> StreamDAG () arrs
         -> CIO arrs
    loop !n !s =
      let !k = stepSize n s
      in if k == 0
         then returnOut s stream
         else do
           (!s', _) <- stepSequence s Empty k stream
           loop n s'

stepSequence :: StreamDAG senv a
             -> Val senv
             -> Int
             -> Stream
             -> CIO (StreamDAG senv a, a)
stepSequence !s !senv !k !stream =
  case s of
    StreamProducer !p !s0 -> do
      (!c', !p') <- produce p senv k stream
      (!s0', a)  <- stepSequence s0 (senv `Push` c') k stream
      return (StreamProducer p' s0', a)
    StreamConsumer !c  -> do
      !c' <- consume c senv stream
      return $ (StreamConsumer c', error "use returnOut")
    StreamReify !f -> do
      !as <- f senv k stream
      return (StreamReify f, as)

stepSize :: Int -> StreamDAG senv arrs -> Int
stepSize !pd !s =
  case s of
    StreamProducer !p !s0 -> min (stepSize pd s0) $
      case p of
        StreamStreamIn !xs -> length (take pd xs)
        StreamMapFin (!i, !m) _ -> ((m - i) `max` 0) `min` pd
        _ -> pd
    _ -> pd

produce :: StreamProducer senv a
        -> Val senv
        -> Int
        -> Stream
        -> CIO (a, StreamProducer senv a)
produce !p !senv !k !stream =
  case p of
    StreamStreamIn xs -> do
      let (!x, !xs') = (head xs, tail xs)
      return (x, StreamStreamIn xs')
    StreamMap !f -> do
      !c <- f senv stream
      return (c, StreamMap f)
    StreamMapFin (!i, !n) !f -> do
      !c <- f senv i stream
      return (c, StreamMapFin (i + k, n) f)
    StreamScan !scanner !a -> do
      (!c', !a') <- scanner senv a stream
      return (c', StreamScan scanner a')

consume :: forall senv arrs. StreamConsumer senv arrs -> Val senv -> Stream -> CIO (StreamConsumer senv arrs)
consume !con !senv !stream = go con
  where
    go :: StreamConsumer senv a -> CIO (StreamConsumer senv a)
    go !c =
      case c of
        StreamFold !f !g !acc ->
          do !acc' <- f senv acc stream
             return (StreamFold f g acc')
        StreamStuple t ->
          let consT :: Atuple (StreamConsumer senv) t -> CIO (Atuple (StreamConsumer senv) t)
              consT !NilAtup           = return (NilAtup)
              consT (SnocAtup !t0 !c0) = do
                !c'  <- go c0
                !t'  <- consT t0
                return (SnocAtup t' c')
          in do
            !t' <- consT t
            return (StreamStuple t')

returnOut :: StreamDAG senv arrs -> Stream -> CIO arrs
returnOut !s !stream =
  case s of
    StreamProducer _ !s0 -> returnOut s0 stream
    StreamConsumer !c -> retC c
    StreamReify _ -> error "absurd"
  where
    retC :: StreamConsumer senv arrs -> CIO arrs
    retC !c =
      case c of
        StreamFold _ !g !accum -> g accum stream
        StreamStuple !t ->
          let retT :: Atuple (StreamConsumer senv) t -> CIO t
              retT !NilAtup = return ()
              retT (SnocAtup !t0 !c0) = (,) <$> retT t0 <*> retC c0
          in toAtuple <$> retT t


-- Evaluating bindings
-- -------------------

executeExtend :: Extend ExecOpenAcc aenv aenv' -> Aval aenv -> CIO (Aval aenv')
executeExtend BaseEnv         !aenv = return aenv
executeExtend (PushEnv !e !a) !aenv = do
  !aenv' <- executeExtend e aenv
  streaming (executeOpenAcc True a aenv') $ \ !a' -> return $ Apush aenv' a'


-- Scalar expression evaluation
-- ----------------------------

executeExp :: ExecExp aenv t -> Aval aenv -> Stream -> CIO t
executeExp !exp !aenv !stream = executeOpenExp exp Empty aenv stream

executeOpenExp :: forall env aenv exp. ExecOpenExp env aenv exp -> Val env -> Aval aenv -> Stream -> CIO exp
executeOpenExp !rootExp !env !aenv !stream = travE rootExp
  where
    travE :: ExecOpenExp env aenv t -> CIO t
    travE exp = case exp of
      Var ix                    -> return (prj ix env)
      Let bnd body              -> travE bnd >>= \x -> executeOpenExp body (env `Push` x) aenv stream
      Const c                   -> return (toElt c)
      PrimConst c               -> return (evalPrimConst c)
      PrimApp f x               -> evalPrim f <$> travE x
      Tuple t                   -> toTuple <$> travT t
      Prj ix e                  -> evalPrj ix . fromTuple <$> travE e
      Cond p t e                -> travE p >>= \x -> if x then travE t else travE e
      While p f x               -> while p f =<< travE x
      IndexAny                  -> return Any
      IndexNil                  -> return Z
      IndexCons sh sz           -> (:.) <$> travE sh <*> travE sz
      IndexHead sh              -> (\(_  :. ix) -> ix) <$> travE sh
      IndexTail sh              -> (\(ix :.  _) -> ix) <$> travE sh
      IndexTrans sh             -> transpose <$> travE sh
      IndexSlice ix slix sh     -> indexSlice ix <$> travE slix <*> travE sh
      IndexFull ix slix sl      -> indexFull  ix <$> travE slix <*> travE sl
      ToIndex sh ix             -> toIndex   <$> travE sh  <*> travE ix
      FromIndex sh ix           -> fromIndex <$> travE sh  <*> travE ix
      Intersect sh1 sh2         -> intersect <$> travE sh1 <*> travE sh2
      Union sh1 sh2             -> union <$> travE sh1 <*> travE sh2
      ShapeSize sh              -> size  <$> travE sh
      Shape acc                 -> shape <$> travA acc
      Index acc ix              -> join $ index      <$> travA acc <*> travE ix
      LinearIndex acc ix        -> join $ indexArray <$> travA acc <*> travE ix
      Foreign _ f x             -> foreign f x

    -- Helpers
    -- -------

    travT :: Tuple (ExecOpenExp env aenv) t -> CIO t
    travT tup = case tup of
      NilTup            -> return ()
      SnocTup !t !e     -> (,) <$> travT t <*> travE e

    travA :: ExecOpenAcc aenv a -> CIO a
    travA !acc = executeOpenAcc True acc aenv stream

    foreign :: ExecFun () (a -> b) -> ExecOpenExp env aenv a -> CIO b
    foreign (Lam (Body f)) x = travE x >>= \e -> executeOpenExp f (Empty `Push` e) Aempty stream
    foreign _              _ = error "I bless the rains down in Africa"

    travF1 :: ExecOpenFun env aenv (a -> b) -> a -> CIO b
    travF1 (Lam (Body f)) x = executeOpenExp f (env `Push` x) aenv stream
    travF1 _              _ = error "Gonna take some time to do the things we never have"

    while :: ExecOpenFun env aenv (a -> Bool) -> ExecOpenFun env aenv (a -> a) -> a -> CIO a
    while !p !f !x = do
      ok <- travF1 p x
      if ok then while p f =<< travF1 f x
            else return x

    indexSlice :: (Elt slix, Elt sh, Elt sl)
               => SliceIndex (EltRepr slix) (EltRepr sl) co (EltRepr sh)
               -> slix
               -> sh
               -> sl
    indexSlice !ix !slix !sh = toElt $! restrict ix (fromElt slix) (fromElt sh)
      where
        restrict :: SliceIndex slix sl co sh -> slix -> sh -> sl
        restrict SliceNil              ()        ()       = ()
        restrict (SliceAll   sliceIdx) (slx, ()) (sl, sz) = (restrict sliceIdx slx sl, sz)
        restrict (SliceFixed sliceIdx) (slx,  _) (sl,  _) = restrict sliceIdx slx sl

    indexFull :: (Elt slix, Elt sh, Elt sl)
              => SliceIndex (EltRepr slix) (EltRepr sl) co (EltRepr sh)
              -> slix
              -> sl
              -> sh
    indexFull !ix !slix !sl = toElt $! extend ix (fromElt slix) (fromElt sl)
      where
        extend :: SliceIndex slix sl co sh -> slix -> sl -> sh
        extend SliceNil              ()        ()       = ()
        extend (SliceAll sliceIdx)   (slx, ()) (sh, sz) = (extend sliceIdx slx sh, sz)
        extend (SliceFixed sliceIdx) (slx, sz) sh       = (extend sliceIdx slx sh, sz)

    index :: (Shape sh, Elt e) => Array sh e -> sh -> CIO e
    index !arr !ix = indexArray arr (toIndex (shape arr) ix)


-- Marshalling data
-- ----------------

-- Data which can be marshalled as function arguments to a kernel invocation.
--
class Marshalable a where
  marshal :: a -> Maybe Stream -> ContT b CIO [CUDA.FunParam]

instance Marshalable () where
  marshal () _ = return []

instance Marshalable CUDA.FunParam where
  marshal !x _ = return [x]

instance ArrayElt e => Marshalable (ArrayData e) where
  marshal !ad ms = ContT $ marshalArrayData ad ms

instance Shape sh => Marshalable sh where
  marshal !sh ms = marshal (reverse (shapeToList sh)) ms

instance Marshalable a => Marshalable [a] where
  marshal xs ms = concatMapM (flip marshal ms) xs

instance (Marshalable sh, Elt e) => Marshalable (Array sh e) where
  marshal !(Array sh ad) ms = (++) <$> marshal (toElt sh :: sh) ms <*> marshal ad ms

instance (Marshalable a, Marshalable b) => Marshalable (a, b) where
  marshal (!a, !b) ms = (++) <$> marshal a ms <*> marshal b ms

instance (Marshalable a, Marshalable b, Marshalable c) => Marshalable (a, b, c) where
  marshal (!a, !b, !c) ms
    = concat <$> sequence [marshal a ms, marshal b ms, marshal c ms]

instance (Marshalable a, Marshalable b, Marshalable c, Marshalable d)
      => Marshalable (a, b, c, d) where
  marshal (!a, !b, !c, !d) ms
    = concat <$> sequence [marshal a ms, marshal b ms, marshal c ms, marshal d ms]


#define primMarshalable(ty)                                                    \
instance Marshalable (ty) where {                                              \
  marshal !x _ = return [CUDA.VArg x] }

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
primMarshalable(CUDA.DevicePtr a)


-- Note [Array references in scalar code]
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--
-- All CUDA devices have between 6-8KB of read-only texture memory per
-- multiprocessor. Since all arrays in Accelerate are immutable, we can always
-- access input arrays through the texture cache to reduce global memory demand
-- when accesses do not follow the regular patterns required for coalescing.
--
-- This is great for older 1.x series devices, but newer devices have a
-- dedicated L2 cache (device dependent, 256KB-1.5MB), as well as a configurable
-- L1 cache combined with shared memory (16-48KB).
--
-- For older 1.x series devices, we pass free array variables as texture
-- references, but for new devices we pass them as standard array arguments so
-- as to use the larger available caches.
--

marshalAccEnvTex :: AccKernel a -> Aval aenv -> Gamma aenv -> Stream -> ContT b CIO [CUDA.FunParam]
marshalAccEnvTex !kernel !aenv (Gamma !gamma) !stream
  = flip concatMapM (Map.toList gamma)
  $ \(Idx_ !(idx :: Idx aenv (Array sh e)), i) ->
        do arr <- after stream (aprj idx aenv)
           ContT $ \f -> marshalAccTex (namesOfArray (groupOfInt i) (undefined :: e)) kernel arr (Just stream) (f ())
           marshal (shape arr) (Just stream)

marshalAccTex :: (Name,[Name]) -> AccKernel a -> Array sh e -> Maybe Stream -> CIO b -> CIO b
marshalAccTex (_, !arrIn) (AccKernel _ _ !lmdl _ _ _ _) (Array !sh !adata) ms run
  = do
      texs <- liftIO $ withLifetime lmdl $ \mdl -> (sequence' $ map (CUDA.getTex mdl) (reverse arrIn))
      marshalTextureData adata (R.size sh) texs ms (const run)

marshalAccEnvArg :: Aval aenv -> Gamma aenv -> Stream -> ContT b CIO [CUDA.FunParam]
marshalAccEnvArg !aenv (Gamma !gamma) !stream
  = concatMapM (\(Idx_ !idx) -> flip marshal (Just stream) =<< after stream (aprj idx aenv)) (Map.keys gamma)


-- A lazier version of 'Control.Monad.sequence'
--
sequence' :: [IO a] -> IO [a]
sequence' = foldr k (return [])
  where k m ms = do { x <- m; xs <- unsafeInterleaveIO ms; return (x:xs) }

-- Generalise concatMap for teh monadz
--
concatMapM :: Monad m => (a -> m [b]) -> [a] -> m [b]
concatMapM f xs = concat `liftM` mapM f xs


-- Kernel execution
-- ----------------

-- What launch parameters should we use to execute the kernel with a number of
-- array elements?
--
configure :: AccKernel a -> Int -> (Int, Int, Int)
configure (AccKernel _ _ _ _ !cta !smem !grid) !n = (cta, grid n, smem)


-- Marshal the kernel arguments. For older 1.x devices this binds free arrays to
-- texture references, and for newer devices adds the parameters to the front of
-- the argument list
--
arguments :: Marshalable args
          => AccKernel a
          -> Aval aenv
          -> Gamma aenv
          -> args
          -> Stream
          -> ContT b CIO [CUDA.FunParam]
arguments !kernel !aenv !gamma !a !stream = do
  dev <- asks deviceProperties
  let marshaller | computeCapability dev < Compute 2 0   = marshalAccEnvTex kernel
                 | otherwise                             = marshalAccEnvArg
  --
  (++) <$> marshaller aenv gamma stream <*> marshal a (Just stream)


-- Link the binary object implementing the computation, configure the kernel
-- launch parameters, and initiate the computation. This also handles lifting
-- and binding of array references from scalar expressions.
--
execute :: Marshalable args
        => AccKernel a                  -- The binary module implementing this kernel
        -> Gamma aenv                   -- variables of arrays embedded in scalar expressions
        -> Aval aenv                    -- the environment
        -> Int                          -- a "size" parameter, typically number of elements in the output
        -> args                         -- arguments to marshal to the kernel function
        -> Stream                       -- Compute stream to execute in
        -> CIO ()
execute !kernel !gamma !aenv !n !a !stream = flip runContT return $ do
  args  <- arguments kernel aenv gamma a stream
  liftIO $ launch kernel (configure kernel n) args stream


-- Execute a device function, with the given thread configuration and function
-- parameters. The tuple contains (threads per block, grid size, shared memory)
--
launch :: AccKernel a -> (Int,Int,Int) -> [CUDA.FunParam] -> Stream -> IO ()
launch (AccKernel entry !fn _ _ _ _ _) !(cta, grid, smem) !args !stream
  = D.timed D.dump_exec msg (Just stream)
  $ CUDA.launchKernel fn (grid,1,1) (cta,1,1) smem (Just stream) args
  where
    msg gpuTime cpuTime
      = "exec: " ++ entry ++ "<<< " ++ shows grid ", " ++ shows cta ", " ++ shows smem " >>> "
                 ++ D.elapsed gpuTime cpuTime


