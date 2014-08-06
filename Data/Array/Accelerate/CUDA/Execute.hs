{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE CPP                        #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GADTs                      #-}
{-# LANGUAGE IncoherentInstances        #-}
{-# LANGUAGE NoForeignFunctionInterface #-}
{-# LANGUAGE PatternGuards              #-}
{-# LANGUAGE ScopedTypeVariables        #-}
{-# LANGUAGE TemplateHaskell            #-}
{-# LANGUAGE TypeOperators              #-}
{-# LANGUAGE TypeSynonymInstances       #-}
{-# LANGUAGE UndecidableInstances       #-}
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
  executeAcc, executeAfun1

) where

-- friends
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.FullList                      ( FullList(..), List(..) )
import Data.Array.Accelerate.CUDA.Array.Data
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
import Data.Array.Accelerate.Tuple
import Data.Array.Accelerate.Interpreter                        ( evalPrim, evalPrimConst, evalPrj )
import Data.Array.Accelerate.Array.Data                         ( ArrayElt, ArrayData )
import Data.Array.Accelerate.Array.Representation               ( SliceIndex(..) )
import qualified Data.Array.Accelerate.Array.Representation     as R


-- standard library
import Prelude                                                  hiding ( exp, sum, iterate )
import Control.Applicative                                      hiding ( Const )
import Control.Monad                                            ( join, when, liftM )
import Control.Monad.Reader                                     ( asks )
import Control.Monad.State                                      ( gets )
import Control.Monad.Trans                                      ( MonadIO, liftIO )
import System.IO.Unsafe                                         ( unsafeInterleaveIO )
import Data.Int
import Data.Word
import Data.Maybe

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


-- Projection of a value from a valuation using a de Bruijn index.
--
aprj :: Idx env t -> Aval env -> Async t
aprj ZeroIdx       (Apush _   x) = x
aprj (SuccIdx idx) (Apush val _) = aprj idx val
aprj _             _             = $internalError "aprj" "inconsistent valuation"


-- All work submitted to the given stream will occur after the asynchronous
-- event for the given array has been fulfilled. Synchronisation is performed
-- efficiently on the device. This function returns immediately.
--
after :: MonadIO m => Stream -> Async a -> m a
after stream (Async event arr) = liftIO $ Event.after event stream >> return arr


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
  Stream.streaming context reservoir first (\e a -> second (Async e a))


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
executeAcc !acc = streaming (executeOpenAcc acc Aempty) wait

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
executeOpenAfun1 (Alam (Abody f)) aenv x = streaming (executeOpenAcc f (aenv `Apush` x)) wait
executeOpenAfun1 _                _    _ = error "the sword comes out after you swallow it, right?"


-- Evaluate an open array computation
--
executeOpenAcc
    :: forall aenv arrs.
       ExecOpenAcc aenv arrs
    -> Aval aenv
    -> Stream
    -> CIO arrs
executeOpenAcc EmbedAcc{} _ _
  = $internalError "execute" "unexpected delayed array"
executeOpenAcc (ExecAcc (FL () kernel more) !gamma !pacc) !aenv !stream
  = case pacc of

      -- Array introduction
      Use arr                   -> return (toArr arr)
      Unit x                    -> newArray Z . const =<< travE x

      -- Environment manipulation
      Avar ix                   -> after stream (aprj ix aenv)
      Alet bnd body             -> streaming (executeOpenAcc bnd aenv) (\x -> executeOpenAcc body (aenv `Apush` x) stream)
      Apply f a                 -> streaming (executeOpenAcc a aenv)   (executeOpenAfun1 f aenv)
      Atuple tup                -> toTuple <$> travT tup
      Aprj ix tup               -> evalPrj ix . fromTuple <$> travA tup
      Acond p t e               -> travE p >>= \x -> if x then travA t else travA e
      Awhile p f a              -> awhile p f =<< travA a

      -- Foreign
      Aforeign ff afun a        -> fromMaybe (executeAfun1 afun) (canExecuteAcc ff) =<< travA a

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

      -- Removed by fusion
      Replicate _ _ _           -> fusionError
      Slice _ _ _               -> fusionError
      ZipWith _ _ _             -> fusionError

  where
    fusionError = $internalError "executeOpenAcc" "unexpected fusible matter"

    -- term traversals
    travA :: ExecOpenAcc aenv a -> CIO a
    travA !acc = executeOpenAcc acc aenv stream

    travE :: ExecExp aenv t -> CIO t
    travE !exp = executeExp exp aenv stream

    travT :: Atuple (ExecOpenAcc aenv) t -> CIO t
    travT NilAtup          = return ()
    travT (SnocAtup !t !a) = (,) <$> travT t <*> travA a

    awhile :: PreOpenAfun ExecOpenAcc aenv (a -> Scalar Bool) -> PreOpenAfun ExecOpenAcc aenv (a -> a) -> a -> CIO a
    awhile p f a = do
      nop <- liftIO Event.create                -- record event never call, so this is a functional no-op
      r   <- executeOpenAfun1 p aenv (Async nop a)
      ok  <- indexArray r 0                     -- TLM TODO: memory manager should remember what is already on the host
      if ok then awhile p f =<< executeOpenAfun1 f aenv (Async nop a)
            else return a

    -- get the extent of an embedded array
    extent :: Shape sh => ExecOpenAcc aenv (Array sh e) -> CIO sh
    extent ExecAcc{}     = $internalError "executeOpenAcc" "expected delayed array"
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

    -- Change the shape of an array without altering its contents. This does not
    -- execute any kernel programs.
    --
    reshapeOp :: Shape sh => sh -> Array sh' e -> Array sh e
    reshapeOp sh (Array sh' adata)
      = $boundsCheck "reshape" "shape mismatch" (size sh == R.size sh')
      $ Array (fromElt sh) adata

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
      out                       <- devicePtrsOfArrayData adata
      let (!body, !sum)
            | left      = (out, advancePtrsOfArrayData adata numElements out)
            | otherwise = (advancePtrsOfArrayData adata 1 out, out)
      --
      scanCore numElements arr body sum
      return arr

    scan1Op :: forall e. Elt e => (Z :. Int) -> CIO (Vector e)
    scan1Op !(Z :. numElements) = do
      arr@(Array _ adata)       <- allocateArray (Z :. numElements + 1) :: CIO (Vector e)
      body                      <- devicePtrsOfArrayData adata
      let sum {- to fix type -} =  advancePtrsOfArrayData adata numElements body
      --
      scanCore numElements arr body sum
      return (Array ((),numElements) adata)

    scan'Op :: forall e. Elt e => (Z :. Int) -> CIO (Vector e, Scalar e)
    scan'Op !(Z :. numElements) = do
      vec@(Array _ ad_vec)      <- allocateArray (Z :. numElements) :: CIO (Vector e)
      sum@(Array _ ad_sum)      <- allocateArray Z                  :: CIO (Scalar e)
      d_vec                     <- devicePtrsOfArrayData ad_vec
      d_sum                     <- devicePtrsOfArrayData ad_sum
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
      ((), d_locks)     <- devicePtrsOfArrayData locks  :: CIO ((), CUDA.DevicePtr Int32)

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
         then marshalAccTex (namesOfArray "Stencil" (undefined :: a)) kernel arr >>
              execute kernel gamma aenv (size sh) (out, sh) stream
         else execute kernel gamma aenv (size sh) (out, arr) stream
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
             then marshalAccTex (namesOfArray "Stencil1" (undefined :: a)) op arr1 >>
                  marshalAccTex (namesOfArray "Stencil2" (undefined :: b)) op arr2 >>
                  execute op gamma aenv (size sh) (out, sh1,  sh2) stream
             else execute op gamma aenv (size sh) (out, arr1, arr2) stream
          --
          return out

      | otherwise
      = $internalError "stencil2Op" "missing stencil specialisation kernel"


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
      IndexSlice ix slix sh     -> indexSlice ix <$> travE slix <*> travE sh
      IndexFull ix slix sl      -> indexFull  ix <$> travE slix <*> travE sl
      ToIndex sh ix             -> toIndex   <$> travE sh  <*> travE ix
      FromIndex sh ix           -> fromIndex <$> travE sh  <*> travE ix
      Intersect sh1 sh2         -> intersect <$> travE sh1 <*> travE sh2
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
    travA !acc = executeOpenAcc acc aenv stream

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
  marshal :: a -> CIO [CUDA.FunParam]

instance Marshalable () where
  marshal () = return []

instance Marshalable CUDA.FunParam where
  marshal !x = return [x]

instance ArrayElt e => Marshalable (ArrayData e) where
  marshal !ad = marshalArrayData ad

instance Shape sh => Marshalable sh where
  marshal !sh = marshal (reverse (shapeToList sh))

instance Marshalable a => Marshalable [a] where
  marshal = concatMapM marshal

instance (Marshalable sh, Elt e) => Marshalable (Array sh e) where
  marshal !(Array sh ad) = (++) <$> marshal (toElt sh :: sh) <*> marshal ad

instance (Marshalable a, Marshalable b) => Marshalable (a, b) where
  marshal (!a, !b) = (++) <$> marshal a <*> marshal b

instance (Marshalable a, Marshalable b, Marshalable c) => Marshalable (a, b, c) where
  marshal (!a, !b, !c)
    = concat <$> sequence [marshal a, marshal b, marshal c]

instance (Marshalable a, Marshalable b, Marshalable c, Marshalable d)
      => Marshalable (a, b, c, d) where
  marshal (!a, !b, !c, !d)
    = concat <$> sequence [marshal a, marshal b, marshal c, marshal d]


#define primMarshalable(ty)                                                    \
instance Marshalable (ty) where {                                              \
  marshal !x = return [CUDA.VArg x] }

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

marshalAccEnvTex :: AccKernel a -> Aval aenv -> Gamma aenv -> Stream -> CIO [CUDA.FunParam]
marshalAccEnvTex !kernel !aenv (Gamma !gamma) !stream
  = flip concatMapM (Map.toList gamma)
  $ \(Idx_ !(idx :: Idx aenv (Array sh e)), i) ->
        do arr <- after stream (aprj idx aenv)
           marshalAccTex (namesOfArray (groupOfInt i) (undefined :: e)) kernel arr
           marshal (shape arr)

marshalAccTex :: (Name,[Name]) -> AccKernel a -> Array sh e -> CIO ()
marshalAccTex (_, !arrIn) (AccKernel _ _ !mdl _ _ _ _) (Array !sh !adata)
  = marshalTextureData adata (R.size sh) =<< liftIO (sequence' $ map (CUDA.getTex mdl) (reverse arrIn))

marshalAccEnvArg :: Aval aenv -> Gamma aenv -> Stream -> CIO [CUDA.FunParam]
marshalAccEnvArg !aenv (Gamma !gamma) !stream
  = concatMapM (\(Idx_ !idx) -> marshal =<< after stream (aprj idx aenv)) (Map.keys gamma)


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
          -> CIO [CUDA.FunParam]
arguments !kernel !aenv !gamma !a !stream = do
  dev <- asks deviceProperties
  let marshaller | computeCapability dev < Compute 2 0   = marshalAccEnvTex kernel
                 | otherwise                             = marshalAccEnvArg
  --
  (++) <$> marshaller aenv gamma stream <*> marshal a


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
execute !kernel !gamma !aenv !n !a !stream = do
  args <- arguments kernel aenv gamma a stream
  launch kernel (configure kernel n) args stream


-- Execute a device function, with the given thread configuration and function
-- parameters. The tuple contains (threads per block, grid size, shared memory)
--
launch :: AccKernel a -> (Int,Int,Int) -> [CUDA.FunParam] -> Stream -> CIO ()
launch (AccKernel entry !fn _ _ _ _ _) !(cta, grid, smem) !args !stream
  = D.timed D.dump_exec msg (Just stream)
  $ liftIO $ CUDA.launchKernel fn (grid,1,1) (cta,1,1) smem (Just stream) args
  where
    msg gpuTime cpuTime
      = "exec: " ++ entry ++ "<<< " ++ shows grid ", " ++ shows cta ", " ++ shows smem " >>> "
                 ++ D.elapsed gpuTime cpuTime

