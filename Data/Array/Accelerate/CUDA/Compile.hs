{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Compile
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Compile (

  -- * generate and compile kernels to realise a computation
  compileAcc, compileAfun

) where

#include "accelerate.h"

-- friends
import Data.Array.Accelerate.Tuple
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.CodeGen
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Analysis.Launch
import Data.Array.Accelerate.CUDA.Persistent                    as KT
import qualified Data.Array.Accelerate.CUDA.FullList            as FL
import qualified Data.Array.Accelerate.CUDA.Debug               as D

-- libraries
import Numeric
import Prelude                                                  hiding ( exp, scanl, scanr )
import Control.Applicative                                      hiding ( Const )
import Control.Exception
import Control.Monad
import Control.Monad.Reader                                     ( asks )
import Control.Monad.State                                      ( gets )
import Control.Monad.Trans                                      ( liftIO, MonadIO )
import Control.Concurrent
import Crypto.Hash.MD5                                          ( hashlazy )
import Data.List                                                ( intercalate )
import Data.Maybe
import Data.Monoid
import System.Directory
import System.Exit                                              ( ExitCode(..) )
import System.FilePath
import System.IO
import System.IO.Error
import System.IO.Unsafe
import System.Process
import Text.PrettyPrint.Mainland                                ( ppr, renderCompact, displayLazyText )
import qualified Data.ByteString                                as B
import qualified Data.Text.Lazy                                 as T
import qualified Data.Text.Lazy.IO                              as T
import qualified Data.Text.Lazy.Encoding                        as T
import qualified Control.Concurrent.MSem                        as Q
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Foreign.CUDA.Analysis                          as CUDA

import GHC.Conc                                                 ( getNumProcessors )

#ifdef VERSION_unix
import System.Posix.Process
#else
import System.Win32.Process
#endif

#ifndef SIZEOF_HSINT
import Foreign.Storable
#endif

import Paths_accelerate_cuda                                    ( getDataDir )


-- Keep track of which kernels have been linked into which contexts. We use the
-- context as a lookup key, which requires equality.
--
instance Eq CUDA.Context where
  CUDA.Context c1 == CUDA.Context c2 = c1 == c2


-- | Initiate code generation, compilation, and data transfer for an array
-- expression. The returned array computation is annotated so to be suitable for
-- execution in the CUDA environment. This includes:
--
--   * list of array variables embedded within scalar expressions
--
--   * kernel object(s) required to executed the kernel
--
compileAcc :: Acc a -> CIO (ExecAcc a)
compileAcc = prepareOpenAcc

compileAfun :: Afun f -> CIO (ExecAfun f)
compileAfun = prepareOpenAfun


prepareOpenAfun :: OpenAfun aenv f -> CIO (PreOpenAfun ExecOpenAcc aenv f)
prepareOpenAfun (Alam l)  = Alam  <$> prepareOpenAfun l
prepareOpenAfun (Abody b) = Abody <$> prepareOpenAcc b


prepareOpenAcc :: OpenAcc aenv a -> CIO (ExecOpenAcc aenv a)
prepareOpenAcc rootAcc = traverseAcc rootAcc
  where
    -- Traverse an open array expression in depth-first order
    --
    -- The applicative combinators are used to gloss over that we are passing
    -- around the AST nodes together with a set of free variable indices that
    -- are merged at every step.
    --
    traverseAcc :: forall aenv arrs. OpenAcc aenv arrs -> CIO (ExecOpenAcc aenv arrs)
    traverseAcc acc@(OpenAcc pacc) =
      case pacc of
        -- Environment and control flow
        Avar ix                 -> node $ pure (Avar ix)
        Alet a b                -> node . pure =<< Alet         <$> traverseAcc a <*> traverseAcc b
        Apply f a               -> node . pure =<< Apply        <$> compileAfun f <*> traverseAcc a
        Acond p t e             -> node =<< liftA3 Acond        <$> travE p <*> travA t <*> travA e
        Atuple tup              -> node =<< liftA Atuple        <$> travAtup tup
        Aprj ix tup             -> node =<< liftA (Aprj ix)     <$> travA    tup

        -- Array injection
        Unit e                  -> node =<< liftA  Unit         <$> travE e
        Use arrs                -> use (arrays (undefined::arrs)) arrs >> node (pure $ Use arrs)

        -- Index space transforms
        Reshape s a             -> node =<< liftA2 Reshape              <$> travE s <*> travA a
        Replicate slix e a      -> exec =<< liftA2 (Replicate slix)     <$> travE e <*> travA a
        Slice slix a e          -> exec =<< liftA2 (Slice slix)         <$> travA a <*> travE e
        Backpermute e f a       -> exec =<< liftA3 Backpermute          <$> travE e <*> travF f <*> travD a

        -- Producers
        Generate e f            -> exec =<< liftA2 Generate             <$> travE e <*> travF f
        Map f a                 -> exec =<< liftA2 Map                  <$> travF f <*> travD a
        ZipWith f a b           -> exec =<< liftA3 ZipWith              <$> travF f <*> travD a <*> travD b
        Transform e p f a       -> exec =<< liftA4 Transform            <$> travE e <*> travF p <*> travF f <*> travD a

        -- Consumers
        Fold f z a              -> exec =<< liftA3 Fold                 <$> travF f <*> travE z <*> travD a
        Fold1 f a               -> exec =<< liftA2 Fold1                <$> travF f <*> travD a
        FoldSeg f e a s         -> exec =<< liftA4 FoldSeg              <$> travF f <*> travE e <*> travD a <*> travD s
        Fold1Seg f a s          -> exec =<< liftA3 Fold1Seg             <$> travF f <*> travD a <*> travD s
        Scanl f e a             -> exec =<< liftA3 Scanl                <$> travF f <*> travE e <*> travD a
        Scanl' f e a            -> exec =<< liftA3 Scanl'               <$> travF f <*> travE e <*> travD a
        Scanl1 f a              -> exec =<< liftA2 Scanl1               <$> travF f <*> travD a
        Scanr f e a             -> exec =<< liftA3 Scanr                <$> travF f <*> travE e <*> travD a
        Scanr' f e a            -> exec =<< liftA3 Scanr'               <$> travF f <*> travE e <*> travD a
        Scanr1 f a              -> exec =<< liftA2 Scanr1               <$> travF f <*> travD a
        Permute f d g a         -> exec =<< liftA4 Permute              <$> travF f <*> travA d <*> travF g <*> travD a
        Stencil f b a           -> exec =<< liftA2 (flip Stencil b)     <$> travF f <*> travA a
        Stencil2 f b1 a1 b2 a2  -> exec =<< liftA3 stencil2             <$> travF f <*> travA a1 <*> travA a2
          where stencil2 f' a1' a2' = Stencil2 f' b1 a1' b2 a2'

      where
        use :: ArraysR a -> a -> CIO ()
        use ArraysRunit         ()       = return ()
        use ArraysRarray        arr      = useArray arr
        use (ArraysRpair r1 r2) (a1, a2) = use r1 a1 >> use r2 a2

        exec :: (Gamma aenv, PreOpenAcc ExecOpenAcc aenv arrs) -> CIO (ExecOpenAcc aenv arrs)
        exec (aenv, eacc) = do
          kernel <- build acc aenv
          return $! ExecAcc (fullOfList kernel) aenv eacc

        node :: (Gamma aenv', PreOpenAcc ExecOpenAcc aenv' arrs') -> CIO (ExecOpenAcc aenv' arrs')
        node = fmap snd . wrap

        wrap :: (Gamma aenv', PreOpenAcc ExecOpenAcc aenv' arrs') -> CIO (Gamma aenv', ExecOpenAcc aenv' arrs')
        wrap = return . liftA (ExecAcc noKernel mempty)

        travA :: OpenAcc aenv' a' -> CIO (Gamma aenv', ExecOpenAcc aenv' a')
        travA a = pure <$> traverseAcc a

        travD :: (Shape sh, Elt e) => OpenAcc aenv (Array sh e) -> CIO (Gamma aenv, ExecOpenAcc aenv (Array sh e))
        travD (OpenAcc delayed) =
          case delayed of
            Avar ix             -> wrap (freevar ix, Avar ix)
            Map f a             -> wrap =<< liftA2 Map                  <$> travF f <*> travD a
            Generate e f        -> wrap =<< liftA2 Generate             <$> travE e <*> travF f
            Backpermute e f a   -> wrap =<< liftA3 Backpermute          <$> travE e <*> travF f <*> travD a
            Transform e p f a   -> wrap =<< liftA4 Transform            <$> travE e <*> travF p <*> travF f <*> travD a
            _                   -> INTERNAL_ERROR(error) "compile" "expected fused/delayable array"

        travAtup :: Atuple (OpenAcc aenv) a -> CIO (Gamma aenv, Atuple (ExecOpenAcc aenv) a)
        travAtup NilAtup        = return (pure NilAtup)
        travAtup (SnocAtup t a) = liftA2 SnocAtup <$> travAtup t <*> travA a

        travF :: OpenFun env aenv t -> CIO (Gamma aenv, PreOpenFun ExecOpenAcc env aenv t)
        travF (Body b)  = liftA Body <$> travE b
        travF (Lam  f)  = liftA Lam  <$> travF f

        noKernel :: FL.FullList () (AccKernel a)
        noKernel =  FL.FL () (INTERNAL_ERROR(error) "compile" "no kernel module for this node") FL.Nil

        fullOfList :: [a] -> FL.FullList () a
        fullOfList []       = INTERNAL_ERROR(error) "fullList" "empty list"
        fullOfList [x]      = FL.singleton () x
        fullOfList (x:xs)   = FL.cons () x (fullOfList xs)

    -- Traverse a scalar expression
    --
    travE :: OpenExp env aenv e
          -> CIO (Gamma aenv, PreOpenExp ExecOpenAcc env aenv e)
    travE exp =
      case exp of
        Var ix                  -> return $ pure (Var ix)
        Const c                 -> return $ pure (Const c)
        PrimConst c             -> return $ pure (PrimConst c)
        IndexAny                -> return $ pure IndexAny
        IndexNil                -> return $ pure IndexNil
        --
        Let a b                 -> liftA2 Let                   <$> travE a <*> travE b
        IndexCons t h           -> liftA2 IndexCons             <$> travE t <*> travE h
        IndexHead h             -> liftA  IndexHead             <$> travE h
        IndexTail t             -> liftA  IndexTail             <$> travE t
        IndexSlice slix x s     -> liftA2 (IndexSlice slix)     <$> travE x <*> travE s
        IndexFull slix x s      -> liftA2 (IndexFull slix)      <$> travE x <*> travE s
        ToIndex s i             -> liftA2 ToIndex               <$> travE s <*> travE i
        FromIndex s i           -> liftA2 FromIndex             <$> travE s <*> travE i
        Tuple t                 -> liftA  Tuple                 <$> travT t
        Prj ix e                -> liftA  (Prj ix)              <$> travE e
        Cond p t e              -> liftA3 Cond                  <$> travE p <*> travE t <*> travE e
        Iterate n f x           -> liftA2 (Iterate n)           <$> travF f <*> travE x
        PrimApp f e             -> liftA  (PrimApp f)           <$> travE e
        Index a e               -> liftA2 Index                 <$> travA a <*> travE e
        LinearIndex a e         -> liftA2 LinearIndex           <$> travA a <*> travE e
        Shape a                 -> liftA  Shape                 <$> travA a
        ShapeSize e             -> liftA  ShapeSize             <$> travE e
        Intersect x y           -> liftA2 Intersect             <$> travE x <*> travE y
      where
        travA :: (Shape sh, Elt e)
              => OpenAcc aenv (Array sh e) -> CIO (Gamma aenv, ExecOpenAcc aenv (Array sh e))
        travA a = do
          a'    <- traverseAcc a
          return $ (bind a', a')

        travF :: OpenFun env aenv t -> CIO (Gamma aenv, PreOpenFun ExecOpenAcc env aenv t)
        travF (Body b)  = liftA Body <$> travE b
        travF (Lam  f)  = liftA Lam  <$> travF f

        travT :: Tuple (OpenExp env aenv) t
              -> CIO (Gamma aenv, Tuple (PreOpenExp ExecOpenAcc env aenv) t)
        travT NilTup        = return (pure NilTup)
        travT (SnocTup t e) = liftA2 SnocTup <$> travT t <*> travE e

        bind :: (Shape sh, Elt e) => ExecOpenAcc aenv (Array sh e) -> Gamma aenv
        bind (ExecAcc _ _ (Avar ix)) = freevar ix
        bind _                       = INTERNAL_ERROR(error) "bind" "expected array variable"


-- Applicative
-- -----------
--
liftA4 :: Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
liftA4 f a b c d = f <$> a <*> b <*> c <*> d


-- Compilation
-- -----------

-- Generate, compile, and link code to evaluate an array computation. We use
-- 'unsafePerformIO' here to leverage laziness, so that the 'link' function
-- evaluates and blocks on the external compiler only once the compiled object
-- is truly needed.
--
build :: OpenAcc aenv a -> Gamma aenv -> CIO [AccKernel a]
build acc aenv = do
  dev   <- asks deviceProps
  mapM (build1 acc) (codegenAcc dev acc aenv)

build1 :: OpenAcc aenv a -> CUTranslSkel aenv a -> CIO (AccKernel a)
build1 acc code = do
  dev           <- asks deviceProps
  table         <- gets kernelTable
  (entry,key)   <- compile table dev code
  let (cta,blocks,smem) = launchConfig acc dev occ
      (mdl,fun,occ)     = unsafePerformIO $ do
        m <- link table key
        f <- CUDA.getFun m entry
        l <- CUDA.requires f CUDA.MaxKernelThreadsPerBlock
        o <- determineOccupancy acc dev f l
        D.when D.dump_cc (stats entry f o)
        return (m,f,o)
  --
  return $ AccKernel entry fun mdl occ cta smem blocks
  where
    stats name fn occ = do
      regs      <- CUDA.requires fn CUDA.NumRegs
      smem      <- CUDA.requires fn CUDA.SharedSizeBytes
      cmem      <- CUDA.requires fn CUDA.ConstSizeBytes
      lmem      <- CUDA.requires fn CUDA.LocalSizeBytes
      let msg1  = "entry function '" ++ name ++ "' used "
                  ++ shows regs " registers, "  ++ shows smem " bytes smem, "
                  ++ shows lmem " bytes lmem, " ++ shows cmem " bytes cmem"
          msg2  = "multiprocessor occupancy " ++ showFFloat (Just 1) (CUDA.occupancy100 occ) "% : "
                  ++ shows (CUDA.activeThreads occ)      " threads over "
                  ++ shows (CUDA.activeWarps occ)        " warps in "
                  ++ shows (CUDA.activeThreadBlocks occ) " blocks"
      --
      -- make sure kernel/stats are printed together. Use 'intercalate' rather
      -- than 'unlines' to avoid a trailing newline.
      --
      message   $ intercalate "\n" [msg1, "     ... " ++ msg2]


-- Link a compiled binary and update the associated kernel entry in the hash
-- table. This may entail waiting for the external compilation process to
-- complete. If successful, the temporary files are removed.
--
link :: KernelTable -> KernelKey -> IO CUDA.Module
link table key =
  let intErr = INTERNAL_ERROR(error) "link" "missing kernel entry"
  in do
    ctx         <- CUDA.get
    entry       <- fromMaybe intErr `fmap` KT.lookup table key
    case entry of
      CompileProcess cufile done -> do
        -- Wait for the compiler to finish and load the binary object into the
        -- current context.
        --
        -- A forked thread will fill the MVar once the external compilation
        -- process completes, but only the main thread executes kernels. Hence,
        -- only one thread will ever attempt to take the MVar in order to link
        -- the binary object.
        --
        message "waiting for nvcc..."
        takeMVar done
        let cubin       =  replaceExtension cufile ".cubin"
        bin             <- B.readFile cubin
        mdl             <- CUDA.loadData bin

        -- Update hash tables and stash the binary object into the persistent
        -- cache
        --
        KT.insert table key $! KernelObject bin (FL.singleton ctx mdl)
        KT.persist cubin key

        -- Remove temporary build products.
        -- If compiling kernels with debugging symbols, leave the source files
        -- in place so that they can be referenced by 'cuda-gdb'.
        --
        D.unless D.debug_cc $ do
          removeFile      cufile
          removeDirectory (dropFileName cufile)
            `catchIOError` \_ -> return ()      -- directory not empty

        return mdl

      -- If we get a real object back, then this will already be in the
      -- persistent cache, since either it was just read in from there, or we
      -- had to generate new code and the link step above has added it.
      --
      KernelObject bin active
        | Just mdl <- FL.lookup ctx active      -> return mdl
        | otherwise                             -> do
            message "re-linking module for current context"
            mdl                 <- CUDA.loadData bin
            KT.insert table key $! KernelObject bin (FL.cons ctx mdl active)
            return mdl


-- Generate and compile code for a single open array expression
--
compile :: KernelTable -> CUDA.DeviceProperties -> CUTranslSkel aenv a -> CIO (String, KernelKey)
compile table dev cunit = do
  exists        <- isJust `fmap` liftIO (KT.lookup table key)
  unless exists $ do
    message     $  unlines [ show key, T.unpack code ]
    nvcc        <- fromMaybe (error "nvcc: command not found") <$> liftIO (findExecutable "nvcc")
    (file,hdl)  <- openTemporaryFile "dragon.cu"   -- rawr!
    flags       <- compileFlags file
    done        <- liftIO $ do
      message $ "execute: " ++ nvcc ++ " " ++ unwords flags
      T.hPutStr hdl code               `finally`     hClose hdl
      enqueueProcess (proc nvcc flags) `onException` removeFile file
    --
    liftIO $ KT.insert table key (CompileProcess file done)
  --
  return (entry, key)
  where
    entry       = show cunit
    key         = (CUDA.computeCapability dev, hashlazy (T.encodeUtf8 code) )
    code        = displayLazyText . renderCompact $ ppr cunit


-- Determine the appropriate command line flags to pass to the compiler process.
-- This is dependent on the host architecture and device capabilities.
--
compileFlags :: FilePath -> CIO [String]
compileFlags cufile = do
  CUDA.Compute m n      <- CUDA.computeCapability `fmap` asks deviceProps
  ddir                  <- liftIO getDataDir
  return                $  filter (not . null) $
    [ "-I", ddir </> "cubits"
    , "-arch=sm_" ++ show m ++ show n
    , "-cubin"
    , "-o", cufile `replaceExtension` "cubin"
    , if D.mode D.dump_cc  then ""   else "--disable-warnings"
    , if D.mode D.debug_cc then "-G" else "-O3"
    , machine
    , cufile ]
  where
#if SIZEOF_HSINT == 4
    machine     = "-m32"
#elif SIZEOF_HSINT == 8
    machine     = "-m64"
#else
    machine     = case sizeOf (undefined :: Int) of
                    4 -> "-m32"
                    8 -> "-m64"
#endif


-- Open a unique file in the temporary directory used for compilation
-- by-products. The directory will be created if it does not exist.
--
openTemporaryFile :: String -> CIO (FilePath, Handle)
openTemporaryFile template = liftIO $ do
  pid <- getProcessID
  dir <- (</>) <$> getTemporaryDirectory <*> pure ("accelerate-cuda-" ++ show pid)
  createDirectoryIfMissing True dir
  openTempFile dir template

#ifndef VERSION_unix
getProcessID :: ProcessHandle -> IO ProcessId
getProcessID = getProcessId
#endif


-- Worker pool
-- -----------

{-# NOINLINE pool #-}
pool :: Q.MSem Int
pool = unsafePerformIO $ Q.new =<< getNumProcessors

-- Queue a system process to be executed and return an MVar flag that will be
-- filled once the process completes. The task will only be launched once there
-- is a worker available from the pool. This ensures we don't run out of process
-- handles or flood the IO bus, degrading performance.
--
enqueueProcess :: CreateProcess -> IO (MVar ())
enqueueProcess cp = do
  mvar  <- newEmptyMVar
  _     <- forkIO $ do

    -- wait for a worker to become available
    Q.wait pool
    (_,_,_,pid) <- createProcess cp

    -- asynchronously notify the queue when the compiler has completed
    _           <- forkIO $ do finally (waitFor pid) (Q.signal pool)
                               putMVar mvar ()
    return ()
  --
  return mvar


-- Wait for a (compilation) process to finish
--
waitFor :: ProcessHandle -> IO ()
waitFor pid = do
  status <- waitForProcess pid
  case status of
    ExitSuccess   -> return ()
    ExitFailure c -> error $ "nvcc terminated abnormally (" ++ show c ++ ")"


-- Debug
-- -----

{-# INLINE message #-}
message :: MonadIO m => String -> m ()
message msg = trace msg $ return ()

{-# INLINE trace #-}
trace :: MonadIO m => String -> m a -> m a
trace msg next = D.message D.dump_cc ("cc: " ++ msg) >> next

