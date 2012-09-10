{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Compile
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-partable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Compile (

  -- * generate and compile kernels to realise a computation
  compileAcc, compileAfun1

) where

#include "accelerate.h"

-- friends
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Tuple

import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.CodeGen
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Analysis.Launch
import Data.Array.Accelerate.CUDA.FullList              as FL
import Data.Array.Accelerate.CUDA.Persistent            as KT
import qualified Data.Array.Accelerate.CUDA.Debug       as D

-- libraries
import Numeric
import Prelude                                          hiding ( exp, catch )
import Control.Applicative                              hiding ( Const )
import Control.Exception
import Control.Monad
import Control.Monad.Trans
import Crypto.Hash.MD5                                  ( hashlazy )
import Data.Label.PureM
import Data.List
import Data.Maybe
import Data.Monoid
import System.Directory
import System.Exit                                      ( ExitCode(..) )
import System.FilePath
import System.IO
import System.IO.Unsafe
import System.Process
import Text.PrettyPrint.Mainland                        ( ppr, renderCompact, displayLazyText )
import qualified Data.HashSet                           as Set
import qualified Data.ByteString                        as B
import qualified Data.Text.Lazy                         as T
import qualified Data.Text.Lazy.IO                      as T
import qualified Data.Text.Lazy.Encoding                as T
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Analysis                  as CUDA

#ifdef VERSION_unix
import System.Posix.Process
#else
import System.Win32.Process
#endif

import Paths_accelerate_cuda                            ( getDataDir )


-- | Initiate code generation, compilation, and data transfer for an array
-- expression. The returned array computation is annotated so to be suitable for
-- execution in the CUDA environment. This includes:
--
--   * list of array variables embedded within scalar expressions
--   * kernel object(s) required to executed the kernel
--
compileAcc :: Acc a -> CIO (ExecAcc a)
compileAcc acc = prepareAcc acc


compileAfun1 :: Afun (a -> b) -> CIO (ExecAfun (a -> b))
compileAfun1 (Alam (Abody b)) = Alam . Abody <$> prepareAcc b
compileAfun1 _                =
  error "Hope (noun): something that happens to facts when the world refuses to agree"


prepareAcc :: OpenAcc aenv a -> CIO (ExecOpenAcc aenv a)
prepareAcc rootAcc = traverseAcc rootAcc
  where
    -- Traverse an open array expression in depth-first order
    --
    -- The applicative combinators are used to gloss over that we are passing
    -- around the AST nodes together with a set of free variable indices that
    -- are merged at every step.
    --
    traverseAcc :: forall aenv a. OpenAcc aenv a -> CIO (ExecOpenAcc aenv a)
    traverseAcc acc@(OpenAcc pacc) = do

      let exec :: (AccBindings aenv, PreOpenAcc ExecOpenAcc aenv a) -> CIO (ExecOpenAcc aenv a)
          exec (var, eacc) = do
            kernel      <- build acc var
            return      $  ExecAcc (FL.singleton () kernel) var eacc

          node :: (AccBindings aenv, PreOpenAcc ExecOpenAcc aenv a) -> CIO (ExecOpenAcc aenv a)
          node (_, eacc) = return $ ExecAcc noKernel mempty eacc

      case pacc of
        --
        -- Environment manipulations
        --
        Avar ix                 -> node $ pure (Avar ix)

        --
        -- Let bindings
        --
        Alet a b                -> node . pure =<< Alet         <$> traverseAcc a  <*> traverseAcc b
        Apply f a               -> node . pure =<< Apply        <$> compileAfun1 f <*> traverseAcc a
        Acond p t e             -> node =<< liftA3 Acond        <$> travE p <*> travA t <*> travA e

        --
        -- Tuples
        --
        Atuple tup              -> node =<< liftA Atuple        <$> travAtup tup
        Aprj ix tup             -> node =<< liftA (Aprj ix)     <$> travA    tup

        --
        -- Array injection
        --
        Use arrs                -> use (arrays (undefined::a)) arrs >> node (pure $ Use arrs)
          where
            use :: ArraysR a' -> a' -> CIO ()
            use ArraysRunit         ()       = return ()
            use ArraysRarray        arr      = useArray arr
            use (ArraysRpair r1 r2) (a1, a2) = use r1 a1 >> use r2 a2

        --
        -- Computation nodes
        --
        Reshape s a             -> node =<< liftA2 Reshape              <$> travE s <*> travA a
        Unit e                  -> node =<< liftA  Unit                 <$> travE e
        Generate e f            -> exec =<< liftA2 Generate             <$> travE e <*> travF f
        Transform e p f a       -> exec =<< liftA4 Transform            <$> travE e <*> travF p <*> travF f <*> travA a
        Replicate slix e a      -> exec =<< liftA2 (Replicate slix)     <$> travE e <*> travA a
        Index slix a e          -> exec =<< liftA2 (Index slix)         <$> travA a <*> travE e
        Map f a                 -> exec =<< liftA2 Map                  <$> travF f <*> travA a
        ZipWith f a b           -> exec =<< liftA3 ZipWith              <$> travF f <*> travA a <*> travA b
        Fold f z a              -> exec =<< liftA3 Fold                 <$> travF f <*> travE z <*> travA a
        Fold1 f a               -> exec =<< liftA2 Fold1                <$> travF f <*> travA a
        FoldSeg f e a s         -> exec =<< liftA4 FoldSeg              <$> travF f <*> travE e <*> travA a <*> travA (segments s)
        Fold1Seg f a s          -> exec =<< liftA3 Fold1Seg             <$> travF f <*> travA a <*> travA (segments s)
        Permute f a g b         -> exec =<< liftA4 Permute              <$> travF f <*> travA a <*> travF g <*> travA b
        Backpermute e f a       -> exec =<< liftA3 Backpermute          <$> travE e <*> travF f <*> travA a
        Stencil f b a           -> exec =<< liftA2 (flip Stencil b)     <$> travF f <*> travA a
        Stencil2 f b1 a1 b2 a2  -> exec =<< liftA3 stencil2             <$> travF f <*> travA a1 <*> travA a2
          where stencil2 f' a1' a2' = Stencil2 f' b1 a1' b2 a2'

        -- TODO: write helper functions to clean these up
        Scanl f e a -> do
          ExecAcc (FL _ scan _) var eacc  <- exec =<< liftA3 Scanl <$> travF f <*> travE e <*> travA a
          add           <- build (OpenAcc (Fold1 f mat)) var
          return        $  ExecAcc (cons () add $ FL.singleton () scan) var eacc

        Scanl' f e a -> do
          ExecAcc (FL _ scan _) var eacc  <- exec =<< liftA3 Scanl' <$> travF f <*> travE e <*> travA a
          add           <- build (OpenAcc (Fold1 f mat)) var
          return        $  ExecAcc (cons () (retag add) $ FL.singleton () scan) var eacc

        Scanl1 f a -> do
          ExecAcc (FL _ scan1 _) var eacc <- exec =<< liftA2 Scanl1 <$> travF f <*> travA a
          add           <- build (OpenAcc (Fold1 f mat)) var
          return        $  ExecAcc (cons () add $ FL.singleton () scan1) var eacc

        Scanr f e a -> do
          ExecAcc (FL _ scan _) var eacc  <- exec =<< liftA3 Scanr <$> travF f <*> travE e <*> travA a
          add           <- build (OpenAcc (Fold1 f mat)) var
          return        $  ExecAcc (cons () add $ FL.singleton () scan) var eacc

        Scanr' f e a -> do
          ExecAcc (FL _ scan _) var eacc  <- exec =<< liftA3 Scanr' <$> travF f <*> travE e <*> travA a
          add           <- build (OpenAcc (Fold1 f mat)) var
          return        $  ExecAcc (cons () (retag add) $ FL.singleton () scan) var eacc

        Scanr1 f a -> do
          ExecAcc (FL _ scan1 _) var eacc <- exec =<< liftA2 Scanr1 <$> travF f <*> travA a
          add           <- build (OpenAcc (Fold1 f mat)) var
          return        $  ExecAcc (cons () add $ FL.singleton () scan1) var eacc

      where
        travA :: OpenAcc aenv' a' -> CIO (AccBindings aenv', ExecOpenAcc aenv' a')
        travA a = pure <$> traverseAcc a

        travAtup :: Atuple (OpenAcc aenv') a' -> CIO (AccBindings aenv', Atuple (ExecOpenAcc aenv') a')
        travAtup NilAtup        = return (pure NilAtup)
        travAtup (SnocAtup t a) = liftA2 SnocAtup <$> travAtup t <*> travA a

        travF :: OpenFun env aenv t -> CIO (AccBindings aenv, PreOpenFun ExecOpenAcc env aenv t)
        travF (Body b)  = liftA Body <$> travE b
        travF (Lam  f)  = liftA Lam  <$> travF f

        segments :: forall i. (Elt i, IsIntegral i)
                 => OpenAcc aenv (Segments i) -> OpenAcc aenv (Segments i)
        segments = OpenAcc . Scanl plus (Const (fromElt (0::i)))

        plus :: (Elt i, IsIntegral i) => PreOpenFun OpenAcc () aenv (i -> i -> i)
        plus = Lam (Lam (Body (PrimAdd numType
                              `PrimApp`
                              Tuple (NilTup `SnocTup` Var (SuccIdx ZeroIdx)
                                            `SnocTup` Var ZeroIdx))))

        mat :: Elt e => OpenAcc aenv (Array DIM2 e)
        mat = OpenAcc $ Use ((), Array (((),0),0) undefined)

        noKernel :: FullList () (AccKernel a)
        noKernel =  FL () (INTERNAL_ERROR(error) "compile" "no kernel module for this node") Nil

    -- Traverse a scalar expression
    --
    travE :: OpenExp env aenv e
          -> CIO (AccBindings aenv, PreOpenExp ExecOpenAcc env aenv e)
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
        IndexScalar a e         -> liftA2 IndexScalar           <$> travA a <*> travE e
        Shape a                 -> liftA  Shape                 <$> travA a
        ShapeSize e             -> liftA  ShapeSize             <$> travE e
        Intersect x y           -> liftA2 Intersect             <$> travE x <*> travE y
      where
        travA :: (Shape sh, Elt e)
              => OpenAcc aenv (Array sh e) -> CIO (AccBindings aenv, ExecOpenAcc aenv (Array sh e))
        travA a = do
          a'    <- traverseAcc a
          return $ (bind a', a')

        travF :: OpenFun env aenv t -> CIO (AccBindings aenv, PreOpenFun ExecOpenAcc env aenv t)
        travF (Body b)  = liftA Body <$> travE b
        travF (Lam  f)  = liftA Lam  <$> travF f

        travT :: Tuple (OpenExp env aenv) t
              -> CIO (AccBindings aenv, Tuple (PreOpenExp ExecOpenAcc env aenv) t)
        travT NilTup        = return (pure NilTup)
        travT (SnocTup t e) = liftA2 SnocTup <$> travT t <*> travE e

        bind :: (Shape sh, Elt e) => ExecOpenAcc aenv (Array sh e) -> AccBindings aenv
        bind (ExecAcc _ _ (Avar ix)) = AccBindings ( Set.singleton (ArrayVar ix) )
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
build :: OpenAcc aenv a -> AccBindings aenv -> CIO (AccKernel a)
build acc fvar = do
  dev           <- gets deviceProps
  table         <- gets kernelTable
  (entry,key)   <- compile table dev acc fvar
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
      -- make sure kernel/stats are printed together
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
      CompileProcess cufile pid -> do
        -- Wait for the compiler to finish and load the binary object into the
        -- current context
        --
        message "waiting for nvcc..."
        waitFor pid
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
            `catch` \(_ :: IOError) -> return ()        -- directory not empty

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
compile :: KernelTable
        -> CUDA.DeviceProperties
        -> OpenAcc aenv a
        -> AccBindings aenv
        -> CIO (String, KernelKey)
compile table dev acc fvar = do
  exists        <- isJust `fmap` liftIO (KT.lookup table key)
  unless exists $ do
    message     $  unlines [ show key, T.unpack code ]
    nvcc        <- fromMaybe (error "nvcc: command not found") <$> liftIO (findExecutable "nvcc")
    (file,hdl)  <- openTemporaryFile "dragon.cu"   -- rawr!
    flags       <- compileFlags file
    (_,_,_,pid) <- liftIO $ do
      message $ "execute: " ++ nvcc ++ " " ++ unwords flags
      T.hPutStr hdl code              `finally`     hClose hdl
      createProcess (proc nvcc flags) `onException` removeFile file
    --
    liftIO $ KT.insert table key (CompileProcess file pid)
  --
  return (entry, key)
  where
    cunit       = codegenAcc dev acc fvar
    entry       = show cunit
    key         = (CUDA.computeCapability dev, hashlazy (T.encodeUtf8 code) )
    code        = displayLazyText . renderCompact $ ppr cunit


-- Wait for the compilation process to finish
--
waitFor :: ProcessHandle -> IO ()
waitFor pid = do
  status <- waitForProcess pid
  case status of
    ExitSuccess   -> return ()
    ExitFailure c -> error $ "nvcc terminated abnormally (" ++ show c ++ ")"


-- Determine the appropriate command line flags to pass to the compiler process.
-- This is dependent on the host architecture and device capabilities.
--
compileFlags :: FilePath -> CIO [String]
compileFlags cufile = do
  CUDA.Compute m n      <- CUDA.computeCapability `fmap` gets deviceProps
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

-- Debug
-- -----

{-# INLINE message #-}
message :: MonadIO m => String -> m ()
message msg = trace msg $ return ()

{-# INLINE trace #-}
trace :: MonadIO m => String -> m a -> m a
trace msg next = D.message D.dump_cc ("cc: " ++ msg) >> next

