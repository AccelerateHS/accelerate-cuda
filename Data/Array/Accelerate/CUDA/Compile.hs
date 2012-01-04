{-# LANGUAGE CPP, GADTs, TupleSections, ScopedTypeVariables #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Compile
-- Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Manuel M T Chakravarty <chak@cse.unsw.edu.au>
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
import qualified Data.Array.Accelerate.CUDA.Debug       as D

-- libraries
import Numeric
import Prelude                                          hiding ( exp, catch )
import Control.Applicative                              hiding ( Const )
import Blaze.ByteString.Builder
import Blaze.ByteString.Builder.Char8
import Control.Exception
import Control.Monad
import Control.Monad.Trans
import Crypto.Hash.MD5                                  ( hashlazy )
import Data.Label.PureM
import Data.Maybe
import Data.Monoid
import Foreign.Storable
import System.Directory
import System.Exit                                      ( ExitCode(..) )
import System.FilePath
import System.IO
import System.IO.Unsafe
import System.Process
import Text.PrettyPrint.Mainland                        ( RDoc(..), ppr, renderCompact )
import Data.ByteString.Internal                         ( w2c )
import qualified Data.ByteString.Lazy                   as L
import qualified Data.HashTable.IO                      as Hash
import qualified Foreign.CUDA.Driver                    as CUDA
import qualified Foreign.CUDA.Analysis                  as CUDA

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
prepareAcc rootAcc = travA rootAcc
  where
    -- Traverse an open array expression in depth-first order
    --
    travA :: OpenAcc aenv a -> CIO (ExecOpenAcc aenv a)
    travA acc@(OpenAcc pacc) =
      case pacc of

        -- Environment manipulations
        --
        Avar ix -> return $ node (Avar ix)

        -- Let bindings
        --
        Alet2 a b -> do
          a'            <- travA a
          b'            <- travA b
          return $ node (Alet2 a' b')

        Alet a b  -> do
          a'            <- travA a
          b'            <- travA b
          return $ node (Alet a' b')

        Apply (Alam (Abody b)) a -> do
          a'            <- travA a
          b'            <- travA b
          return $ node (Apply (Alam (Abody b')) a')
        Apply _                _ -> error "I made you a cookie, but I eated it"

        PairArrays arr1 arr2 -> do
          arr1'         <- travA arr1
          arr2'         <- travA arr2
          return $ node (PairArrays arr1' arr2')

        Acond c t e -> do
          (c', _)       <- travE c []
          t'            <- travA t
          e'            <- travA e
          return $ node (Acond c' t' e')

        -- Array injection
        --
        Use arr@(Array _ _) -> do
          useArray arr
          return $ node (Use arr)

        -- Computation nodes
        --
        Reshape sh a -> do
          (sh', _)      <- travE sh []
          a'            <- travA a
          return $ node (Reshape sh' a')

        Unit e  -> do
          (e', _)       <- travE e []
          return $ node (Unit e')

        Generate e f -> do
          (e', _)       <- travE e []
          (f', var1)    <- travF f []
          kernel        <- build acc var1
          return $ exec kernel var1 (Generate e' f')

        Replicate slix e a -> do
          (e', _)       <- travE e []
          a'            <- travA a
          kernel        <- build acc []
          return $ exec kernel [] (Replicate slix e' a')

        Index slix a e -> do
          a'            <- travA a
          (e', _)       <- travE e []
          kernel        <- build acc []
          return $ exec kernel [] (Index slix a' e')

        Map f a -> do
          (f', var1)    <- travF f []
          a'            <- travA a
          kernel        <- build acc var1
          return $ exec kernel var1 (Map f' a')

        ZipWith f a b -> do
          (f', var1)    <- travF f []
          a'            <- travA a
          b'            <- travA b
          kernel        <- build acc var1
          return $ exec kernel var1 (ZipWith f' a' b')

        Fold f e a -> do
          (f', var1)    <- travF f []
          (e', var2)    <- travE e var1
          a'            <- travA a
          kernel        <- build acc var2
          return $ exec kernel var2 (Fold f' e' a')

        Fold1 f a -> do
          (f', var1)    <- travF f []
          a'            <- travA a
          kernel        <- build acc var1
          return $ exec kernel var1 (Fold1 f' a')

        FoldSeg f e a s -> do
          (f', var1)    <- travF f []
          (e', var2)    <- travE e var1
          a'            <- travA a
          s'            <- travA (segments s)
          kernel        <- build acc var2
          return $ exec kernel var2 (FoldSeg f' e' a' s')

        Fold1Seg f a s -> do
          (f', var1)    <- travF f []
          a'            <- travA a
          s'            <- travA (segments s)
          kernel        <- build acc var1
          return $ exec kernel var1 (Fold1Seg f' a' s')

        Scanl f e a -> do
          (f', var1)    <- travF f []
          (e', var2)    <- travE e var1
          a'            <- travA a
          add           <- build (OpenAcc (Fold1 f mat)) var2
          scan1         <- build (OpenAcc (Scanl1 f a))  var2
          scan          <- build acc var2
          return $ ExecAcc (FL add (scan1 :> scan :> Nil)) var2 (Scanl f' e' a')

        Scanl' f e a -> do
          (f', var1)    <- travF f []
          (e', var2)    <- travE e var1
          a'            <- travA a
          add           <- build (OpenAcc (Fold1 f mat)) var2
          scan1         <- build (OpenAcc (Scanl1 f a))  var2
          scan          <- build acc var2
          return $ ExecAcc (FL (retag add) (retag scan1 :> scan :> Nil)) var2 (Scanl' f' e' a')

        Scanl1 f a -> do
          (f', var1)    <- travF f []
          a'            <- travA a
          add           <- build (OpenAcc (Fold1 f mat)) var1
          scan1         <- build acc var1
          return $ ExecAcc (FL add (scan1 :> Nil)) var1 (Scanl1 f' a')

        Scanr f e a -> do
          (f', var1)    <- travF f []
          (e', var2)    <- travE e var1
          a'            <- travA a
          add           <- build (OpenAcc (Fold1 f mat)) var2
          scan1         <- build (OpenAcc (Scanr1 f a))  var2
          scan          <- build acc var2
          return $ ExecAcc (FL add (scan1 :> scan :> Nil)) var2 (Scanr f' e' a')

        Scanr' f e a -> do
          (f', var1)    <- travF f []
          (e', var2)    <- travE e var1
          a'            <- travA a
          add           <- build (OpenAcc (Fold1 f mat)) var2
          scan1         <- build (OpenAcc (Scanr1 f a))  var2
          scan          <- build acc var2
          return $ ExecAcc (FL (retag add) (retag scan1 :> scan :> Nil)) var2 (Scanr' f' e' a')

        Scanr1 f a -> do
          (f', var1)    <- travF f []
          a'            <- travA a
          add           <- build (OpenAcc (Fold1 f mat)) var1
          scan1         <- build acc var1
          return $ ExecAcc (FL add (scan1 :> Nil)) var1 (Scanr1 f' a')

        Permute f a g b -> do
          (f', var1)    <- travF f []
          (g', var2)    <- travF g var1
          a'            <- travA a
          b'            <- travA b
          kernel        <- build acc var2
          return $ exec kernel var2 (Permute f' a' g' b')

        Backpermute e f a -> do
          (e', _)       <- travE e []
          (f', var2)    <- travF f []
          a'            <- travA a
          kernel        <- build acc var2
          return $ exec kernel var2 (Backpermute e' f' a')

        Stencil f b a -> do
          (f', var1)    <- travF f []
          a'            <- travA a
          kernel        <- build acc var1
          return $ exec kernel var1 (Stencil f' b a')

        Stencil2 f b1 a1 b2 a2 -> do
          (f', var1)    <- travF f []
          a1'           <- travA a1
          a2'           <- travA a2
          kernel        <- build acc var1
          return $ exec kernel var1 (Stencil2 f' b1 a1' b2 a2')


    -- Traverse a scalar expression
    --
    travE :: OpenExp env aenv e
          -> [AccBinding aenv]
          -> CIO (PreOpenExp ExecOpenAcc env aenv e, [AccBinding aenv])
    travE exp vars =
      case exp of
        Let _ _         -> INTERNAL_ERROR(error) "prepareAcc" "Let: not implemented yet"
        Var ix          -> return (Var ix, vars)
        Const c         -> return (Const c, vars)
        PrimConst c     -> return (PrimConst c, vars)
        IndexAny        -> return (IndexAny, vars)
        IndexNil        -> return (IndexNil, vars)
        IndexCons ix i  -> do
          (ix', var1) <- travE ix vars
          (i',  var2) <- travE i  var1
          return (IndexCons ix' i', var2)

        IndexHead ix    -> do
          (ix', var1) <- travE ix vars
          return (IndexHead ix', var1)

        IndexTail ix    -> do
          (ix', var1) <- travE ix vars
          return (IndexTail ix', var1)

        Tuple t         -> do
          (t', var1) <- travT t vars
          return (Tuple t', var1)

        Prj idx e       -> do
          (e', var1) <- travE e vars
          return (Prj idx e', var1)

        Cond p t e      -> do
          (p', var1) <- travE p vars
          (t', var2) <- travE t var1
          (e', var3) <- travE e var2
          return (Cond p' t' e', var3)

        PrimApp f e     -> do
          (e', var1) <- travE e vars
          return (PrimApp f e', var1)

        IndexScalar a e -> do
          a'         <- travA a
          (e', var2) <- travE e vars
          return (IndexScalar a' e', bind a' `cons` var2)

        Shape a         -> do
          a' <- travA a
          return (Shape a', bind a' `cons` vars)

        ShapeSize e     -> do
          (e', var1) <- travE e vars
          return (ShapeSize e', var1)


    travT :: Tuple (OpenExp env aenv) t
          -> [AccBinding aenv]
          -> CIO (Tuple (PreOpenExp ExecOpenAcc env aenv) t, [AccBinding aenv])
    travT NilTup        vars = return (NilTup, vars)
    travT (SnocTup t e) vars = do
      (e', var1) <- travE e vars
      (t', var2) <- travT t var1
      return (SnocTup t' e', var2)

    travF :: OpenFun env aenv t
          -> [AccBinding aenv]
          -> CIO (PreOpenFun ExecOpenAcc env aenv t, [AccBinding aenv])
    travF (Body b) vars = do
      (b', var1) <- travE b vars
      return (Body b', var1)
    travF (Lam  f) vars = do
      (f', var1) <- travF f vars
      return (Lam f', var1)


    -- Auxiliary
    --
    segments :: OpenAcc aenv Segments -> OpenAcc aenv Segments
    segments = OpenAcc . Scanl plus (Const ((),0))

    plus :: PreOpenFun OpenAcc () aenv (Int -> Int -> Int)
    plus = Lam (Lam (Body (PrimAdd numType
                          `PrimApp`
                          Tuple (NilTup `SnocTup` Var (SuccIdx ZeroIdx)
                                        `SnocTup` Var ZeroIdx))))

    mat :: Elt a => OpenAcc aenv (Array DIM2 a)
    mat = OpenAcc $ Use (Array (((),0),0) undefined)

    node :: PreOpenAcc ExecOpenAcc aenv a -> ExecOpenAcc aenv a
    node = ExecAcc noKernel []

    exec :: AccKernel a -> [AccBinding aenv] -> PreOpenAcc ExecOpenAcc aenv a -> ExecOpenAcc aenv a
    exec k = ExecAcc (FL k Nil)

    noKernel :: FullList (AccKernel a)
    noKernel =  FL (INTERNAL_ERROR(error) "compile" "no kernel module for this node") Nil

    cons :: AccBinding aenv -> [AccBinding aenv] -> [AccBinding aenv]
    cons x xs | x `notElem` xs = x : xs
              | otherwise      = xs

    bind :: (Shape sh, Elt e) => ExecOpenAcc aenv (Array sh e) -> AccBinding aenv
    bind (ExecAcc _ _ (Avar ix)) = ArrayVar ix
    bind _                       = INTERNAL_ERROR(error) "bind" "expected array variable"


-- Compilation
-- -----------

-- Initiate compilation and provide a closure to later link the compiled module
-- when it is required.
--
build :: OpenAcc aenv a -> [AccBinding aenv] -> CIO (AccKernel a)
build acc fvar = do
  dev           <- gets deviceProps
  table         <- gets kernelTable
  (entry,key)   <- compile table dev acc fvar
  let (mdl,fun,occ) = unsafePerformIO $ do
        m <- link table key
        f <- CUDA.getFun m entry
        l <- CUDA.requires f CUDA.MaxKernelThreadsPerBlock
        o <- determineOccupancy acc dev f l
        D.when D.dump_cc (stats entry f o)
        return (m,f,o)
  --
  return $ Kernel entry mdl fun occ (launchConfig acc dev occ)
  where
    stats name fn occ = do
      regs      <- CUDA.requires fn CUDA.NumRegs
      smem      <- CUDA.requires fn CUDA.SharedSizeBytes
      cmem      <- CUDA.requires fn CUDA.ConstSizeBytes
      lmem      <- CUDA.requires fn CUDA.LocalSizeBytes
      message   $ "entry function '" ++ name ++ "' used "
        ++ shows regs " registers, "  ++ shows smem " bytes smem, "
        ++ shows lmem " bytes lmem, " ++ shows cmem " bytes cmem"
      message   $ "multiprocessor occupancy " ++ showFFloat (Just 1) (CUDA.occupancy100 occ) "% : "
        ++ shows (CUDA.activeThreads occ)       " threads over "
        ++ shows (CUDA.activeWarps occ)         " warps in "
        ++ shows (CUDA.activeThreadBlocks occ)  " blocks"


-- Link a compiled binary and update the associated kernel entry in the hash
-- table. This may entail waiting for the external compilation process to
-- complete. If successfully, the temporary files are removed.
--
link :: KernelTable -> KernelKey -> IO CUDA.Module
link table key =
  let intErr = INTERNAL_ERROR(error) "link" "missing kernel entry"
  in do
    (KernelEntry cufile stat) <- fromMaybe intErr `fmap` Hash.lookup table key
    case stat of
      Right mdl -> return mdl
      Left  pid -> do
        -- wait for compiler to finish and load binary object
        --
        waitFor pid
        mdl <- CUDA.loadFile (replaceExtension cufile ".cubin")

        -- remove build products
        --
        removeFile      cufile
        removeFile      (replaceExtension cufile ".cubin")
        removeDirectory (dropFileName cufile)
          `catch` \(_ :: IOError) -> return ()          -- directory not empty

        -- update hash table
        --
        Hash.insert table key (KernelEntry cufile (Right mdl))
        return mdl


-- Generate and compile code for a single open array expression
--
compile :: KernelTable
        -> CUDA.DeviceProperties
        -> OpenAcc aenv a
        -> [AccBinding aenv]
        -> CIO (String, KernelKey)
compile table dev acc fvar = do
  exists        <- isJust `fmap` liftIO (Hash.lookup table key)
  unless exists $ do
    message     $  unlines [ show key, map w2c (L.unpack code) ]
    nvcc        <- fromMaybe (error "nvcc: command not found") <$> liftIO (findExecutable "nvcc")
    (file,hdl)  <- openOutputFile "dragon.cu"   -- rawr!
    flags       <- compileFlags file
    (_,_,_,pid) <- liftIO $ do
      L.hPut hdl code                 `finally`     hClose hdl
      createProcess (proc nvcc flags) `onException` removeFile file
    --
    liftIO $ Hash.insert table key (KernelEntry file (Left pid))
  --
  return (entry, key)
  where
    cunit       = codegenAcc dev acc fvar
    entry       = show cunit
    key         = hashlazy code
    code        = toLazyByteString
                . layout . renderCompact $ ppr cunit
    --
    layout (RText _ s next)     = fromString s  `mappend` layout next
    layout (RChar c   next)     = fromChar c    `mappend` layout next
    layout (RLine _   next)     = fromChar '\n' `mappend` layout next   -- no indenting
    layout (RPos _    next)     = layout next                           -- no line markers
    layout REmpty               = mempty                                -- done


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
  arch <- CUDA.computeCapability `fmap` gets deviceProps
  ddir <- liftIO getDataDir
  return $ filter (not . null) $
    [ "-I", ddir </> "cubits"
    , "--compiler-options", "-fno-strict-aliasing"
    , "-arch=sm_" ++ show (round (arch * 10) :: Int)
    , "-cubin"
    , "-o", cufile `replaceExtension` "cubin"
    , if D.mode D.verbose then ""   else "--disable-warnings"
    , if D.mode D.debug   then "-G" else "-O2"
    , machine
    , cufile ]
  where
    wordSize                    = sizeOf (undefined::Int)
    machine | wordSize == 4     = "-m32"
            | wordSize == 8     = "-m64"
            | otherwise         = error "recreational scolding?"


-- Open a unique file in the temporary directory used for compilation
-- by-products. The directory will be created if it does not exist.
--
openOutputFile :: String -> CIO (FilePath, Handle)
openOutputFile template = liftIO $ do
#ifdef ACCELERATE_CUDA_PERSISTENT_CACHE
  dir <- (</>) <$> getDataDir            <*> pure "cache"
#else
  dir <- (</>) <$> getTemporaryDirectory <*> pure "accelerate-cuda"
#endif
  createDirectoryIfMissing True dir
  openTempFile dir template


-- Debug
-- -----

{-# INLINE message #-}
message :: MonadIO m => String -> m ()
message msg = trace ("cc: " ++ msg) $ return ()

{-# INLINE trace #-}
trace :: MonadIO m => String -> m a -> m a
trace msg next = D.message D.dump_cc msg >> next

