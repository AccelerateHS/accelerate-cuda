{-# LANGUAGE RankNTypes               #-}
{-# LANGUAGE ScopedTypeVariables      #-}
{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE GADTs                    #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
{-# LANGUAGE QuasiQuotes              #-}
{-# LANGUAGE TypeFamilies             #-}
{-# LANGUAGE UndecidableInstances     #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing  #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Foreign.Export
-- Copyright   : [2013] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell, Robert Clifton-Everest
-- License     : BSD3
--
-- Maintainer  : Robert Clifton-Everest <robertce@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--
-- This module is intended to allow the calling of Accelerate functions from
-- within CUDA C/C++code. See the nbody/visualizer example in the accelerate-examples
-- package to see how it is used.
--

module Data.Array.Accelerate.CUDA.Foreign.Export (

  -- ** Functions callable from foreign code
  -- In order to call these from from C, see the corresponding C function signature.
  accelerateCreate, accelerateDestroy, freeResult, freeProgram, getShape,
  getDevicePtrs,

  -- ** Exporting
  Exported, foreignAccModule, exportAfun1, buildExported,

  -- ** Types
  ResultArray, ShapeBuffer, DevicePtrBuffer, Device, ForeignContext,

) where

import Data.Functor
import System.FilePath
import Control.Exception
import Control.Applicative
import Foreign.StablePtr
import Foreign.C.Types
import Foreign.Ptr
import Foreign.Storable                                 ( poke )
import Foreign.Marshal.Array                            ( peekArray, pokeArray )
import Control.Monad.State                              ( liftIO )
import qualified Foreign.CUDA.Driver                    as CUDA
import System.IO.Unsafe
import Language.Haskell.TH                              hiding ( ppr )
import Language.C.Quote.C
import Language.C.Syntax                                ( Definition )
import Text.PrettyPrint.Mainland

-- friends
import Data.Array.Accelerate.Smart                      ( Acc )
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.CUDA.Array.Sugar           hiding ( shape, size )
import Data.Array.Accelerate.CUDA.Array.Data            hiding ( pokeArray, peekArray )
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Compile
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Execute

#include "accelerate.h"

-- |A handle foreign code can use to call accelerate functions.
type AccHandle = StablePtr Context

-- |A result array from an accelerate program.
type ResultArray sh e = StablePtr (Array sh e)

-- |A foreign buffer that represents a shape as an array of ints.
type ShapeBuffer = CArray CInt

-- |A buffer of device pointers
type DevicePtrBuffer = CArray WordPtr

-- |Foreign exportable representation of a CUDA device
type Device = Int32

-- |Foreign representation of a CUDA context.
type ForeignContext = Ptr ()

type CArray a = Ptr a

-- |Given a left-nested tuple of argument types and a result type, generate the
-- corresponding function type. This method for building functions of arbitrary
-- arity proved the most effective in this particular instance.
type family Fun a r
type instance Fun ()    r = r
type instance Fun (a,x) r = Fun a (x -> r)

-- |The structure of the above type, reified.
data FunR a r where
  ResultF :: r               -> FunR ()    r
  PushF   :: FunR a (x -> r) -> FunR (a,x) r

instance Functor (FunR a) where
  fmap f (ResultF a) = ResultF  (f a)
  fmap f (PushF f')  = PushF (fmap (f .) f')

-- |Convert the reified function into an actual haskell function.
toFun :: FunR a r -> Fun a r
toFun (ResultF r) = r
toFun (PushF f)   = toFun f

-- |Given the type of a monomorphic accelerate function, yields the type of the
-- foreign version.
-- For example a function of type
--  ''Acc (Array DIM1 (Float, Double), Array DIM0 Int) -> Acc (Array DIM0 Int, Array DIM1 Float)''
-- has foreign version
--  ''DevicePtrBuffer -> ShapeBuffer -> DevicePtrBuffer -> ShapeBuffer -> Ptr (ResultArray DIM0 Int) -> Ptr (ResultArray DIM1 Float) -> IO ()''
-- Where the first 6 args correspond to input, and the last 6 to output.
--
-- NB: In the the input function ''Acc a -> Acc b'' both ''a'' and ''b'' have
-- to be flat-- i.e Not nested.
type family Exported f
type instance Exported (Acc a -> Acc b) = Fun (InputArgsOf (ArrRepr a)) (Fun (OutputArgsOf (ArrRepr b)) (IO ()))

type family InputArgsOf a
type instance InputArgsOf ()      = ()
type instance InputArgsOf (xs, x) = ((InputArgsOf xs, DevicePtrBuffer), ShapeBuffer)

type family OutputArgsOf a
type instance OutputArgsOf ()      = ()
type instance OutputArgsOf (xs, Array sh e) = (OutputArgsOf xs, Ptr (ResultArray sh e))

-- |Create an Accelerate handle given a device and a cuda context.
--
-- @AccHandle accelerateCreate(int device, CUcontext ctx);@
accelerateCreate :: Device -> ForeignContext -> IO AccHandle
accelerateCreate dev ctx = fromDeviceContext (CUDA.Device $ CInt dev) (CUDA.Context ctx) >>= newStablePtr

-- |Releases all resources used by the accelerate library.
--
-- @void accelerateDestroy(AccHandle hndl);@
accelerateDestroy :: AccHandle -> IO ()
accelerateDestroy = freeStablePtr

-- |Function callable from foreign code to 'free' a ResultArray returned after executing
-- an Accelerate computation.
--
-- Once freed, the device pointers associated with an array are no longer valid.
--
-- @void freeResult(ResultArray arr);@
freeResult :: ResultArray sh e -> IO ()
freeResult arr = freeStablePtr arr

-- |Free a compiled accelerate program.
--
-- @void freeProgram(Program prg);@
freeProgram :: StablePtr a -> IO ()
freeProgram = freeStablePtr

-- |Get the shape of the result array and write it to the given buffer.
--
-- @void getShape(ResultArray arr, int *sh);@
getShape :: ResultArray sh e -> ShapeBuffer -> IO ()
getShape arr psh = do
  a <- deRefStablePtr arr
  forSh a
  where
    forSh :: forall sh e. Array sh e -> IO ()
    forSh (Array sh _) = do
      let sh' = toElt sh :: sh
      pokeArray psh (map fromIntegral $ shapeToList sh')

-- |Get the device pointers associated with the result array and
-- write them to the given buffer.
--
-- @void getDevicePtrs(AccHandle hndl, ResultArray arr, void **buffer);@
getDevicePtrs :: AccHandle -> ResultArray sh e -> DevicePtrBuffer -> IO ()
getDevicePtrs hndl arr pbuf = do
  (Array _ adata) <- deRefStablePtr arr
  ctx   <- deRefStablePtr hndl
  dptrs <- evalCUDA ctx (devicePtrsOfArrayData adata)
  pokeArray pbuf (devicePtrsToWordPtrs adata dptrs)

-- |A template haskell stub that generates foreign export statements for the
-- functions above.
foreignAccModule :: Q [Dec]
foreignAccModule = createHfile >> exports
  where
    createHfile = do
      Loc hsFile _ _ _ _ <- location
      let hFile = replaceExtension hsFile ".h"
      let contents = (show . ppr) $ typedefs (takeBaseName hFile ++ "_stub.h")
      runIO $ writeFile hFile contents

    typedefs file = [cunit|
        $esc:("#include \"" ++ file ++ "\"")

        typedef typename HsStablePtr AccHandle;
        typedef typename HsStablePtr AccProgram;
        typedef typename HsStablePtr ResultArray;
      |]

    exports = sequence $ uncurry exportf <$>
      [
        ('accelerateCreate,  [t| Device -> ForeignContext -> IO AccHandle |])
      , ('accelerateDestroy, [t| AccHandle -> IO () |])
      , ('freeResult,        [t| forall sh e. ResultArray sh e -> IO () |])
      , ('freeProgram,       [t| forall a. StablePtr a -> IO ()|])
      , ('getShape,          [t| forall sh e. ResultArray sh e -> ShapeBuffer -> IO () |])
      , ('getDevicePtrs,     [t| forall sh e. AccHandle -> ResultArray sh e -> DevicePtrBuffer -> IO () |])
      ]
      where
        exportf name ty = ForeignD <$> (ExportF cCall (nameBase name) name <$> ty)

-- |Given the name, ''f'', of an Accelerate function (a function of type ''Acc a -> Acc b'')
-- generate two top level bindings, ''f_compile'' and ''f_run'', and export them. When given
-- a ''AccHandle'', ''f_compile'' will compile the function and return a handle to the program
-- that can be called with ''f_run''.
exportAfun1 :: Name -> Q [Dec]
exportAfun1 fname = do
  (VarI n ty _ _) <- reify fname

  -- Generate initialisation function
  initFun <- genCompileFun n ty

  -- Generate calling function
  callFun <- genCallFun n ty

  return $ initFun ++ callFun

genCallFun :: Name -> Type -> Q [Dec]
genCallFun fname ty@(AppT (AppT ArrowT (AppT _ a)) (AppT _ b))
  = sequence [sig, dec, expt]
  where
    body = [| \sfun ->
        unsafePerformIO (deRefStablePtr sfun)
      |]
    dec  = FunD callName . (:[]) <$> cls
    cls  = Clause [] <$> (NormalB <$> body) <*> return []
    sig  = SigD callName <$> ety
    expt = ForeignD <$> (ExportF cCall (nameBase callName) callName <$> ety)

    callName = mkName $ nameBase fname ++ "_run"

    ety   = [t| StablePtr (Exported $(return ty)) -> Exported $(return ty) |]
genCallFun _      _
  = error "Invalid accelerate function"

genCompileFun :: Name -> Type -> Q [Dec]
genCompileFun fname ty@(AppT (AppT ArrowT (AppT _ a)) (AppT _ b))
  = sequence [sig, dec, expt]
  where
    initName = mkName $ nameBase fname ++ "_compile"

    body = [| \hndl ->
        let
            ff = buildExported hndl $(varE fname)
        in newStablePtr ff |]
    dec  = FunD initName . (:[]) <$> cls
    cls  = Clause [] <$> (NormalB <$> body) <*> return []
    sig  = SigD initName <$> ety
    expt = ForeignD <$> (ExportF cCall (nameBase initName) initName <$> ety)

    ety   = [t| AccHandle -> IO (StablePtr (Exported $(return ty))) |]
genCompileFun _     _
  = error "Invalid accelerate function"

-- |Given a handle and an Accelerate function, generate an exportable version.
-- This can be used as an alternative to genCallFun and genCompileFun if more
-- control over execution is desired.
buildExported :: forall a b. (Arrays a, Arrays b) => AccHandle -> (Acc a -> Acc b) -> Exported (Acc a -> Acc b)
buildExported hndl f = toFun (toFun . fmap eval <$> ffun)
  where
    !fun = executeAfun1 afun
    !acc = convertAfunWith config f
    !afun = unsafePerformIO $ eval (compileAfun acc)

    funr :: ArrRepr a -> CIO (ArrRepr b)
    funr a = fromArr <$> fun (toArr a)

    eval :: CIO x -> IO x
    eval cio = do
      ctx <- deRefStablePtr hndl
      evalCUDA ctx cio
      `catch`
        \e -> INTERNAL_ERROR(error) "unhandled" (show (e :: CUDA.CUDAException))

    ffun :: FunR (InputArgsOf (ArrRepr a)) (FunR (OutputArgsOf (ArrRepr b)) (CIO ()))
    ffun = g <$> b
      where
        g :: CIO (ArrRepr b) -> FunR (OutputArgsOf (ArrRepr b)) (CIO ())
        g i = fmap (i >>=) (buildOut (arrays (undefined :: b)))

        b :: FunR (InputArgsOf (ArrRepr a)) (CIO (ArrRepr b))
        b = (>>= funr) <$> buildIn (arrays (undefined :: a))

    buildIn :: forall a. ArraysR a -> FunR (InputArgsOf a) (CIO a)
    buildIn ArraysRunit                   = ResultF (return ())
    buildIn (ArraysRpair aR ArraysRarray)
      = (PushF . PushF) (fmap g (buildIn aR))
      where
        g :: a ~ (l,arr) => CIO l -> DevicePtrBuffer -> ShapeBuffer -> CIO a
        g i dps sh = do
          l <- i
          arr <- arrayFromForeignData dps sh
          return (l,arr)
    buildIn _                              = exportError

    buildOut :: forall b. ArraysR b -> FunR (OutputArgsOf b) (b -> CIO ())
    buildOut ArraysRunit                   = ResultF (const $ return ())
    buildOut (ArraysRpair aR ArraysRarray)
      = PushF (fmap g (buildOut aR))
      where
        g :: b ~ (l, Array sh e) => (l -> CIO ()) -> Ptr (ResultArray sh e) -> b -> CIO ()
        g i ptr (l, arr) = write arr ptr >> i l

        write :: Array sh e -> Ptr (ResultArray sh e) -> CIO ()
        write a p = liftIO $ do
          sp <- newStablePtr a
          poke p sp
    buildOut _                             = exportError

    exportError = error "Unable to export a function over nested tuples of arrays."

-- Utility functions
-- ------------------

arrayFromForeignData :: forall sh e. (Shape sh, Elt e) => DevicePtrBuffer -> ShapeBuffer -> CIO (Array sh e)
arrayFromForeignData ptrs shape = do
   let d  = dim (ignore :: sh) -- Using ignore as using dim requires a non-dummy argument
   let sz = eltSize (eltType (undefined :: e))
   lst <- liftIO (peekArray d shape)
   liftIO $ putStrLn $ "Shape: " ++ (show lst)
   let sh = listToShape (map fromIntegral lst) :: sh
   plst <- liftIO $ peekArray sz ptrs
   liftIO $ putStrLn $ "Pointers: " ++ (show plst)
   let ptrs' = devicePtrsFromList (arrayElt :: ArrayEltR (EltRepr e)) plst
   useDevicePtrs (fromElt sh) ptrs'

config :: Phase
config =  Phase
  { recoverAccSharing      = True
  , recoverExpSharing      = True
  , floatOutAccFromExp     = True
  , enableAccFusion        = True
  , convertOffsetOfSegment = True
  }

eltSize :: TupleType e -> Int
eltSize UnitTuple         = 0
eltSize (SingleTuple _  ) = 1
eltSize (PairTuple   a b) = eltSize a + eltSize b

