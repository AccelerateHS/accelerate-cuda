{-# LANGUAGE RankNTypes               #-}
{-# LANGUAGE ScopedTypeVariables      #-}
{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE GADTs                    #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
{-# LANGUAGE QuasiQuotes              #-}
{-# LANGUAGE TypeFamilies             #-}
{-# LANGUAGE FlexibleInstances        #-}
{-# LANGUAGE ImpredicativeTypes       #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing -fno-warn-orphans #-}
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
  foreignAccModule, exportAfun, buildExported,

  -- ** Types
  InputArray, ResultArray, ShapeBuffer, DevicePtrBuffer,

) where

import Data.Functor
import System.FilePath
import Control.Applicative
import Foreign.StablePtr
import Foreign.C.Types
import Foreign.Ptr
import Foreign.Storable                                 ( Storable(..) )
import Foreign.Marshal.Array                            ( peekArray, pokeArray )
import Control.Monad.State                              ( liftIO )
import qualified Foreign.CUDA.Driver                    as CUDA
import Language.Haskell.TH                              hiding ( ppr )
import Language.C.Quote.C
import Text.PrettyPrint.Mainland

-- friends
import Data.Array.Accelerate.Smart                      ( Acc )
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.CUDA                       ( run1In )
import Data.Array.Accelerate.CUDA.Array.Sugar           hiding ( shape, size )
import Data.Array.Accelerate.CUDA.Array.Data            hiding ( pokeArray, peekArray )
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Context

-- |A handle foreign code can use to call accelerate functions.
type AccHandle = StablePtr Context

-- |A result array from an accelerate program.
type ResultArray = StablePtr EArray

-- |A foreign buffer that represents a shape as an array of ints.
type ShapeBuffer = Ptr CInt

-- |A buffer of device pointers
type DevicePtrBuffer = Ptr WordPtr

-- |The input required from foreign code.
type InputArray = (ShapeBuffer, DevicePtrBuffer)

-- |Foreign exportable representation of a CUDA device
type Device = Int32

-- |Foreign representation of a CUDA context.
type ForeignContext = Ptr ()

-- We need to capture the Arrays constraint
data Afun where
  Afun :: (Arrays a, Arrays b)
       => (a -> b)
       -> a {- dummy -}
       -> b {- dummy -}
       -> Afun

data EArray where
  EArray :: (Shape sh, Elt e) => Array sh e -> EArray

instance Storable InputArray where
  sizeOf (sh, ptrs) = sizeOf sh + sizeOf ptrs
  alignment _ = 0
  peek ptr = do
    sh <- peek (castPtr ptr :: Ptr ShapeBuffer)
    ptrs <- peek ((castPtr ptr :: Ptr DevicePtrBuffer) `plusPtr` (sizeOf sh))
    return (sh, ptrs)
  poke ptr (sh, ptrs) = do
    poke (castPtr ptr :: Ptr ShapeBuffer) sh
    poke ((castPtr ptr :: Ptr DevicePtrBuffer) `plusPtr` (sizeOf sh)) ptrs

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
freeResult :: ResultArray -> IO ()
freeResult arr = freeStablePtr arr

-- |Free a compiled accelerate program.
--
-- @void freeProgram(Program prg);@
freeProgram :: StablePtr a -> IO ()
freeProgram = freeStablePtr

-- |Get the shape of the result array and write it to the given buffer.
--
-- @void getShape(ResultArray arr, int *sh);@
getShape :: ResultArray -> ShapeBuffer -> IO ()
getShape arr psh = do
  a <- deRefStablePtr arr
  getSh a
  where
    getSh :: EArray -> IO ()
    getSh (EArray a) = forSh a

    forSh :: forall sh e. Array sh e -> IO ()
    forSh (Array sh _) = do
      let sh' = toElt sh :: sh
      pokeArray psh (map fromIntegral $ shapeToList sh')

-- |Get the device pointers associated with the result array and
-- write them to the given buffer.
--
-- @void getDevicePtrs(AccHandle hndl, ResultArray arr, void **buffer);@
getDevicePtrs :: AccHandle -> ResultArray -> DevicePtrBuffer -> IO ()
getDevicePtrs hndl arr pbuf = do
  (EArray (Array _ adata)) <- deRefStablePtr arr
  ctx   <- deRefStablePtr hndl
  dptrs <- evalCUDA ctx (devicePtrsOfArrayData adata)
  pokeArray pbuf (devicePtrsToWordPtrs adata dptrs)

-- |Execute the given accelerate program with @is@ as the input and @os@ as the output.
--
-- @void runProgram(AccHandle hndl, AccProgram p, InputArray* is, ResultArray* os);@
runProgram :: AccHandle -> StablePtr Afun -> Ptr InputArray -> Ptr ResultArray -> IO ()
runProgram hndl fun input output = do
  ctx <- deRefStablePtr hndl
  af <- deRefStablePtr fun
  run ctx af
  where
    run :: Context -> Afun -> IO ()
    run ctx (Afun f (_ :: a) (_ :: b)) = do
      (a, _) <- evalCUDA ctx $ marshalIn (arrays (undefined :: a)) input
      let !b = f (toArr a)
      _ <- marshalOut (arrays (undefined :: b)) (fromArr b) output
      return ()

    marshalIn :: ArraysR a -> Ptr InputArray -> CIO (a, Ptr InputArray)
    marshalIn ArraysRunit  ptr = return ((), ptr)
    marshalIn ArraysRarray ptr = do
      (sh, ptrs) <- liftIO (peek ptr)
      a <- arrayFromForeignData ptrs sh
      let ptr' = plusPtr ptr (sizeOf (sh, ptrs))
      return (a, ptr')
    marshalIn (ArraysRpair aR1 aR2) ptr = do
      (x, ptr')  <- marshalIn aR1 ptr
      (y, ptr'') <- marshalIn aR2 ptr'
      return ((x,y), ptr'')

    marshalOut :: ArraysR b -> b -> Ptr ResultArray -> IO (Ptr ResultArray)
    marshalOut ArraysRunit  () ptr = return ptr
    marshalOut ArraysRarray a  ptr = do
      sptr <- newStablePtr (EArray a)
      poke ptr sptr
      return (plusPtr ptr (sizeOf sptr))
    marshalOut (ArraysRpair aR1 aR2) (x,y) ptr = do
      ptr' <- marshalOut aR1 x ptr
      marshalOut aR2 y ptr'


-- |A template haskell stub that generates a .h file in the directory in which
-- the module is being compiled. This file will include the C definitions of
-- the types below, the functions above, and
--
-- @AccHandle accelerateInit(int argc, char** argv, int device, CUcontext ctx);@
--
-- Which, in addition to creating an Accelerate handle, also initialises the
-- Haskell runtime.
foreignAccModule :: Q [Dec]
foreignAccModule = createHfile >> exports
  where
    createHfile = do
      Loc hsFile _ _ _ _ <- location
      let hFile = replaceExtension hsFile ".h"
      let contents = (show . ppr) $ typedefs (takeBaseName hFile ++ "_stub.h")
      runIO $ writeFile hFile contents

    typedefs file = [cunit|
        $esc:("#include \"HsFFI.h\"")
        $esc:("#include \"" ++ file ++ "\"")
        $esc:("#include <cuda.h>")

        typedef typename HsStablePtr AccHandle;
        typedef typename HsStablePtr AccProgram;
        typedef typename HsStablePtr ResultArray;

        typedef struct {
          int*    shape;
          void**  ptrs;
        } InputArray;

        AccHandle accelerateInit(int argc, char** argv, int device, typename CUcontext ctx) {
          extern void __stginit_Dotp ( void );
          hs_init(&argc, &argv);
          return accelerateCreate(device, ctx);
        }
      |]

    exports = sequence $ uncurry exportf <$>
      [
        ('accelerateCreate,  [t| Device -> ForeignContext -> IO AccHandle |])
      , ('accelerateDestroy, [t| AccHandle -> IO () |])
      , ('runProgram,        [t| AccHandle -> StablePtr Afun -> Ptr InputArray -> Ptr ResultArray -> IO () |])
      , ('freeResult,        [t| ResultArray -> IO () |])
      , ('freeProgram,       [t| forall a. StablePtr a -> IO ()|])
      , ('getShape,          [t| ResultArray -> ShapeBuffer -> IO () |])
      , ('getDevicePtrs,     [t| AccHandle -> ResultArray -> DevicePtrBuffer -> IO () |])
      ]
      where
        exportf name ty = ForeignD <$> (ExportF cCall (nameBase name) name <$> ty)

-- |Given the name, ''f'', of an Accelerate function (a function of type ''Acc a -> Acc b'')
-- generate two top level bindings, ''f_compile'' and ''f_run'', and export them. When given
-- a ''AccHandle'', ''f_compile'' will compile the function and return a handle to the program
-- that can be called with ''f_run''.
exportAfun :: Name -> String -> Q [Dec]
exportAfun fname ename = do
  (VarI n ty _ _) <- reify fname

  -- Generate initialisation function
  initFun <- genCompileFun n ename ty

  return initFun

genCompileFun :: Name -> String -> Type -> Q [Dec]
genCompileFun fname ename ty@(AppT (AppT ArrowT (AppT _ a)) (AppT _ b))
  = sequence [sig, dec, expt]
  where
    initName = mkName ename

    body = [| \hndl -> buildExported hndl $(varE fname) |]
    dec  = FunD initName . (:[]) <$> cls
    cls  = Clause [] <$> (NormalB <$> body) <*> return []
    sig  = SigD initName <$> ety
    expt = ForeignD <$> (ExportF cCall (nameBase initName) initName <$> ety)

    ety   = [t| AccHandle -> IO (StablePtr Afun) |]
genCompileFun _     _     _
  = error "Invalid accelerate function"

-- |Given a handle and an Accelerate function, generate an exportable version.
buildExported :: forall a b. (Arrays a, Arrays b) => AccHandle -> (Acc a -> Acc b) -> IO (StablePtr Afun)
buildExported hndl f = ef
  where
    ef :: IO (StablePtr Afun)
    ef = do
      ctx <- deRefStablePtr hndl
      newStablePtr (Afun (run1In ctx f) (undefined :: a) (undefined :: b))

-- Utility functions
-- ------------------

arrayFromForeignData :: forall sh e. (Shape sh, Elt e) => DevicePtrBuffer -> ShapeBuffer -> CIO (Array sh e)
arrayFromForeignData ptrs shape = do
   let d  = dim (ignore :: sh) -- Using ignore as using dim requires a non-dummy argument
   let sz = eltSize (eltType (undefined :: e))
   lst <- liftIO (peekArray d shape)
   let sh = listToShape (map fromIntegral lst) :: sh
   plst <- liftIO $ peekArray sz ptrs
   let ptrs' = devicePtrsFromList (arrayElt :: ArrayEltR (EltRepr e)) plst
   useDevicePtrs (fromElt sh) ptrs'

eltSize :: TupleType e -> Int
eltSize UnitTuple         = 0
eltSize (SingleTuple _  ) = 1
eltSize (PairTuple   a b) = eltSize a + eltSize b

