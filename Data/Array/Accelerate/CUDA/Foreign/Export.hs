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
{-# LANGUAGE ViewPatterns             #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing #-}
{-# OPTIONS_GHC -fno-warn-orphans        #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Foreign.Export
-- Copyright   : [2013..2014] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell, Robert Clifton-Everest
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
  accelerateCreate, accelerateDestroy, freeOutput, freeProgram,

  -- ** Exporting
  exportAfun, buildExported,

  -- ** Types
  InputArray, OutputArray, ShapeBuffer, DevicePtrBuffer,

) where

import Data.Functor
import Control.Applicative
import Control.Monad.State                              ( liftIO )
import Foreign.StablePtr
import Foreign.C.Types
import Foreign.Ptr
import Foreign.Storable                                 ( Storable(..) )
import Foreign.Marshal.Array                            ( peekArray, pokeArray, mallocArray )
import Foreign.Marshal.Alloc                            ( free )
import Language.Haskell.TH                              hiding ( ppr )
import qualified Foreign.CUDA.Driver                    as CUDA

import Prelude                                          as P

-- friends
import Data.Array.Accelerate.Smart                      ( Acc )
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.CUDA                       ( run1With )
import Data.Array.Accelerate.CUDA.Array.Sugar           hiding ( shape, size )
import Data.Array.Accelerate.CUDA.Array.Data            hiding ( pokeArray, peekArray, mallocArray )
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Context

-- |A handle foreign code can use to call accelerate functions.
type AccHandle = StablePtr Context

-- |A foreign buffer that represents a shape as an array of ints.
type ShapeBuffer = Ptr CInt

-- |A buffer of device pointers
type DevicePtrBuffer = Ptr WordPtr

-- |The input required from foreign code.
type InputArray = (ShapeBuffer, DevicePtrBuffer)

-- |A result array from an accelerate program.
type OutputArray = (ShapeBuffer, DevicePtrBuffer, StablePtr EArray)

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

-- We need to export these
foreign export ccall accelerateCreate  :: Device -> ForeignContext -> IO AccHandle
foreign export ccall accelerateDestroy :: AccHandle -> IO ()
foreign export ccall runProgram        :: AccHandle -> StablePtr Afun -> Ptr InputArray -> Ptr OutputArray -> IO ()
foreign export ccall freeOutput        :: Ptr OutputArray -> IO ()
foreign export ccall freeProgram       :: StablePtr a -> IO ()

instance Storable InputArray where
  sizeOf (sh, ptrs) = sizeOf sh + sizeOf ptrs

  alignment _ = 0

  peek ptr = do
    let p_sh   = castPtr ptr :: Ptr ShapeBuffer
    sh         <- peek p_sh
    let p_ptrs = (castPtr p_sh :: Ptr DevicePtrBuffer) `plusPtr` sizeOf sh
    ptrs       <- peek p_ptrs
    return (sh, ptrs)

  poke ptr (sh, ptrs) = do
    let p_sh   = castPtr ptr :: Ptr ShapeBuffer
        p_ptrs = (castPtr p_sh :: Ptr DevicePtrBuffer) `plusPtr` sizeOf sh
    poke p_sh sh
    poke p_ptrs ptrs

instance Storable OutputArray where
  sizeOf (sh, ptrs, sa) = sizeOf sh + sizeOf ptrs + sizeOf sa
  alignment _ = 0
  peek ptr = do
    let p_sh   = castPtr ptr :: Ptr ShapeBuffer
    sh         <- peek p_sh
    let p_ptrs = (castPtr p_sh :: Ptr DevicePtrBuffer) `plusPtr` sizeOf sh
    ptrs       <- peek p_ptrs
    let p_sa   = (castPtr p_ptrs :: Ptr (StablePtr a)) `plusPtr` sizeOf ptrs
    sa         <- peek p_sa
    return (sh, ptrs, sa)
  poke ptr (sh, ptrs, sa) = do
    let p_sh   = castPtr ptr :: Ptr ShapeBuffer
        p_ptrs = (castPtr p_sh :: Ptr DevicePtrBuffer) `plusPtr` sizeOf sh
        p_sa   = (castPtr p_ptrs :: Ptr (StablePtr a)) `plusPtr` sizeOf ptrs
    poke p_sh sh
    poke p_ptrs ptrs
    poke p_sa sa

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

-- |Function callable from foreign code to 'free' a OutputArray returned after executing
-- an Accelerate computation.
--
-- Once freed, the device pointers associated with an array are no longer valid.
--
-- @void freeOutput(OutputArray arr);@
freeOutput :: Ptr OutputArray -> IO ()
freeOutput o = do
  (sh, dptrs, sa) <- peek o
  free sh
  free dptrs
  freeStablePtr sa

-- |Free a compiled accelerate program.
--
-- @void freeProgram(Program prg);@
freeProgram :: StablePtr a -> IO ()
freeProgram = freeStablePtr

-- |Execute the given accelerate program with @is@ as the input and @os@ as the output.
--
-- @void runProgram(AccHandle hndl, AccProgram p, InputArray* is, OutputArray* os);@
runProgram :: AccHandle -> StablePtr Afun -> Ptr InputArray -> Ptr OutputArray -> IO ()
runProgram hndl fun input output = do
  ctx <- deRefStablePtr hndl
  af <- deRefStablePtr fun
  run ctx af
  where
    run :: Context -> Afun -> IO ()
    run ctx (Afun f (_ :: a) (_ :: b)) = do
      _ <- evalCUDA ctx $ do
        (a, _) <- marshalIn (arrays (undefined :: a)) input
        let !b = f (toArr a)
        marshalOut (arrays (undefined :: b)) (fromArr b) output
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

    marshalOut :: ArraysR b -> b -> Ptr OutputArray -> CIO (Ptr OutputArray)
    marshalOut ArraysRunit  () ptr = return ptr
    marshalOut ArraysRarray a  ptr = do
      oarr <- mkOutput a
      liftIO $ poke ptr oarr
      return (plusPtr ptr (sizeOf oarr))
      where
        mkOutput :: forall sh e. Shape sh => Array sh e -> CIO OutputArray
        mkOutput (Array sh adata) = do
          let sh' = shapeToList (toElt sh :: sh)
          shbuf <- liftIO $ mallocArray (P.length sh')
          liftIO $ pokeArray shbuf (map fromIntegral sh')

          withDevicePtrs adata Nothing $ \dptrs -> do
            let wptrs = devicePtrsToWordPtrs adata dptrs
            pbuf  <- liftIO $ mallocArray (P.length wptrs)
            liftIO $ pokeArray pbuf wptrs

            sa <- liftIO $ newStablePtr (EArray a)
            return (shbuf, pbuf, sa)

    marshalOut (ArraysRpair aR1 aR2) (x,y) ptr = do
      ptr' <- marshalOut aR1 x ptr
      marshalOut aR2 y ptr'


-- |Given the 'Name' of an Accelerate function (a function of type ''Acc a -> Acc b'') generate a
-- a function callable from foreign code with the second argument specifying it's name.
exportAfun :: Name -> String -> Q [Dec]
exportAfun fname ename = do
  (VarI n ty _ _) <- reify fname

  -- Generate initialisation function
  genCompileFun n ename ty

genCompileFun :: Name -> String -> Type -> Q [Dec]
genCompileFun fname ename (AppT (AppT ArrowT (AppT _ _)) (AppT _ _))
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
      newStablePtr (Afun (run1With ctx f) (undefined :: a) (undefined :: b))

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

