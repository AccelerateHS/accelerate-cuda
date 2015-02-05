{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TemplateHaskell #-} 
{-# LANGUAGE GADTs #-}
{-# LANGUAGE BangPatterns #-} 


-- Implementation experiments regarding multidevice execution and
-- scheduling
module Data.Array.Accelerate.CUDA.ExecuteMulti where

-- import qualified Data.Array.Accelerate.CUDA.Execute as E
-- import Data.Array.Accelerate.CUDA.Execute.Stream                ( Stream )
import Data.Array.Accelerate.CUDA.AST hiding (Idx_) 
import Data.Array.Accelerate.CUDA.State


import Data.Array.Accelerate.Trafo  hiding (strengthen) 
import Data.Array.Accelerate.Trafo.Base
import Data.Array.Accelerate.Array.Sugar  ( Array
                                          , Shape
                                          , Elt
                                          , Arrays
                                          , Vector
                                          , EltRepr
                                          , Atuple(..)
                                          , Tuple(..)
                                          , TupleRepr
                                          , IsAtuple
                                          , Scalar )

-- import qualified Data.Array.Accelerate.CUDA.Debug               as D
import Data.Array.Accelerate.Error
--
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Analysis.Device
import qualified Foreign.CUDA.Driver    as CUDA
-- import Data.Array.Accelerate.CUDA.Analysis.Device

import Data.Array.Accelerate.CUDA.Context 

import Data.IORef

import Data.Word
import Data.Map as M
import Data.Set as S
import Data.List as L
import Data.Function
import Data.Ord 
-- import Debug.Trace

import Control.Monad.State
import Control.Applicative hiding (Const) 
-- import Control.Concurrent.MVar 

-- Scheduler related datatypes
-- ---------------------------
type DevID  = Int 
type MemID  = Int 

-- will we use TaskIDs using this approach ? 
type TaskID = Int 
type Size   = Word


-- | Which memory space are we in, this can be any kind of unique ID.
-- data MemID = CPU | GPU1 | GPU2    deriving (Show,Eq,Ord,Read)

data Device = Device { did :: DevID   
                     , mem :: MemID   -- ^ From which memory does this
                                      -- device consume and produce
                                      -- results?
                     , prio :: Double -- ^ A bias factor, tend towards
                                      -- using this device if it is
                                      -- higher priority.
                     }
              deriving (Show,Eq,Ord)

-- As I understand it, we will first experiment without having a
-- TaskGraph

-- | Scheduler state 
data SchedState =
  SchedState {
    freeDevs    :: Set DevID         -- ^ Free devices 
  , workingDevs :: Map DevID TaskID  -- ^ Busy devices  
  , arrays      :: Map TaskID (Size, Set MemID)
    -- ^ Resident arrays, indexed by the taskID that produced them.
    -- Note, a given array can be on more than one device.  Every
    -- taskId eventually appears in here, so it is also our
    -- "completed" list.
  , waiting     :: Set TaskID -- ^ Work waiting for executor
  , outstanding :: Set TaskID -- ^ Work assigned to executor

      
  -- A map from DevID to the devices available on the system
  -- The CUDA context identifies a device
  , allDevices  :: Map DevID Context 
  } 

-- I dont think this SchedState is exactly what we need for this setup.
-- 

-- Need some way of keeping track of actual devices.  What identifies
-- these devices ?  
-- ------------------------------------------------------------------
enumerateDevices :: IO [(CUDA.Device)]
enumerateDevices = do
  devices    <- mapM CUDA.device . enumFromTo 0 . subtract 1 =<< CUDA.count
  properties <- mapM CUDA.props devices
  return . L.map fst . sortBy (flip cmp `on` snd) $ zip devices properties 
  where
    compute     = computeCapability
    flops d     = multiProcessorCount d * (coresPerMP . deviceResources) d * clockRate d
    cmp x y
      | compute x == compute y  = comparing flops   x y
      | otherwise               = comparing compute x y

-- Initialize a starting state
-- ---------------------------

-- What flags to use ?
contextFlags :: [CUDA.ContextFlag]
contextFlags = [CUDA.SchedAuto]

initState :: IO SchedState
initState = do
  devs <- enumerateDevices
  devs' <- mapM (flip create contextFlags) devs 
  let numDevs = L.length devs

  let assocList = zip [0..numDevs-1] devs'
  return $ SchedState (S.fromList (L.map fst assocList))
                      M.empty
                      M.empty
                      S.empty
                      S.empty
                      (M.fromList assocList)


-- at any point along the traversal of the AST the executeOpenAccMulti
-- function will be under the influence of the SchedState
newtype SchedMonad a = SchedMonad (StateT SchedState CIO a)
          deriving ( MonadState SchedState
                   , Monad
                   , MonadIO
                   , Functor 
                   , Applicative )

runSched :: SchedMonad a -> SchedState -> CIO a
runSched (SchedMonad m) = evalStateT m
  

-- Evaluate an PreOpenAcc or ExecAcc or something under the influence
-- of the scheduler
-- ------------------------------------------------------------------

-- The plan:
-- Traverse DelayedOpenAcc
--  compileAcc on subtrees (that the scheduler decides to execute)
--   gives: ExecAcc
--   Tie up all arrays.. Scheduler knows of all arrays and where they are
--   Create env to pass to execOpenAcc (with the ExecAcc object) 

runDelayedAccMulti :: DelayedAcc arrs
                   -> SchedState
                   -> SchedMonad arrs
runDelayedAccMulti acc st =
  runDelayedOpenAccMulti acc Aempty st 



data Env env where
  Aempty :: Env ()
  Apush  :: Env env -> (t, IORef (Set MemID)) -> Env (env, t)
      -- Async t 

-- Async MemID 


runDelayedOpenAccMulti :: DelayedOpenAcc aenv arrs
                       -> Env aenv 
                       -> SchedState
                       -> SchedMonad arrs
runDelayedOpenAccMulti = traverseAcc 
  where
    traverseAcc :: forall aenv arrs. DelayedOpenAcc aenv arrs
                -> Env aenv
                -> SchedState
                -> SchedMonad arrs
    traverseAcc Delayed{} _ _ = $internalError "runDelayedOpenAccMulti" "unexpected delayed array"
    traverseAcc (Manifest pacc) _ _ =
      case pacc of

        -- look up what is at position ix in the environment
        -- Walk environment see if 
        Avar ix -> $internalError "runDelayedOpenAccMulti" "Not implemented"

        -- Let binding.
        --
        -- The format is: 
        -- let real_work in a
        -- So approach we will follow is, enqueue a for computation
        -- keep going into b and keep enqueing work
        
        Alet a b -> $internalError "runDelayedOpenAccMulti" "Not implemented"


        -- Another approach would be launch work
        -- at the operator level. 
        --
        -- This function of course needs to handle these cases
        -- since let a = expensive in map f a
        -- can occur (as an example)
        -- In this case we would enqueu expensive and map f a for
        -- execution.
        
        Map f a -> $internalError "runDelayedOpenAccMulti" "Not implemented"
        
        -- Array injection 
        Unit e   -> $internalError "runDelayedOpenAccMulti" "Not implemented"
        Use arrs -> $internalError "runDelayedOpenAccMulti" "Not implemented"
        
        _       -> $internalError "runDelayedOpenAccMulti" "Not implemented" 

-- Traverse a DelayedOpenAcc and figure out what arrays are being referenced
-- Those arrays must be copied to the device where that DelayedOpenAcc
-- will execute.

arrayRefs :: forall aenv arrs. DelayedOpenAcc aenv arrs -> S.Set (Idx_ aenv) -- change type
arrayRefs (Delayed{}) = $internalError "arrayRefs" "unexpected delayed array"
arrayRefs (Manifest pacc) =
  case pacc of
    Use  arr -> S.empty
    Unit x   -> S.empty
    
    Avar ix -> addFree ix
    Alet a b -> arrayRefs a `S.union` (strengthen (arrayRefs b))
    
    Apply f a -> arrayRefsAF f `S.union` arrayRefs a 
       
    Atuple tup -> travT tup
    Aprj ix tup -> arrayRefs tup

    Awhile p f a -> arrayRefsAF p `S.union`
                    arrayRefsAF f `S.union`
                    arrayRefs a
    Acond p t e  -> travE p `S.union`
                    arrayRefs t `S.union`
                    arrayRefs e

    Aforeign ff afun a -> $internalError "arrayRefs" "Aforeign"

    Reshape s a -> travE s `S.union` arrayRefs a
    Replicate _ e a -> travE e `S.union` arrayRefs a
    Slice _ a e -> arrayRefs a `S.union` travE e
    Backpermute e f a -> travE e `S.union`
                         travF f `S.union` arrayRefs a

    Generate e f -> travE e `S.union` travF f
    Map f a -> travF f `S.union` arrayRefs a
    ZipWith f a b -> travF f `S.union`
                     arrayRefs a `S.union`
                     arrayRefs b
    Transform e p f a -> travE e `S.union`
                         travF p `S.union`
                         travF f `S.union`
                         arrayRefs a

    Fold f z a -> travF f `S.union`
                  travE z `S.union`
                  arrayRefs a
    Fold1 f a -> travF f `S.union`
                 arrayRefs a
    FoldSeg f e a s -> travF f `S.union`
                       travE e `S.union`
                       arrayRefs a `S.union`
                       arrayRefs s
    Fold1Seg f a s -> travF f `S.union`
                      arrayRefs a `S.union`
                      arrayRefs s
    Scanl f e a -> travF f `S.union`
                   travE e `S.union`
                   arrayRefs a
    Scanl' f e a -> travF f `S.union`
                    travE e `S.union`
                    arrayRefs a
    Scanl1 f a -> travF f `S.union`
                  arrayRefs a
    Scanr f e a -> travF f `S.union`
                   travE e `S.union`
                   arrayRefs a
    Scanr' f e a -> travF f `S.union`
                    travE e `S.union`
                    arrayRefs a
    Scanr1 f a -> travF f `S.union`
                  arrayRefs a
    Permute f d g a -> travF f `S.union`
                       arrayRefs d `S.union`
                       travF g `S.union`
                       arrayRefs a

    Stencil f _ a -> travF f `S.union`
                     arrayRefs a
    Stencil2 f _ a1 _ a2 -> travF f `S.union`
                            arrayRefs a1 `S.union`
                            arrayRefs a2

    Collect l -> $internalError "arrayRefs" "Collect" 
                 
  where
    arrayRefsAF :: DelayedOpenAfun aenv' arrs' -> S.Set (Idx_ aenv')
    arrayRefsAF (Alam l) = strengthen $ arrayRefsAF l
    arrayRefsAF (Abody b) = arrayRefs b 

    travT :: Atuple (DelayedOpenAcc aenv') a -> S.Set (Idx_ aenv')
    travT NilAtup  = S.empty
    travT (SnocAtup !t !a) = travT t `S.union` arrayRefs a 

    travE :: DelayedOpenExp env aenv' t -> S.Set (Idx_ aenv')
    travE = arrayRefsE

    travF :: DelayedOpenFun env aenv' t -> S.Set (Idx_ aenv') 
    travF (Body b)  = travE b
    travF (Lam  f)  = travF f


arrayRefsE :: DelayedOpenExp env aenv e -> S.Set (Idx_ aenv)
arrayRefsE exp =
  case exp of
    Index       a e -> arrayRefs a `S.union` arrayRefsE e 
    LinearIndex a e -> arrayRefs a `S.union` arrayRefsE e 
    Shape       a   -> arrayRefs a

    
    -- Just recurse through
    -- --------------------
    Var        _     -> S.empty
    Const      _     -> S.empty
    PrimConst  _     -> S.empty
    IndexAny         -> S.empty
    IndexNil         -> S.empty
    Foreign   ff f x -> $internalError "arrayRefsE" "Foreign"
    Let        a b   -> arrayRefsE a `S.union` arrayRefsE b 
    IndexCons  t h   -> arrayRefsE t `S.union` arrayRefsE h
    IndexHead  h     -> arrayRefsE h
    IndexSlice _ x s -> arrayRefsE x `S.union` arrayRefsE s
    IndexFull  _ x s -> arrayRefsE x `S.union` arrayRefsE s
    ToIndex    s i   -> arrayRefsE s `S.union` arrayRefsE i
    FromIndex  s i   -> arrayRefsE s `S.union` arrayRefsE i
    Tuple      t     -> travT t
    Prj        _ e   -> arrayRefsE e
    Cond       p t e -> arrayRefsE p `S.union`
                        arrayRefsE t `S.union`
                        arrayRefsE e
    While      p f x -> travF p `S.union`
                        travF f `S.union`
                        arrayRefsE x
    PrimApp    _ e   -> arrayRefsE e

    ShapeSize  e     -> arrayRefsE e
    Intersect  x y   -> arrayRefsE x `S.union` arrayRefsE y
    Union      x y   -> arrayRefsE x `S.union` arrayRefsE y 
    
                        

  where
    travT :: Tuple (DelayedOpenExp env aenv) t -> S.Set (Idx_ aenv)
    travT NilTup = S.empty
    travT (SnocTup t e) = travT t `S.union` arrayRefsE e 

    travF :: DelayedOpenFun env aenv t -> S.Set (Idx_ aenv)
    travF (Body b) = arrayRefsE b
    travF (Lam f)  = travF f

    
    

-- Various
-- -------
strengthen ::  Arrays a => S.Set (Idx_ (aenv, a)) -> S.Set (Idx_ aenv)
strengthen s = S.map (\(Idx_ (SuccIdx v)) -> Idx_ v )
                     (S.delete (Idx_ ZeroIdx) s)

addFree :: Arrays a => Idx aenv a -> S.Set (Idx_ aenv) 
addFree = S.singleton . Idx_


data Idx_ aenv where
  Idx_ :: (Arrays a) => Idx aenv a -> Idx_ aenv

instance Eq (Idx_ aenv) where
  Idx_ ix1 == Idx_ ix2 = idxToInt ix1 == idxToInt ix2

instance Ord (Idx_ aenv) where
  Idx_ ix1 `compare` Idx_ ix2 = idxToInt ix1 `compare` idxToInt ix2 


    





