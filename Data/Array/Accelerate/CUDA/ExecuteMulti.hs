{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TemplateHaskell #-} 
{-# LANGUAGE GADTs #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE EmptyDataDecls #-} 


-- Implementation experiments regarding multidevice execution and
-- scheduling
module Data.Array.Accelerate.CUDA.ExecuteMulti
       (  
         runDelayedOpenAccMulti,
         runDelayedAccMulti,
       ) where

import Data.Array.Accelerate.CUDA.AST hiding (Idx_, prj) 
import qualified Data.Array.Accelerate.CUDA.State as CUDA 
import Data.Array.Accelerate.CUDA.Compile
import qualified Data.Array.Accelerate.CUDA.Execute as E
import qualified Data.Array.Accelerate.CUDA.Execute.Event as E 
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Array.Data

import Data.Array.Accelerate.Trafo  hiding (strengthen) 
import Data.Array.Accelerate.Trafo.Base hiding (inject) 
import Data.Array.Accelerate.Array.Sugar  ( Array
                                          , Arrays(..), ArraysR(..)
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

import Data.Array.Accelerate.Error
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Analysis.Device
import qualified Foreign.CUDA.Driver       as CUDA
import qualified Foreign.CUDA.Driver.Event as CUDA

import Foreign.Ptr (nullPtr)

import Data.IORef

import Data.Word
import Data.Map as M
import Data.Set as S
import Data.List as L
import Data.Function
import Data.Ord


-- import qualified Data.Array as A
import qualified Data.Array.IArray as A 

import Control.Monad.Reader
import Control.Applicative hiding (Const) 

-- Concurrency 
import Control.Concurrent.MVar
import Control.Concurrent
import Control.Concurrent.Chan

import System.IO 
import System.IO.Unsafe 

debug = True 

debugLock :: MVar Integer 
debugLock = unsafePerformIO $ newMVar 0 

debugMsg str =
  when debug $ 
  do
    n <- takeMVar debugLock
    hPutStrLn stderr $ show n ++ ": " ++ str
    putMVar debugLock (n+1) 


-- Datastructures for Gang of worker threads
-- One thread per participating device
-- -----------------------------------------
data Done = Done -- Free to accept work  
data Work = ShutDown
          | Work (IO ()) 

data DeviceState = DeviceState { devCtx      :: Context
                               , devDoneMVar :: MVar Done
                               , devWorkMVar :: MVar Work
                               , devThread   :: ThreadId
                               }
                   deriving Show

instance Show Context where
  show _ = "Context"

instance Show (Chan a) where
  show _ = "Chan"

instance Show (MVar a) where
  show _ = "MVar" 
                   
-- Each device is associated a worker thread
-- that performs workloads passed to it from the
-- Scheduler
createDeviceThread :: CUDA.Device -> IO DeviceState
createDeviceThread dev =
  do
    debugMsg $ "Creating context on device"
    ctxMVar <- newEmptyMVar 
    
    work <- newEmptyMVar 
    done <- newEmptyMVar

    debugMsg $ "Forking device work thread" 
    tid <- runInBoundThread $ forkIO $
           do
             ctx <- create dev contextFlags
             putMVar ctxMVar ctx 
             -- Bind the created context to this thread
             -- I assume that means operations within
             -- this thread will default to this context
             -- CUDA.set (deviceContext ctx)
             -- Enter the workloop 
             deviceLoop done work

    ctx <- takeMVar ctxMVar
    return $ DeviceState ctx done work tid 

    -- The worker thread 
    where deviceLoop done work =
            do
              debugMsg $ "Entered device workloop" 
              x <- takeMVar work
              debugMsg $ "Work available!" 
              case x of
                ShutDown -> putMVar done Done >> return () 
                Work w ->
                  do debugMsg $ "Entering work loop" 
                     w
                     debugMsg $ "Exiting work loop"

              debugMsg $ "Device reporting done"
              putMVar done Done
              deviceLoop done work 
  

-- Initialize a starting state
-- ---------------------------
initScheduler :: IO SchedState
initScheduler = do
  devs <- enumerateDevices
  let numDevs = L.length devs
  debugMsg $ "InitScheduler: found " ++ show numDevs ++ " devices."
  
   -- Create a device thread for each device 
  devs' <- mapM createDeviceThread devs 
  debugMsg $ "InitScheduler: Created device threads."
  --
    
  let assocList = zip [0..numDevs-1] devs'

  -- All devices start out free
  free <- newChan
  writeList2Chan free (L.map fst assocList) 

  let st = SchedState (A.array (0,numDevs-1) assocList)
                      free

  debugMsg $ "InitScheduler: " ++ show st 
  
  return $ st


-- Scheduler related datatypes
-- ---------------------------
type DevID  = Int 
type MemID  = Int 

data SchedState =
  SchedState {
      deviceState  :: A.Array Int DeviceState
      -- this is a channel, for now. 
    , freeDevs     :: Chan DevID 
                      
    }
  deriving Show 
      

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

-- What flags to use ?
contextFlags :: [CUDA.ContextFlag]
contextFlags = [CUDA.SchedAuto]

-- at any point along the traversal of the AST the executeOpenAccMulti
-- function will be under the influence of the SchedState
newtype SchedMonad a = SchedMonad (ReaderT SchedState IO a)
          deriving ( MonadReader SchedState
                   , Monad
                   , MonadIO
                   , Functor 
                   , Applicative )

runSched :: SchedMonad a -> SchedState -> IO a
runSched (SchedMonad m) = runReaderT m
  

-- Environments and operations thereupon
-- ------------------------------------- 

type Async t = MVar (t, Set MemID)

-- Environment augmented with information about where
-- arrays exist. 
data Env env where
  Aempty :: Env ()
  -- Empty MVar signifies that array has not yet been computed.  
  Apush  :: Arrays t => Env env -> Async t -> Env (env, t)
  -- MVar to wait on for computation of the desired array.
  -- Once array is computed, the set indicates where it exists.
  

-- traverse Env and create E.Aval while doing the transfers.
transferArrays :: forall aenv.
                  A.Array Int DeviceState
               -> DevID 
               -> S.Set (Idx_ aenv)
               -> Env aenv
               -> CUDA.CIO (E.Aval aenv) 
transferArrays alldevices devid dependencies env =
  trav env dependencies 
  where
    allContexts = A.amap devCtx alldevices
    myContext   = allContexts A.! devid
    
    trav :: forall aenv . Env aenv -> S.Set (Idx_ aenv) -> CUDA.CIO (E.Aval aenv)
    trav Aempty _ = return E.Aempty
    trav a@(Apush e tloc) reqset =
      do
        let (needArray,newset) = isNeeded reqset 
        env <- trav e newset

        if needArray
          then
          do
            (t,loc) <- liftIO $ takeMVar tloc
            
            let existsOnDevice = S.member devid loc
            case existsOnDevice of
              True ->
                do
                  -- Here everything should be fine. the
                  -- array is already on the machine.
                  -- Still need to create a valid Async though!
                  -- So, a real event is needed (is it ?) 

                  -- The array already exists here
                  liftIO $ putMVar tloc (t,loc)
                  evt <- liftIO $ E.create 
                  return (E.Apush env (E.Async evt t))
                 
              False ->
                do
                  -- copy arrays into device
                  let srcContext = allContexts A.! (head $ S.elems loc)
                      
                  copyArrays t srcContext myContext
                  evt <- liftIO $ E.create 

                  -- now this array will exist here as well. 
                  liftIO $ putMVar tloc (t,(S.insert devid loc)) 
                  return (E.Apush env (E.Async evt t))
          else
          -- We do not need this array,
          -- So this location in the env will never be touched
          -- by the computation. So putting trash here should be fine. 
          return (E.Apush env dummyAsync)

    -- Is the "current" array needed by the computation.
    -- Figure this out by checking if zero is in the set.
    -- Then "Strengthen" the set, remove zero and decrement all remaining
    -- indices. 
    isNeeded :: Arrays t => S.Set (Idx_ (aenv',t)) -> (Bool,S.Set (Idx_ aenv'))
    isNeeded s =
      let needed = S.member (Idx_ ZeroIdx) s
      in (needed, strengthen s) 

copyArrays :: forall t. Arrays t => t -> Context -> Context -> CUDA.CIO ()
copyArrays arrs src dst = copyArraysR (arrays (undefined :: t)) (fromArr arrs)
  where
    -- MOCK-UP 
    copyArraysR :: ArraysR a -> a -> CUDA.CIO () 
    copyArraysR ArraysRunit () = return ()
    copyArraysR ArraysRarray arr =
      do
        mallocArray arr
        copyArrayPeer arr src arr dst 
        
    copyArraysR (ArraysRpair r1 r2) (arrs1, arrs2) =
      do copyArraysR r1 arrs1
         copyArraysR r2 arrs2 
  
-- Wait for all arrays in the set to be computed.
waitOnArrays :: Env aenv -> S.Set (Idx_ aenv) -> IO ()
waitOnArrays Aempty _ = return ()
waitOnArrays (Apush e tloc) reqset =
  do
    debugMsg $ "Waiting on arrays" 
    let needed = S.member (Idx_ ZeroIdx) reqset
        ns     = strengthen reqset 
    
    case needed of
      True -> do s <- takeMVar tloc
                 putMVar tloc s
                 waitOnArrays e ns
      False -> waitOnArrays e ns
    debugMsg $ "All arrays arrived" 
              
          
-- Maybe actually create a real "this is nothing event" 
nilEvent = CUDA.Event nullPtr 

dummyAsync :: E.Async t 
dummyAsync = E.Async nilEvent undefined


-- Evaluate an PreOpenAcc or ExecAcc or something under the influence
-- of the scheduler
-- ------------------------------------------------------------------

runDelayedAccMulti :: Arrays arrs => DelayedAcc arrs
                   -> IO arrs
runDelayedAccMulti acc =
  do
    CUDA.initialise []
    st <- initScheduler
    flip runSched st $ 
      do mv <- runDelayedOpenAccMulti acc Aempty
         liftIO $ debugMsg $ "Waiting for final result" 
         (arr,ixset) <- liftIO $ takeMVar mv
         -- This means that arr exists on all machines
         -- in ixset
         schedState <- ask 
         let machine = deviceState schedState A.! (head (S.toList ixset))
             mCtx    = devCtx machine
         -- Bind the device context to transfer result array from 
         liftIO $ collectArrs mCtx arr


collectArrs :: forall arrs. Arrays arrs => Context -> arrs -> IO arrs
collectArrs ctx !arrs =
  do
    arrs' <- CUDA.evalCUDA ctx $ collectR (arrays (undefined :: arrs)) (fromArr arrs)
    return $ toArr arrs'   
  where
    collectR :: ArraysR a -> a -> CUDA.CIO a
    collectR ArraysRunit         ()             = return ()
    collectR ArraysRarray        arr            = peekArray arr >> return arr
    collectR (ArraysRpair r1 r2) (arrs1, arrs2) = (,) <$> collectR r1 arrs1
                                                      <*> collectR r2 arrs2

-- This is the multirunner for DelayedOpenAcc
-- ------------------------------------------
runDelayedOpenAccMulti :: Arrays arrs => DelayedOpenAcc aenv arrs
                       -> Env aenv 
                       -> SchedMonad (Async arrs) 
runDelayedOpenAccMulti = traverseAcc 
  where
    traverseAcc :: forall aenv arrs. DelayedOpenAcc aenv arrs
                -> Env aenv
                -> SchedMonad (Async arrs)
    traverseAcc Delayed{} _ = $internalError "runDelayedOpenAccMulti" "unexpected delayed array"
    traverseAcc (Manifest pacc) env =
      case pacc of
        Alet a b ->
          do res <- perform a env
             traverseAcc b (env `Apush` res)


        -- Avar ix  

        -- Atuple (Avar ix, AVar ix2) ->
        -- Atuple tup -> travTup
          
        _ -> perform (Manifest pacc) env 
    
    registerAsFree :: SchedState -> DevID -> IO () 
    registerAsFree st dev = writeChan (freeDevs st) dev 

    perform :: forall aenv arrs . DelayedOpenAcc aenv arrs -> Env aenv -> SchedMonad (Async arrs) 
    perform a env = do
      arrayOnTheWay <- liftIO $ newEmptyMVar
      schedState <- ask

      -- Here! Fork of a thread that waits for all the
      -- arrays that a depends upon to be computed.
      -- Otherwise there will be deadlocks!
      -- This is before deciding what device to use.
      -- Here, spawn off a worker thread ,that is not tied to a device
      -- it is tied to the Work!
      tid <- liftIO $ forkIO $
             do
                     
               -- What arrays are needed to perform this piece of work 
               let dependencies = arrayRefs a
               -- Wait for those arrays to be computed     
               waitOnArrays env dependencies   

               -- Replace following code
               -- with a "getSuitableWorker" function 
               -- Wait for at least one free device.
               devid <- liftIO $ readChan (freeDevs schedState)
               -- To get somewhere, grab head.
                    
               let mydevstate = alldevices A.! devid
                   alldevices = deviceState schedState
                         
               -- Send away work to the device
               debugMsg $ "Launching work on device: " ++ show devid
               liftIO $ putMVar (devWorkMVar mydevstate) $
                 Work $
                 CUDA.evalCUDA (devCtx mydevstate) $
                 -- We are now in CIO 
                 do
                   -- Transfer all arrays to chosen device.
                   liftIO $ debugMsg $ "   Transfer arrays to device " ++ show devid
                   aenv <- transferArrays alldevices devid dependencies env
                   -- Compile workload
                   liftIO $ debugMsg $ "   Compiling OpenAcc" 
                   compiled <- compileOpenAcc a
                   -- Execute workload in a fresh stream and wait for work to finish
                   liftIO $ debugMsg $ "   Executing work on stream"
                   result <- E.streaming (E.executeOpenAcc compiled aenv) E.waitForIt

                   -- Update environment with the result and where it exists
                   liftIO $ debugMsg $ "   Updating environment with computed array" 
                   liftIO $ putMVar arrayOnTheWay (result, S.singleton devid)  
                   -- Work is over! 
                   return () 

               -- wait on the done signal
               debugMsg $ "Waiting for device to report done." 
               Done <- takeMVar (devDoneMVar mydevstate)
               -- This device is now free
               registerAsFree schedState devid
      return arrayOnTheWay 
        

-- Traverse a DelayedOpenAcc and figure out what arrays are being referenced
-- Those arrays must be copied to the device where that DelayedOpenAcc
-- will execute.

arrayRefs :: forall aenv arrs. DelayedOpenAcc aenv arrs -> S.Set (Idx_ aenv) 
arrayRefs (Delayed extent index lin) =
  travE extent `S.union`
  travF index  `S.union`
  travF lin 
--   $internalError "arrayRefs" "unexpected delayed array"
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

    Aforeign _ _ _ -> $internalError "arrayRefs" "Aforeign"

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
    Foreign    _ _ _ -> $internalError "arrayRefsE" "Foreign"
    Let        a b   -> arrayRefsE a `S.union` arrayRefsE b 
    IndexCons  t h   -> arrayRefsE t `S.union` arrayRefsE h
    IndexHead  h     -> arrayRefsE h
    IndexSlice _ x s -> arrayRefsE x `S.union` arrayRefsE s
    IndexFull  _ x s -> arrayRefsE x `S.union` arrayRefsE s
    IndexTail  x     -> arrayRefsE x 
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

    -- travF :: DelayedOpenFun env aenv t -> S.Set (Idx_ aenv)
    -- travF (Body b) = arrayRefsE b
    -- travF (Lam f)  = travF f

    
    

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
