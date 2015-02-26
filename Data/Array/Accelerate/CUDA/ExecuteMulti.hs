{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE EmptyDataDecls             #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE GADTs                      #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE ScopedTypeVariables        #-}
{-# LANGUAGE TemplateHaskell            #-}
{-# LANGUAGE TypeFamilies               #-}
{-# LANGUAGE TypeOperators              #-}
{-# LANGUAGE PatternGuards              #-} 

-- Implementation experiments regarding multidevice execution and
-- scheduling
--
module Data.Array.Accelerate.CUDA.ExecuteMulti
       -- (  
       --   runDelayedOpenAccMulti,
       --   runDelayedAccMulti,
       -- )
       where

import Data.Array.Accelerate.CUDA.AST hiding (Idx_, prj) 
import qualified Data.Array.Accelerate.CUDA.State as CUDA 
import Data.Array.Accelerate.CUDA.Compile
import qualified Data.Array.Accelerate.CUDA.Execute as E
import Data.Array.Accelerate.CUDA.Execute.Stream (Stream)
import qualified Data.Array.Accelerate.CUDA.Execute.Event as E 
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.Array.Data

import Data.Array.Accelerate.Trafo  hiding (strengthen)
import Data.Array.Accelerate.Trafo.Base
-- import Data.Array.Accelerate.Trafo.Base hiding (inject) 
import Data.Array.Accelerate.Array.Sugar
import Data.Array.Accelerate.Analysis.Match


import Data.Array.Accelerate.Error
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Analysis.Device
import qualified Foreign.CUDA.Driver       as CUDA
import qualified Foreign.CUDA.Driver.Event as CUDA

import Foreign.Ptr (nullPtr)

import Data.Function
import Data.List                                                as L
import Data.Ord
import Data.Set                                                 as S
import Data.Typeable
import Data.Maybe
import Data.Map as M 

-- import qualified Data.Array as A
import qualified Data.Array.IArray as A 

import Control.Monad.Reader
import Control.Applicative hiding (Const) 

-- Concurrency 
import Control.Concurrent.MVar
import Control.Concurrent

import System.IO 
import System.IO.Unsafe
import System.Environment 

debug :: Bool
debug = False 

{-# NOINLINE debugLock #-} 
debugLock :: MVar Integer 
debugLock = unsafePerformIO $ keepAlive =<< newMVar 0 

debugMsg :: String -> IO () 
debugMsg str =
  when debug $ 
    modifyMVar_ debugLock $ \n ->
      do 
        hPutStrLn stderr $ show n ++ ": " ++ str
        return (n+1)

debugMsg_ :: String -> IO () 
debugMsg_ str =
  when (not debug) $ 
    modifyMVar_ debugLock $ \n ->
      do 
        hPutStrLn stderr $ show n ++ ": " ++ str
        return (n+1)


-- Datastructures for Gang of worker threads
-- One thread per participating device
-- -----------------------------------------
data Done = Done -- Free to accept work  
data Work = ShutDown
          | Work (() -> IO ()) 

data DeviceState = DeviceState { devCtx      :: !Context
                               , devDoneMVar :: !(MVar Done)
                               , devWorkMVar :: !(MVar Work)
                               , devID       :: !DevID
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

createDeviceThreads :: Maybe [Int] -> Int -> IO [(DevID,DeviceState)]
createDeviceThreads use_devices n  = do
  --realdevs <- enumerateDevices
  realdevs <- selectDevices use_devices 
  
  let numRealDevs = L.length realdevs
      -- virtualized devices 
      devs = concatMap (replicate n) realdevs
      numDevs = L.length devs
  
      
  debugMsg $ "createDeviceThreads: found " ++  -- show numRealDevs ++
             " real devices.\n"  -- ++
             -- "    Creating " ++ show numDevs ++ " devices after virtualization" 
  
  devs' <- zipWithM createDeviceThread [0..] devs
  
  let assocList = zip [0..numDevs-1] devs'
      
  return assocList
  
createDeviceThread :: DevID -> CUDA.Device -> IO DeviceState
createDeviceThread devid dev = do
    debugMsg $ "Creating context on device"
    ctxMVar <- newEmptyMVar 
    
    work <- newEmptyMVar 
    done <- newMVar Done 

    debugMsg $ "Forking device work thread" 
    _ <- runInBoundThread $ forkOn devid $
           do
             !ctx <- create dev contextFlags
             putMVar ctxMVar ctx 
             -- Bind the created context to this thread
             -- I assume that means operations within
             -- this thread will default to this context
             -- CUDA.set (deviceContext ctx)
             -- Enter the workloop 
             deviceLoop done work

    ctx <- takeMVar ctxMVar
    return $ DeviceState ctx done work devid 

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
                     -- w launches of a thread of its own! 
                     w () 
                     debugMsg $ "Exiting work loop"
                     deviceLoop done work
                
-- Register a device as being free.
-- --------------------------------
registerAsFree :: SchedState  -> IO () 
registerAsFree st =
  do
    -- debugMsg $ "***Putting  () on free chan" 
    writeChan (freeDevs st) () 
    -- debugMsg $ "***writeChan complete!" 




-- Initialize a starting state
-- ---------------------------

{- NOINLINE initScheduler #-} 
initScheduler :: SchedState
initScheduler = unsafePerformIO $ keepAlive =<< do
  CUDA.initialise []
  debugMsg $ "InitScheduler"

  var <- lookupEnv "MULTI_USE_DEVICES"
  let use_devices = parseDevs var
  hPutStrLn stderr $ "MULTI_USE_DEVICES: "  ++ show var
  hPutStrLn stderr $ "Devices to use: " ++ show use_devices
  
  -- It is possible to virtualize each device. 
  -- It is not implemented properly though.
  -- We need to create one context per real device
  -- and share between the virtual workers. 
  devs <- createDeviceThreads use_devices 1
  let numDevs = length devs 
      --devids  = L.map fst devs
      
  free <- newChan
  writeList2Chan free [()| _ <- [0..numDevs-1]] 
  debugMsg $ "" -- "Write " ++ show numDevs ++ " ()s into Free devices chan!"

  s_lock <- newMVar () 

  let st = SchedState (A.array (0,numDevs-1) devs)
                      free
                      s_lock 
      
  debugMsg $ "InitScheduler: " -- ++ show st 
  
  return $ st
  where
    parseDevs :: Maybe String -> Maybe [Int]
    parseDevs Nothing = Nothing
    parseDevs (Just []) = Nothing 
    parseDevs (Just xs) = Just $ L.map read $ L.words xs 


-- Scheduler related datatypes
-- ---------------------------
type DevID  = Int 
type MemID  = Int 

data SchedState =
  SchedState {
      deviceState  :: !(A.Array Int DeviceState)
      -- this is a channel, for now. 
    , freeDevs     :: !(Chan ())
      -- as long as there are free devices, there are ()s on this chan. 
    , schedLock    :: !(MVar ())
    }
  deriving Show 

getDeviceCtx :: SchedState -> DevID -> Context
getDeviceCtx st devid = devCtx $ deviceState st A.! devid 
           

-- Need some way of keeping track of actual devices.  What identifies
-- these devices ?  
-- ------------------------------------------------------------------
enumerateDevices :: [Int] -> IO [(CUDA.Device)]
enumerateDevices use_devices = do
  devices    <- mapM CUDA.device use_devices 
  properties <- mapM CUDA.props devices
  return . L.map fst . sortBy (flip cmp `on` snd) $ zip devices properties 
  where
    compute     = computeCapability
    flops d     = multiProcessorCount d * (coresPerMP . deviceResources) d * clockRate d
    cmp x y
      | compute x == compute y  = comparing flops   x y
      | otherwise               = comparing compute x y
                                  
selectDevices :: Maybe [Int] -> IO [(CUDA.Device)]
selectDevices use_devices =
  do num_devices <- CUDA.count 
     let all_devices = [0..num_devices-1] 
     case use_devices of
       Nothing -> enumerateDevices all_devices
       Just devs -> enumerateDevices devs 
                                  

-- What flags to use ?
contextFlags :: [CUDA.ContextFlag]
contextFlags = [CUDA.SchedAuto]

  
-- Environments and operations thereupon
-- -------------------------------------
-- Environment augmented with information about where arrays exist.
data Env env where
  Aempty :: Env ()
  -- Empty MVar signifies that array has not yet been computed.  
  Apush  :: Arrays t => Env env -> Asyncs t -> Env (env, t)

prj :: Idx env t -> Env env -> Asyncs t
prj ZeroIdx        (Apush  _ v)  = v
prj (SuccIdx idx)  (Apush val _) = prj idx val
prj _              _            = $internalError "prj" "inconsistency" 

-- traverse Env and create E.Aval while doing the transfers.
transferArrays :: forall aenv.
                  A.Array Int DeviceState
               -> DevID 
               -> S.Set (Idx_ aenv)
               -> Env aenv
               -> Stream 
               -> CUDA.CIO (E.Aval aenv) 
transferArrays alldevices devid dependencies env stream =
  trav env dependencies stream 
  where
    allContexts = A.amap devCtx alldevices
    -- myContext   = allContexts A.! devid
    
    trav :: Env aenv' -> S.Set (Idx_ aenv') -> Stream -> CUDA.CIO (E.Aval aenv')
    trav Aempty _ _ = return E.Aempty
    trav (Apush e arrs) reqset stream =
      do
        let (needArrays,newset) = isNeeded reqset 
        env' <- trav e newset stream

        if needArrays
          then do 
            !arrs' <- copyArrays arrs allContexts devid stream
            !evt <- liftIO $ E.create 
            return $ E.Apush env' (E.Async evt arrs') 
               
          else return (E.Apush env' dummyAsync)
          
    dummyAsync :: E.Async t
    dummyAsync = E.Async (CUDA.Event nullPtr) undefined
    
    isNeeded :: Arrays t => S.Set (Idx_ (aenv',t)) -> (Bool,S.Set (Idx_ aenv'))
    isNeeded s =
      let needed = S.member (Idx_ ZeroIdx) s
      in (needed, strengthen s) 


copyArrays :: forall t. Arrays t => Asyncs t -> A.Array Int Context -> DevID -> Stream -> CUDA.CIO t
copyArrays (Asyncs !arrs) allContexts devid stream = toArr <$> copyArraysR (arrays (undefined :: t)) arrs stream 
  where
    copyArraysR :: ArraysR a -> AsyncsR a -> Stream -> CUDA.CIO a 
    copyArraysR ArraysRunit A_Unit  stream = return ()
    copyArraysR ArraysRarray (A_Array (Async a)) stream =
      do
        liftIO $ debugMsg "copyArrays: taking mvar" 
        (arr,loc) <- liftIO $ takeMVar a 
        case (devid `S.member` loc) of 
           True ->
             do
               -- Array exists on device.
               liftIO $ debugMsg "copyArrays: putting mvar" 
               liftIO $ putMVar a (arr,loc)
               return arr 
           False -> 
             do let src = allContexts A.! (S.findMin loc) 
                    dst = allContexts A.! devid

                mallocArray arr
                copyArrayPeerAsync arr src arr dst (Just stream)
                liftIO $ debugMsg "copyArrays: putting mvar" 
                liftIO $ putMVar a (arr, S.insert devid loc)
                return arr 
        
    copyArraysR (ArraysRpair r1 r2) (A_Pair arrs1 arrs2) stream =
      do (,) <$> copyArraysR r1 arrs1 stream
             <*> copyArraysR r2 arrs2 stream 
 

-- Wait for arrays while computing a score for
-- each device. A high score means more arrays
-- are computed on that very device. 
waitOnArrays :: Env aenv
                -> S.Set (Idx_ aenv)
                -> IO (M.Map MemID Int)
waitOnArrays env s =
  waitOnArrays' env s (M.empty) 
  where
    waitOnArrays' :: Env aenv
                  -> S.Set (Idx_ aenv)
                  -> M.Map MemID Int
                  -> IO (M.Map MemID Int)

    waitOnArrays' Aempty _ sm = return sm 
    waitOnArrays' (Apush e arr) reqset sm =
      do
        let needed = S.member (Idx_ ZeroIdx) reqset
            ns     = strengthen reqset 
    
        case needed of
          True -> do sm' <- awaitAll arr sm 
                     waitOnArrays' e ns sm'
          False -> waitOnArrays' e ns sm
 
    awaitAll :: forall arrs. Arrays arrs
             => Asyncs arrs -> M.Map MemID Int -> IO (M.Map MemID Int) 
    awaitAll (Asyncs a) sm = go (arrays (undefined :: arrs)) a sm
    
    go :: ArraysR a -> AsyncsR a -> M.Map MemID Int -> IO (M.Map MemID Int) 
    go ArraysRunit         A_Unit sm = return sm
    go (ArraysRpair a1 a2) (A_Pair b1 b2) sm =
      do sm1 <- go a1 b1 sm
         go a2 b2 sm1 
    go ArraysRarray        (A_Array a) sm =
      do
        loc <- waitAsyncLoc a
        let elts = S.elems loc
            sm' = incrementAll elts sm 
        return sm' 
    incrementAll [] sm = sm
    incrementAll (x:xs) sm =
      let sm' = M.alter (\val ->
                          -- I Have a feeling there is a function for this
                          -- in the prelude 
                          case val of
                            Nothing -> Just 1
                            Just a -> Just (a + 1)) x sm
      in  incrementAll xs sm'

                  
-- waitOnArrays :: Env aenv
--              -> S.Set (Idx_ aenv)
--              -> IO (M.Map MemID Int)

-- waitOnArrays :: Env aenv -> S.Set (Idx_ aenv) -> IO ()
-- waitOnArrays Aempty _ = return ()
-- waitOnArrays (Apush e arr) reqset =
--   do
--     let needed = S.member (Idx_ ZeroIdx) reqset
--         ns     = strengthen reqset 
    
--     case needed of
--       True -> do awaitAll arr
--                  waitOnArrays e ns 
--       False -> waitOnArrays e ns
--   where
--     awaitAll :: forall arrs. Arrays arrs => Asyncs arrs -> IO ()
--     awaitAll (Asyncs a) = go (arrays (undefined :: arrs)) a
    
--     go :: ArraysR a -> AsyncsR a -> IO ()
--     go ArraysRunit         A_Unit = return ()
--     go (ArraysRpair a1 a2) (A_Pair b1 b2) =
--       do go a1 b1
--          go a2 b2
--     go ArraysRarray        (A_Array a) =
--       waitAsync a >> return ()


-- Evaluate an PreOpenAcc or ExecAcc or something under the influence
-- of the scheduler
-- ------------------------------------------------------------------
schedState :: SchedState 
schedState = initScheduler

runDelayedAccMulti :: Arrays arrs => DelayedAcc arrs
                   -> Scheduler 
                   -> IO arrs
runDelayedAccMulti !acc scheduler =
  do
    schedState `seq` return () 
    debugMsg $ "runDelayedAccMulti: "
    collectAsyncs =<< runDelayedOpenAccMulti acc Aempty scheduler 

-- Magic
-- ----- 
matchArrayType
    :: forall sh1 sh2 e1 e2. (Shape sh1, Shape sh2, Elt e1, Elt e2)
    => Array sh1 e1 {- dummy -}
    -> Array sh2 e2 {- dummy -}
    -> Maybe (Array sh1 e1 :=: Array sh2 e2)
matchArrayType _ _ -- a1 a2
  | Just REFL <- matchTupleType (eltType (undefined::sh1)) (eltType (undefined::sh2))
  , Just REFL <- matchTupleType (eltType (undefined::e1))  (eltType (undefined::e2))
  = gcast REFL

matchArrayType _ _
  = Nothing


matchArraysR :: ArraysR s -> ArraysR t -> Maybe (s :=: t)
matchArraysR ArraysRunit ArraysRunit
  = Just REFL

matchArraysR (ArraysRarray :: ArraysR s) (ArraysRarray :: ArraysR t)
  | Just REFL <- matchArrayType (undefined::s) (undefined::t)
  = Just REFL

matchArraysR (ArraysRpair s1 s2) (ArraysRpair t1 t2)
  | Just REFL <- matchArraysR s1 t1
  , Just REFL <- matchArraysR s2 t2
  = Just REFL

matchArraysR _ _
  = Nothing

------------------------------------------------------------
-- Schedulers 

type ArrayScore = M.Map MemID Int 

type Scheduler = ArrayScore -> IO DeviceState


-- A silly scheduler that performs no smarts 
sillySched :: Scheduler 
sillySched _ = do
  -- wait for at least one free device 
  () <- readChan (freeDevs schedState) --schedState
  
  withMVar (schedLock schedState) $ \_ ->
   do
    -- Do scheduler work here
    -- Select suitable device
    let alldevices = A.elems $ deviceState schedState
        -- numdevs    = length alldevices
                       
    freedevs <- filterM (\x ->
                          do m <- tryReadMVar (devDoneMVar x)
                             return $ isJust m) alldevices
    let mydev = head freedevs 

    Done <- takeMVar (devDoneMVar mydev)
    return mydev

-- a scheduler that performs a minimum of smarts 
affinitySched :: Scheduler
affinitySched arrayScore = do
  -- wait for at least one free device 
  () <- readChan (freeDevs schedState) --schedState

  withMVar (schedLock schedState)  $ \_ ->
   do 
    let alldevices = A.elems $ deviceState schedState
        -- numdevs    = length alldevices
                       
    freedevs <- filterM (\x ->
                          do m <- tryReadMVar (devDoneMVar x)
                             return $ isJust m) alldevices
    
    case freedevs of
      [] -> $internalError "affinitySched" "no free devices"
      [x] -> takeMVar (devDoneMVar x) >> return x 
      xs ->
        let devScores' = [(x,fromMaybe 0 y) | x <- xs
                         , let y = M.lookup (devID x) arrayScore]
            -- use `on` here 
            devScores = reverse $ L.sortBy (\ (_,b) (_,d) -> b `compare` d) devScores'
            device = fst $ head devScores
        in
         do
--           debugMsg $ "Device Scores: " ++ show (L.map (\ (x,y) -> (devID x,y)) devScores )

           takeMVar (devDoneMVar device) >> return device  
      

-- Even smarter scheduler ?
smartSched :: Scheduler
smartSched = undefined 
  


-- This is the multirunner for DelayedOpenAcc
-- ------------------------------------------
runDelayedOpenAccMulti :: Arrays arrs
                       => DelayedOpenAcc aenv arrs
                       -> Env aenv
                       -> Scheduler
                       -> IO (Asyncs arrs) 
runDelayedOpenAccMulti !acc !aenv scheduler =
  do
    debugMsg $ "runDelayedOpenAccMulti: "
    traverseAcc acc aenv 
  where
    traverseAcc :: forall aenv arrs. Arrays arrs => DelayedOpenAcc aenv arrs
                -> Env aenv
                -> IO (Asyncs arrs)
    traverseAcc Delayed{} _ = $internalError "runDelayedOpenAccMulti" "unexpected delayed array"
    traverseAcc dacc@(Manifest !pacc) env =
      case pacc of
        -- Use a -> a `seq` perform dacc env
        
        Alet a b ->
          do res <- perform a env
             traverseAcc b (env `Apush` res)

        Avar ix -> return $ prj ix env 

        Atuple tup -> travT tup env
          
        _ -> perform (dacc) env 

    -- Traverse Array tuple
    -- --------------------
    travT :: forall aenv arrs. (Arrays arrs, IsAtuple arrs)
          => Atuple (DelayedOpenAcc aenv) (TupleRepr arrs)
          -> Env aenv
          -> IO (Asyncs arrs)
    travT tup env = Asyncs <$> go (arrays (undefined::arrs)) tup
      where
        go :: ArraysR a -> Atuple (DelayedOpenAcc aenv) atup -> IO (AsyncsR a)
        go ArraysRunit NilAtup
          = return A_Unit

        go (ArraysRpair ar2 ar1) (SnocAtup a2 (a1 :: DelayedOpenAcc aenv a1))
          | Just REFL <- matchArraysR ar1 (arrays (undefined :: a1))
          = do
               Asyncs a1' <- traverseAcc a1 env
               a2'        <- go ar2 a2
               return      $ A_Pair a2' a1'

        go _ _
          = $internalError "travT" "unexpected case"

    -- This performs the main part of the scheduler work. 
    -- --------------------------------------------------
    perform :: forall aenv arrs. Arrays arrs => DelayedOpenAcc aenv arrs -> Env aenv -> IO (Asyncs arrs) 
    perform a env = do
      arrayOnTheWay <- asyncs (undefined :: arrs)   

      -- Here! Fork of a thread that waits for all the
      -- arrays that "a" depends upon to be computed.
      -- Otherwise there will be deadlocks!
      -- This is before deciding what device to use.
      -- Here, spawn off a worker thread ,that is not tied to a device
      -- it is tied to the Work!
      -- _ <- runInBoundThread $ forkIO $
      _ <- forkOn 2 $ 
             do
                
               -- What arrays are needed to perform this piece of work 
               let !dependencies = arrayRefs a
               debugMsg $ "" -- "Waiting for: " ++ show (length (S.toList dependencies))
               -- Wait for those arrays to be computed     
               !arrayScore <- waitOnArrays env dependencies   

               -- Wait for at least one free device.
               debugMsg $ "Waiting for a free device"

               -- Let scheduler do its job
               !mydevstate <- scheduler arrayScore 
               
               -- We still need this info for copies 
               let alldevices = deviceState schedState
                   devid      = devID mydevstate 
                                                 
               -- Send away work to the device
               debugMsg $ "" -- "Launching work on device: " ++ show devid
               putMVar (devWorkMVar mydevstate) $
                 Work $ \ () ->  
                 CUDA.evalCUDA (devCtx mydevstate) $
                 -- We are now in CIO 
                 do
                   -- Transfer all arrays to chosen device.
                   -- liftIO $ debugMsg $ "" -- "   Transfer arrays to device: " ++ show devid
                   -- !exec_env <- transferArrays alldevices devid dependencies env
                   -- Compile workload
                   -- liftIO $ debugMsg $ "   Compiling OpenAcc" 
                   
                   -- Execute workload in a fresh stream and wait for work to finish
                   liftIO $ debugMsg $ "   Executing work on stream"
                   !result <- E.streaming
                              (\stream ->
                                do
                                  !exec_env <- transferArrays alldevices devid dependencies env stream
                                  !compiled <- compileOpenAcc a
                                  E.executeOpenAcc compiled exec_env stream) E.waitForIt

                   -- liftIO $ debugMsg $ "" -- "   Transfer arrays to device: " ++ show devid
                   -- !exec_env <- transferArrays alldevices devid dependencies env
                   -- -- Compile workload
                   -- liftIO $ debugMsg $ "   Compiling OpenAcc" 
                   -- !compiled <- compileOpenAcc a
                   -- -- Execute workload in a fresh stream and wait for work to finish
                   -- liftIO $ debugMsg $ "   Executing work on stream"
                   -- !result <- E.streaming (E.executeOpenAcc compiled exec_env) E.waitForIt

                   -- Update environment with the result and where it exists
                   liftIO $ debugMsg $ "   Updating environment with computed array" 
                   --liftIO $ putMVar arrayOnTheWay (result, S.singleton devid)
                   liftIO $ putAsyncs devid result arrayOnTheWay  
                   -- Work is over!

                   -- Do this here for now!
                   liftIO $ debugMsg $ "   Work is done, Write devDone and add to ready queue"
                   liftIO $ putMVar (devDoneMVar mydevstate) Done
                   liftIO $ registerAsFree schedState 
                 
                   --return ()
                   -- DONE

               -- wait on the done signal
               --debugMsg $ "Waiting for device to report done." 
               -- Done <- takeMVar (devDoneMVar mydevstate)
               
               -- putMVar (devDoneMVar mydevstate) Done
               --debugMsg $ "******************************************\n" ++
               --           " Device Reported Done, adding to freeChan.\n" ++
               --           "******************************************" 
                -- schedState devid
      return arrayOnTheWay
        

-- Traverse a DelayedOpenAcc and figure out what arrays are being referenced
-- Those arrays must be copied to the device where that DelayedOpenAcc
-- will execute.
deeplySeq :: forall arrs. ArraysR arrs -> arrs -> arrs 
deeplySeq ArraysRunit         ()         = ()  
deeplySeq (ArraysRpair a1 a2) (a,b)      = let !a' = deeplySeq a1 a
                                               !b' = deeplySeq a2 b
                                           in (a', b')
deeplySeq ArraysRarray        !a         = a



arrayRefs :: forall aenv arrs. DelayedOpenAcc aenv arrs -> S.Set (Idx_ aenv) 
arrayRefs (Delayed extent index lin) =
  travE extent `S.union`
  travF index  `S.union`
  travF lin 
arrayRefs (Manifest pacc) =
  case pacc of
    Use  a    -> (deeplySeq (arrays (undefined :: arrs)) a) `seq` S.empty
    Unit !_   -> S.empty
    
    Avar ix -> addFree ix
    Alet a b -> arrayRefs a `S.union` (strengthen (arrayRefs b))
    
    Apply f a -> arrayRefsAF f `S.union` arrayRefs a 
       
    Atuple tup -> travT tup
    Aprj _ tup -> arrayRefs tup

    Awhile p f a -> arrayRefsAF p `S.union`
                    arrayRefsAF f `S.union`
                    arrayRefs a
    Acond p t e  -> travE p `S.union`
                    arrayRefs t `S.union`
                    arrayRefs e

    Aforeign _ _ a -> arrayRefs a -- $internalError "arrayRefs" "Aforeign"

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

    Collect _ -> $internalError "arrayRefs" "Collect" 
                 
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
arrayRefsE expr =
  case expr of
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
    Foreign    _ _ a -> travE a 
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

-- --------------------------------------------------
--
-- --------------------------------------------------

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


-- Async and Asyncs
-- ---------------- 
data Async t = Async (MVar (t, Set MemID))

newEmptyAsync :: IO (Async t)
newEmptyAsync = Async <$> newEmptyMVar 

-- wait on an Async 
waitAsync :: Async t -> IO ()
waitAsync a =
  waitAsyncLoc a >> return ()  

-- wait on an Async and provide its location data 
waitAsyncLoc :: Async t -> IO (Set MemID)
waitAsyncLoc (Async tloc) =
  withMVar tloc $ \ (_,loc) ->
    return loc

waitAsyncTLoc :: Async t -> IO (t,Set MemID)
waitAsyncTLoc (Async tloc) =
  withMVar tloc $ \ (t,loc) ->
    return (t,loc)

takeAsync :: Async t -> IO (t,Set MemID)
takeAsync (Async tloc) = takeMVar tloc 


data Asyncs a where
  Asyncs :: AsyncsR (ArrRepr a)
         -> Asyncs a 

data family AsyncsR :: * -> * 
data instance AsyncsR ()            = A_Unit
data instance AsyncsR (Array sh e)  = A_Array (Async (Array sh e))
data instance AsyncsR (a,b)         = A_Pair (AsyncsR a) (AsyncsR b)

-- This is mysterious to me ! 
asyncs :: forall a. Arrays a => a -> IO (Asyncs a)
asyncs a =
  do Asyncs <$> go (arrays a) -- (undefined :: a)) 
  where
    go :: ArraysR t -> IO (AsyncsR t)
    go ArraysRunit         = return A_Unit
    go (ArraysRpair a1 a2) = A_Pair <$> go a1 <*> go a2
    go ArraysRarray        = A_Array <$> newEmptyAsync  
    


-- Fill in an Async..
-- Assumes the Async is empty! 
putAsyncs :: forall arrs. Arrays arrs => DevID -> arrs -> Asyncs arrs -> IO ()
putAsyncs devid a (Asyncs !arrs) =
  go (arrays (undefined :: arrs)) (fromArr a) arrs
  where
    go :: ArraysR a -> a -> AsyncsR a -> IO ()
    go ArraysRunit         ()     A_Unit    = return ()
    go (ArraysRpair a1 a2) (b1,b2) (A_Pair c1 c2) = 
      do go a1 b1 c1
         go a2 b2 c2
    go ArraysRarray        !arr     (A_Array (Async a'))  =
      do
         --debugMsg "putAsyncs: adding array to env"
         putMVar a' (arr,S.singleton devid)
         --debugMsg "putAsyncs: DONE!" 

-- copy a collection of Async arrays back to host
collectAsyncs :: forall arrs. Arrays arrs => Asyncs arrs -> IO arrs
collectAsyncs (Asyncs !arrs) =
  do debugMsg "Collecting result arrays" 
     !arrs' <- collectR (arrays (undefined :: arrs)) arrs
     debugMsg "Collecting result arrays: DONE!"
     return $ toArr arrs'
  where
    collectR :: ArraysR a -> AsyncsR a -> IO a
    collectR ArraysRunit         A_Unit         = return () 
    collectR (ArraysRpair a1 a2) (A_Pair a b)   = (,) <$> collectR a1 a <*> collectR a2 b
    collectR ArraysRarray        (A_Array a)    =
      do
        -- Take out and do not put back, we are done with this
        -- here 
        -- !(t,loc) <- takeAsync a
        !(t,loc) <- waitAsyncTLoc a 
        -- get min from set (because the min device is likely to the
        -- most capable device) 
        let devid = S.findMin loc
            ctx = getDeviceCtx schedState devid 

        -- Copy out from device 
        CUDA.evalCUDA ctx $ peekArray t >> return t 

------------------------------------------------------- 

