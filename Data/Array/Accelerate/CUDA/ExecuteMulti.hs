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

         ) where


import Data.Array.Accelerate.CUDA.AST hiding (Idx_, prj) 
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Compile
import qualified Data.Array.Accelerate.CUDA.Execute as E 
import Data.Array.Accelerate.CUDA.Context 
-- import Data.Array.Accelerate.CUDA.Execute.Stream                ( Stream )

import Data.Array.Accelerate.Trafo  hiding (strengthen) 
import Data.Array.Accelerate.Trafo.Base hiding (inject) 
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
import qualified Foreign.CUDA.Driver.Event as CUDA
-- import Data.Array.Accelerate.CUDA.Analysis.Device

import Foreign.Ptr



import Data.IORef

import Data.Word
import Data.Map as M
import Data.Set as S
import Data.List as L
import Data.Function
import Data.Ord

import qualified Data.Array as A

import Control.Monad.State
import Control.Applicative hiding (Const) 

-- Concurrency 
import Control.Concurrent.MVar
import Control.Concurrent 

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

-- Each device is associated a worker thread
-- that performs workloads passed to it from the
-- Scheduler 
createDeviceThread :: CUDA.Device -> IO DeviceState
createDeviceThread dev =
  do
    ctx <- create dev contextFlags
    
    work <- newEmptyMVar 
    done <- newMVar Done
    
    tid <- forkIO $
           do
             -- Bind the created context to this thread
             -- I assume that means operations within
             -- this thread will default to this context
             CUDA.set (deviceContext ctx)
             -- Enter the workloop 
             deviceLoop done work

    return $ DeviceState ctx done work tid 

    -- The worker thread 
    where deviceLoop done work =
            do
              x <- takeMVar work
              case x of
                ShutDown -> putMVar done Done >> return () 
                Work w -> w

              putMVar done Done 
                    
-- Create the initial SchedState 
initScheduler :: IO SchedState
initScheduler = do
  devs <- enumerateDevices
   -- Create a device thread for each device 
  devs' <- mapM createDeviceThread devs 
  let numDevs = L.length devs

  let assocList = zip [0..numDevs-1] devs'

  return $ SchedState 
                       -- (S.fromList (L.map fst assocList))
                       -- M.empty
                      M.empty
                      S.empty
                      S.empty
                      (A.array (0,numDevs-1) assocList)



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
-- Why not say DevID = MemID
-- Then any device will use memory associated via a lookup somewhere
-- with that DevID.


-- As I understand it, we will first experiment without having a
-- TaskGraph

-- | Scheduler state 
data SchedState =
  SchedState {
--     freeDevs    :: Set DevID         -- ^ Free devices 
--    workingDevs :: Map DevID TaskID  -- ^ Busy devices  
   arrays      :: Map TaskID (Size, Set MemID)
    -- ^ Resident arrays, indexed by the taskID that produced them.
    -- Note, a given array can be on more than one device.  Every
    -- taskId eventually appears in here, so it is also our
    -- "completed" list.
  , waiting     :: Set TaskID -- ^ Work waiting for executor
  , outstanding :: Set TaskID -- ^ Work assigned to executor
      
  -- A map from DevID to the devices available on the system
  -- The CUDA context identifies a device
  , allDevices  :: A.Array Int DeviceState 
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

-- at any point along the traversal of the AST the executeOpenAccMulti
-- function will be under the influence of the SchedState
newtype SchedMonad a = SchedMonad (StateT SchedState IO a)
          deriving ( MonadState SchedState
                   , Monad
                   , MonadIO
                   , Functor 
                   , Applicative )

runSched :: SchedMonad a -> SchedState -> IO a
runSched (SchedMonad m) = evalStateT m
  

-- Environments and operations thereupon
-- ------------------------------------- 

-- Environment augmented with information about where
-- arrays exist. 
data Env env where
  Aempty :: Env ()
  Apush  :: Arrays t => Env env -> (t, MVar (Set MemID)) -> Env (env, t)
  -- MVar to wait on for computation of the desired array.
  -- Once array is computed, the set indicates where it exists.
  
  -- Arrays t => 
      -- Async t 
      -- Async MemID

  -- Array sh a 

-- new approach
-- traverse Env and create E.Aval while doing the transfers.
-- Create Idx_ s during recurse, perform lookups. 


-- ---------------------------------------------------------------
-- TODO: 
-- Gah! Use the strengthening on the set (change S.Set (Idx_ aenv)
-- in each step, so that types match up) and
-- always lookup index zero..

transExperiment :: forall aenv. MemID -> S.Set (Idx_ aenv) -> Env aenv -> SchedMonad (E.Aval aenv)
transExperiment memid ixset aenv =
  do
    -- Assume for now that MemID and DevID are the same.
    -- Which they probably will be in this setup. 
    allDevs <- get >>= (return . allDevices)
    -- Get the context we are moving arrays into.
    -- This should, however, be the context bound
    -- to the thread that is executing this.
    -- If it is not, I have no idea how to allocate memory
    -- in that context. a withContext function or inContext
    -- function could be nice.
    let targetContext = devCtx $ allDevs A.! memid

    -- Traverse environment and copy all arrays
    -- that are needed. 
    trav targetContext aenv ixset 

--   trav _ (Apush Aempty _) = return (Idx_ ZeroIdx, E.Apush E.Aempty undefined )
  where
    trav :: forall aenv . Context -> Env aenv -> S.Set (Idx_ aenv) -> SchedMonad (E.Aval aenv)
    trav _ Aempty _ = return E.Aempty
    trav ctx a@(Apush e (t,loc)) reqset =
      do
        let (needArray,newset) = isNeeded reqset 
        env <- trav ctx e newset

        if needArray
          then
          do
            -- when take this lock ? 
            s <- liftIO $ takeMVar loc
            -- And when release it ? 
            
            let existsOnDevice = S.member memid s 
            case existsOnDevice of
              True ->
                do
                  -- Here everything should be fine. the
                  -- array is already on the machine.
                  -- Still need to create a valid Async though!
                  -- So, a real event is needed

                  -- The array already exists here
                  liftIO $ putMVar loc s
                  
                  return (E.Apush env (E.Async nilEvent t))
                 
              False ->
                do
                  -- copy arrays into device
                  -- Traverse Arrays 
                  copyArrays t s

                  -- now this array will exist here as well. 
                  liftIO $ putMVar loc (S.insert memid s )  
                  return (E.Apush env (E.Async nilEvent undefined))
          else
          -- We do not need this array,
          -- So this location in the env will never be touched
          -- by the computation. So putting trash here should be fine. 
          return (E.Apush env (E.Async nilEvent t))

    -- Is the "current" array needed by the computation.
    -- Figure this out by checking if zero is in the set.
    -- Then "Strengthen" the set, remove zero and decrement all remaining
    -- indices. 
    isNeeded :: Arrays t => S.Set (Idx_ (aenv',t)) -> (Bool,S.Set (Idx_ aenv'))
    isNeeded s =
      let needed = S.member (Idx_ ZeroIdx) s
      in (needed, strengthen s) 

copyArrays :: Arrays t => t -> Set MemID -> SchedMonad ()
copyArrays = undefined
-- do
--    allDevs <- get >>= (return . allDevices)

    
--    intset = S.map idx_ToInt  ixset

-- DANGER ZONE -- DANGER ZONE -- DANGER ZONE -- DANGER ZONE --
    -- dist :: forall aenv. Env aenv -> Int
    -- dist Aempty = -1 
    -- dist (Apush e _) = 1 + dist e 

    -- idx_ToInt :: Idx_ env -> Int
    -- idx_ToInt (Idx_ idx) = idxToInt idx
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 



          
-- OLD EXPERIMENTS 
-- Function that transforms a Env to a Aval
-- Transfer arrays to MemID
-- Transfer those arrays that are in the S.Set (Idx_ aenv)
-- Take Env environment of where all arrays are located
-- Start transfers and output a E.Aval environment (for execution) 
-- transferArrays :: MemID -> S.Set (Idx_ aenv) -> Env aenv -> SchedMonad (E.Aval aenv) 
-- transferArrays memid ixs aenv =
--   do
--     -- Really implement moving of data to where it belongs
--     -- Create real Async t objects for operations depending
--     -- on these to wait on. 
    
--     uploadedEnv <- upload ixlist newEnv 
     
--     return uploadedEnv 
--   where
--     newEnv = nilAval aenv
--     ixlist = S.toList ixs

-- upload :: [Idx_ aenv] -> E.Aval aenv -> SchedMonad (E.Aval aenv) 
-- upload [] env = return env
-- upload (Idx_ x:xs) env =
--   let env' = inject x dummyAsync env
--   in upload xs env'
    
-- -- Not possible with the Idx_ 
-- inject :: Idx env t -> E.Async t -> E.Aval env -> E.Aval env
-- inject ZeroIdx        t  (E.Apush env _) = E.Apush env t
-- inject (SuccIdx idx)  t  (E.Apush env a) = E.Apush (inject idx t env) a
-- inject _ _ _ = $internalError "inject" "Nooo!"

-- prj :: Idx aenv t -> Env aenv -> (t, MVar (Set MemID))
-- prj ZeroIdx  (Apush _ x) = x
-- prj (SuccIdx idx) (Apush env _) = prj idx env
-- prj _ _ = $internalError "prj" "Nooo!" 

nilAval :: Env aenv -> E.Aval aenv
nilAval Aempty = E.Aempty
nilAval (Apush e (t,_)) = E.Apush (nilAval e) (E.Async nilEvent t)

-- Maybe actually create a real "this is nothing event" 
nilEvent = CUDA.Event nullPtr 

dummyAsync :: E.Async t 
dummyAsync = E.Async nilEvent undefined

----------------------------------------------------------------------
-- Info:
-- The type of executeOpenAcc
-- executeOpenAcc
--     :: forall aenv arrs.
--        ExecOpenAcc aenv arrs
--     -> Aval aenv
--     -> Stream
--     -> CIO arrs



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


-- Lots of comments associated with this function
-- contains questions for rest of team to answer.
-- 
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
    traverseAcc (Manifest pacc) env _ =
      case pacc of

        -- Avar ix
        -- look up what is at position ix in the environment. 
        -- When will this happen ?
        --
        -- let a = expensive1
        -- in let b = expensive2 
        -- in Avar ix
        --
        -- The above would mean that array at ix is the result
        -- of the whole program (right ?).
        --
        -- What we want to do in runDelayedOpenAccMulti is
        -- traverse "shallowly" the tree. That is we wont
        -- look into a in let a b, just send it off to compileAcc
        -- and then run it.
        -- So what do we do when we hit an Avar constructor?
        --
        -- Is it safe to assume that if we do hit an Avar constructor
        -- it is the looking up of the result of the program
        -- and maybe it could be handled by more or less doing nothing?
        -- just create some way of reading that array from wherever it
        -- may be ? 
        
        Avar ix -> $internalError "runDelayedOpenAccMulti" "Not implemented"

        -- Let binding.
        --
        -- The format is: 
        -- let real_work in a
        -- So approach we will follow is, enqueue a for computation
        -- keep going into b and keep enqueing work
        
        Alet a b ->
          do
            schedState <- get
            
            let exec_a = compileOpenAcc a

                -- These are the arrays needed for computing a 
                free   = arrayRefs a

                -- we need to keep track of what Arrays have been computed.
                -- How do we identify an array ?
                -- 
                
                
                -- Find out where they are
                -- prefer to launch a on device that
                -- has most of them

                -- execOpenAcc . compileOpenAcc
                
            -- Fire off IO Thread     
            -- Create a CIO (runCIO) 
            -- runWithContext (context is the one from the device chosen) 
                
            -- Create a stream for data copy
            -- Create events for copy complete
            -- Copy 


          
            -- create a stream for kernel execute
            -- create event for execute 
            -- wait for copy complete
            -- compute  (executeOpenAcc . compileOpenAcc)
            -- record event execute done
            -- block on that event. and update free devices 
            

                
                  


            
            -- How can a device report back that it is free.

            -- Can we extend after
            -- after :: Event -> Stream -> IO ()
            -- to
            -- after' :: Event -> Stream -> IO () -> IO ()
            -- So that after Event.wait the passed in IO ()
            -- action is performed. Use that IO action to fill out some MVar. 
            --
            -- Is an event fired off when a device finishes computing?
            -- There is the cudaStreamSynchronize function.
            -- And the cudaEventSynchronize function
            --    I think this is called "block" in Foreign.CUDA

           

                
            
                
            
            -- Choose device to execute this on.
            -- Copy arrays to memory associated with that device 
                
                         
          
            return $ undefined -- $internalError "runDelayedOpenAccMulti" "Not implemented"

        -- Another approach would be launch work
        -- at the operator level. 
        --
        -- This function of course needs to handle these cases
        -- since let a = expensive in map f a
        -- can occur (as an example)
        -- In this case we would enqueu expensive and map f a for
        -- execution.
        
        Map f a -> $internalError "runDelayedOpenAccMulti" "Not implemented"
        -- It does the normal thing:
        --   Free vars
        --   Copy to device
        --   executeOpenAcc
        --
        --   decide where to copy this or not. 

        -- What should we do if we hit a Map here.
        -- Can we make any assumptions about what
        -- constructors we will possibly see, when
        -- doing this "shallow" traversal of the AST ?

        
        
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


    





