{-# LANGUAGE TypeFamilies #-} 
{-# LANGUAGE DeriveDataTypeable #-} 

module Data.Array.Accelerate.SimpleCUDA where




import           Data.Array.Accelerate.BackendClass
import qualified Data.Array.Accelerate.BackendKit.IRs.SimpleAcc   as S
import qualified Data.Array.Accelerate.BackendKit.SimpleArray     as SA
import           Data.Array.Accelerate.BackendKit.Utils.Helpers (dbgPrint, dbg)
--import           Data.Array.Accelerate.BackendKit.CompilerPipeline
--  (phase0, phase1, phase2, repackAcc, unpackArray, Phantom(..), defaultTrafoConfig)

 
import Data.Array.Accelerate.CUDA.AST ( ExecAcc(..)  ) -- a candidate for the role of a Blob
import Data.Array.Accelerate.CUDA.State  -- holds evalCUDA, defaultContext

import Data.Array.Accelerate.CUDA.CompileSimpleCUDA
import Data.Array.Accelerate.CUDA.ExecuteSimpleCUDA
import Data.Typeable (Typeable)

import Control.Monad 
import System.IO.Unsafe (unsafePerformIO) 

-- TODO: Implement a SimpleBackend instance for CUDA.

---------------------------------------------------------------------------
-- Backend and Blob 

data SimpleCUDABackend = SimpleCUDABackend
                         deriving (Show, Typeable) 

data SimpleCUDABlob = SimpleCUDABlob ExecAcc  -- not sure what to put here! 
                
---------------------------------------------------------------------------
-- SimpleBackend instance

instance SimpleBackend SimpleCUDABackend where
  type SimpleRemote SimpleCUDABackend = [SA.AccArray]  -- Guess 1
  type SimpleBlob   SimpleCUDABackend = SimpleCUDABlob   

  -- SACC = Data.Array.Accelerate.BackendKit.IRs.SimpleAcc 
  -- simpleCompile :: b -> FilePath -> SACC.Prog () -> IO (SimpleBlob b)
  simpleCompile _ path prog = do b <- evalCUDA defaultContext (compileSimpleAcc prog )
                                 return $ SimpleCUDABlob b 

  --simpleCompile :: b -> FilePath -> SACC.Prog () -> IO (SimpleBlob b)

  --simpleRunRaw :: b -> DebugName -> SACC.Prog () -> Maybe (SimpleBlob b) -> IO [SimpleRemote b]
  simpleRunRaw b mname prog (Just (SimpleCUDABlob blob)) = evalCUDA defaultContext $ executeSimpleAcc blob
  simpleRunRaw b mname prog Nothing = undefined

  --simpleRunRawFun1 :: b -> Int -> ([SACC.AVar] -> SACC.Prog ()) -> Maybe (SimpleBlob b) -> [SimpleRemote b] -> IO [SimpleRemote b]            

  -- simpleCopyToHost :: b -> SimpleRemote b -> IO SACC.AccArray
  simpleCopyToHost _b [arr] = undefined

  --simpleCopyToDevice :: b -> SACC.AccArray -> IO (SimpleRemote b)
  simpleCopyToDevice _b arr = undefined

  --simpleCopyToPeer :: b -> SimpleRemote b -> IO (SimpleRemote b)
  simpleCopyToPeer _ x = undefined

  --simpleUseRemote :: b -> SimpleRemote b -> IO SACC.AExp
  simpleUseRemote _ [arr] = undefined

  --simpleWaitRemote :: b -> SimpleRemote b -> IO ()
  simpleWaitRemote _ _ = undefined

  --simpleSeparateMemorySpace :: b -> Bool
  simpleSeparateMemorySpace _ = undefined
  


                                        
