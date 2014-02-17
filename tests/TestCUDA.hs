
-- | "Test" the main accelerate interpreter.  This is really more a test of the
--   testing infrastructure itself, because the Accelerate interpreter is actually
--   used as the gold standard from which the test "answers" are derived.

module Main where

import Data.Array.Accelerate.BackendKit.Tests (testCompiler, allProgs)
import Data.Array.Accelerate.BackendKit.ConsoleTester 
import Data.Array.Accelerate.BackendClass
import qualified Data.Array.Accelerate.CUDA  as CUDA
import qualified Data.Array.Accelerate.CUDA.Debug as Dbg
import System.Environment (withArgs) 

main :: IO ()
main = do
       -- Simulate command line args passed in for debugging;
       flags <- withArgs ["-fflush-cache", "-dverbose"] $ Dbg.initialise 
       putStrLn$ "Proceeding with Accelerate flags: "++show flags

       makeMain $ BackendTestConf {
         backend  = CUDA.defaultBackend,
         sbackend = Nothing,
         knownTests = KnownBad [],
         extraTests = []
       }
