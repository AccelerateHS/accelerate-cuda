
-- | "Test" the main accelerate interpreter.  This is really more a test of the
--   testing infrastructure itself, because the Accelerate interpreter is actually
--   used as the gold standard from which the test "answers" are derived.

module Main where

import Data.Array.Accelerate.BackendKit.Tests (testCompiler, allProgs)
import Data.Array.Accelerate.BackendKit.ConsoleTester 
import Data.Array.Accelerate.BackendClass
import qualified Data.Array.Accelerate.CUDA as CUDA

main :: IO ()
main = do
       makeMain $ BackendTestConf {
         backend  = CUDA.defaultBackend,
         sbackend = Nothing,
         knownTests = KnownBad [],
         extraTests = []
       }
