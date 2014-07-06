{-# LANGUAGE CPP, NamedFieldPuns #-}

-- | This test programs runs a number of very small Accelerate
-- programs through the full compiler (front-end plus CUDA backend),
-- and it checks their answers against the reference interpreter.
-- 
-- The program defined by this module responds to various command line flags.

module Main where 

import Data.Array.Accelerate.BackendKit.ConsoleTester 
import Data.Array.Accelerate.BackendClass (SomeSimpleBackend(..), DropBackend(..))
import Data.Array.Accelerate.CUDA (defaultBackend)

main :: IO ()
main = do 
       -- system "rm -rf .genC_*" -- Remove remaining output from last time, if any
       makeMain $ BackendTestConf { 
         backend  = defaultBackend,
         sbackend = Just (SomeSimpleBackend (DropBackend defaultBackend)),
         knownTests = KnownBad knownProblems,
         extraTests = [],
         frontEndFusion = False
       }

knownProblems :: [String]
knownProblems = words $ "" 
  -- UNFINISHED, not bugs:
  ----------------------------------------
  ++ " p20a p20b p20c " -- UNFINISHED error printed, strides for foldsegs [2014.02.16]
  -- These fali with:
     -- run test 59/88 p20a:: [Failed]
     -- Printed Accelerate result should match expected
     -- expected: "Array (Z :. 2 :. 2) [3.0,12.0,13.0,27.0]"
     --  but got: "Array (Z :. 2 :. 1) [3.0,8.0]"
