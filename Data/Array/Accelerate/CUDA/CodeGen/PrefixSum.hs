{-# LANGUAGE QuasiQuotes #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.CodeGen.PrefixSum
-- Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
--               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module  Data.Array.Accelerate.CUDA.CodeGen.PrefixSum (

  -- skeletons
  mkScanl, mkScanr,

  -- closets
  scanBlock

) where

import Language.C.Syntax
import Language.C.Quote.CUDA
import Foreign.CUDA.Analysis
import Data.Maybe
import qualified Data.Loc
import qualified Data.Symbol

import Data.Array.Accelerate.CUDA.CodeGen.Base
import Data.Array.Accelerate.CUDA.CodeGen.Monad


data Direction = L | R
  deriving Eq

instance Show Direction where
  show L = "l"
  show R = "r"


mkScanl, mkScanr :: DeviceProperties -> [Type] -> [Exp] -> Maybe [Exp] -> CGM CUTranslSkel
mkScanl = mkScan L
mkScanr = mkScan R


-- [OVERVIEW]
--
-- Data.List-style exclusive scan, with the additional restriction that the
-- first argument needs to be an /associative/ function to enable efficient
-- parallel implementation. The initial value may be arbitrary.
--
-- scanl :: Elt a
--       => (Exp a -> Exp a -> Exp a)
--       -> Exp a
--       -> Acc (Vector a)
--       -> Acc (Vector a)
--
-- > scanl (+) 10 (use xs)
-- >   where
-- >     xs = fromList (Z:.10) (cycle [1])
-- >
-- > ==> Array (Z:.11) [10,11,12,13,14,15,16,17,18,19,20]
--
-- Data.List-style inclusive scan without an initial value
--
-- scanl1 :: Elt a
--        => (Exp a -> Exp a -> Exp a)
--        -> Acc (Vector a)
--        -> Acc (Vector a)
--
-- > scanl1 (+) (use xs)
-- >   where
-- >     xs = fromList (Z:.10) (cycle [1])
-- >
-- > ==> Array (Z:.10) [1,2,3,4,5,6,7,8,9,10]
--
-- Variant of 'scanl' where the final result is returned separately.
--
-- scanl' :: Elt a
--        => (Exp a -> Exp a -> Exp a)
--        -> Exp a
--        -> Acc (Vector a)
--        -> (Acc (Vector a), Acc (Scalar a))
--
-- Denotationally, we have:
--
-- > scanl' f z xs = (init res, last res)
-- >   where
-- >     res = scanl f z xs
--
--
-- [IMPLEMENTATION]
--
-- This code handles all the above cases, in both left and right-handed
-- variants. This is the _downsweep_ phase to a multi-block scan procedure.
-- We require a work distribution such that there is a _single_ thread block for
-- each interval. For multi-block scans, we have an array of interval sums that
-- are used to determine the carry-in value from the previous interval. Note
-- that 'argBlk' will not be accessed by a single-block scan, so may be null.
--
-- We require some pointer manipulation from the calling code in order to
-- support all types of scans:
--
--   * scanl          : argSum should point to the last position of argOut
--   * scanr          : argSum should be the start of argOut, argOut should be incremented by one
--   * scanl1, scanr1 : no change (argSum is required, even though it will not be used Haskell-side)
--   * scanl', scanr' : no change
--
mkScan :: Direction -> DeviceProperties -> [Type] -> [Exp] -> Maybe [Exp] -> CGM CUTranslSkel
mkScan dir dev elt combine mseed = do
  env                                   <- environment
  (argIn0, x0, decl0, getIn0, _)        <- getters 0 elt
  return $ CUTranslSkel name [cunit|
    extern "C"
    __global__ void
    $id:name
    (
        $params:argOut,
        $params:argSum,
        $params:argIn0,
        $params:argBlk,
              typename Ix interval_size,
        const typename Ix num_elements
    )
    {
        $decls:smem
        $decls:decl0
        $decls:decl1

        /*
         * Read in previous result partial sum. We store the carry value in x0
         * and read new values from the input array into x1, since 'scanBlock'
         * will store its results into x1 on completion.
         */
        int carry_in = 0;

        if ( threadIdx.x == 0 ) {
            $stms:(initialise mseed x0)
        }

        const int start = blockIdx.x * interval_size;
        const int end   = min(start + interval_size, num_elements);
        interval_size   = end - start;

        for (int i = threadIdx.x; i < interval_size; i += blockDim.x)
        {
            const int j = $id:(if left then "start + i" else "end - i - 1");
            $stms:(x1 .=. getIn0 "j")

            if (threadIdx.x == 0 && carry_in) {
                $decls:env
                $stms:(x1 .=. combine)
            }

            /*
             * Store our input into shared memory and cooperatively scan
             */
            $stms:(sdata "threadIdx.x" .=. x1)
            __syncthreads();

            $stms:(scanBlock dev elt Nothing (cvar "blockDim.x") sdata env combine)

            if ( $exp:(cbool exclusive) && threadIdx.x != 0 ) {
                $stms:(x1 .=. sdata "threadIdx.x - 1")
            }
            $stms:(setOut "j" x1)

            /*
             * Carry the final result from this block through x0. If this is the
             * last section of the interval, this is the value to write out as
             * the final (reduction) result.
             */
            if ( threadIdx.x == 0 ) {
                const int last = min(interval_size - i, blockDim.x) - 1;
                $stms:(x0 .=. sdata "last")
            }
            carry_in = 1;
        }

        /*
         * for exclusive scans, set the overall scan result and reapply the
         * initial element at the boundaries of each interval
         */
        $stms:(finalise mseed x0)
    }
  |]
  where
    name                        = "scan" ++ show dir ++ maybe "1" (const "") mseed
    (argOut, _, setOut)         = setters elt
    setSum                      = totalSum "0"
    (argSum, totalSum)          = arrays "d_sum" elt
    (argBlk, blkSum)            = arrays "d_blk" elt
    (x1,   decl1)               = locals "x1" elt
    (smem, sdata)               = shared 0 Nothing [cexp| blockDim.x |] elt
    --
    exclusive                   = isJust mseed
    left                        = dir == L
    firstBlock                  = if     left then "0" else "gridDim.x - 1"
    lastBlock                   = if not left then "0" else "gridDim.x - 1"
    --
    initialise Nothing x0       = [cstm|
        if ( blockIdx.x != $id:firstBlock ) {
            $stms:(x0 .=. blkSum (if left then "blockIdx.x - 1" else "blockIdx.x + 1"))
            carry_in = 1;
        }
      |] : []
    initialise (Just seed) x0   = [cstm|
        if ( gridDim.x > 1 ) {
            $stms:(x0 .=. blkSum "blockIdx.x")
        } else {
            $stms:(x0 .=. seed)
        }
      |] : [cstm| carry_in = 1; |] : []
    --
    finalise Nothing     _      = []
    finalise (Just seed) x0     = [[cstm|
      if ( threadIdx.x == 0 ) {
          if ( blockIdx.x == $id:lastBlock ) {
              $stms:(setSum .=. x0)
          }

          if ( num_elements > 0 ) {
              if ( gridDim.x > 1 ) {
                  $stms:(setOut (if left then "start" else "end - 1") (blkSum "blockIdx.x"))
              }
              else {
                  $stms:(setOut (if left then "start" else "end - 1") seed)
              }
          }
      } |]]


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- Introduce some new array arguments and a way to index them
--
arrays :: String -> [Type] -> ([Param], String -> [Exp])
arrays base elt =
  ( zipWith (\t a -> [cparam| $ty:(cptr t) $id:a |]) elt arrs
  , \ix -> map (\a -> [cexp| $id:a [$id:ix] |]) arrs
  )
  where
    n           = length elt
    arrs        = map (\x -> base ++ "_a" ++ show x) [n-1, n-2 .. 0]


-- Scan a block of results in shared memory. We hijack the standard local
-- variables (x0 and x1) for the combination function. This thread must have
-- already stored its initial value into shared memory. The final result for
-- this thread will be stored in x1 as well as the appropriate place in shared
-- memory.
--
scanBlock :: DeviceProperties
          -> [Type]                     -- element type
          -> Maybe Exp                  -- partially-full block bounds check?
          -> Exp                        -- CTA size
          -> (String -> [Exp])          -- index shared memory area
          -> Environment                -- local environment for the..
          -> [Exp]                      -- ..binary function
          -> [Stm]
scanBlock dev elt mlim cta sdata env combine = map (scan . pow2) [0 .. maxThreads]
  where
    maxThreads  = floor (logBase 2 (fromIntegral $ maxThreadsPerBlock dev :: Double)) :: Int
    (x0, _)     = locals "x0" elt
    (x1, _)     = locals "x1" elt
    pow2 x      = (2::Int) ^ x
    scan n      =
      let inrange = maybe [cexp| threadIdx.x >= $int:n|]
                   (\m -> [cexp| threadIdx.x >= $int:n && threadIdx.x < $exp:m |]) mlim
          ix      = "threadIdx.x - " ++ show n
      in
      [cstm|
        if ( $exp:cta > $int:n ) {
            if ( $exp:inrange ) {
                $stms:(x0 .=. sdata ix)
                $decls:env
                $stms:(x1 .=. combine)
            }
            __syncthreads();
            $stms:(sdata "threadIdx.x" .=. x1)
            __syncthreads();
        }
      |]


