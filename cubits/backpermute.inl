/* -----------------------------------------------------------------------------
 *
 * Kernel      : Backpermute
 * Copyright   : [2008..2011] Manuel M T Chakravarty, Gabriele Keller, Sean Lee, Trevor L. McDonell
 * License     : BSD3
 *
 * Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
 * Stability   : experimental
 *
 * Backwards permutation (gather) an array according to the permutation
 * function. The input `shape' is that of the output array.
 *
 *   bpermute :: [a] -> [Int] -> [a]
 *   bpermute v is = [ v!i | i <- is ]
 *
 * ---------------------------------------------------------------------------*/

extern "C"
__global__ void
backpermute
(
    ArrOut              d_out,
    const ArrIn0        d_in0,
    const DimOut        shOut,
    const DimIn0        shIn0
)
{
    const int shapeSize = size(shOut);
    const int gridSize  = __umul24(blockDim.x, gridDim.x);

    for (int ix = __umul24(blockDim.x, blockIdx.x) + threadIdx.x; ix < shapeSize; ix += gridSize)
    {
        DimOut dst = fromIndex(shOut, ix);
        DimIn0 src = project(dst);

        set(d_out, ix, get0(d_in0, toIndex(shIn0, src)));
    }
}

