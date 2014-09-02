/* -----------------------------------------------------------------------------
 *
 * Module      : Assert
 * Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
 *               [2009..2014] Trevor L. McDonell
 * License     : BSD3
 *
 * Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
 * Stability   : experimental
 *
 * ---------------------------------------------------------------------------*/

#ifndef __ACCELERATE_CUDA_ASSERT_H__
#define __ACCELERATE_CUDA_ASSERT_H__

#include <assert.h>
#include <cuda_runtime.h>

/*
 * assert() is only supported for devices of compute capability 2.0 and higher
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef  assert
#define assert(arg)
#endif

#endif  // __ACCELERATE_CUDA_ASSERT_H__

