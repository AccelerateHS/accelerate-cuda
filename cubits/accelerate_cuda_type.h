/* -----------------------------------------------------------------------------
 *
 * Module      : Type
 * Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
 *               [2009..2014] Trevor L. McDonell
 * License     : BSD3
 *
 * Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
 * Stability   : experimental
 *
 * ---------------------------------------------------------------------------*/

#ifndef __ACCELERATE_CUDA_TYPE_H__
#define __ACCELERATE_CUDA_TYPE_H__

/*
 * The word size on CUDA devices is always 32-bits. If the host is a 64-bit
 * architecture, the system stdint.h implementation can define these incorrectly
 * for device code.
 *
 */
typedef signed char            Int8;
typedef short                 Int16;
typedef int                   Int32;
typedef long long             Int64;

typedef unsigned char         Word8;
typedef unsigned short       Word16;
typedef unsigned int         Word32;
typedef unsigned long long   Word64;

#endif

