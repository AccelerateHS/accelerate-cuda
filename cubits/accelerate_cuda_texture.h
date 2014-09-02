/* -----------------------------------------------------------------------------
 *
 * Module      : Texture
 * Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
 *               [2009..2014] Trevor L. McDonell
 * License     : BSD3
 *
 * Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
 * Stability   : experimental
 *
 * CUDA texture definitions and access functions are defined in terms of
 * templates, and hence only available through the C++ interface.
 *
 * We don't have a definition for `Int' or `Word', since the bitwidth of the
 * Haskell and C types may be different.
 *
 * NOTE ON 64-BIT TYPES
 *   The CUDA device uses little-endian arithmetic. We haven't accounted for the
 *   fact that this may be different on the host, for both initial data transfer
 *   and the unpacking below.
 *
 * ---------------------------------------------------------------------------*/

#ifndef __ACCELERATE_CUDA_TEXTURE_H__
#define __ACCELERATE_CUDA_TEXTURE_H__

#include <cuda_runtime.h>
#include "accelerate_cuda_type.h"

#if defined(__cplusplus) && defined(__CUDACC__)

typedef texture<uint2,  1> TexWord64;
typedef texture<Word32, 1> TexWord32;
typedef texture<Word16, 1> TexWord16;
typedef texture<Word8,  1> TexWord8;
typedef texture<int2,   1> TexInt64;
typedef texture<Int32,  1> TexInt32;
typedef texture<Int16,  1> TexInt16;
typedef texture<Int8,   1> TexInt8;
typedef texture<float,  1> TexFloat;
typedef texture<char,   1> TexCChar;

static __inline__ __device__  Word8 indexArray(TexWord8  t, const int x) { return tex1Dfetch(t,x); }
static __inline__ __device__ Word16 indexArray(TexWord16 t, const int x) { return tex1Dfetch(t,x); }
static __inline__ __device__ Word32 indexArray(TexWord32 t, const int x) { return tex1Dfetch(t,x); }
static __inline__ __device__ Word64 indexArray(TexWord64 t, const int x)
{
  union { uint2 x; Word64 y; } v;
  v.x = tex1Dfetch(t,x);
  return v.y;
}

static __inline__ __device__  Int8 indexArray(TexInt8  t, const int x) { return tex1Dfetch(t,x); }
static __inline__ __device__ Int16 indexArray(TexInt16 t, const int x) { return tex1Dfetch(t,x); }
static __inline__ __device__ Int32 indexArray(TexInt32 t, const int x) { return tex1Dfetch(t,x); }
static __inline__ __device__ Int64 indexArray(TexInt64 t, const int x)
{
  union { int2 x; Int64 y; } v;
  v.x = tex1Dfetch(t,x);
  return v.y;
}

static __inline__ __device__ float indexArray(TexFloat  t, const int x) { return tex1Dfetch(t,x); }
static __inline__ __device__ char  indexArray(TexCChar  t, const int x) { return tex1Dfetch(t,x); }

#if defined(__LP64__)
typedef TexInt64  TexCLong;
typedef TexWord64 TexCULong;
#else
typedef TexInt32  TexCLong;
typedef TexWord32 TexCULong;
#endif

/*
 * Synonyms for C-types. NVCC will force Ints to be 32-bits.
 */
typedef TexInt8   TexCSChar;
typedef TexWord8  TexCUChar;
typedef TexInt16  TexCShort;
typedef TexWord16 TexCUShort;
typedef TexInt32  TexCInt;
typedef TexWord32 TexCUInt;
typedef TexInt64  TexCLLong;
typedef TexWord64 TexCULLong;
typedef TexFloat  TexCFloat;

/*
 * Doubles, only available when compiled for Compute 1.3 and greater
 */
typedef texture<int2, 1> TexDouble;
typedef TexDouble        TexCDouble;

#if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
static __inline__ __device__ double indexDArray(TexDouble t, const int x)
{
  int2 v = tex1Dfetch(t,x);
  return __hiloint2double(v.y,v.x);
}
#endif

#endif  // defined(__cplusplus) && defined(__CUDACC__)
#endif  // __ACCELERATE_CUDA_TEXTURE_H__

