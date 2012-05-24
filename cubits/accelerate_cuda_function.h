/* -----------------------------------------------------------------------------
 *
 * Module      : Function
 * Copyright   : [2008..2010] Manuel M T Chakravarty, Gabriele Keller, Sean Lee
 *               [2009..2012] Manuel M T Chakravarty, Gabriele Keller, Trevor L. McDonell
 * License     : BSD3
 *
 * Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
 * Stability   : experimental
 *
 * ---------------------------------------------------------------------------*/

#ifndef __ACCELERATE_CUDA_FUNCTION_H__
#define __ACCELERATE_CUDA_FUNCTION_H__

#include <cuda_runtime.h>
#include "accelerate_cuda_type.h"

#ifdef __cplusplus

/* -----------------------------------------------------------------------------
 * Device functions required to support generated code
 * -------------------------------------------------------------------------- */

/*
 * Left/Right bitwise rotation
 */
template <typename T>
static __inline__ __device__ T rotateL(const T x, const Int32 i)
{
    const Int32 i8 = i & 8 * sizeof(x) - 1;
    return i8 == 0 ? x : x << i8 | x >> 8 * sizeof(x) - i8;
}

template <typename T>
static __inline__ __device__ T rotateR(const T x, const Int32 i)
{
    const Int32 i8 = i & 8 * sizeof(x) - 1;
    return i8 == 0 ? x : x >> i8 | x << 8 * sizeof(x) - i8;
}

/*
 * Integer division, truncated towards negative infinity
 */
template <typename T>
static __inline__ __device__ T idiv(const T x, const T y)
{
    return x > 0 && y < 0 ? (x - y - 1) / y : (x < 0 && y > 0 ? (x - y + 1) / y : x / y);
}

/*
 * Integer modulus, Haskell style
 */
template <typename T>
static __inline__ __device__ T mod(const T x, const T y)
{
    const T r = x % y;
    return x > 0 && y < 0 || x < 0 && y > 0 ? (r != 0 ? r + y : 0) : r;
}


/*
 * Type coercion
 */
template <typename T>
static __inline__ __device__ Word32 reinterpret32(const T x)
{
    union { T a; Word32 b; } u;

    u.a = x;
    return u.b;
}

template <>
static __inline__ __device__ Word32 reinterpret32(const float x)
{
    return __float_as_int(x);
}

template <typename T>
static __inline__ __device__ Word64 reinterpret64(const T x)
{
    union { T a; Word64 b; } u;

    u.a = x;
    return u.b;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 130
template <>
static __inline__ __device__ Word64 reinterpret64(const double x)
{
    return __double_as_longlong(x);
}
#endif


/*
 * Atomic compare-and-swap, with type coercion
 */
template <typename T>
static __inline__ __device__ T atomicCAS32(T* address, T compare, T val)
{
    union { T a; Word32 b; } u;

    u.b = atomicCAS((Word32*) address, reinterpret32<T>(compare), reinterpret32<T>(val));
    return u.a;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 110
template <>
static __inline__ __device__ Int32 atomicCAS32(Int32* address, Int32 compare, Int32 val)
{
    return atomicCAS(address, compare, val);
}

template <>
static __inline__ __device__ Word32 atomicCAS32(Word32* address, Word32 compare, Word32 val)
{
    return atomicCAS(address, compare, val);
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 120
template <typename T>
static __inline__ __device__ T atomicCAS64(T* address, T compare, T val)
{
    union { T a; Word64 b; } u;

    u.b = atomicCAS((Word64*) address, reinterpret64<T>(compare), reinterpret64<T>(val));
    return u.a;
}

template <>
static __inline__ __device__ Word64 atomicCAS64(Word64* address, Word64 compare, Word64 val)
{
    return atomicCAS(address, compare, val);
}
#endif

#if 0
/* -----------------------------------------------------------------------------
 * Additional helper functions
 * -------------------------------------------------------------------------- */

/*
 * Determine if the input is a power of two
 */
template <typename T>
static __inline__ __host__ __device__ T isPow2(const T x)
{
    return ((x&(x-1)) == 0);
}

/*
 * Compute the next highest power of two
 */
template <typename T>
static __inline__ __host__ T ceilPow2(const T x)
{
#if 0
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
#endif

    return (isPow2(x)) ? x : 1u << (int) ceil(log2((double)x));
}

/*
 * Compute the next lowest power of two
 */
template <typename T>
static __inline__ __host__ T floorPow2(const T x)
{
#if 0
    float nf = (float) n;
    return 1 << (((*(int*)&nf) >> 23) - 127);
#endif

    int exp;
    frexp(x, &exp);
    return 1 << (exp - 1);
}

/*
 * computes next highest multiple of f from x
 */
template <typename T>
static __inline__ __host__ T multiple(const T x, const T f)
{
    return ((x + (f-1)) / f);
}

/*
 * MS Excel-style CEIL() function. Rounds x up to nearest multiple of f
 */
template <typename T>
static __inline__ __host__ T ceiling(const T x, const T f)
{
    return multiple(x, f) * f;
}
#endif

#endif  // __cplusplus
#endif  // __ACCELERATE_CUDA_FUNCTION_H__

