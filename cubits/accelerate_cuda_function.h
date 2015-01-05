/* -----------------------------------------------------------------------------
 *
 * Module      : Function
 * Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
 *               [2009..2014] Trevor L. McDonell
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
__inline__ __device__ Word32 reinterpret32(const Word32 x)
{
    return x;
}

template <>
__inline__ __device__ Word32 reinterpret32(const float x)
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

template <>
__inline__ __device__ Word64 reinterpret64(const Word64 x)
{
    return x;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 130
template <>
__inline__ __device__ Word64 reinterpret64(const double x)
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
__inline__ __device__ Int32 atomicCAS32(Int32* address, Int32 compare, Int32 val)
{
    return atomicCAS(address, compare, val);
}

template <>
__inline__ __device__ Word32 atomicCAS32(Word32* address, Word32 compare, Word32 val)
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
__inline__ __device__ Word64 atomicCAS64(Word64* address, Word64 compare, Word64 val)
{
    return atomicCAS(address, compare, val);
}
#endif


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300
/*
 * Warp shuffle functions
 */
template <typename T>
static __inline__ __device__ T shfl_up32(T var, unsigned int delta, int width=warpSize)
{
    union { T a; Int32 b; } u, v;

    v.a = var;
    u.b = __shfl_up(v.b, delta, warpSize);

    return u.a;
}

template <>
__inline__ __device__ int shfl_up32(int var, unsigned int delta, int width)
{
    return __shfl_up(var, delta, width);
}

template <>
__inline__ __device__ float shfl_up32(float var, unsigned int delta, int width)
{
    return __shfl_up(var, delta, width);
}


template <typename T>
static __inline__ __device__ T shfl_up64(T var, unsigned int delta, int width=warpSize)
{
    union { T a; struct { Int32 lo; Int32 hi; }; } u, v;

    v.a  = var;
    u.lo = __shfl_up(v.lo, delta, warpSize);
    u.hi = __shfl_up(v.hi, delta, warpSize);

    return u.a;
}


template <typename T>
static __inline__ __device__ T shfl_xor32(T var, int laneMask, int width=warpSize)
{
    union { T a; Int32 b; } u, v;

    v.a = var;
    u.b = __shfl_xor(v.b, laneMask, warpSize);

    return u.a;
}

template <>
__inline__ __device__ int shfl_xor32(int var, int laneMask, int width)
{
    return __shfl_xor(var, laneMask, width);
}

template <>
__inline__ __device__ float shfl_xor32(float var, int laneMask, int width)
{
    return __shfl_xor(var, laneMask, width);
}


template <typename T>
static __inline__ __device__ T shfl_xor64(T var, int laneMask, int width=warpSize)
{
    union { T a; struct { Int32 lo; Int32 hi; }; } u, v;

    v.a  = var;
    u.lo = __shfl_xor(v.lo, laneMask, warpSize);
    u.hi = __shfl_xor(v.hi, laneMask, warpSize);

    return u.a;
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

