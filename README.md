A CUDA Backend for the Accelerate Array Language
================================================

[![Build Status](https://travis-ci.org/AccelerateHS/accelerate-cuda.svg?branch=master)](https://travis-ci.org/AccelerateHS/accelerate-cuda)

***NOTE:***
_This package is being deprecated in favour of
[accelerate-llvm][github-accelerate-llvm], which supports execution on multicore
CPUs as well as CUDA-capable GPUs. See the
[accelerate-llvm][github-accelerate-llvm] package for details._

___


This package compiles Accelerate code down to NVIDIA's CUDA language for general-purpose GPU programming. For details on Accelerate, refer to the [main repository][github-accelerate]. Please also file bug reports and feature requests with the [issue tracker][accelerate-issues] of the main repository.

To use this package, you need a CUDA-enabled NVIDIA GPU and NVIDIA's CUDA SDK version 3.* or later. You can find the SDK at the [NVIDIA Developer Zone][CUDA]. We recommend to use hardware with compute capability 1.2 or greater — see the [table on Wikipedia][wiki-cc].

  [github-accelerate-llvm]: https://github.com/AccelerateHS/accelerate-llvm
  [github-accelerate]:      https://github.com/AccelerateHS/accelerate
  [accelerate-issues]:      https://github.com/AccelerateHS/accelerate/issues
  [CUDA]:                   http://developer.nvidia.com/cuda-downloads
  [wiki-cc]:                http://en.wikipedia.org/wiki/CUDA#Supported_GPUs

