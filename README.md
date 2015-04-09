A CUDA Backend for the Accelerate Array Language
================================================

[![Build Status](https://travis-ci.org/tmcdonell/accelerate-cuda.svg?branch=master)](https://travis-ci.org/tmcdonell/accelerate-cuda)

This package compiles Accelerate code down to NVIDIA's CUDA language for general-purpose GPU programming. For details on Accelerate, refer to the [main repository][GitHub]. Please also file bug reports and feature requests with the [issue tracker][Issues] of the main repository.

To use this package, you need a CUDA-enabled NVIDIA GPU and NVIDIA's CUDA SDK version 3.* or later. You can find the SDK at the [NVIDIA Developer Zone][CUDA]. We recommend to use hardware with compute capability 1.2 or greater — see the [table on Wikipedia][wiki-cc].

  [GitHub]:  https://github.com/AccelerateHS/accelerate
  [Issues]:  https://github.com/AccelerateHS/accelerate/issues
  [CUDA]:    http://developer.nvidia.com/cuda-downloads
  [wiki-cc]: http://en.wikipedia.org/wiki/CUDA#Supported_GPUs

