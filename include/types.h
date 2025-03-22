#pragma once
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef CUDA_MODE
typedef CUdeviceptr ypred_t;
#else
typedef float* ypred_t;
#endif
