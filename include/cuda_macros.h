#pragma once
#include <iostream>
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char* errorString; \
            cuGetErrorString(result, &errorString); \
            std::cerr << "CUDA error: " << errorString << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_NVRTC(call) \
    do { \
        nvrtcResult result = call; \
        if (result != NVRTC_SUCCESS) { \
            std::cerr << "NVRTC error: " << nvrtcGetErrorString(result) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#ifdef USE_CUDA
#define CUDA_MODE
#endif
