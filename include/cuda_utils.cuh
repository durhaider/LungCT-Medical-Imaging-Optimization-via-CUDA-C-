#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * CUDA error checking macros
 */

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d\n", __FILE__, __LINE__); \
        fprintf(stderr, "  %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_LAST_ERROR() \
do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA KERNEL ERROR] %s:%d\n", __FILE__, __LINE__); \
        fprintf(stderr, "  %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * Device capability checker
 */
inline bool CheckComputeCapability(int major, int minor) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int deviceCapability = prop.major * 10 + prop.minor;
    int requiredCapability = major * 10 + minor;
    
    return deviceCapability >= requiredCapability;
}

/**
 * Memory availability checker
 */
inline bool CheckAvailableMemory(size_t requiredBytes) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    if (requiredBytes > free) {
        fprintf(stderr, "[GPU MEM] Insufficient memory\n");
        fprintf(stderr, "  Required: %.2f GB\n", requiredBytes / 1e9);
        fprintf(stderr, "  Available: %.2f GB\n", free / 1e9);
        return false;
    }
    
    return true;
}

/**
 * Optimal block/grid size calculator
 */
inline dim3 CalculateGridSize(int totalThreads, int blockSize = 256) {
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    return dim3(numBlocks);
}

/**
 * GPU info printer
 */
inline void PrintDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("\n[GPU %d] %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  SM Count: %d\n", prop.multiProcessorCount);
        printf("  Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Shared Memory/Block: %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  Warp Size: %d\n", prop.warpSize);
    }
}
