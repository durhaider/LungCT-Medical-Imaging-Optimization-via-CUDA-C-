#include <cuda_runtime.h>

/**
 * Parallel sum reduction using shared memory
 * Generic pattern for aggregating data on GPU
 */

template<typename T, int BLOCK_SIZE>
__global__ void SumReductionKernel(
    const T* input,
    T* output,
    int n
) {
    extern __shared__ T sharedData[];
    
    int tid = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sharedData[tid] = (globalId < n) ? input[globalId] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

/**
 * Parallel min/max reduction
 */
template<typename T>
__global__ void MinMaxReductionKernel(
    const T* input,
    T* minOutput,
    T* maxOutput,
    int n
) {
    extern __shared__ T sharedMin[];
    T* sharedMax = &sharedMin[blockDim.x];
    
    int tid = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (globalId < n) {
        sharedMin[tid] = input[globalId];
        sharedMax[tid] = input[globalId];
    } else {
        sharedMin[tid] = 1e30f;  // Large value for min
        sharedMax[tid] = -1e30f; // Small value for max
    }
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMin[tid] = min(sharedMin[tid], sharedMin[tid + stride]);
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        minOutput[blockIdx.x] = sharedMin[0];
        maxOutput[blockIdx.x] = sharedMax[0];
    }
}

/**
 * Parallel prefix sum (scan) - work-efficient algorithm
 */
__global__ void ExclusiveScanKernel(
    const int* input,
    int* output,
    int n
) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    temp[tid] = (globalId < n) ? input[globalId] : 0;
    __syncthreads();
    
    // Up-sweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Clear last element
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int tmp = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += tmp;
        }
        __syncthreads();
    }
    
    // Write result
    if (globalId < n) {
        output[globalId] = temp[tid];
    }
}
```

---

## **Push these to GitHub as:**
```
cuda_samples/
├── parallel_graph_builder.cu      # Adjacency construction
├── parallel_smoothing.cu          # Laplacian operator
├── gpu_memory_pool.cu             # Memory management
├── cuda_utils.cuh                 # Error checking
└── parallel_reduction.cu          # Reduction patterns
