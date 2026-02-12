#include <cuda_runtime.h>
#include <stdio.h>

/**
 * GPU memory pool for efficient allocation
 * Reduces cudaMalloc/cudaFree overhead
 */

class DeviceMemoryPool {
private:
    void* basePtr;
    size_t totalSize;
    size_t currentOffset;
    
public:
    DeviceMemoryPool(size_t sizeBytes) {
        cudaMalloc(&basePtr, sizeBytes);
        totalSize = sizeBytes;
        currentOffset = 0;
        
        printf("[GPU POOL] Allocated %.2f MB\n", sizeBytes / 1024.0 / 1024.0);
    }
    
    ~DeviceMemoryPool() {
        cudaFree(basePtr);
    }
    
    void* Allocate(size_t bytes) {
        // Align to 256 bytes for coalesced access
        size_t alignedBytes = (bytes + 255) & ~255;
        
        if (currentOffset + alignedBytes > totalSize) {
            fprintf(stderr, "[GPU POOL ERROR] Out of memory\n");
            return nullptr;
        }
        
        void* ptr = (char*)basePtr + currentOffset;
        currentOffset += alignedBytes;
        
        return ptr;
    }
    
    void Reset() {
        currentOffset = 0;
    }
    
    size_t GetUsage() const {
        return currentOffset;
    }
    
    float GetUsagePercent() const {
        return 100.0f * currentOffset / totalSize;
    }
};

/**
 * Async memory transfer manager
 */
class AsyncMemoryTransfer {
private:
    cudaStream_t stream;
    
public:
    AsyncMemoryTransfer() {
        cudaStreamCreate(&stream);
    }
    
    ~AsyncMemoryTransfer() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    
    template<typename T>
    void TransferToDevice(T* d_dst, const T* h_src, size_t count) {
        cudaMemcpyAsync(
            d_dst, h_src,
            count * sizeof(T),
            cudaMemcpyHostToDevice,
            stream
        );
    }
    
    template<typename T>
    void TransferToHost(T* h_dst, const T* d_src, size_t count) {
        cudaMemcpyAsync(
            h_dst, d_src,
            count * sizeof(T),
            cudaMemcpyDeviceToHost,
            stream
        );
    }
    
    void Synchronize() {
        cudaStreamSynchronize(stream);
    }
};

/**
 * Pinned memory allocator for zero-copy transfers
 */
template<typename T>
class PinnedHostBuffer {
private:
    T* hostPtr;
    size_t count;
    
public:
    PinnedHostBuffer(size_t n) : count(n) {
        cudaMallocHost(&hostPtr, n * sizeof(T));
    }
    
    ~PinnedHostBuffer() {
        cudaFreeHost(hostPtr);
    }
    
    T* Get() { return hostPtr; }
    const T* Get() const { return hostPtr; }
    size_t Size() const { return count; }
};
