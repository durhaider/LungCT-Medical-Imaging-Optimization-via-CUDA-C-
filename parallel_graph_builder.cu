#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * Parallel edge-based graph construction
 * Generic algorithm for building vertex adjacency from triangle mesh
 */

__global__ void ComputeVertexDegreeKernel(
    const int3* triangles,
    int numTriangles,
    int* vertexDegrees
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= numTriangles) return;
    
    int3 tri = triangles[tid];
    
    // Each triangle contributes 2 neighbors per vertex
    atomicAdd(&vertexDegrees[tri.x], 2);
    atomicAdd(&vertexDegrees[tri.y], 2);
    atomicAdd(&vertexDegrees[tri.z], 2);
}

__global__ void BuildEdgeListKernel(
    const int3* triangles,
    int numTriangles,
    const int* offsets,
    int* edgeList,
    int* writePositions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= numTriangles) return;
    
    int3 tri = triangles[tid];
    
    // Insert 6 directed edges (3 undirected edges = 6 directed)
    int pos;
    
    // Edge v0 -> v1
    pos = atomicAdd(&writePositions[tri.x], 1);
    edgeList[offsets[tri.x] + pos] = tri.y;
    
    // Edge v1 -> v0
    pos = atomicAdd(&writePositions[tri.y], 1);
    edgeList[offsets[tri.y] + pos] = tri.x;
    
    // Edge v1 -> v2
    pos = atomicAdd(&writePositions[tri.y], 1);
    edgeList[offsets[tri.y] + pos] = tri.z;
    
    // Edge v2 -> v1
    pos = atomicAdd(&writePositions[tri.z], 1);
    edgeList[offsets[tri.z] + pos] = tri.y;
    
    // Edge v2 -> v0
    pos = atomicAdd(&writePositions[tri.z], 1);
    edgeList[offsets[tri.z] + pos] = tri.x;
    
    // Edge v0 -> v2
    pos = atomicAdd(&writePositions[tri.x], 1);
    edgeList[offsets[tri.x] + pos] = tri.z;
}

__global__ void RemoveDuplicateEdgesKernel(
    int* edgeList,
    const int* offsets,
    int numVertices,
    int* compactedOffsets
) {
    int vertexId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vertexId >= numVertices) return;
    
    int start = offsets[vertexId];
    int end = offsets[vertexId + 1];
    int count = end - start;
    
    if (count == 0) {
        compactedOffsets[vertexId] = 0;
        return;
    }
    
    // Simple bubble sort for small neighbor lists (typically < 20)
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (edgeList[start + i] > edgeList[start + j]) {
                int temp = edgeList[start + i];
                edgeList[start + i] = edgeList[start + j];
                edgeList[start + j] = temp;
            }
        }
    }
    
    // Remove duplicates
    int writePos = start;
    int lastValue = edgeList[start];
    
    for (int i = start + 1; i < end; i++) {
        if (edgeList[i] != lastValue) {
            writePos++;
            edgeList[writePos] = edgeList[i];
            lastValue = edgeList[i];
        }
    }
    
    compactedOffsets[vertexId] = writePos - start + 1;
}
