#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * Parallel geometric Laplacian computation
 * Generic smoothing operator for 3D point clouds
 */

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__global__ void ComputeLaplacianKernel(
    const float3* points,
    const int* adjacency,
    const int* offsets,
    float3* laplacianVectors,
    int numPoints
) {
    int pointId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pointId >= numPoints) return;
    
    float3 point = points[pointId];
    
    int start = offsets[pointId];
    int end = offsets[pointId + 1];
    int neighborCount = end - start;
    
    if (neighborCount == 0) {
        laplacianVectors[pointId] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    
    // Compute centroid of neighbors
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int i = start; i < end; i++) {
        int neighborId = adjacency[i];
        centroid = centroid + points[neighborId];
    }
    
    centroid = centroid * (1.0f / neighborCount);
    
    // Laplacian = centroid - current point
    laplacianVectors[pointId] = centroid - point;
}

__global__ void ApplySmoothingKernel(
    const float3* points,
    const float3* laplacianVectors,
    const float* weights,
    float3* smoothedPoints,
    int numPoints,
    float alpha,
    float maxDisplacement
) {
    int pointId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pointId >= numPoints) return;
    
    float3 point = points[pointId];
    float3 laplacian = laplacianVectors[pointId];
    
    // Apply feature-based weighting
    float weight = weights ? weights[pointId] : 1.0f;
    float effectiveAlpha = alpha * weight;
    
    // Compute displacement
    float3 displacement = laplacian * effectiveAlpha;
    
    // Clamp displacement magnitude
    float displacementLength = length(displacement);
    if (displacementLength > maxDisplacement) {
        displacement = displacement * (maxDisplacement / displacementLength);
    }
    
    // Apply smoothing
    smoothedPoints[pointId] = point + displacement;
}

__global__ void ComputeLocalFeatureKernel(
    const float3* points,
    const int* adjacency,
    const int* offsets,
    float* featureMetric,
    int numPoints
) {
    int pointId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pointId >= numPoints) return;
    
    int start = offsets[pointId];
    int end = offsets[pointId + 1];
    int neighborCount = end - start;
    
    if (neighborCount < 3) {
        featureMetric[pointId] = 0.0f;
        return;
    }
    
    float3 point = points[pointId];
    
    // Compute variance of neighbor distances
    float sumDist = 0.0f;
    float sumDistSq = 0.0f;
    
    for (int i = start; i < end; i++) {
        int neighborId = adjacency[i];
        float3 neighbor = points[neighborId];
        float dist = length(neighbor - point);
        
        sumDist += dist;
        sumDistSq += dist * dist;
    }
    
    float meanDist = sumDist / neighborCount;
    float variance = (sumDistSq / neighborCount) - (meanDist * meanDist);
    
    featureMetric[pointId] = sqrtf(variance);
}
