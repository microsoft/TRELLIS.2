#pragma once

#include "fdg_gpu_common.h"

namespace intersection_qef {

struct IntersectionOccResult {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<int> voxels;   // [size,3] flattened
};

struct IntersectQEFResult {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<int> voxels;            // [size,3] flattened
    fdg_gpu::DeviceBuffer<float> mean_sum;        // [size,3] flattened
    fdg_gpu::DeviceBuffer<float> cnt;             // [size]
    fdg_gpu::DeviceBuffer<uint8_t> intersected;   // [size], bitmask for bool3
    fdg_gpu::DeviceBuffer<fdg_gpu::SymQEF10> qefs;// [size]
};

IntersectionOccResult intersection_occ_gpu(
    const float* triangles,           // [num_triangles, 3, 3] flattened
    int64_t num_triangles,
    float3 voxel_size,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    int64_t chunk_triangles = 4096,
    cudaStream_t stream = nullptr);

IntersectQEFResult intersect_qef_gpu(
    const float* triangles,           // [num_triangles, 3, 3] flattened
    int64_t num_triangles,
    float3 voxel_size,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    int64_t chunk_triangles = 4096,
    cudaStream_t stream = nullptr);

}  // namespace intersection_qef
