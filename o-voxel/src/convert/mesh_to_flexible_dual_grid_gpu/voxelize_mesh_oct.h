#pragma once

#include "fdg_gpu_common.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace oct_pairs {

fdg_gpu::PrimitivePairResult voxelize_mesh_oct_gpu(
    const float* vertices,
    int64_t num_vertices,
    const int32_t* faces,
    int64_t num_faces,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_size,
    float3 voxel_size,
    cudaStream_t stream = nullptr);

fdg_gpu::PrimitivePairResult voxelize_edge_oct_gpu(
    const float* vertices,
    int64_t num_vertices,
    const int32_t* edges,
    int64_t num_edges,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_size,
    float3 voxel_size,
    cudaStream_t stream = nullptr);

struct FaceQEFResult {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<fdg_gpu::SymQEF10> qefs;
};

FaceQEFResult face_qef_gpu(
    float3 voxel_size,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    const float* triangles,
    int64_t num_triangles,
    const int* voxels,
    int64_t num_voxels,
    cudaStream_t stream = nullptr);

} // namespace oct_pairs
