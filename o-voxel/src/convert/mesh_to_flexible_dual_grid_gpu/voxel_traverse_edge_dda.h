#pragma once

#include "fdg_gpu_common.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace edge_dda {

fdg_gpu::PrimitivePairResult voxel_traverse_edge_dda_gpu(
    const float* vertices,
    int64_t num_vertices,
    const int32_t* edges,
    int64_t num_edges,
    float3 voxel_size,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    int chunk_steps,
    cudaStream_t stream = nullptr);

struct BoundaryQEFResult {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<fdg_gpu::SymQEF10> qefs;
};

BoundaryQEFResult boundary_qef_gpu(
    float3 voxel_size,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    const float* boundaries,
    int64_t num_boundaries,
    float boundary_weight,
    const int* voxels,
    int64_t num_voxels,
    int chunk_steps,
    cudaStream_t stream = nullptr);

} // namespace edge_dda
