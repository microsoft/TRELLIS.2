#pragma once

#include "fdg_gpu_common.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace fdg_gpu {

struct FlexibleDualGridGPUOutput {
    int64_t size = 0;
    int32_t* voxel_coords = nullptr;  // [size, 3]
    float* dual_vertices = nullptr;   // [size, 3]
    bool* intersected = nullptr;      // [size, 3]
};

cudaError_t mesh_to_flexible_dual_grid_gpu(
    const float* vertices,
    int64_t num_vertices,
    const int32_t* faces,
    int64_t num_faces,
    float3 voxel_size,
    int3_ grid_min,
    int3_ grid_max,
    float face_weight,
    float boundary_weight,
    float regularization_weight,
    int64_t intersect_chunk_triangles,
    int boundary_chunk_steps,
    cudaStream_t stream,
    FlexibleDualGridGPUOutput* out);

void free_flexible_dual_grid_gpu_output(FlexibleDualGridGPUOutput* out) noexcept;

} // namespace fdg_gpu
