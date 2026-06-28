#include <torch/extension.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "../api.h"
#include "flexible_dual_grid_gpu.h"
#include "intersection_qef.h"
#include "voxelize_mesh_oct.h"
#include "voxel_traverse_edge_dda.h"

struct bool3 { bool x, y, z; bool& operator[](int i) { return (&x)[i]; } };

struct VoxelCoord {
    int x, y, z;

    int& operator[](int i) { return (&x)[i]; }

    bool operator==(const VoxelCoord& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
template <>
struct hash<VoxelCoord> {
    size_t operator()(const VoxelCoord& v) const {
        const std::size_t p1 = 73856093;
        const std::size_t p2 = 19349663;
        const std::size_t p3 = 83492791;
        return static_cast<std::size_t>(v.x) * p1 ^
               static_cast<std::size_t>(v.y) * p2 ^
               static_cast<std::size_t>(v.z) * p3;
    }
};
} // namespace std

void intersect_qef(
    const Eigen::Vector3f& voxel_size,
    const Eigen::Vector3i& grid_min,
    const Eigen::Vector3i& grid_max,
    const std::vector<Eigen::Vector3f>& triangles,
    std::unordered_map<VoxelCoord, size_t>& hash_table,
    std::vector<int3>& voxels,
    std::vector<Eigen::Vector3f>& means,
    std::vector<float>& cnt,
    std::vector<bool3>& intersected,
    std::vector<Eigen::Matrix4f>& qefs
);

void face_qef(
    const Eigen::Vector3f& voxel_size,
    const Eigen::Vector3i& grid_min,
    const Eigen::Vector3i& grid_max,
    const std::vector<Eigen::Vector3f>& triangles,
    std::unordered_map<VoxelCoord, size_t>& hash_table,
    std::vector<Eigen::Matrix4f>& qefs
);

void boundry_qef(
    const Eigen::Vector3f& voxel_size,
    const Eigen::Vector3i& grid_min,
    const Eigen::Vector3i& grid_max,
    const std::vector<Eigen::Vector3f>& boundries,
    float boundary_weight,
    std::unordered_map<VoxelCoord, size_t>& hash_table,
    std::vector<Eigen::Matrix4f>& qefs
);

namespace {

inline void check_cuda_success(cudaError_t err, const char* context) {
    TORCH_CHECK(err == cudaSuccess, context, ": ", cudaGetErrorString(err));
}

inline float3 tensor_to_float3_cpu(const torch::Tensor& t) {
    auto tc = t.to(torch::kFloat32).contiguous().cpu();
    TORCH_CHECK(tc.dim() == 1 && tc.size(0) == 3, "voxel_size must have shape [3]");
    const float* p = tc.data_ptr<float>();
    return float3{p[0], p[1], p[2]};
}

inline void tensor_to_grid_min_max_cpu(
    const torch::Tensor& t,
    fdg_gpu::int3_& grid_min,
    fdg_gpu::int3_& grid_max
) {
    auto tc = t.to(torch::kInt32).contiguous().cpu();
    TORCH_CHECK(tc.dim() == 2 && tc.size(0) == 2 && tc.size(1) == 3, "grid_range must have shape [2, 3]");
    const int32_t* p = tc.data_ptr<int32_t>();
    grid_min = fdg_gpu::int3_{p[0], p[1], p[2]};
    grid_max = fdg_gpu::int3_{p[3], p[4], p[5]};
}

inline fdg_gpu::int3_ grid_size_from_min_max(
    const fdg_gpu::int3_& grid_min,
    const fdg_gpu::int3_& grid_max
) {
    return fdg_gpu::int3_{
        grid_max.x - grid_min.x,
        grid_max.y - grid_min.y,
        grid_max.z - grid_min.z,
    };
}

__global__ void unpack_intersected_mask_kernel(
    const uint8_t* mask,
    int64_t n,
    bool* out_bool3
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t m = mask[i];
    out_bool3[3 * i + 0] = (m & (1u << 0)) != 0;
    out_bool3[3 * i + 1] = (m & (1u << 1)) != 0;
    out_bool3[3 * i + 2] = (m & (1u << 2)) != 0;
}

inline void check_triangles_tensor(const torch::Tensor& triangles_c) {
    TORCH_CHECK(
        triangles_c.dim() == 3 && triangles_c.size(1) == 3 && triangles_c.size(2) == 3,
        "triangles must have shape [T, 3, 3]"
    );
}

inline void check_voxels_tensor(const torch::Tensor& voxels_c) {
    TORCH_CHECK(
        voxels_c.dim() == 2 && voxels_c.size(1) == 3,
        "voxels must have shape [N, 3]"
    );
}

inline void check_edges_tensor(const torch::Tensor& edges_c) {
    TORCH_CHECK(edges_c.dim() == 2 && edges_c.size(1) == 2, "edges must have shape [E, 2]");
}

inline void check_boundaries_tensor(const torch::Tensor& boundaries_c) {
    TORCH_CHECK(
        boundaries_c.dim() == 3 && boundaries_c.size(1) == 2 && boundaries_c.size(2) == 3,
        "boundaries must have shape [B, 2, 3]"
    );
}

inline void check_cpu_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(!t.is_cuda(), name, " must be a CPU tensor");
}

inline Eigen::Vector3f tensor_to_eigen_vec3_cpu(const torch::Tensor& t) {
    auto tc = t.to(torch::kFloat32).contiguous().cpu();
    TORCH_CHECK(tc.dim() == 1 && tc.size(0) == 3, "voxel_size must have shape [3]");
    const float* p = tc.data_ptr<float>();
    return Eigen::Vector3f(p[0], p[1], p[2]);
}

inline void tensor_to_eigen_grid_min_max_cpu(
    const torch::Tensor& t,
    Eigen::Vector3i& grid_min,
    Eigen::Vector3i& grid_max
) {
    auto tc = t.to(torch::kInt32).contiguous().cpu();
    TORCH_CHECK(tc.dim() == 2 && tc.size(0) == 2 && tc.size(1) == 3, "grid_range must have shape [2, 3]");
    const int32_t* p = tc.data_ptr<int32_t>();
    grid_min = Eigen::Vector3i(p[0], p[1], p[2]);
    grid_max = Eigen::Vector3i(p[3], p[4], p[5]);
}

inline std::vector<Eigen::Vector3f> triangles_tensor_to_vector_cpu(const torch::Tensor& triangles) {
    auto triangles_c = triangles.to(torch::kFloat32).contiguous().cpu();
    check_triangles_tensor(triangles_c);
    const float* p = triangles_c.data_ptr<float>();
    const int64_t n = triangles_c.size(0);
    std::vector<Eigen::Vector3f> out;
    out.reserve(static_cast<size_t>(n) * 3);
    for (int64_t i = 0; i < n; ++i) {
        for (int v = 0; v < 3; ++v) {
            const int64_t base = (i * 3 + v) * 3;
            out.emplace_back(p[base + 0], p[base + 1], p[base + 2]);
        }
    }
    return out;
}

inline std::vector<Eigen::Vector3f> boundaries_tensor_to_vector_cpu(const torch::Tensor& boundaries) {
    auto boundaries_c = boundaries.to(torch::kFloat32).contiguous().cpu();
    check_boundaries_tensor(boundaries_c);
    const float* p = boundaries_c.data_ptr<float>();
    const int64_t n = boundaries_c.size(0);
    std::vector<Eigen::Vector3f> out;
    out.reserve(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        for (int v = 0; v < 2; ++v) {
            const int64_t base = (i * 2 + v) * 3;
            out.emplace_back(p[base + 0], p[base + 1], p[base + 2]);
        }
    }
    return out;
}

inline std::vector<int3> voxels_tensor_to_vector_cpu(
    const torch::Tensor& voxels,
    std::unordered_map<VoxelCoord, size_t>& hash_table
) {
    auto voxels_c = voxels.to(torch::kInt32).contiguous().cpu();
    check_voxels_tensor(voxels_c);
    const int32_t* p = voxels_c.data_ptr<int32_t>();
    const int64_t n = voxels_c.size(0);
    std::vector<int3> out;
    out.reserve(static_cast<size_t>(n));
    hash_table.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const VoxelCoord coord{p[3 * i + 0], p[3 * i + 1], p[3 * i + 2]};
        hash_table[coord] = static_cast<size_t>(i);
        out.push_back(int3{coord.x, coord.y, coord.z});
    }
    return out;
}

inline torch::Tensor int3_vector_to_tensor_cpu(const std::vector<int3>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size()), 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    int32_t* p = out.data_ptr<int32_t>();
    for (size_t i = 0; i < values.size(); ++i) {
        p[3 * i + 0] = values[i].x;
        p[3 * i + 1] = values[i].y;
        p[3 * i + 2] = values[i].z;
    }
    return out;
}

inline torch::Tensor vec3f_vector_to_tensor_cpu(const std::vector<Eigen::Vector3f>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size()), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* p = out.data_ptr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        p[3 * i + 0] = values[i].x();
        p[3 * i + 1] = values[i].y();
        p[3 * i + 2] = values[i].z();
    }
    return out;
}

inline torch::Tensor float_vector_to_tensor_cpu(const std::vector<float>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size())}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* p = out.data_ptr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        p[i] = values[i];
    }
    return out;
}

inline torch::Tensor bool3_vector_to_tensor_cpu(const std::vector<bool3>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size()), 3}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    bool* p = out.data_ptr<bool>();
    for (size_t i = 0; i < values.size(); ++i) {
        p[3 * i + 0] = values[i].x;
        p[3 * i + 1] = values[i].y;
        p[3 * i + 2] = values[i].z;
    }
    return out;
}

inline torch::Tensor matrix4f_vector_to_tensor_cpu(const std::vector<Eigen::Matrix4f>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size()), 4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* p = out.data_ptr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                p[i * 16 + r * 4 + c] = values[i](r, c);
            }
        }
    }
    return out;
}

inline std::tuple<torch::Tensor, torch::Tensor> primitive_pair_to_tensors(
    const fdg_gpu::PrimitivePairResult& pairs,
    const torch::Device& device,
    cudaStream_t stream
) {
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    torch::Tensor prim_id = torch::empty({pairs.size}, opts_i32);
    torch::Tensor voxels_axis_major = torch::empty({3, pairs.size}, opts_i32);

    if (pairs.size > 0) {
        check_cuda_success(
            cudaMemcpyAsync(
                prim_id.data_ptr<int32_t>(),
                pairs.prim_id.data(),
                static_cast<size_t>(pairs.size) * sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync primitive prim_id"
        );

        check_cuda_success(
            cudaMemcpyAsync(
                voxels_axis_major.data_ptr<int32_t>() + pairs.size * 0,
                pairs.voxel_i.data(),
                static_cast<size_t>(pairs.size) * sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync primitive voxel_i"
        );
        check_cuda_success(
            cudaMemcpyAsync(
                voxels_axis_major.data_ptr<int32_t>() + pairs.size * 1,
                pairs.voxel_j.data(),
                static_cast<size_t>(pairs.size) * sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync primitive voxel_j"
        );
        check_cuda_success(
            cudaMemcpyAsync(
                voxels_axis_major.data_ptr<int32_t>() + pairs.size * 2,
                pairs.voxel_k.data(),
                static_cast<size_t>(pairs.size) * sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync primitive voxel_k"
        );

        check_cuda_success(cudaStreamSynchronize(stream), "cudaStreamSynchronize primitive_pair_to_tensors");
    }

    torch::Tensor voxels = voxels_axis_major.transpose(0, 1).contiguous();
    return std::make_tuple(prim_id, voxels);
}

} // namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mesh_to_flexible_dual_grid_gpu(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    float face_weight,
    float boundary_weight,
    float regularization_weight,
    int64_t intersect_chunk_triangles,
    int boundary_chunk_steps
) {
    TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
    TORCH_CHECK(vertices.device() == faces.device(), "vertices and faces must be on the same CUDA device");

    auto vertices_c = vertices.to(torch::kFloat32).contiguous();
    auto faces_c = faces.to(torch::kInt32).contiguous();

    TORCH_CHECK(vertices_c.dim() == 2 && vertices_c.size(1) == 3, "vertices must have shape [V, 3]");
    TORCH_CHECK(faces_c.dim() == 2 && faces_c.size(1) == 3, "faces must have shape [F, 3]");

    float3 voxel_size_h = tensor_to_float3_cpu(voxel_size);
    fdg_gpu::int3_ grid_min{};
    fdg_gpu::int3_ grid_max{};
    tensor_to_grid_min_max_cpu(grid_range, grid_min, grid_max);

    fdg_gpu::FlexibleDualGridGPUOutput out{};
    cudaStream_t stream = nullptr;

    cudaError_t status = fdg_gpu::mesh_to_flexible_dual_grid_gpu(
        vertices_c.data_ptr<float>(),
        vertices_c.size(0),
        faces_c.data_ptr<int32_t>(),
        faces_c.size(0),
        voxel_size_h,
        grid_min,
        grid_max,
        face_weight,
        boundary_weight,
        regularization_weight,
        intersect_chunk_triangles,
        boundary_chunk_steps,
        stream,
        &out
    );

    if (status != cudaSuccess) {
        fdg_gpu::free_flexible_dual_grid_gpu_output(&out);
        TORCH_CHECK(false, "mesh_to_flexible_dual_grid_gpu failed: ", cudaGetErrorString(status));
    }

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(vertices_c.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(vertices_c.device());
    auto opts_b = torch::TensorOptions().dtype(torch::kBool).device(vertices_c.device());

    torch::Tensor voxel_coords = torch::empty({out.size, 3}, opts_i32);
    torch::Tensor dual_vertices = torch::empty({out.size, 3}, opts_f32);
    torch::Tensor intersected = torch::empty({out.size, 3}, opts_b);

    if (out.size > 0) {
        check_cuda_success(
            cudaMemcpyAsync(
                voxel_coords.data_ptr<int32_t>(),
                out.voxel_coords,
                static_cast<size_t>(out.size) * 3 * sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync voxel_coords"
        );

        check_cuda_success(
            cudaMemcpyAsync(
                dual_vertices.data_ptr<float>(),
                out.dual_vertices,
                static_cast<size_t>(out.size) * 3 * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync dual_vertices"
        );

        check_cuda_success(
            cudaMemcpyAsync(
                intersected.data_ptr<bool>(),
                out.intersected,
                static_cast<size_t>(out.size) * 3 * sizeof(bool),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync intersected"
        );

        check_cuda_success(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    fdg_gpu::free_flexible_dual_grid_gpu_output(&out);
    return std::make_tuple(voxel_coords, dual_vertices, intersected);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> intersect_qef_cpu(
    const torch::Tensor& triangles,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range
) {
    check_cpu_tensor(triangles, "triangles");
    auto triangles_c = triangles.to(torch::kFloat32).contiguous();
    check_triangles_tensor(triangles_c);

    Eigen::Vector3f voxel_size_h = tensor_to_eigen_vec3_cpu(voxel_size);
    Eigen::Vector3i grid_min, grid_max;
    tensor_to_eigen_grid_min_max_cpu(grid_range, grid_min, grid_max);

    std::vector<Eigen::Vector3f> triangles_vec = triangles_tensor_to_vector_cpu(triangles_c);
    std::unordered_map<VoxelCoord, size_t> hash_table;
    std::vector<int3> voxels_vec;
    std::vector<Eigen::Vector3f> mean_sum;
    std::vector<float> cnt;
    std::vector<bool3> intersected_vec;
    std::vector<Eigen::Matrix4f> qefs;

    intersect_qef(
        voxel_size_h,
        grid_min,
        grid_max,
        triangles_vec,
        hash_table,
        voxels_vec,
        mean_sum,
        cnt,
        intersected_vec,
        qefs
    );

    return std::make_tuple(
        int3_vector_to_tensor_cpu(voxels_vec),
        vec3f_vector_to_tensor_cpu(mean_sum),
        float_vector_to_tensor_cpu(cnt),
        bool3_vector_to_tensor_cpu(intersected_vec),
        matrix4f_vector_to_tensor_cpu(qefs)
    );
}


torch::Tensor face_qef_cpu(
    const torch::Tensor& triangles,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    const torch::Tensor& voxels
) {
    check_cpu_tensor(triangles, "triangles");
    check_cpu_tensor(voxels, "voxels");
    auto triangles_c = triangles.to(torch::kFloat32).contiguous();
    auto voxels_c = voxels.to(torch::kInt32).contiguous();
    check_triangles_tensor(triangles_c);
    check_voxels_tensor(voxels_c);

    Eigen::Vector3f voxel_size_h = tensor_to_eigen_vec3_cpu(voxel_size);
    Eigen::Vector3i grid_min, grid_max;
    tensor_to_eigen_grid_min_max_cpu(grid_range, grid_min, grid_max);

    std::vector<Eigen::Vector3f> triangles_vec = triangles_tensor_to_vector_cpu(triangles_c);
    std::unordered_map<VoxelCoord, size_t> hash_table;
    std::vector<int3> voxels_vec = voxels_tensor_to_vector_cpu(voxels_c, hash_table);
    std::vector<Eigen::Matrix4f> qefs(voxels_vec.size(), Eigen::Matrix4f::Zero());

    face_qef(
        voxel_size_h,
        grid_min,
        grid_max,
        triangles_vec,
        hash_table,
        qefs
    );

    return matrix4f_vector_to_tensor_cpu(qefs);
}


torch::Tensor boundary_qef_cpu(
    const torch::Tensor& boundaries,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    float boundary_weight,
    const torch::Tensor& voxels
) {
    check_cpu_tensor(boundaries, "boundaries");
    check_cpu_tensor(voxels, "voxels");
    auto boundaries_c = boundaries.to(torch::kFloat32).contiguous();
    auto voxels_c = voxels.to(torch::kInt32).contiguous();
    check_boundaries_tensor(boundaries_c);
    check_voxels_tensor(voxels_c);

    Eigen::Vector3f voxel_size_h = tensor_to_eigen_vec3_cpu(voxel_size);
    Eigen::Vector3i grid_min, grid_max;
    tensor_to_eigen_grid_min_max_cpu(grid_range, grid_min, grid_max);

    std::vector<Eigen::Vector3f> boundaries_vec = boundaries_tensor_to_vector_cpu(boundaries_c);
    std::unordered_map<VoxelCoord, size_t> hash_table;
    std::vector<int3> voxels_vec = voxels_tensor_to_vector_cpu(voxels_c, hash_table);
    std::vector<Eigen::Matrix4f> qefs(voxels_vec.size(), Eigen::Matrix4f::Zero());

    boundry_qef(
        voxel_size_h,
        grid_min,
        grid_max,
        boundaries_vec,
        boundary_weight,
        hash_table,
        qefs
    );

    return matrix4f_vector_to_tensor_cpu(qefs);
}


torch::Tensor intersection_occ_gpu(
    const torch::Tensor& triangles,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    int64_t chunk_triangles
) {
    TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");
    TORCH_CHECK(chunk_triangles > 0, "chunk_triangles must be > 0");

    auto triangles_c = triangles.to(torch::kFloat32).contiguous();
    check_triangles_tensor(triangles_c);

    float3 voxel_size_h = tensor_to_float3_cpu(voxel_size);
    fdg_gpu::int3_ grid_min{};
    fdg_gpu::int3_ grid_max{};
    tensor_to_grid_min_max_cpu(grid_range, grid_min, grid_max);

    cudaStream_t stream = nullptr;
    auto out = intersection_qef::intersection_occ_gpu(
        triangles_c.data_ptr<float>(),
        triangles_c.size(0),
        voxel_size_h,
        grid_min,
        grid_max,
        chunk_triangles,
        stream
    );

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles_c.device());
    torch::Tensor voxels = torch::empty({out.size, 3}, opts_i32);
    if (out.size > 0) {
        check_cuda_success(
            cudaMemcpyAsync(
                voxels.data_ptr<int32_t>(),
                out.voxels.data(),
                static_cast<size_t>(out.size) * 3 * sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync intersection_occ voxels"
        );
        check_cuda_success(cudaStreamSynchronize(stream), "cudaStreamSynchronize intersection_occ");
    }
    return voxels;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> intersect_qef_gpu(
    const torch::Tensor& triangles,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    int64_t chunk_triangles
) {
    TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");
    TORCH_CHECK(chunk_triangles > 0, "chunk_triangles must be > 0");

    auto triangles_c = triangles.to(torch::kFloat32).contiguous();
    check_triangles_tensor(triangles_c);

    float3 voxel_size_h = tensor_to_float3_cpu(voxel_size);
    fdg_gpu::int3_ grid_min{};
    fdg_gpu::int3_ grid_max{};
    tensor_to_grid_min_max_cpu(grid_range, grid_min, grid_max);

    cudaStream_t stream = nullptr;
    auto out = intersection_qef::intersect_qef_gpu(
        triangles_c.data_ptr<float>(),
        triangles_c.size(0),
        voxel_size_h,
        grid_min,
        grid_max,
        chunk_triangles,
        stream
    );

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles_c.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(triangles_c.device());
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(triangles_c.device());
    auto opts_b = torch::TensorOptions().dtype(torch::kBool).device(triangles_c.device());

    static_assert(sizeof(fdg_gpu::SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");

    torch::Tensor voxels = torch::empty({out.size, 3}, opts_i32);
    torch::Tensor mean_sum = torch::empty({out.size, 3}, opts_f32);
    torch::Tensor cnt = torch::empty({out.size}, opts_f32);
    torch::Tensor intersected_mask = torch::empty({out.size}, opts_u8);
    torch::Tensor qefs = torch::empty({out.size, 10}, opts_f32);

    if (out.size > 0) {
        check_cuda_success(
            cudaMemcpyAsync(
                voxels.data_ptr<int32_t>(),
                out.voxels.data(),
                static_cast<size_t>(out.size) * 3 * sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync intersect_qef voxels"
        );
        check_cuda_success(
            cudaMemcpyAsync(
                mean_sum.data_ptr<float>(),
                out.mean_sum.data(),
                static_cast<size_t>(out.size) * 3 * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync intersect_qef mean_sum"
        );
        check_cuda_success(
            cudaMemcpyAsync(
                cnt.data_ptr<float>(),
                out.cnt.data(),
                static_cast<size_t>(out.size) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync intersect_qef cnt"
        );
        check_cuda_success(
            cudaMemcpyAsync(
                intersected_mask.data_ptr<uint8_t>(),
                out.intersected.data(),
                static_cast<size_t>(out.size) * sizeof(uint8_t),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync intersect_qef intersected"
        );
        check_cuda_success(
            cudaMemcpyAsync(
                qefs.data_ptr<float>(),
                out.qefs.data(),
                static_cast<size_t>(out.size) * sizeof(fdg_gpu::SymQEF10),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync intersect_qef qefs"
        );
    }

    torch::Tensor intersected = torch::empty({out.size, 3}, opts_b);
    if (out.size > 0) {
        const int kBlock = 256;
        const int grid = static_cast<int>((out.size + kBlock - 1) / kBlock);
        unpack_intersected_mask_kernel<<<grid, kBlock, 0, stream>>>(
            intersected_mask.data_ptr<uint8_t>(),
            out.size,
            intersected.data_ptr<bool>()
        );
        check_cuda_success(cudaGetLastError(), "unpack_intersected_mask_kernel");
        check_cuda_success(cudaStreamSynchronize(stream), "cudaStreamSynchronize intersect_qef");
    }

    return std::make_tuple(voxels, mean_sum, cnt, intersected, qefs);
}


std::tuple<torch::Tensor, torch::Tensor> voxelize_mesh_oct_gpu(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range
) {
    TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
    TORCH_CHECK(vertices.device() == faces.device(), "vertices and faces must be on the same CUDA device");

    auto vertices_c = vertices.to(torch::kFloat32).contiguous();
    auto faces_c = faces.to(torch::kInt32).contiguous();
    TORCH_CHECK(vertices_c.dim() == 2 && vertices_c.size(1) == 3, "vertices must have shape [V, 3]");
    TORCH_CHECK(faces_c.dim() == 2 && faces_c.size(1) == 3, "faces must have shape [F, 3]");

    float3 voxel_size_h = tensor_to_float3_cpu(voxel_size);
    fdg_gpu::int3_ grid_min{};
    fdg_gpu::int3_ grid_max{};
    tensor_to_grid_min_max_cpu(grid_range, grid_min, grid_max);
    fdg_gpu::int3_ grid_size = grid_size_from_min_max(grid_min, grid_max);

    cudaStream_t stream = nullptr;
    auto out = oct_pairs::voxelize_mesh_oct_gpu(
        vertices_c.data_ptr<float>(),
        vertices_c.size(0),
        faces_c.data_ptr<int32_t>(),
        faces_c.size(0),
        grid_min,
        grid_size,
        voxel_size_h,
        stream
    );

    return primitive_pair_to_tensors(out, vertices_c.device(), stream);
}


std::tuple<torch::Tensor, torch::Tensor> voxelize_edge_oct_gpu(
    const torch::Tensor& vertices,
    const torch::Tensor& edges,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range
) {
    TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");
    TORCH_CHECK(vertices.device() == edges.device(), "vertices and edges must be on the same CUDA device");

    auto vertices_c = vertices.to(torch::kFloat32).contiguous();
    auto edges_c = edges.to(torch::kInt32).contiguous();
    TORCH_CHECK(vertices_c.dim() == 2 && vertices_c.size(1) == 3, "vertices must have shape [V, 3]");
    check_edges_tensor(edges_c);

    float3 voxel_size_h = tensor_to_float3_cpu(voxel_size);
    fdg_gpu::int3_ grid_min{};
    fdg_gpu::int3_ grid_max{};
    tensor_to_grid_min_max_cpu(grid_range, grid_min, grid_max);
    fdg_gpu::int3_ grid_size = grid_size_from_min_max(grid_min, grid_max);

    cudaStream_t stream = nullptr;
    auto out = oct_pairs::voxelize_edge_oct_gpu(
        vertices_c.data_ptr<float>(),
        vertices_c.size(0),
        edges_c.data_ptr<int32_t>(),
        edges_c.size(0),
        grid_min,
        grid_size,
        voxel_size_h,
        stream
    );

    return primitive_pair_to_tensors(out, vertices_c.device(), stream);
}


torch::Tensor face_qef_gpu(
    const torch::Tensor& triangles,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    const torch::Tensor& voxels
) {
    TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");
    TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
    TORCH_CHECK(triangles.device() == voxels.device(), "triangles and voxels must be on the same CUDA device");

    auto triangles_c = triangles.to(torch::kFloat32).contiguous();
    auto voxels_c = voxels.to(torch::kInt32).contiguous();
    check_triangles_tensor(triangles_c);
    check_voxels_tensor(voxels_c);

    float3 voxel_size_h = tensor_to_float3_cpu(voxel_size);
    fdg_gpu::int3_ grid_min{};
    fdg_gpu::int3_ grid_max{};
    tensor_to_grid_min_max_cpu(grid_range, grid_min, grid_max);

    cudaStream_t stream = nullptr;
    auto out = oct_pairs::face_qef_gpu(
        voxel_size_h,
        grid_min,
        grid_max,
        triangles_c.data_ptr<float>(),
        triangles_c.size(0),
        voxels_c.data_ptr<int>(),
        voxels_c.size(0),
        stream
    );

    static_assert(sizeof(fdg_gpu::SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(triangles_c.device());
    torch::Tensor qefs = torch::empty({out.size, 10}, opts_f32);
    if (out.size > 0) {
        check_cuda_success(
            cudaMemcpyAsync(
                qefs.data_ptr<float>(),
                out.qefs.data(),
                static_cast<size_t>(out.size) * sizeof(fdg_gpu::SymQEF10),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync face_qef qefs"
        );
        check_cuda_success(cudaStreamSynchronize(stream), "cudaStreamSynchronize face_qef");
    }
    return qefs;
}


std::tuple<torch::Tensor, torch::Tensor> voxel_traverse_edge_dda_gpu(
    const torch::Tensor& vertices,
    const torch::Tensor& edges,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    int chunk_steps
) {
    TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");
    TORCH_CHECK(vertices.device() == edges.device(), "vertices and edges must be on the same CUDA device");
    TORCH_CHECK(chunk_steps > 0, "chunk_steps must be > 0");

    auto vertices_c = vertices.to(torch::kFloat32).contiguous();
    auto edges_c = edges.to(torch::kInt32).contiguous();
    TORCH_CHECK(vertices_c.dim() == 2 && vertices_c.size(1) == 3, "vertices must have shape [V, 3]");
    check_edges_tensor(edges_c);

    float3 voxel_size_h = tensor_to_float3_cpu(voxel_size);
    fdg_gpu::int3_ grid_min{};
    fdg_gpu::int3_ grid_max{};
    tensor_to_grid_min_max_cpu(grid_range, grid_min, grid_max);

    cudaStream_t stream = nullptr;
    auto out = edge_dda::voxel_traverse_edge_dda_gpu(
        vertices_c.data_ptr<float>(),
        vertices_c.size(0),
        edges_c.data_ptr<int32_t>(),
        edges_c.size(0),
        voxel_size_h,
        grid_min,
        grid_max,
        chunk_steps,
        stream
    );

    return primitive_pair_to_tensors(out, vertices_c.device(), stream);
}


torch::Tensor boundary_qef_gpu(
    const torch::Tensor& boundaries,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    float boundary_weight,
    const torch::Tensor& voxels,
    int chunk_steps
) {
    TORCH_CHECK(boundaries.is_cuda(), "boundaries must be a CUDA tensor");
    TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
    TORCH_CHECK(boundaries.device() == voxels.device(), "boundaries and voxels must be on the same CUDA device");
    TORCH_CHECK(chunk_steps > 0, "chunk_steps must be > 0");

    auto boundaries_c = boundaries.to(torch::kFloat32).contiguous();
    auto voxels_c = voxels.to(torch::kInt32).contiguous();
    check_boundaries_tensor(boundaries_c);
    check_voxels_tensor(voxels_c);

    float3 voxel_size_h = tensor_to_float3_cpu(voxel_size);
    fdg_gpu::int3_ grid_min{};
    fdg_gpu::int3_ grid_max{};
    tensor_to_grid_min_max_cpu(grid_range, grid_min, grid_max);

    cudaStream_t stream = nullptr;
    auto out = edge_dda::boundary_qef_gpu(
        voxel_size_h,
        grid_min,
        grid_max,
        boundaries_c.data_ptr<float>(),
        boundaries_c.size(0),
        boundary_weight,
        voxels_c.data_ptr<int>(),
        voxels_c.size(0),
        chunk_steps,
        stream
    );

    static_assert(sizeof(fdg_gpu::SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(boundaries_c.device());
    torch::Tensor qefs = torch::empty({out.size, 10}, opts_f32);
    if (out.size > 0) {
        check_cuda_success(
            cudaMemcpyAsync(
                qefs.data_ptr<float>(),
                out.qefs.data(),
                static_cast<size_t>(out.size) * sizeof(fdg_gpu::SymQEF10),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "cudaMemcpyAsync boundary_qef qefs"
        );
        check_cuda_success(cudaStreamSynchronize(stream), "cudaStreamSynchronize boundary_qef");
    }

    return qefs;
}
