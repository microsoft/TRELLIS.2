#include "flexible_dual_grid_gpu.h"

#include "intersection_qef.h"
#include "voxel_traverse_edge_dda.h"
#include "voxelize_mesh_oct.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include "fdg_gpu_small_cpqr_device.cuh"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace fdg_gpu {
namespace {

constexpr int kBlockSize = 128;

struct FlatTriangles {
    int64_t num_triangles = 0;
    DeviceBuffer<float> triangles;  // [3 * num_triangles, 3]
};

struct BoundaryEdgeIndexResult {
    int64_t size = 0;
    DeviceBuffer<int32_t> edge_vertex_ids;  // [size, 2]
};

struct BoundarySegments {
    int64_t size = 0;
    DeviceBuffer<float> segments;  // [2 * size, 3]
};

struct IsOne {
    __host__ __device__ bool operator()(int v) const { return v == 1; }
};

__host__ __device__ __forceinline__ uint64_t pack_edge_key(int a, int b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
           static_cast<uint32_t>(b);
}

__host__ __device__ __forceinline__ int edge_key_v0(uint64_t key) {
    return static_cast<int>(key >> 32);
}

__host__ __device__ __forceinline__ int edge_key_v1(uint64_t key) {
    return static_cast<int>(key & 0xffffffffu);
}

__host__ __device__ __forceinline__ SymQEF10 sym10_zero() {
    return SymQEF10{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

__host__ __device__ __forceinline__ SymQEF10 sym10_add(const SymQEF10& a, const SymQEF10& b) {
    return SymQEF10{
        a.q00 + b.q00,
        a.q01 + b.q01,
        a.q02 + b.q02,
        a.q03 + b.q03,
        a.q11 + b.q11,
        a.q12 + b.q12,
        a.q13 + b.q13,
        a.q22 + b.q22,
        a.q23 + b.q23,
        a.q33 + b.q33,
    };
}

__host__ __device__ __forceinline__ SymQEF10 sym10_scale(const SymQEF10& a, float s) {
    return SymQEF10{
        a.q00 * s,
        a.q01 * s,
        a.q02 * s,
        a.q03 * s,
        a.q11 * s,
        a.q12 * s,
        a.q13 * s,
        a.q22 * s,
        a.q23 * s,
        a.q33 * s,
    };
}

__global__ void gather_flat_triangles_kernel(
    const float* vertices,
    const int32_t* faces,
    int64_t num_faces,
    float* triangles) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= 3 * num_faces) return;

    const int64_t f = tid / 3;
    const int lv = static_cast<int>(tid % 3);
    const int32_t vid = faces[3 * f + lv];

    triangles[3 * tid + 0] = vertices[3 * vid + 0];
    triangles[3 * tid + 1] = vertices[3 * vid + 1];
    triangles[3 * tid + 2] = vertices[3 * vid + 2];
}

__global__ void emit_face_edges_kernel(
    const int32_t* faces,
    int64_t num_faces,
    uint64_t* edge_keys) {
    const int64_t f = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int32_t e00 = faces[3 * f + 0];
    int32_t e01 = faces[3 * f + 1];
    int32_t e10 = faces[3 * f + 1];
    int32_t e11 = faces[3 * f + 2];
    int32_t e20 = faces[3 * f + 2];
    int32_t e21 = faces[3 * f + 0];

    if (e00 > e01) { const int32_t t = e00; e00 = e01; e01 = t; }
    if (e10 > e11) { const int32_t t = e10; e10 = e11; e11 = t; }
    if (e20 > e21) { const int32_t t = e20; e20 = e21; e21 = t; }

    edge_keys[3 * f + 0] = pack_edge_key(e00, e01);
    edge_keys[3 * f + 1] = pack_edge_key(e10, e11);
    edge_keys[3 * f + 2] = pack_edge_key(e20, e21);
}

__global__ void unpack_boundary_keys_kernel(
    const uint64_t* boundary_keys,
    int64_t size,
    int32_t* edge_vertex_ids) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const uint64_t key = boundary_keys[i];
    edge_vertex_ids[2 * i + 0] = edge_key_v0(key);
    edge_vertex_ids[2 * i + 1] = edge_key_v1(key);
}

__global__ void gather_boundary_segments_kernel(
    const float* vertices,
    const int32_t* edge_vertex_ids,
    int64_t num_boundary_edges,
    float* segments) {
    const int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (eid >= num_boundary_edges) return;

    const int32_t v0 = edge_vertex_ids[2 * eid + 0];
    const int32_t v1 = edge_vertex_ids[2 * eid + 1];

    segments[6 * eid + 0] = vertices[3 * v0 + 0];
    segments[6 * eid + 1] = vertices[3 * v0 + 1];
    segments[6 * eid + 2] = vertices[3 * v0 + 2];
    segments[6 * eid + 3] = vertices[3 * v1 + 0];
    segments[6 * eid + 4] = vertices[3 * v1 + 1];
    segments[6 * eid + 5] = vertices[3 * v1 + 2];
}

__global__ void zero_qef_kernel(SymQEF10* qefs, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    qefs[i] = sym10_zero();
}

__global__ void sum_qef_kernel(
    const SymQEF10* qef_init,
    const SymQEF10* qef_face,
    const SymQEF10* qef_boundary,
    int64_t n,
    float face_weight,
    SymQEF10* qef_total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    SymQEF10 q = qef_init[i];
    q = sym10_add(q, sym10_scale(qef_face[i], face_weight));
    q = sym10_add(q, qef_boundary[i]);
    qef_total[i] = q;
}

__global__ void unpack_intersected_kernel(
    const uint8_t* intersected_mask,
    int64_t n,
    bool* intersected_bool) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const uint8_t m = intersected_mask[i];
    intersected_bool[3 * i + 0] = (m & (1u << 0)) != 0;
    intersected_bool[3 * i + 1] = (m & (1u << 1)) != 0;
    intersected_bool[3 * i + 2] = (m & (1u << 2)) != 0;
}


__device__ __forceinline__ int idx4(int r, int c) { return r * 4 + c; }
__device__ __forceinline__ int idx3(int r, int c) { return r * 3 + c; }
__device__ __forceinline__ int idx2(int r, int c) { return r * 2 + c; }

__device__ __forceinline__ void sym10_to_dense4x4(const SymQEF10& q, float Q[16]) {
    Q[idx4(0,0)] = q.q00; Q[idx4(0,1)] = q.q01; Q[idx4(0,2)] = q.q02; Q[idx4(0,3)] = q.q03;
    Q[idx4(1,0)] = q.q01; Q[idx4(1,1)] = q.q11; Q[idx4(1,2)] = q.q12; Q[idx4(1,3)] = q.q13;
    Q[idx4(2,0)] = q.q02; Q[idx4(2,1)] = q.q12; Q[idx4(2,2)] = q.q22; Q[idx4(2,3)] = q.q23;
    Q[idx4(3,0)] = q.q03; Q[idx4(3,1)] = q.q13; Q[idx4(3,2)] = q.q23; Q[idx4(3,3)] = q.q33;
}

__device__ __forceinline__ bool point_inside_box3(
    const float v[3],
    const float min_corner[3],
    const float max_corner[3]) {
    return (
        v[0] >= min_corner[0] && v[0] <= max_corner[0] &&
        v[1] >= min_corner[1] && v[1] <= max_corner[1] &&
        v[2] >= min_corner[2] && v[2] <= max_corner[2]);
}

__device__ __forceinline__ float qef_error4(const float Q[16], const float p[4]) {
    const float y0 = Q[idx4(0,0)] * p[0] + Q[idx4(0,1)] * p[1] + Q[idx4(0,2)] * p[2] + Q[idx4(0,3)] * p[3];
    const float y1 = Q[idx4(1,0)] * p[0] + Q[idx4(1,1)] * p[1] + Q[idx4(1,2)] * p[2] + Q[idx4(1,3)] * p[3];
    const float y2 = Q[idx4(2,0)] * p[0] + Q[idx4(2,1)] * p[1] + Q[idx4(2,2)] * p[2] + Q[idx4(2,3)] * p[3];
    const float y3 = Q[idx4(3,0)] * p[0] + Q[idx4(3,1)] * p[1] + Q[idx4(3,2)] * p[2] + Q[idx4(3,3)] * p[3];
    return p[0] * y0 + p[1] * y1 + p[2] * y2 + p[3] * y3;
}

__device__ __forceinline__ void add_qef_regularization_inplace(
    float Q[16],
    const float mean_sum[3],
    float cnt,
    float regularization_weight) {
    if (regularization_weight <= 0.0f || cnt <= 0.0f) {
        return;
    }

    const float px = mean_sum[0] / cnt;
    const float py = mean_sum[1] / cnt;
    const float pz = mean_sum[2] / cnt;
    const float w = regularization_weight * cnt;

    Q[idx4(0,0)] += w;
    Q[idx4(1,1)] += w;
    Q[idx4(2,2)] += w;

    Q[idx4(0,3)] += -w * px;
    Q[idx4(1,3)] += -w * py;
    Q[idx4(2,3)] += -w * pz;

    Q[idx4(3,0)] += -w * px;
    Q[idx4(3,1)] += -w * py;
    Q[idx4(3,2)] += -w * pz;

    Q[idx4(3,3)] += w * (px * px + py * py + pz * pz);
}

__device__ __forceinline__ void try_single_constraint(
    const float Q[16],
    int fixed_axis,
    const float min_corner[3],
    const float max_corner[3],
    float& best,
    float v_new[3]) {
    const int ax1 = (fixed_axis + 1) % 3;
    const int ax2 = (fixed_axis + 2) % 3;

    float A2[4];
    float B2[4];
    float q2[2];
    float rhs2[2];
    float x2[2];

    A2[idx2(0,0)] = Q[idx4(ax1, ax1)];
    A2[idx2(0,1)] = Q[idx4(ax1, ax2)];
    A2[idx2(1,0)] = Q[idx4(ax2, ax1)];
    A2[idx2(1,1)] = Q[idx4(ax2, ax2)];

    B2[idx2(0,0)] = Q[idx4(ax1, fixed_axis)];
    B2[idx2(0,1)] = Q[idx4(ax1, 3)];
    B2[idx2(1,0)] = Q[idx4(ax2, fixed_axis)];
    B2[idx2(1,1)] = Q[idx4(ax2, 3)];

    q2[0] = min_corner[fixed_axis];
    q2[1] = 1.0f;
    rhs2[0] = -(B2[idx2(0,0)] * q2[0] + B2[idx2(0,1)] * q2[1]);
    rhs2[1] = -(B2[idx2(1,0)] * q2[0] + B2[idx2(1,1)] * q2[1]);
    fdg_gpu::small_cpqr::cpqr_solve_2x2(A2, rhs2, x2);
    if (x2[0] >= min_corner[ax1] && x2[0] <= max_corner[ax1] &&
        x2[1] >= min_corner[ax2] && x2[1] <= max_corner[ax2]) {
        float p4[4];
        p4[fixed_axis] = min_corner[fixed_axis];
        p4[ax1] = x2[0];
        p4[ax2] = x2[1];
        p4[3] = 1.0f;
        const float err = qef_error4(Q, p4);
        if (err < best) {
            best = err;
            v_new[0] = p4[0];
            v_new[1] = p4[1];
            v_new[2] = p4[2];
        }
    }

    q2[0] = max_corner[fixed_axis];
    q2[1] = 1.0f;
    rhs2[0] = -(B2[idx2(0,0)] * q2[0] + B2[idx2(0,1)] * q2[1]);
    rhs2[1] = -(B2[idx2(1,0)] * q2[0] + B2[idx2(1,1)] * q2[1]);
    fdg_gpu::small_cpqr::cpqr_solve_2x2(A2, rhs2, x2);
    if (x2[0] >= min_corner[ax1] && x2[0] <= max_corner[ax1] &&
        x2[1] >= min_corner[ax2] && x2[1] <= max_corner[ax2]) {
        float p4[4];
        p4[fixed_axis] = max_corner[fixed_axis];
        p4[ax1] = x2[0];
        p4[ax2] = x2[1];
        p4[3] = 1.0f;
        const float err = qef_error4(Q, p4);
        if (err < best) {
            best = err;
            v_new[0] = p4[0];
            v_new[1] = p4[1];
            v_new[2] = p4[2];
        }
    }
}

__device__ __forceinline__ void try_two_constraint(
    const float Q[16],
    int free_axis,
    const float min_corner[3],
    const float max_corner[3],
    float& best,
    float v_new[3]) {
    const int ax1 = (free_axis + 1) % 3;
    const int ax2 = (free_axis + 2) % 3;

    const float a = Q[idx4(free_axis, free_axis)];
    const float b0 = Q[idx4(free_axis, ax1)];
    const float b1 = Q[idx4(free_axis, ax2)];
    const float b2 = Q[idx4(free_axis, 3)];

    float rhs = -(b0 * min_corner[ax1] + b1 * min_corner[ax2] + b2);
    float x = fdg_gpu::small_cpqr::solve_1x1_unchecked(a, rhs);
    if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
        float p4[4];
        p4[free_axis] = x;
        p4[ax1] = min_corner[ax1];
        p4[ax2] = min_corner[ax2];
        p4[3] = 1.0f;
        const float err = qef_error4(Q, p4);
        if (err < best) {
            best = err;
            v_new[0] = p4[0];
            v_new[1] = p4[1];
            v_new[2] = p4[2];
        }
    }

    rhs = -(b0 * min_corner[ax1] + b1 * max_corner[ax2] + b2);
    x = fdg_gpu::small_cpqr::solve_1x1_unchecked(a, rhs);
    if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
        float p4[4];
        p4[free_axis] = x;
        p4[ax1] = min_corner[ax1];
        p4[ax2] = max_corner[ax2];
        p4[3] = 1.0f;
        const float err = qef_error4(Q, p4);
        if (err < best) {
            best = err;
            v_new[0] = p4[0];
            v_new[1] = p4[1];
            v_new[2] = p4[2];
        }
    }

    rhs = -(b0 * max_corner[ax1] + b1 * min_corner[ax2] + b2);
    x = fdg_gpu::small_cpqr::solve_1x1_unchecked(a, rhs);
    if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
        float p4[4];
        p4[free_axis] = x;
        p4[ax1] = max_corner[ax1];
        p4[ax2] = min_corner[ax2];
        p4[3] = 1.0f;
        const float err = qef_error4(Q, p4);
        if (err < best) {
            best = err;
            v_new[0] = p4[0];
            v_new[1] = p4[1];
            v_new[2] = p4[2];
        }
    }

    rhs = -(b0 * max_corner[ax1] + b1 * max_corner[ax2] + b2);
    x = fdg_gpu::small_cpqr::solve_1x1_unchecked(a, rhs);
    if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
        float p4[4];
        p4[free_axis] = x;
        p4[ax1] = max_corner[ax1];
        p4[ax2] = max_corner[ax2];
        p4[3] = 1.0f;
        const float err = qef_error4(Q, p4);
        if (err < best) {
            best = err;
            v_new[0] = p4[0];
            v_new[1] = p4[1];
            v_new[2] = p4[2];
        }
    }
}

__device__ __forceinline__ void try_three_constraint(
    const float Q[16],
    const float min_corner[3],
    const float max_corner[3],
    float& best,
    float v_new[3]) {
    for (int x_constraint = 0; x_constraint < 2; ++x_constraint) {
        for (int y_constraint = 0; y_constraint < 2; ++y_constraint) {
            for (int z_constraint = 0; z_constraint < 2; ++z_constraint) {
                float p4[4];
                p4[0] = x_constraint ? min_corner[0] : max_corner[0];
                p4[1] = y_constraint ? min_corner[1] : max_corner[1];
                p4[2] = z_constraint ? min_corner[2] : max_corner[2];
                p4[3] = 1.0f;
                const float err = qef_error4(Q, p4);
                if (err < best) {
                    best = err;
                    v_new[0] = p4[0];
                    v_new[1] = p4[1];
                    v_new[2] = p4[2];
                }
            }
        }
    }
}

__global__ void solve_qef_full_kernel(
    const int* voxel_coords,
    const float* mean_sum,
    const float* cnt,
    const SymQEF10* qef_total,
    int64_t n,
    float3 voxel_size,
    float regularization_weight,
    float* dual_vertices) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int x = voxel_coords[3 * i + 0];
    const int y = voxel_coords[3 * i + 1];
    const int z = voxel_coords[3 * i + 2];

    float min_corner[3] = {
        x * voxel_size.x,
        y * voxel_size.y,
        z * voxel_size.z,
    };
    float max_corner[3] = {
        (x + 1) * voxel_size.x,
        (y + 1) * voxel_size.y,
        (z + 1) * voxel_size.z,
    };

    float Q[16];
    sym10_to_dense4x4(qef_total[i], Q);

    const float mean_i[3] = {
        mean_sum[3 * i + 0],
        mean_sum[3 * i + 1],
        mean_sum[3 * i + 2],
    };
    add_qef_regularization_inplace(Q, mean_i, cnt[i], regularization_weight);

    float A3[9];
    float b3[3];
    float v_new[3];

    A3[idx3(0,0)] = Q[idx4(0,0)];
    A3[idx3(0,1)] = Q[idx4(0,1)];
    A3[idx3(0,2)] = Q[idx4(0,2)];
    A3[idx3(1,0)] = Q[idx4(1,0)];
    A3[idx3(1,1)] = Q[idx4(1,1)];
    A3[idx3(1,2)] = Q[idx4(1,2)];
    A3[idx3(2,0)] = Q[idx4(2,0)];
    A3[idx3(2,1)] = Q[idx4(2,1)];
    A3[idx3(2,2)] = Q[idx4(2,2)];

    b3[0] = -Q[idx4(0,3)];
    b3[1] = -Q[idx4(1,3)];
    b3[2] = -Q[idx4(2,3)];

    fdg_gpu::small_cpqr::cpqr_solve_3x3(A3, b3, v_new);

    if (!point_inside_box3(v_new, min_corner, max_corner)) {
        float best = CUDART_INF_F;
        try_single_constraint(Q, 0, min_corner, max_corner, best, v_new);
        try_single_constraint(Q, 1, min_corner, max_corner, best, v_new);
        try_single_constraint(Q, 2, min_corner, max_corner, best, v_new);
        try_two_constraint(Q, 0, min_corner, max_corner, best, v_new);
        try_two_constraint(Q, 1, min_corner, max_corner, best, v_new);
        try_two_constraint(Q, 2, min_corner, max_corner, best, v_new);
        try_three_constraint(Q, min_corner, max_corner, best, v_new);
    }

    dual_vertices[3 * i + 0] = v_new[0];
    dual_vertices[3 * i + 1] = v_new[1];
    dual_vertices[3 * i + 2] = v_new[2];
}

inline FlatTriangles build_flat_triangles_gpu(
    const float* vertices,
    const int32_t* faces,
    int64_t num_faces,
    cudaStream_t stream) {
    FlatTriangles out;
    out.num_triangles = num_faces;
    out.triangles.allocate(9 * num_faces);
    if (num_faces == 0) return out;

    gather_flat_triangles_kernel<<<ceil_div_i64(3 * num_faces, kBlockSize), kBlockSize, 0, stream>>>(
        vertices,
        faces,
        num_faces,
        out.triangles.data());
    throw_cuda_error(cudaGetLastError(), "gather_flat_triangles_kernel");
    return out;
}

inline BoundaryEdgeIndexResult detect_boundary_edges_gpu(
    const int32_t* faces,
    int64_t num_faces,
    cudaStream_t stream) {
    BoundaryEdgeIndexResult out;
    if (num_faces == 0) return out;

    DeviceBuffer<uint64_t> edge_keys(3 * num_faces);
    emit_face_edges_kernel<<<ceil_div_i64(num_faces, kBlockSize), kBlockSize, 0, stream>>>(
        faces,
        num_faces,
        edge_keys.data());
    throw_cuda_error(cudaGetLastError(), "emit_face_edges_kernel");

    auto policy = thrust::cuda::par.on(stream);
    auto edge_keys_begin = thrust::device_pointer_cast(edge_keys.data());
    thrust::sort(policy, edge_keys_begin, edge_keys_begin + 3 * num_faces);

    DeviceBuffer<uint64_t> unique_keys(3 * num_faces);
    DeviceBuffer<int> counts(3 * num_faces);
    auto reduce_end = thrust::reduce_by_key(
        policy,
        edge_keys_begin,
        edge_keys_begin + 3 * num_faces,
        thrust::make_constant_iterator<int>(1),
        thrust::device_pointer_cast(unique_keys.data()),
        thrust::device_pointer_cast(counts.data()));
    const int64_t unique_size = reduce_end.first - thrust::device_pointer_cast(unique_keys.data());
    if (unique_size == 0) return out;

    const int64_t boundary_count = thrust::count_if(
        policy,
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(counts.data()) + unique_size,
        IsOne{});
    out.size = boundary_count;
    out.edge_vertex_ids.allocate(2 * boundary_count);
    if (boundary_count == 0) return out;

    DeviceBuffer<uint64_t> boundary_keys(boundary_count);
    auto copied_end = thrust::copy_if(
        policy,
        thrust::device_pointer_cast(unique_keys.data()),
        thrust::device_pointer_cast(unique_keys.data()) + unique_size,
        thrust::device_pointer_cast(counts.data()),
        thrust::device_pointer_cast(boundary_keys.data()),
        IsOne{});
    const int64_t copied = copied_end - thrust::device_pointer_cast(boundary_keys.data());
    if (copied != boundary_count) {
        throw std::runtime_error("boundary edge count mismatch");
    }

    unpack_boundary_keys_kernel<<<ceil_div_i64(boundary_count, kBlockSize), kBlockSize, 0, stream>>>(
        boundary_keys.data(),
        boundary_count,
        out.edge_vertex_ids.data());
    throw_cuda_error(cudaGetLastError(), "unpack_boundary_keys_kernel");
    return out;
}

inline BoundarySegments gather_boundary_segments_gpu(
    const float* vertices,
    const BoundaryEdgeIndexResult& boundary_edges,
    cudaStream_t stream) {
    BoundarySegments out;
    out.size = boundary_edges.size;
    out.segments.allocate(6 * out.size);
    if (out.size == 0) return out;

    gather_boundary_segments_kernel<<<ceil_div_i64(out.size, kBlockSize), kBlockSize, 0, stream>>>(
        vertices,
        boundary_edges.edge_vertex_ids.data(),
        out.size,
        out.segments.data());
    throw_cuda_error(cudaGetLastError(), "gather_boundary_segments_kernel");
    return out;
}

inline DeviceBuffer<SymQEF10> make_zero_qef_buffer(int64_t n, cudaStream_t stream) {
    DeviceBuffer<SymQEF10> out(n);
    if (n > 0) {
        zero_qef_kernel<<<ceil_div_i64(n, kBlockSize), kBlockSize, 0, stream>>>(out.data(), n);
        throw_cuda_error(cudaGetLastError(), "zero_qef_kernel");
    }
    return out;
}

}  // namespace

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
    FlexibleDualGridGPUOutput* out) {
    if (out == nullptr) {
        return cudaErrorInvalidValue;
    }
    out->size = 0;
    out->voxel_coords = nullptr;
    out->dual_vertices = nullptr;
    out->intersected = nullptr;

    if (num_vertices < 0 || num_faces < 0) {
        return cudaErrorInvalidValue;
    }
    if (!(voxel_size.x > 0.0f && voxel_size.y > 0.0f && voxel_size.z > 0.0f)) {
        return cudaErrorInvalidValue;
    }
    if (grid_max.x <= grid_min.x || grid_max.y <= grid_min.y || grid_max.z <= grid_min.z) {
        return cudaErrorInvalidValue;
    }
    if (num_vertices > 0 && vertices == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (num_faces > 0 && faces == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (intersect_chunk_triangles <= 0 || boundary_chunk_steps <= 0) {
        return cudaErrorInvalidValue;
    }

    try {
        FlatTriangles flat = build_flat_triangles_gpu(vertices, faces, num_faces, stream);

        auto surface = intersection_qef::intersect_qef_gpu(
            flat.triangles.data(),
            flat.num_triangles,
            voxel_size,
            grid_min,
            grid_max,
            intersect_chunk_triangles,
            stream);

        auto face_qefs = make_zero_qef_buffer(surface.size, stream);
        if (surface.size > 0 && face_weight > 0.0f) {
            auto face_result = oct_pairs::face_qef_gpu(
                voxel_size,
                grid_min,
                grid_max,
                flat.triangles.data(),
                flat.num_triangles,
                surface.voxels.data(),
                surface.size,
                stream);
            face_qefs = std::move(face_result.qefs);
        }

        auto boundary_qefs = make_zero_qef_buffer(surface.size, stream);
        if (surface.size > 0 && boundary_weight > 0.0f) {
            BoundaryEdgeIndexResult boundary_edges = detect_boundary_edges_gpu(faces, num_faces, stream);
            BoundarySegments boundary_segments = gather_boundary_segments_gpu(vertices, boundary_edges, stream);
            auto boundary_result = edge_dda::boundary_qef_gpu(
                voxel_size,
                grid_min,
                grid_max,
                boundary_segments.segments.data(),
                boundary_segments.size,
                boundary_weight,
                surface.voxels.data(),
                surface.size,
                boundary_chunk_steps,
                stream);
            boundary_qefs = std::move(boundary_result.qefs);
        }

        DeviceBuffer<SymQEF10> qef_total(surface.size);
        if (surface.size > 0) {
            sum_qef_kernel<<<ceil_div_i64(surface.size, kBlockSize), kBlockSize, 0, stream>>>(
                surface.qefs.data(),
                face_qefs.data(),
                boundary_qefs.data(),
                surface.size,
                face_weight,
                qef_total.data());
            throw_cuda_error(cudaGetLastError(), "sum_qef_kernel");
        }

        DeviceBuffer<float> dual_vertices(3 * surface.size);
        if (surface.size > 0) {
            solve_qef_full_kernel<<<ceil_div_i64(surface.size, kBlockSize), kBlockSize, 0, stream>>>(
                surface.voxels.data(),
                surface.mean_sum.data(),
                surface.cnt.data(),
                qef_total.data(),
                surface.size,
                voxel_size,
                regularization_weight,
                dual_vertices.data());
            throw_cuda_error(cudaGetLastError(), "solve_qef_full_kernel");
        }

        DeviceBuffer<bool> intersected_bool(3 * surface.size);
        if (surface.size > 0) {
            unpack_intersected_kernel<<<ceil_div_i64(surface.size, kBlockSize), kBlockSize, 0, stream>>>(
                surface.intersected.data(),
                surface.size,
                intersected_bool.data());
            throw_cuda_error(cudaGetLastError(), "unpack_intersected_kernel");
        }

        out->size = surface.size;
        out->voxel_coords = surface.voxels.release_ownership();
        out->dual_vertices = dual_vertices.release_ownership();
        out->intersected = intersected_bool.release_ownership();
        return cudaSuccess;
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (const std::invalid_argument&) {
        return cudaErrorInvalidValue;
    } catch (const std::exception&) {
        return cudaErrorUnknown;
    }
}

void free_flexible_dual_grid_gpu_output(FlexibleDualGridGPUOutput* out) noexcept {
    if (out == nullptr) return;
    if (out->voxel_coords) cudaFree(out->voxel_coords);
    if (out->dual_vertices) cudaFree(out->dual_vertices);
    if (out->intersected) cudaFree(out->intersected);
    out->size = 0;
    out->voxel_coords = nullptr;
    out->dual_vertices = nullptr;
    out->intersected = nullptr;
}

}  // namespace fdg_gpu
