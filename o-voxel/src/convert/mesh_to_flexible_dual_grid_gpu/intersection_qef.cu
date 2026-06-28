#include "intersection_qef.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace intersection_qef {
namespace {

using fdg_gpu::DeviceBuffer;
using fdg_gpu::SymQEF10;
using fdg_gpu::int3_;
using fdg_gpu::throw_cuda_error;

#define IQ_CUDA_CHECK(expr) ::fdg_gpu::throw_cuda_error((expr), #expr)

struct D2 {
    double x;
    double z;
};

struct D3 {
    double x;
    double y;
    double z;

    __host__ __device__ double operator[](int i) const { return (&x)[i]; }
};

struct QEFEventValue {
    float mean_sum_x;
    float mean_sum_y;
    float mean_sum_z;
    float cnt;
    uint8_t intersected;
    SymQEF10 qef;
};

struct OccChunk {
    DeviceBuffer<uint64_t> keys;
    int64_t size = 0;
};

struct QEFChunk {
    DeviceBuffer<uint64_t> keys;
    DeviceBuffer<QEFEventValue> values;
    int64_t size = 0;
};

struct AddQEFEventValue {
    __host__ __device__ QEFEventValue operator()(const QEFEventValue& a, const QEFEventValue& b) const {
        QEFEventValue out;
        out.mean_sum_x = a.mean_sum_x + b.mean_sum_x;
        out.mean_sum_y = a.mean_sum_y + b.mean_sum_y;
        out.mean_sum_z = a.mean_sum_z + b.mean_sum_z;
        out.cnt = a.cnt + b.cnt;
        out.intersected = static_cast<uint8_t>(a.intersected | b.intersected);
        out.qef.q00 = a.qef.q00 + b.qef.q00;
        out.qef.q01 = a.qef.q01 + b.qef.q01;
        out.qef.q02 = a.qef.q02 + b.qef.q02;
        out.qef.q03 = a.qef.q03 + b.qef.q03;
        out.qef.q11 = a.qef.q11 + b.qef.q11;
        out.qef.q12 = a.qef.q12 + b.qef.q12;
        out.qef.q13 = a.qef.q13 + b.qef.q13;
        out.qef.q22 = a.qef.q22 + b.qef.q22;
        out.qef.q23 = a.qef.q23 + b.qef.q23;
        out.qef.q33 = a.qef.q33 + b.qef.q33;
        return out;
    }
};

__host__ __device__ inline double lerp_scalar(double a, double b, double t, double va, double vb) {
    if (a == b) return va;
    const double alpha = (t - a) / (b - a);
    return (1.0 - alpha) * va + alpha * vb;
}

__host__ __device__ inline D2 lerp_vec2(double a, double b, double t, D2 va, D2 vb) {
    if (a == b) return va;
    const double alpha = (t - a) / (b - a);
    return D2{(1.0 - alpha) * va.x + alpha * vb.x, (1.0 - alpha) * va.z + alpha * vb.z};
}

__host__ __device__ inline int clamp_int(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__host__ __device__ inline void swap_d3(D3& a, D3& b) {
    D3 t = a;
    a = b;
    b = t;
}

__host__ __device__ inline void sort_by_y(D3& t0, D3& t1, D3& t2) {
    if (t0.y > t1.y) swap_d3(t0, t1);
    if (t1.y > t2.y) swap_d3(t1, t2);
    if (t0.y > t1.y) swap_d3(t0, t1);
}

__device__ inline void normalize3(double& x, double& y, double& z) {
    const double n = sqrt(x * x + y * y + z * z);
    if (n > 0.0) {
        x /= n;
        y /= n;
        z /= n;
    }
}

__device__ inline SymQEF10 make_plane_qef_from_triangle(const double v0[3], const double v1[3], const double v2[3]) {
    const double e0x = v1[0] - v0[0];
    const double e0y = v1[1] - v0[1];
    const double e0z = v1[2] - v0[2];

    const double e1x = v2[0] - v1[0];
    const double e1y = v2[1] - v1[1];
    const double e1z = v2[2] - v1[2];

    double nx = e0y * e1z - e0z * e1y;
    double ny = e0z * e1x - e0x * e1z;
    double nz = e0x * e1y - e0y * e1x;
    normalize3(nx, ny, nz);

    const double d = -(nx * v0[0] + ny * v0[1] + nz * v0[2]);

    SymQEF10 q;
    q.q00 = static_cast<float>(nx * nx);
    q.q01 = static_cast<float>(nx * ny);
    q.q02 = static_cast<float>(nx * nz);
    q.q03 = static_cast<float>(nx * d);
    q.q11 = static_cast<float>(ny * ny);
    q.q12 = static_cast<float>(ny * nz);
    q.q13 = static_cast<float>(ny * d);
    q.q22 = static_cast<float>(nz * nz);
    q.q23 = static_cast<float>(nz * d);
    q.q33 = static_cast<float>(d * d);
    return q;
}

__device__ inline int64_t count_triangle_axis_surface_voxels(
    const float* tri,
    int ax2,
    const float voxel_size[3],
    int3_ grid_min,
    int3_ grid_max) {
    const double v0[3] = {static_cast<double>(tri[0]), static_cast<double>(tri[1]), static_cast<double>(tri[2])};
    const double v1[3] = {static_cast<double>(tri[3]), static_cast<double>(tri[4]), static_cast<double>(tri[5])};
    const double v2[3] = {static_cast<double>(tri[6]), static_cast<double>(tri[7]), static_cast<double>(tri[8])};

    const int ax0 = (ax2 + 1) % 3;
    const int ax1 = (ax2 + 2) % 3;

    D3 t0{v0[ax0], v0[ax1], v0[ax2]};
    D3 t1{v1[ax0], v1[ax1], v1[ax2]};
    D3 t2{v2[ax0], v2[ax1], v2[ax2]};
    sort_by_y(t0, t1, t2);

    const int start = clamp_int(static_cast<int>(t0.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
    const int mid = clamp_int(static_cast<int>(t1.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
    const int end = clamp_int(static_cast<int>(t2.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);

    int64_t total = 0;
    auto scan_half = [&](int row_start, int row_end, D3 a, D3 b, D3 c) {
        for (int y_idx = row_start; y_idx < row_end; ++y_idx) {
            const double y = (static_cast<double>(y_idx) + 1.0) * voxel_size[ax1];
            D2 t3 = lerp_vec2(a.y, b.y, y, D2{a.x, a.z}, D2{b.x, b.z});
            D2 t4 = lerp_vec2(a.y, c.y, y, D2{a.x, a.z}, D2{c.x, c.z});
            if (t3.x > t4.x) {
                D2 tmp = t3;
                t3 = t4;
                t4 = tmp;
            }

            const int line_start = clamp_int(static_cast<int>(t3.x / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
            const int line_end = clamp_int(static_cast<int>(t4.x / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
            for (int x_idx = line_start; x_idx < line_end; ++x_idx) {
                const double x = (static_cast<double>(x_idx) + 1.0) * voxel_size[ax0];
                const double z = lerp_scalar(t3.x, t4.x, x, t3.z, t4.z);
                const int z_idx = static_cast<int>(z / voxel_size[ax2]);
                if (z_idx < grid_min[ax2] || z_idx >= grid_max[ax2]) continue;
                total += 4;
            }
        }
    };

    scan_half(start, mid, t0, t1, t2);
    scan_half(mid, end, t2, t1, t0);
    return total;
}

__global__ void intersection_count_kernel(
    const float* triangles,
    int64_t tri_begin,
    int64_t tri_count,
    float vx,
    float vy,
    float vz,
    int3_ grid_min,
    int3_ grid_max,
    int64_t* counts) {
    const int64_t local_t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_t >= tri_count) return;

    const int64_t t = tri_begin + local_t;
    const float* tri = triangles + t * 9;
    const float voxel_size[3] = {vx, vy, vz};

    int64_t total = 0;
    total += count_triangle_axis_surface_voxels(tri, 0, voxel_size, grid_min, grid_max);
    total += count_triangle_axis_surface_voxels(tri, 1, voxel_size, grid_min, grid_max);
    total += count_triangle_axis_surface_voxels(tri, 2, voxel_size, grid_min, grid_max);
    counts[local_t] = total;
}

__global__ void intersection_occ_emit_kernel(
    const float* triangles,
    int64_t tri_begin,
    int64_t tri_count,
    float vx,
    float vy,
    float vz,
    int3_ grid_min,
    int3_ grid_max,
    const int64_t* offsets,
    uint64_t* event_keys) {
    const int64_t local_t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_t >= tri_count) return;

    const int64_t t = tri_begin + local_t;
    const float* tri = triangles + t * 9;
    const double v0[3] = {static_cast<double>(tri[0]), static_cast<double>(tri[1]), static_cast<double>(tri[2])};
    const double v1[3] = {static_cast<double>(tri[3]), static_cast<double>(tri[4]), static_cast<double>(tri[5])};
    const double v2[3] = {static_cast<double>(tri[6]), static_cast<double>(tri[7]), static_cast<double>(tri[8])};
    const float voxel_size[3] = {vx, vy, vz};

    int64_t out = offsets[local_t];

    for (int ax2 = 0; ax2 < 3; ++ax2) {
        const int ax0 = (ax2 + 1) % 3;
        const int ax1 = (ax2 + 2) % 3;

        D3 t0{v0[ax0], v0[ax1], v0[ax2]};
        D3 t1{v1[ax0], v1[ax1], v1[ax2]};
        D3 t2{v2[ax0], v2[ax1], v2[ax2]};
        sort_by_y(t0, t1, t2);

        const int start = clamp_int(static_cast<int>(t0.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
        const int mid = clamp_int(static_cast<int>(t1.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
        const int end = clamp_int(static_cast<int>(t2.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);

        auto emit_one = [&](int x_idx, int y_idx, int z_idx) {
            int coord[3];
            coord[ax0] = x_idx;
            coord[ax1] = y_idx;
            coord[ax2] = z_idx;
            event_keys[out++] = fdg_gpu::pack_voxel_key(coord[0], coord[1], coord[2], grid_min, grid_max);
        };

        auto scan_half = [&](int row_start, int row_end, D3 a, D3 b, D3 c) {
            for (int y_idx = row_start; y_idx < row_end; ++y_idx) {
                const double y = (static_cast<double>(y_idx) + 1.0) * voxel_size[ax1];
                D2 t3 = lerp_vec2(a.y, b.y, y, D2{a.x, a.z}, D2{b.x, b.z});
                D2 t4 = lerp_vec2(a.y, c.y, y, D2{a.x, a.z}, D2{c.x, c.z});
                if (t3.x > t4.x) {
                    D2 tmp = t3;
                    t3 = t4;
                    t4 = tmp;
                }

                const int line_start = clamp_int(static_cast<int>(t3.x / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
                const int line_end = clamp_int(static_cast<int>(t4.x / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);

                for (int x_idx = line_start; x_idx < line_end; ++x_idx) {
                    const double x = (static_cast<double>(x_idx) + 1.0) * voxel_size[ax0];
                    const double z = lerp_scalar(t3.x, t4.x, x, t3.z, t4.z);
                    const int z_idx = static_cast<int>(z / voxel_size[ax2]);
                    if (z_idx < grid_min[ax2] || z_idx >= grid_max[ax2]) continue;

                    emit_one(x_idx + 0, y_idx + 0, z_idx);
                    emit_one(x_idx + 1, y_idx + 0, z_idx);
                    emit_one(x_idx + 0, y_idx + 1, z_idx);
                    emit_one(x_idx + 1, y_idx + 1, z_idx);
                }
            }
        };

        scan_half(start, mid, t0, t1, t2);
        scan_half(mid, end, t2, t1, t0);
    }
}

__global__ void intersect_qef_emit_kernel(
    const float* triangles,
    int64_t tri_begin,
    int64_t tri_count,
    float vx,
    float vy,
    float vz,
    int3_ grid_min,
    int3_ grid_max,
    const int64_t* offsets,
    uint64_t* event_keys,
    QEFEventValue* event_values) {
    const int64_t local_t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_t >= tri_count) return;

    const int64_t t = tri_begin + local_t;
    const float* tri = triangles + t * 9;
    const double v0[3] = {static_cast<double>(tri[0]), static_cast<double>(tri[1]), static_cast<double>(tri[2])};
    const double v1[3] = {static_cast<double>(tri[3]), static_cast<double>(tri[4]), static_cast<double>(tri[5])};
    const double v2[3] = {static_cast<double>(tri[6]), static_cast<double>(tri[7]), static_cast<double>(tri[8])};
    const float voxel_size[3] = {vx, vy, vz};
    const SymQEF10 qef = make_plane_qef_from_triangle(v0, v1, v2);

    int64_t out = offsets[local_t];

    for (int ax2 = 0; ax2 < 3; ++ax2) {
        const int ax0 = (ax2 + 1) % 3;
        const int ax1 = (ax2 + 2) % 3;

        D3 t0{v0[ax0], v0[ax1], v0[ax2]};
        D3 t1{v1[ax0], v1[ax1], v1[ax2]};
        D3 t2{v2[ax0], v2[ax1], v2[ax2]};
        sort_by_y(t0, t1, t2);

        const int start = clamp_int(static_cast<int>(t0.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
        const int mid = clamp_int(static_cast<int>(t1.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
        const int end = clamp_int(static_cast<int>(t2.y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);

        auto emit_one = [&](int x_idx, int y_idx, int z_idx, double x, double y, double z, uint8_t mask) {
            int coord[3];
            coord[ax0] = x_idx;
            coord[ax1] = y_idx;
            coord[ax2] = z_idx;
            event_keys[out] = fdg_gpu::pack_voxel_key(coord[0], coord[1], coord[2], grid_min, grid_max);
            event_values[out].mean_sum_x = static_cast<float>(ax0 == 0 ? x : (ax1 == 0 ? y : z));
            event_values[out].mean_sum_y = static_cast<float>(ax0 == 1 ? x : (ax1 == 1 ? y : z));
            event_values[out].mean_sum_z = static_cast<float>(ax0 == 2 ? x : (ax1 == 2 ? y : z));
            event_values[out].cnt = 1.0f;
            event_values[out].intersected = mask;
            event_values[out].qef = qef;
            ++out;
        };

        auto scan_half = [&](int row_start, int row_end, D3 a, D3 b, D3 c) {
            for (int y_idx = row_start; y_idx < row_end; ++y_idx) {
                const double y = (static_cast<double>(y_idx) + 1.0) * voxel_size[ax1];
                D2 t3 = lerp_vec2(a.y, b.y, y, D2{a.x, a.z}, D2{b.x, b.z});
                D2 t4 = lerp_vec2(a.y, c.y, y, D2{a.x, a.z}, D2{c.x, c.z});
                if (t3.x > t4.x) {
                    D2 tmp = t3;
                    t3 = t4;
                    t4 = tmp;
                }

                const int line_start = clamp_int(static_cast<int>(t3.x / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
                const int line_end = clamp_int(static_cast<int>(t4.x / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);

                for (int x_idx = line_start; x_idx < line_end; ++x_idx) {
                    const double x = (static_cast<double>(x_idx) + 1.0) * voxel_size[ax0];
                    const double z = lerp_scalar(t3.x, t4.x, x, t3.z, t4.z);
                    const int z_idx = static_cast<int>(z / voxel_size[ax2]);
                    if (z_idx < grid_min[ax2] || z_idx >= grid_max[ax2]) continue;

                    emit_one(x_idx + 0, y_idx + 0, z_idx, x, y, z, static_cast<uint8_t>(1u << ax2));
                    emit_one(x_idx + 1, y_idx + 0, z_idx, x, y, z, static_cast<uint8_t>(0u));
                    emit_one(x_idx + 0, y_idx + 1, z_idx, x, y, z, static_cast<uint8_t>(0u));
                    emit_one(x_idx + 1, y_idx + 1, z_idx, x, y, z, static_cast<uint8_t>(0u));
                }
            }
        };

        scan_half(start, mid, t0, t1, t2);
        scan_half(mid, end, t2, t1, t0);
    }
}

__global__ void decode_occ_output_kernel(
    const uint64_t* keys,
    int64_t size,
    int3_ grid_min,
    int3_ grid_max,
    int* out_voxels) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    int x, y, z;
    const uint64_t key = keys[i];
    const uint64_t sx = static_cast<uint64_t>(grid_max.x - grid_min.x);
    const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
    const uint64_t yz = sx * sy;
    const uint64_t zz = key / yz;
    const uint64_t rem = key - zz * yz;
    const uint64_t yy = rem / sx;
    const uint64_t xx = rem - yy * sx;
    x = static_cast<int>(xx) + grid_min.x;
    y = static_cast<int>(yy) + grid_min.y;
    z = static_cast<int>(zz) + grid_min.z;
    out_voxels[3 * i + 0] = x;
    out_voxels[3 * i + 1] = y;
    out_voxels[3 * i + 2] = z;
}

__global__ void decode_qef_output_kernel(
    const uint64_t* keys,
    const QEFEventValue* values,
    int64_t size,
    int3_ grid_min,
    int3_ grid_max,
    int* out_voxels,
    float* out_mean_sum,
    float* out_cnt,
    uint8_t* out_intersected,
    SymQEF10* out_qefs) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;

    const uint64_t key = keys[i];
    const uint64_t sx = static_cast<uint64_t>(grid_max.x - grid_min.x);
    const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
    const uint64_t yz = sx * sy;
    const uint64_t zz = key / yz;
    const uint64_t rem = key - zz * yz;
    const uint64_t yy = rem / sx;
    const uint64_t xx = rem - yy * sx;

    out_voxels[3 * i + 0] = static_cast<int>(xx) + grid_min.x;
    out_voxels[3 * i + 1] = static_cast<int>(yy) + grid_min.y;
    out_voxels[3 * i + 2] = static_cast<int>(zz) + grid_min.z;

    out_mean_sum[3 * i + 0] = values[i].mean_sum_x;
    out_mean_sum[3 * i + 1] = values[i].mean_sum_y;
    out_mean_sum[3 * i + 2] = values[i].mean_sum_z;
    out_cnt[i] = values[i].cnt;
    out_intersected[i] = values[i].intersected;
    out_qefs[i] = values[i].qef;
}

inline int64_t copy_last_i64(const int64_t* ptr, int64_t count, cudaStream_t stream) {
    if (count <= 0) return 0;
    int64_t value = 0;
    IQ_CUDA_CHECK(cudaMemcpyAsync(&value, ptr + (count - 1), sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    IQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    return value;
}

OccChunk make_occ_chunk_exact(DeviceBuffer<uint64_t>& src_keys, int64_t size, cudaStream_t stream) {
    OccChunk out;
    out.size = size;
    if (size <= 0) return out;
    out.keys.allocate(size);
    thrust::copy_n(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(src_keys.data()),
        size,
        thrust::device_pointer_cast(out.keys.data()));
    return out;
}

QEFChunk make_qef_chunk_exact(
    DeviceBuffer<uint64_t>& src_keys,
    DeviceBuffer<QEFEventValue>& src_values,
    int64_t size,
    cudaStream_t stream) {
    QEFChunk out;
    out.size = size;
    if (size <= 0) return out;
    out.keys.allocate(size);
    out.values.allocate(size);
    thrust::copy_n(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(src_keys.data()),
        size,
        thrust::device_pointer_cast(out.keys.data()));
    thrust::copy_n(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(src_values.data()),
        size,
        thrust::device_pointer_cast(out.values.data()));
    return out;
}

OccChunk merge_occ_two_chunks(OccChunk a, OccChunk b, cudaStream_t stream) {
    if (a.size == 0) return std::move(b);
    if (b.size == 0) return std::move(a);

    DeviceBuffer<uint64_t> merged(a.size + b.size);
    auto merged_end = thrust::merge(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(a.keys.data()),
        thrust::device_pointer_cast(a.keys.data()) + a.size,
        thrust::device_pointer_cast(b.keys.data()),
        thrust::device_pointer_cast(b.keys.data()) + b.size,
        thrust::device_pointer_cast(merged.data()));
    const int64_t merged_size = merged_end - thrust::device_pointer_cast(merged.data());

    auto unique_end = thrust::unique(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(merged.data()),
        thrust::device_pointer_cast(merged.data()) + merged_size);

    OccChunk out;
    out.size = unique_end - thrust::device_pointer_cast(merged.data());
    out.keys = std::move(merged);
    return out;
}

QEFChunk merge_qef_two_chunks(QEFChunk a, QEFChunk b, cudaStream_t stream) {
    if (a.size == 0) return std::move(b);
    if (b.size == 0) return std::move(a);

    DeviceBuffer<uint64_t> merged_keys(a.size + b.size);
    DeviceBuffer<QEFEventValue> merged_values(a.size + b.size);

    auto merged_end = thrust::merge_by_key(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(a.keys.data()),
        thrust::device_pointer_cast(a.keys.data()) + a.size,
        thrust::device_pointer_cast(b.keys.data()),
        thrust::device_pointer_cast(b.keys.data()) + b.size,
        thrust::device_pointer_cast(a.values.data()),
        thrust::device_pointer_cast(b.values.data()),
        thrust::device_pointer_cast(merged_keys.data()),
        thrust::device_pointer_cast(merged_values.data()));
    const int64_t merged_size = merged_end.first - thrust::device_pointer_cast(merged_keys.data());

    DeviceBuffer<uint64_t> next_keys(merged_size);
    DeviceBuffer<QEFEventValue> next_values(merged_size);
    auto next_end = thrust::reduce_by_key(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(merged_keys.data()),
        thrust::device_pointer_cast(merged_keys.data()) + merged_size,
        thrust::device_pointer_cast(merged_values.data()),
        thrust::device_pointer_cast(next_keys.data()),
        thrust::device_pointer_cast(next_values.data()),
        thrust::equal_to<uint64_t>(),
        AddQEFEventValue());

    QEFChunk out;
    out.size = next_end.first - thrust::device_pointer_cast(next_keys.data());
    out.keys = std::move(next_keys);
    out.values = std::move(next_values);
    return out;
}

OccChunk final_merge_occ_chunks(std::vector<OccChunk> chunks, cudaStream_t stream) {
    if (chunks.empty()) return OccChunk{};
    while (chunks.size() > 1) {
        std::vector<OccChunk> next_level;
        next_level.reserve((chunks.size() + 1) / 2);
        for (size_t i = 0; i < chunks.size(); i += 2) {
            if (i + 1 >= chunks.size()) {
                next_level.push_back(std::move(chunks[i]));
            } else {
                next_level.push_back(merge_occ_two_chunks(std::move(chunks[i]), std::move(chunks[i + 1]), stream));
            }
        }
        chunks = std::move(next_level);
    }
    return std::move(chunks[0]);
}

QEFChunk final_merge_qef_chunks(std::vector<QEFChunk> chunks, cudaStream_t stream) {
    if (chunks.empty()) return QEFChunk{};
    while (chunks.size() > 1) {
        std::vector<QEFChunk> next_level;
        next_level.reserve((chunks.size() + 1) / 2);
        for (size_t i = 0; i < chunks.size(); i += 2) {
            if (i + 1 >= chunks.size()) {
                next_level.push_back(std::move(chunks[i]));
            } else {
                next_level.push_back(merge_qef_two_chunks(std::move(chunks[i]), std::move(chunks[i + 1]), stream));
            }
        }
        chunks = std::move(next_level);
    }
    return std::move(chunks[0]);
}

IntersectionOccResult run_occ_impl(
    const float* triangles,
    int64_t num_triangles,
    float3 voxel_size,
    int3_ grid_min,
    int3_ grid_max,
    int64_t chunk_triangles,
    cudaStream_t stream) {
    if (num_triangles < 0) throw std::invalid_argument("num_triangles must be non-negative");
    if (chunk_triangles <= 0) throw std::invalid_argument("chunk_triangles must be positive");

    constexpr int threads = 256;
    std::vector<OccChunk> chunks;
    chunks.reserve(static_cast<size_t>((num_triangles + chunk_triangles - 1) / chunk_triangles));

    for (int64_t tri_begin = 0; tri_begin < num_triangles; tri_begin += chunk_triangles) {
        const int64_t tri_count = std::min<int64_t>(chunk_triangles, num_triangles - tri_begin);
        if (tri_count == 0) continue;

        DeviceBuffer<int64_t> counts(tri_count);
        const int blocks = fdg_gpu::ceil_div_i64(tri_count, threads);
        intersection_count_kernel<<<blocks, threads, 0, stream>>>(
            triangles,
            tri_begin,
            tri_count,
            voxel_size.x,
            voxel_size.y,
            voxel_size.z,
            grid_min,
            grid_max,
            counts.data());
        IQ_CUDA_CHECK(cudaGetLastError());

        DeviceBuffer<int64_t> offsets(tri_count);
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(counts.data()),
            thrust::device_pointer_cast(counts.data()) + tri_count,
            thrust::device_pointer_cast(offsets.data()));

        const int64_t last_count = copy_last_i64(counts.data(), tri_count, stream);
        const int64_t last_offset = copy_last_i64(offsets.data(), tri_count, stream);
        const int64_t raw_size = last_offset + last_count;
        if (raw_size == 0) continue;

        DeviceBuffer<uint64_t> partial_keys(raw_size);
        intersection_occ_emit_kernel<<<blocks, threads, 0, stream>>>(
            triangles,
            tri_begin,
            tri_count,
            voxel_size.x,
            voxel_size.y,
            voxel_size.z,
            grid_min,
            grid_max,
            offsets.data(),
            partial_keys.data());
        IQ_CUDA_CHECK(cudaGetLastError());

        thrust::sort(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(partial_keys.data()),
            thrust::device_pointer_cast(partial_keys.data()) + raw_size);

        auto partial_end = thrust::unique(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(partial_keys.data()),
            thrust::device_pointer_cast(partial_keys.data()) + raw_size);
        const int64_t partial_size = partial_end - thrust::device_pointer_cast(partial_keys.data());
        if (partial_size == 0) continue;

        chunks.push_back(make_occ_chunk_exact(partial_keys, partial_size, stream));
    }

    OccChunk final_chunk = final_merge_occ_chunks(std::move(chunks), stream);

    IntersectionOccResult out;
    out.size = final_chunk.size;
    out.voxels.allocate(out.size * 3);
    if (out.size == 0) return out;

    const int blocks = fdg_gpu::ceil_div_i64(out.size, threads);
    decode_occ_output_kernel<<<blocks, threads, 0, stream>>>(
        final_chunk.keys.data(), out.size, grid_min, grid_max, out.voxels.data());
    IQ_CUDA_CHECK(cudaGetLastError());
    return out;
}

IntersectQEFResult run_qef_impl(
    const float* triangles,
    int64_t num_triangles,
    float3 voxel_size,
    int3_ grid_min,
    int3_ grid_max,
    int64_t chunk_triangles,
    cudaStream_t stream) {
    if (num_triangles < 0) throw std::invalid_argument("num_triangles must be non-negative");
    if (chunk_triangles <= 0) throw std::invalid_argument("chunk_triangles must be positive");

    constexpr int threads = 256;
    std::vector<QEFChunk> chunks;
    chunks.reserve(static_cast<size_t>((num_triangles + chunk_triangles - 1) / chunk_triangles));

    for (int64_t tri_begin = 0; tri_begin < num_triangles; tri_begin += chunk_triangles) {
        const int64_t tri_count = std::min<int64_t>(chunk_triangles, num_triangles - tri_begin);
        if (tri_count == 0) continue;

        DeviceBuffer<int64_t> counts(tri_count);
        const int blocks = fdg_gpu::ceil_div_i64(tri_count, threads);
        intersection_count_kernel<<<blocks, threads, 0, stream>>>(
            triangles,
            tri_begin,
            tri_count,
            voxel_size.x,
            voxel_size.y,
            voxel_size.z,
            grid_min,
            grid_max,
            counts.data());
        IQ_CUDA_CHECK(cudaGetLastError());

        DeviceBuffer<int64_t> offsets(tri_count);
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(counts.data()),
            thrust::device_pointer_cast(counts.data()) + tri_count,
            thrust::device_pointer_cast(offsets.data()));

        const int64_t last_count = copy_last_i64(counts.data(), tri_count, stream);
        const int64_t last_offset = copy_last_i64(offsets.data(), tri_count, stream);
        const int64_t raw_size = last_offset + last_count;
        if (raw_size == 0) continue;

        DeviceBuffer<uint64_t> partial_keys(raw_size);
        DeviceBuffer<QEFEventValue> partial_values(raw_size);
        intersect_qef_emit_kernel<<<blocks, threads, 0, stream>>>(
            triangles,
            tri_begin,
            tri_count,
            voxel_size.x,
            voxel_size.y,
            voxel_size.z,
            grid_min,
            grid_max,
            offsets.data(),
            partial_keys.data(),
            partial_values.data());
        IQ_CUDA_CHECK(cudaGetLastError());

        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(partial_keys.data()),
            thrust::device_pointer_cast(partial_keys.data()) + raw_size,
            thrust::device_pointer_cast(partial_values.data()));

        DeviceBuffer<uint64_t> reduced_keys(raw_size);
        DeviceBuffer<QEFEventValue> reduced_values(raw_size);
        auto reduce_end = thrust::reduce_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(partial_keys.data()),
            thrust::device_pointer_cast(partial_keys.data()) + raw_size,
            thrust::device_pointer_cast(partial_values.data()),
            thrust::device_pointer_cast(reduced_keys.data()),
            thrust::device_pointer_cast(reduced_values.data()),
            thrust::equal_to<uint64_t>(),
            AddQEFEventValue());
        const int64_t reduced_size = reduce_end.first - thrust::device_pointer_cast(reduced_keys.data());
        if (reduced_size == 0) continue;

        chunks.push_back(make_qef_chunk_exact(reduced_keys, reduced_values, reduced_size, stream));
    }

    QEFChunk final_chunk = final_merge_qef_chunks(std::move(chunks), stream);

    IntersectQEFResult out;
    out.size = final_chunk.size;
    out.voxels.allocate(out.size * 3);
    out.mean_sum.allocate(out.size * 3);
    out.cnt.allocate(out.size);
    out.intersected.allocate(out.size);
    out.qefs.allocate(out.size);
    if (out.size == 0) return out;

    const int blocks = fdg_gpu::ceil_div_i64(out.size, threads);
    decode_qef_output_kernel<<<blocks, threads, 0, stream>>>(
        final_chunk.keys.data(),
        final_chunk.values.data(),
        out.size,
        grid_min,
        grid_max,
        out.voxels.data(),
        out.mean_sum.data(),
        out.cnt.data(),
        out.intersected.data(),
        out.qefs.data());
    IQ_CUDA_CHECK(cudaGetLastError());
    return out;
}

} // namespace

IntersectionOccResult intersection_occ_gpu(
    const float* triangles,
    int64_t num_triangles,
    float3 voxel_size,
    int3_ grid_min,
    int3_ grid_max,
    int64_t chunk_triangles,
    cudaStream_t stream) {
    if (triangles == nullptr && num_triangles > 0) throw std::invalid_argument("triangles is null");
    if (!(voxel_size.x > 0.0f && voxel_size.y > 0.0f && voxel_size.z > 0.0f)) {
        throw std::invalid_argument("voxel_size must be positive");
    }
    return run_occ_impl(triangles, num_triangles, voxel_size, grid_min, grid_max, chunk_triangles, stream);
}

IntersectQEFResult intersect_qef_gpu(
    const float* triangles,
    int64_t num_triangles,
    float3 voxel_size,
    int3_ grid_min,
    int3_ grid_max,
    int64_t chunk_triangles,
    cudaStream_t stream) {
    if (triangles == nullptr && num_triangles > 0) throw std::invalid_argument("triangles is null");
    if (!(voxel_size.x > 0.0f && voxel_size.y > 0.0f && voxel_size.z > 0.0f)) {
        throw std::invalid_argument("voxel_size must be positive");
    }
    return run_qef_impl(triangles, num_triangles, voxel_size, grid_min, grid_max, chunk_triangles, stream);
}

}  // namespace intersection_qef
