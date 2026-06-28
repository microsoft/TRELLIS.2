#include "voxelize_mesh_oct.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace voxelize_oct_impl {

#define VOX_CUDA_CHECK(expr)                                                        \
    do {                                                                            \
        cudaError_t _err = (expr);                                                  \
        if (_err != cudaSuccess) {                                                  \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(_err) +                     \
                                     " at " + __FILE__ + ":" +                  \
                                     std::to_string(__LINE__));                     \
        }                                                                           \
    } while (0)

constexpr int kRootNeighborCount = 27;
constexpr int kDefaultBlockSize = 128;

struct FaceDesc {
    float3 v0;
    float3 v1;
    float3 v2;

    float3 e0;
    float3 e1;
    float3 e2;

    float3 n_unit;

    float3 tri_bmin;
    float3 tri_bmax;
};

struct EdgeDesc {
    float3 p0;
    float3 p1;

    float3 seg;
    float seg_len;
    float3 dir_unit;

    float3 seg_bmin;
    float3 seg_bmax;
};

struct JobQueue {
    int32_t* prim_id = nullptr;
    uint8_t* level = nullptr;
    int32_t* i = nullptr;
    int32_t* j = nullptr;
    int32_t* k = nullptr;
    int64_t size = 0;
    int64_t capacity = 0;
};

struct RoundBuffers {
    uint8_t* job_hit = nullptr;
    int32_t* child_count = nullptr;
    int32_t* result_count = nullptr;
    int32_t* child_offset = nullptr;
    int32_t* result_offset = nullptr;

    void* cub_temp_storage = nullptr;
    size_t cub_temp_bytes = 0;
    int64_t capacity = 0;
};

struct ResultBuffer {
    int32_t* prim_id = nullptr;
    int32_t* vi = nullptr;
    int32_t* vj = nullptr;
    int32_t* vk = nullptr;
    int64_t size = 0;
};

struct DeviceResult {
    int32_t* prim_id = nullptr;
    int32_t* voxel_i = nullptr;
    int32_t* voxel_j = nullptr;
    int32_t* voxel_k = nullptr;
    int64_t size = 0;
};

struct VoxelizeWorkspace {
    int32_t* leaf_ix = nullptr;
    int32_t* leaf_iy = nullptr;
    int32_t* leaf_iz = nullptr;

    JobQueue queue_a;
    JobQueue queue_b;
    RoundBuffers round;
    std::vector<ResultBuffer> result_rounds;
};

inline int ceil_div_i64(int64_t n, int block) {
    return static_cast<int>((n + block - 1) / block);
}

inline int max3_int(int a, int b, int c) {
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

inline int ceil_log2_pos_int(int x) {
    int d = 0;
    int v = 1;
    while (v < x) {
        v <<= 1;
        ++d;
    }
    return d;
}

inline int compute_grid_depth_from_grid_size(int3 grid_size) {
    const int max_dim = max3_int(grid_size.x, grid_size.y, grid_size.z);
    return ceil_log2_pos_int(max_dim);
}

inline float3 reciprocal_voxel_size(float3 voxel_size) {
    return make_float3(1.0f / voxel_size.x, 1.0f / voxel_size.y, 1.0f / voxel_size.z);
}

inline void free_ptr(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

inline void alloc_i32(int32_t** ptr, int64_t n) {
    *ptr = nullptr;
    if (n > 0) {
        VOX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(int32_t) * n));
    }
}

inline void alloc_u8(uint8_t** ptr, int64_t n) {
    *ptr = nullptr;
    if (n > 0) {
        VOX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(uint8_t) * n));
    }
}

inline void alloc_face_desc(FaceDesc** ptr, int64_t n) {
    *ptr = nullptr;
    if (n > 0) {
        VOX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(FaceDesc) * n));
    }
}

inline void alloc_edge_desc(EdgeDesc** ptr, int64_t n) {
    *ptr = nullptr;
    if (n > 0) {
        VOX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(EdgeDesc) * n));
    }
}

inline void release_job_queue(JobQueue& q) {
    free_ptr(q.prim_id);
    free_ptr(q.level);
    free_ptr(q.i);
    free_ptr(q.j);
    free_ptr(q.k);
    q = {};
}

inline void release_round_buffers(RoundBuffers& b) {
    free_ptr(b.job_hit);
    free_ptr(b.child_count);
    free_ptr(b.result_count);
    free_ptr(b.child_offset);
    free_ptr(b.result_offset);
    free_ptr(b.cub_temp_storage);
    b = {};
}

inline void release_result_buffer(ResultBuffer& r) {
    free_ptr(r.prim_id);
    free_ptr(r.vi);
    free_ptr(r.vj);
    free_ptr(r.vk);
    r = {};
}

inline void release_workspace(VoxelizeWorkspace& ws) {
    free_ptr(ws.leaf_ix);
    free_ptr(ws.leaf_iy);
    free_ptr(ws.leaf_iz);
    release_job_queue(ws.queue_a);
    release_job_queue(ws.queue_b);
    release_round_buffers(ws.round);
    for (auto& r : ws.result_rounds) {
        release_result_buffer(r);
    }
    ws.result_rounds.clear();
}

inline void ensure_job_queue_capacity(JobQueue& q, int64_t capacity) {
    if (capacity <= q.capacity) {
        return;
    }
    release_job_queue(q);
    alloc_i32(&q.prim_id, capacity);
    alloc_u8(&q.level, capacity);
    alloc_i32(&q.i, capacity);
    alloc_i32(&q.j, capacity);
    alloc_i32(&q.k, capacity);
    q.capacity = capacity;
    q.size = 0;
}

inline void ensure_round_capacity(RoundBuffers& b, int64_t capacity) {
    if (capacity <= b.capacity) {
        return;
    }

    free_ptr(b.job_hit);
    free_ptr(b.child_count);
    free_ptr(b.result_count);
    free_ptr(b.child_offset);
    free_ptr(b.result_offset);

    alloc_u8(&b.job_hit, capacity);
    alloc_i32(&b.child_count, capacity);
    alloc_i32(&b.result_count, capacity);
    alloc_i32(&b.child_offset, capacity);
    alloc_i32(&b.result_offset, capacity);
    b.capacity = capacity;
}

inline void ensure_scan_temp_storage(
    RoundBuffers& b,
    int32_t* d_in,
    int32_t* d_out,
    int64_t count,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    if (count > INT32_MAX) {
        throw std::runtime_error("CUB scan count exceeds int32 range");
    }
    size_t bytes = 0;
    VOX_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr,
        bytes,
        d_in,
        d_out,
        static_cast<int>(count),
        stream));
    if (bytes > b.cub_temp_bytes) {
        free_ptr(b.cub_temp_storage);
        VOX_CUDA_CHECK(cudaMalloc(&b.cub_temp_storage, bytes));
        b.cub_temp_bytes = bytes;
    }
}

inline void exclusive_scan_i32(
    RoundBuffers& b,
    int32_t* d_in,
    int32_t* d_out,
    int64_t count,
    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    ensure_scan_temp_storage(b, d_in, d_out, count, stream);
    VOX_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        b.cub_temp_storage,
        b.cub_temp_bytes,
        d_in,
        d_out,
        static_cast<int>(count),
        stream));
}

inline int32_t copy_last_i32(const int32_t* ptr, int64_t count, cudaStream_t stream) {
    if (count <= 0) {
        return 0;
    }
    int32_t value = 0;
    VOX_CUDA_CHECK(cudaMemcpyAsync(
        &value,
        ptr + (count - 1),
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream));
    VOX_CUDA_CHECK(cudaStreamSynchronize(stream));
    return value;
}

inline ResultBuffer make_result_buffer(int64_t count) {
    ResultBuffer r;
    if (count <= 0) {
        return r;
    }
    alloc_i32(&r.prim_id, count);
    alloc_i32(&r.vi, count);
    alloc_i32(&r.vj, count);
    alloc_i32(&r.vk, count);
    r.size = count;
    return r;
}

inline DeviceResult gather_result_rounds(
    const std::vector<ResultBuffer>& rounds,
    cudaStream_t stream) {
    DeviceResult out;
    int64_t total = 0;
    for (const auto& r : rounds) {
        total += r.size;
    }
    out.size = total;
    if (total == 0) {
        return out;
    }

    alloc_i32(&out.prim_id, total);
    alloc_i32(&out.voxel_i, total);
    alloc_i32(&out.voxel_j, total);
    alloc_i32(&out.voxel_k, total);

    int64_t cursor = 0;
    for (const auto& r : rounds) {
        if (r.size == 0) {
            continue;
        }
        VOX_CUDA_CHECK(cudaMemcpyAsync(
            out.prim_id + cursor,
            r.prim_id,
            sizeof(int32_t) * r.size,
            cudaMemcpyDeviceToDevice,
            stream));
        VOX_CUDA_CHECK(cudaMemcpyAsync(
            out.voxel_i + cursor,
            r.vi,
            sizeof(int32_t) * r.size,
            cudaMemcpyDeviceToDevice,
            stream));
        VOX_CUDA_CHECK(cudaMemcpyAsync(
            out.voxel_j + cursor,
            r.vj,
            sizeof(int32_t) * r.size,
            cudaMemcpyDeviceToDevice,
            stream));
        VOX_CUDA_CHECK(cudaMemcpyAsync(
            out.voxel_k + cursor,
            r.vk,
            sizeof(int32_t) * r.size,
            cudaMemcpyDeviceToDevice,
            stream));
        cursor += r.size;
    }

    VOX_CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
}

__device__ inline float3 add3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 mul3(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float3 mul3_comp(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float2 dot2_pair(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

__device__ inline float dot2(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}

__device__ inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ inline float3 min3(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ inline float3 max3(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ inline float3 normalize3(const float3& a) {
    const float z = dot3(a, a);
    if (z > 0.0f) {
        const float n = sqrtf(z);
        return make_float3(a.x / n, a.y / n, a.z / n);
    } else {
        return a;
    }
}

__device__ inline bool bbox_overlap_closed(
    const float3& a_min,
    const float3& a_max,
    const float3& b_min,
    const float3& b_max) {
    return !(a_max.x < b_min.x || b_max.x < a_min.x ||
             a_max.y < b_min.y || b_max.y < a_min.y ||
             a_max.z < b_min.z || b_max.z < a_min.z);
}

__device__ inline float2 max2_zero(const float2& a) {
    return make_float2(fmaxf(a.x, 0.0f), fmaxf(a.y, 0.0f));
}

__device__ inline void compute_face_min_level_and_root(
    int ix0, int iy0, int iz0,
    int ix1, int iy1, int iz1,
    int ix2, int iy2, int iz2,
    int d,
    uint8_t& level,
    int& root_i,
    int& root_j,
    int& root_k) {
    uint32_t diff =
        static_cast<uint32_t>(ix0 ^ ix1) | static_cast<uint32_t>(ix0 ^ ix2) |
        static_cast<uint32_t>(iy0 ^ iy1) | static_cast<uint32_t>(iy0 ^ iy2) |
        static_cast<uint32_t>(iz0 ^ iz1) | static_cast<uint32_t>(iz0 ^ iz2);

    if (diff == 0) {
        level = static_cast<uint8_t>(d);
        root_i = ix0;
        root_j = iy0;
        root_k = iz0;
        return;
    }

    int msb = 31 - __clz(diff);
    int l = d - 1 - msb;
    level = static_cast<uint8_t>(l);

    int shift = d - l;
    root_i = ix0 >> shift;
    root_j = iy0 >> shift;
    root_k = iz0 >> shift;
}

__device__ inline void compute_edge_min_level_and_root(
    int ix0, int iy0, int iz0,
    int ix1, int iy1, int iz1,
    int d,
    uint8_t& level,
    int& root_i,
    int& root_j,
    int& root_k) {
    uint32_t diff =
        static_cast<uint32_t>(ix0 ^ ix1) |
        static_cast<uint32_t>(iy0 ^ iy1) |
        static_cast<uint32_t>(iz0 ^ iz1);

    if (diff == 0) {
        level = static_cast<uint8_t>(d);
        root_i = ix0;
        root_j = iy0;
        root_k = iz0;
        return;
    }

    int msb = 31 - __clz(diff);
    int l = d - 1 - msb;
    level = static_cast<uint8_t>(l);

    int shift = d - l;
    root_i = ix0 >> shift;
    root_j = iy0 >> shift;
    root_k = iz0 >> shift;
}

__device__ inline bool node_intersects_valid_domain(
    int d,
    int level,
    int i,
    int j,
    int k,
    int3 grid_size) {
    int cells = 1 << level;
    if (i < 0 || i >= cells || j < 0 || j >= cells || k < 0 || k >= cells) {
        return false;
    }

    int node_span = 1 << (d - level);
    int x0 = i * node_span;
    int y0 = j * node_span;
    int z0 = k * node_span;

    return (x0 < grid_size.x) && (y0 < grid_size.y) && (z0 < grid_size.z);
}

__device__ inline void compute_node_box_world(
    int d,
    int level,
    int i,
    int j,
    int k,
    int3 grid_min,
    float3 voxel_size,
    float3& box_min,
    float3& box_size,
    float3& box_max) {
    int node_span = 1 << (d - level);

    int global_base_x = grid_min.x + i * node_span;
    int global_base_y = grid_min.y + j * node_span;
    int global_base_z = grid_min.z + k * node_span;

    int global_end_x = global_base_x + node_span;
    int global_end_y = global_base_y + node_span;
    int global_end_z = global_base_z + node_span;

    box_min = make_float3(
        static_cast<float>(global_base_x) * voxel_size.x,
        static_cast<float>(global_base_y) * voxel_size.y,
        static_cast<float>(global_base_z) * voxel_size.z);
    box_max = make_float3(
        static_cast<float>(global_end_x) * voxel_size.x,
        static_cast<float>(global_end_y) * voxel_size.y,
        static_cast<float>(global_end_z) * voxel_size.z);
    box_size = sub3(box_max, box_min);
}

__device__ inline bool face_qef_style_triangle_box_hit(
    const FaceDesc& f,
    const float3& box_min,
    const float3& box_size,
    const float3& box_max) {
    if (!bbox_overlap_closed(f.tri_bmin, f.tri_bmax, box_min, box_max)) {
        return false;
    }

    const float3& n = f.n_unit;

    float3 c = make_float3(
        n.x > 0.0f ? box_size.x : 0.0f,
        n.y > 0.0f ? box_size.y : 0.0f,
        n.z > 0.0f ? box_size.z : 0.0f);

    float d1 = dot3(n, sub3(c, f.v0));
    float d2 = dot3(n, sub3(sub3(box_size, c), f.v0));

    int mul_xy = (n.z < 0.0f) ? -1 : 1;
    float2 n_xy_e0 = make_float2(-mul_xy * f.e0.y, mul_xy * f.e0.x);
    float2 n_xy_e1 = make_float2(-mul_xy * f.e1.y, mul_xy * f.e1.x);
    float2 n_xy_e2 = make_float2(-mul_xy * f.e2.y, mul_xy * f.e2.x);

    float d_xy_e0 = -dot2(n_xy_e0, make_float2(f.v0.x, f.v0.y)) +
                    dot2(max2_zero(n_xy_e0), make_float2(box_size.x, box_size.y));
    float d_xy_e1 = -dot2(n_xy_e1, make_float2(f.v1.x, f.v1.y)) +
                    dot2(max2_zero(n_xy_e1), make_float2(box_size.x, box_size.y));
    float d_xy_e2 = -dot2(n_xy_e2, make_float2(f.v2.x, f.v2.y)) +
                    dot2(max2_zero(n_xy_e2), make_float2(box_size.x, box_size.y));

    int mul_yz = (n.x < 0.0f) ? -1 : 1;
    float2 n_yz_e0 = make_float2(-mul_yz * f.e0.z, mul_yz * f.e0.y);
    float2 n_yz_e1 = make_float2(-mul_yz * f.e1.z, mul_yz * f.e1.y);
    float2 n_yz_e2 = make_float2(-mul_yz * f.e2.z, mul_yz * f.e2.y);

    float d_yz_e0 = -dot2(n_yz_e0, make_float2(f.v0.y, f.v0.z)) +
                    dot2(max2_zero(n_yz_e0), make_float2(box_size.y, box_size.z));
    float d_yz_e1 = -dot2(n_yz_e1, make_float2(f.v1.y, f.v1.z)) +
                    dot2(max2_zero(n_yz_e1), make_float2(box_size.y, box_size.z));
    float d_yz_e2 = -dot2(n_yz_e2, make_float2(f.v2.y, f.v2.z)) +
                    dot2(max2_zero(n_yz_e2), make_float2(box_size.y, box_size.z));

    int mul_zx = (n.y < 0.0f) ? -1 : 1;
    float2 n_zx_e0 = make_float2(-mul_zx * f.e0.x, mul_zx * f.e0.z);
    float2 n_zx_e1 = make_float2(-mul_zx * f.e1.x, mul_zx * f.e1.z);
    float2 n_zx_e2 = make_float2(-mul_zx * f.e2.x, mul_zx * f.e2.z);

    float d_zx_e0 = -dot2(n_zx_e0, make_float2(f.v0.z, f.v0.x)) +
                    dot2(max2_zero(n_zx_e0), make_float2(box_size.z, box_size.x));
    float d_zx_e1 = -dot2(n_zx_e1, make_float2(f.v1.z, f.v1.x)) +
                    dot2(max2_zero(n_zx_e1), make_float2(box_size.z, box_size.x));
    float d_zx_e2 = -dot2(n_zx_e2, make_float2(f.v2.z, f.v2.x)) +
                    dot2(max2_zero(n_zx_e2), make_float2(box_size.z, box_size.x));

    float n_dot_p = dot3(n, box_min);
    if (((n_dot_p + d1) * (n_dot_p + d2)) > 0.0f) {
        return false;
    }

    float2 p_xy = make_float2(box_min.x, box_min.y);
    if (dot2(n_xy_e0, p_xy) + d_xy_e0 < 0.0f) return false;
    if (dot2(n_xy_e1, p_xy) + d_xy_e1 < 0.0f) return false;
    if (dot2(n_xy_e2, p_xy) + d_xy_e2 < 0.0f) return false;

    float2 p_yz = make_float2(box_min.y, box_min.z);
    if (dot2(n_yz_e0, p_yz) + d_yz_e0 < 0.0f) return false;
    if (dot2(n_yz_e1, p_yz) + d_yz_e1 < 0.0f) return false;
    if (dot2(n_yz_e2, p_yz) + d_yz_e2 < 0.0f) return false;

    float2 p_zx = make_float2(box_min.z, box_min.x);
    if (dot2(n_zx_e0, p_zx) + d_zx_e0 < 0.0f) return false;
    if (dot2(n_zx_e1, p_zx) + d_zx_e1 < 0.0f) return false;
    if (dot2(n_zx_e2, p_zx) + d_zx_e2 < 0.0f) return false;

    return true;
}

__device__ inline bool segment_box_overlap_world(
    const EdgeDesc& e,
    const float3& box_min,
    const float3& box_max) {
    if (e.seg_len < 1.0e-6f) {
        return false;
    }

    if (!bbox_overlap_closed(e.seg_bmin, e.seg_bmax, box_min, box_max)) {
        return false;
    }

    float tmin = 0.0f;
    float tmax = 1.0f;

    if (e.seg.x == 0.0f) {
        if (!(box_min.x <= e.p0.x && e.p0.x <= box_max.x)) return false;
    } else {
        float inv_d = 1.0f / e.seg.x;
        float t1 = (box_min.x - e.p0.x) * inv_d;
        float t2 = (box_max.x - e.p0.x) * inv_d;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax) return false;
    }

    if (e.seg.y == 0.0f) {
        if (!(box_min.y <= e.p0.y && e.p0.y <= box_max.y)) return false;
    } else {
        float inv_d = 1.0f / e.seg.y;
        float t1 = (box_min.y - e.p0.y) * inv_d;
        float t2 = (box_max.y - e.p0.y) * inv_d;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax) return false;
    }

    if (e.seg.z == 0.0f) {
        if (!(box_min.z <= e.p0.z && e.p0.z <= box_max.z)) return false;
    } else {
        float inv_d = 1.0f / e.seg.z;
        float t1 = (box_min.z - e.p0.z) * inv_d;
        float t2 = (box_max.z - e.p0.z) * inv_d;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax) return false;
    }

    return true;
}

__global__ void kernel_build_leaf_coords(
    const float* __restrict__ vertices,
    int64_t num_vertices,
    float3 inv_voxel_size,
    int3 grid_min,
    int3 grid_size,
    int32_t* __restrict__ leaf_ix,
    int32_t* __restrict__ leaf_iy,
    int32_t* __restrict__ leaf_iz) {
    int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vid >= num_vertices) {
        return;
    }

    float x = vertices[3 * vid + 0] * inv_voxel_size.x;
    float y = vertices[3 * vid + 1] * inv_voxel_size.y;
    float z = vertices[3 * vid + 2] * inv_voxel_size.z;

    int ix = static_cast<int>(floorf(x)) - grid_min.x;
    int iy = static_cast<int>(floorf(y)) - grid_min.y;
    int iz = static_cast<int>(floorf(z)) - grid_min.z;

    ix = max(0, min(ix, grid_size.x - 1));
    iy = max(0, min(iy, grid_size.y - 1));
    iz = max(0, min(iz, grid_size.z - 1));

    leaf_ix[vid] = ix;
    leaf_iy[vid] = iy;
    leaf_iz[vid] = iz;
}

__global__ void kernel_init_faces_and_emit_root27(
    const float* __restrict__ vertices,
    const int32_t* __restrict__ faces,
    const int32_t* __restrict__ leaf_ix,
    const int32_t* __restrict__ leaf_iy,
    const int32_t* __restrict__ leaf_iz,
    int64_t num_faces,
    int d,
    FaceDesc* __restrict__ face_desc,
    JobQueue out_q) {
    int64_t fid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (fid >= num_faces) {
        return;
    }

    int v0_id = faces[3 * fid + 0];
    int v1_id = faces[3 * fid + 1];
    int v2_id = faces[3 * fid + 2];

    float3 v0 = make_float3(vertices[3 * v0_id + 0], vertices[3 * v0_id + 1], vertices[3 * v0_id + 2]);
    float3 v1 = make_float3(vertices[3 * v1_id + 0], vertices[3 * v1_id + 1], vertices[3 * v1_id + 2]);
    float3 v2 = make_float3(vertices[3 * v2_id + 0], vertices[3 * v2_id + 1], vertices[3 * v2_id + 2]);

    FaceDesc fd;
    fd.v0 = v0;
    fd.v1 = v1;
    fd.v2 = v2;
    fd.e0 = sub3(v1, v0);
    fd.e1 = sub3(v2, v1);
    fd.e2 = sub3(v0, v2);
    fd.n_unit = normalize3(cross3(fd.e0, fd.e1));
    fd.tri_bmin = min3(v0, min3(v1, v2));
    fd.tri_bmax = max3(v0, max3(v1, v2));
    face_desc[fid] = fd;

    uint8_t level;
    int root_i, root_j, root_k;
    compute_face_min_level_and_root(
        leaf_ix[v0_id], leaf_iy[v0_id], leaf_iz[v0_id],
        leaf_ix[v1_id], leaf_iy[v1_id], leaf_iz[v1_id],
        leaf_ix[v2_id], leaf_iy[v2_id], leaf_iz[v2_id],
        d,
        level,
        root_i,
        root_j,
        root_k);

    int64_t base = static_cast<int64_t>(kRootNeighborCount) * fid;
    int slot = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int64_t out = base + slot++;
                out_q.prim_id[out] = static_cast<int32_t>(fid);
                out_q.level[out] = level;
                out_q.i[out] = root_i + dx;
                out_q.j[out] = root_j + dy;
                out_q.k[out] = root_k + dz;
            }
        }
    }
}

__global__ void kernel_init_edges_and_emit_root27(
    const float* __restrict__ vertices,
    const int32_t* __restrict__ edges,
    const int32_t* __restrict__ leaf_ix,
    const int32_t* __restrict__ leaf_iy,
    const int32_t* __restrict__ leaf_iz,
    int64_t num_edges,
    int d,
    EdgeDesc* __restrict__ edge_desc,
    JobQueue out_q) {
    int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (eid >= num_edges) {
        return;
    }

    int v0_id = edges[2 * eid + 0];
    int v1_id = edges[2 * eid + 1];

    float3 p0 = make_float3(vertices[3 * v0_id + 0], vertices[3 * v0_id + 1], vertices[3 * v0_id + 2]);
    float3 p1 = make_float3(vertices[3 * v1_id + 0], vertices[3 * v1_id + 1], vertices[3 * v1_id + 2]);

    EdgeDesc ed;
    ed.p0 = p0;
    ed.p1 = p1;
    ed.seg = sub3(p1, p0);
    ed.seg_len = sqrtf(dot3(ed.seg, ed.seg));
    ed.dir_unit = (ed.seg_len >= 1.0e-6f) ? mul3(ed.seg, 1.0f / ed.seg_len) : make_float3(0.0f, 0.0f, 0.0f);
    ed.seg_bmin = min3(p0, p1);
    ed.seg_bmax = max3(p0, p1);
    edge_desc[eid] = ed;

    uint8_t level;
    int root_i, root_j, root_k;
    compute_edge_min_level_and_root(
        leaf_ix[v0_id], leaf_iy[v0_id], leaf_iz[v0_id],
        leaf_ix[v1_id], leaf_iy[v1_id], leaf_iz[v1_id],
        d,
        level,
        root_i,
        root_j,
        root_k);

    int64_t base = static_cast<int64_t>(kRootNeighborCount) * eid;
    int slot = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int64_t out = base + slot++;
                out_q.prim_id[out] = static_cast<int32_t>(eid);
                out_q.level[out] = level;
                out_q.i[out] = root_i + dx;
                out_q.j[out] = root_j + dy;
                out_q.k[out] = root_k + dz;
            }
        }
    }
}

__global__ void kernel_count_face_jobs(
    JobQueue curr_q,
    const FaceDesc* __restrict__ face_desc,
    int d,
    int3 grid_min,
    int3 grid_size,
    float3 voxel_size,
    uint8_t* __restrict__ job_hit,
    int32_t* __restrict__ child_count,
    int32_t* __restrict__ result_count) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= curr_q.size) {
        return;
    }

    int fid = curr_q.prim_id[idx];
    int level = static_cast<int>(curr_q.level[idx]);
    int i = curr_q.i[idx];
    int j = curr_q.j[idx];
    int k = curr_q.k[idx];

    if (!node_intersects_valid_domain(d, level, i, j, k, grid_size)) {
        job_hit[idx] = 0;
        child_count[idx] = 0;
        result_count[idx] = 0;
        return;
    }

    float3 box_min, box_size, box_max;
    compute_node_box_world(d, level, i, j, k, grid_min, voxel_size, box_min, box_size, box_max);

    bool hit = face_qef_style_triangle_box_hit(face_desc[fid], box_min, box_size, box_max);
    job_hit[idx] = static_cast<uint8_t>(hit ? 1 : 0);

    if (!hit) {
        child_count[idx] = 0;
        result_count[idx] = 0;
    } else if (level < d) {
        child_count[idx] = 8;
        result_count[idx] = 0;
    } else {
        child_count[idx] = 0;
        result_count[idx] = 1;
    }
}

__global__ void kernel_count_edge_jobs(
    JobQueue curr_q,
    const EdgeDesc* __restrict__ edge_desc,
    int d,
    int3 grid_min,
    int3 grid_size,
    float3 voxel_size,
    uint8_t* __restrict__ job_hit,
    int32_t* __restrict__ child_count,
    int32_t* __restrict__ result_count) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= curr_q.size) {
        return;
    }

    int eid = curr_q.prim_id[idx];
    int level = static_cast<int>(curr_q.level[idx]);
    int i = curr_q.i[idx];
    int j = curr_q.j[idx];
    int k = curr_q.k[idx];

    if (!node_intersects_valid_domain(d, level, i, j, k, grid_size)) {
        job_hit[idx] = 0;
        child_count[idx] = 0;
        result_count[idx] = 0;
        return;
    }

    float3 box_min, box_size, box_max;
    compute_node_box_world(d, level, i, j, k, grid_min, voxel_size, box_min, box_size, box_max);

    bool hit = segment_box_overlap_world(edge_desc[eid], box_min, box_max);
    job_hit[idx] = static_cast<uint8_t>(hit ? 1 : 0);

    if (!hit) {
        child_count[idx] = 0;
        result_count[idx] = 0;
    } else if (level < d) {
        child_count[idx] = 8;
        result_count[idx] = 0;
    } else {
        child_count[idx] = 0;
        result_count[idx] = 1;
    }
}

__global__ void kernel_emit_jobs(
    JobQueue curr_q,
    const uint8_t* __restrict__ job_hit,
    const int32_t* __restrict__ child_offset,
    const int32_t* __restrict__ result_offset,
    int d,
    JobQueue next_q,
    ResultBuffer out_res) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= curr_q.size) {
        return;
    }
    if (job_hit[idx] == 0) {
        return;
    }

    int prim_id = curr_q.prim_id[idx];
    int level = static_cast<int>(curr_q.level[idx]);
    int i = curr_q.i[idx];
    int j = curr_q.j[idx];
    int k = curr_q.k[idx];

    if (level < d) {
        int32_t base = child_offset[idx];
        int child_level = level + 1;
        int slot = 0;
        for (int bz = 0; bz < 2; ++bz) {
            for (int by = 0; by < 2; ++by) {
                for (int bx = 0; bx < 2; ++bx) {
                    int32_t out = base + slot++;
                    next_q.prim_id[out] = prim_id;
                    next_q.level[out] = static_cast<uint8_t>(child_level);
                    next_q.i[out] = 2 * i + bx;
                    next_q.j[out] = 2 * j + by;
                    next_q.k[out] = 2 * k + bz;
                }
            }
        }
    } else {
        int32_t out = result_offset[idx];
        out_res.prim_id[out] = prim_id;
        out_res.vi[out] = i;
        out_res.vj[out] = j;
        out_res.vk[out] = k;
    }
}

inline void reset_output(DeviceResult& out) {
    out.prim_id = nullptr;
    out.voxel_i = nullptr;
    out.voxel_j = nullptr;
    out.voxel_k = nullptr;
    out.size = 0;
}

inline void release_device_result(DeviceResult& out) {
    free_ptr(out.prim_id);
    free_ptr(out.voxel_i);
    free_ptr(out.voxel_j);
    free_ptr(out.voxel_k);
    out = {};
}

}  // namespace voxelize_oct_impl


namespace {

inline fdg_gpu::PrimitivePairResult to_primitive_pair(voxelize_oct_impl::DeviceResult&& r) {
    fdg_gpu::PrimitivePairResult out;
    out.size = r.size;
    out.prim_id.adopt(r.prim_id, r.size);
    out.voxel_i.adopt(r.voxel_i, r.size);
    out.voxel_j.adopt(r.voxel_j, r.size);
    out.voxel_k.adopt(r.voxel_k, r.size);
    r.prim_id = nullptr;
    r.voxel_i = nullptr;
    r.voxel_j = nullptr;
    r.voxel_k = nullptr;
    r.size = 0;
    return out;
}

struct SurfaceLookup {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<uint64_t> keys_sorted;
    fdg_gpu::DeviceBuffer<int32_t> ids_sorted;
};

struct FacePairKeys {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<uint64_t> keys;
};

struct FaceContribStream {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<int32_t> voxel_id;
    fdg_gpu::DeviceBuffer<fdg_gpu::SymQEF10> qef;
};

__host__ __device__ inline fdg_gpu::SymQEF10 symqef10_zero() {
    return fdg_gpu::SymQEF10{0,0,0,0,0,0,0,0,0,0};
}

struct SymQEF10Add {
    __host__ __device__ fdg_gpu::SymQEF10 operator()(const fdg_gpu::SymQEF10& a, const fdg_gpu::SymQEF10& b) const {
        return fdg_gpu::SymQEF10{
            a.q00 + b.q00, a.q01 + b.q01, a.q02 + b.q02, a.q03 + b.q03,
            a.q11 + b.q11, a.q12 + b.q12, a.q13 + b.q13,
            a.q22 + b.q22, a.q23 + b.q23,
            a.q33 + b.q33};
    }
};

__host__ __device__ inline uint64_t pack_pair_key(int32_t voxel_id, int32_t face_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(voxel_id)) << 32) |
           static_cast<uint32_t>(face_id);
}

__host__ __device__ inline int32_t unpack_pair_voxel_id(uint64_t k) {
    return static_cast<int32_t>(k >> 32);
}

__host__ __device__ inline int32_t unpack_pair_face_id(uint64_t k) {
    return static_cast<int32_t>(k & 0xffffffffu);
}

__host__ __device__ inline fdg_gpu::SymQEF10 symqef10_from_plane(float4 p) {
    const float a = p.x, b = p.y, c = p.z, d = p.w;
    return fdg_gpu::SymQEF10{
        a*a, a*b, a*c, a*d,
        b*b, b*c, b*d,
        c*c, c*d,
        d*d
    };
}

__global__ void build_synth_faces_kernel(int64_t num_triangles, int32_t* __restrict__ faces) {
    int64_t fid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (fid >= num_triangles) return;
    faces[3 * fid + 0] = static_cast<int32_t>(3 * fid + 0);
    faces[3 * fid + 1] = static_cast<int32_t>(3 * fid + 1);
    faces[3 * fid + 2] = static_cast<int32_t>(3 * fid + 2);
}

__global__ void build_surface_keys_kernel(
    const int* __restrict__ voxels,
    int64_t num_voxels,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ ids) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_voxels) return;
    int x = voxels[3 * i + 0];
    int y = voxels[3 * i + 1];
    int z = voxels[3 * i + 2];
    keys[i] = fdg_gpu::pack_voxel_key(x, y, z, grid_min, grid_max);
    ids[i] = static_cast<int32_t>(i);
}

__global__ void build_raw_pair_keys_kernel(
    const int32_t* __restrict__ voxel_i,
    const int32_t* __restrict__ voxel_j,
    const int32_t* __restrict__ voxel_k,
    const int32_t* __restrict__ face_id,
    int64_t num_pairs,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    uint64_t* __restrict__ voxel_keys,
    int32_t* __restrict__ pair_face_ids) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;
    const int32_t gx = voxel_i[i] + grid_min.x;
    const int32_t gy = voxel_j[i] + grid_min.y;
    const int32_t gz = voxel_k[i] + grid_min.z;
    voxel_keys[i] = fdg_gpu::pack_voxel_key(gx, gy, gz, grid_min, grid_max);
    pair_face_ids[i] = face_id[i];
}

__device__ inline int lower_bound_u64(const uint64_t* arr, int64_t n, uint64_t key) {
    int64_t lo = 0;
    int64_t hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        uint64_t v = arr[mid];
        if (v < key) lo = mid + 1;
        else hi = mid;
    }
    return static_cast<int>(lo);
}

__global__ void map_pair_to_voxel_id_kernel(
    const uint64_t* __restrict__ pair_keys,
    const int32_t* __restrict__ pair_face_ids,
    int64_t num_pairs,
    const uint64_t* __restrict__ surface_keys_sorted,
    const int32_t* __restrict__ surface_ids_sorted,
    int64_t num_voxels,
    int32_t* __restrict__ mapped_voxel_id,
    int32_t* __restrict__ mapped_face_id,
    int32_t* __restrict__ valid) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;
    uint64_t key = pair_keys[i];
    int pos = lower_bound_u64(surface_keys_sorted, num_voxels, key);
    if (pos < num_voxels && surface_keys_sorted[pos] == key) {
        mapped_voxel_id[i] = surface_ids_sorted[pos];
        mapped_face_id[i] = pair_face_ids[i];
        valid[i] = 1;
    } else {
        mapped_voxel_id[i] = -1;
        mapped_face_id[i] = -1;
        valid[i] = 0;
    }
}

__global__ void compact_valid_pairs_kernel(
    const int32_t* __restrict__ mapped_voxel_id,
    const int32_t* __restrict__ mapped_face_id,
    const int32_t* __restrict__ valid,
    const int32_t* __restrict__ offsets,
    int64_t num_pairs,
    uint64_t* __restrict__ pair_keys_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs || valid[i] == 0) return;
    int32_t out = offsets[i];
    pair_keys_out[out] = pack_pair_key(mapped_voxel_id[i], mapped_face_id[i]);
}

__global__ void build_face_qef_contrib_kernel(
    const uint64_t* __restrict__ pair_keys,
    int64_t num_pairs,
    const float* __restrict__ vertices,
    const int32_t* __restrict__ faces,
    int32_t* __restrict__ voxel_id_out,
    fdg_gpu::SymQEF10* __restrict__ qef_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;

    int32_t voxel_id = unpack_pair_voxel_id(pair_keys[i]);
    int32_t fid = unpack_pair_face_id(pair_keys[i]);

    int32_t i0 = faces[3 * fid + 0];
    int32_t i1 = faces[3 * fid + 1];
    int32_t i2 = faces[3 * fid + 2];

    float3 v0 = make_float3(vertices[3 * i0 + 0], vertices[3 * i0 + 1], vertices[3 * i0 + 2]);
    float3 v1 = make_float3(vertices[3 * i1 + 0], vertices[3 * i1 + 1], vertices[3 * i1 + 2]);
    float3 v2 = make_float3(vertices[3 * i2 + 0], vertices[3 * i2 + 1], vertices[3 * i2 + 2]);

    float3 e0 = voxelize_oct_impl::sub3(v1, v0);
    float3 e1 = voxelize_oct_impl::sub3(v2, v1);
    float3 n = voxelize_oct_impl::normalize3(voxelize_oct_impl::cross3(e0, e1));
    float4 plane = make_float4(n.x, n.y, n.z, -voxelize_oct_impl::dot3(n, v0));

    voxel_id_out[i] = voxel_id;
    qef_out[i] = symqef10_from_plane(plane);
}

__global__ void scatter_reduced_face_qef_kernel(
    const int32_t* __restrict__ reduced_voxel_id,
    const fdg_gpu::SymQEF10* __restrict__ reduced_qef,
    int64_t num_reduced,
    fdg_gpu::SymQEF10* __restrict__ full_qefs) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_reduced) return;
    full_qefs[reduced_voxel_id[i]] = reduced_qef[i];
}

inline SurfaceLookup build_surface_lookup(
    const int* voxels,
    int64_t num_voxels,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    cudaStream_t stream) {
    SurfaceLookup out;
    out.size = num_voxels;
    out.keys_sorted.allocate(num_voxels);
    out.ids_sorted.allocate(num_voxels);
    constexpr int kBlock = 256;
    build_surface_keys_kernel<<<fdg_gpu::ceil_div_i64(num_voxels, kBlock), kBlock, 0, stream>>>(
        voxels, num_voxels, grid_min, grid_max, out.keys_sorted.data(), out.ids_sorted.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "build_surface_keys_kernel");
    thrust::device_ptr<uint64_t> kptr(out.keys_sorted.data());
    thrust::device_ptr<int32_t> iptr(out.ids_sorted.data());
    thrust::sort_by_key(thrust::cuda::par.on(stream), kptr, kptr + num_voxels, iptr);
    return out;
}

inline FacePairKeys map_and_unique_face_pairs(
    const fdg_gpu::PrimitivePairResult& raw_pairs,
    const SurfaceLookup& lookup,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    cudaStream_t stream) {
    FacePairKeys out;
    const int64_t N = raw_pairs.size;
    if (N == 0) return out;
    fdg_gpu::DeviceBuffer<uint64_t> pair_voxel_keys(N);
    fdg_gpu::DeviceBuffer<int32_t> pair_face_ids(N);
    fdg_gpu::DeviceBuffer<int32_t> mapped_voxel_id(N);
    fdg_gpu::DeviceBuffer<int32_t> mapped_face_id(N);
    fdg_gpu::DeviceBuffer<int32_t> valid(N);
    fdg_gpu::DeviceBuffer<int32_t> offsets(N);
    constexpr int kBlock = 256;

    build_raw_pair_keys_kernel<<<fdg_gpu::ceil_div_i64(N, kBlock), kBlock, 0, stream>>>(
        raw_pairs.voxel_i.data(), raw_pairs.voxel_j.data(), raw_pairs.voxel_k.data(), raw_pairs.prim_id.data(),
        N, grid_min, grid_max, pair_voxel_keys.data(), pair_face_ids.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "build_raw_pair_keys_kernel");

    map_pair_to_voxel_id_kernel<<<fdg_gpu::ceil_div_i64(N, kBlock), kBlock, 0, stream>>>(
        pair_voxel_keys.data(), pair_face_ids.data(), N,
        lookup.keys_sorted.data(), lookup.ids_sorted.data(), lookup.size,
        mapped_voxel_id.data(), mapped_face_id.data(), valid.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "map_pair_to_voxel_id_kernel");

    void* temp = nullptr;
    size_t temp_bytes = 0;
    VOX_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, valid.data(), offsets.data(), static_cast<int>(N), stream));
    VOX_CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
    VOX_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, temp_bytes, valid.data(), offsets.data(), static_cast<int>(N), stream));
    int32_t last_off = voxelize_oct_impl::copy_last_i32(offsets.data(), N, stream);
    int32_t last_valid = voxelize_oct_impl::copy_last_i32(valid.data(), N, stream);
    int64_t M = static_cast<int64_t>(last_off + last_valid);
    cudaFree(temp);

    out.size = M;
    out.keys.allocate(M);
    compact_valid_pairs_kernel<<<fdg_gpu::ceil_div_i64(N, kBlock), kBlock, 0, stream>>>(
        mapped_voxel_id.data(), mapped_face_id.data(), valid.data(), offsets.data(), N, out.keys.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "compact_valid_pairs_kernel");

    thrust::device_ptr<uint64_t> kptr(out.keys.data());
    thrust::sort(thrust::cuda::par.on(stream), kptr, kptr + M);
    auto new_end = thrust::unique(thrust::cuda::par.on(stream), kptr, kptr + M);
    out.size = static_cast<int64_t>(new_end - kptr);
    return out;
}

inline FaceContribStream build_face_contrib_stream(
    const FacePairKeys& pair_keys,
    const float* triangles_world,
    const int32_t* faces_synth,
    cudaStream_t stream) {
    FaceContribStream out;
    out.size = pair_keys.size;
    if (out.size == 0) return out;
    out.voxel_id.allocate(out.size);
    out.qef.allocate(out.size);
    constexpr int kBlock = 256;
    build_face_qef_contrib_kernel<<<fdg_gpu::ceil_div_i64(out.size, kBlock), kBlock, 0, stream>>>(
        pair_keys.keys.data(), out.size, triangles_world, faces_synth, out.voxel_id.data(), out.qef.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "build_face_qef_contrib_kernel");
    return out;
}

inline oct_pairs::FaceQEFResult reduce_face_contribs(
    FaceContribStream&& contrib,
    int64_t num_voxels,
    cudaStream_t stream) {
    oct_pairs::FaceQEFResult out;
    out.size = num_voxels;
    out.qefs.allocate(num_voxels);
    out.qefs.clear_async(stream);
    if (contrib.size == 0) return out;

    thrust::device_ptr<int32_t> kptr(contrib.voxel_id.data());
    thrust::device_ptr<fdg_gpu::SymQEF10> vptr(contrib.qef.data());
    thrust::sort_by_key(thrust::cuda::par.on(stream), kptr, kptr + contrib.size, vptr);

    fdg_gpu::DeviceBuffer<int32_t> reduced_ids(contrib.size);
    fdg_gpu::DeviceBuffer<fdg_gpu::SymQEF10> reduced_qefs(contrib.size);
    auto end_pair = thrust::reduce_by_key(
        thrust::cuda::par.on(stream),
        kptr, kptr + contrib.size,
        vptr,
        thrust::device_pointer_cast(reduced_ids.data()),
        thrust::device_pointer_cast(reduced_qefs.data()),
        thrust::equal_to<int32_t>(),
        SymQEF10Add());
    int64_t M = end_pair.first - thrust::device_pointer_cast(reduced_ids.data());

    constexpr int kBlock = 256;
    scatter_reduced_face_qef_kernel<<<fdg_gpu::ceil_div_i64(M, kBlock), kBlock, 0, stream>>>(
        reduced_ids.data(), reduced_qefs.data(), M, out.qefs.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "scatter_reduced_face_qef_kernel");
    return out;
}

} // anonymous namespace

namespace oct_pairs {

fdg_gpu::PrimitivePairResult voxelize_mesh_oct_gpu(
    const float* d_vertices,
    int64_t num_vertices,
    const int32_t* d_faces,
    int64_t num_faces,
    fdg_gpu::int3_ grid_min_,
    fdg_gpu::int3_ grid_size_,
    float3 voxel_size,
    cudaStream_t stream) {
    using namespace voxelize_oct_impl;
    int3 grid_min{grid_min_.x, grid_min_.y, grid_min_.z};
    int3 grid_size{grid_size_.x, grid_size_.y, grid_size_.z};
    if (d_vertices == nullptr || d_faces == nullptr || num_vertices < 0 || num_faces < 0) {
        throw std::invalid_argument("invalid mesh inputs");
    }
    if (grid_size.x <= 0 || grid_size.y <= 0 || grid_size.z <= 0) {
        throw std::invalid_argument("invalid grid_size");
    }
    if (!(voxel_size.x > 0.0f && voxel_size.y > 0.0f && voxel_size.z > 0.0f)) {
        throw std::invalid_argument("invalid voxel_size");
    }
    if (num_vertices == 0 || num_faces == 0) {
        return {};
    }
    const int d = compute_grid_depth_from_grid_size(grid_size);
    if (d < 0 || d > 21) throw std::invalid_argument("grid depth exceeds 21");
    const float3 inv_voxel_size = reciprocal_voxel_size(voxel_size);

    VoxelizeWorkspace ws;
    FaceDesc* face_desc = nullptr;
    DeviceResult gathered; reset_output(gathered);

    alloc_i32(&ws.leaf_ix, num_vertices);
    alloc_i32(&ws.leaf_iy, num_vertices);
    alloc_i32(&ws.leaf_iz, num_vertices);
    alloc_face_desc(&face_desc, num_faces);

    kernel_build_leaf_coords<<<ceil_div_i64(num_vertices, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(
        d_vertices, num_vertices, inv_voxel_size, grid_min, grid_size, ws.leaf_ix, ws.leaf_iy, ws.leaf_iz);
    VOX_CUDA_CHECK(cudaGetLastError());

    ensure_job_queue_capacity(ws.queue_a, static_cast<int64_t>(kRootNeighborCount) * num_faces);
    ws.queue_a.size = static_cast<int64_t>(kRootNeighborCount) * num_faces;

    kernel_init_faces_and_emit_root27<<<ceil_div_i64(num_faces, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(
        d_vertices, d_faces, ws.leaf_ix, ws.leaf_iy, ws.leaf_iz, num_faces, d, face_desc, ws.queue_a);
    VOX_CUDA_CHECK(cudaGetLastError());

    JobQueue* curr = &ws.queue_a; JobQueue* next = &ws.queue_b;
    while (curr->size > 0) {
        int64_t nj = curr->size;
        ensure_round_capacity(ws.round, nj);
        kernel_count_face_jobs<<<ceil_div_i64(nj, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(
            *curr, face_desc, d, grid_min, grid_size, voxel_size, ws.round.job_hit, ws.round.child_count, ws.round.result_count);
        VOX_CUDA_CHECK(cudaGetLastError());
        exclusive_scan_i32(ws.round, ws.round.child_count, ws.round.child_offset, nj, stream);
        exclusive_scan_i32(ws.round, ws.round.result_count, ws.round.result_offset, nj, stream);
        const int32_t num_children_total = copy_last_i32(ws.round.child_offset, nj, stream) + copy_last_i32(ws.round.child_count, nj, stream);
        const int32_t num_results_total = copy_last_i32(ws.round.result_offset, nj, stream) + copy_last_i32(ws.round.result_count, nj, stream);
        ensure_job_queue_capacity(*next, num_children_total);
        next->size = num_children_total;
        ResultBuffer round_res = make_result_buffer(num_results_total);
        kernel_emit_jobs<<<ceil_div_i64(nj, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(
            *curr, ws.round.job_hit, ws.round.child_offset, ws.round.result_offset, d, *next, round_res);
        VOX_CUDA_CHECK(cudaGetLastError());
        if (num_results_total > 0) ws.result_rounds.push_back(round_res);
        std::swap(curr, next);
    }
    gathered = gather_result_rounds(ws.result_rounds, stream);
    free_ptr(face_desc);
    release_workspace(ws);
    return to_primitive_pair(std::move(gathered));
}

fdg_gpu::PrimitivePairResult voxelize_edge_oct_gpu(
    const float* d_vertices,
    int64_t num_vertices,
    const int32_t* d_edges,
    int64_t num_edges,
    fdg_gpu::int3_ grid_min_,
    fdg_gpu::int3_ grid_size_,
    float3 voxel_size,
    cudaStream_t stream) {
    using namespace voxelize_oct_impl;
    int3 grid_min{grid_min_.x, grid_min_.y, grid_min_.z};
    int3 grid_size{grid_size_.x, grid_size_.y, grid_size_.z};
    if (d_vertices == nullptr || d_edges == nullptr || num_vertices < 0 || num_edges < 0) {
        throw std::invalid_argument("invalid edge inputs");
    }
    if (grid_size.x <= 0 || grid_size.y <= 0 || grid_size.z <= 0) {
        throw std::invalid_argument("invalid grid_size");
    }
    if (!(voxel_size.x > 0.0f && voxel_size.y > 0.0f && voxel_size.z > 0.0f)) {
        throw std::invalid_argument("invalid voxel_size");
    }
    if (num_vertices == 0 || num_edges == 0) {
        return {};
    }
    const int d = compute_grid_depth_from_grid_size(grid_size);
    if (d < 0 || d > 21) throw std::invalid_argument("grid depth exceeds 21");
    const float3 inv_voxel_size = reciprocal_voxel_size(voxel_size);

    VoxelizeWorkspace ws;
    EdgeDesc* edge_desc = nullptr;
    DeviceResult gathered; reset_output(gathered);

    alloc_i32(&ws.leaf_ix, num_vertices);
    alloc_i32(&ws.leaf_iy, num_vertices);
    alloc_i32(&ws.leaf_iz, num_vertices);
    alloc_edge_desc(&edge_desc, num_edges);
    kernel_build_leaf_coords<<<ceil_div_i64(num_vertices, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(
        d_vertices, num_vertices, inv_voxel_size, grid_min, grid_size, ws.leaf_ix, ws.leaf_iy, ws.leaf_iz);
    VOX_CUDA_CHECK(cudaGetLastError());
    ensure_job_queue_capacity(ws.queue_a, static_cast<int64_t>(kRootNeighborCount) * num_edges);
    ws.queue_a.size = static_cast<int64_t>(kRootNeighborCount) * num_edges;
    kernel_init_edges_and_emit_root27<<<ceil_div_i64(num_edges, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(
        d_vertices, d_edges, ws.leaf_ix, ws.leaf_iy, ws.leaf_iz, num_edges, d, edge_desc, ws.queue_a);
    VOX_CUDA_CHECK(cudaGetLastError());
    JobQueue* curr = &ws.queue_a; JobQueue* next = &ws.queue_b;
    while (curr->size > 0) {
        int64_t nj = curr->size;
        ensure_round_capacity(ws.round, nj);
        kernel_count_edge_jobs<<<ceil_div_i64(nj, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(
            *curr, edge_desc, d, grid_min, grid_size, voxel_size, ws.round.job_hit, ws.round.child_count, ws.round.result_count);
        VOX_CUDA_CHECK(cudaGetLastError());
        exclusive_scan_i32(ws.round, ws.round.child_count, ws.round.child_offset, nj, stream);
        exclusive_scan_i32(ws.round, ws.round.result_count, ws.round.result_offset, nj, stream);
        const int32_t num_children_total = copy_last_i32(ws.round.child_offset, nj, stream) + copy_last_i32(ws.round.child_count, nj, stream);
        const int32_t num_results_total = copy_last_i32(ws.round.result_offset, nj, stream) + copy_last_i32(ws.round.result_count, nj, stream);
        ensure_job_queue_capacity(*next, num_children_total);
        next->size = num_children_total;
        ResultBuffer round_res = make_result_buffer(num_results_total);
        kernel_emit_jobs<<<ceil_div_i64(nj, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(
            *curr, ws.round.job_hit, ws.round.child_offset, ws.round.result_offset, d, *next, round_res);
        VOX_CUDA_CHECK(cudaGetLastError());
        if (num_results_total > 0) ws.result_rounds.push_back(round_res);
        std::swap(curr, next);
    }
    gathered = gather_result_rounds(ws.result_rounds, stream);
    free_ptr(edge_desc);
    release_workspace(ws);
    return to_primitive_pair(std::move(gathered));
}

FaceQEFResult face_qef_gpu(
    float3 voxel_size,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    const float* triangles,
    int64_t num_triangles,
    const int* voxels,
    int64_t num_voxels,
    cudaStream_t stream) {
    FaceQEFResult out;
    out.size = num_voxels;
    out.qefs.allocate(num_voxels);
    out.qefs.clear_async(stream);
    if (num_voxels == 0 || num_triangles == 0) return out;
    if (triangles == nullptr || voxels == nullptr) throw std::invalid_argument("null face_qef inputs");

    const int64_t num_tri_vertices = num_triangles * 3;
    fdg_gpu::DeviceBuffer<int32_t> faces_synth(num_triangles * 3);

    constexpr int kBlock = 256;
    build_synth_faces_kernel<<<fdg_gpu::ceil_div_i64(num_triangles, kBlock), kBlock, 0, stream>>>(
        num_triangles, faces_synth.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "build_synth_faces_kernel");

    fdg_gpu::int3_ grid_size{grid_max.x - grid_min.x, grid_max.y - grid_min.y, grid_max.z - grid_min.z};
    auto raw_pairs = voxelize_mesh_oct_gpu(
        triangles, num_tri_vertices, faces_synth.data(), num_triangles, grid_min, grid_size, voxel_size, stream);

    auto lookup = build_surface_lookup(voxels, num_voxels, grid_min, grid_max, stream);
    auto face_pair_keys = map_and_unique_face_pairs(raw_pairs, lookup, grid_min, grid_max, stream);
    auto contrib = build_face_contrib_stream(face_pair_keys, triangles, faces_synth.data(), stream);
    return reduce_face_contribs(std::move(contrib), num_voxels, stream);
}

} // namespace oct_pairs
