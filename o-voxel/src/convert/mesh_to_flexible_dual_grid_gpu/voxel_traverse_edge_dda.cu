#include "voxel_traverse_edge_dda.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/device/device_scan.cuh>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>


#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace voxel_traverse_edge_dda_impl {

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

constexpr int kDefaultBlockSize = 128;

struct EdgeDesc {
    float3 v0_ws;
    float3 v1_ws;

    double3 dir_unit;
    double segment_length;

    int32_t start_x;
    int32_t start_y;
    int32_t start_z;

    int8_t step_x;
    int8_t step_y;
    int8_t step_z;

    double tmax0_x;
    double tmax0_y;
    double tmax0_z;

    double tdelta_x;
    double tdelta_y;
    double tdelta_z;
};

struct DDAJobQueue {
    int32_t* edge_id = nullptr;

    int32_t* cur_x = nullptr;
    int32_t* cur_y = nullptr;
    int32_t* cur_z = nullptr;

    double* tmax_x = nullptr;
    double* tmax_y = nullptr;
    double* tmax_z = nullptr;

    int64_t size = 0;
    int64_t capacity = 0;
};

struct RoundBuffers {
    int32_t* pair_count = nullptr;
    int32_t* next_job_count = nullptr;

    int32_t* pair_offset = nullptr;
    int32_t* next_job_offset = nullptr;

    void* cub_temp_storage = nullptr;
    size_t cub_temp_bytes = 0;
    int64_t capacity = 0;
};

struct ResultBuffer {
    int32_t* edge_id = nullptr;
    int32_t* vi = nullptr;
    int32_t* vj = nullptr;
    int32_t* vk = nullptr;
    int64_t size = 0;
};

struct DeviceResult {
    int32_t* edge_id = nullptr;
    int32_t* voxel_i = nullptr;
    int32_t* voxel_j = nullptr;
    int32_t* voxel_k = nullptr;
    int64_t size = 0;
};

struct Workspace {
    EdgeDesc* edge_desc = nullptr;

    uint8_t* edge_valid = nullptr;
    int32_t* init_count = nullptr;
    int32_t* init_offset = nullptr;

    DDAJobQueue queue_a;
    DDAJobQueue queue_b;

    RoundBuffers round;
    std::vector<ResultBuffer> result_rounds;
};

inline int ceil_div_i64(int64_t n, int block) {
    return static_cast<int>((n + block - 1) / block);
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

inline void alloc_double(double** ptr, int64_t n) {
    *ptr = nullptr;
    if (n > 0) {
        VOX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(double) * n));
    }
}

inline void alloc_edge_desc(EdgeDesc** ptr, int64_t n) {
    *ptr = nullptr;
    if (n > 0) {
        VOX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(EdgeDesc) * n));
    }
}

inline void release_dda_job_queue(DDAJobQueue& q) {
    free_ptr(q.edge_id);
    free_ptr(q.cur_x);
    free_ptr(q.cur_y);
    free_ptr(q.cur_z);
    free_ptr(q.tmax_x);
    free_ptr(q.tmax_y);
    free_ptr(q.tmax_z);
    q = {};
}

inline void release_round_buffers(RoundBuffers& b) {
    free_ptr(b.pair_count);
    free_ptr(b.next_job_count);
    free_ptr(b.pair_offset);
    free_ptr(b.next_job_offset);
    free_ptr(b.cub_temp_storage);
    b = {};
}

inline void release_result_buffer(ResultBuffer& r) {
    free_ptr(r.edge_id);
    free_ptr(r.vi);
    free_ptr(r.vj);
    free_ptr(r.vk);
    r = {};
}

inline void release_workspace(Workspace& ws) {
    free_ptr(ws.edge_desc);
    free_ptr(ws.edge_valid);
    free_ptr(ws.init_count);
    free_ptr(ws.init_offset);
    release_dda_job_queue(ws.queue_a);
    release_dda_job_queue(ws.queue_b);
    release_round_buffers(ws.round);
    for (auto& r : ws.result_rounds) {
        release_result_buffer(r);
    }
    ws.result_rounds.clear();
}

inline void ensure_dda_job_queue_capacity(DDAJobQueue& q, int64_t capacity) {
    if (capacity <= q.capacity) {
        return;
    }
    release_dda_job_queue(q);
    alloc_i32(&q.edge_id, capacity);
    alloc_i32(&q.cur_x, capacity);
    alloc_i32(&q.cur_y, capacity);
    alloc_i32(&q.cur_z, capacity);
    alloc_double(&q.tmax_x, capacity);
    alloc_double(&q.tmax_y, capacity);
    alloc_double(&q.tmax_z, capacity);
    q.capacity = capacity;
    q.size = 0;
}

inline void ensure_round_capacity(RoundBuffers& b, int64_t capacity) {
    if (capacity <= b.capacity) {
        return;
    }
    free_ptr(b.pair_count);
    free_ptr(b.next_job_count);
    free_ptr(b.pair_offset);
    free_ptr(b.next_job_offset);

    alloc_i32(&b.pair_count, capacity);
    alloc_i32(&b.next_job_count, capacity);
    alloc_i32(&b.pair_offset, capacity);
    alloc_i32(&b.next_job_offset, capacity);
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
    alloc_i32(&r.edge_id, count);
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

    alloc_i32(&out.edge_id, total);
    alloc_i32(&out.voxel_i, total);
    alloc_i32(&out.voxel_j, total);
    alloc_i32(&out.voxel_k, total);

    int64_t cursor = 0;
    for (const auto& r : rounds) {
        if (r.size == 0) {
            continue;
        }
        VOX_CUDA_CHECK(cudaMemcpyAsync(
            out.edge_id + cursor,
            r.edge_id,
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

__device__ inline int argmin_axis_strict(double tx, double ty, double tz) {
    if (tx < ty) {
        return (tx < tz) ? 0 : 2;
    }
    return (ty < tz) ? 1 : 2;
}

__device__ inline bool in_bounds_voxel_abs(
    int x,
    int y,
    int z,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max) {
    return (grid_min.x <= x && x < grid_max.x) &&
           (grid_min.y <= y && y < grid_max.y) &&
           (grid_min.z <= z && z < grid_max.z);
}

__global__ void kernel_build_edge_desc(
    const float* __restrict__ vertices,
    const int32_t* __restrict__ edges,
    int64_t num_edges,
    float3 voxel_size,
    EdgeDesc* __restrict__ edge_desc,
    uint8_t* __restrict__ edge_valid) {
    int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (eid >= num_edges) {
        return;
    }

    int v0_id = edges[2 * eid + 0];
    int v1_id = edges[2 * eid + 1];

    float3 v0 = make_float3(vertices[3 * v0_id + 0], vertices[3 * v0_id + 1], vertices[3 * v0_id + 2]);
    float3 v1 = make_float3(vertices[3 * v1_id + 0], vertices[3 * v1_id + 1], vertices[3 * v1_id + 2]);

    double dx = static_cast<double>(v1.x) - static_cast<double>(v0.x);
    double dy = static_cast<double>(v1.y) - static_cast<double>(v0.y);
    double dz = static_cast<double>(v1.z) - static_cast<double>(v0.z);
    double segment_length = sqrt(dx * dx + dy * dy + dz * dz);

    if (segment_length < 1e-6) {
        edge_valid[eid] = 0;
        return;
    }

    double3 dir_unit = make_double3(dx / segment_length, dy / segment_length, dz / segment_length);

    int32_t sx = static_cast<int32_t>(floor(static_cast<double>(v0.x) / static_cast<double>(voxel_size.x)));
    int32_t sy = static_cast<int32_t>(floor(static_cast<double>(v0.y) / static_cast<double>(voxel_size.y)));
    int32_t sz = static_cast<int32_t>(floor(static_cast<double>(v0.z) / static_cast<double>(voxel_size.z)));

    int8_t step_x = (dir_unit.x > 0.0) ? 1 : -1;
    int8_t step_y = (dir_unit.y > 0.0) ? 1 : -1;
    int8_t step_z = (dir_unit.z > 0.0) ? 1 : -1;

    double tmax_x, tmax_y, tmax_z;
    double tdelta_x, tdelta_y, tdelta_z;

    if (dir_unit.x == 0.0) {
        tmax_x = CUDART_INF;
        tdelta_x = CUDART_INF;
    } else {
        double voxel_border = static_cast<double>(voxel_size.x) * static_cast<double>(sx + (step_x > 0 ? 1 : 0));
        tmax_x = (voxel_border - static_cast<double>(v0.x)) / dir_unit.x;
        tdelta_x = static_cast<double>(voxel_size.x) / fabs(dir_unit.x);
    }

    if (dir_unit.y == 0.0) {
        tmax_y = CUDART_INF;
        tdelta_y = CUDART_INF;
    } else {
        double voxel_border = static_cast<double>(voxel_size.y) * static_cast<double>(sy + (step_y > 0 ? 1 : 0));
        tmax_y = (voxel_border - static_cast<double>(v0.y)) / dir_unit.y;
        tdelta_y = static_cast<double>(voxel_size.y) / fabs(dir_unit.y);
    }

    if (dir_unit.z == 0.0) {
        tmax_z = CUDART_INF;
        tdelta_z = CUDART_INF;
    } else {
        double voxel_border = static_cast<double>(voxel_size.z) * static_cast<double>(sz + (step_z > 0 ? 1 : 0));
        tmax_z = (voxel_border - static_cast<double>(v0.z)) / dir_unit.z;
        tdelta_z = static_cast<double>(voxel_size.z) / fabs(dir_unit.z);
    }

    EdgeDesc desc;
    desc.v0_ws = v0;
    desc.v1_ws = v1;
    desc.dir_unit = dir_unit;
    desc.segment_length = segment_length;
    desc.start_x = sx;
    desc.start_y = sy;
    desc.start_z = sz;
    desc.step_x = step_x;
    desc.step_y = step_y;
    desc.step_z = step_z;
    desc.tmax0_x = tmax_x;
    desc.tmax0_y = tmax_y;
    desc.tmax0_z = tmax_z;
    desc.tdelta_x = tdelta_x;
    desc.tdelta_y = tdelta_y;
    desc.tdelta_z = tdelta_z;

    edge_desc[eid] = desc;
    edge_valid[eid] = 1;
}

__global__ void kernel_count_init_jobs(
    const uint8_t* __restrict__ edge_valid,
    int64_t num_edges,
    int32_t* __restrict__ init_count) {
    int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (eid >= num_edges) {
        return;
    }
    init_count[eid] = edge_valid[eid] ? 1 : 0;
}

__global__ void kernel_emit_init_jobs(
    const uint8_t* __restrict__ edge_valid,
    const EdgeDesc* __restrict__ edge_desc,
    const int32_t* __restrict__ init_offset,
    int64_t num_edges,
    DDAJobQueue out_q) {
    int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (eid >= num_edges || !edge_valid[eid]) {
        return;
    }

    int32_t out = init_offset[eid];
    const EdgeDesc& desc = edge_desc[eid];
    out_q.edge_id[out] = static_cast<int32_t>(eid);
    out_q.cur_x[out] = desc.start_x;
    out_q.cur_y[out] = desc.start_y;
    out_q.cur_z[out] = desc.start_z;
    out_q.tmax_x[out] = desc.tmax0_x;
    out_q.tmax_y[out] = desc.tmax0_y;
    out_q.tmax_z[out] = desc.tmax0_z;
}

__global__ void kernel_count_dda_jobs(
    DDAJobQueue curr_q,
    const EdgeDesc* __restrict__ edge_desc,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    int chunk_steps,
    int32_t* __restrict__ pair_count,
    int32_t* __restrict__ next_job_count) {
    int64_t jid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (jid >= curr_q.size) {
        return;
    }

    int32_t eid = curr_q.edge_id[jid];
    const EdgeDesc& desc = edge_desc[eid];

    int32_t cx = curr_q.cur_x[jid];
    int32_t cy = curr_q.cur_y[jid];
    int32_t cz = curr_q.cur_z[jid];
    double tx = curr_q.tmax_x[jid];
    double ty = curr_q.tmax_y[jid];
    double tz = curr_q.tmax_z[jid];

    int32_t local_pairs = 0;
    bool alive = true;

    if (in_bounds_voxel_abs(cx, cy, cz, grid_min, grid_max)) {
        local_pairs += 1;
    }

    for (int step_idx = 0; step_idx < chunk_steps; ++step_idx) {
        int axis = argmin_axis_strict(tx, ty, tz);
        double t_axis = (axis == 0) ? tx : (axis == 1 ? ty : tz);
        if (t_axis > desc.segment_length) {
            alive = false;
            break;
        }

        if (axis == 0) {
            cx += static_cast<int32_t>(desc.step_x);
            tx += desc.tdelta_x;
        } else if (axis == 1) {
            cy += static_cast<int32_t>(desc.step_y);
            ty += desc.tdelta_y;
        } else {
            cz += static_cast<int32_t>(desc.step_z);
            tz += desc.tdelta_z;
        }

        if (in_bounds_voxel_abs(cx, cy, cz, grid_min, grid_max)) {
            local_pairs += 1;
        }
    }

    pair_count[jid] = local_pairs;
    next_job_count[jid] = alive ? 1 : 0;
}

__global__ void kernel_emit_dda_jobs(
    DDAJobQueue curr_q,
    const EdgeDesc* __restrict__ edge_desc,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    int chunk_steps,
    const int32_t* __restrict__ pair_offset,
    const int32_t* __restrict__ next_job_offset,
    ResultBuffer out_res,
    DDAJobQueue next_q) {
    int64_t jid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (jid >= curr_q.size) {
        return;
    }

    int32_t eid = curr_q.edge_id[jid];
    const EdgeDesc& desc = edge_desc[eid];

    int32_t cx = curr_q.cur_x[jid];
    int32_t cy = curr_q.cur_y[jid];
    int32_t cz = curr_q.cur_z[jid];
    double tx = curr_q.tmax_x[jid];
    double ty = curr_q.tmax_y[jid];
    double tz = curr_q.tmax_z[jid];

    int32_t out_pair = pair_offset[jid];
    bool alive = true;

    if (in_bounds_voxel_abs(cx, cy, cz, grid_min, grid_max)) {
        out_res.edge_id[out_pair] = eid;
        out_res.vi[out_pair] = cx;
        out_res.vj[out_pair] = cy;
        out_res.vk[out_pair] = cz;
        out_pair += 1;
    }

    for (int step_idx = 0; step_idx < chunk_steps; ++step_idx) {
        int axis = argmin_axis_strict(tx, ty, tz);
        double t_axis = (axis == 0) ? tx : (axis == 1 ? ty : tz);
        if (t_axis > desc.segment_length) {
            alive = false;
            break;
        }

        if (axis == 0) {
            cx += static_cast<int32_t>(desc.step_x);
            tx += desc.tdelta_x;
        } else if (axis == 1) {
            cy += static_cast<int32_t>(desc.step_y);
            ty += desc.tdelta_y;
        } else {
            cz += static_cast<int32_t>(desc.step_z);
            tz += desc.tdelta_z;
        }

        if (in_bounds_voxel_abs(cx, cy, cz, grid_min, grid_max)) {
            out_res.edge_id[out_pair] = eid;
            out_res.vi[out_pair] = cx;
            out_res.vj[out_pair] = cy;
            out_res.vk[out_pair] = cz;
            out_pair += 1;
        }
    }

    if (alive) {
        int32_t out_job = next_job_offset[jid];
        next_q.edge_id[out_job] = eid;
        next_q.cur_x[out_job] = cx;
        next_q.cur_y[out_job] = cy;
        next_q.cur_z[out_job] = cz;
        next_q.tmax_x[out_job] = tx;
        next_q.tmax_y[out_job] = ty;
        next_q.tmax_z[out_job] = tz;
    }
}

inline void release_device_result(DeviceResult& out) {
    free_ptr(out.edge_id);
    free_ptr(out.voxel_i);
    free_ptr(out.voxel_j);
    free_ptr(out.voxel_k);
    out = {};
}

}  // namespace voxel_traverse_edge_dda_impl

namespace {

inline fdg_gpu::PrimitivePairResult to_primitive_pair(voxel_traverse_edge_dda_impl::DeviceResult&& r) {
    fdg_gpu::PrimitivePairResult out;
    out.size = r.size;
    out.prim_id.adopt(r.edge_id, r.size);
    out.voxel_i.adopt(r.voxel_i, r.size);
    out.voxel_j.adopt(r.voxel_j, r.size);
    out.voxel_k.adopt(r.voxel_k, r.size);
    r.edge_id = nullptr;
    r.voxel_i = nullptr;
    r.voxel_j = nullptr;
    r.voxel_k = nullptr;
    r.size = 0;
    return out;
}

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

struct SurfaceLookup {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<uint64_t> keys_sorted;
    fdg_gpu::DeviceBuffer<int32_t> ids_sorted;
};

struct EdgePairKeys {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<uint64_t> pair_keys;
};

struct BoundaryContribStream {
    int64_t size = 0;
    fdg_gpu::DeviceBuffer<int32_t> voxel_id;
    fdg_gpu::DeviceBuffer<fdg_gpu::SymQEF10> qef;
};


__global__ void copy_boundaries_to_vertices_kernel(const float* boundaries, int64_t num_boundaries, float* vertices_out) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= 2 * num_boundaries) return;
    vertices_out[3 * tid + 0] = boundaries[3 * tid + 0];
    vertices_out[3 * tid + 1] = boundaries[3 * tid + 1];
    vertices_out[3 * tid + 2] = boundaries[3 * tid + 2];
}

__global__ void build_synth_edges_kernel(int64_t num_boundaries, int32_t* edges_out) {
    int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (eid >= num_boundaries) return;
    edges_out[2 * eid + 0] = static_cast<int32_t>(2 * eid + 0);
    edges_out[2 * eid + 1] = static_cast<int32_t>(2 * eid + 1);
}

__global__ void build_surface_keys_kernel(const int* voxels, int64_t num_voxels, fdg_gpu::int3_ grid_min, fdg_gpu::int3_ grid_max, uint64_t* keys, int32_t* ids) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_voxels) return;
    int x = voxels[3 * i + 0];
    int y = voxels[3 * i + 1];
    int z = voxels[3 * i + 2];
    keys[i] = fdg_gpu::pack_voxel_key(x, y, z, grid_min, grid_max);
    ids[i] = static_cast<int32_t>(i);
}

__global__ void build_raw_pair_voxel_keys_kernel(const int32_t* voxel_i, const int32_t* voxel_j, const int32_t* voxel_k, int64_t num_pairs, fdg_gpu::int3_ grid_min, fdg_gpu::int3_ grid_max, uint64_t* pair_voxel_keys) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;
    pair_voxel_keys[i] = fdg_gpu::pack_voxel_key(voxel_i[i], voxel_j[i], voxel_k[i], grid_min, grid_max);
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

__global__ void map_pair_to_voxel_id_kernel(const uint64_t* pair_voxel_keys, const int32_t* edge_id, int64_t num_pairs, const uint64_t* surface_keys_sorted, const int32_t* surface_ids_sorted, int64_t num_voxels, int32_t* mapped_voxel_id, int32_t* mapped_edge_id, int32_t* valid) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;
    uint64_t key = pair_voxel_keys[i];
    int pos = lower_bound_u64(surface_keys_sorted, num_voxels, key);
    if (pos < num_voxels && surface_keys_sorted[pos] == key) {
        mapped_voxel_id[i] = surface_ids_sorted[pos];
        mapped_edge_id[i] = edge_id[i];
        valid[i] = 1;
    } else {
        mapped_voxel_id[i] = -1;
        mapped_edge_id[i] = -1;
        valid[i] = 0;
    }
}

__host__ __device__ inline uint64_t pack_edge_voxel_pair_key(int32_t edge_id, uint64_t voxel_key) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(edge_id)) << 32) ^ voxel_key;
}

__global__ void compact_valid_pairs_kernel(const int32_t* mapped_voxel_id, const int32_t* mapped_edge_id, const int32_t* valid, const int32_t* offsets, int64_t num_pairs, uint64_t* pair_keys_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs || valid[i] == 0) return;
    int32_t out = offsets[i];
    uint64_t voxel_key = static_cast<uint32_t>(mapped_voxel_id[i]);
    pair_keys_out[out] = (static_cast<uint64_t>(static_cast<uint32_t>(mapped_edge_id[i])) << 32) | voxel_key;
}

__global__ void decode_pair_keys_kernel(const uint64_t* pair_keys, int64_t num_pairs, int32_t* voxel_id, int32_t* edge_id) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;
    edge_id[i] = static_cast<int32_t>(pair_keys[i] >> 32);
    voxel_id[i] = static_cast<int32_t>(pair_keys[i] & 0xffffffffu);
}

__device__ inline fdg_gpu::SymQEF10 symqef10_from_boundary(float3 p0, float3 p1, float boundary_weight) {
    double dx = static_cast<double>(p1.x) - static_cast<double>(p0.x);
    double dy = static_cast<double>(p1.y) - static_cast<double>(p0.y);
    double dz = static_cast<double>(p1.z) - static_cast<double>(p0.z);
    double L = sqrt(dx * dx + dy * dy + dz * dz);
    if (L < 1e-6) return symqef10_zero();
    double ux = dx / L;
    double uy = dy / L;
    double uz = dz / L;
    double A00 = 1.0 - ux * ux;
    double A01 = -ux * uy;
    double A02 = -ux * uz;
    double A11 = 1.0 - uy * uy;
    double A12 = -uy * uz;
    double A22 = 1.0 - uz * uz;
    double bx = -(A00 * p0.x + A01 * p0.y + A02 * p0.z);
    double by = -(A01 * p0.x + A11 * p0.y + A12 * p0.z);
    double bz = -(A02 * p0.x + A12 * p0.y + A22 * p0.z);
    double c = p0.x * (A00 * p0.x + A01 * p0.y + A02 * p0.z) +
               p0.y * (A01 * p0.x + A11 * p0.y + A12 * p0.z) +
               p0.z * (A02 * p0.x + A12 * p0.y + A22 * p0.z);
    float w = boundary_weight;
    return fdg_gpu::SymQEF10{
        static_cast<float>(w * A00), static_cast<float>(w * A01), static_cast<float>(w * A02), static_cast<float>(w * bx),
        static_cast<float>(w * A11), static_cast<float>(w * A12), static_cast<float>(w * by),
        static_cast<float>(w * A22), static_cast<float>(w * bz),
        static_cast<float>(w * c)
    };
}

__global__ void build_boundary_qef_contrib_kernel(const int32_t* voxel_id, const int32_t* edge_id, int64_t num_pairs, const float* boundary_vertices, const int32_t* boundary_edges, float boundary_weight, int32_t* out_voxel_id, fdg_gpu::SymQEF10* out_qef) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;
    int32_t eid = edge_id[i];
    int32_t i0 = boundary_edges[2 * eid + 0];
    int32_t i1 = boundary_edges[2 * eid + 1];
    float3 p0 = make_float3(boundary_vertices[3 * i0 + 0], boundary_vertices[3 * i0 + 1], boundary_vertices[3 * i0 + 2]);
    float3 p1 = make_float3(boundary_vertices[3 * i1 + 0], boundary_vertices[3 * i1 + 1], boundary_vertices[3 * i1 + 2]);
    out_voxel_id[i] = voxel_id[i];
    out_qef[i] = symqef10_from_boundary(p0, p1, boundary_weight);
}

__global__ void scatter_reduced_qef_kernel(const int32_t* reduced_voxel_id, const fdg_gpu::SymQEF10* reduced_qef, int64_t M, fdg_gpu::SymQEF10* full_qef) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= M) return;
    full_qef[reduced_voxel_id[i]] = reduced_qef[i];
}

inline SurfaceLookup build_surface_lookup(const int* voxels, int64_t num_voxels, fdg_gpu::int3_ grid_min, fdg_gpu::int3_ grid_max, cudaStream_t stream) {
    SurfaceLookup out;
    out.size = num_voxels;
    out.keys_sorted.allocate(num_voxels);
    out.ids_sorted.allocate(num_voxels);
    constexpr int kBlock = 128;
    build_surface_keys_kernel<<<fdg_gpu::ceil_div_i64(num_voxels, kBlock), kBlock, 0, stream>>>(voxels, num_voxels, grid_min, grid_max, out.keys_sorted.data(), out.ids_sorted.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "build_surface_keys_kernel");
    thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(out.keys_sorted.data()), thrust::device_pointer_cast(out.keys_sorted.data()) + num_voxels, thrust::device_pointer_cast(out.ids_sorted.data()));
    return out;
}

inline EdgePairKeys map_and_unique_edge_pairs(const fdg_gpu::PrimitivePairResult& raw_pairs, const SurfaceLookup& lookup, fdg_gpu::int3_ grid_min, fdg_gpu::int3_ grid_max, cudaStream_t stream) {
    EdgePairKeys out;
    const int64_t N = raw_pairs.size;
    if (N == 0) return out;
    fdg_gpu::DeviceBuffer<uint64_t> pair_voxel_keys(N);
    fdg_gpu::DeviceBuffer<int32_t> mapped_voxel_id(N);
    fdg_gpu::DeviceBuffer<int32_t> mapped_edge_id(N);
    fdg_gpu::DeviceBuffer<int32_t> valid(N);
    fdg_gpu::DeviceBuffer<int32_t> offsets(N);
    constexpr int kBlock = 128;
    build_raw_pair_voxel_keys_kernel<<<fdg_gpu::ceil_div_i64(N, kBlock), kBlock, 0, stream>>>(raw_pairs.voxel_i.data(), raw_pairs.voxel_j.data(), raw_pairs.voxel_k.data(), N, grid_min, grid_max, pair_voxel_keys.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "build_raw_pair_voxel_keys_kernel");
    map_pair_to_voxel_id_kernel<<<fdg_gpu::ceil_div_i64(N, kBlock), kBlock, 0, stream>>>(pair_voxel_keys.data(), raw_pairs.prim_id.data(), N, lookup.keys_sorted.data(), lookup.ids_sorted.data(), lookup.size, mapped_voxel_id.data(), mapped_edge_id.data(), valid.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "map_pair_to_voxel_id_kernel");
    size_t temp_bytes = 0;
    void* temp = nullptr;
    VOX_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, valid.data(), offsets.data(), static_cast<int>(N), stream));
    VOX_CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
    VOX_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, temp_bytes, valid.data(), offsets.data(), static_cast<int>(N), stream));
    int32_t last_off = voxel_traverse_edge_dda_impl::copy_last_i32(offsets.data(), N, stream);
    int32_t last_valid = voxel_traverse_edge_dda_impl::copy_last_i32(valid.data(), N, stream);
    cudaFree(temp);
    int64_t M = static_cast<int64_t>(last_off) + static_cast<int64_t>(last_valid);
    out.size = M;
    out.pair_keys.allocate(M);
    compact_valid_pairs_kernel<<<fdg_gpu::ceil_div_i64(N, kBlock), kBlock, 0, stream>>>(mapped_voxel_id.data(), mapped_edge_id.data(), valid.data(), offsets.data(), N, out.pair_keys.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "compact_valid_pairs_kernel");
    auto ptr = thrust::device_pointer_cast(out.pair_keys.data());
    thrust::sort(thrust::cuda::par.on(stream), ptr, ptr + M);
    auto new_end = thrust::unique(thrust::cuda::par.on(stream), ptr, ptr + M);
    out.size = static_cast<int64_t>(new_end - ptr);
    return out;
}

inline BoundaryContribStream build_boundary_contrib_stream(const EdgePairKeys& pair_keys, const float* boundary_vertices, const int32_t* boundary_edges, float boundary_weight, cudaStream_t stream) {
    BoundaryContribStream out;
    out.size = pair_keys.size;
    out.voxel_id.allocate(out.size);
    out.qef.allocate(out.size);
    fdg_gpu::DeviceBuffer<int32_t> edge_id(out.size);
    constexpr int kBlock = 128;
    decode_pair_keys_kernel<<<fdg_gpu::ceil_div_i64(out.size, kBlock), kBlock, 0, stream>>>(pair_keys.pair_keys.data(), out.size, out.voxel_id.data(), edge_id.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "decode_pair_keys_kernel");
    build_boundary_qef_contrib_kernel<<<fdg_gpu::ceil_div_i64(out.size, kBlock), kBlock, 0, stream>>>(out.voxel_id.data(), edge_id.data(), out.size, boundary_vertices, boundary_edges, boundary_weight, out.voxel_id.data(), out.qef.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "build_boundary_qef_contrib_kernel");
    return out;
}

inline edge_dda::BoundaryQEFResult reduce_boundary_contribs(BoundaryContribStream&& contrib, int64_t num_voxels, cudaStream_t stream) {
    edge_dda::BoundaryQEFResult out;
    out.size = num_voxels;
    out.qefs.allocate(num_voxels);
    out.qefs.clear_async(stream);
    if (contrib.size == 0) return out;
    auto kptr = thrust::device_pointer_cast(contrib.voxel_id.data());
    auto vptr = thrust::device_pointer_cast(contrib.qef.data());
    thrust::sort_by_key(thrust::cuda::par.on(stream), kptr, kptr + contrib.size, vptr);
    fdg_gpu::DeviceBuffer<int32_t> reduced_ids(contrib.size);
    fdg_gpu::DeviceBuffer<fdg_gpu::SymQEF10> reduced_qefs(contrib.size);
    auto end_pair = thrust::reduce_by_key(thrust::cuda::par.on(stream), kptr, kptr + contrib.size, vptr, thrust::device_pointer_cast(reduced_ids.data()), thrust::device_pointer_cast(reduced_qefs.data()), thrust::equal_to<int32_t>(), SymQEF10Add{});
    int64_t M = end_pair.first - thrust::device_pointer_cast(reduced_ids.data());
    constexpr int kBlock = 128;
    scatter_reduced_qef_kernel<<<fdg_gpu::ceil_div_i64(M, kBlock), kBlock, 0, stream>>>(reduced_ids.data(), reduced_qefs.data(), M, out.qefs.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "scatter_reduced_qef_kernel");
    return out;
}

inline fdg_gpu::PrimitivePairResult dedup_pairs(fdg_gpu::PrimitivePairResult&& in, cudaStream_t stream) {
    auto pid = thrust::device_pointer_cast(in.prim_id.data());
    auto vi = thrust::device_pointer_cast(in.voxel_i.data());
    auto vj = thrust::device_pointer_cast(in.voxel_j.data());
    auto vk = thrust::device_pointer_cast(in.voxel_k.data());
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(pid, vi, vj, vk));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(pid + in.size, vi + in.size, vj + in.size, vk + in.size));
    thrust::sort(thrust::cuda::par.on(stream), begin, end);
    auto new_end = thrust::unique(thrust::cuda::par.on(stream), begin, end);
    in.size = static_cast<int64_t>(new_end - begin);
    return std::move(in);
}

} // anonymous namespace

namespace edge_dda {

fdg_gpu::PrimitivePairResult voxel_traverse_edge_dda_gpu(
    const float* d_vertices,
    int64_t num_vertices,
    const int32_t* d_edges,
    int64_t num_edges,
    float3 voxel_size,
    fdg_gpu::int3_ grid_min,
    fdg_gpu::int3_ grid_max,
    int chunk_steps,
    cudaStream_t stream) {
    using namespace voxel_traverse_edge_dda_impl;
    if (d_vertices == nullptr || d_edges == nullptr || num_vertices < 0 || num_edges < 0) {
        throw std::invalid_argument("invalid edge inputs");
    }
    if (!(voxel_size.x > 0.0f && voxel_size.y > 0.0f && voxel_size.z > 0.0f)) {
        throw std::invalid_argument("invalid voxel_size");
    }
    if (grid_max.x <= grid_min.x || grid_max.y <= grid_min.y || grid_max.z <= grid_min.z) {
        throw std::invalid_argument("invalid grid range");
    }
    if (chunk_steps <= 0) {
        throw std::invalid_argument("chunk_steps must be positive");
    }
    if (num_vertices == 0 || num_edges == 0) return {};

    Workspace ws;
    DeviceResult gathered{};
    gathered.edge_id = nullptr;
    gathered.voxel_i = nullptr;
    gathered.voxel_j = nullptr;
    gathered.voxel_k = nullptr;
    gathered.size = 0;

    VOX_CUDA_CHECK(cudaGetLastError());
    try {
        alloc_edge_desc(&ws.edge_desc, num_edges);
        alloc_u8(&ws.edge_valid, num_edges);
        alloc_i32(&ws.init_count, num_edges);
        alloc_i32(&ws.init_offset, num_edges);
        kernel_build_edge_desc<<<ceil_div_i64(num_edges, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(d_vertices, d_edges, num_edges, voxel_size,  ws.edge_desc, ws.edge_valid);
        VOX_CUDA_CHECK(cudaGetLastError());
        kernel_count_init_jobs<<<ceil_div_i64(num_edges, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(ws.edge_valid, num_edges, ws.init_count);
        VOX_CUDA_CHECK(cudaGetLastError());
        ensure_round_capacity(ws.round, num_edges);
        exclusive_scan_i32(ws.round, ws.init_count, ws.init_offset, num_edges, stream);
        int32_t last_init_offset = copy_last_i32(ws.init_offset, num_edges, stream);
        int32_t last_init_count = copy_last_i32(ws.init_count, num_edges, stream);
        int64_t num_init_jobs = static_cast<int64_t>(last_init_offset) + static_cast<int64_t>(last_init_count);
        ensure_dda_job_queue_capacity(ws.queue_a, num_init_jobs);
        ws.queue_a.size = num_init_jobs;
        kernel_emit_init_jobs<<<ceil_div_i64(num_edges, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(ws.edge_valid, ws.edge_desc, ws.init_offset, num_edges, ws.queue_a);
        VOX_CUDA_CHECK(cudaGetLastError());
        DDAJobQueue* curr = &ws.queue_a;
        DDAJobQueue* next = &ws.queue_b;
        while (curr->size > 0) {
            int64_t nj = curr->size;
            ensure_round_capacity(ws.round, nj);
            kernel_count_dda_jobs<<<ceil_div_i64(nj, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(*curr, ws.edge_desc, grid_min, grid_max, chunk_steps, ws.round.pair_count, ws.round.next_job_count);
            VOX_CUDA_CHECK(cudaGetLastError());
            exclusive_scan_i32(ws.round, ws.round.pair_count, ws.round.pair_offset, nj, stream);
            exclusive_scan_i32(ws.round, ws.round.next_job_count, ws.round.next_job_offset, nj, stream);
            int32_t last_pair_offset = copy_last_i32(ws.round.pair_offset, nj, stream);
            int32_t last_pair_count = copy_last_i32(ws.round.pair_count, nj, stream);
            int64_t num_pairs = static_cast<int64_t>(last_pair_offset) + static_cast<int64_t>(last_pair_count);
            int32_t last_next_offset = copy_last_i32(ws.round.next_job_offset, nj, stream);
            int32_t last_next_count = copy_last_i32(ws.round.next_job_count, nj, stream);
            int64_t num_next_jobs = static_cast<int64_t>(last_next_offset) + static_cast<int64_t>(last_next_count);
            ensure_dda_job_queue_capacity(*next, num_next_jobs);
            next->size = num_next_jobs;
            ResultBuffer round_result = make_result_buffer(num_pairs);
            kernel_emit_dda_jobs<<<ceil_div_i64(nj, kDefaultBlockSize), kDefaultBlockSize, 0, stream>>>(*curr, ws.edge_desc, grid_min, grid_max, chunk_steps, ws.round.pair_offset, ws.round.next_job_offset, round_result, *next);
            VOX_CUDA_CHECK(cudaGetLastError());
            if (num_pairs > 0) ws.result_rounds.push_back(round_result);
            else release_result_buffer(round_result);
            std::swap(curr, next);
        }
        gathered = gather_result_rounds(ws.result_rounds, stream);
        release_workspace(ws);
        return dedup_pairs(to_primitive_pair(std::move(gathered)), stream);
    } catch (...) {
   
        release_workspace(ws);
        throw;
    }
}

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
    cudaStream_t stream) {
    BoundaryQEFResult out;
    out.size = num_voxels;
    out.qefs.allocate(num_voxels);
    out.qefs.clear_async(stream);
    if (num_voxels == 0 || num_boundaries == 0) return out;
    if (boundaries == nullptr || voxels == nullptr) throw std::invalid_argument("null boundary_qef inputs");
    fdg_gpu::DeviceBuffer<float> boundary_vertices(2 * num_boundaries * 3);
    fdg_gpu::DeviceBuffer<int32_t> boundary_edges(num_boundaries * 2);
    constexpr int kBlock = 128;
    copy_boundaries_to_vertices_kernel<<<fdg_gpu::ceil_div_i64(2 * num_boundaries, kBlock), kBlock, 0, stream>>>(boundaries, num_boundaries, boundary_vertices.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "copy_boundaries_to_vertices_kernel");
    build_synth_edges_kernel<<<fdg_gpu::ceil_div_i64(num_boundaries, kBlock), kBlock, 0, stream>>>(num_boundaries, boundary_edges.data());
    fdg_gpu::throw_cuda_error(cudaGetLastError(), "build_synth_edges_kernel");
    auto raw_pairs = voxel_traverse_edge_dda_gpu(boundary_vertices.data(), 2 * num_boundaries, boundary_edges.data(), num_boundaries, voxel_size, grid_min, grid_max, chunk_steps, stream);
    auto lookup = build_surface_lookup(voxels, num_voxels, grid_min, grid_max, stream);
    auto pair_keys = map_and_unique_edge_pairs(raw_pairs, lookup, grid_min, grid_max, stream);
    auto contrib = build_boundary_contrib_stream(pair_keys, boundary_vertices.data(), boundary_edges.data(), boundary_weight, stream);
    return reduce_boundary_contribs(std::move(contrib), num_voxels, stream);
}

} // namespace edge_dda
