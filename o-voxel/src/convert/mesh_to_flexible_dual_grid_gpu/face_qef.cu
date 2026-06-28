#include "../api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>

// Face QEFs are accumulated as a triangle-brick task stream. Each task checks
// one triangle against one active brick-sized region, scans only voxels that are
// already active in that brick, and adds face_weight * plane_qef directly into
// the running qefs tensor.
namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;

        // A compact work item: triangle tri_id tested against brick (bx, by, bz).
        struct FaceBrickTask
        {
            int32_t tri_id;
            int32_t bx;
            int32_t by;
            int32_t bz;
        };

        // Matches CPU e0.cross(e1).normalized(); explicit RN ops prevent FMA residuals.
        __device__ __forceinline__ float3 aligned_cross_normalize3(
            float e0x,
            float e0y,
            float e0z,
            float e1x,
            float e1y,
            float e1z)
        {
            const float nx = __fsub_rn(__fmul_rn(e0y, e1z), __fmul_rn(e0z, e1y));
            const float ny = __fsub_rn(__fmul_rn(e0z, e1x), __fmul_rn(e0x, e1z));
            const float nz = __fsub_rn(__fmul_rn(e0x, e1y), __fmul_rn(e0y, e1x));
            const float n2 = __fadd_rn(
                __fadd_rn(__fmul_rn(nx, nx), __fmul_rn(ny, ny)),
                __fmul_rn(nz, nz));
            if (n2 > 0.0f)
            {
                const float len = sqrtf(n2);
                return make_float3(nx / len, ny / len, nz / len);
            }
            return make_float3(nx, ny, nz);
        }

        __global__ void count_face_brick_tasks_kernel(
            const float *__restrict__ triangles,
            int64_t num_triangles,
            GridSpec grid,
            int64_t *__restrict__ task_counts)
        {
            const int64_t tri_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (tri_id >= num_triangles)
                return;

            const float *tri = triangles + tri_id * 9;
            const float vs[3] = {grid.voxel_size.x, grid.voxel_size.y, grid.voxel_size.z};
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;

            // The first pass only counts how many active-brick-sized regions a
            // triangle's voxel bbox can overlap. CUB scan then gives each
            // triangle a compact output range for task emission.
            float min_x = tri[0] < tri[3] ? tri[0] : tri[3];
            min_x = min_x < tri[6] ? min_x : tri[6];
            float min_y = tri[1] < tri[4] ? tri[1] : tri[4];
            min_y = min_y < tri[7] ? min_y : tri[7];
            float min_z = tri[2] < tri[5] ? tri[2] : tri[5];
            min_z = min_z < tri[8] ? min_z : tri[8];
            float max_x = tri[0] > tri[3] ? tri[0] : tri[3];
            max_x = max_x > tri[6] ? max_x : tri[6];
            float max_y = tri[1] > tri[4] ? tri[1] : tri[4];
            max_y = max_y > tri[7] ? max_y : tri[7];
            float max_z = tri[2] > tri[5] ? tri[2] : tri[5];
            max_z = max_z > tri[8] ? max_z : tri[8];

            const int bb_min_x = max(static_cast<int>(min_x / vs[0]), grid_min.x);
            const int bb_min_y = max(static_cast<int>(min_y / vs[1]), grid_min.y);
            const int bb_min_z = max(static_cast<int>(min_z / vs[2]), grid_min.z);
            const int bb_max_x = min(static_cast<int>(max_x / vs[0] + 1.0f), grid_max.x);
            const int bb_max_y = min(static_cast<int>(max_y / vs[1] + 1.0f), grid_max.y);
            const int bb_max_z = min(static_cast<int>(max_z / vs[2] + 1.0f), grid_max.z);
            if (bb_max_x <= bb_min_x || bb_max_y <= bb_min_y || bb_max_z <= bb_min_z)
            {
                task_counts[tri_id] = 0;
                return;
            }

            const int bx0 = (bb_min_x - grid_min.x) / kBrickSize;
            const int by0 = (bb_min_y - grid_min.y) / kBrickSize;
            const int bz0 = (bb_min_z - grid_min.z) / kBrickSize;
            const int bx1 = (bb_max_x - 1 - grid_min.x) / kBrickSize;
            const int by1 = (bb_max_y - 1 - grid_min.y) / kBrickSize;
            const int bz1 = (bb_max_z - 1 - grid_min.z) / kBrickSize;
            // One FaceBrickTask is cheaper than asking every active voxel to
            // test every triangle. Later, the task only scans active bits inside
            // that brick.
            task_counts[tri_id] =
                static_cast<int64_t>(bx1 - bx0 + 1) *
                static_cast<int64_t>(by1 - by0 + 1) *
                static_cast<int64_t>(bz1 - bz0 + 1);
        }

        __global__ void emit_face_brick_tasks_kernel(
            const float *__restrict__ triangles,
            int64_t num_triangles,
            GridSpec grid,
            const int64_t *__restrict__ task_offsets,
            FaceBrickTask *__restrict__ tasks)
        {
            const int64_t tri_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (tri_id >= num_triangles)
                return;

            const float *tri = triangles + tri_id * 9;
            const float vs[3] = {grid.voxel_size.x, grid.voxel_size.y, grid.voxel_size.z};
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;

            float min_x = tri[0] < tri[3] ? tri[0] : tri[3];
            min_x = min_x < tri[6] ? min_x : tri[6];
            float min_y = tri[1] < tri[4] ? tri[1] : tri[4];
            min_y = min_y < tri[7] ? min_y : tri[7];
            float min_z = tri[2] < tri[5] ? tri[2] : tri[5];
            min_z = min_z < tri[8] ? min_z : tri[8];
            float max_x = tri[0] > tri[3] ? tri[0] : tri[3];
            max_x = max_x > tri[6] ? max_x : tri[6];
            float max_y = tri[1] > tri[4] ? tri[1] : tri[4];
            max_y = max_y > tri[7] ? max_y : tri[7];
            float max_z = tri[2] > tri[5] ? tri[2] : tri[5];
            max_z = max_z > tri[8] ? max_z : tri[8];

            const int bb_min_x = max(static_cast<int>(min_x / vs[0]), grid_min.x);
            const int bb_min_y = max(static_cast<int>(min_y / vs[1]), grid_min.y);
            const int bb_min_z = max(static_cast<int>(min_z / vs[2]), grid_min.z);
            const int bb_max_x = min(static_cast<int>(max_x / vs[0] + 1.0f), grid_max.x);
            const int bb_max_y = min(static_cast<int>(max_y / vs[1] + 1.0f), grid_max.y);
            const int bb_max_z = min(static_cast<int>(max_z / vs[2] + 1.0f), grid_max.z);
            if (bb_max_x <= bb_min_x || bb_max_y <= bb_min_y || bb_max_z <= bb_min_z)
                return;

            const int bx0 = (bb_min_x - grid_min.x) / kBrickSize;
            const int by0 = (bb_min_y - grid_min.y) / kBrickSize;
            const int bz0 = (bb_min_z - grid_min.z) / kBrickSize;
            const int bx1 = (bb_max_x - 1 - grid_min.x) / kBrickSize;
            const int by1 = (bb_max_y - 1 - grid_min.y) / kBrickSize;
            const int bz1 = (bb_max_z - 1 - grid_min.z) / kBrickSize;

            int64_t out = task_offsets[tri_id];
            // The task grid is triangle bbox clipped to brick coordinates. It is
            // intentionally conservative; accumulate_face_qef_kernel performs
            // the triangle/voxel-box overlap test.
            for (int bz = bz0; bz <= bz1; ++bz)
                for (int by = by0; by <= by1; ++by)
                    for (int bx = bx0; bx <= bx1; ++bx)
                        tasks[out++] = FaceBrickTask{static_cast<int32_t>(tri_id), bx, by, bz};
        }

        __global__ void accumulate_face_qef_kernel(
            const FaceBrickTask *__restrict__ tasks,
            int64_t num_tasks,
            const float *__restrict__ triangles,
            GridSpec grid,
            BrickLookup lookup,
            float face_weight,
            float *__restrict__ out_qefs)
        {
            const int64_t task_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (task_id >= num_tasks)
                return;

            const FaceBrickTask task = tasks[task_id];
            const uint32_t *bits;
            int64_t base;
            // If the triangle bbox brick was never activated by intersect_qef,
            // it has no output rows and the whole task can be skipped.
            if (!lookup_brick_bits_and_base(task.bx, task.by, task.bz, grid, lookup, &bits, &base))
                return;

            const float *tri = triangles + static_cast<int64_t>(task.tri_id) * 9;
            const float3 v0 = make_float3(tri[0], tri[1], tri[2]);
            const float3 v1 = make_float3(tri[3], tri[4], tri[5]);
            const float3 v2 = make_float3(tri[6], tri[7], tri[8]);
            const float vs[3] = {grid.voxel_size.x, grid.voxel_size.y, grid.voxel_size.z};
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;

            const float e0x = v1.x - v0.x;
            const float e0y = v1.y - v0.y;
            const float e0z = v1.z - v0.z;
            const float e1x = v2.x - v1.x;
            const float e1y = v2.y - v1.y;
            const float e1z = v2.z - v1.z;
            const float e2x = v0.x - v2.x;
            const float e2y = v0.y - v2.y;
            const float e2z = v0.z - v2.z;

            const float3 n = aligned_cross_normalize3(e0x, e0y, e0z, e1x, e1y, e1z);
            const float nx = n.x;
            const float ny = n.y;
            const float nz = n.z;
            // Face QEF is the squared distance to the triangle plane, scaled
            // once here before all matching voxels receive it.
            const SymQEF10 qef = qef_scale(
                qef_from_plane(make_float4(nx, ny, nz, -(nx * v0.x + ny * v0.y + nz * v0.z))),
                face_weight);

            float min_x = v0.x < v1.x ? v0.x : v1.x;
            min_x = min_x < v2.x ? min_x : v2.x;
            float min_y = v0.y < v1.y ? v0.y : v1.y;
            min_y = min_y < v2.y ? min_y : v2.y;
            float min_z = v0.z < v1.z ? v0.z : v1.z;
            min_z = min_z < v2.z ? min_z : v2.z;
            float max_x = v0.x > v1.x ? v0.x : v1.x;
            max_x = max_x > v2.x ? max_x : v2.x;
            float max_y = v0.y > v1.y ? v0.y : v1.y;
            max_y = max_y > v2.y ? max_y : v2.y;
            float max_z = v0.z > v1.z ? v0.z : v1.z;
            max_z = max_z > v2.z ? max_z : v2.z;

            const int bb_min_x = max(static_cast<int>(min_x / vs[0]), grid_min.x);
            const int bb_min_y = max(static_cast<int>(min_y / vs[1]), grid_min.y);
            const int bb_min_z = max(static_cast<int>(min_z / vs[2]), grid_min.z);
            const int bb_max_x = min(static_cast<int>(max_x / vs[0] + 1.0f), grid_max.x);
            const int bb_max_y = min(static_cast<int>(max_y / vs[1] + 1.0f), grid_max.y);
            const int bb_max_z = min(static_cast<int>(max_z / vs[2] + 1.0f), grid_max.z);

            const float c_x = nx > 0.0f ? vs[0] : 0.0f;
            const float c_y = ny > 0.0f ? vs[1] : 0.0f;
            const float c_z = nz > 0.0f ? vs[2] : 0.0f;
            const float d1 = nx * (c_x - v0.x) + ny * (c_y - v0.y) + nz * (c_z - v0.z);
            const float d2 = nx * (vs[0] - c_x - v0.x) + ny * (vs[1] - c_y - v0.y) + nz * (vs[2] - c_z - v0.z);

            // Plane slab test: choose the two voxel-box corners that are most
            // separated along the triangle normal. If both signed distances have
            // the same sign, the box cannot cross the triangle plane.
            const int mul_xy = nz < 0.0f ? -1 : 1;
            const float n_xy_e0_x = -mul_xy * e0y;
            const float n_xy_e0_y = mul_xy * e0x;
            const float n_xy_e1_x = -mul_xy * e1y;
            const float n_xy_e1_y = mul_xy * e1x;
            const float n_xy_e2_x = -mul_xy * e2y;
            const float n_xy_e2_y = mul_xy * e2x;
            const float d_xy_e0 = -(n_xy_e0_x * v0.x + n_xy_e0_y * v0.y) + fmaxf(n_xy_e0_x, 0.0f) * vs[0] + fmaxf(n_xy_e0_y, 0.0f) * vs[1];
            const float d_xy_e1 = -(n_xy_e1_x * v1.x + n_xy_e1_y * v1.y) + fmaxf(n_xy_e1_x, 0.0f) * vs[0] + fmaxf(n_xy_e1_y, 0.0f) * vs[1];
            const float d_xy_e2 = -(n_xy_e2_x * v2.x + n_xy_e2_y * v2.y) + fmaxf(n_xy_e2_x, 0.0f) * vs[0] + fmaxf(n_xy_e2_y, 0.0f) * vs[1];

            const int mul_yz = nx < 0.0f ? -1 : 1;
            const float n_yz_e0_x = -mul_yz * e0z;
            const float n_yz_e0_y = mul_yz * e0y;
            const float n_yz_e1_x = -mul_yz * e1z;
            const float n_yz_e1_y = mul_yz * e1y;
            const float n_yz_e2_x = -mul_yz * e2z;
            const float n_yz_e2_y = mul_yz * e2y;
            const float d_yz_e0 = -(n_yz_e0_x * v0.y + n_yz_e0_y * v0.z) + fmaxf(n_yz_e0_x, 0.0f) * vs[1] + fmaxf(n_yz_e0_y, 0.0f) * vs[2];
            const float d_yz_e1 = -(n_yz_e1_x * v1.y + n_yz_e1_y * v1.z) + fmaxf(n_yz_e1_x, 0.0f) * vs[1] + fmaxf(n_yz_e1_y, 0.0f) * vs[2];
            const float d_yz_e2 = -(n_yz_e2_x * v2.y + n_yz_e2_y * v2.z) + fmaxf(n_yz_e2_x, 0.0f) * vs[1] + fmaxf(n_yz_e2_y, 0.0f) * vs[2];

            const int mul_zx = ny < 0.0f ? -1 : 1;
            const float n_zx_e0_x = -mul_zx * e0x;
            const float n_zx_e0_y = mul_zx * e0z;
            const float n_zx_e1_x = -mul_zx * e1x;
            const float n_zx_e1_y = mul_zx * e1z;
            const float n_zx_e2_x = -mul_zx * e2x;
            const float n_zx_e2_y = mul_zx * e2z;
            const float d_zx_e0 = -(n_zx_e0_x * v0.z + n_zx_e0_y * v0.x) + fmaxf(n_zx_e0_x, 0.0f) * vs[2] + fmaxf(n_zx_e0_y, 0.0f) * vs[0];
            const float d_zx_e1 = -(n_zx_e1_x * v1.z + n_zx_e1_y * v1.x) + fmaxf(n_zx_e1_x, 0.0f) * vs[2] + fmaxf(n_zx_e1_y, 0.0f) * vs[0];
            const float d_zx_e2 = -(n_zx_e2_x * v2.z + n_zx_e2_y * v2.x) + fmaxf(n_zx_e2_x, 0.0f) * vs[2] + fmaxf(n_zx_e2_y, 0.0f) * vs[0];

            // The xy/yz/zx edge functions are projected half-space tests. The
            // fmax terms move the sampled box corner to the side most favorable
            // to overlap, making this a voxel-box vs triangle test rather than
            // a point-in-triangle test.
            int rank_before_word = 0;
            for (int word = 0; word < kBrickBitWords; ++word)
            {
                const uint32_t word_bits = bits[word];
                uint32_t active = word_bits;
                int rank_in_word = 0;
                while (active != 0)
                {
                    const int bit = __ffs(active) - 1;
                    const int local_id = word * 32 + bit;
                    const int local_rank = rank_before_word + rank_in_word;
                    // local_rank is the number of active bits before this voxel
                    // in the brick, so base + local_rank is the compact qef row.
                    const int lz = local_id / (kBrickSize * kBrickSize);
                    const int rem = local_id - lz * kBrickSize * kBrickSize;
                    const int ly = rem / kBrickSize;
                    const int lx = rem - ly * kBrickSize;
                    const int x = grid_min.x + task.bx * kBrickSize + lx;
                    const int y = grid_min.y + task.by * kBrickSize + ly;
                    const int z = grid_min.z + task.bz * kBrickSize + lz;

                    if (x >= bb_min_x && x < bb_max_x &&
                        y >= bb_min_y && y < bb_max_y &&
                        z >= bb_min_z && z < bb_max_z)
                    {
                        const float px = x * vs[0];
                        const float py = y * vs[1];
                        const float pz = z * vs[2];
                        const float n_dot_p = nx * px + ny * py + nz * pz;
                        bool hit = true;
                        // Reject if the voxel box is entirely on one side of
                        // the triangle plane, or outside any projected edge
                        // half-space.
                        if ((n_dot_p + d1) * (n_dot_p + d2) > 0.0f)
                            hit = false;
                        if (n_xy_e0_x * px + n_xy_e0_y * py + d_xy_e0 < 0.0f)
                            hit = false;
                        if (n_xy_e1_x * px + n_xy_e1_y * py + d_xy_e1 < 0.0f)
                            hit = false;
                        if (n_xy_e2_x * px + n_xy_e2_y * py + d_xy_e2 < 0.0f)
                            hit = false;
                        if (n_yz_e0_x * py + n_yz_e0_y * pz + d_yz_e0 < 0.0f)
                            hit = false;
                        if (n_yz_e1_x * py + n_yz_e1_y * pz + d_yz_e1 < 0.0f)
                            hit = false;
                        if (n_yz_e2_x * py + n_yz_e2_y * pz + d_yz_e2 < 0.0f)
                            hit = false;
                        if (n_zx_e0_x * pz + n_zx_e0_y * px + d_zx_e0 < 0.0f)
                            hit = false;
                        if (n_zx_e1_x * pz + n_zx_e1_y * px + d_zx_e1 < 0.0f)
                            hit = false;
                        if (n_zx_e2_x * pz + n_zx_e2_y * px + d_zx_e2 < 0.0f)
                            hit = false;
                        if (hit)
                        {
                            // Different triangle-brick tasks can contribute to
                            // the same voxel, so the total QEF is accumulated
                            // with atomic adds.
                            float *dst = out_qefs + 10 * (base + local_rank);
                            atomicAdd(dst + 0, qef.q00);
                            atomicAdd(dst + 1, qef.q01);
                            atomicAdd(dst + 2, qef.q02);
                            atomicAdd(dst + 3, qef.q03);
                            atomicAdd(dst + 4, qef.q11);
                            atomicAdd(dst + 5, qef.q12);
                            atomicAdd(dst + 6, qef.q13);
                            atomicAdd(dst + 7, qef.q22);
                            atomicAdd(dst + 8, qef.q23);
                            atomicAdd(dst + 9, qef.q33);
                        }
                    }

                    active &= active - 1u;
                    ++rank_in_word;
                }
                rank_before_word += __popc(word_bits);
            }
        }

    } // namespace

    torch::Tensor face_qef_cuda(
        const torch::Tensor &triangles,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range,
        const torch::Tensor &voxels,
        const torch::Tensor &qefs,
        float face_weight,
        const torch::Tensor &brick_hash_keys,
        const torch::Tensor &brick_hash_vals,
        const torch::Tensor &brick_bits,
        const torch::Tensor &brick_base)
    {
        // qefs is an in-place accumulator. No separate face-QEF buffer is
        // allocated; the caller should pass the running total QEF tensor.
        TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");
        TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");

        const c10::cuda::CUDAGuard guard(triangles.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(triangles.get_device()).stream();
        const torch::Device device = triangles.device();
        const int64_t num_triangles = triangles.size(0);
        const int64_t num_voxels = voxels.size(0);
        if (num_triangles == 0 || num_voxels == 0)
            return qefs;

        const auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
        const auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
        const auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);
        const GridSpec grid{
            float3{voxel_size[0], voxel_size[1], voxel_size[2]},
            Int3{
                static_cast<int32_t>(grid_range[0]),
                static_cast<int32_t>(grid_range[1]),
                static_cast<int32_t>(grid_range[2])},
            Int3{
                static_cast<int32_t>(grid_range[3]),
                static_cast<int32_t>(grid_range[4]),
                static_cast<int32_t>(grid_range[5])},
        };
        const BrickLookup lookup{
            brick_hash_keys.data_ptr<uint64_t>(),
            brick_hash_vals.data_ptr<uint32_t>(),
            brick_bits.data_ptr<uint32_t>(),
            brick_base.data_ptr<int64_t>(),
            static_cast<uint64_t>(brick_hash_keys.numel()),
        };

        auto task_counts = torch::empty({num_triangles}, opts_i64);
        auto task_offsets = torch::empty({num_triangles}, opts_i64);
        int blocks = static_cast<int>((num_triangles + kThreads - 1) / kThreads);
        // count -> exclusive scan -> emit keeps task storage exact while still
        // letting every triangle count its brick work independently.
        count_face_brick_tasks_kernel<<<blocks, kThreads, 0, stream>>>(
            triangles.data_ptr<float>(),
            num_triangles,
            grid,
            task_counts.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        size_t temp_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            temp_bytes,
            task_counts.data_ptr<int64_t>(),
            task_offsets.data_ptr<int64_t>(),
            static_cast<int>(num_triangles),
            stream));
        auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, opts_u8);
        C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            temp.data_ptr<uint8_t>(),
            temp_bytes,
            task_counts.data_ptr<int64_t>(),
            task_offsets.data_ptr<int64_t>(),
            static_cast<int>(num_triangles),
            stream));

        int64_t tail[2] = {0, 0};
        C10_CUDA_CHECK(cudaMemcpyAsync(tail, task_counts.data_ptr<int64_t>() + num_triangles - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(tail + 1, task_offsets.data_ptr<int64_t>() + num_triangles - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        const int64_t num_tasks = tail[0] + tail[1];
        if (num_tasks == 0)
            return qefs;

        auto tasks = torch::empty({num_tasks, 4}, opts_i32);
        emit_face_brick_tasks_kernel<<<blocks, kThreads, 0, stream>>>(
            triangles.data_ptr<float>(),
            num_triangles,
            grid,
            task_offsets.data_ptr<int64_t>(),
            reinterpret_cast<FaceBrickTask *>(tasks.data_ptr<int32_t>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        blocks = static_cast<int>((num_tasks + kThreads - 1) / kThreads);
        accumulate_face_qef_kernel<<<blocks, kThreads, 0, stream>>>(
            reinterpret_cast<const FaceBrickTask *>(tasks.data_ptr<int32_t>()),
            num_tasks,
            triangles.data_ptr<float>(),
            grid,
            lookup,
            face_weight,
            qefs.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return qefs;
    }

} // namespace o_voxel::fdg
