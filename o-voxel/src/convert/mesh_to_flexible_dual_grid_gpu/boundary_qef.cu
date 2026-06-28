#include "../api.h"

#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

// Boundary QEFs are accumulated by walking each boundary segment through the
// voxel grid. A thread visits the voxels crossed by one segment, looks them up
// in the active-brick structure built by intersect_qef_cuda, and adds directly
// into the running qefs tensor.
namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;

        __device__ __forceinline__ void add_boundary_voxel(
            int x,
            int y,
            int z,
            GridSpec grid,
            BrickLookup lookup,
            SymQEF10 qef,
            int &cached_bx,
            int &cached_by,
            int &cached_bz,
            const uint32_t *&cached_bits,
            int64_t &cached_base,
            bool &cached_found,
            float *out_qefs)
        {
            // Map a voxel coordinate to its compact row and add one QEF. The
            // brick cache avoids repeating the hash lookup while a segment stays
            // within the same 8x8x8 brick.
            if (x < grid.grid_min.x || x >= grid.grid_max.x)
                return;
            if (y < grid.grid_min.y || y >= grid.grid_max.y)
                return;
            if (z < grid.grid_min.z || z >= grid.grid_max.z)
                return;

            const int rx = x - grid.grid_min.x;
            const int ry = y - grid.grid_min.y;
            const int rz = z - grid.grid_min.z;
            const int bx = rx / kBrickSize;
            const int by = ry / kBrickSize;
            const int bz = rz / kBrickSize;
            const int lx = rx - bx * kBrickSize;
            const int ly = ry - by * kBrickSize;
            const int lz = rz - bz * kBrickSize;
            const int local_id = lx + kBrickSize * (ly + kBrickSize * lz);

            // Boundary DDA often visits many consecutive voxels in the same
            // brick. Reuse the previous lookup until the brick coordinate
            // changes.
            if (bx != cached_bx || by != cached_by || bz != cached_bz)
            {
                cached_bx = bx;
                cached_by = by;
                cached_bz = bz;
                cached_found = lookup_brick_bits_and_base(bx, by, bz, grid, lookup, &cached_bits, &cached_base);
            }
            if (!cached_found)
                return;

            const int word = local_id / 32;
            const int bit = local_id - word * 32;
            if ((cached_bits[word] & (1u << bit)) == 0)
                return;

            int rank = 0;
            for (int i = 0; i < word; ++i)
                rank += __popc(cached_bits[i]);
            const uint32_t mask = bit == 0 ? 0u : ((1u << bit) - 1u);
            rank += __popc(cached_bits[word] & mask);

            // Compact row inside qefs is the brick's base row plus the number
            // of active local bits before this voxel.
            float *dst = out_qefs + 10 * (cached_base + rank);
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

        __global__ void accumulate_boundary_qef_kernel(
            const float *__restrict__ boundaries,
            int64_t num_boundaries,
            GridSpec grid,
            float boundary_weight,
            BrickLookup lookup,
            float *__restrict__ out_qefs)
        {
            // One thread handles one boundary segment and advances through grid
            // cells in DDA order until the segment length is reached.
            const int64_t eid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (eid >= num_boundaries)
                return;

            const float *seg = boundaries + 6 * eid;
            const float3 p0 = make_float3(seg[0], seg[1], seg[2]);
            const float3 p1 = make_float3(seg[3], seg[4], seg[5]);

            const double dx = static_cast<double>(p1.x) - static_cast<double>(p0.x);
            const double dy = static_cast<double>(p1.y) - static_cast<double>(p0.y);
            const double dz = static_cast<double>(p1.z) - static_cast<double>(p0.z);
            const double segment_length = sqrt(dx * dx + dy * dy + dz * dz);
            if (segment_length < 1e-6)
                return;

            const double dir_x = dx / segment_length;
            const double dir_y = dy / segment_length;
            const double dir_z = dz / segment_length;
            // Squared distance to a line through p0 with unit direction d:
            // ||(I - d d^T)(x - p0)||^2. The symmetric matrix below stores
            // A = I - d d^T, b = -A*p0, c = p0^T*A*p0 in QEF form.
            const float a00 = 1.0f - static_cast<float>(dir_x * dir_x);
            const float a01 = -static_cast<float>(dir_x * dir_y);
            const float a02 = -static_cast<float>(dir_x * dir_z);
            const float a11 = 1.0f - static_cast<float>(dir_y * dir_y);
            const float a12 = -static_cast<float>(dir_y * dir_z);
            const float a22 = 1.0f - static_cast<float>(dir_z * dir_z);
            const float b0 = -(a00 * p0.x + a01 * p0.y + a02 * p0.z);
            const float b1 = -(a01 * p0.x + a11 * p0.y + a12 * p0.z);
            const float b2 = -(a02 * p0.x + a12 * p0.y + a22 * p0.z);
            const float av0_x = a00 * p0.x + a01 * p0.y + a02 * p0.z;
            const float av0_y = a01 * p0.x + a11 * p0.y + a12 * p0.z;
            const float av0_z = a02 * p0.x + a12 * p0.y + a22 * p0.z;
            const float c = p0.x * av0_x + p0.y * av0_y + p0.z * av0_z;
            const SymQEF10 qef{
                boundary_weight * a00,
                boundary_weight * a01,
                boundary_weight * a02,
                boundary_weight * b0,
                boundary_weight * a11,
                boundary_weight * a12,
                boundary_weight * b1,
                boundary_weight * a22,
                boundary_weight * b2,
                boundary_weight * c,
            };
            const int step_x = dir_x > 0.0 ? 1 : -1;
            const int step_y = dir_y > 0.0 ? 1 : -1;
            const int step_z = dir_z > 0.0 ? 1 : -1;
            int cur_x = static_cast<int>(floorf(p0.x / grid.voxel_size.x));
            int cur_y = static_cast<int>(floorf(p0.y / grid.voxel_size.y));
            int cur_z = static_cast<int>(floorf(p0.z / grid.voxel_size.z));

            // tmax_* is the distance along the segment to the next grid plane on
            // that axis. tdelta_* is the distance between later grid-plane
            // crossings on the same axis.
            double tmax_x;
            double tmax_y;
            double tmax_z;
            double tdelta_x;
            double tdelta_y;
            double tdelta_z;

            if (dir_x == 0.0)
            {
                tmax_x = CUDART_INF;
                tdelta_x = CUDART_INF;
            }
            else
            {
                const float border = grid.voxel_size.x * static_cast<float>(cur_x + (step_x > 0 ? 1 : 0));
                tmax_x = static_cast<double>(border - p0.x) / dir_x;
                tdelta_x = static_cast<double>(grid.voxel_size.x) / fabs(dir_x);
            }
            if (dir_y == 0.0)
            {
                tmax_y = CUDART_INF;
                tdelta_y = CUDART_INF;
            }
            else
            {
                const float border = grid.voxel_size.y * static_cast<float>(cur_y + (step_y > 0 ? 1 : 0));
                tmax_y = static_cast<double>(border - p0.y) / dir_y;
                tdelta_y = static_cast<double>(grid.voxel_size.y) / fabs(dir_y);
            }
            if (dir_z == 0.0)
            {
                tmax_z = CUDART_INF;
                tdelta_z = CUDART_INF;
            }
            else
            {
                const float border = grid.voxel_size.z * static_cast<float>(cur_z + (step_z > 0 ? 1 : 0));
                tmax_z = static_cast<double>(border - p0.z) / dir_z;
                tdelta_z = static_cast<double>(grid.voxel_size.z) / fabs(dir_z);
            }

            int cached_bx = -1;
            int cached_by = -1;
            int cached_bz = -1;
            const uint32_t *cached_bits = nullptr;
            int64_t cached_base = 0;
            bool cached_found = false;

            add_boundary_voxel(
                cur_x,
                cur_y,
                cur_z,
                grid,
                lookup,
                qef,
                cached_bx,
                cached_by,
                cached_bz,
                cached_bits,
                cached_base,
                cached_found,
                out_qefs);

            while (true)
            {
                int axis;
                // Advance to the nearest next grid plane, visit the voxel on the
                // other side, and stop once that crossing would be past p1.
                if (tmax_x < tmax_y)
                    axis = (tmax_x < tmax_z) ? 0 : 2;
                else
                    axis = (tmax_y < tmax_z) ? 1 : 2;

                if (axis == 0 && tmax_x > segment_length)
                    break;
                if (axis == 1 && tmax_y > segment_length)
                    break;
                if (axis == 2 && tmax_z > segment_length)
                    break;

                if (axis == 0)
                {
                    cur_x += step_x;
                    tmax_x += tdelta_x;
                }
                else if (axis == 1)
                {
                    cur_y += step_y;
                    tmax_y += tdelta_y;
                }
                else
                {
                    cur_z += step_z;
                    tmax_z += tdelta_z;
                }

                add_boundary_voxel(
                    cur_x,
                    cur_y,
                    cur_z,
                    grid,
                    lookup,
                    qef,
                    cached_bx,
                    cached_by,
                    cached_bz,
                    cached_bits,
                    cached_base,
                    cached_found,
                    out_qefs);
            }
        }

    } // namespace

    torch::Tensor boundary_qef_cuda(
        const torch::Tensor &boundaries,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range,
        float boundary_weight,
        const torch::Tensor &voxels,
        const torch::Tensor &qefs,
        const torch::Tensor &brick_hash_keys,
        const torch::Tensor &brick_hash_vals,
        const torch::Tensor &brick_bits,
        const torch::Tensor &brick_base)
    {
        // qefs is an in-place accumulator. boundary_weight is already folded
        // into each segment QEF before the atomic adds.
        TORCH_CHECK(boundaries.is_cuda(), "boundaries must be a CUDA tensor");
        TORCH_CHECK(voxels.is_cuda(), "voxels must be a CUDA tensor");
        static_assert(sizeof(SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");

        const c10::cuda::CUDAGuard guard(boundaries.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(boundaries.get_device()).stream();
        const int64_t num_boundaries = boundaries.size(0);
        const int64_t num_voxels = voxels.size(0);
        if (num_boundaries == 0 || num_voxels == 0 || boundary_weight <= 0.0f)
            return qefs;

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

        const int blocks = static_cast<int>((num_boundaries + kThreads - 1) / kThreads);
        accumulate_boundary_qef_kernel<<<blocks, kThreads, 0, stream>>>(
            boundaries.data_ptr<float>(),
            num_boundaries,
            grid,
            boundary_weight,
            lookup,
            qefs.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return qefs;
    }

} // namespace o_voxel::fdg
