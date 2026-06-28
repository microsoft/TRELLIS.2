#include "../api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

// Triangle/voxel intersection is expressed as a stream of small scan tasks.
// A large triangle can cover many grid cells, so each axis projection is split
// into 16x16 tiles; GPU threads then process tiles rather than whole triangles.
// The occupancy pass first marks active voxels in brick bitsets and only later
// compacts those bits into the final voxel rows used by QEF arrays.
namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;
        constexpr int kTileU = 16;
        constexpr int kTileV = 16;

        // One triangle-axis tile. u/v are the two scan axes for the chosen axis,
        // and the half-open range [u0, u1) x [v0, v1) bounds the work item.
        struct ScanTask
        {
            int32_t tri_id;
            int32_t axis;
            int32_t u0;
            int32_t u1;
            int32_t v0;
            int32_t v1;
        };

        // Tensor-owned result of the shared occupancy build. intersect_occ_cuda
        // returns only voxels; intersect_qef_cuda also reuses tasks and brick
        // lookup tensors to accumulate intersection QEFs.
        struct IntersectionOccupancy
        {
            torch::Tensor tasks;
            torch::Tensor hash_keys;
            torch::Tensor hash_vals;
            torch::Tensor brick_coords;
            torch::Tensor brick_bits;
            torch::Tensor brick_base;
            torch::Tensor voxels;
            torch::Tensor overflow_flag;
            int64_t num_tasks = 0;
            int64_t num_bricks = 0;
            int64_t num_voxels = 0;
            uint64_t hash_capacity = 0;
        };

        __host__ __device__ __forceinline__ int div_up_i32(int n, int d)
        {
            return (n + d - 1) / d;
        }

        __host__ __device__ __forceinline__ int64_t div_up_i64(int64_t n, int64_t d)
        {
            return (n + d - 1) / d;
        }

        __device__ __forceinline__ int clamp_int(int v, int lo, int hi)
        {
            return max(min(v, hi), lo);
        }

        __device__ __forceinline__ uint64_t mix64(uint64_t x)
        {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;
            return x;
        }

        __device__ bool compute_scan_bbox(
            const float *tri,
            int axis,
            GridSpec grid,
            int &u0,
            int &u1,
            int &v0,
            int &v1)
        {
            // For one depth axis, scan the triangle in the other two axes. The
            // returned u/v box is conservative because each scan event later
            // touches a 2x2 voxel neighborhood in the projection plane.
            const int ax0 = (axis + 1) % 3;
            const int ax1 = (axis + 2) % 3;
            const float3 voxel_size = grid.voxel_size;
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;
            const float vs[3] = {voxel_size.x, voxel_size.y, voxel_size.z};
            float min_u = tri[ax0];
            float max_u = tri[ax0];
            float min_v = tri[ax1];
            float max_v = tri[ax1];
            for (int i = 1; i < 3; ++i)
            {
                const float u = tri[3 * i + ax0];
                const float v = tri[3 * i + ax1];
                min_u = fminf(min_u, u);
                max_u = fmaxf(max_u, u);
                min_v = fminf(min_v, v);
                max_v = fmaxf(max_v, v);
            }

            // u is expanded by one cell on the low side and two on the high
            // side to preserve scanline event coverage near triangle edges and
            // voxel boundaries.
            u0 = clamp_int(static_cast<int>(min_u / vs[ax0]) - 1, grid_min[ax0], grid_max[ax0] - 1);
            u1 = clamp_int(static_cast<int>(max_u / vs[ax0]) + 2, grid_min[ax0], grid_max[ax0] - 1);
            v0 = clamp_int(static_cast<int>(min_v / vs[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            v1 = clamp_int(static_cast<int>(max_v / vs[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            return u1 > u0 && v1 > v0;
        }

        __device__ int64_t task_brick_bound(
            const float *tri,
            const ScanTask &task,
            GridSpec grid)
        {
            // This is an allocation bound, not the exact output count. For the
            // tile's u/v range, estimate the depth interval covered by the
            // triangle plane, then count all bricks overlapped by that 3D box.
            const int axis = task.axis;
            const int ax0 = (axis + 1) % 3;
            const int ax1 = (axis + 2) % 3;
            const int ax2 = axis;
            const float3 voxel_size = grid.voxel_size;
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;
            const float vs[3] = {voxel_size.x, voxel_size.y, voxel_size.z};
            int lo[3];
            int hi[3];
            lo[ax0] = max(task.u0, grid_min[ax0]);
            hi[ax0] = min(task.u1 + 1, grid_max[ax0]);
            lo[ax1] = max(task.v0, grid_min[ax1]);
            hi[ax1] = min(task.v1 + 1, grid_max[ax1]);

            const double u0 = tri[ax0];
            const double v0 = tri[ax1];
            const double z0 = tri[ax2];
            const double u1 = tri[3 + ax0];
            const double v1 = tri[3 + ax1];
            const double z1 = tri[3 + ax2];
            const double u2 = tri[6 + ax0];
            const double v2 = tri[6 + ax1];
            const double z2 = tri[6 + ax2];
            const double denom = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
            double z_min = fmin(fmin(z0, z1), z2);
            double z_max = fmax(fmax(z0, z1), z2);
            if (fabs(denom) > 1e-20)
            {
                // z = a*u + b*v + c is the triangle plane written over the
                // current projection. Evaluating the four tile corners gives a
                // conservative depth interval for all scan events in this tile.
                const double a = ((z1 - z0) * (v2 - v0) - (z2 - z0) * (v1 - v0)) / denom;
                const double b = ((u1 - u0) * (z2 - z0) - (u2 - u0) * (z1 - z0)) / denom;
                const double c = z0 - a * u0 - b * v0;
                const double ru0 = static_cast<double>(task.u0) * vs[ax0];
                const double ru1 = static_cast<double>(task.u1 + 1) * vs[ax0];
                const double rv0 = static_cast<double>(task.v0) * vs[ax1];
                const double rv1 = static_cast<double>(task.v1 + 1) * vs[ax1];
                const double d0 = a * ru0 + b * rv0 + c;
                const double d1 = a * ru0 + b * rv1 + c;
                const double d2 = a * ru1 + b * rv0 + c;
                const double d3 = a * ru1 + b * rv1 + c;
                z_min = fmin(fmin(d0, d1), fmin(d2, d3));
                z_max = fmax(fmax(d0, d1), fmax(d2, d3));
            }

            lo[ax2] = clamp_int(static_cast<int>(z_min / vs[ax2]) - 2, grid_min[ax2], grid_max[ax2]);
            hi[ax2] = clamp_int(static_cast<int>(z_max / vs[ax2]) + 3, grid_min[ax2], grid_max[ax2]);
            if (hi[0] <= lo[0] || hi[1] <= lo[1] || hi[2] <= lo[2])
                return 0;

            // Convert the conservative voxel box into a conservative brick box.
            // Summing these bounds lets the host allocate hash/bitset storage
            // once, with overflow treated as a bug guard.
            const int64_t bx0 = (lo[0] - grid_min.x) / kBrickSize;
            const int64_t by0 = (lo[1] - grid_min.y) / kBrickSize;
            const int64_t bz0 = (lo[2] - grid_min.z) / kBrickSize;
            const int64_t bx1 = div_up_i64(hi[0] - grid_min.x, kBrickSize);
            const int64_t by1 = div_up_i64(hi[1] - grid_min.y, kBrickSize);
            const int64_t bz1 = div_up_i64(hi[2] - grid_min.z, kBrickSize);
            return (bx1 - bx0) * (by1 - by0) * (bz1 - bz0);
        }

        template <typename Emit>
        __device__ void scan_triangle_events_tiled(
            const float *tri,
            const ScanTask &task,
            GridSpec grid,
            Emit emit)
        {
            // Shared scanline generator used by occupancy and QEF passes. The
            // template callback keeps both passes on the exact same event stream.
            const int ax2 = task.axis;
            const int ax0 = (ax2 + 1) % 3;
            const int ax1 = (ax2 + 2) % 3;
            const float3 voxel_size = grid.voxel_size;
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;
            double t[3][3] = {
                {static_cast<double>(tri[ax0]), static_cast<double>(tri[ax1]), static_cast<double>(tri[ax2])},
                {static_cast<double>(tri[3 + ax0]), static_cast<double>(tri[3 + ax1]), static_cast<double>(tri[3 + ax2])},
                {static_cast<double>(tri[6 + ax0]), static_cast<double>(tri[6 + ax1]), static_cast<double>(tri[6 + ax2])},
            };
            int order[3] = {0, 1, 2};
            // Sort vertices by the scan row axis so the triangle can be scanned
            // as two monotonic halves: top->middle and middle->bottom.
            if (t[order[0]][1] > t[order[1]][1])
            {
                const int tmp = order[0];
                order[0] = order[1];
                order[1] = tmp;
            }
            if (t[order[1]][1] > t[order[2]][1])
            {
                const int tmp = order[1];
                order[1] = order[2];
                order[2] = tmp;
            }
            if (t[order[0]][1] > t[order[1]][1])
            {
                const int tmp = order[0];
                order[0] = order[1];
                order[1] = tmp;
            }

            const double *t0 = t[order[0]];
            const double *t1 = t[order[1]];
            const double *t2 = t[order[2]];
            const float vs[3] = {voxel_size.x, voxel_size.y, voxel_size.z};
            const int start = max(min(static_cast<int>(t0[1] / vs[ax1]), grid_max[ax1] - 1), grid_min[ax1]);
            const int mid = max(min(static_cast<int>(t1[1] / vs[ax1]), grid_max[ax1] - 1), grid_min[ax1]);
            const int end = max(min(static_cast<int>(t2[1] / vs[ax1]), grid_max[ax1] - 1), grid_min[ax1]);

            auto scan_half = [&](int row_start, int row_end, const double *a, const double *b, const double *c)
            {
                // For each scan row, intersect the row with two triangle edges,
                // then interpolate along the horizontal span to recover depth.
                row_start = max(row_start, task.v0);
                row_end = min(row_end, task.v1);
                for (int y_idx = row_start; y_idx < row_end; ++y_idx)
                {
                    // y and x use the high cell boundary, matching the original
                    // event placement for voxel face crossings.
                    const double y = (static_cast<double>(y_idx) + 1.0) * vs[ax1];
                    const double ab = fabs(a[1] - b[1]) < 1e-12 ? 0.0 : (y - a[1]) / (b[1] - a[1]);
                    const double ac = fabs(a[1] - c[1]) < 1e-12 ? 0.0 : (y - a[1]) / (c[1] - a[1]);
                    double t3x = (1.0 - ab) * a[0] + ab * b[0];
                    double t3z = (1.0 - ab) * a[2] + ab * b[2];
                    double t4x = (1.0 - ac) * a[0] + ac * c[0];
                    double t4z = (1.0 - ac) * a[2] + ac * c[2];
                    if (t3x > t4x)
                    {
                        double tmp = t3x;
                        t3x = t4x;
                        t4x = tmp;
                        tmp = t3z;
                        t3z = t4z;
                        t4z = tmp;
                    }

                    int line_start = max(min(static_cast<int>(t3x / vs[ax0]), grid_max[ax0] - 1), grid_min[ax0]);
                    int line_end = max(min(static_cast<int>(t4x / vs[ax0]), grid_max[ax0] - 1), grid_min[ax0]);
                    line_start = max(line_start, task.u0);
                    line_end = min(line_end, task.u1);
                    for (int x_idx = line_start; x_idx < line_end; ++x_idx)
                    {
                        const double x = (static_cast<double>(x_idx) + 1.0) * vs[ax0];
                        // alpha moves across the row segment; z is the point
                        // where the triangle plane crosses this projected event.
                        const double alpha = fabs(t4x - t3x) < 1e-12 ? 0.0 : (x - t3x) / (t4x - t3x);
                        const double z = (1.0 - alpha) * t3z + alpha * t4z;
                        const int z_idx = static_cast<int>(z / vs[ax2]);
                        if (z_idx >= grid_min[ax2] && z_idx < grid_max[ax2])
                            emit(ax0, ax1, ax2, x_idx, y_idx, z_idx, x, y, z);
                    }
                }
            };
            scan_half(start, mid, t0, t1, t2);
            scan_half(mid, end, t2, t1, t0);
        }

        __device__ __forceinline__ uint64_t voxel_to_brick(
            int x,
            int y,
            int z,
            GridSpec grid,
            Int3 &brick,
            int &local_id)
        {
            // Split one voxel coordinate into a brick key plus a local bit id.
            // The key addresses the hash table; local_id addresses the 512-bit
            // occupancy mask inside that brick.
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;
            const int rx = x - grid_min.x;
            const int ry = y - grid_min.y;
            const int rz = z - grid_min.z;
            brick.x = rx / kBrickSize;
            brick.y = ry / kBrickSize;
            brick.z = rz / kBrickSize;
            const int lx = rx - brick.x * kBrickSize;
            const int ly = ry - brick.y * kBrickSize;
            const int lz = rz - brick.z * kBrickSize;
            local_id = lx + kBrickSize * (ly + kBrickSize * lz);
            const uint64_t nbx = div_up_i64(grid_max.x - grid_min.x, kBrickSize);
            const uint64_t nby = div_up_i64(grid_max.y - grid_min.y, kBrickSize);
            return static_cast<uint64_t>(brick.x) + nbx * (static_cast<uint64_t>(brick.y) + nby * static_cast<uint64_t>(brick.z));
        }

        __device__ uint32_t get_or_create_brick(
            uint64_t key,
            Int3 brick,
            uint64_t *hash_keys,
            uint32_t *hash_vals,
            uint32_t *brick_count,
            int32_t *brick_coords,
            int32_t *overflow_flag,
            uint64_t hash_capacity,
            uint32_t max_bricks)
        {
            uint64_t slot = mix64(key) & (hash_capacity - 1);
            for (uint64_t probe = 0; probe < hash_capacity; ++probe)
            {
                // The first thread that installs the key owns brick allocation.
                // Other threads finding the same key wait until hash_vals is
                // published, then reuse the existing compact brick index.
                const uint64_t prev = atomicCAS(
                    reinterpret_cast<unsigned long long *>(hash_keys + slot),
                    static_cast<unsigned long long>(kEmptyBrickKey),
                    static_cast<unsigned long long>(key));
                if (prev == kEmptyBrickKey)
                {
                    const uint32_t idx = atomicAdd(brick_count, 1u);
                    if (idx >= max_bricks)
                    {
                        atomicExch(overflow_flag, 1);
                        __threadfence();
                        hash_vals[slot] = kOverflowBrickVal;
                        return kEmptyBrickVal;
                    }
                    brick_coords[3 * idx + 0] = brick.x;
                    brick_coords[3 * idx + 1] = brick.y;
                    brick_coords[3 * idx + 2] = brick.z;
                    __threadfence();
                    hash_vals[slot] = idx;
                    return idx;
                }
                if (prev == key)
                {
                    volatile uint32_t *val_ptr = hash_vals + slot;
                    uint32_t val = *val_ptr;
                    // key becomes visible before value; spin only on that slot
                    // until the creating thread publishes the brick index.
                    while (val == kEmptyBrickVal)
                        val = *val_ptr;
                    return val == kOverflowBrickVal ? kEmptyBrickVal : val;
                }
                slot = (slot + 1u) & (hash_capacity - 1);
            }
            atomicExch(overflow_flag, 1);
            return kEmptyBrickVal;
        }

        __device__ __forceinline__ SymQEF10 triangle_qef(const float *tri)
        {
            // Plane QEF for the triangle itself. The normal uses the same edge
            // order as the CPU path so the plane sign and d term stay aligned.
            const float e0x = tri[3] - tri[0];
            const float e0y = tri[4] - tri[1];
            const float e0z = tri[5] - tri[2];
            const float e1x = tri[6] - tri[3];
            const float e1y = tri[7] - tri[4];
            const float e1z = tri[8] - tri[5];
            float nx = e0y * e1z - e0z * e1y;
            float ny = e0z * e1x - e0x * e1z;
            float nz = e0x * e1y - e0y * e1x;
            const float inv_len = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-30f);
            nx *= inv_len;
            ny *= inv_len;
            nz *= inv_len;
            return qef_from_plane(float4{nx, ny, nz, -(nx * tri[0] + ny * tri[1] + nz * tri[2])});
        }

        __device__ __forceinline__ void atomic_add_qef(float *dst, const SymQEF10 &q)
        {
            // Multiple scan tasks can hit the same compact voxel row, so every
            // matrix coefficient is accumulated atomically.
            atomicAdd(dst + 0, q.q00);
            atomicAdd(dst + 1, q.q01);
            atomicAdd(dst + 2, q.q02);
            atomicAdd(dst + 3, q.q03);
            atomicAdd(dst + 4, q.q11);
            atomicAdd(dst + 5, q.q12);
            atomicAdd(dst + 6, q.q13);
            atomicAdd(dst + 7, q.q22);
            atomicAdd(dst + 8, q.q23);
            atomicAdd(dst + 9, q.q33);
        }

        __global__ void count_scan_tasks_kernel(
            const float *triangles,
            int64_t num_triangles,
            GridSpec grid,
            int64_t *task_counts)
        {
            const int64_t pair_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (pair_id >= num_triangles * 3)
                return;
            const int64_t tri_id = pair_id / 3;
            const int axis = static_cast<int>(pair_id - tri_id * 3);
            // Parallel unit is (triangle, depth axis). Large projected boxes
            // become many fixed-size tile tasks instead of one long thread.
            int u0, u1, v0, v1;
            if (!compute_scan_bbox(triangles + tri_id * 9, axis, grid, u0, u1, v0, v1))
            {
                task_counts[pair_id] = 0;
                return;
            }
            task_counts[pair_id] = static_cast<int64_t>(div_up_i32(u1 - u0, kTileU)) *
                                   static_cast<int64_t>(div_up_i32(v1 - v0, kTileV));
        }

        __global__ void emit_scan_tasks_and_brick_bounds_kernel(
            const float *triangles,
            int64_t num_triangles,
            GridSpec grid,
            const int64_t *task_offsets,
            ScanTask *tasks,
            int64_t *task_brick_bounds)
        {
            const int64_t pair_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (pair_id >= num_triangles * 3)
                return;
            const int64_t tri_id = pair_id / 3;
            const int axis = static_cast<int>(pair_id - tri_id * 3);
            int u0, u1, v0, v1;
            if (!compute_scan_bbox(triangles + tri_id * 9, axis, grid, u0, u1, v0, v1))
                return;

            int64_t out = task_offsets[pair_id];
            // task_offsets is the CUB exclusive scan of per-pair task counts.
            // Each thread owns a disjoint output range and writes all tiles for
            // its (triangle, axis) pair.
            for (int v = v0; v < v1; v += kTileV)
            {
                for (int u = u0; u < u1; u += kTileU)
                {
                    ScanTask task{
                        static_cast<int32_t>(tri_id),
                        static_cast<int32_t>(axis),
                        static_cast<int32_t>(u),
                        static_cast<int32_t>(min(u + kTileU, u1)),
                        static_cast<int32_t>(v),
                        static_cast<int32_t>(min(v + kTileV, v1)),
                    };
                    tasks[out] = task;
                    task_brick_bounds[out] = task_brick_bound(triangles + tri_id * 9, task, grid);
                    ++out;
                }
            }
        }

        __global__ void mark_occupied_voxel_bits_kernel(
            const ScanTask *tasks,
            int64_t num_tasks,
            const float *triangles,
            GridSpec grid,
            uint64_t *hash_keys,
            uint32_t *hash_vals,
            uint32_t *brick_count,
            int32_t *brick_coords,
            uint32_t *brick_bits,
            int32_t *overflow_flag,
            uint64_t hash_capacity,
            uint32_t max_bricks)
        {
            const int64_t task_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (task_id >= num_tasks)
                return;
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;
            const ScanTask task = tasks[task_id];
            const float *tri = triangles + static_cast<int64_t>(task.tri_id) * 9;
            auto emit = [&](int ax0, int ax1, int ax2, int x_idx, int y_idx, int z_idx, double, double, double)
            {
                // One geometric event activates the four voxels around the
                // crossed face in the projection plane. This is the occupancy
                // subset later reused by face and boundary QEF stages.
                for (int dx = 0; dx < 2; ++dx)
                {
                    for (int dy = 0; dy < 2; ++dy)
                    {
                        int coord[3];
                        coord[ax0] = x_idx + dx;
                        coord[ax1] = y_idx + dy;
                        coord[ax2] = z_idx;
                        if (coord[0] < grid_min.x || coord[0] >= grid_max.x)
                            continue;
                        if (coord[1] < grid_min.y || coord[1] >= grid_max.y)
                            continue;
                        if (coord[2] < grid_min.z || coord[2] >= grid_max.z)
                            continue;
                        Int3 brick;
                        int local_id;
                        const uint64_t key = voxel_to_brick(coord[0], coord[1], coord[2], grid, brick, local_id);
                        const uint32_t brick_idx = get_or_create_brick(
                            key, brick, hash_keys, hash_vals, brick_count, brick_coords, overflow_flag, hash_capacity, max_bricks);
                        if (brick_idx == kEmptyBrickVal)
                            continue;
                        atomicOr(brick_bits + static_cast<int64_t>(brick_idx) * kBrickBitWords + local_id / 32, 1u << (local_id & 31));
                    }
                }
            };
            scan_triangle_events_tiled(tri, task, grid, emit);
        }

        __global__ void count_brick_voxels_kernel(
            const uint32_t *brick_bits,
            int64_t num_bricks,
            int64_t *brick_counts)
        {
            const int64_t brick_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (brick_idx >= num_bricks)
                return;
            // Popcount turns each brick bitset into the number of compact voxel
            // rows owned by that brick.
            int64_t count = 0;
            const uint32_t *bits = brick_bits + brick_idx * kBrickBitWords;
            for (int i = 0; i < kBrickBitWords; ++i)
                count += __popc(bits[i]);
            brick_counts[brick_idx] = count;
        }

        __global__ void emit_occupied_voxels_kernel(
            const int32_t *brick_coords,
            const uint32_t *brick_bits,
            const int64_t *brick_base,
            int64_t num_bricks,
            GridSpec grid,
            int32_t *voxels)
        {
            const int64_t brick_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (brick_idx >= num_bricks)
                return;
            const Int3 grid_min = grid.grid_min;
            const int bx = brick_coords[3 * brick_idx + 0];
            const int by = brick_coords[3 * brick_idx + 1];
            const int bz = brick_coords[3 * brick_idx + 2];
            const uint32_t *bits = brick_bits + brick_idx * kBrickBitWords;
            int64_t out = brick_base[brick_idx];
            for (int local_id = 0; local_id < kBrickLocalCells; ++local_id)
            {
                if ((bits[local_id / 32] & (1u << (local_id & 31))) == 0)
                    continue;
                // Enumerating local_id in increasing order defines the local
                // row order inside this compact active brick.
                const int lz = local_id / (kBrickSize * kBrickSize);
                const int rem = local_id - lz * kBrickSize * kBrickSize;
                const int ly = rem / kBrickSize;
                const int lx = rem - ly * kBrickSize;
                voxels[3 * out + 0] = grid_min.x + bx * kBrickSize + lx;
                voxels[3 * out + 1] = grid_min.y + by * kBrickSize + ly;
                voxels[3 * out + 2] = grid_min.z + bz * kBrickSize + lz;
                ++out;
            }
        }

        __global__ void accumulate_intersection_qef_kernel(
            const ScanTask *tasks,
            int64_t num_tasks,
            const float *triangles,
            GridSpec grid,
            const uint64_t *hash_keys,
            const uint32_t *hash_vals,
            const uint32_t *brick_bits,
            const int64_t *brick_base,
            float *mean_sum,
            float *cnt,
            uint32_t *intersected_mask,
            float *qefs,
            uint64_t hash_capacity)
        {
            const int64_t task_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (task_id >= num_tasks)
                return;
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;
            const BrickLookup lookup{hash_keys, hash_vals, brick_bits, brick_base, hash_capacity};
            const ScanTask task = tasks[task_id];
            const float *tri = triangles + static_cast<int64_t>(task.tri_id) * 9;
            const SymQEF10 qef = triangle_qef(tri);
            auto emit = [&](int ax0, int ax1, int ax2, int x_idx, int y_idx, int z_idx, double x, double y, double z)
            {
                // Replay the same event stream used for occupancy. This pass now
                // maps each active voxel to its compact row and accumulates the
                // intersection point and triangle plane QEF.
                for (int dx = 0; dx < 2; ++dx)
                {
                    for (int dy = 0; dy < 2; ++dy)
                    {
                        int coord[3];
                        coord[ax0] = x_idx + dx;
                        coord[ax1] = y_idx + dy;
                        coord[ax2] = z_idx;
                        if (coord[0] < grid_min.x || coord[0] >= grid_max.x)
                            continue;
                        if (coord[1] < grid_min.y || coord[1] >= grid_max.y)
                            continue;
                        if (coord[2] < grid_min.z || coord[2] >= grid_max.z)
                            continue;
                        const int64_t out_idx = lookup_voxel_row_in_bricks(
                            coord[0],
                            coord[1],
                            coord[2],
                            grid,
                            lookup);
                        if (out_idx < 0)
                            continue;
                        float p[3];
                        p[ax0] = static_cast<float>(x);
                        p[ax1] = static_cast<float>(y);
                        p[ax2] = static_cast<float>(z);
                        atomicAdd(mean_sum + 3 * out_idx + 0, p[0]);
                        atomicAdd(mean_sum + 3 * out_idx + 1, p[1]);
                        atomicAdd(mean_sum + 3 * out_idx + 2, p[2]);
                        atomicAdd(cnt + out_idx, 1.0f);
                        // The base event marks which grid edge direction was
                        // crossed. Neighbor voxels receive QEF/mean updates but
                        // should not duplicate the axis flag.
                        if (dx == 0 && dy == 0)
                            atomicOr(intersected_mask + out_idx, 1u << ax2);
                        atomic_add_qef(qefs + 10 * out_idx, qef);
                    }
                }
            };
            scan_triangle_events_tiled(tri, task, grid, emit);
        }

        __global__ void decode_intersection_masks_kernel(
            const uint32_t *mask,
            int64_t n,
            bool *intersected)
        {
            // Convert one uint32 bitfield per voxel into the public [N, 3] bool
            // layout: bit 0/1/2 means the voxel was intersected along x/y/z.
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const uint32_t m = mask[i];
            intersected[3 * i + 0] = (m & 1u) != 0;
            intersected[3 * i + 1] = (m & 2u) != 0;
            intersected[3 * i + 2] = (m & 4u) != 0;
        }

        int64_t next_power_of_two_i64(int64_t x)
        {
            int64_t p = 1;
            while (p < x)
                p <<= 1;
            return p;
        }

        int64_t read_i64(const torch::Tensor &t, cudaStream_t stream)
        {
            int64_t value = 0;
            C10_CUDA_CHECK(cudaMemcpyAsync(&value, t.data_ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            return value;
        }

        int64_t read_scan_total(const torch::Tensor &counts, const torch::Tensor &offsets, int64_t n, cudaStream_t stream)
        {
            // After an exclusive scan, total = last_count + last_offset. This
            // small host readback gives the exact tensor size for the next pass.
            int64_t tail[2] = {0, 0};
            C10_CUDA_CHECK(cudaMemcpyAsync(
                tail,
                counts.data_ptr<int64_t>() + n - 1,
                sizeof(int64_t),
                cudaMemcpyDeviceToHost,
                stream));
            C10_CUDA_CHECK(cudaMemcpyAsync(
                tail + 1,
                offsets.data_ptr<int64_t>() + n - 1,
                sizeof(int64_t),
                cudaMemcpyDeviceToHost,
                stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            return tail[0] + tail[1];
        }

        void cub_exclusive_sum_i64(const torch::Tensor &in, const torch::Tensor &out, int64_t n, cudaStream_t stream)
        {
            if (n == 0)
                return;
            TORCH_CHECK(n <= std::numeric_limits<int>::max(), "CUB item count exceeds int");
            size_t temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                nullptr,
                temp_bytes,
                in.data_ptr<int64_t>(),
                out.data_ptr<int64_t>(),
                static_cast<int>(n),
                stream));
            auto temp = torch::empty(
                {static_cast<int64_t>(temp_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(in.device()));
            C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                temp.data_ptr<uint8_t>(),
                temp_bytes,
                in.data_ptr<int64_t>(),
                out.data_ptr<int64_t>(),
                static_cast<int>(n),
                stream));
        }

        int64_t cub_sum_i64(const torch::Tensor &in, int64_t n, cudaStream_t stream)
        {
            if (n == 0)
                return 0;
            TORCH_CHECK(n <= std::numeric_limits<int>::max(), "CUB item count exceeds int");
            auto out = torch::empty({1}, in.options());
            size_t temp_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceReduce::Sum(
                nullptr,
                temp_bytes,
                in.data_ptr<int64_t>(),
                out.data_ptr<int64_t>(),
                static_cast<int>(n),
                stream));
            auto temp = torch::empty(
                {static_cast<int64_t>(temp_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(in.device()));
            C10_CUDA_CHECK(cub::DeviceReduce::Sum(
                temp.data_ptr<uint8_t>(),
                temp_bytes,
                in.data_ptr<int64_t>(),
                out.data_ptr<int64_t>(),
                static_cast<int>(n),
                stream));
            return read_i64(out, stream);
        }

        IntersectionOccupancy build_intersection_occupancy(
            const torch::Tensor &triangles,
            GridSpec grid,
            const torch::Device &device,
            cudaStream_t stream)
        {
            // Pipeline:
            // 1. Count and emit scan tasks for triangle-axis tiles.
            // 2. Estimate a strict active-brick bound from those tasks.
            // 3. Mark occupied voxel bits in active bricks through a hash table.
            // 4. Prefix-sum brick popcounts and emit compact voxel coordinates.
            IntersectionOccupancy out;
            const Int3 grid_min = grid.grid_min;
            const Int3 grid_max = grid.grid_max;
            const int64_t num_triangles = triangles.size(0);
            const auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
            const auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
            const auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(device);
            const auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(device);
            out.tasks = torch::empty({0, 6}, opts_i32);
            out.hash_keys = torch::empty({0}, opts_u64);
            out.hash_vals = torch::empty({0}, opts_u32);
            out.brick_coords = torch::empty({0, 3}, opts_i32);
            out.brick_bits = torch::empty({0, kBrickBitWords}, opts_u32);
            out.brick_base = torch::empty({0}, opts_i64);
            out.overflow_flag = torch::empty({0}, opts_i32);
            out.voxels = torch::empty({0, 3}, opts_i32);
            if (num_triangles == 0)
                return out;

            const int64_t pair_count = num_triangles * 3;
            auto task_counts = torch::empty({pair_count}, opts_i64);
            auto task_offsets = torch::empty({pair_count}, opts_i64);
            int blocks = static_cast<int>((pair_count + kThreads - 1) / kThreads);
            count_scan_tasks_kernel<<<blocks, kThreads, 0, stream>>>(
                triangles.data_ptr<float>(),
                num_triangles,
                grid,
                task_counts.data_ptr<int64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            cub_exclusive_sum_i64(task_counts, task_offsets, pair_count, stream);
            const int64_t num_tasks = read_scan_total(task_counts, task_offsets, pair_count, stream);
            out.num_tasks = num_tasks;
            if (num_tasks == 0)
                return out;

            out.tasks = torch::empty({num_tasks, 6}, opts_i32);
            auto task_brick_bounds = torch::empty({num_tasks}, opts_i64);
            emit_scan_tasks_and_brick_bounds_kernel<<<blocks, kThreads, 0, stream>>>(
                triangles.data_ptr<float>(),
                num_triangles,
                grid,
                task_offsets.data_ptr<int64_t>(),
                reinterpret_cast<ScanTask *>(out.tasks.data_ptr<int32_t>()),
                task_brick_bounds.data_ptr<int64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            const int64_t sum_brick_bounds = cub_sum_i64(task_brick_bounds, num_tasks, stream);
            const int64_t nbx = div_up_i64(grid_max.x - grid_min.x, kBrickSize);
            const int64_t nby = div_up_i64(grid_max.y - grid_min.y, kBrickSize);
            const int64_t nbz = div_up_i64(grid_max.z - grid_min.z, kBrickSize);
            const int64_t total_grid_bricks = nbx * nby * nbz;
            const int64_t max_bricks = std::min(sum_brick_bounds, total_grid_bricks);
            if (max_bricks == 0)
                return out;
            TORCH_CHECK(max_bricks <= std::numeric_limits<uint32_t>::max(), "active brick bound exceeds uint32_t");
            const int64_t hash_capacity_i64 = next_power_of_two_i64(std::max<int64_t>(2, max_bricks * 2));
            out.hash_capacity = static_cast<uint64_t>(hash_capacity_i64);

            out.hash_keys = torch::empty({hash_capacity_i64}, opts_u64);
            out.hash_vals = torch::empty({hash_capacity_i64}, opts_u32);
            auto brick_count = torch::zeros({1}, opts_u32);
            out.brick_coords = torch::empty({max_bricks, 3}, opts_i32);
            out.brick_bits = torch::zeros({max_bricks, kBrickBitWords}, opts_u32);
            out.overflow_flag = torch::zeros({1}, opts_i32);
            C10_CUDA_CHECK(cudaMemsetAsync(out.hash_keys.data_ptr<uint64_t>(), 0xff, hash_capacity_i64 * sizeof(uint64_t), stream));
            C10_CUDA_CHECK(cudaMemsetAsync(out.hash_vals.data_ptr<uint32_t>(), 0xff, hash_capacity_i64 * sizeof(uint32_t), stream));

            blocks = static_cast<int>((num_tasks + kThreads - 1) / kThreads);
            mark_occupied_voxel_bits_kernel<<<blocks, kThreads, 0, stream>>>(
                reinterpret_cast<const ScanTask *>(out.tasks.data_ptr<int32_t>()),
                num_tasks,
                triangles.data_ptr<float>(),
                grid,
                out.hash_keys.data_ptr<uint64_t>(),
                out.hash_vals.data_ptr<uint32_t>(),
                brick_count.data_ptr<uint32_t>(),
                out.brick_coords.data_ptr<int32_t>(),
                out.brick_bits.data_ptr<uint32_t>(),
                out.overflow_flag.data_ptr<int32_t>(),
                out.hash_capacity,
                static_cast<uint32_t>(max_bricks));
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            int32_t overflow = 0;
            uint32_t brick_count_h = 0;
            C10_CUDA_CHECK(cudaMemcpyAsync(&overflow, out.overflow_flag.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaMemcpyAsync(&brick_count_h, brick_count.data_ptr<uint32_t>(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            TORCH_CHECK(overflow == 0, "brick occupancy hash overflow");

            const int64_t num_bricks = static_cast<int64_t>(brick_count_h);
            out.num_bricks = num_bricks;
            TORCH_CHECK(num_bricks <= max_bricks, "active brick count exceeds bound");
            if (num_bricks == 0)
                return out;

            auto brick_counts = torch::empty({max_bricks}, opts_i64);
            out.brick_base = torch::empty({max_bricks}, opts_i64);
            blocks = static_cast<int>((num_bricks + kThreads - 1) / kThreads);
            count_brick_voxels_kernel<<<blocks, kThreads, 0, stream>>>(
                out.brick_bits.data_ptr<uint32_t>(),
                num_bricks,
                brick_counts.data_ptr<int64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            cub_exclusive_sum_i64(brick_counts, out.brick_base, num_bricks, stream);
            const int64_t num_voxels = read_scan_total(brick_counts, out.brick_base, num_bricks, stream);
            out.num_voxels = num_voxels;
            if (num_voxels == 0)
                return out;

            out.voxels = torch::empty({num_voxels, 3}, opts_i32);
            emit_occupied_voxels_kernel<<<blocks, kThreads, 0, stream>>>(
                out.brick_coords.data_ptr<int32_t>(),
                out.brick_bits.data_ptr<uint32_t>(),
                out.brick_base.data_ptr<int64_t>(),
                num_bricks,
                grid,
                out.voxels.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return out;
        }

    } // namespace

    torch::Tensor intersect_occ_cuda(
        const torch::Tensor &triangles,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range)
    {
        // Same occupancy construction as intersect_qef_cuda, but no QEF,
        // mean/cnt, or intersected mask work is performed.
        TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");

        const c10::cuda::CUDAGuard guard(triangles.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(triangles.get_device()).stream();
        const torch::Device device = triangles.device();
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
        return build_intersection_occupancy(triangles, grid, device, stream).voxels;
    }

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    intersect_qef_cuda(
        const torch::Tensor &triangles,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range)
    {
        // Start from the shared occupancy pass, then accumulate the intersection
        // planes and per-axis flags needed by the full flexible dual grid solve.
        TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");

        const c10::cuda::CUDAGuard guard(triangles.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(triangles.get_device()).stream();
        const torch::Device device = triangles.device();
        const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(device);
        const auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(device);
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

        IntersectionOccupancy occ = build_intersection_occupancy(triangles, grid, device, stream);
        const int64_t n = occ.num_voxels;
        auto mean_sum = torch::zeros({n, 3}, opts_f32);
        auto cnt = torch::zeros({n}, opts_f32);
        auto intersected = torch::empty({n, 3}, opts_bool);
        auto qefs = torch::zeros({n, 10}, opts_f32);
        if (n == 0)
            return std::make_tuple(
                occ.voxels,
                mean_sum,
                cnt,
                intersected,
                qefs,
                occ.hash_keys,
                occ.hash_vals,
                occ.brick_bits,
                occ.brick_base);

        auto intersected_mask = torch::zeros({n}, opts_u32);
        const int blocks_tasks = static_cast<int>((occ.num_tasks + kThreads - 1) / kThreads);
        accumulate_intersection_qef_kernel<<<blocks_tasks, kThreads, 0, stream>>>(
            reinterpret_cast<const ScanTask *>(occ.tasks.data_ptr<int32_t>()),
            occ.num_tasks,
            triangles.data_ptr<float>(),
            grid,
            occ.hash_keys.data_ptr<uint64_t>(),
            occ.hash_vals.data_ptr<uint32_t>(),
            occ.brick_bits.data_ptr<uint32_t>(),
            occ.brick_base.data_ptr<int64_t>(),
            mean_sum.data_ptr<float>(),
            cnt.data_ptr<float>(),
            intersected_mask.data_ptr<uint32_t>(),
            qefs.data_ptr<float>(),
            occ.hash_capacity);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const int blocks_voxels = static_cast<int>((n + kThreads - 1) / kThreads);
        decode_intersection_masks_kernel<<<blocks_voxels, kThreads, 0, stream>>>(
            intersected_mask.data_ptr<uint32_t>(),
            n,
            intersected.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return std::make_tuple(
            occ.voxels,
            mean_sum,
            cnt,
            intersected,
            qefs,
            occ.hash_keys,
            occ.hash_vals,
            occ.brick_bits,
            occ.brick_base);
    }

} // namespace o_voxel::fdg
