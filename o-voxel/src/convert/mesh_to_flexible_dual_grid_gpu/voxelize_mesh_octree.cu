#include "../api.h"

#include "types.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

// Standalone octree voxelization. It is kept as a public CUDA module, but the
// main flexible dual grid CUDA path does not call it. Work is represented as an
// octree job stream that expands from coarse cells to fine cells on the GPU.
namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;
        constexpr int kRootNeighborCount = 27;

        // Per-face data reused while octree jobs for that face are refined.
        struct FaceDesc
        {
            float3 v0;
            float3 v1;
            float3 v2;
            float3 e0;
            float3 e1;
            float3 e2;
            float3 n;
            float3 bmin;
            float3 bmax;
        };

        __host__ __device__ __forceinline__ int64_t div_up_i64(int64_t n, int64_t d)
        {
            return (n + d - 1) / d;
        }

        int ceil_log2_pos(int x)
        {
            int d = 0;
            int v = 1;
            while (v < x)
            {
                v <<= 1;
                ++d;
            }
            return d;
        }

        int grid_depth(Int3 grid_size)
        {
            return ceil_log2_pos(std::max(grid_size.x, std::max(grid_size.y, grid_size.z)));
        }

        int64_t read_scan_total(const torch::Tensor &counts, const torch::Tensor &offsets, int64_t n, cudaStream_t stream)
        {
            // After an exclusive scan, total = last_count + last_offset. The
            // host loop needs this exact size before allocating next_jobs/results.
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

        __device__ __forceinline__ float3 sub3(float3 a, float3 b)
        {
            return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        }

        __device__ __forceinline__ float dot3(float3 a, float3 b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        __device__ __forceinline__ float dot2(float2 a, float2 b)
        {
            return a.x * b.x + a.y * b.y;
        }

        __device__ __forceinline__ float3 cross3(float3 a, float3 b)
        {
            return make_float3(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
        }

        __device__ __forceinline__ float3 min3(float3 a, float3 b)
        {
            return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
        }

        __device__ __forceinline__ float3 max3(float3 a, float3 b)
        {
            return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
        }

        __device__ __forceinline__ float2 max2_zero(float2 a)
        {
            return make_float2(fmaxf(a.x, 0.0f), fmaxf(a.y, 0.0f));
        }

        __device__ __forceinline__ float3 normalize3(float3 a)
        {
            const float n2 = dot3(a, a);
            if (n2 <= 0.0f)
                return a;
            const float n = sqrtf(n2);
            return make_float3(a.x / n, a.y / n, a.z / n);
        }

        __device__ __forceinline__ bool bbox_overlap_closed(float3 a_min, float3 a_max, float3 b_min, float3 b_max)
        {
            return !(a_max.x < b_min.x || b_max.x < a_min.x ||
                     a_max.y < b_min.y || b_max.y < a_min.y ||
                     a_max.z < b_min.z || b_max.z < a_min.z);
        }

        __device__ bool triangle_box_hit(const FaceDesc &f, float3 box_min, float3 box_size, float3 box_max)
        {
            // Triangle-box overlap test. AABB rejects obvious misses first; the
            // remaining tests check whether the triangle plane crosses the box
            // and whether the box overlaps the triangle in all three projections.
            if (!bbox_overlap_closed(f.bmin, f.bmax, box_min, box_max))
                return false;

            const float3 n = f.n;
            const float3 c = make_float3(
                n.x > 0.0f ? box_size.x : 0.0f,
                n.y > 0.0f ? box_size.y : 0.0f,
                n.z > 0.0f ? box_size.z : 0.0f);
            const float d1 = dot3(n, sub3(c, f.v0));
            const float d2 = dot3(n, sub3(sub3(box_size, c), f.v0));

            // Projected edge half-spaces. The sign flip chooses inward-facing
            // edge normals for each projection, and max2_zero shifts the tested
            // box corner to the side most favorable to overlap.
            const int mul_xy = n.z < 0.0f ? -1 : 1;
            const float2 n_xy_e0 = make_float2(-mul_xy * f.e0.y, mul_xy * f.e0.x);
            const float2 n_xy_e1 = make_float2(-mul_xy * f.e1.y, mul_xy * f.e1.x);
            const float2 n_xy_e2 = make_float2(-mul_xy * f.e2.y, mul_xy * f.e2.x);
            const float d_xy_e0 = -dot2(n_xy_e0, make_float2(f.v0.x, f.v0.y)) +
                                  dot2(max2_zero(n_xy_e0), make_float2(box_size.x, box_size.y));
            const float d_xy_e1 = -dot2(n_xy_e1, make_float2(f.v1.x, f.v1.y)) +
                                  dot2(max2_zero(n_xy_e1), make_float2(box_size.x, box_size.y));
            const float d_xy_e2 = -dot2(n_xy_e2, make_float2(f.v2.x, f.v2.y)) +
                                  dot2(max2_zero(n_xy_e2), make_float2(box_size.x, box_size.y));

            const int mul_yz = n.x < 0.0f ? -1 : 1;
            const float2 n_yz_e0 = make_float2(-mul_yz * f.e0.z, mul_yz * f.e0.y);
            const float2 n_yz_e1 = make_float2(-mul_yz * f.e1.z, mul_yz * f.e1.y);
            const float2 n_yz_e2 = make_float2(-mul_yz * f.e2.z, mul_yz * f.e2.y);
            const float d_yz_e0 = -dot2(n_yz_e0, make_float2(f.v0.y, f.v0.z)) +
                                  dot2(max2_zero(n_yz_e0), make_float2(box_size.y, box_size.z));
            const float d_yz_e1 = -dot2(n_yz_e1, make_float2(f.v1.y, f.v1.z)) +
                                  dot2(max2_zero(n_yz_e1), make_float2(box_size.y, box_size.z));
            const float d_yz_e2 = -dot2(n_yz_e2, make_float2(f.v2.y, f.v2.z)) +
                                  dot2(max2_zero(n_yz_e2), make_float2(box_size.y, box_size.z));

            const int mul_zx = n.y < 0.0f ? -1 : 1;
            const float2 n_zx_e0 = make_float2(-mul_zx * f.e0.x, mul_zx * f.e0.z);
            const float2 n_zx_e1 = make_float2(-mul_zx * f.e1.x, mul_zx * f.e1.z);
            const float2 n_zx_e2 = make_float2(-mul_zx * f.e2.x, mul_zx * f.e2.z);
            const float d_zx_e0 = -dot2(n_zx_e0, make_float2(f.v0.z, f.v0.x)) +
                                  dot2(max2_zero(n_zx_e0), make_float2(box_size.z, box_size.x));
            const float d_zx_e1 = -dot2(n_zx_e1, make_float2(f.v1.z, f.v1.x)) +
                                  dot2(max2_zero(n_zx_e1), make_float2(box_size.z, box_size.x));
            const float d_zx_e2 = -dot2(n_zx_e2, make_float2(f.v2.z, f.v2.x)) +
                                  dot2(max2_zero(n_zx_e2), make_float2(box_size.z, box_size.x));

            const float n_dot_p = dot3(n, box_min);
            // Plane slab test: the two extreme box corners along the triangle
            // normal must lie on opposite sides, or one side exactly on plane.
            if (((n_dot_p + d1) * (n_dot_p + d2)) > 0.0f)
                return false;

            const float2 p_xy = make_float2(box_min.x, box_min.y);
            if (dot2(n_xy_e0, p_xy) + d_xy_e0 < 0.0f)
                return false;
            if (dot2(n_xy_e1, p_xy) + d_xy_e1 < 0.0f)
                return false;
            if (dot2(n_xy_e2, p_xy) + d_xy_e2 < 0.0f)
                return false;

            const float2 p_yz = make_float2(box_min.y, box_min.z);
            if (dot2(n_yz_e0, p_yz) + d_yz_e0 < 0.0f)
                return false;
            if (dot2(n_yz_e1, p_yz) + d_yz_e1 < 0.0f)
                return false;
            if (dot2(n_yz_e2, p_yz) + d_yz_e2 < 0.0f)
                return false;

            const float2 p_zx = make_float2(box_min.z, box_min.x);
            if (dot2(n_zx_e0, p_zx) + d_zx_e0 < 0.0f)
                return false;
            if (dot2(n_zx_e1, p_zx) + d_zx_e1 < 0.0f)
                return false;
            if (dot2(n_zx_e2, p_zx) + d_zx_e2 < 0.0f)
                return false;

            return true;
        }

        __device__ void compute_face_root(
            int ix0,
            int iy0,
            int iz0,
            int ix1,
            int iy1,
            int iz1,
            int ix2,
            int iy2,
            int iz2,
            int d,
            int &level,
            int &root_i,
            int &root_j,
            int &root_k)
        {
            // XOR of vertex leaf coordinates tells which octree bits differ
            // across the triangle vertices. The highest differing bit selects
            // the smallest common ancestor node containing all three vertex cells.
            const uint32_t diff =
                static_cast<uint32_t>(ix0 ^ ix1) | static_cast<uint32_t>(ix0 ^ ix2) |
                static_cast<uint32_t>(iy0 ^ iy1) | static_cast<uint32_t>(iy0 ^ iy2) |
                static_cast<uint32_t>(iz0 ^ iz1) | static_cast<uint32_t>(iz0 ^ iz2);
            if (diff == 0)
            {
                level = d;
                root_i = ix0;
                root_j = iy0;
                root_k = iz0;
                return;
            }
            const int msb = 31 - __clz(diff);
            level = d - 1 - msb;
            const int shift = d - level;
            root_i = ix0 >> shift;
            root_j = iy0 >> shift;
            root_k = iz0 >> shift;
        }

        __device__ bool node_in_domain(int d, int level, int i, int j, int k, Int3 grid_size)
        {
            const int cells = 1 << level;
            if (i < 0 || i >= cells || j < 0 || j >= cells || k < 0 || k >= cells)
                return false;
            const int span = 1 << (d - level);
            return i * span < grid_size.x && j * span < grid_size.y && k * span < grid_size.z;
        }

        __device__ void node_box(int d, int level, int i, int j, int k, Int3 grid_min, float3 voxel_size, float3 &box_min, float3 &box_size, float3 &box_max)
        {
            const int span = 1 << (d - level);
            const int gx0 = grid_min.x + i * span;
            const int gy0 = grid_min.y + j * span;
            const int gz0 = grid_min.z + k * span;
            const int gx1 = gx0 + span;
            const int gy1 = gy0 + span;
            const int gz1 = gz0 + span;
            box_min = make_float3(gx0 * voxel_size.x, gy0 * voxel_size.y, gz0 * voxel_size.z);
            box_max = make_float3(gx1 * voxel_size.x, gy1 * voxel_size.y, gz1 * voxel_size.z);
            box_size = sub3(box_max, box_min);
        }

        __global__ void build_leaf_coords_kernel(
            const float *__restrict__ vertices,
            int64_t num_vertices,
            float3 inv_voxel_size,
            Int3 grid_min,
            Int3 grid_size,
            int32_t *__restrict__ leaf_coords)
        {
            const int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (vid >= num_vertices)
                return;
            // Convert each vertex to a leaf voxel coordinate in grid-local
            // space. These coordinates are only used to seed each face's octree
            // root; geometric overlap is still checked later by triangle_box_hit.
            int ix = static_cast<int>(floorf(vertices[3 * vid + 0] * inv_voxel_size.x)) - grid_min.x;
            int iy = static_cast<int>(floorf(vertices[3 * vid + 1] * inv_voxel_size.y)) - grid_min.y;
            int iz = static_cast<int>(floorf(vertices[3 * vid + 2] * inv_voxel_size.z)) - grid_min.z;
            ix = max(0, min(ix, grid_size.x - 1));
            iy = max(0, min(iy, grid_size.y - 1));
            iz = max(0, min(iz, grid_size.z - 1));
            leaf_coords[3 * vid + 0] = ix;
            leaf_coords[3 * vid + 1] = iy;
            leaf_coords[3 * vid + 2] = iz;
        }

        __global__ void init_faces_kernel(
            const float *__restrict__ vertices,
            const int32_t *__restrict__ faces,
            const int32_t *__restrict__ leaf_coords,
            int64_t num_faces,
            int d,
            FaceDesc *__restrict__ face_desc,
            int32_t *__restrict__ jobs)
        {
            const int64_t fid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (fid >= num_faces)
                return;

            const int v0_id = faces[3 * fid + 0];
            const int v1_id = faces[3 * fid + 1];
            const int v2_id = faces[3 * fid + 2];
            const float3 v0 = make_float3(vertices[3 * v0_id + 0], vertices[3 * v0_id + 1], vertices[3 * v0_id + 2]);
            const float3 v1 = make_float3(vertices[3 * v1_id + 0], vertices[3 * v1_id + 1], vertices[3 * v1_id + 2]);
            const float3 v2 = make_float3(vertices[3 * v2_id + 0], vertices[3 * v2_id + 1], vertices[3 * v2_id + 2]);

            FaceDesc f;
            f.v0 = v0;
            f.v1 = v1;
            f.v2 = v2;
            f.e0 = sub3(v1, v0);
            f.e1 = sub3(v2, v1);
            f.e2 = sub3(v0, v2);
            f.n = normalize3(cross3(f.e0, f.e1));
            f.bmin = min3(v0, min3(v1, v2));
            f.bmax = max3(v0, max3(v1, v2));
            face_desc[fid] = f;

            // Start from the smallest octree node that contains the triangle's
            // vertex leaf cells. The 3x3x3 neighbor seed around that root keeps
            // conservative coverage when the triangle crosses nearby cells.
            int level;
            int root_i;
            int root_j;
            int root_k;
            compute_face_root(
                leaf_coords[3 * v0_id + 0],
                leaf_coords[3 * v0_id + 1],
                leaf_coords[3 * v0_id + 2],
                leaf_coords[3 * v1_id + 0],
                leaf_coords[3 * v1_id + 1],
                leaf_coords[3 * v1_id + 2],
                leaf_coords[3 * v2_id + 0],
                leaf_coords[3 * v2_id + 1],
                leaf_coords[3 * v2_id + 2],
                d,
                level,
                root_i,
                root_j,
                root_k);

            const int64_t base = fid * kRootNeighborCount;
            int slot = 0;
            for (int dz = -1; dz <= 1; ++dz)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const int64_t out = base + slot++;
                        jobs[5 * out + 0] = static_cast<int32_t>(fid);
                        jobs[5 * out + 1] = level;
                        jobs[5 * out + 2] = root_i + dx;
                        jobs[5 * out + 3] = root_j + dy;
                        jobs[5 * out + 4] = root_k + dz;
                    }
                }
            }
        }

        __global__ void count_jobs_kernel(
            const int32_t *__restrict__ jobs,
            int64_t num_jobs,
            const FaceDesc *__restrict__ face_desc,
            int d,
            Int3 grid_min,
            Int3 grid_size,
            float3 voxel_size,
            int64_t *__restrict__ child_count,
            int64_t *__restrict__ result_count)
        {
            const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= num_jobs)
                return;
            const int fid = jobs[5 * idx + 0];
            const int level = jobs[5 * idx + 1];
            const int i = jobs[5 * idx + 2];
            const int j = jobs[5 * idx + 3];
            const int k = jobs[5 * idx + 4];

            if (!node_in_domain(d, level, i, j, k, grid_size))
            {
                child_count[idx] = 0;
                result_count[idx] = 0;
                return;
            }

            float3 box_min;
            float3 box_size;
            float3 box_max;
            node_box(d, level, i, j, k, grid_min, voxel_size, box_min, box_size, box_max);
            if (!triangle_box_hit(face_desc[fid], box_min, box_size, box_max))
            {
                // Miss: no children and no leaf result.
                child_count[idx] = 0;
                result_count[idx] = 0;
            }
            else if (level < d)
            {
                // Hit at an internal octree node: split into eight children for
                // the next breadth-first refinement level.
                child_count[idx] = 8;
                result_count[idx] = 0;
            }
            else
            {
                // Hit at leaf level: this face intersects this voxel.
                child_count[idx] = 0;
                result_count[idx] = 1;
            }
        }

        __global__ void emit_jobs_kernel(
            const int32_t *__restrict__ jobs,
            int64_t num_jobs,
            const int64_t *__restrict__ child_count,
            const int64_t *__restrict__ result_count,
            const int64_t *__restrict__ child_offsets,
            const int64_t *__restrict__ result_offsets,
            int32_t *__restrict__ next_jobs,
            int32_t *__restrict__ result_prim,
            int32_t *__restrict__ result_voxels)
        {
            const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= num_jobs)
                return;
            const int prim_id = jobs[5 * idx + 0];
            const int level = jobs[5 * idx + 1];
            const int i = jobs[5 * idx + 2];
            const int j = jobs[5 * idx + 3];
            const int k = jobs[5 * idx + 4];

            if (child_count[idx] != 0)
            {
                const int64_t base = child_offsets[idx];
                const int child_level = level + 1;
                int slot = 0;
                // child_offsets is the scan of child_count, so every surviving
                // job writes its eight children into a disjoint range.
                for (int bz = 0; bz < 2; ++bz)
                {
                    for (int by = 0; by < 2; ++by)
                    {
                        for (int bx = 0; bx < 2; ++bx)
                        {
                            const int64_t out = base + slot++;
                            next_jobs[5 * out + 0] = prim_id;
                            next_jobs[5 * out + 1] = child_level;
                            next_jobs[5 * out + 2] = 2 * i + bx;
                            next_jobs[5 * out + 3] = 2 * j + by;
                            next_jobs[5 * out + 4] = 2 * k + bz;
                        }
                    }
                }
            }
            else if (result_count[idx] != 0)
            {
                const int64_t out = result_offsets[idx];
                // result_offsets compacts leaf hits from this level into the
                // chunk returned to the host loop.
                result_prim[out] = prim_id;
                result_voxels[3 * out + 0] = i;
                result_voxels[3 * out + 1] = j;
                result_voxels[3 * out + 2] = k;
            }
        }

        void copy_chunk(torch::Tensor &dst, int64_t dst_offset, const torch::Tensor &src, int64_t values, cudaStream_t stream)
        {
            if (values == 0)
                return;
            C10_CUDA_CHECK(cudaMemcpyAsync(
                dst.data_ptr<int32_t>() + dst_offset,
                src.data_ptr<int32_t>(),
                values * sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                stream));
        }

    } // namespace

    std::tuple<torch::Tensor, torch::Tensor>
    voxelize_mesh_octree_cuda(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range)
    {
        // Returns one row per primitive/voxel hit:
        // prim_ids [K] int32 and voxels [K, 3] int32.
        TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
        TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");

        const c10::cuda::CUDAGuard guard(vertices.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        const torch::Device device = vertices.device();
        const auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
        const auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
        const auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);
        const int64_t num_vertices = vertices.size(0);
        const int64_t num_faces = faces.size(0);
        auto empty_prim = torch::empty({0}, opts_i32);
        auto empty_voxels = torch::empty({0, 3}, opts_i32);
        if (num_vertices == 0 || num_faces == 0)
            return std::make_tuple(empty_prim, empty_voxels);

        const float3 voxel_size_h{voxel_size[0], voxel_size[1], voxel_size[2]};
        const float3 inv_voxel_size{1.0f / voxel_size_h.x, 1.0f / voxel_size_h.y, 1.0f / voxel_size_h.z};
        const Int3 grid_min{
            static_cast<int32_t>(grid_range[0]),
            static_cast<int32_t>(grid_range[1]),
            static_cast<int32_t>(grid_range[2])};
        const Int3 grid_max{
            static_cast<int32_t>(grid_range[3]),
            static_cast<int32_t>(grid_range[4]),
            static_cast<int32_t>(grid_range[5])};
        const Int3 grid_size{grid_max.x - grid_min.x, grid_max.y - grid_min.y, grid_max.z - grid_min.z};
        const int d = grid_depth(grid_size);
        // d is the full leaf depth of the cubic octree that covers the grid.
        // A leaf node corresponds to one voxel-sized cell.
        TORCH_CHECK(d <= 21, "grid depth exceeds 21");

        auto leaf_coords = torch::empty({num_vertices, 3}, opts_i32);
        int blocks = static_cast<int>(div_up_i64(num_vertices, kThreads));
        build_leaf_coords_kernel<<<blocks, kThreads, 0, stream>>>(
            vertices.data_ptr<float>(),
            num_vertices,
            inv_voxel_size,
            grid_min,
            grid_size,
            leaf_coords.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto face_desc_storage = torch::empty(
            {num_faces * static_cast<int64_t>(sizeof(FaceDesc))},
            opts_u8);
        FaceDesc *face_desc = reinterpret_cast<FaceDesc *>(face_desc_storage.data_ptr<uint8_t>());
        int64_t num_jobs = num_faces * kRootNeighborCount;
        torch::Tensor jobs = torch::empty({num_jobs, 5}, opts_i32);
        blocks = static_cast<int>(div_up_i64(num_faces, kThreads));
        init_faces_kernel<<<blocks, kThreads, 0, stream>>>(
            vertices.data_ptr<float>(),
            faces.data_ptr<int32_t>(),
            leaf_coords.data_ptr<int32_t>(),
            num_faces,
            d,
            face_desc,
            jobs.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        std::vector<torch::Tensor> prim_chunks;
        std::vector<torch::Tensor> voxel_chunks;
        std::vector<int64_t> chunk_sizes;
        int64_t total_results = 0;

        while (num_jobs > 0)
        {
            // Breadth-first refinement. Each iteration classifies all current
            // jobs in parallel, scans counts with CUB, then emits the next level
            // jobs and any leaf hits.
            auto child_count = torch::empty({num_jobs}, opts_i64);
            auto result_count = torch::empty({num_jobs}, opts_i64);
            auto child_offsets = torch::empty({num_jobs}, opts_i64);
            auto result_offsets = torch::empty({num_jobs}, opts_i64);
            blocks = static_cast<int>(div_up_i64(num_jobs, kThreads));
            count_jobs_kernel<<<blocks, kThreads, 0, stream>>>(
                jobs.data_ptr<int32_t>(),
                num_jobs,
                face_desc,
                d,
                grid_min,
                grid_size,
                voxel_size_h,
                child_count.data_ptr<int64_t>(),
                result_count.data_ptr<int64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            cub_exclusive_sum_i64(child_count, child_offsets, num_jobs, stream);
            cub_exclusive_sum_i64(result_count, result_offsets, num_jobs, stream);
            const int64_t next_count = read_scan_total(child_count, child_offsets, num_jobs, stream);
            const int64_t result_count_h = read_scan_total(result_count, result_offsets, num_jobs, stream);

            auto next_jobs = torch::empty({next_count, 5}, opts_i32);
            auto result_prim = torch::empty({result_count_h}, opts_i32);
            auto result_voxels = torch::empty({result_count_h, 3}, opts_i32);
            emit_jobs_kernel<<<blocks, kThreads, 0, stream>>>(
                jobs.data_ptr<int32_t>(),
                num_jobs,
                child_count.data_ptr<int64_t>(),
                result_count.data_ptr<int64_t>(),
                child_offsets.data_ptr<int64_t>(),
                result_offsets.data_ptr<int64_t>(),
                next_jobs.data_ptr<int32_t>(),
                result_prim.data_ptr<int32_t>(),
                result_voxels.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            if (result_count_h > 0)
            {
                total_results += result_count_h;
                prim_chunks.push_back(result_prim);
                voxel_chunks.push_back(result_voxels);
                chunk_sizes.push_back(result_count_h);
            }

            jobs = next_jobs;
            num_jobs = next_count;
        }

        if (total_results == 0)
            return std::make_tuple(empty_prim, empty_voxels);

        auto prim_ids = torch::empty({total_results}, opts_i32);
        auto voxels = torch::empty({total_results, 3}, opts_i32);
        int64_t cursor = 0;
        // Leaf hits are produced level by level, so collect chunks first and
        // concatenate once the total result count is known.
        for (size_t i = 0; i < chunk_sizes.size(); ++i)
        {
            const int64_t n = chunk_sizes[i];
            copy_chunk(prim_ids, cursor, prim_chunks[i], n, stream);
            copy_chunk(voxels, cursor * 3, voxel_chunks[i], n * 3, stream);
            cursor += n;
        }
        return std::make_tuple(prim_ids, voxels);
    }

} // namespace o_voxel::fdg
