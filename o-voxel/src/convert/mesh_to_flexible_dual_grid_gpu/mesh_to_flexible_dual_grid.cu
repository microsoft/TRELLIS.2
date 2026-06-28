#include "../api.h"

#include "qef.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

// Full CUDA flexible dual grid pipeline. It follows the CPU algorithm's
// semantics, but each stage is expressed as GPU work: gather triangles, build
// intersection QEFs, add face and boundary QEFs in-place, then solve one QEF per
// active voxel.
namespace o_voxel::fdg
{
    namespace
    {

        constexpr int kThreads = 256;
        constexpr uint64_t kEmptyEdgeKey = UINT64_MAX;

        __host__ __device__ __forceinline__ int64_t div_up_i64(int64_t n, int64_t d)
        {
            return (n + d - 1) / d;
        }

        int64_t next_power_of_two_i64(int64_t x)
        {
            int64_t out = 1;
            while (out < x)
                out <<= 1;
            return out;
        }

        __host__ __device__ __forceinline__ uint64_t pack_edge_key(int32_t a, int32_t b)
        {
            // Callers pass endpoints sorted as (min, max), so the same
            // undirected mesh edge from two faces lands in the same hash slot.
            return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
                   static_cast<uint32_t>(b);
        }

        __host__ __device__ __forceinline__ int32_t edge_key_v0(uint64_t key)
        {
            return static_cast<int32_t>(key >> 32);
        }

        __host__ __device__ __forceinline__ int32_t edge_key_v1(uint64_t key)
        {
            return static_cast<int32_t>(key & 0xffffffffu);
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

        namespace small_cpqr
        {
            // Small column-pivoted QR routines used by the constrained QEF
            // solver. This keeps the CUDA solve aligned with the CPU path
            // instead of using a closed-form solver with different numerical behavior.
            namespace detail
            {

                template <int N>
                __device__ __forceinline__ float absf(float x)
                {
                    return x < 0.0f ? -x : x;
                }

                template <int N>
                __device__ __forceinline__ void swap_cols(
                    float *qr,
                    float *col_norms_updated,
                    float *col_norms_direct,
                    int *perm,
                    int c0,
                    int c1)
                {
                    if (c0 == c1)
                        return;
                    for (int r = 0; r < N; ++r)
                    {
                        const float tmp = qr[r * N + c0];
                        qr[r * N + c0] = qr[r * N + c1];
                        qr[r * N + c1] = tmp;
                    }
                    const float tmp_u = col_norms_updated[c0];
                    col_norms_updated[c0] = col_norms_updated[c1];
                    col_norms_updated[c1] = tmp_u;
                    const float tmp_d = col_norms_direct[c0];
                    col_norms_direct[c0] = col_norms_direct[c1];
                    col_norms_direct[c1] = tmp_d;
                    const int tmp_p = perm[c0];
                    perm[c0] = perm[c1];
                    perm[c1] = tmp_p;
                }

                template <int N>
                __device__ __forceinline__ void make_householder_real(
                    float x0,
                    const float *tail_in,
                    int tail_len,
                    float *beta,
                    float *tau,
                    float *essential_out)
                {
                    // Build a Householder reflector that zeroes the tail of a
                    // column. This is the small-matrix version of QR
                    // factorization used by the voxel-local QEF solve.
                    float tail_sq_norm = 0.0f;
                    for (int i = 0; i < N - 1; ++i)
                        if (i < tail_len)
                            tail_sq_norm += tail_in[i] * tail_in[i];

                    if (tail_sq_norm <= FLT_MIN)
                    {
                        *beta = x0;
                        *tau = 0.0f;
                        for (int i = 0; i < N - 1; ++i)
                            essential_out[i] = 0.0f;
                        return;
                    }

                    float b = sqrtf(x0 * x0 + tail_sq_norm);
                    if (x0 >= 0.0f)
                        b = -b;
                    const float denom = x0 - b;
                    for (int i = 0; i < N - 1; ++i)
                        essential_out[i] = (i < tail_len) ? (tail_in[i] / denom) : 0.0f;
                    *beta = b;
                    *tau = (b - x0) / b;
                }

                template <int N>
                __device__ __forceinline__ void apply_householder_left_matrix(
                    float *qr,
                    int row0,
                    int col0,
                    const float *essential,
                    int tail_len,
                    float tau)
                {
                    if (tau == 0.0f)
                        return;
                    for (int j = 0; j < N; ++j)
                    {
                        if (j < col0)
                            continue;
                        float tmp = qr[row0 * N + j];
                        for (int i = 0; i < N - 1; ++i)
                            if (i < tail_len)
                                tmp += essential[i] * qr[(row0 + 1 + i) * N + j];
                        qr[row0 * N + j] -= tau * tmp;
                        for (int i = 0; i < N - 1; ++i)
                            if (i < tail_len)
                                qr[(row0 + 1 + i) * N + j] -= tau * essential[i] * tmp;
                    }
                }

                template <int N>
                __device__ __forceinline__ void apply_householder_left_vector(
                    float *c,
                    int row0,
                    const float *essential,
                    int tail_len,
                    float tau)
                {
                    if (tau == 0.0f)
                        return;
                    float tmp = c[row0];
                    for (int i = 0; i < N - 1; ++i)
                        if (i < tail_len)
                            tmp += essential[i] * c[row0 + 1 + i];
                    c[row0] -= tau * tmp;
                    for (int i = 0; i < N - 1; ++i)
                        if (i < tail_len)
                            c[row0 + 1 + i] -= tau * essential[i] * tmp;
                }

                template <int N>
                __device__ __forceinline__ void backsolve_upper_ranked(
                    const float *qr,
                    int rank,
                    const float *c,
                    const int *perm,
                    float *x_out)
                {
                    float y[N];
                    for (int i = 0; i < N; ++i)
                    {
                        y[i] = 0.0f;
                        x_out[i] = 0.0f;
                    }
                    for (int i = rank - 1; i >= 0; --i)
                    {
                        float s = c[i];
                        for (int j = 0; j < N; ++j)
                            if (j > i && j < rank)
                                s -= qr[i * N + j] * y[j];
                        y[i] = s / qr[i * N + i];
                    }
                    for (int i = 0; i < N; ++i)
                        x_out[perm[i]] = (i < rank) ? y[i] : 0.0f;
                }

                template <int N>
                __device__ __forceinline__ void cpqr_solve_small_impl(
                    const float *A_in,
                    const float *b_in,
                    float *x_out)
                {
                    // Column-pivoted QR handles rank-deficient QEF systems: if
                    // the plane constraints do not determine all coordinates,
                    // low-norm columns are moved to the end and dropped.
                    float qr[N * N];
                    float c[N];
                    int perm[N];
                    float col_norms_direct[N];
                    float col_norms_updated[N];
                    float essential[N > 1 ? N - 1 : 1];

                    for (int i = 0; i < N * N; ++i)
                        qr[i] = A_in[i];
                    for (int i = 0; i < N; ++i)
                    {
                        c[i] = b_in[i];
                        perm[i] = i;
                        x_out[i] = 0.0f;
                    }

                    // Track both exact and downdated column norms. Pivoting on
                    // the largest remaining norm improves stability for nearly
                    // degenerate local systems.
                    for (int j = 0; j < N; ++j)
                    {
                        float norm_sq = 0.0f;
                        for (int r = 0; r < N; ++r)
                            norm_sq += qr[r * N + j] * qr[r * N + j];
                        const float norm = sqrtf(norm_sq);
                        col_norms_direct[j] = norm;
                        col_norms_updated[j] = norm;
                    }

                    float max_norm_updated = col_norms_updated[0];
                    for (int j = 1; j < N; ++j)
                        if (col_norms_updated[j] > max_norm_updated)
                            max_norm_updated = col_norms_updated[j];
                    const float threshold_helper = (max_norm_updated * FLT_EPSILON) * (max_norm_updated * FLT_EPSILON) / float(N);
                    const float norm_downdate_threshold = sqrtf(FLT_EPSILON);
                    int nonzero_pivots = N;

                    for (int k = 0; k < N; ++k)
                    {
                        // Choose the strongest remaining column as the next
                        // pivot, then apply a Householder reflector to form R.
                        int biggest_col_index = k;
                        float best_updated = col_norms_updated[k];
                        for (int j = 0; j < N; ++j)
                            if (j > k && col_norms_updated[j] > best_updated)
                            {
                                best_updated = col_norms_updated[j];
                                biggest_col_index = j;
                            }
                        if (nonzero_pivots == N && best_updated * best_updated < threshold_helper * float(N - k))
                            nonzero_pivots = k;

                        swap_cols<N>(qr, col_norms_updated, col_norms_direct, perm, k, biggest_col_index);
                        const int tail_len = N - k - 1;
                        float tail_local[N > 1 ? N - 1 : 1];
                        for (int i = 0; i < N - 1; ++i)
                            tail_local[i] = (i < tail_len) ? qr[(k + 1 + i) * N + k] : 0.0f;

                        float beta = 0.0f;
                        float tau = 0.0f;
                        make_householder_real<N>(qr[k * N + k], tail_local, tail_len, &beta, &tau, essential);
                        qr[k * N + k] = beta;
                        for (int i = 0; i < N - 1; ++i)
                            if (i < tail_len)
                                qr[(k + 1 + i) * N + k] = essential[i];

                        apply_householder_left_matrix<N>(qr, k, k + 1, essential, tail_len, tau);
                        if (k < nonzero_pivots)
                            apply_householder_left_vector<N>(c, k, essential, tail_len, tau);

                        // Cheap norm downdates avoid recomputing every column
                        // norm after each reflector; recompute only when the
                        // downdate becomes unreliable.
                        for (int j = 0; j < N; ++j)
                        {
                            if (j <= k || col_norms_updated[j] == 0.0f)
                                continue;
                            float temp = absf<N>(qr[k * N + j]) / col_norms_updated[j];
                            temp = (1.0f + temp) * (1.0f - temp);
                            if (temp < 0.0f)
                                temp = 0.0f;
                            const float ratio = col_norms_updated[j] / col_norms_direct[j];
                            const float temp2 = temp * ratio * ratio;
                            if (temp2 <= norm_downdate_threshold)
                            {
                                float norm_sq = 0.0f;
                                for (int r = 0; r < N; ++r)
                                    if (r > k)
                                        norm_sq += qr[r * N + j] * qr[r * N + j];
                                const float norm = sqrtf(norm_sq);
                                col_norms_direct[j] = norm;
                                col_norms_updated[j] = norm;
                            }
                            else
                            {
                                col_norms_updated[j] *= sqrtf(temp);
                            }
                        }
                    }

                    if (nonzero_pivots == 0)
                        return;
                    // Solve the retained upper-triangular part and undo the
                    // pivot permutation. Dropped columns stay zero.
                    backsolve_upper_ranked<N>(qr, nonzero_pivots, c, perm, x_out);
                }

            } // namespace detail

            __device__ __forceinline__ void cpqr_solve_3x3(const float A[9], const float b[3], float x[3])
            {
                detail::cpqr_solve_small_impl<3>(A, b, x);
            }

            __device__ __forceinline__ void cpqr_solve_2x2(const float A[4], const float b[2], float x[2])
            {
                detail::cpqr_solve_small_impl<2>(A, b, x);
            }

            __device__ __forceinline__ float solve_1x1_unchecked(float a, float rhs)
            {
                return rhs / a;
            }

        } // namespace small_cpqr

        __host__ __device__ __forceinline__ int idx4(int r, int c) { return r * 4 + c; }
        __host__ __device__ __forceinline__ int idx2(int r, int c) { return r * 2 + c; }

        __device__ __forceinline__ void sym10_to_dense4x4(const SymQEF10 &q, float Q[16])
        {
            // Expand compact symmetric storage so the solver can evaluate
            // p^T Q p and extract 3x3/2x2 constrained systems directly.
            Q[idx4(0, 0)] = q.q00;
            Q[idx4(0, 1)] = q.q01;
            Q[idx4(0, 2)] = q.q02;
            Q[idx4(0, 3)] = q.q03;
            Q[idx4(1, 0)] = q.q01;
            Q[idx4(1, 1)] = q.q11;
            Q[idx4(1, 2)] = q.q12;
            Q[idx4(1, 3)] = q.q13;
            Q[idx4(2, 0)] = q.q02;
            Q[idx4(2, 1)] = q.q12;
            Q[idx4(2, 2)] = q.q22;
            Q[idx4(2, 3)] = q.q23;
            Q[idx4(3, 0)] = q.q03;
            Q[idx4(3, 1)] = q.q13;
            Q[idx4(3, 2)] = q.q23;
            Q[idx4(3, 3)] = q.q33;
        }

        __device__ __forceinline__ bool point_inside_box3(const float v[3], const float min_corner[3], const float max_corner[3])
        {
            return v[0] >= min_corner[0] && v[0] <= max_corner[0] &&
                   v[1] >= min_corner[1] && v[1] <= max_corner[1] &&
                   v[2] >= min_corner[2] && v[2] <= max_corner[2];
        }

        __device__ __forceinline__ float qef_error4(const float Q[16], const float p[4])
        {
            // Homogeneous point p=(x,y,z,1). The scalar p^T Q p is the total
            // squared plane/line distance represented by the accumulated QEF.
            const float y0 = Q[idx4(0, 0)] * p[0] + Q[idx4(0, 1)] * p[1] + Q[idx4(0, 2)] * p[2] + Q[idx4(0, 3)] * p[3];
            const float y1 = Q[idx4(1, 0)] * p[0] + Q[idx4(1, 1)] * p[1] + Q[idx4(1, 2)] * p[2] + Q[idx4(1, 3)] * p[3];
            const float y2 = Q[idx4(2, 0)] * p[0] + Q[idx4(2, 1)] * p[1] + Q[idx4(2, 2)] * p[2] + Q[idx4(2, 3)] * p[3];
            const float y3 = Q[idx4(3, 0)] * p[0] + Q[idx4(3, 1)] * p[1] + Q[idx4(3, 2)] * p[2] + Q[idx4(3, 3)] * p[3];
            return p[0] * y0 + p[1] * y1 + p[2] * y2 + p[3] * y3;
        }

        __device__ __forceinline__ void add_qef_regularization_inplace(
            float Q[16],
            const float mean_sum[3],
            float cnt,
            float regularization_weight)
        {
            if (regularization_weight <= 0.0f || cnt <= 0.0f)
                return;

            // Regularization adds w * ||x - mean_intersection||^2. It keeps the
            // solve near the observed intersection points when QEF planes alone
            // are under-constrained.
            const float px = mean_sum[0] / cnt;
            const float py = mean_sum[1] / cnt;
            const float pz = mean_sum[2] / cnt;
            const float w = regularization_weight * cnt;

            Q[idx4(0, 0)] += w;
            Q[idx4(1, 1)] += w;
            Q[idx4(2, 2)] += w;
            Q[idx4(0, 3)] += -w * px;
            Q[idx4(1, 3)] += -w * py;
            Q[idx4(2, 3)] += -w * pz;
            Q[idx4(3, 0)] += -w * px;
            Q[idx4(3, 1)] += -w * py;
            Q[idx4(3, 2)] += -w * pz;
            Q[idx4(3, 3)] += w * (px * px + py * py + pz * pz);
        }

        __device__ __forceinline__ void try_single_constraint(
            const float Q[16],
            int fixed_axis,
            const float min_corner[3],
            const float max_corner[3],
            float &best,
            float v_new[3])
        {
            const int ax1 = (fixed_axis + 1) % 3;
            const int ax2 = (fixed_axis + 2) % 3;
            // Candidate on one voxel face: fix one coordinate to min or max and
            // solve the remaining 2D QEF on that face.
            float A2[4] = {
                Q[idx4(ax1, ax1)], Q[idx4(ax1, ax2)],
                Q[idx4(ax2, ax1)], Q[idx4(ax2, ax2)]};
            float B2[4] = {
                Q[idx4(ax1, fixed_axis)], Q[idx4(ax1, 3)],
                Q[idx4(ax2, fixed_axis)], Q[idx4(ax2, 3)]};
            float x2[2];

            for (int bound = 0; bound < 2; ++bound)
            {
                const float fixed = bound == 0 ? min_corner[fixed_axis] : max_corner[fixed_axis];
                const float rhs2[2] = {
                    -(B2[idx2(0, 0)] * fixed + B2[idx2(0, 1)]),
                    -(B2[idx2(1, 0)] * fixed + B2[idx2(1, 1)])};
                small_cpqr::cpqr_solve_2x2(A2, rhs2, x2);
                if (x2[0] >= min_corner[ax1] && x2[0] <= max_corner[ax1] &&
                    x2[1] >= min_corner[ax2] && x2[1] <= max_corner[ax2])
                {
                    float p4[4];
                    p4[fixed_axis] = fixed;
                    p4[ax1] = x2[0];
                    p4[ax2] = x2[1];
                    p4[3] = 1.0f;
                    const float err = qef_error4(Q, p4);
                    if (err < best)
                    {
                        best = err;
                        v_new[0] = p4[0];
                        v_new[1] = p4[1];
                        v_new[2] = p4[2];
                    }
                }
            }
        }

        __device__ __forceinline__ void try_two_constraint(
            const float Q[16],
            int free_axis,
            const float min_corner[3],
            const float max_corner[3],
            float &best,
            float v_new[3])
        {
            const int ax1 = (free_axis + 1) % 3;
            const int ax2 = (free_axis + 2) % 3;
            // Candidate on one voxel edge: fix two coordinates to box bounds and
            // solve the remaining 1D minimizer along the free axis.
            const float a = Q[idx4(free_axis, free_axis)];
            const float b0 = Q[idx4(free_axis, ax1)];
            const float b1 = Q[idx4(free_axis, ax2)];
            const float b2 = Q[idx4(free_axis, 3)];

            for (int c0 = 0; c0 < 2; ++c0)
            {
                for (int c1 = 0; c1 < 2; ++c1)
                {
                    const float v0 = c0 == 0 ? min_corner[ax1] : max_corner[ax1];
                    const float v1 = c1 == 0 ? min_corner[ax2] : max_corner[ax2];
                    const float x = small_cpqr::solve_1x1_unchecked(a, -(b0 * v0 + b1 * v1 + b2));
                    if (x >= min_corner[free_axis] && x <= max_corner[free_axis])
                    {
                        float p4[4];
                        p4[free_axis] = x;
                        p4[ax1] = v0;
                        p4[ax2] = v1;
                        p4[3] = 1.0f;
                        const float err = qef_error4(Q, p4);
                        if (err < best)
                        {
                            best = err;
                            v_new[0] = p4[0];
                            v_new[1] = p4[1];
                            v_new[2] = p4[2];
                        }
                    }
                }
            }
        }

        __device__ __forceinline__ void try_three_constraint(
            const float Q[16],
            const float min_corner[3],
            const float max_corner[3],
            float &best,
            float v_new[3])
        {
            // Final fallback candidates are the eight box corners. This makes
            // the constrained search complete over the voxel box boundary.
            for (int cx = 0; cx < 2; ++cx)
            {
                for (int cy = 0; cy < 2; ++cy)
                {
                    for (int cz = 0; cz < 2; ++cz)
                    {
                        float p4[4];
                        p4[0] = cx ? min_corner[0] : max_corner[0];
                        p4[1] = cy ? min_corner[1] : max_corner[1];
                        p4[2] = cz ? min_corner[2] : max_corner[2];
                        p4[3] = 1.0f;
                        const float err = qef_error4(Q, p4);
                        if (err < best)
                        {
                            best = err;
                            v_new[0] = p4[0];
                            v_new[1] = p4[1];
                            v_new[2] = p4[2];
                        }
                    }
                }
            }
        }

        __global__ void gather_triangles_kernel(
            const float *__restrict__ vertices,
            const int32_t *__restrict__ faces,
            int64_t num_faces,
            float *__restrict__ triangles)
        {
            const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (tid >= 3 * num_faces)
                return;

            const int64_t f = tid / 3;
            const int lv = static_cast<int>(tid - 3 * f);
            const int32_t vid = faces[3 * f + lv];
            // One thread copies one face vertex, producing a dense [F,3,3]
            // triangle tensor consumed by the three QEF stages.
            triangles[3 * tid + 0] = vertices[3 * static_cast<int64_t>(vid) + 0];
            triangles[3 * tid + 1] = vertices[3 * static_cast<int64_t>(vid) + 1];
            triangles[3 * tid + 2] = vertices[3 * static_cast<int64_t>(vid) + 2];
        }

        __global__ void count_edges_kernel(
            int64_t num_faces,
            const int32_t *__restrict__ faces,
            uint64_t *__restrict__ hash_keys,
            uint32_t *__restrict__ edge_counts,
            int32_t *__restrict__ overflow,
            uint64_t hash_capacity)
        {
            const int64_t fid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (fid >= num_faces)
                return;

            const int32_t v[3] = {
                faces[3 * fid + 0],
                faces[3 * fid + 1],
                faces[3 * fid + 2],
            };
            for (int e = 0; e < 3; ++e)
            {
                int32_t a = v[e];
                int32_t b = v[(e + 1) % 3];
                if (a > b)
                {
                    const int32_t tmp = a;
                    a = b;
                    b = tmp;
                }
                // Count each undirected edge. Interior manifold edges are seen
                // twice; boundary edges are seen once.
                const uint64_t key = pack_edge_key(a, b);
                uint64_t slot = mix64(key) & (hash_capacity - 1);
                bool inserted = false;
                for (uint64_t probe = 0; probe < hash_capacity; ++probe)
                {
                    const uint64_t old = atomicCAS(
                        reinterpret_cast<unsigned long long *>(hash_keys + slot),
                        static_cast<unsigned long long>(kEmptyEdgeKey),
                        static_cast<unsigned long long>(key));
                    if (old == kEmptyEdgeKey || old == key)
                    {
                        atomicAdd(edge_counts + slot, 1u);
                        inserted = true;
                        break;
                    }
                    slot = (slot + 1u) & (hash_capacity - 1);
                }
                if (!inserted)
                    atomicExch(overflow, 1);
            }
        }

        __global__ void count_boundary_edges_kernel(
            uint64_t hash_capacity,
            const uint64_t *__restrict__ hash_keys,
            const uint32_t *__restrict__ edge_counts,
            uint32_t *__restrict__ boundary_count)
        {
            const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= hash_capacity)
                return;
            // A valid hash slot with count one is exactly one boundary edge.
            if (hash_keys[i] != kEmptyEdgeKey && edge_counts[i] == 1u)
                atomicAdd(boundary_count, 1u);
        }

        __global__ void emit_boundaries_kernel(
            uint64_t hash_capacity,
            const uint64_t *__restrict__ hash_keys,
            const uint32_t *__restrict__ edge_counts,
            const float *__restrict__ vertices,
            uint32_t *__restrict__ boundary_count,
            float *__restrict__ boundaries)
        {
            const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= hash_capacity)
                return;
            if (hash_keys[i] == kEmptyEdgeKey || edge_counts[i] != 1u)
                return;

            const uint32_t out = atomicAdd(boundary_count, 1u);
            const int32_t v0 = edge_key_v0(hash_keys[i]);
            const int32_t v1 = edge_key_v1(hash_keys[i]);
            // Emit the two endpoint positions in grid-local coordinates for the
            // boundary DDA kernel.
            boundaries[6 * static_cast<int64_t>(out) + 0] = vertices[3 * static_cast<int64_t>(v0) + 0];
            boundaries[6 * static_cast<int64_t>(out) + 1] = vertices[3 * static_cast<int64_t>(v0) + 1];
            boundaries[6 * static_cast<int64_t>(out) + 2] = vertices[3 * static_cast<int64_t>(v0) + 2];
            boundaries[6 * static_cast<int64_t>(out) + 3] = vertices[3 * static_cast<int64_t>(v1) + 0];
            boundaries[6 * static_cast<int64_t>(out) + 4] = vertices[3 * static_cast<int64_t>(v1) + 1];
            boundaries[6 * static_cast<int64_t>(out) + 5] = vertices[3 * static_cast<int64_t>(v1) + 2];
        }

        __global__ void solve_qef_kernel(
            const int32_t *__restrict__ voxels,
            const float *__restrict__ mean_sum,
            const float *__restrict__ cnt,
            const SymQEF10 *__restrict__ qefs,
            int64_t n,
            float3 voxel_size,
            float regularization_weight,
            float *__restrict__ dual_vertices)
        {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;

            const int x = voxels[3 * i + 0];
            const int y = voxels[3 * i + 1];
            const int z = voxels[3 * i + 2];
            // The dual vertex for this voxel must remain inside this voxel box.
            const float min_corner[3] = {
                x * voxel_size.x,
                y * voxel_size.y,
                z * voxel_size.z,
            };
            const float max_corner[3] = {
                (x + 1) * voxel_size.x,
                (y + 1) * voxel_size.y,
                (z + 1) * voxel_size.z,
            };

            float Q[16];
            sym10_to_dense4x4(qefs[i], Q);
            const float mean_i[3] = {
                mean_sum[3 * i + 0],
                mean_sum[3 * i + 1],
                mean_sum[3 * i + 2],
            };
            add_qef_regularization_inplace(Q, mean_i, cnt[i], regularization_weight);

            // First solve the unconstrained 3D minimizer of p^T Q p with p.w=1.
            const float A3[9] = {
                Q[idx4(0, 0)], Q[idx4(0, 1)], Q[idx4(0, 2)],
                Q[idx4(1, 0)], Q[idx4(1, 1)], Q[idx4(1, 2)],
                Q[idx4(2, 0)], Q[idx4(2, 1)], Q[idx4(2, 2)]};
            const float b3[3] = {-Q[idx4(0, 3)], -Q[idx4(1, 3)], -Q[idx4(2, 3)]};
            float v_new[3];
            small_cpqr::cpqr_solve_3x3(A3, b3, v_new);

            if (!point_inside_box3(v_new, min_corner, max_corner))
            {
                // If the best unconstrained point leaves the voxel, search the
                // voxel boundary: first faces, then edges, then corners. Each
                // candidate is scored with the full 4x4 QEF.
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

        torch::Tensor extract_boundaries_cuda(
            const torch::Tensor &vertices,
            const torch::Tensor &faces)
        {
            // Boundary extraction uses a hash table over undirected mesh edges.
            // An edge seen exactly once is a boundary edge and is emitted as a
            // [2, 3] segment for boundary_qef_cuda.
            const c10::cuda::CUDAGuard guard(vertices.device());
            const cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
            const torch::Device device = vertices.device();
            const int64_t num_faces = faces.size(0);
            const auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(device);
            const auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(device);
            const auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
            const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
            if (num_faces == 0)
                return torch::empty({0, 2, 3}, opts_f32);

            const int64_t num_edges = num_faces * 3;
            const int64_t hash_capacity_i64 = next_power_of_two_i64(num_edges * 2 > 2 ? num_edges * 2 : 2);
            auto hash_keys = torch::empty({hash_capacity_i64}, opts_u64);
            auto edge_counts = torch::zeros({hash_capacity_i64}, opts_u32);
            auto boundary_count_t = torch::zeros({1}, opts_u32);
            auto overflow = torch::zeros({1}, opts_i32);
            C10_CUDA_CHECK(cudaMemsetAsync(hash_keys.data_ptr<uint64_t>(), 0xff, hash_capacity_i64 * sizeof(uint64_t), stream));

            int blocks = static_cast<int>(div_up_i64(num_faces, kThreads));
            count_edges_kernel<<<blocks, kThreads, 0, stream>>>(
                num_faces,
                faces.data_ptr<int32_t>(),
                hash_keys.data_ptr<uint64_t>(),
                edge_counts.data_ptr<uint32_t>(),
                overflow.data_ptr<int32_t>(),
                static_cast<uint64_t>(hash_capacity_i64));
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            blocks = static_cast<int>(div_up_i64(hash_capacity_i64, kThreads));
            count_boundary_edges_kernel<<<blocks, kThreads, 0, stream>>>(
                static_cast<uint64_t>(hash_capacity_i64),
                hash_keys.data_ptr<uint64_t>(),
                edge_counts.data_ptr<uint32_t>(),
                boundary_count_t.data_ptr<uint32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            int32_t overflow_h = 0;
            uint32_t boundary_count = 0;
            C10_CUDA_CHECK(cudaMemcpyAsync(&overflow_h, overflow.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaMemcpyAsync(&boundary_count, boundary_count_t.data_ptr<uint32_t>(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            TORCH_CHECK(overflow_h == 0, "edge hash overflow while extracting boundaries");

            auto boundaries = torch::empty({static_cast<int64_t>(boundary_count), 2, 3}, opts_f32);
            if (boundary_count == 0)
                return boundaries;

            C10_CUDA_CHECK(cudaMemsetAsync(boundary_count_t.data_ptr<uint32_t>(), 0, sizeof(uint32_t), stream));
            emit_boundaries_kernel<<<blocks, kThreads, 0, stream>>>(
                static_cast<uint64_t>(hash_capacity_i64),
                hash_keys.data_ptr<uint64_t>(),
                edge_counts.data_ptr<uint32_t>(),
                vertices.data_ptr<float>(),
                boundary_count_t.data_ptr<uint32_t>(),
                boundaries.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return boundaries;
        }

    } // namespace

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    mesh_to_flexible_dual_grid_cuda(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range,
        float face_weight,
        float boundary_weight,
        float regularization_weight)
    {
        // CUDA version of the Python/CPU flexible dual grid entry point:
        // vertices/faces -> triangles -> intersection QEF -> face QEF ->
        // boundary QEF -> constrained solve.
        TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
        TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
        static_assert(sizeof(SymQEF10) == sizeof(float) * 10, "Unexpected SymQEF10 layout");

        const c10::cuda::CUDAGuard guard(vertices.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        const torch::Device device = vertices.device();
        const auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        const int64_t num_faces = faces.size(0);
        auto triangles = torch::empty({num_faces, 3, 3}, opts_f32);
        if (num_faces > 0)
        {
            const int blocks = static_cast<int>(div_up_i64(num_faces * 3, kThreads));
            gather_triangles_kernel<<<blocks, kThreads, 0, stream>>>(
                vertices.data_ptr<float>(),
                faces.data_ptr<int32_t>(),
                num_faces,
                triangles.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        auto intersect = intersect_qef_cuda(triangles, voxel_size, grid_range);
        torch::Tensor voxels = std::get<0>(intersect);
        torch::Tensor mean_sum = std::get<1>(intersect);
        torch::Tensor cnt = std::get<2>(intersect);
        torch::Tensor intersected = std::get<3>(intersect);
        torch::Tensor total_qefs = std::get<4>(intersect);
        torch::Tensor brick_hash_keys = std::get<5>(intersect);
        torch::Tensor brick_hash_vals = std::get<6>(intersect);
        torch::Tensor brick_bits = std::get<7>(intersect);
        torch::Tensor brick_base = std::get<8>(intersect);

        const int64_t num_voxels = voxels.size(0);
        if (num_voxels == 0)
        {
            auto dual_vertices = torch::empty({0, 3}, opts_f32);
            return std::make_tuple(voxels, dual_vertices, intersected);
        }

        if (face_weight > 0.0f)
            face_qef_cuda(
                triangles,
                voxel_size,
                grid_range,
                voxels,
                total_qefs,
                face_weight,
                brick_hash_keys,
                brick_hash_vals,
                brick_bits,
                brick_base);

        if (boundary_weight > 0.0f)
        {
            torch::Tensor boundaries = extract_boundaries_cuda(vertices, faces);
            if (boundaries.size(0) > 0)
                boundary_qef_cuda(
                    boundaries,
                    voxel_size,
                    grid_range,
                    boundary_weight,
                    voxels,
                    total_qefs,
                    brick_hash_keys,
                    brick_hash_vals,
                    brick_bits,
                    brick_base);
        }

        int blocks = static_cast<int>(div_up_i64(num_voxels, kThreads));
        const float3 voxel_size_h = make_float3(voxel_size[0], voxel_size[1], voxel_size[2]);
        auto dual_vertices = torch::empty({num_voxels, 3}, opts_f32);
        solve_qef_kernel<<<blocks, kThreads, 0, stream>>>(
            voxels.data_ptr<int32_t>(),
            mean_sum.data_ptr<float>(),
            cnt.data_ptr<float>(),
            reinterpret_cast<const SymQEF10 *>(total_qefs.data_ptr<float>()),
            num_voxels,
            voxel_size_h,
            regularization_weight,
            dual_vertices.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return std::make_tuple(voxels, dual_vertices, intersected);
    }

} // namespace o_voxel::fdg
