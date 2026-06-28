#pragma once

#include <cfloat>
#include <cmath>

namespace fdg_gpu::small_cpqr {
namespace detail {

template<int N>
__device__ __forceinline__ float absf(float x) {
    return x < 0.0f ? -x : x;
}

template<int N>
__device__ __forceinline__ void swap_cols(
    float* qr,
    float* col_norms_updated,
    float* col_norms_direct,
    int* perm,
    int c0,
    int c1) {
    if (c0 == c1) return;
    #pragma unroll
    for (int r = 0; r < N; ++r) {
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

template<int N>
__device__ __forceinline__ void make_householder_real(
    float x0,
    const float* tail_in,
    int tail_len,
    float* beta,
    float* tau,
    float* essential_out) {
    float tail_sq_norm = 0.0f;
    #pragma unroll
    for (int i = 0; i < N - 1; ++i) {
        if (i < tail_len) {
            tail_sq_norm += tail_in[i] * tail_in[i];
        }
    }

    const float tol = FLT_MIN;
    if (tail_sq_norm <= tol) {
        *beta = x0;
        *tau = 0.0f;
        #pragma unroll
        for (int i = 0; i < N - 1; ++i) {
            essential_out[i] = 0.0f;
        }
        return;
    }

    float b = sqrtf(x0 * x0 + tail_sq_norm);
    if (x0 >= 0.0f) {
        b = -b;
    }
    const float denom = x0 - b;
    #pragma unroll
    for (int i = 0; i < N - 1; ++i) {
        essential_out[i] = (i < tail_len) ? (tail_in[i] / denom) : 0.0f;
    }
    *beta = b;
    *tau = (b - x0) / b;
}

template<int N>
__device__ __forceinline__ void apply_householder_left_matrix(
    float* qr,
    int row0,
    int col0,
    const float* essential,
    int tail_len,
    float tau) {
    if (tau == 0.0f) return;
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        if (j < col0) continue;
        float tmp = qr[row0 * N + j];
        #pragma unroll
        for (int i = 0; i < N - 1; ++i) {
            if (i < tail_len) {
                tmp += essential[i] * qr[(row0 + 1 + i) * N + j];
            }
        }
        qr[row0 * N + j] -= tau * tmp;
        #pragma unroll
        for (int i = 0; i < N - 1; ++i) {
            if (i < tail_len) {
                qr[(row0 + 1 + i) * N + j] -= tau * essential[i] * tmp;
            }
        }
    }
}

template<int N>
__device__ __forceinline__ void apply_householder_left_vector(
    float* c,
    int row0,
    const float* essential,
    int tail_len,
    float tau) {
    if (tau == 0.0f) return;
    float tmp = c[row0];
    #pragma unroll
    for (int i = 0; i < N - 1; ++i) {
        if (i < tail_len) {
            tmp += essential[i] * c[row0 + 1 + i];
        }
    }
    c[row0] -= tau * tmp;
    #pragma unroll
    for (int i = 0; i < N - 1; ++i) {
        if (i < tail_len) {
            c[row0 + 1 + i] -= tau * essential[i] * tmp;
        }
    }
}

template<int N>
__device__ __forceinline__ void backsolve_upper_ranked(
    const float* qr,
    int rank,
    const float* c,
    const int* perm,
    float* x_out) {
    float y[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        y[i] = 0.0f;
        x_out[i] = 0.0f;
    }
    for (int i = rank - 1; i >= 0; --i) {
        float s = c[i];
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            if (j > i && j < rank) {
                s -= qr[i * N + j] * y[j];
            }
        }
        y[i] = s / qr[i * N + i];
    }
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        if (i < rank) {
            x_out[perm[i]] = y[i];
        } else {
            x_out[perm[i]] = 0.0f;
        }
    }
}

template<int N>
__device__ __forceinline__ void cpqr_solve_small_impl(
    const float* A_in,
    const float* b_in,
    float* x_out) {
    float qr[N * N];
    float c[N];
    int perm[N];
    float col_norms_direct[N];
    float col_norms_updated[N];
    float essential[N > 1 ? N - 1 : 1];

    #pragma unroll
    for (int i = 0; i < N * N; ++i) {
        qr[i] = A_in[i];
    }
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        c[i] = b_in[i];
        perm[i] = i;
        x_out[i] = 0.0f;
    }

    #pragma unroll
    for (int j = 0; j < N; ++j) {
        float norm_sq = 0.0f;
        #pragma unroll
        for (int r = 0; r < N; ++r) {
            const float v = qr[r * N + j];
            norm_sq += v * v;
        }
        const float norm = sqrtf(norm_sq);
        col_norms_direct[j] = norm;
        col_norms_updated[j] = norm;
    }

    float max_norm_updated = col_norms_updated[0];
    #pragma unroll
    for (int j = 1; j < N; ++j) {
        if (col_norms_updated[j] > max_norm_updated) {
            max_norm_updated = col_norms_updated[j];
        }
    }

    const float threshold_helper = (max_norm_updated * FLT_EPSILON) * (max_norm_updated * FLT_EPSILON) / float(N);
    const float norm_downdate_threshold = sqrtf(FLT_EPSILON);
    int nonzero_pivots = N;
    float maxpivot = 0.0f;

    #pragma unroll
    for (int k = 0; k < N; ++k) {
        int biggest_col_index = k;
        float best_updated = col_norms_updated[k];
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            if (j > k && col_norms_updated[j] > best_updated) {
                best_updated = col_norms_updated[j];
                biggest_col_index = j;
            }
        }
        const float biggest_col_sq_norm = best_updated * best_updated;
        if (nonzero_pivots == N && biggest_col_sq_norm < threshold_helper * float(N - k)) {
            nonzero_pivots = k;
        }

        swap_cols<N>(qr, col_norms_updated, col_norms_direct, perm, k, biggest_col_index);

        const int tail_len = N - k - 1;
        float tail_local[N > 1 ? N - 1 : 1];
        #pragma unroll
        for (int i = 0; i < N - 1; ++i) {
            tail_local[i] = (i < tail_len) ? qr[(k + 1 + i) * N + k] : 0.0f;
        }

        float beta = 0.0f;
        float tau = 0.0f;
        make_householder_real<N>(qr[k * N + k], tail_local, tail_len, &beta, &tau, essential);

        qr[k * N + k] = beta;
        #pragma unroll
        for (int i = 0; i < N - 1; ++i) {
            if (i < tail_len) {
                qr[(k + 1 + i) * N + k] = essential[i];
            }
        }
        const float abs_beta = absf<N>(beta);
        if (abs_beta > maxpivot) {
            maxpivot = abs_beta;
        }

        apply_householder_left_matrix<N>(qr, k, k + 1, essential, tail_len, tau);
        if (k < nonzero_pivots) {
            apply_householder_left_vector<N>(c, k, essential, tail_len, tau);
        }

        #pragma unroll
        for (int j = 0; j < N; ++j) {
            if (j <= k) continue;
            if (col_norms_updated[j] != 0.0f) {
                float temp = absf<N>(qr[k * N + j]) / col_norms_updated[j];
                temp = (1.0f + temp) * (1.0f - temp);
                if (temp < 0.0f) temp = 0.0f;
                const float ratio = col_norms_updated[j] / col_norms_direct[j];
                const float temp2 = temp * ratio * ratio;
                if (temp2 <= norm_downdate_threshold) {
                    float norm_sq = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < N; ++r) {
                        if (r > k) {
                            const float v = qr[r * N + j];
                            norm_sq += v * v;
                        }
                    }
                    const float norm = sqrtf(norm_sq);
                    col_norms_direct[j] = norm;
                    col_norms_updated[j] = norm;
                } else {
                    col_norms_updated[j] *= sqrtf(temp);
                }
            }
        }
    }

    if (nonzero_pivots == 0) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            x_out[i] = 0.0f;
        }
        return;
    }

    backsolve_upper_ranked<N>(qr, nonzero_pivots, c, perm, x_out);
}

} // namespace detail

__device__ __forceinline__ void cpqr_solve_3x3(const float A[9], const float b[3], float x[3]) {
    detail::cpqr_solve_small_impl<3>(A, b, x);
}

__device__ __forceinline__ void cpqr_solve_2x2(const float A[4], const float b[2], float x[2]) {
    detail::cpqr_solve_small_impl<2>(A, b, x);
}

__device__ __forceinline__ float solve_1x1_unchecked(float a, float rhs) {
    return rhs / a;
}

} // namespace fdg_gpu::small_cpqr
