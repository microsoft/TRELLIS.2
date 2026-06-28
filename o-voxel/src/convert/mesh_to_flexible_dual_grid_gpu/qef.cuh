#pragma once

#include "types.cuh"

namespace o_voxel::fdg
{

    __host__ __device__ __forceinline__ SymQEF10 qef_zero()
    {
        return SymQEF10{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Small value helpers kept host/device so the same QEF math can be used in
    // kernels and in any light host-side setup code.
    __host__ __device__ __forceinline__ SymQEF10 qef_add(
        const SymQEF10 &a,
        const SymQEF10 &b)
    {
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

    // Scaling folds a stage weight into a QEF before it is accumulated into the
    // running per-voxel total.
    __host__ __device__ __forceinline__ SymQEF10 qef_scale(
        const SymQEF10 &q,
        float s)
    {
        return SymQEF10{
            q.q00 * s,
            q.q01 * s,
            q.q02 * s,
            q.q03 * s,
            q.q11 * s,
            q.q12 * s,
            q.q13 * s,
            q.q22 * s,
            q.q23 * s,
            q.q33 * s,
        };
    }

    __host__ __device__ __forceinline__ SymQEF10 qef_from_plane(float4 p)
    {
        // Plane ax + by + cz + d = 0 contributes p * p^T in SymQEF10 layout.
        const float a = p.x;
        const float b = p.y;
        const float c = p.z;
        const float d = p.w;
        return SymQEF10{
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        };
    }

} // namespace o_voxel::fdg
