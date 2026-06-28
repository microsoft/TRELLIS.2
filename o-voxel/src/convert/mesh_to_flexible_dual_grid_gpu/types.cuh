#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace o_voxel::fdg
{

    struct Int3
    {
        int x;
        int y;
        int z;

        __host__ __device__ int &operator[](int i) { return (&x)[i]; }
        __host__ __device__ int operator[](int i) const { return (&x)[i]; }
    };

    struct GridSpec
    {
        // Grid-local coordinates use voxel indices in [grid_min, grid_max).
        float3 voxel_size;
        Int3 grid_min;
        Int3 grid_max;
    };

    // Upper-triangle storage for a symmetric 4x4 QEF matrix:
    // q00 q01 q02 q03
    //     q11 q12 q13
    //         q22 q23
    //             q33
    struct SymQEF10
    {
        float q00;
        float q01;
        float q02;
        float q03;
        float q11;
        float q12;
        float q13;
        float q22;
        float q23;
        float q33;
    };

    // Active voxels are grouped into 8x8x8 bricks. Each brick stores 512
    // occupancy bits as 16 uint32 words, and active bricks are addressed by an
    // open-addressing hash table.
    inline constexpr int kBrickSize = 8;
    inline constexpr int kBrickLocalCells = kBrickSize * kBrickSize * kBrickSize;
    inline constexpr int kBrickBitWords = kBrickLocalCells / 32;
    inline constexpr uint64_t kEmptyBrickKey = UINT64_MAX;
    inline constexpr uint32_t kEmptyBrickVal = UINT32_MAX;
    inline constexpr uint32_t kOverflowBrickVal = UINT32_MAX - 1u;

    // Lookup data returned by intersect_qef_cuda. hash_keys/hash_vals map a
    // brick key to a compact brick index, brick_bits stores local occupancy, and
    // brick_base gives the first row for that brick in compacted voxel/QEF arrays.
    struct BrickLookup
    {
        const uint64_t *hash_keys;
        const uint32_t *hash_vals;
        const uint32_t *brick_bits;
        const int64_t *brick_base;
        uint64_t hash_capacity;
    };

    __host__ __device__ __forceinline__ uint64_t pack_voxel_key(
        int x,
        int y,
        int z,
        Int3 grid_min,
        Int3 grid_max)
    {
        // Dense row-major key for a voxel coordinate inside the current grid.
        const uint64_t sx = static_cast<uint64_t>(grid_max.x - grid_min.x);
        const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
        const uint64_t ux = static_cast<uint64_t>(x - grid_min.x);
        const uint64_t uy = static_cast<uint64_t>(y - grid_min.y);
        const uint64_t uz = static_cast<uint64_t>(z - grid_min.z);
        return ux + sx * (uy + sy * uz);
    }

    __host__ __device__ __forceinline__ Int3 unpack_voxel_key(
        uint64_t key,
        Int3 grid_min,
        Int3 grid_max)
    {
        // Inverse of pack_voxel_key, used when compact voxel rows are emitted.
        const uint64_t sx = static_cast<uint64_t>(grid_max.x - grid_min.x);
        const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
        const uint64_t yz = sx * sy;
        const uint64_t z = key / yz;
        const uint64_t rem = key - z * yz;
        const uint64_t y = rem / sx;
        const uint64_t x = rem - y * sx;
        return Int3{
            static_cast<int>(x) + grid_min.x,
            static_cast<int>(y) + grid_min.y,
            static_cast<int>(z) + grid_min.z,
        };
    }

    __device__ __forceinline__ bool lookup_brick_bits_and_base(
        int bx,
        int by,
        int bz,
        GridSpec grid,
        BrickLookup lookup,
        const uint32_t **bits,
        int64_t *base)
    {
        // Find one active brick and return both its occupancy bitset and its
        // first compact voxel row. Callers can then test local bits cheaply.
        if (lookup.hash_capacity == 0)
            return false;

        const uint64_t nbx = static_cast<uint64_t>(
            (grid.grid_max.x - grid.grid_min.x + kBrickSize - 1) / kBrickSize);
        const uint64_t nby = static_cast<uint64_t>(
            (grid.grid_max.y - grid.grid_min.y + kBrickSize - 1) / kBrickSize);
        const uint64_t key =
            static_cast<uint64_t>(bx) + nbx * (static_cast<uint64_t>(by) + nby * static_cast<uint64_t>(bz));

        uint64_t slot_key = key;
        slot_key ^= slot_key >> 33;
        slot_key *= 0xff51afd7ed558ccdULL;
        slot_key ^= slot_key >> 33;
        slot_key *= 0xc4ceb9fe1a85ec53ULL;
        slot_key ^= slot_key >> 33;

        uint64_t slot = slot_key & (lookup.hash_capacity - 1);
        for (uint64_t probe = 0; probe < lookup.hash_capacity; ++probe)
        {
            const uint64_t found = lookup.hash_keys[slot];
            if (found == kEmptyBrickKey)
                return false;
            if (found == key)
            {
                const uint32_t brick_idx = lookup.hash_vals[slot];
                if (brick_idx == kEmptyBrickVal || brick_idx == kOverflowBrickVal)
                    return false;
                *bits = lookup.brick_bits + static_cast<int64_t>(brick_idx) * kBrickBitWords;
                *base = lookup.brick_base[brick_idx];
                return true;
            }
            slot = (slot + 1u) & (lookup.hash_capacity - 1);
        }
        return false;
    }

    __device__ __forceinline__ int64_t lookup_voxel_row_in_bricks(
        int x,
        int y,
        int z,
        GridSpec grid,
        BrickLookup lookup)
    {
        // Returns the row in compacted voxels/qefs, or -1 when the voxel is not active.
        if (lookup.hash_capacity == 0)
            return -1;
        if (x < grid.grid_min.x || x >= grid.grid_max.x)
            return -1;
        if (y < grid.grid_min.y || y >= grid.grid_max.y)
            return -1;
        if (z < grid.grid_min.z || z >= grid.grid_max.z)
            return -1;

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

        const uint32_t *bits;
        int64_t base;
        if (!lookup_brick_bits_and_base(bx, by, bz, grid, lookup, &bits, &base))
            return -1;
        const int word = local_id / 32;
        const int bit = local_id - word * 32;
        if ((bits[word] & (1u << bit)) == 0)
            return -1;

        // Rank is the number of active local voxels before local_id. Since
        // emit_occupied_voxels_kernel writes voxels in the same bit order, this
        // reproduces the compact row without a per-voxel hash table.
        int rank = 0;
        for (int i = 0; i < word; ++i)
            rank += __popc(bits[i]);
        const uint32_t mask = bit == 0 ? 0u : ((1u << bit) - 1u);
        rank += __popc(bits[word] & mask);
        return base + rank;
    }

} // namespace o_voxel::fdg
