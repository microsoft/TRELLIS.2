#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace fdg_gpu {

inline void throw_cuda_error(cudaError_t error, const char* context) {
    if (error == cudaSuccess) return;
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(error));
}

struct int2_ {
    int x;
    int y;
};

struct int3_ {
    int x;
    int y;
    int z;

    __host__ __device__ int& operator[](int i) { return (&x)[i]; }
    __host__ __device__ int operator[](int i) const { return (&x)[i]; }
};

struct bool3_ {
    bool x;
    bool y;
    bool z;

    __host__ __device__ bool& operator[](int i) { return (&x)[i]; }
    __host__ __device__ bool operator[](int i) const { return (&x)[i]; }
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(int64_t count) { allocate(count); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), owns_(other.owns_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.owns_ = true;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            owns_ = other.owns_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.owns_ = true;
        }
        return *this;
    }

    void allocate(int64_t count) {
        if (count < 0) {
            throw std::invalid_argument("DeviceBuffer::allocate count must be non-negative");
        }
        release();
        size_ = count;
        owns_ = true;
        if (count == 0) return;
        throw_cuda_error(cudaMalloc(reinterpret_cast<void**>(&ptr_), static_cast<size_t>(count) * sizeof(T)),
                         "cudaMalloc failed in DeviceBuffer::allocate");
    }

    void adopt(T* ptr, int64_t count) {
        release();
        ptr_ = ptr;
        size_ = count;
        owns_ = true;
    }

    void clear_async(cudaStream_t stream = nullptr) {
        if (size_ == 0) return;
        throw_cuda_error(cudaMemsetAsync(ptr_, 0, static_cast<size_t>(size_) * sizeof(T), stream),
                         "cudaMemsetAsync failed in DeviceBuffer::clear_async");
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    int64_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    T* release_ownership() noexcept {
        T* out = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        owns_ = true;
        return out;
    }

private:
    void release() noexcept {
        if (ptr_ != nullptr && owns_) {
            cudaFree(ptr_);
        }
        ptr_ = nullptr;
        size_ = 0;
        owns_ = true;
    }

    T* ptr_ = nullptr;
    int64_t size_ = 0;
    bool owns_ = true;
};

struct SymQEF10 {
    float q00, q01, q02, q03;
    float q11, q12, q13;
    float q22, q23;
    float q33;
};

struct PrimitivePairResult {
    int64_t size = 0;
    DeviceBuffer<int32_t> prim_id;
    DeviceBuffer<int32_t> voxel_i;
    DeviceBuffer<int32_t> voxel_j;
    DeviceBuffer<int32_t> voxel_k;
};

__host__ __device__ __forceinline__ int ceil_div_i64(int64_t n, int block) {
    return static_cast<int>((n + block - 1) / block);
}

__host__ __device__ __forceinline__ uint64_t pack_voxel_key(
    int x, int y, int z, int3_ grid_min, int3_ grid_max) {
    const uint64_t sx = static_cast<uint64_t>(grid_max.x - grid_min.x);
    const uint64_t sy = static_cast<uint64_t>(grid_max.y - grid_min.y);
    const uint64_t ux = static_cast<uint64_t>(x - grid_min.x);
    const uint64_t uy = static_cast<uint64_t>(y - grid_min.y);
    const uint64_t uz = static_cast<uint64_t>(z - grid_min.z);
    return ux + sx * (uy + sy * uz);
}

} // namespace fdg_gpu
