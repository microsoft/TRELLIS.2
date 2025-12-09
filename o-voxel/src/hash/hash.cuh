// 32 bit Murmur3 hash
__forceinline__ __device__ size_t hash(uint32_t k, size_t N) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % N;
}


// 64 bit Murmur3 hash
__forceinline__ __device__ size_t hash(uint64_t k, size_t N) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k % N;
}


template<typename K, typename V>
__forceinline__ __device__ void linear_probing_insert(
    K* hashmap_keys,
    V* hashmap_values,
    const K key,
    const V value,
    const size_t N
) {
    size_t slot = hash(key, N);
    while (true) {
        K prev = atomicCAS(&hashmap_keys[slot], std::numeric_limits<K>::max(), key);
        if (prev == std::numeric_limits<K>::max() || prev == key) {
            hashmap_values[slot] = value;
            return;
        }
        slot = slot + 1;
        if (slot >= N) slot = 0;
    }
}


template<typename V>
__forceinline__ __device__ void linear_probing_insert(
    uint64_t* hashmap_keys,
    V* hashmap_values,
    const uint64_t key,
    const V value,
    const size_t N
) {
    size_t slot = hash(key, N);
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hashmap_keys[slot]),
            static_cast<unsigned long long>(std::numeric_limits<uint64_t>::max()),
            static_cast<unsigned long long>(key)
        );
        if (prev == std::numeric_limits<uint64_t>::max() || prev == key) {
            hashmap_values[slot] = value;
            return;
        }
        slot = (slot + 1) % N;
    }
}


template<typename K, typename V>
__forceinline__ __device__ V linear_probing_lookup(
    const K* hashmap_keys,
    const V* hashmap_values,
    const K key,
    const size_t N
) {
    size_t slot = hash(key, N);
    while (true) {
        K prev = hashmap_keys[slot];
        if (prev == std::numeric_limits<K>::max()) {
            return std::numeric_limits<V>::max();
        }
        if (prev == key) {
            return hashmap_values[slot];
        }
        slot = slot + 1;
        if (slot >= N) slot = 0;
    }
}
