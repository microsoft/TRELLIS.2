#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <ctime>

#include "api.h"


constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();


static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}


template <typename T, typename U>
static inline U lerp(const T& a, const T& b, const T& t, const U& val_a, const U& val_b) {
    if (a == b) return val_a; // Avoid divide by zero
    T alpha = (t - a) / (b - a);
    return (1 - alpha) * val_a + alpha * val_b;
}


template <typename Map, typename Key, typename Default>
static auto get_or_default(const Map& map, const Key& key, const Default& default_val) -> typename Map::mapped_type {
    auto it = map.find(key);
    return (it != map.end()) ? it->second : default_val;
}


// 3D voxel coordinate
struct VoxelCoord {
    int x, y, z;

    int& operator[](int i) {
        return (&x)[i];
    }

    bool operator==(const VoxelCoord& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Hash function for VoxelCoord to use in unordered_map
namespace std {
template <>
struct hash<VoxelCoord> {
    size_t operator()(const VoxelCoord& v) const {
        const std::size_t p1 = 73856093;
        const std::size_t p2 = 19349663;
        const std::size_t p3 = 83492791;
        return (std::size_t)(v.x) * p1 ^ (std::size_t)(v.y) * p2 ^ (std::size_t)(v.z) * p3;
    }
};
}


/**
 * Compute the Normal Tangent and Bitangent vectors for a triangle.
 * 
 * @param v0 The first vertex of the triangle.
 * @param v1 The second vertex of the triangle.
 * @param v2 The third vertex of the triangle.
 * @param uv0 The texture coordinates of the first vertex.
 * @param uv1 The texture coordinates of the second vertex.
 * @param uv2 The texture coordinates of the third vertex.
 * 
 * @return A tuple containing:
 *  - t The tangent vector.
 *  - b The bitangent vector.
 *  - n The normal vector.
 *  - mip_length The norms of the partial derivatives of the 3D coordinates with respect to the 2D texture coordinates.
 */
static std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector2f> compute_TBN(
    const Eigen::Vector3f& v0,
    const Eigen::Vector3f& v1,
    const Eigen::Vector3f& v2,
    const Eigen::Vector2f& uv0,
    const Eigen::Vector2f& uv1,
    const Eigen::Vector2f& uv2
) {
    Eigen::Vector3f e1 = v1 - v0;
    Eigen::Vector3f e2 = v2 - v0;
    Eigen::Vector2f duv1 = uv1 - uv0;
    Eigen::Vector2f duv2 = uv2 - uv0;
    Eigen::Vector3f n = e1.cross(e2).normalized();

    float det = duv1.x() * duv2.y() - duv1.y() * duv2.x();
    if (fabs(det) < 1e-6) {
        // Use default
        Eigen::Vector3f t(1.0f, 0.0f, 0.0f);
        Eigen::Vector3f b(0.0f, 1.0f, 0.0f);
        Eigen::Vector2f mip_length(1e6, 1e6);
        return std::make_tuple(t, b, n, mip_length);
    }

    float invDet = 1.0f / det;
    Eigen::Vector3f t = (duv2.y() * e1 - duv1.y() * e2);
    Eigen::Vector3f b = (duv1.x() * e2 - duv2.x() * e1);
    float t_norm = t.norm();
    float b_norm = b.norm();
    t = t / t_norm;
    b = b / b_norm;
    Eigen::Vector2f mip_length(invDet * t_norm, invDet * b_norm);

    return std::make_tuple(t, b, n, mip_length);
}


/**
 * Project a point onto a triangle defined by three vertices.
 * 
 * @param p The point to project.
 * @param a The first vertex of the triangle.
 * @param b The second vertex of the triangle.
 * @param c The third vertex of the triangle.
 * @param n The normal of the triangle.
 * 
 * @return The projected point represented as barycentric coordinates (u, v, w) and distance from the plane.
 */
static Eigen::Vector4f project_onto_triangle(
    const Eigen::Vector3f& p,
    const Eigen::Vector3f& a,
    const Eigen::Vector3f& b,
    const Eigen::Vector3f& c,
    const Eigen::Vector3f& n
) {
    float d = (p - a).dot(n);

    Eigen::Vector3f p_proj = p - d * n;
    Eigen::Vector3f ab = b - a;
    Eigen::Vector3f ac = c - a;
    Eigen::Vector3f ap = p_proj - a;

    float d00 = ab.dot(ab);
    float d01 = ab.dot(ac);
    float d11 = ac.dot(ac);
    float d20 = ap.dot(ab);
    float d21 = ap.dot(ac);

    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return Eigen::Vector4f(u, v, w, d);
}


static inline int wrap_texcoord(const int& x, const int& W, const int& filter) {
    if (filter == 0) {          // REPEAT
        return (x % W + W) % W;
    } else if (filter == 1) {   // CLAMP_TO_EDGE
        return std::max(0, std::min(x, W - 1));
    } else if (filter == 2) {   // MIRROR_REPEAT
        int period = 2 * W;
        int x_mod = (x % period + period) % period;
        return (x_mod < W) ? x_mod : (period - x_mod - 1);
    } else {
        // Default to repeat
        return (x % W + W) % W;
    }
}


static std::vector<std::vector<uint8_t>> build_mipmaps(
    const uint8_t* texture,
    const int& H, const int& W, const int& C
) {
    if (H != W || !is_power_of_two(H)) {
        throw std::invalid_argument("Texture width and height must be equal and a power of two.");
    }
    std::vector<std::vector<uint8_t>> mipmaps;
    const uint8_t* cur_map = texture;
    int cur_H = H;
    int cur_W = W;
    int next_H = cur_H >> 1;
    int next_W = cur_W >> 1;
    while (next_H > 0 && next_W > 0) {   
        std::vector<uint8_t> next_map(next_H * next_W * C);
        for (int y = 0; y < next_H; y++) {
            for (int x = 0; x < next_W; x++) {
                for (int c = 0; c < C; c++) {
                    size_t sum = 0;
                    size_t xx = static_cast<size_t>(x) << 1;
                    size_t yy = static_cast<size_t>(y) << 1;
                    sum += cur_map[yy * static_cast<size_t>(cur_W) * C + xx * C + c];
                    sum += cur_map[(yy + 1) * static_cast<size_t>(cur_W) * C + xx * C + c];
                    sum += cur_map[yy * static_cast<size_t>(cur_W) * C + (xx + 1) * C + c];
                    sum += cur_map[(yy + 1) * static_cast<size_t>(cur_W) * C + (xx + 1) * C + c];
                    next_map[y * next_W * C + x * C + c] = static_cast<uint8_t>(sum / 4);
                }
            }
        }
        mipmaps.push_back(std::move(next_map));
        cur_map = mipmaps.back().data();
        cur_H = next_H;
        cur_W = next_W;
        next_H = cur_H >> 1;
        next_W = cur_W >> 1;
    }
    return mipmaps;
}


static void sample_texture(
    const uint8_t* texture,
    const int& H, const int& W, const int& C,
    const float& u, const float& v,
    const int& filter, const int& wrap,
    float* color
) {
    float x = u * W;
    float y = (1 - v) * H;
    if (filter == 0) {      // NEAREST
        int x_int = floorf(x);
        int y_int = floorf(y);
        x_int = wrap_texcoord(x_int, W, wrap);
        y_int = wrap_texcoord(y_int, H, wrap);
        for (int c = 0; c < C; c++) {
            color[c] = texture[y_int * W * C + x_int * C + c] / 255.0f;
        }
    }
    else {                  // LINEAR
        int x_low = floorf(x - 0.5);
        int x_high = x_low + 1;
        int y_low = floorf(y - 0.5);
        int y_high = y_low + 1;
        float w_x = x - x_low - 0.5;
        float w_y = y - y_low - 0.5;
        x_low = wrap_texcoord(x_low, W, wrap);
        x_high = wrap_texcoord(x_high, W, wrap);
        y_low = wrap_texcoord(y_low, H, wrap);
        y_high = wrap_texcoord(y_high, H, wrap);
        for (int c = 0; c < C; c++) {
            color[c] = (1 - w_x) * (1 - w_y) * texture[y_low * W * C + x_low * C + c] +
                    w_x * (1 - w_y) * texture[y_low * W * C + x_high * C + c] +
                    (1 - w_x) * w_y * texture[y_high * W * C + x_low * C + c] +
                    w_x * w_y * texture[y_high * W * C + x_high * C + c];
            color[c] /= 255.0f;
        }
    }
}


static void sample_texture_mipmap(
    const uint8_t* texture,
    const int& H, const int& W, const int& C,
    const std::vector<std::vector<uint8_t>>& mipmaps,
    const float& u, const float& v, const float& mip_length, const float& mipLevelOffset,
    const int& filter, const int& wrap,
    float* color
) {
    if (filter == 0) {      // NEAREST
        sample_texture(texture, H, W, C, u, v, filter, wrap, color);
    }
    else {                  // LINEAR
        float mip_level = std::log2(mip_length * H) + mipLevelOffset;
        if (!std::isfinite(mip_level) || mip_level <= 0 || mipmaps.empty()) {
            sample_texture(texture, H, W, C, u, v, filter, wrap, color);
        }
        else if (mip_level >= mipmaps.size()) {
            sample_texture(mipmaps[mipmaps.size() - 1].data(), H >> mipmaps.size(), W >> mipmaps.size(), C, u, v, filter, wrap, color);
        }
        else {
            int lower_mip_level = std::floor(mip_level);
            int upper_mip_level = lower_mip_level + 1;
            float mip_frac = mip_level - lower_mip_level;
            const uint8_t* lower_mip_ptr = lower_mip_level == 0 ? texture : mipmaps[lower_mip_level - 1].data();
            const uint8_t* upper_mip_ptr = mipmaps[upper_mip_level - 1].data();
            int lower_mip_H = H >> lower_mip_level;
            int lower_mip_W = W >> lower_mip_level;
            int upper_mip_H = H >> upper_mip_level;
            int upper_mip_W = W >> upper_mip_level;
            std::vector<float> lower_mip_sample(C);
            std::vector<float> upper_mip_sample(C);
            sample_texture(lower_mip_ptr, lower_mip_H, lower_mip_W, C, u, v, filter, wrap, lower_mip_sample.data());
            sample_texture(upper_mip_ptr, upper_mip_H, upper_mip_W, C, u, v, filter, wrap, upper_mip_sample.data());
            for (int c = 0; c < C; c++) {
                color[c] = (1 - mip_frac) * lower_mip_sample[c] + mip_frac * upper_mip_sample[c];
            }
        }
    }
}


static std::tuple<std::vector<int>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
voxelize_trimesh_pbr_impl(
    const float* voxel_size,
    const int* grid_range,
    const int N_tri,
    const float* vertices,
    const float* normals,
    const float* uvs,
    const int* materialIds,
    const std::vector<float*> baseColorFactor,
    const std::vector<uint8_t*> baseColorTexture,
    const std::vector<int> H_bcTex, const std::vector<int> W_bcTex,
    const std::vector<int> baseColorTextureFilter,
    const std::vector<int> baseColorTextureWrap,
    const std::vector<float> metallicFactor,
    const std::vector<uint8_t*> metallicTexture,    
    const std::vector<int> H_mtlTex, const std::vector<int> W_mtlTex,
    const std::vector<int> metallicTextureFilter,
    const std::vector<int> metallicTextureWrap,
    const std::vector<float> roughnessFactor,
    const std::vector<uint8_t*> roughnessTexture,
    const std::vector<int> H_rghTex, const std::vector<int> W_rghTex,
    const std::vector<int> roughnessTextureFilter,
    const std::vector<int> roughnessTextureWrap,
    const std::vector<float*> emissiveFactor,
    const std::vector<uint8_t*> emissiveTexture,
    const std::vector<int> H_emTex, const std::vector<int> W_emTex,
    const std::vector<int> emissiveTextureFilter,
    const std::vector<int> emissiveTextureWrap,
    const std::vector<int> alphaMode,
    const std::vector<float> alphaCutoff,
    const std::vector<float> alphaFactor,
    const std::vector<uint8_t*> alphaTexture,
    const std::vector<int> H_aTex, const std::vector<int> W_aTex,
    const std::vector<int> alphaTextureFilter,
    const std::vector<int> alphaTextureWrap,
    const std::vector<uint8_t*> normalTexture,
    const std::vector<int> H_nTex, const std::vector<int> W_nTex,
    const std::vector<int> normalTextureFilter,
    const std::vector<int> normalTextureWrap,
    const float mipLevelOffset,
    const bool timing
) {
    clock_t start, end;

    // Common variables used in the voxelization process
    Eigen::Vector3f delta_p(voxel_size[0], voxel_size[1], voxel_size[2]);
    Eigen::Vector3i grid_min(grid_range[0], grid_range[1], grid_range[2]);
    Eigen::Vector3i grid_max(grid_range[3], grid_range[4], grid_range[5]);

    // Construct Mipmaps
    start = clock();
    std::vector<std::vector<std::vector<uint8_t>>> baseColorMipmaps(baseColorTexture.size());
    std::vector<std::vector<std::vector<uint8_t>>> metallicMipmaps(metallicTexture.size());
    std::vector<std::vector<std::vector<uint8_t>>> roughnessMipmaps(roughnessTexture.size());
    std::vector<std::vector<std::vector<uint8_t>>> emissiveMipmaps(emissiveTexture.size());
    std::vector<std::vector<std::vector<uint8_t>>> alphaMipmaps(alphaTexture.size());
    std::vector<std::vector<std::vector<uint8_t>>> normalMipmaps(normalTexture.size());
    for (size_t i = 0; i < baseColorTexture.size(); i++) {
        if (baseColorTexture[i] != nullptr && baseColorTextureFilter[i] != 0) {
            baseColorMipmaps[i] = build_mipmaps(baseColorTexture[i], H_bcTex[i], W_bcTex[i], 3);
        }
    }
    for (size_t i = 0; i < metallicTexture.size(); i++) {
        if (metallicTexture[i] != nullptr && metallicTextureFilter[i] != 0) {
            metallicMipmaps[i] = build_mipmaps(metallicTexture[i], H_mtlTex[i], W_mtlTex[i], 1);
        }
    }
    for (size_t i = 0; i < roughnessTexture.size(); i++) {
        if (roughnessTexture[i] != nullptr && roughnessTextureFilter[i] != 0) {
            roughnessMipmaps[i] = build_mipmaps(roughnessTexture[i], H_rghTex[i], W_rghTex[i], 1);
        }
    }
    for (size_t i = 0; i < emissiveTexture.size(); i++) {
        if (emissiveTexture[i] != nullptr && emissiveTextureFilter[i] != 0) {
            emissiveMipmaps[i] = build_mipmaps(emissiveTexture[i], H_emTex[i], W_emTex[i], 3);
        }
    }
    for (size_t i = 0; i < alphaTexture.size(); i++) {
        if (alphaTexture[i] != nullptr && alphaTextureFilter[i] != 0) {
            alphaMipmaps[i] = build_mipmaps(alphaTexture[i], H_aTex[i], W_aTex[i], 1);
        }
    }
    for (size_t i = 0; i < normalTexture.size(); i++) {
        if (normalTexture[i] != nullptr && normalTextureFilter[i] != 0) {
            normalMipmaps[i] = build_mipmaps(normalTexture[i], H_nTex[i], W_nTex[i], 3);
        }
    }
    end = clock();
    if (timing) std::cout << "Mipmaps construction took " << double(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    // Buffers
    std::unordered_map<VoxelCoord, size_t> hash_table;
    std::vector<VoxelCoord> coords;
    std::vector<float> buf_weights;
    std::vector<Eigen::Vector3f> buf_baseColors;
    std::vector<float> buf_metallics;
    std::vector<float> buf_roughnesses;
    std::vector<Eigen::Vector3f> buf_emissives;
    std::vector<float> buf_alphas;
    std::vector<Eigen::Vector3f> buf_normals;

    // Enumerate all triangles
    start = clock();
    for (size_t tid = 0; tid < N_tri; tid++) {
        // COMPUTE COMMON TRIANGLE PROPERTIES
        // Move vertices to origin using bbox
        size_t ptr = tid * 9;
        Eigen::Vector3f v0(vertices[ptr], vertices[ptr + 1], vertices[ptr + 2]);
        Eigen::Vector3f v1(vertices[ptr + 3], vertices[ptr + 4], vertices[ptr + 5]);
        Eigen::Vector3f v2(vertices[ptr + 6], vertices[ptr + 7], vertices[ptr + 8]);
        // Normals
        Eigen::Vector3f n0(normals[ptr], normals[ptr + 1], normals[ptr + 2]);
        Eigen::Vector3f n1(normals[ptr + 3], normals[ptr + 4], normals[ptr + 5]);
        Eigen::Vector3f n2(normals[ptr + 6], normals[ptr + 7], normals[ptr + 8]);
        // UV vectors
        ptr = tid * 6;
        Eigen::Vector2f uv0(uvs[ptr], uvs[ptr + 1]);
        Eigen::Vector2f uv1(uvs[ptr + 2], uvs[ptr + 3]);
        Eigen::Vector2f uv2(uvs[ptr + 4], uvs[ptr + 5]);
        // TBN
        auto tbn = compute_TBN(v0, v1, v2, uv0, uv1, uv2);
        Eigen::Vector3f t = std::get<0>(tbn);
        Eigen::Vector3f b = std::get<1>(tbn);
        Eigen::Vector3f n = std::get<2>(tbn);
        Eigen::Vector2f v_mip_length = std::get<3>(tbn);
        float mip_length = delta_p.maxCoeff() / std::sqrt(v_mip_length.x() * v_mip_length.y());
        // Material ID
        int mid = materialIds[tid];

        // Find intersected voxel for each triangle
        std::unordered_set<VoxelCoord> intersected_voxels;
        // Scan-line algorithm to find intersections with the voxel grid from three directions
        /*
          t0
          | \
          |  t1
          | /
          t2
         */
        auto scan_line_fill = [&] (const int ax2) {
            int ax0 = (ax2 + 1) % 3;
            int ax1 = (ax2 + 2) % 3;

            // Canonical question
            std::array<Eigen::Vector3d, 3> t = {
                Eigen::Vector3d(v0[ax0], v0[ax1], v0[ax2]),
                Eigen::Vector3d(v1[ax0], v1[ax1], v1[ax2]),
                Eigen::Vector3d(v2[ax0], v2[ax1], v2[ax2])
            };
            std::sort(t.begin(), t.end(), [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return a.y() < b.y(); });

            // Scan-line algorithm
            int start = std::clamp(int(t[0].y() / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            int mid = std::clamp(int(t[1].y() / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            int end = std::clamp(int(t[2].y() / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);

            auto scan_line_half = [&] (const int row_start, const int row_end, const Eigen::Vector3d t0, const Eigen::Vector3d t1, const Eigen::Vector3d t2) {
            /*
             t0
             | \
             t3-t4
             |   \
             t1---t2
             */
                for (int y_idx = row_start; y_idx < row_end; ++y_idx) {
                    double y = (y_idx + 1) * voxel_size[ax1];
                    Eigen::Vector2d t3 = lerp(t0.y(), t1.y(), y, Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t1.x(), t1.z()));
                    Eigen::Vector2d t4 = lerp(t0.y(), t2.y(), y, Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t2.x(), t2.z()));
                    if (t3.x() > t4.x()) std::swap(t3, t4);
                    int line_start = std::clamp(int(t3.x() / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
                    int line_end = std::clamp(int(t4.x() / voxel_size[ax0]), grid_min[ax0], grid_max[ax0] - 1);
                    for (int x_idx = line_start; x_idx < line_end; ++x_idx) {
                        double x = (x_idx + 1) * voxel_size[ax0];
                        double z = lerp(t3.x(), t4.x(), x, t3.y(), t4.y());
                        int z_idx = int(z / voxel_size[ax2]);
                        if (z_idx >= grid_min[ax2] && z_idx < grid_max[ax2]) {
                            // For 4-connected voxels
                            for (int dx = 0; dx < 2; ++dx) {
                                for (int dy = 0; dy < 2; ++dy) {
                                    VoxelCoord coord;
                                    coord[ax0] = x_idx + dx; coord[ax1] = y_idx + dy; coord[ax2] = z_idx;
                                    intersected_voxels.insert(coord);
                                }
                            }
                        }
                    }
                }
            };
            scan_line_half(start, mid, t[0], t[1], t[2]);
            scan_line_half(mid, end, t[2], t[1], t[0]);   
        };
        scan_line_fill(0);
        scan_line_fill(1);
        scan_line_fill(2);

        // For all intersected voxels, ample texture and write to voxel grid
        for (auto voxel : intersected_voxels) {
            int x = voxel.x;
            int y = voxel.y;
            int z = voxel.z;

            // Compute barycentric coordinates and weight
            Eigen::Vector4f barycentric = project_onto_triangle(
                Eigen::Vector3f((x + 0.5f) * delta_p.x(), (y + 0.5f) * delta_p.y(), (z + 0.5f) * delta_p.z()),
                v0, v1, v2, n
            );
            Eigen::Vector2f uv = {
                barycentric.x() * uv0.x() + barycentric.y() * uv1.x() + barycentric.z() * uv2.x(),
                barycentric.x() * uv0.y() + barycentric.y() * uv1.y() + barycentric.z() * uv2.y()
            };
            Eigen::Vector3f int_n = {
                barycentric.x() * n0.x() + barycentric.y() * n1.x() + barycentric.z() * n2.x(),
                barycentric.x() * n0.y() + barycentric.y() * n1.y() + barycentric.z() * n2.y(),
                barycentric.x() * n0.z() + barycentric.y() * n1.z() + barycentric.z() * n2.z()
            };
            float weight = 1 - barycentric.w();

            /// base color
            float baseColor[3] = {1, 1, 1};
            if (baseColorTexture[mid]) {
                sample_texture_mipmap(
                    baseColorTexture[mid],
                    H_bcTex[mid], W_bcTex[mid], 3,
                    baseColorMipmaps[mid],
                    uv.x(), uv.y(), mip_length, mipLevelOffset,
                    baseColorTextureFilter[mid], baseColorTextureWrap[mid],
                    baseColor
                );
            }
            baseColor[0] *= baseColorFactor[mid][0];
            baseColor[1] *= baseColorFactor[mid][1];
            baseColor[2] *= baseColorFactor[mid][2];

            /// metallic
            float metallic = 1.0f;
            if (metallicTexture[mid]) {
                sample_texture_mipmap(
                    metallicTexture[mid],
                    H_mtlTex[mid], W_mtlTex[mid], 1,
                    metallicMipmaps[mid],
                    uv.x(), uv.y(), mip_length, mipLevelOffset,
                    metallicTextureFilter[mid], metallicTextureWrap[mid],
                    &metallic
                );
            }
            metallic *= metallicFactor[mid];

            /// roughness
            float roughness = 1.0f;
            if (roughnessTexture[mid]) {
                sample_texture_mipmap(
                    roughnessTexture[mid],
                    H_rghTex[mid], W_rghTex[mid], 1,
                    roughnessMipmaps[mid],
                    uv.x(), uv.y(), mip_length, mipLevelOffset,
                    roughnessTextureFilter[mid], roughnessTextureWrap[mid],
                    &roughness
                );
            }
            roughness *= roughnessFactor[mid];

            /// emissive
            float emissive[3] = {1, 1, 1};
            if (emissiveTexture[mid]) {
                sample_texture_mipmap(
                    emissiveTexture[mid],
                    H_emTex[mid], W_emTex[mid], 3,
                    roughnessMipmaps[mid],
                    uv.x(), uv.y(), mip_length, mipLevelOffset,
                    emissiveTextureFilter[mid], emissiveTextureWrap[mid],
                    emissive
                );
            }
            emissive[0] *= emissiveFactor[mid][0];
            emissive[1] *= emissiveFactor[mid][1];
            emissive[2] *= emissiveFactor[mid][2];

            /// alpha
            float alpha = 1.0f;
            if (alphaMode[mid] != 0) {
                if (alphaTexture[mid]) {
                        sample_texture_mipmap(
                        alphaTexture[mid],
                        H_aTex[mid], W_aTex[mid], 1,
                        alphaMipmaps[mid],
                        uv.x(), uv.y(), mip_length, mipLevelOffset,
                        alphaTextureFilter[mid], alphaTextureWrap[mid],
                        &alpha
                    );
                }
                alpha *= alphaFactor[mid];
                if (alphaMode[mid] == 1) {       // MASK
                    alpha = alpha < alphaCutoff[mid] ? 0.0f : 1.0f;
                }
            }

            /// normal
            float normal[3] = {int_n.x(), int_n.y(), int_n.z()};
            if (normalTexture[mid]) {
                sample_texture_mipmap(
                    normalTexture[mid],
                    H_nTex[mid], W_nTex[mid], 3,
                    normalMipmaps[mid],
                    uv.x(), uv.y(), mip_length, mipLevelOffset,
                    normalTextureFilter[mid], normalTextureWrap[mid],
                    normal
                );
                normal[0] = normal[0] * 2 - 1;
                normal[1] = normal[1] * 2 - 1;
                normal[2] = normal[2] * 2 - 1;
                Eigen::Vector3f _n = (normal[0] * t + normal[1] * b + normal[2] * int_n).normalized();
                normal[0] = _n.x();
                normal[1] = _n.y();
                normal[2] = _n.z();
            }

            // Write to voxel grid
            auto coord = VoxelCoord{x-grid_min.x(), y-grid_min.y(), z-grid_min.z()};
            auto kv = hash_table.find(coord);
            if (kv == hash_table.end()) {
                hash_table[coord] = coords.size();
                coords.push_back({coord.x, coord.y, coord.z});
                buf_weights.push_back(weight);
                buf_baseColors.push_back(Eigen::Vector3f(baseColor[0], baseColor[1], baseColor[2]) * weight);
                buf_metallics.push_back(metallic * weight);
                buf_roughnesses.push_back(roughness * weight);
                buf_emissives.push_back(Eigen::Vector3f(emissive[0], emissive[1], emissive[2]) * weight);
                buf_alphas.push_back(alpha * weight);
                buf_normals.push_back(Eigen::Vector3f(normal[0], normal[1], normal[2]) * weight);
            }
            else {
                auto i = kv->second;
                buf_weights[i] += weight;
                buf_baseColors[i] += Eigen::Vector3f(baseColor[0], baseColor[1], baseColor[2]) * weight;
                buf_metallics[i] += metallic * weight;
                buf_roughnesses[i] += roughness * weight;
                buf_emissives[i] += Eigen::Vector3f(emissive[0], emissive[1], emissive[2]) * weight;
                buf_alphas[i] += alpha * weight;
                buf_normals[i] += Eigen::Vector3f(normal[0], normal[1], normal[2]) * weight;
            }
        }
    }
    end = clock();
    if (timing) std::cout << "Voxelization took " << double(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    // Normalize buffers
    start = clock();
    std::vector<int> out_coord(coords.size() * 3);
    std::vector<float> out_baseColor(coords.size() * 3);
    std::vector<float> out_metallic(coords.size());
    std::vector<float> out_roughness(coords.size());
    std::vector<float> out_emissive(coords.size() * 3);
    std::vector<float> out_alpha(coords.size());
    std::vector<float> out_normal(coords.size() * 3);
    for (int i = 0; i < coords.size(); i++) {
        out_coord[i * 3 + 0] = coords[i].x;
        out_coord[i * 3 + 1] = coords[i].y;
        out_coord[i * 3 + 2] = coords[i].z;
        out_baseColor[i * 3 + 0] = buf_baseColors[i].x() / buf_weights[i];
        out_baseColor[i * 3 + 1] = buf_baseColors[i].y() / buf_weights[i];
        out_baseColor[i * 3 + 2] = buf_baseColors[i].z() / buf_weights[i];
        out_metallic[i] = buf_metallics[i] / buf_weights[i];
        out_roughness[i] = buf_roughnesses[i] / buf_weights[i];
        out_emissive[i * 3 + 0] = buf_emissives[i].x() / buf_weights[i];
        out_emissive[i * 3 + 1] = buf_emissives[i].y() / buf_weights[i];
        out_emissive[i * 3 + 2] = buf_emissives[i].z() / buf_weights[i];
        out_alpha[i] = buf_alphas[i] / buf_weights[i];
        out_normal[i * 3 + 0] = buf_normals[i].x() / buf_weights[i];
        out_normal[i * 3 + 1] = buf_normals[i].y() / buf_weights[i];
        out_normal[i * 3 + 2] = buf_normals[i].z() / buf_weights[i];
    }
    end = clock();
    if (timing) std::cout << "Normalization took " << double(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    return std::make_tuple(
        std::move(out_coord),
        std::move(out_baseColor),
        std::move(out_metallic),
        std::move(out_roughness),
        std::move(out_emissive),
        std::move(out_alpha),
        std::move(out_normal)
    );
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
textured_mesh_to_volumetric_attr_cpu(
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    const torch::Tensor& vertices,
    const torch::Tensor& normals,
    const torch::Tensor& uvs,
    const torch::Tensor& materialIds,
    const std::vector<torch::Tensor>& baseColorFactor,
    const std::vector<torch::Tensor>& baseColorTexture,
    const std::vector<int>& baseColorTextureFilter,
    const std::vector<int>& baseColorTextureWrap,
    const std::vector<float>& metallicFactor,
    const std::vector<torch::Tensor>& metallicTexture,
    const std::vector<int>& metallicTextureFilter,
    const std::vector<int>& metallicTextureWrap,
    const std::vector<float>& roughnessFactor,
    const std::vector<torch::Tensor>& roughnessTexture,
    const std::vector<int>& roughnessTextureFilter,
    const std::vector<int>& roughnessTextureWrap,
    const std::vector<torch::Tensor>& emissiveFactor,
    const std::vector<torch::Tensor>& emissiveTexture,
    const std::vector<int>& emissiveTextureFilter,
    const std::vector<int>& emissiveTextureWrap,
    const std::vector<int>& alphaMode,
    const std::vector<float>& alphaCutoff,
    const std::vector<float>& alphaFactor,
    const std::vector<torch::Tensor>& alphaTexture,
    const std::vector<int>& alphaTextureFilter,
    const std::vector<int>& alphaTextureWrap,
    const std::vector<torch::Tensor>& normalTexture,
    const std::vector<int>& normalTextureFilter,
    const std::vector<int>& normalTextureWrap,
    const float mipLevelOffset,
    const bool timing
) {
    auto N_mat = baseColorFactor.size();
    int N_tri = vertices.size(0);

    // Get the size of the input tensors
    std::vector<float*> baseColorFactor_ptrs(N_mat);
    std::vector<uint8_t*> baseColorTexture_ptrs(N_mat);
    std::vector<int> H_bcTex(N_mat), W_bcTex(N_mat);
    std::vector<float> metallicFactor_vec(N_mat);
    std::vector<uint8_t*> metallicTexture_ptrs(N_mat);
    std::vector<int> H_mtlTex(N_mat), W_mtlTex(N_mat);
    std::vector<float> roughnessFactor_vec(N_mat);
    std::vector<uint8_t*> roughnessTexture_ptrs(N_mat);
    std::vector<int> H_rghTex(N_mat), W_rghTex(N_mat);
    std::vector<float*> emissiveFactor_ptrs(N_mat);
    std::vector<uint8_t*> emissiveTexture_ptrs(N_mat);
    std::vector<int> H_emTex(N_mat), W_emTex(N_mat);
    std::vector<int> alphaMode_vec(N_mat);
    std::vector<float> alphaCutoff_vec(N_mat);
    std::vector<float> alphaFactor_vec(N_mat);
    std::vector<uint8_t*> alphaTexture_ptrs(N_mat);
    std::vector<int> H_aTex(N_mat), W_aTex(N_mat);
    std::vector<uint8_t*> normalTexture_ptrs(N_mat);
    std::vector<int> H_nTex(N_mat), W_nTex(N_mat);

    for (int i = 0; i < N_mat; ++i) {
        baseColorFactor_ptrs[i] = baseColorFactor[i].contiguous().data_ptr<float>();
        if (baseColorTexture[i].numel() > 0) {
            baseColorTexture_ptrs[i] = baseColorTexture[i].contiguous().data_ptr<uint8_t>();
            H_bcTex[i] = baseColorTexture[i].size(0);
            W_bcTex[i] = baseColorTexture[i].size(1);
        }
        else {
            baseColorTexture_ptrs[i] = nullptr;
            H_bcTex[i] = 0;
            W_bcTex[i] = 0;
        }
        metallicFactor_vec[i] = metallicFactor[i];
        if (metallicTexture[i].numel() > 0) {
            metallicTexture_ptrs[i] = metallicTexture[i].contiguous().data_ptr<uint8_t>();
            H_mtlTex[i] = metallicTexture[i].size(0);
            W_mtlTex[i] = metallicTexture[i].size(1);
        }
        else {
            metallicTexture_ptrs[i] = nullptr;
            H_mtlTex[i] = 0;
            W_mtlTex[i] = 0;
        }
        roughnessFactor_vec[i] = roughnessFactor[i];
        if (roughnessTexture[i].numel() > 0) {
            roughnessTexture_ptrs[i] = roughnessTexture[i].contiguous().data_ptr<uint8_t>();
            H_rghTex[i] = roughnessTexture[i].size(0);
            W_rghTex[i] = roughnessTexture[i].size(1);
        }
        else {
            roughnessTexture_ptrs[i] = nullptr;
            H_rghTex[i] = 0;
            W_rghTex[i] = 0;
        }
        emissiveFactor_ptrs[i] = emissiveFactor[i].contiguous().data_ptr<float>();
        if (emissiveTexture[i].numel() > 0) {
            emissiveTexture_ptrs[i] = emissiveTexture[i].contiguous().data_ptr<uint8_t>();
            H_emTex[i] = emissiveTexture[i].size(0);
            W_emTex[i] = emissiveTexture[i].size(1);
        }
        else {
            emissiveTexture_ptrs[i] = nullptr;
            H_emTex[i] = 0;
            W_emTex[i] = 0;
        }
        alphaMode_vec[i] = alphaMode[i];
        alphaCutoff_vec[i] = alphaCutoff[i];
        alphaFactor_vec[i] = alphaFactor[i];
        if (alphaTexture[i].numel() > 0) {
            alphaTexture_ptrs[i] = alphaTexture[i].contiguous().data_ptr<uint8_t>();
            H_aTex[i] = alphaTexture[i].size(0);
            W_aTex[i] = alphaTexture[i].size(1);
        }
        else {
            alphaTexture_ptrs[i] = nullptr;
            H_aTex[i] = 0;
            W_aTex[i] = 0;
        }
        if (normalTexture[i].numel() > 0) {
            normalTexture_ptrs[i] = normalTexture[i].contiguous().data_ptr<uint8_t>();
            H_nTex[i] = normalTexture[i].size(0);
            W_nTex[i] = normalTexture[i].size(1);
        }
        else {
            normalTexture_ptrs[i] = nullptr;
            H_nTex[i] = 0;
            W_nTex[i] = 0;
        }
    }

    auto outputs = voxelize_trimesh_pbr_impl(
        voxel_size.contiguous().data_ptr<float>(),
        grid_range.contiguous().data_ptr<int>(),
        N_tri,
        vertices.contiguous().data_ptr<float>(),
        normals.contiguous().data_ptr<float>(),
        uvs.contiguous().data_ptr<float>(),
        materialIds.contiguous().data_ptr<int>(),
        baseColorFactor_ptrs,
        baseColorTexture_ptrs,
        H_bcTex, W_bcTex,
        baseColorTextureFilter, baseColorTextureWrap,
        metallicFactor_vec,
        metallicTexture_ptrs,
        H_mtlTex, W_mtlTex,
        metallicTextureFilter, metallicTextureWrap,
        roughnessFactor_vec,
        roughnessTexture_ptrs,
        H_rghTex, W_rghTex,
        roughnessTextureFilter, roughnessTextureWrap,
        emissiveFactor_ptrs,
        emissiveTexture_ptrs,
        H_emTex, W_emTex,
        emissiveTextureFilter, emissiveTextureWrap,
        alphaMode_vec,
        alphaCutoff_vec,
        alphaFactor_vec,
        alphaTexture_ptrs,
        H_aTex, W_aTex,
        alphaTextureFilter, alphaTextureWrap,
        normalTexture_ptrs,
        H_nTex, W_nTex,
        normalTextureFilter, normalTextureWrap,
        mipLevelOffset,
        timing
    );

    std::vector<int> coords_vec = std::get<0>(outputs);
    std::vector<float> baseColors_vec = std::get<1>(outputs);
    std::vector<float> metallics_vec = std::get<2>(outputs);
    std::vector<float> roughnesses_vec = std::get<3>(outputs);
    std::vector<float> emissives_vec = std::get<4>(outputs);
    std::vector<float> alphas_vec = std::get<5>(outputs);
    std::vector<float> normals_vec = std::get<6>(outputs);
    
    // Create output tensors
    auto out_coords = torch::from_blob(coords_vec.data(), {static_cast<int64_t>(coords_vec.size() / 3), 3}, torch::kInt32).clone();
    auto out_baseColors = torch::from_blob(baseColors_vec.data(), {static_cast<int64_t>(baseColors_vec.size() / 3), 3}, torch::kFloat32).clone();
    auto out_metallics = torch::from_blob(metallics_vec.data(), {static_cast<int64_t>(metallics_vec.size())}, torch::kFloat32).clone();
    auto out_roughnesses = torch::from_blob(roughnesses_vec.data(), {static_cast<int64_t>(roughnesses_vec.size())}, torch::kFloat32).clone();
    auto out_emissives = torch::from_blob(emissives_vec.data(), {static_cast<int64_t>(emissives_vec.size() / 3), 3}, torch::kFloat32).clone();
    auto out_alphas = torch::from_blob(alphas_vec.data(), {static_cast<int64_t>(alphas_vec.size())}, torch::kFloat32).clone();
    auto out_normals = torch::from_blob(normals_vec.data(), {static_cast<int64_t>(normals_vec.size() / 3), 3}, torch::kFloat32).clone();

    return std::make_tuple(
        out_coords,
        out_baseColors,
        out_metallics,
        out_roughnesses,
        out_emissives,
        out_alphas,
        out_normals
    );
}

