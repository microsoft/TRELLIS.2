#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <ctime>

#include "api.h"


constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();


struct float3 {float x, y, z; float& operator[](int i) {return (&x)[i];}};
struct int3 {int x, y, z; int& operator[](int i) {return (&x)[i];}};
struct int4 {int x, y, z, w; int& operator[](int i) {return (&x)[i];}};
struct bool3 {bool x, y, z; bool& operator[](int i) {return (&x)[i];}};


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


void intersect_qef(
    const Eigen::Vector3f& voxel_size,
    const Eigen::Vector3i& grid_min,
    const Eigen::Vector3i& grid_max,
    const std::vector<Eigen::Vector3f>& triangles, // 3 vertices per triangle
    std::unordered_map<VoxelCoord, size_t>& hash_table, // Hash table for voxel lookup
    std::vector<int3>& voxels, // Output: Voxel coordinates
    std::vector<Eigen::Vector3f>& means, // Output: Mean vertex positions for each voxel
    std::vector<float>& cnt, // Output: Number of intersections for each voxel
    std::vector<bool3>& intersected, // Output: Whether edge of voxel intersects with triangle
    std::vector<Eigen::Matrix4f>& qefs // Output: QEF matrices for each voxel
) {
    const size_t N_tri = triangles.size() / 3;

    for (size_t i = 0; i < N_tri; ++i) {
        const Eigen::Vector3f& v0 = triangles[i * 3 + 0];
        const Eigen::Vector3f& v1 = triangles[i * 3 + 1];
        const Eigen::Vector3f& v2 = triangles[i * 3 + 2];

        // Compute edge vectors and face normal
        Eigen::Vector3f e0 = v1 - v0;
        Eigen::Vector3f e1 = v2 - v1;
        Eigen::Vector3f n = e0.cross(e1).normalized();
        Eigen::Vector4f plane;
        plane << n.x(), n.y(), n.z(), -n.dot(v0);
        auto Q = plane * plane.transpose();

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
                                    Eigen::Vector3d intersect;
                                    intersect[ax0] = x; intersect[ax1] = y; intersect[ax2] = z;
                                    auto kv = hash_table.find(coord);
                                    if (kv == hash_table.end()) {
                                        hash_table[coord] = voxels.size();
                                        voxels.push_back({coord.x, coord.y, coord.z});
                                        means.push_back(intersect.cast<float>());
                                        cnt.push_back(1);
                                        intersected.push_back({false, false, false});
                                        qefs.push_back(Q);
                                        if (dx == 0 && dy == 0)
                                            intersected.back()[ax2] = true;
                                    }
                                    else {
                                        auto i = kv->second;
                                        means[i] += intersect.cast<float>();
                                        cnt[i] += 1;
                                        if (dx == 0 && dy == 0)
                                            intersected[i][ax2] = true;
                                        qefs[i] += Q;
                                    }
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
    }
}


void face_qef(
    const Eigen::Vector3f& voxel_size,
    const Eigen::Vector3i& grid_min,
    const Eigen::Vector3i& grid_max,
    const std::vector<Eigen::Vector3f>& triangles, // 3 vertices per triangle
    std::unordered_map<VoxelCoord, size_t>& hash_table, // Hash table for voxel lookup
    std::vector<Eigen::Matrix4f>& qefs // Output: QEF matrices for each voxel
) {
    const size_t N_tri = triangles.size() / 3;

    for (size_t i = 0; i < N_tri; ++i) {
        const Eigen::Vector3f& v0 = triangles[i * 3 + 0];
        const Eigen::Vector3f& v1 = triangles[i * 3 + 1];
        const Eigen::Vector3f& v2 = triangles[i * 3 + 2];

        // Compute edge vectors and face normal
        Eigen::Vector3f e0 = v1 - v0;
        Eigen::Vector3f e1 = v2 - v1;
        Eigen::Vector3f e2 = v0 - v2;
        Eigen::Vector3f n = e0.cross(e1).normalized();
        Eigen::Vector4f plane;
        plane << n.x(), n.y(), n.z(), -n.dot(v0);
        auto Q = plane * plane.transpose();

        // Compute triangle bounding box in voxel coordinates
        Eigen::Vector3f bb_min_f = v0.cwiseMin(v1).cwiseMin(v2).cwiseQuotient(voxel_size);
        Eigen::Vector3f bb_max_f = v0.cwiseMax(v1).cwiseMax(v2).cwiseQuotient(voxel_size);

        Eigen::Vector3i bb_min(std::max(static_cast<int>(bb_min_f.x()), grid_min.x()),
                               std::max(static_cast<int>(bb_min_f.y()), grid_min.y()),
                               std::max(static_cast<int>(bb_min_f.z()), grid_min.z()));
        Eigen::Vector3i bb_max(std::min(static_cast<int>(bb_max_f.x() + 1), grid_max.x()),
                               std::min(static_cast<int>(bb_max_f.y() + 1), grid_max.y()),
                               std::min(static_cast<int>(bb_max_f.z() + 1), grid_max.z()));

        // Plane test setup
        Eigen::Vector3f c(
            n.x() > 0.0f ? voxel_size.x() : 0.0f,
            n.y() > 0.0f ? voxel_size.y() : 0.0f,
            n.z() > 0.0f ? voxel_size.z() : 0.0f
        );
        float d1 = n.dot(c - v0);
        float d2 = n.dot(voxel_size - c - v0);

        // XY plane projection test setup
        int mul_xy = (n.z() < 0.0f) ? -1 : 1;
        Eigen::Vector2f n_xy_e0(-mul_xy * e0.y(), mul_xy * e0.x());
        Eigen::Vector2f n_xy_e1(-mul_xy * e1.y(), mul_xy * e1.x());
        Eigen::Vector2f n_xy_e2(-mul_xy * e2.y(), mul_xy * e2.x());

        float d_xy_e0 = -n_xy_e0.dot(v0.head<2>()) + n_xy_e0.cwiseMax(0.0f).dot(voxel_size.head<2>());
        float d_xy_e1 = -n_xy_e1.dot(v1.head<2>()) + n_xy_e1.cwiseMax(0.0f).dot(voxel_size.head<2>());
        float d_xy_e2 = -n_xy_e2.dot(v2.head<2>()) + n_xy_e2.cwiseMax(0.0f).dot(voxel_size.head<2>());

        // YZ plane projection test setup
        int mul_yz = (n.x() < 0.0f) ? -1 : 1;
        Eigen::Vector2f n_yz_e0(-mul_yz * e0.z(), mul_yz * e0.y());
        Eigen::Vector2f n_yz_e1(-mul_yz * e1.z(), mul_yz * e1.y());
        Eigen::Vector2f n_yz_e2(-mul_yz * e2.z(), mul_yz * e2.y());

        float d_yz_e0 = -n_yz_e0.dot(Eigen::Vector2f(v0.y(), v0.z())) + n_yz_e0.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.y(), voxel_size.z()));
        float d_yz_e1 = -n_yz_e1.dot(Eigen::Vector2f(v1.y(), v1.z())) + n_yz_e1.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.y(), voxel_size.z()));
        float d_yz_e2 = -n_yz_e2.dot(Eigen::Vector2f(v2.y(), v2.z())) + n_yz_e2.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.y(), voxel_size.z()));

        // ZX plane projection test setup
        int mul_zx = (n.y() < 0.0f) ? -1 : 1;
        Eigen::Vector2f n_zx_e0(-mul_zx * e0.x(), mul_zx * e0.z());
        Eigen::Vector2f n_zx_e1(-mul_zx * e1.x(), mul_zx * e1.z());
        Eigen::Vector2f n_zx_e2(-mul_zx * e2.x(), mul_zx * e2.z());

        float d_zx_e0 = -n_zx_e0.dot(Eigen::Vector2f(v0.z(), v0.x())) + n_zx_e0.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.z(), voxel_size.x()));
        float d_zx_e1 = -n_zx_e1.dot(Eigen::Vector2f(v1.z(), v1.x())) + n_zx_e1.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.z(), voxel_size.x()));
        float d_zx_e2 = -n_zx_e2.dot(Eigen::Vector2f(v2.z(), v2.x())) + n_zx_e2.cwiseMax(0.0f).dot(Eigen::Vector2f(voxel_size.z(), voxel_size.x()));

        // Loop over candidate voxels inside bounding box
        for (int z = bb_min.z(); z < bb_max.z(); ++z) {
            for (int y = bb_min.y(); y < bb_max.y(); ++y) {
                for (int x = bb_min.x(); x < bb_max.x(); ++x) {
                    // Voxel center
                    Eigen::Vector3f p = voxel_size.cwiseProduct(Eigen::Vector3f(x, y, z));

                    // Plane through box test
                    float nDOTp = n.dot(p);
                    if (((nDOTp + d1) * (nDOTp + d2)) > 0.0f) continue;

                    // XY projection test
                    Eigen::Vector2f p_xy(p.x(), p.y());
                    if (n_xy_e0.dot(p_xy) + d_xy_e0 < 0) continue;
                    if (n_xy_e1.dot(p_xy) + d_xy_e1 < 0) continue;
                    if (n_xy_e2.dot(p_xy) + d_xy_e2 < 0) continue;

                    // YZ projection test
                    Eigen::Vector2f p_yz(p.y(), p.z());
                    if (n_yz_e0.dot(p_yz) + d_yz_e0 < 0) continue;
                    if (n_yz_e1.dot(p_yz) + d_yz_e1 < 0) continue;
                    if (n_yz_e2.dot(p_yz) + d_yz_e2 < 0) continue;

                    // ZX projection test
                    Eigen::Vector2f p_zx(p.z(), p.x());
                    if (n_zx_e0.dot(p_zx) + d_zx_e0 < 0) continue;
                    if (n_zx_e1.dot(p_zx) + d_zx_e1 < 0) continue;
                    if (n_zx_e2.dot(p_zx) + d_zx_e2 < 0) continue;

                    // Passed all tests — mark voxel
                    auto coord = VoxelCoord{x, y, z};
                    auto kv = hash_table.find(coord);
                    if (kv != hash_table.end()) {
                        qefs[kv->second] += Q;
                    }
                }
            }
        }
    }
}


void boundry_qef(
    const Eigen::Vector3f& voxel_size,
    const Eigen::Vector3i& grid_min,
    const Eigen::Vector3i& grid_max,
    const std::vector<Eigen::Vector3f>& boundries, // 2 vertices per segment
    const float boundary_weight,    // Weight for boundary edges
    std::unordered_map<VoxelCoord, size_t>& hash_table, // Hash table for voxel lookup
    std::vector<Eigen::Matrix4f>& qefs // Output: QEF matrices for each voxel
) {
    for (size_t i = 0; i < boundries.size() / 2; ++i) {
        const Eigen::Vector3f& v0 = boundries[i * 2 + 0];
        const Eigen::Vector3f& v1 = boundries[i * 2 + 1];

        // Calculate the QEF for the edge (boundary) defined by v0 and v1
        Eigen::Vector3d dir(v1.x() - v0.x(), v1.y() - v0.y(), v1.z() - v0.z());
        double segment_length = dir.norm();
        if (segment_length < 1e-6d) continue; // Skip degenerate edges (zero-length)
        dir.normalize();  // unit direction vector

        // Projection matrix orthogonal to the direction: I - d d^T
        Eigen::Matrix3f A = Eigen::Matrix3f::Identity() - (dir * dir.transpose()).cast<float>();

        // b = -A * v0
        Eigen::Vector3f b = -A * v0;

        // c = v0^T * A * v0
        float c = v0.transpose() * A * v0;

        // Now pack this into a 4x4 QEF matrix
        Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
        Q.block<3, 3>(0, 0) = A;
        Q.block<3, 1>(0, 3) = b;
        Q.block<1, 3>(3, 0) = b.transpose();
        Q(3, 3) = c;

        // DDA Traversal logic directly inside the function

        // Starting and ending voxel coordinates
        Eigen::Vector3i v0_voxel = (v0.cwiseQuotient(voxel_size)).array().floor().cast<int>();
        Eigen::Vector3i v1_voxel = (v1.cwiseQuotient(voxel_size)).array().floor().cast<int>();

        // Determine step direction for each axis based on the line direction
        Eigen::Vector3i step = (dir.array() > 0).select(Eigen::Vector3i(1, 1, 1), Eigen::Vector3i(-1, -1, -1));

        Eigen::Vector3d tMax, tDelta;
        for (int axis = 0; axis < 3; ++axis) {
            if (dir[axis] == 0.0d) {
                tMax[axis] = std::numeric_limits<double>::infinity();
                tDelta[axis] = std::numeric_limits<double>::infinity();
            } else {
                float voxel_border = voxel_size[axis] * (v0_voxel[axis] + (step[axis] > 0 ? 1 : 0));
                tMax[axis] = (voxel_border - v0[axis]) / dir[axis];
                tDelta[axis] = voxel_size[axis] / std::abs(dir[axis]);
            }
        }

        // Current voxel position
        Eigen::Vector3i current = v0_voxel;

        // Store the voxel we start at
        std::vector<VoxelCoord> voxels;
        voxels.push_back({current.x(), current.y(), current.z()});

        // Traverse the voxels
        while (true) {
            int axis;
            if (tMax.x() < tMax.y()) {
                axis = (tMax.x() < tMax.z()) ? 0 : 2;
            } else {
                axis = (tMax.y() < tMax.z()) ? 1 : 2;
            }

            if (tMax[axis] > segment_length) break;

            current[axis] += step[axis];
            tMax[axis] += tDelta[axis];

            voxels.push_back({current.x(), current.y(), current.z()});
        }

        // Accumulate QEF for each voxel passed through
        for (const auto& coord : voxels) {
            // Make sure the voxel is within bounds
            if ((coord.x < grid_min.x() || coord.x >= grid_max.x()) ||
                (coord.y < grid_min.y() || coord.y >= grid_max.y()) ||
                (coord.z < grid_min.z() || coord.z >= grid_max.z())) continue;
            if (!hash_table.count(coord)) continue; // Skip if voxel not in hash table

            // Accumulate the QEF for this voxel
            qefs[hash_table[coord]] += boundary_weight * Q; // Scale by boundary weight
        }
    }
}


std::array<int3, 2> quad_to_2tri(
    const std::vector<float3>& vertices,
    const int4& quad_indices
) {
    int ia = quad_indices.x;
    int ib = quad_indices.y;
    int ic = quad_indices.z;
    int id = quad_indices.w;

    Eigen::Vector3f a(vertices[ia].x, vertices[ia].y, vertices[ia].z);
    Eigen::Vector3f b(vertices[ib].x, vertices[ib].y, vertices[ib].z);
    Eigen::Vector3f c(vertices[ic].x, vertices[ic].y, vertices[ic].z);
    Eigen::Vector3f d(vertices[id].x, vertices[id].y, vertices[id].z);

    // diagonal AC
    Eigen::Vector3f n_abc = (b - a).cross(c - a).normalized();
    Eigen::Vector3f n_acd = (c - a).cross(d - a).normalized();
    float angle_ac = std::acos(std::clamp(n_abc.dot(n_acd), -1.0f, 1.0f));

    // diagonal BD
    Eigen::Vector3f n_abd = (b - a).cross(d - a).normalized();
    Eigen::Vector3f n_bcd = (c - b).cross(d - b).normalized();
    float angle_bd = std::acos(std::clamp(n_abd.dot(n_bcd), -1.0f, 1.0f));

    if (angle_ac <= angle_bd) {
        return {int3{ia, ib, ic}, int3{ia, ic, id}};
    } else {
        return {int3{ia, ib, id}, int3{ib, ic, id}};
    }
}


void face_from_dual_vertices(
    const std::unordered_map<VoxelCoord, size_t>& hash_table,
    const std::vector<int3>& voxels,
    const std::vector<float3>& dual_vertices,
    const std::vector<bool3>& intersected,
    std::vector<int3>& face_indices
) {
    for (int i = 0; i < dual_vertices.size(); ++i) {
        int3 coord = voxels[i];
        bool3 is_intersected = intersected[i];

        // Check existence of neighboring 6 voxels
        size_t neigh_indices[6] = {
            get_or_default(hash_table, VoxelCoord{coord.x + 1, coord.y, coord.z}, kInvalidIndex),
            get_or_default(hash_table, VoxelCoord{coord.x, coord.y + 1, coord.z}, kInvalidIndex),
            get_or_default(hash_table, VoxelCoord{coord.x + 1, coord.y + 1, coord.z}, kInvalidIndex),
            get_or_default(hash_table, VoxelCoord{coord.x, coord.y, coord.z + 1}, kInvalidIndex),
            get_or_default(hash_table, VoxelCoord{coord.x + 1, coord.y, coord.z + 1}, kInvalidIndex),
            get_or_default(hash_table, VoxelCoord{coord.x, coord.y + 1, coord.z + 1}, kInvalidIndex)
        };

        // xy-plane
        if (is_intersected[2] && neigh_indices[0] != kInvalidIndex && neigh_indices[1] != kInvalidIndex && neigh_indices[2] != kInvalidIndex) {
            int4 quad_indices{i, neigh_indices[0], neigh_indices[2], neigh_indices[1]};
            auto tri_indices = quad_to_2tri(dual_vertices, quad_indices);
            face_indices.insert(face_indices.end(), tri_indices.begin(), tri_indices.end());
        }
        // yz-plane
        if (is_intersected[0] && neigh_indices[1] != kInvalidIndex && neigh_indices[3] != kInvalidIndex && neigh_indices[5] != kInvalidIndex) {
            int4 quad_indices{i, neigh_indices[1], neigh_indices[5], neigh_indices[3]};
            auto tri_indices = quad_to_2tri(dual_vertices, quad_indices);
            face_indices.insert(face_indices.end(), tri_indices.begin(), tri_indices.end());
        }
        // xz-plane
        if (is_intersected[1] && neigh_indices[0] != kInvalidIndex && neigh_indices[3] != kInvalidIndex && neigh_indices[4] != kInvalidIndex) {
            int4 quad_indices{i, neigh_indices[0], neigh_indices[4], neigh_indices[3]};
            auto tri_indices = quad_to_2tri(dual_vertices, quad_indices);
            face_indices.insert(face_indices.end(), tri_indices.begin(), tri_indices.end());
        }
    }
}

/**
 * Extract flexible dual grid from a triangle mesh.
 *
 * @param vertices: Tensor of shape (N, 3) containing vertex positions.
 * @param faces: Tensor of shape (M, 3) containing triangle vertex indices.
 * @param voxel_size: Tensor of shape (3,) containing the voxel size in each dimension.
 * @param grid_range: Tensor of shape (2, 3) containing the minimum and maximum coordinates of the grid range.
 * @param face_weight: Weight for the face edges in the QEF computation.
 * @param boundary_weight: Weight for the boundary edges in the QEF computation.
 * @param regularization_weight: Regularization factor to apply to the QEF matrices.
 * @param timing: Boolean flag to indicate whether to print timing information.
 *
 * @return a tuple ((x, y, z), vertices, intersected, faces) containing the remeshed vertices and the corresponding voxel grid.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mesh_to_flexible_dual_grid_cpu(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& voxel_size,
    const torch::Tensor& grid_range,
    float face_weight,
    float boundary_weight,
    float regularization_weight,
    bool timing
) {
    const int F = faces.size(0);
    const float* v_ptr = vertices.data_ptr<float>();
    const int* f_ptr = faces.data_ptr<int>();
    const float* voxel_size_ptr = voxel_size.data_ptr<float>();
    const int* grid_range_ptr = grid_range.data_ptr<int>();
    clock_t start, end;
    std::unordered_map<VoxelCoord, size_t> hash_table;
    std::vector<int3> voxels; // Voxel coordinates
    std::vector<Eigen::Vector3f> means; // Mean vertex positions for each voxel
    std::vector<float> cnt; // Number of intersections for each voxel
    std::vector<bool3> intersected; // Indicate whether edges of voxels intersect with surface
    std::vector<Eigen::Matrix4f> qefs; // QEF matrices for each voxel

    // Convert tensors to Eigen types
    Eigen::Vector3f e_voxel_size(voxel_size_ptr[0], voxel_size_ptr[1], voxel_size_ptr[2]);
    Eigen::Vector3i e_grid_min(grid_range_ptr[0], grid_range_ptr[1], grid_range_ptr[2]);
    Eigen::Vector3i e_grid_max(grid_range_ptr[3], grid_range_ptr[4], grid_range_ptr[5]);
    
    // Intersect QEF computation
    start = clock();
    std::vector<Eigen::Vector3f> triangles;
    triangles.reserve(F * 3);
    for (int f = 0; f < F; ++f) {
        for (int v = 0; v < 3; ++v) {
            triangles.push_back(Eigen::Vector3f(
                v_ptr[f_ptr[f * 3 + v] * 3 + 0],
                v_ptr[f_ptr[f * 3 + v] * 3 + 1],
                v_ptr[f_ptr[f * 3 + v] * 3 + 2]
            ));
        }
    }
    intersect_qef(e_voxel_size, e_grid_min, e_grid_max, triangles, hash_table, voxels, means, cnt, intersected, qefs);
    end = clock();
    if (timing) std::cout << "Intersect QEF computation took " << double(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    // Face QEF computation
    if (face_weight > 0.0f) {
        start = clock();
        face_qef(e_voxel_size, e_grid_min, e_grid_max, triangles, hash_table, qefs);
        end = clock();
        if (timing) std::cout << "Face QEF computation took " << double(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;
    }

    // Boundary QEF computation
    if (boundary_weight > 0.0f) {
        start = clock();
        std::map<std::pair<int, int>, int> edge_count;
        for (int f = 0; f < F; ++f) {
            for (int v0 = 0; v0 < 3; ++v0) {
                int e0 = f_ptr[f * 3 + v0];
                int e1 = f_ptr[f * 3 + (v0 + 1) % 3];
                if (e0 > e1) std::swap(e0, e1);
                edge_count[std::make_pair(e0, e1)]++;
            }
        }
        std::vector<Eigen::Vector3f> boundries;
        for (const auto& e : edge_count) {
            if (e.second == 1) {
                int v0 = e.first.first;
                int v1 = e.first.second;
                boundries.push_back(Eigen::Vector3f(
                    v_ptr[v0 * 3 + 0],
                    v_ptr[v0 * 3 + 1],
                    v_ptr[v0 * 3 + 2]
                ));
                boundries.push_back(Eigen::Vector3f(
                    v_ptr[v1 * 3 + 0],
                    v_ptr[v1 * 3 + 1],
                    v_ptr[v1 * 3 + 2]
                ));
            }
        }
        boundry_qef(e_voxel_size, e_grid_min, e_grid_max, boundries, boundary_weight, hash_table, qefs);
        end = clock();
        if (timing) std::cout << "Boundary QEF computation took " << double(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;
    }

    // Solve the QEF system to obtain final dual vertices
    start = clock();
    std::vector<float3> dual_vertices(voxels.size());
    for (int i = 0; i < voxels.size(); ++i) {
        int3 coord = voxels[i];
        Eigen::Matrix4f Q = qefs[i];
        float min_corner[3] = {
            coord.x * e_voxel_size.x(),
            coord.y * e_voxel_size.y(),
            coord.z * e_voxel_size.z()
        };
        float max_corner[3] = {
            (coord.x + 1) * e_voxel_size.x(),
            (coord.y + 1) * e_voxel_size.y(),
            (coord.z + 1) * e_voxel_size.z()
        };

        // Add regularization term
        if (regularization_weight > 0.0f) {
            Eigen::Vector3f p = means[i] / cnt[i];

            // Construct the QEF matrix for this vertex
            Eigen::Matrix4f Qreg = Eigen::Matrix4f::Zero();
            Qreg.topLeftCorner<3,3>() = Eigen::Matrix3f::Identity();
            Qreg.block<3,1>(0,3)    = -p;
            Qreg.block<1,3>(3,0)    = -p.transpose();
            Qreg(3,3)               = p.dot(p);

            Q += regularization_weight * cnt[i] * Qreg; // Scale by regularization weight
        }

        // Solve unconstrained
        Eigen::Matrix3f A = Q.topLeftCorner<3, 3>();
        Eigen::Vector3f b = -Q.block<3, 1>(0, 3);
        Eigen::Vector3f v_new = A.colPivHouseholderQr().solve(b);

        if (!(
            v_new.x() >= min_corner[0] && v_new.x() <= max_corner[0] &&
            v_new.y() >= min_corner[1] && v_new.y() <= max_corner[1] &&
            v_new.z() >= min_corner[2] && v_new.z() <= max_corner[2]
        )) {
            // Starting enumeration of constraints
            float best = std::numeric_limits<float>::infinity();

            // Solve single-constraint
            auto solve_single_constraint = [&](int fixed_axis) {
                int ax1 = (fixed_axis + 1) % 3;
                int ax2 = (fixed_axis + 2) % 3;

                Eigen::Matrix2f A;
                Eigen::Matrix2f B;
                Eigen::Vector2f q, b, x;

                A << Q(ax1, ax1), Q(ax1, ax2),
                     Q(ax2, ax1), Q(ax2, ax2);
                B << Q(ax1, fixed_axis), Q(ax1, 3),
                     Q(ax2, fixed_axis), Q(ax2, 3);
                auto Asol = A.colPivHouseholderQr();

                // if lower bound
                q << min_corner[fixed_axis], 1;
                b = -B * q;
                x = Asol.solve(b);
                if (
                    x.x() >= min_corner[ax1] && x.x() <= max_corner[ax1] &&
                    x.y() >= min_corner[ax2] && x.y() <= max_corner[ax2]
                ) {
                    Eigen::Vector4f p;
                    p[fixed_axis] = min_corner[fixed_axis];
                    p[ax1] = x.x();
                    p[ax2] = x.y();
                    p[3] = 1.0f;
                    float err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }

                // if upper bound
                q << max_corner[fixed_axis], 1;
                b = -B * q;
                x = Asol.solve(b);
                if (
                    x.x() >= min_corner[ax1] && x.x() <= max_corner[ax1] &&
                    x.y() >= min_corner[ax2] && x.y() <= max_corner[ax2]
                ) {
                    Eigen::Vector4f p;
                    p[fixed_axis] = max_corner[fixed_axis];
                    p[ax1] = x.x();
                    p[ax2] = x.y();
                    p[3] = 1.0f;
                    float err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }
            };
            solve_single_constraint(0); // fix x
            solve_single_constraint(1); // fix y
            solve_single_constraint(2); // fix z

            // Solve two-constraint
            auto solve_two_constraint = [&](int free_axis) {
                int ax1 = (free_axis + 1) % 3;
                int ax2 = (free_axis + 2) % 3;

                float a, x;
                Eigen::Vector3f b, q;

                a = Q(free_axis, free_axis);
                b << Q(free_axis, ax1), Q(free_axis, ax2), Q(free_axis, 3);

                // if lower-lower bound
                q << min_corner[ax1], min_corner[ax2], 1;
                x = -(b.dot(q)) / a;
                if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                    Eigen::Vector4f p;
                    p[free_axis] = x;
                    p[ax1] = min_corner[ax1];
                    p[ax2] = min_corner[ax2];
                    p[3] = 1.0f;
                    float err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }

                // if lower-upper bound
                q << min_corner[ax1], max_corner[ax2], 1;
                x = -(b.dot(q)) / a;
                if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                    Eigen::Vector4f p;
                    p[free_axis] = x;
                    p[ax1] = min_corner[ax1];
                    p[ax2] = max_corner[ax2];
                    p[3] = 1.0f;
                    float err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }

                // if upper-lower bound
                q << max_corner[ax1], min_corner[ax2], 1;
                x = -(b.dot(q)) / a;
                if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                    Eigen::Vector4f p;
                    p[free_axis] = x;
                    p[ax1] = max_corner[ax1];
                    p[ax2] = min_corner[ax2];
                    p[3] = 1.0f;
                    float err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }

                // if upper-upper bound
                q << max_corner[ax1], max_corner[ax2], 1;
                x = -(b.dot(q)) / a;
                if (x >= min_corner[free_axis] && x <= max_corner[free_axis]) {
                    Eigen::Vector4f p;
                    p[free_axis] = x;
                    p[ax1] = max_corner[ax1];
                    p[ax2] = max_corner[ax2];
                    p[3] = 1.0f;
                    float err = p.transpose() * Q * p;
                    if (err < best) {
                        best = err;
                        v_new << p[0], p[1], p[2];
                    }
                }
            };
            solve_two_constraint(0); // free x
            solve_two_constraint(1); // free y
            solve_two_constraint(2); // free z

            // Solve three-constraint
            for (int x_constraint = 0; x_constraint < 2; ++x_constraint) {
                for (int y_constraint = 0; y_constraint < 2; ++y_constraint) {
                    for (int z_constraint = 0; z_constraint < 2; ++z_constraint) {
                        Eigen::Vector4f p;
                        p[0] = x_constraint ? min_corner[0] : max_corner[0];
                        p[1] = y_constraint ? min_corner[1] : max_corner[1];
                        p[2] = z_constraint ? min_corner[2] : max_corner[2];
                        p[3] = 1.0f;

                        float err = p.transpose() * Q * p;
                        if (err < best) {
                            best = err;
                            v_new << p[0], p[1], p[2];
                        }
                    }
                }
            }
        }

        // Store the dual vertex and voxel grid coordinates
        dual_vertices[i] = float3{v_new.x(), v_new.y(), v_new.z()};
    }
    end = clock();
    if (timing) std::cout << "Dual vertices computation took " << double(end - start) / CLOCKS_PER_SEC << " seconds." << std::endl;

    return std::make_tuple(
        torch::from_blob(voxels.data(), {int(voxels .size()), 3}, torch::kInt32).clone(),
        torch::from_blob(dual_vertices.data(), {int(dual_vertices.size()), 3}, torch::kFloat32).clone(),
        torch::from_blob(intersected.data(), {int(intersected.size()), 3}, torch::kBool).clone()
    );
}

