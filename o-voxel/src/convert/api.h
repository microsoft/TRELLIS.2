/*
 * O-Voxel Convertion API
 *
 * Copyright (C) 2025, Jianfeng XIANG <belljig@outlook.com>
 * All rights reserved.
 *
 * Licensed under The MIT License [see LICENSE for details]
 *
 * Written by Jianfeng XIANG
 */

#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <vector>

/**
 * Extract flexible dual grid from a triangle mesh.
 *
 * @param vertices: Tensor of shape (N, 3) containing vertex positions.
 * @param faces: Tensor of shape (M, 3) containing triangle vertex indices.
 * @param voxel_size: Host vector of length 3 containing the voxel size in each dimension.
 * @param grid_range: Host vector of length 6 containing the minimum and maximum coordinates of the grid range.
 * @param face_weight: Weight for the face edges in the QEM computation.
 * @param boundary_weight: Weight for the boundary edges in the QEM computation.
 * @param regularization_weight: Regularization factor to apply to the QEM matrices.
 * @param timing: Boolean flag to indicate whether to print timing information.
 *
 * @return a tuple (voxels, dual_vertices, intersected) containing the sparse dual grid.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mesh_to_flexible_dual_grid_cpu(
    const torch::Tensor &vertices,
    const torch::Tensor &faces,
    const std::vector<float> &voxel_size,
    const std::vector<int64_t> &grid_range,
    float face_weight,
    float boundary_weight,
    float regularization_weight,
    bool timing);

/**
 * CPU-only occupancy pass for pre-gathered triangles.
 *
 * Input triangles are [T, 3, 3] float32 in grid-local coordinates. The function
 * uses the same triangle/voxel intersection rules as the CPU flexible dual grid
 * pipeline, but returns only occupied voxel coordinates [N, 3] int32.
 */
torch::Tensor intersect_occ_cpu(
    const torch::Tensor &triangles,
    const std::vector<float> &voxel_size,
    const std::vector<int64_t> &grid_range);

namespace o_voxel::fdg
{

    /**
     * CUDA occupancy pass for pre-gathered triangles.
     *
     * This shares the same active-brick construction used by intersect_qef_cuda,
     * but stops after compacting occupied voxels. It is useful when only
     * occupancy is needed and mean/QEF/intersection flags would be wasted work.
     */
    torch::Tensor intersect_occ_cuda(
        const torch::Tensor &triangles,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range);

    /**
     * CUDA triangle intersection and QEF pass.
     *
     * Input triangles are [T, 3, 3] float32 in grid-local coordinates. Large
     * triangles are split into small scan tasks so many GPU threads can share
     * their work. The return tuple is:
     *   0 voxels          [N, 3]  int32
     *   1 mean_sum        [N, 3]  float32
     *   2 cnt             [N]     float32
     *   3 intersected     [N, 3]  bool
     *   4 qefs            [N, 10] float32, SymQEF10 layout
     *   5 brick_hash_keys [H]     uint64
     *   6 brick_hash_vals [H]     uint32
     *   7 brick_bits      [B, 16] uint32
     *   8 brick_base      [B]     int64
     *
     * The brick hash, bitset, and base tensors are lookup data for later
     * face_qef_cuda and boundary_qef_cuda calls.
     */
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
        const std::vector<int64_t> &grid_range);

    /**
     * In-place CUDA face QEF accumulation.
     *
     * Each triangle is paired with the active bricks overlapped by its bounding
     * box. Threads then inspect only occupied voxels inside those bricks and add
     * face_weight * face_qef directly into qefs [N, 10].
     */
    torch::Tensor face_qef_cuda(
        const torch::Tensor &triangles,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range,
        const torch::Tensor &voxels,
        const torch::Tensor &qefs,
        float face_weight,
        const torch::Tensor &brick_hash_keys,
        const torch::Tensor &brick_hash_vals,
        const torch::Tensor &brick_bits,
        const torch::Tensor &brick_base);

    /**
     * In-place CUDA boundary QEF accumulation.
     *
     * Boundaries are [E, 2, 3] float32 segments in grid-local coordinates. Each
     * thread walks one segment through the voxel grid and adds boundary_weight *
     * boundary_qef only to voxels found through the active brick lookup.
     */
    torch::Tensor boundary_qef_cuda(
        const torch::Tensor &boundaries,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range,
        float boundary_weight,
        const torch::Tensor &voxels,
        const torch::Tensor &qefs,
        const torch::Tensor &brick_hash_keys,
        const torch::Tensor &brick_hash_vals,
        const torch::Tensor &brick_bits,
        const torch::Tensor &brick_base);

    /**
     * Standalone CUDA octree voxelization.
     *
     * This is not part of mesh_to_flexible_dual_grid_cuda. It expands octree
     * jobs from coarse to fine cells on the GPU and returns (prim_ids, voxels),
     * where prim_ids is [K] int32 and voxels is [K, 3] int32.
     */
    std::tuple<torch::Tensor, torch::Tensor>
    voxelize_mesh_octree_cuda(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range);

    /**
     * Full CUDA flexible dual grid pipeline.
     *
     * This is a parallel implementation of the CPU pipeline semantics:
     * gather triangles, compute intersection QEFs, accumulate face and boundary
     * QEFs in-place, then solve the constrained QEF for each voxel.
     *
     * @return (voxels [N, 3] int32, dual_vertices [N, 3] float32,
     *          intersected [N, 3] bool)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    mesh_to_flexible_dual_grid_cuda(
        const torch::Tensor &vertices,
        const torch::Tensor &faces,
        const std::vector<float> &voxel_size,
        const std::vector<int64_t> &grid_range,
        float face_weight,
        float boundary_weight,
        float regularization_weight);

} // namespace o_voxel::fdg

/**
 * Voxelizes a triangle mesh with PBR materials
 *
 * @param voxel_size                    [3] tensor containing the size of a voxel
 * @param grid_range                    [6] tensor containing the size of the grid
 * @param vertices                      [N_tri, 3, 3] array containing the triangle vertices
 * @param normals                       [N_tri, 3, 3] array containing the triangle vertex normals
 * @param uvs                           [N_tri, 3, 2] tensor containing the texture coordinates
 * @param materialIds                   [N_tri] tensor containing the material ids
 * @param baseColorFactor               list of [3] tensor containing the base color factor
 * @param baseColorTexture              list of [H, W, 3] tensor containing the base color texture
 * @param baseColorTextureFilter        list of int indicating the base color texture filter (0: NEAREST, 1: LINEAR)
 * @param baseColorTextureWrap          list of int indicating the base color texture wrap (0: REPEAT, 1: CLAMP_TO_EDGE, 2: MIRRORED_REPEAT)
 * @param metallicFactor                list of float containing the metallic factor
 * @param metallicTexture               list of [H, W] tensor containing the metallic texture
 * @param metallicTextureFilter         list of int indicating the metallic texture filter (0: NEAREST, 1: LINEAR)
 * @param metallicTextureWrap           list of int indicating the metallic texture wrap (0: REPEAT, 1: CLAMP_TO_EDGE, 2: MIRRORED_REPEAT)
 * @param roughnessFactor               list of float containing the roughness factor
 * @param roughnessTexture              list of [H, W] tensor containing the roughness texture
 * @param roughnessTextureFilter        list of int indicating the roughness texture filter (0: NEAREST, 1: LINEAR)
 * @param roughnessTextureWrap          list of int indicating the roughness texture wrap (0: REPEAT, 1: CLAMP_TO_EDGE, 2: MIRRORED_REPEAT)
 * @param emissiveFactor                list of [3] tensor containing the emissive factor
 * @param emissiveTexture               list of [H, W, 3] tensor containing the emissive texture
 * @param emissiveTextureFilter         list of int indicating the emissive texture filter (0: NEAREST, 1: LINEAR)
 * @param emissiveTextureWrap           list of int indicating the emissive texture wrap (0: REPEAT, 1: CLAMP_TO_EDGE, 2: MIRRORED_REPEAT)
 * @param alphaMode                     list of int indicating the alpha mode (0: OPAQUE, 1: MASK, 2: BLEND)
 * @param alphaCutoff                   list of float containing the alpha cutoff
 * @param alphaFactor                   list of float containing the alpha factor
 * @param alphaTexture                  list of [H, W] tensor containing the alpha texture
 * @param alphaTextureFilter            list of int indicating the alpha texture filter (0: NEAREST, 1: LINEAR)
 * @param alphaTextureWrap              list of int indicating the alpha texture wrap (0: REPEAT, 1: CLAMP_TO_EDGE, 2: MIRRORED_REPEAT)
 * @param normalTexture                 list of [H, W, 3] tensor containing the normal texture
 * @param normalTextureFilter           list of int indicating the normal texture filter (0: NEAREST, 1: LINEAR)
 * @param normalTextureWrap             list of int indicating the normal texture wrap (0: REPEAT, 1: CLAMP_TO_EDGE, 2: MIRRORED_REPEAT)
 * @param mipLevelOffset                float indicating the mip level offset for texture mipmap
 *
 * @return tuple containing:
 *   - coords: tensor of shape [N, 3] containing the voxel coordinates
 *   - out_baseColor: tensor of shape [N, 3] containing the base color of each voxel
 *   - out_metallic: tensor of shape [N, 1] containing the metallic of each voxel
 *   - out_roughness: tensor of shape [N, 1] containing the roughness of each voxel
 *   - out_emissive: tensor of shape [N, 3] containing the emissive of each voxel
 *   - out_alpha: tensor of shape [N, 1] containing the alpha of each voxel
 *   - out_normal: tensor of shape [N, 3] containing the normal of each voxel
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
textured_mesh_to_volumetric_attr_cpu(
    const torch::Tensor &voxel_size,
    const torch::Tensor &grid_range,
    const torch::Tensor &vertices,
    const torch::Tensor &normals,
    const torch::Tensor &uvs,
    const torch::Tensor &materialIds,
    const std::vector<torch::Tensor> &baseColorFactor,
    const std::vector<torch::Tensor> &baseColorTexture,
    const std::vector<int> &baseColorTextureFilter,
    const std::vector<int> &baseColorTextureWrap,
    const std::vector<float> &metallicFactor,
    const std::vector<torch::Tensor> &metallicTexture,
    const std::vector<int> &metallicTextureFilter,
    const std::vector<int> &metallicTextureWrap,
    const std::vector<float> &roughnessFactor,
    const std::vector<torch::Tensor> &roughnessTexture,
    const std::vector<int> &roughnessTextureFilter,
    const std::vector<int> &roughnessTextureWrap,
    const std::vector<torch::Tensor> &emissiveFactor,
    const std::vector<torch::Tensor> &emissiveTexture,
    const std::vector<int> &emissiveTextureFilter,
    const std::vector<int> &emissiveTextureWrap,
    const std::vector<int> &alphaMode,
    const std::vector<float> &alphaCutoff,
    const std::vector<float> &alphaFactor,
    const std::vector<torch::Tensor> &alphaTexture,
    const std::vector<int> &alphaTextureFilter,
    const std::vector<int> &alphaTextureWrap,
    const std::vector<torch::Tensor> &normalTexture,
    const std::vector<int> &normalTextureFilter,
    const std::vector<int> &normalTextureWrap,
    const float mipLevelOffset,
    const bool timing);
