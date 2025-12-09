/*
 * Efficient Sparse Voxel storage as Sparse Voxel Zip files (.svz)
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


/**
 * Encode a list of sparse voxel morton codes into a sparse voxel octree
 * NOTE: The input indices must be sorted in ascending order
 * 
 * @param codes    [N] uint32 tensor containing the morton codes
 * @param depth    The depth of the sparse voxel octree
 * 
 * @return         uint8 tensor containing the sparse voxel octree
 */
torch::Tensor encode_sparse_voxel_octree_cpu(
    const torch::Tensor& codes,
    const uint32_t depth
);


/**
 * Decode a sparse voxel octree into a list of sparse voxel morton codes
 * 
 * @param octree   uint8 tensor containing the sparse voxel octree
 * @param depth    The depth of the sparse voxel octree
 * 
 * @return         [N] uint32 tensor containing the morton codes
 *                 The codes are sorted in ascending order
 */
torch::Tensor decode_sparse_voxel_octree_cpu(
    const torch::Tensor& octree,
    const uint32_t depth
);



/**
 * Encode the attribute of a sparse voxel octree into deltas from its parent node.
 * 
 * @param octree   uint8 tensor containing the sparse voxel octree
 * @param depth    The depth of the sparse voxel octree
 * @param attr     [N, C] tensor containing the attribute of each sparse voxel
 * 
 * @return         uint8 tensor containing the deltas
 */
torch::Tensor encode_sparse_voxel_octree_attr_parent_cpu(
    const torch::Tensor& octree,
    const uint32_t depth,
    const torch::Tensor& attr
);


/**
 * Decode the attribute of a sparse voxel octree from its parent node and its deltas.
 * 
 * @param octree   uint8 tensor containing the sparse voxel octree
 * @param depth    The depth of the sparse voxel octree
 * @param delta    uint8 tensor containing the deltas
 * 
 * @return         [N, C] tensor containing the attribute of each sparse voxel
 */
torch::Tensor decode_sparse_voxel_octree_attr_parent_cpu(
    const torch::Tensor& octree,
    const uint32_t depth,
    const torch::Tensor& delta
);


/**
 * Encode the attribute of a sparse voxel octree into deltas from its neighbors.
 * 
 * @param coord    [N, 3] tensor containing the coordinates of each sparse voxel
 * @param res      The resolution of the sparse voxel grid
 * @param attr     [N, C] tensor containing the attribute of each sparse voxel
 * 
 * @return         uint8 tensor containing the deltas
 */
torch::Tensor encode_sparse_voxel_octree_attr_neighbor_cpu(
    const torch::Tensor& coord,
    const uint32_t res,
    const torch::Tensor& attr
);


/**
 * Decode the attribute of a sparse voxel octree from its neighbors and deltas.
 * 
 * @param coord    [N, 3] tensor containing the coordinates of each sparse voxel
 * @param res      The resolution of the sparse voxel grid
 * @param delta    [N, C] tensor containing the deltas
 * 
 * @return         [N, C] tensor containing the attribute of each sparse voxel
 */
torch::Tensor decode_sparse_voxel_octree_attr_neighbor_cpu(
    const torch::Tensor& coord,
    const uint32_t res,
    const torch::Tensor& delta
);
