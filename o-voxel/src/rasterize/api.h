/*
 * Sparse Voxel Rasterizer
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


/**
 * Rasterize a sparse voxel octree with CUDA backend
 * 
 * @param positions         Tensor of shape (N, 3) containing the positions of the octree nodes in [0, 1]^3
 * @param attrs             Tensor of shape (N, 1) containing the attributes of the octree nodes
 * @param voxel_size        Float containing the size of the voxels
 * @param viewmatrix        Tensor of shape (4, 4) containing the view matrix
 * @param projmatrix        Tensor of shape (4, 4) containing the projection matrix
 * @param campos            Tensor of shape (3) containing the camera position
 * @param tan_fovx          Float containing the tangent of the horizontal field of view
 * @param tan_fovy          Float containing the tangent of the vertical field of view
 * @param image_height      Integer containing the image height
 * @param image_width       Integer containing the image width
 * 
 * @return A tuple containing:
 *  - Tensor of shape (C, H, W) containing the output color
 *  - Tensor of shape (H, W) containing the output depth
 *  - Tensor of shape (H, W) containing the output alpha
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_voxels_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& attrs,
    const float voxel_size,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width
);
