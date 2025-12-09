/*
 * Hashmap
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


#define BLOCK_SIZE 256


/**
 * Insert keys into the hashmap
 * 
 * @param hashmap_keys      [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_values    [N] uint32/uint64 tensor containing the hashmap values
 * @param keys              [M] uint32/uint64 tensor containing the keys to be inserted
 * @param values            [M] uint32/uint64 tensor containing the values to be inserted
 */
void hashmap_insert_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    const torch::Tensor& values
);


/**
 * Lookup keys in the hashmap
 * 
 * @param hashmap_keys      [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_values    [N] uint32/uint64 tensor containing the hashmap values
 * @param keys              [M] uint32/uint64 tensor containing the keys to be looked up
 * @return                  [M] uint32/uint64 tensor containing the values of the keys
 */
torch::Tensor hashmap_lookup_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& keys
);


/**
 * Insert 3D coordinates into the hashmap
 * 
 * @param hashmap_keys      [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_values    [N] uint32/uint64 tensor containing the hashmap values
 * @param coords            [M, 4] int32 tensor containing the keys to be inserted
 * @param values            [M] uint32/uint64 tensor containing the values to be inserted
 * @param W                 the number of width dimensions
 * @param H                 the number of height dimensions
 * @param D                 the number of depth dimensions
 */
void hashmap_insert_3d_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    const torch::Tensor& values,
    int W,
    int H,
    int D
);


/**
 * Lookup 3D coordinates in the hashmap
 * 
 * @param hashmap_keys      [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_values    [N] uint32/uint64 tensor containing the hashmap values
 * @param coords            [M, 4] int32 tensor containing the keys to be looked up
 * @param W                 the number of width dimensions
 * @param H                 the number of height dimensions
 * @param D                 the number of depth dimensions
 * 
 * @return                  [M] uint32/uint64 tensor containing the values of the keys
 */
torch::Tensor hashmap_lookup_3d_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W,
    int H,
    int D
);


/**
 * Insert 3D coordinates into the hashmap using index as value
 * 
 * @param hashmap_keys      [N] uint32/uint64 tensor containing the hashmap keys
 * @param hashmap_values    [N] uint32/uint64 tensor containing the hashmap values
 * @param coords            [M, 4] int32 tensor containing the keys to be inserted
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 */
void hashmap_insert_3d_idx_as_val_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W,
    int H,
    int D
);
