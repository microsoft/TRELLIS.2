#include <torch/extension.h>
#include "api.h"
#include "lut.h"

#include <cstdint>
#include <cmath>
#include <vector>


std::vector<uint8_t> encode_recursive(
    const uint8_t* svo,
    const uint32_t depth,
    const uint8_t* attr,
    const size_t C,
    uint32_t& svo_ptr,
    uint32_t& attr_ptr,
    uint32_t& delta_ptr,
    uint32_t self_delta_ptr,
    uint32_t cur_depth,
    uint8_t* delta
) {
    std::vector<uint8_t> node_attr(C, 0);
    if (cur_depth == depth) {
        // Leaf node
        for (size_t i = 0; i < C; i++) {
            node_attr[i] = attr[attr_ptr + i];
            if (self_delta_ptr != 0 || cur_depth == 0) {
                delta[self_delta_ptr + i] = node_attr[i];
            }
        }
        attr_ptr += C;
    }
    else {
        // Internal node
        uint8_t node = svo[svo_ptr];
        uint32_t child_delta_ptr = delta_ptr;
        uint8_t cnt = lut_1cnt[node];
        svo_ptr++;
        delta_ptr += C * (cnt - 1);
        for (uint8_t i = 0; i < cnt; i++) {
            auto child_attr = encode_recursive(
                svo, depth, attr, C, svo_ptr, attr_ptr, delta_ptr, i == cnt-1 ? 0 : child_delta_ptr+i*C, cur_depth+1, delta
            );
            for (size_t j = 0; j < C; j++) {
                if (i == 0) {
                    node_attr[j] = child_attr[j];
                }
                else {
                    delta[child_delta_ptr + (i-1)*C + j] = child_attr[j] - delta[child_delta_ptr + (i-1)*C + j];
                }
            }
        }
        if (self_delta_ptr != 0 || cur_depth == 0) {
            for (size_t i = 0; i < C; i++) {
                delta[self_delta_ptr + i] = node_attr[i];
            }
        }
    }
    return node_attr;
}


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
) {
    size_t N_leaf = attr.size(0);
    size_t N_node = octree.size(0);
    size_t C = attr.size(1);
    uint8_t* octree_data = octree.data_ptr<uint8_t>();
    uint8_t* attr_data = attr.data_ptr<uint8_t>();

    torch::Tensor delta = torch::zeros({N_leaf, C}, torch::kUInt8);
    uint32_t svo_ptr = 0;
    uint32_t attr_ptr = 0;
    uint32_t delta_ptr = C;
    encode_recursive(octree_data, depth, attr_data, C, svo_ptr, attr_ptr, delta_ptr, 0, 0, delta.data_ptr<uint8_t>());

    return delta;
}


void decode_recursive(
    const uint8_t* svo,
    const uint32_t depth,
    const uint8_t* delta,
    const size_t C,
    uint32_t& svo_ptr,
    uint32_t& attr_ptr,
    uint32_t& delta_ptr,
    uint32_t cur_depth,
    uint8_t* cur_attr,
    uint8_t* attr
) {
    if (cur_depth == depth) {
        // Leaf node
        for (size_t i = 0; i < C; i++) {
            attr[attr_ptr + i] = cur_attr[i];
        }
        attr_ptr += C;
    }
    else {
        // Internal node
        uint8_t node = svo[svo_ptr];
        uint32_t child_delta_ptr = delta_ptr;
        std::vector<uint8_t> child_attr(cur_attr, cur_attr + C);
        uint8_t cnt = lut_1cnt[node];
        svo_ptr++;
        delta_ptr += C * (cnt - 1);
        for (uint8_t i = 0; i < cnt; i++) {
            for (size_t j = 0; j < C; j++) {
                if (i > 0) {
                    child_attr[j] += delta[child_delta_ptr + (i-1)*C + j];
                }
            }
            decode_recursive(
                svo, depth, delta, C, svo_ptr, attr_ptr, delta_ptr, cur_depth+1, child_attr.data(), attr
            );
        }
    }
}


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
) {
    size_t N_node = octree.size(0);
    size_t N_leaf = delta.size(0);
    size_t C = delta.size(1);
    uint8_t* octree_data = octree.data_ptr<uint8_t>();
    uint8_t* delta_data = delta.data_ptr<uint8_t>();

    torch::Tensor attr = torch::zeros({N_leaf, C}, torch::kUInt8);
    uint32_t svo_ptr = 0;
    uint32_t attr_ptr = 0;
    uint32_t delta_ptr = C;

    // Recursively decode the attribute
    decode_recursive(
        octree_data, depth, delta_data, C, svo_ptr, attr_ptr, delta_ptr, 0, delta_data, attr.data_ptr<uint8_t>()
    );

    return attr;
}
