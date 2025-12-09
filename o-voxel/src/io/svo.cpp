#include <torch/extension.h>
#include "api.h"

#include <cstdint>
#include <vector>


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
) {
    size_t N_leaf = codes.size(0);
    int* codes_data = codes.data_ptr<int>();
    
    std::vector<uint8_t> svo;
    std::vector<uint8_t> stack(depth-1);
    std::vector<uint32_t> insert_stack(depth);
    std::vector<uint32_t> stack_ptr(depth);
    uint32_t code, insert_from;

    // Root node
    svo.push_back(0);
    stack_ptr[0] = 0;

    // Iterate over all codes and encode them into SVO
    for (int i = 0; i < N_leaf; i++) {
        code = codes_data[i];

        // Convert code to insert stack (3bit per level)
        for (uint32_t j = 0; j < depth; j++) {
            insert_stack[j] = (code >> (3*(depth-1-j))) & 0x7;
        }

        // Compare insert stack to stack to determine which level to insert
        if (i == 0) {
            // First code, insert at level 0
            insert_from = 0;
        }
        else {
            // Compare insert stack to stack
            for (insert_from = 0; insert_from < depth-1; insert_from++) {
                if (insert_stack[insert_from] != stack[insert_from]) {
                    break;
                }
            }
        }

        // Insert new nodes from insert_from to depth-1
        for (uint32_t j = insert_from; j < depth; j++) {
            // Add new node to SVO
            if (j > insert_from) {
                svo.push_back(0);
                stack_ptr[j] = svo.size()-1;
            }
            // Update parent pointers
            svo[stack_ptr[j]] |= (1 << insert_stack[j]);
            // Update stack
            if (j < depth-1) {
                stack[j] = insert_stack[j];
            }
        }
    }

    // Convert SVO to tensor
    torch::Tensor svo_tensor = torch::from_blob(svo.data(), {svo.size()}, torch::kUInt8).clone();
    return svo_tensor;
}


void decode_sparse_voxel_octree_cpu_recursive(
    const uint8_t* svo,
    const uint32_t depth,
    uint32_t& ptr,
    std::vector<uint8_t>& stack,
    std::vector<uint32_t>& codes
) {
    uint8_t node = svo[ptr];
    if (stack.size() == depth-1) {
        // Leaf node, add code to list
        uint32_t code = 0;
        for (uint32_t i = 0; i < depth-1; i++) {
            code |= (static_cast<uint32_t>(stack[i]) << (3*(depth-1-i)));
        }
        for (uint8_t i = 0; i < 8; i++) {
            if (node & (1 << i)) {
                code = (code & ~0x7) | i;
                codes.push_back(code);
            }
        }
        ptr++;
    }
    else {
        // Internal node, recurse
        ptr++;
        for (uint8_t i = 0; i < 8; i++) {
            if (node & (1 << i)) {
                stack.push_back(i);
                decode_sparse_voxel_octree_cpu_recursive(svo, depth, ptr, stack, codes);
                stack.pop_back();
            }
        }
    }
}


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
) {
    uint8_t* octree_data = octree.data_ptr<uint8_t>();
    std::vector<uint32_t> codes;
    std::vector<uint8_t> stack;
    stack.reserve(depth-2);
    uint32_t ptr = 0;
    // Decode SVO into list of codes
    decode_sparse_voxel_octree_cpu_recursive(octree_data, depth, ptr, stack, codes);
    // Convert codes to tensor
    torch::Tensor codes_tensor = torch::from_blob(codes.data(), {codes.size()}, torch::kInt32).clone();
    return codes_tensor;
}
