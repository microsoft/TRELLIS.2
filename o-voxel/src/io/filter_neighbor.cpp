#include <torch/extension.h>
#include "api.h"

#include <cstdint>
#include <cmath>
#include <vector>


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
) {
    size_t N = coord.size(0);
    size_t C = attr.size(1);
    int* coord_data = coord.data_ptr<int>();
    uint8_t* attr_data = attr.data_ptr<uint8_t>();
    std::vector<uint8_t> buffer(res * res * res * (C + 1), 0);
    
    // Densify the coordinates
    for (int i = 0; i < N; i++) {
        int x = coord_data[i * 3 + 0];
        int y = coord_data[i * 3 + 1];
        int z = coord_data[i * 3 + 2];
        int ptr = (z * res * res + y * res + x) * (C + 1);
        buffer[ptr + C] = 1;
        for (int c = 0; c < C; c++) {
            buffer[ptr + c] = attr_data[i * C + c];
        }
    }
    
    // Compute the deltas
    for (int z = res-1; z >= 0; z--) {
        for (int y = res-1; y >= 0; y--) {
            for (int x = res-1; x >= 0; x--) {
                int ptr = (z * res * res + y * res + x) * (C + 1);
                int neignbor_ptr = -1;
                int tmp_ptr;
                if (!buffer[ptr + C]) continue;
                // x
                tmp_ptr = (z * res * res + y * res + (x - 1)) * (C + 1);
                if (x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // y
                tmp_ptr = (z * res * res + (y - 1) * res + x) * (C + 1);
                if (y > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // z
                tmp_ptr = ((z - 1) * res * res + y * res + x) * (C + 1);
                if (z > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // xy
                tmp_ptr = (z * res * res + (y - 1) * res + (x - 1)) * (C + 1);
                if (y > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // xz
                tmp_ptr = ((z - 1) * res * res + y * res + (x - 1)) * (C + 1);
                if (z > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // yz
                tmp_ptr = ((z - 1) * res * res + (y - 1) * res + x) * (C + 1);
                if (z > 0 && y > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // xyz
                tmp_ptr = ((z - 1) * res * res + (y - 1) * res + (x - 1)) * (C + 1);
                if (z > 0 && y > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                if (neignbor_ptr >= 0) {
                    for (int c = 0; c < C; c++) {
                        buffer[ptr + c] -= buffer[neignbor_ptr + c];
                    }
                }
            }
        }
    }

    // Pack the deltas into a uint8 tensor
    torch::Tensor delta = torch::zeros({N, C}, torch::dtype(torch::kUInt8));
    uint8_t* delta_data = delta.data_ptr<uint8_t>();
    for (int i = 0; i < N; i++) {
        int x = coord_data[i * 3 + 0];
        int y = coord_data[i * 3 + 1];
        int z = coord_data[i * 3 + 2];
        int ptr = (z * res * res + y * res + x) * (C + 1);
        for (int c = 0; c < C; c++) {
            delta_data[i * C + c] = buffer[ptr + c];
        }
    }
    return delta;
}


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
) {
    size_t N = coord.size(0);
    size_t C = delta.size(1);
    int* coord_data = coord.data_ptr<int>();
    uint8_t* delta_data = delta.data_ptr<uint8_t>();
    std::vector<uint8_t> buffer(res * res * res * (C + 1), 0);
    
    // Densify the coordinates
    for (int i = 0; i < N; i++) {
        int x = coord_data[i * 3 + 0];
        int y = coord_data[i * 3 + 1];
        int z = coord_data[i * 3 + 2];
        int ptr = (z * res * res + y * res + x) * (C + 1);
        buffer[ptr + C] = 1;
        for (int c = 0; c < C; c++) {
            buffer[ptr + c] = delta_data[i * C + c];
        }
    }
    
    // Reconstruct the attribute
    for (int z = 0; z < res; z++) {
        for (int y = 0; y < res; y++) {
            for (int x = 0; x < res; x++) {
                int ptr = (z * res * res + y * res + x) * (C + 1);
                int neignbor_ptr = -1;
                int tmp_ptr;
                if (!buffer[ptr + C]) continue;
                // x
                tmp_ptr = (z * res * res + y * res + (x - 1)) * (C + 1);
                if (x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // y
                tmp_ptr = (z * res * res + (y - 1) * res + x) * (C + 1);
                if (y > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // z
                tmp_ptr = ((z - 1) * res * res + y * res + x) * (C + 1);
                if (z > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // xy
                tmp_ptr = (z * res * res + (y - 1) * res + (x - 1)) * (C + 1);
                if (y > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // xz
                tmp_ptr = ((z - 1) * res * res + y * res + (x - 1)) * (C + 1);
                if (z > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // yz
                tmp_ptr = ((z - 1) * res * res + (y - 1) * res + x) * (C + 1);
                if (z > 0 && y > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                // xyz
                tmp_ptr = ((z - 1) * res * res + (y - 1) * res + (x - 1)) * (C + 1);
                if (z > 0 && y > 0 && x > 0 && buffer[tmp_ptr + C]) neignbor_ptr = tmp_ptr;
                if (neignbor_ptr >= 0) {
                    for (int c = 0; c < C; c++) {
                        buffer[ptr + c] += buffer[neignbor_ptr + c];
                    }
                }
            }
        }
    }

    // Pack the attribute into a uint8 tensor
    torch::Tensor attr = torch::zeros({N, C}, torch::dtype(torch::kUInt8));
    uint8_t* attr_data = attr.data_ptr<uint8_t>();
    for (int i = 0; i < N; i++) {
        int x = coord_data[i * 3 + 0];
        int y = coord_data[i * 3 + 1];
        int z = coord_data[i * 3 + 2];
        int ptr = (z * res * res + y * res + x) * (C + 1);
        for (int c = 0; c < C; c++) {
            attr_data[i * C + c] = buffer[ptr + c];
        }
    }
    return attr;
}
