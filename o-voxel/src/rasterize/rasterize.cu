#include <cstdint>

#include <cuda.h>
#include "cuda_runtime.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "config.h"
#include "auxiliary.h"
#include "api.h"


/**
 * Preprocess input 3D points
 */
static __global__ void preprocess(
    const int num_nodes,
    const float* positions,
    const float voxel_size,
    const float* viewmatrix,
    const float* projmatrix,
    const int width,
    const int height,
    const dim3 grid,
    int4* bboxes,
    float* depths,
    uint32_t* tiles_touched
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_nodes)
        return;

    // Initialize bboxes and touched tiles to 0. If this isn't changed,
    // this voxel will not be processed further.
    bboxes[idx] = { 0, 0, 0, 0 };
    tiles_touched[idx] = 0;

    // Perform near culling, quit if outside.
    float3 p_orig = {
        positions[3 * idx],
        positions[3 * idx + 1],
        positions[3 * idx + 2]
    };
    float3 p_view;
    if (!in_frustum(idx, p_orig, viewmatrix, projmatrix, p_view))
        return;

    // Project 8 vertices of the voxel to screen space to find the
    // bounding box of the projected points.
    float3 scale = { voxel_size, voxel_size, voxel_size };
    int4 bbox = get_bbox(p_orig, scale, projmatrix, width, height);
    uint2 rect_min, rect_max;
    getRect(bbox, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // Store some useful helper data for the next steps.
    depths[idx] = p_view.z;
    bboxes[idx] = bbox;
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


/**
 * Generates one key/value pair for all voxel / tile overlaps. 
 * Run once per voxel (1:N mapping).
 * 
 * @param P Number of points.
 * @param grid Grid size.
 * @param depths Depths of points.
 * @param offsets Offsets for writing keys/values.
 * @param bboxes Bounding boxes of voxels.
 * @param keys_unsorted Unsorted keys.
 * @param values_unsorted Unsorted values.
 */
static __global__ void duplicateWithKeys(
    int P, dim3 grid,
    const float* depths,
    const int64_t* offsets,
    const int4* bboxes,
    int64_t* keys_unsorted,
    uint32_t* values_unsorted
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Generate no key/value pair for invisible voxels
    if (bboxes[idx].w > 0)
    {
        // Find this voxel's offset in buffer for writing keys/values.
        int64_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;
        getRect(bboxes[idx], rect_min, rect_max, grid);

        // For each tile that the bounding rect overlaps, emit a 
        // key/value pair. The key is |  tile ID  |      depth      |,
        // and the value is the ID of the voxel. Sorting the values 
        // with this key yields voxel IDs in a list, such that they
        // are first sorted by tile and then by depth. 
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                int64_t key = y * grid.x + x;
                key <<= 32;
                key |= *((uint32_t*)&depths[idx]);
                keys_unsorted[off] = key;
                values_unsorted[off] = idx;
                off++;
            }
        }
    }
}


/**
 * Check keys to see if it is at the start/end of one tile's range in the full sorted list. If yes, write start/end of this tile.
 * 
 * @param L Number of points.
 * @param point_list_keys List of keys.
 * @param ranges Ranges of tiles.
 */
static __global__ void identifyTileRanges(int L, int64_t* point_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    int64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32;
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}


/**
 * Main rasterization method. Collaboratively works on one tile per
 * block, each thread treats one pixel. Alternates between fetching 
 * and rasterizing data.
 * 
 * @param ranges Ranges of voxel instances for each tile.
 * @param point_list List of voxel instances.
 * @param C Number of channels.
 * @param W Width of the image.
 * @param H Height of the image.
 * @param cam_pos Camera position.
 * @param tan_fovx Tangent of the horizontal field of view.
 * @param tan_fovy Tangent of the vertical field of view.
 * @param viewmatrix View matrix.
 * @param positions Centers of voxels.
 * @param attrs Attributes of voxels.
 * @param voxel_size Size of voxels.
 * @param out_color Output color.
 * @param out_depth Output depth.
 * @param out_alpha Output alpha.
 */
static __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
render(
    const uint2* ranges,
    const uint32_t* point_list,
    const int C,
    const int W,
    const int H,
    const float* cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const float* viewmatrix,
    const float* positions,
    const float* attrs,
    const float voxel_size,
    float* out_color,
    float* out_depth,
    float* out_alpha
) {
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;

    // Get ray direction and origin for this pixel.
    float3 ray_dir = getRayDir(pix, W, H, tan_fovx, tan_fovy, viewmatrix);

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float3 collected_xyz[BLOCK_SIZE];

    // Initialize helper variables
    int hit = -1;
    float D;

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch per-voxel data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xyz[block.thread_rank()] = {
                positions[3 * coll_id],
                positions[3 * coll_id + 1],
                positions[3 * coll_id + 2]
            };
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            // Get ray-voxel intersection
            float3 p = collected_xyz[j];
            float3 scale = { voxel_size, voxel_size, voxel_size };
            float3 voxel_min = { p.x - 0.5f * scale.x, p.y - 0.5f * scale.y, p.z - 0.5f * scale.z };
            float3 voxel_max = { p.x + 0.5f * scale.x, p.y + 0.5f * scale.y, p.z + 0.5f * scale.z };
            float2 itsc = get_ray_voxel_intersection(*(float3*)cam_pos, ray_dir, voxel_min, voxel_max);
            float itsc_dist = (itsc.y >= itsc.x) ? itsc.y - itsc.x : -1.0f;
            if (itsc_dist <= 0.0f)
                continue;

            hit = collected_id[j];
            D = itsc.x;
            done = true;
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        for (int ch = 0; ch < C; ch++)
            if (hit >= 0) out_color[ch * H * W + pix_id] = attrs[hit * C + ch];
        out_depth[pix_id] = D;
        out_alpha[pix_id] = hit >= 0 ? 1.0f : 0.0f;
    }
}

void forward(
    const int num_nodes,
    const int num_channels,
    const int width,
    const int height,
    const float* positions,
    const float* attrs,
    const float voxel_size,
    const float* viewmatrix,
    const float* projmatrix,
    const float* campos,
    const float tan_fovx,
    const float tan_fovy,
    float* out_color,
    float* out_depth,
    float* out_alpha
) {
    // Parrallel config (2D grid of 2D blocks)
    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Run preprocessing kernel
    auto pt_bboxes = torch::zeros({num_nodes, 4}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto pt_depths = torch::zeros({num_nodes}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto pt_tiles_touched = torch::zeros({num_nodes}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    preprocess<<<(num_nodes+255)/256, 256>>>(
        num_nodes, positions, voxel_size, viewmatrix, projmatrix, width, height, grid,
        reinterpret_cast<int4*>(pt_bboxes.data_ptr<int>()),
        pt_depths.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(pt_tiles_touched.data_ptr<int>())
    );

    // Compute prefix sum over full list of touched tile counts by voxels
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    auto pt_offsets = torch::cumsum(pt_tiles_touched, 0);

    // Retrieve total number of voxel instances to launch
    int num_rendered = pt_offsets[num_nodes - 1].item<int>();
    if (num_rendered == 0) return;

    // For each instance to be rendered, produce adequate [ tile | depth ] key 
    auto pt_keys_unsorted = torch::zeros({num_rendered}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto pt_indices_unsorted = torch::zeros({num_rendered}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    duplicateWithKeys<<<(num_nodes+255)/256, 256>>>(
        num_nodes, grid,
        pt_depths.data_ptr<float>(),
        pt_offsets.data_ptr<int64_t>(),
        reinterpret_cast<int4*>(pt_bboxes.data_ptr<int>()),
        pt_keys_unsorted.data_ptr<int64_t>(),
        reinterpret_cast<uint32_t*>(pt_indices_unsorted.data_ptr<int>())
    );

    // Sort complete list of (duplicated) voxel indices by keys
    auto pt_sorted = torch::sort(pt_keys_unsorted, 0);
    auto pt_keys = std::get<0>(pt_sorted);
    auto pt_order = std::get<1>(pt_sorted);
    auto pt_indices = torch::index_select(pt_indices_unsorted, 0, pt_order);

    // Identify start and end of per-tile workloads in sorted list
    auto tile_ranges = torch::zeros({grid.x * grid.y, 2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    identifyTileRanges<<<(num_rendered+255)/256, 256>>>(
        num_rendered,
        pt_keys.data_ptr<int64_t>(),
        reinterpret_cast<uint2*>(tile_ranges.data_ptr<int>())
    );

    // Let each tile blend its range of voxels independently in parallel
    render<<<grid, block>>>(
        reinterpret_cast<uint2*>(tile_ranges.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(pt_indices.data_ptr<int>()),
        num_channels, width, height,
        campos, tan_fovx, tan_fovy, viewmatrix,
        positions, attrs, voxel_size,
        out_color, out_depth, out_alpha
    );
}


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
) {
    // Sizes
    const int P = positions.size(0);
    const int C = attrs.size(1);
    const int H = image_height;
    const int W = image_width;

    // Types
    torch::TensorOptions float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(positions.device());
    torch::TensorOptions byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(positions.device());

    // Allocate output tensors
    torch::Tensor out_color = torch::zeros({C, H, W}, float_opts);
    torch::Tensor out_depth = torch::zeros({H, W}, float_opts);
    torch::Tensor out_alpha = torch::zeros({H, W}, float_opts);

    // Call Forward
    if (P > 0) {
        forward(
            P, C, W, H,
            positions.contiguous().data_ptr<float>(),
            attrs.contiguous().data_ptr<float>(),
            voxel_size,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx, tan_fovy,
            out_color.contiguous().data_ptr<float>(),
            out_depth.contiguous().data_ptr<float>(),
            out_alpha.contiguous().data_ptr<float>()
        );
    }

    return std::make_tuple(
        out_color, out_depth, out_alpha
    );
}
