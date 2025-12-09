#include <torch/extension.h>
#include "hash/api.h"
#include "convert/api.h"
#include "io/api.h"
#include "serialize/api.h"
#include "rasterize/api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Hash functions
    m.def("hashmap_insert_cuda", &hashmap_insert_cuda);
    m.def("hashmap_lookup_cuda", &hashmap_lookup_cuda);
    m.def("hashmap_insert_3d_cuda", &hashmap_insert_3d_cuda);
    m.def("hashmap_lookup_3d_cuda", &hashmap_lookup_3d_cuda);
    m.def("hashmap_insert_3d_idx_as_val_cuda", &hashmap_insert_3d_idx_as_val_cuda);
    // Convert functions
    m.def("mesh_to_flexible_dual_grid_cpu", &mesh_to_flexible_dual_grid_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("textured_mesh_to_volumetric_attr_cpu", &textured_mesh_to_volumetric_attr_cpu, py::call_guard<py::gil_scoped_release>());
    // Serialization functions
    m.def("z_order_encode_cpu", &z_order_encode_cpu, py::call_guard<py::gil_scoped_release>());
	m.def("z_order_decode_cpu", &z_order_decode_cpu, py::call_guard<py::gil_scoped_release>());
	m.def("hilbert_encode_cpu", &hilbert_encode_cpu, py::call_guard<py::gil_scoped_release>());
	m.def("hilbert_decode_cpu", &hilbert_decode_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("z_order_encode_cuda", &z_order_encode_cuda, py::call_guard<py::gil_scoped_release>());
	m.def("z_order_decode_cuda", &z_order_decode_cuda, py::call_guard<py::gil_scoped_release>());
	m.def("hilbert_encode_cuda", &hilbert_encode_cuda, py::call_guard<py::gil_scoped_release>());
	m.def("hilbert_decode_cuda", &hilbert_decode_cuda, py::call_guard<py::gil_scoped_release>());
    // IO functions
    m.def("encode_sparse_voxel_octree_cpu", &encode_sparse_voxel_octree_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_cpu", &decode_sparse_voxel_octree_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("encode_sparse_voxel_octree_attr_parent_cpu", &encode_sparse_voxel_octree_attr_parent_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_attr_parent_cpu", &decode_sparse_voxel_octree_attr_parent_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("encode_sparse_voxel_octree_attr_neighbor_cpu", &encode_sparse_voxel_octree_attr_neighbor_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_attr_neighbor_cpu", &decode_sparse_voxel_octree_attr_neighbor_cpu, py::call_guard<py::gil_scoped_release>());
    // Rasterization functions
    m.def("rasterize_voxels_cuda", &rasterize_voxels_cuda);
}