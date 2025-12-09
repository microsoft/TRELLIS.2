import os
import numpy as np
import pickle
import torch
import utils3d
from .components import StandardDatasetBase
from ..modules import sparse as sp
from ..renderers import MeshRenderer
from ..representations import Mesh
from ..utils.data_utils import load_balanced_group_indices
import o_voxel


class FlexiDualGridVisMixin:
    @torch.no_grad()
    def visualize_sample(self, x: dict):
        mesh = x['mesh']
        
        renderer = MeshRenderer({'near': 1, 'far': 3})
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)
        
        # Build each representation
        images = []
        for m in mesh:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = \
                    renderer.render(m.cuda(), ext, intr)['normal']
            images.append(image)
        images = torch.stack(images)
        
        return images
    

class FlexiDualGridDataset(FlexiDualGridVisMixin, StandardDatasetBase):
    """
    Flexible Dual Grid Dataset
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(
        self,
        roots,
        resolution: int = 1024,
        max_active_voxels: int = 1000000,
        max_num_faces: int = None,
        min_aesthetic_score: float = 5.0,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_active_voxels = max_active_voxels
        self.max_num_faces = max_num_faces
        self.value_range = (0, 1)

        super().__init__(roots)
        
        self.loads = [self.metadata.loc[sha256, f'dual_grid_size'] for _, sha256 in self.instances]
        
    def __str__(self):
        lines = [
            super().__str__(),
            f'  - Resolution: {self.resolution}',
        ]
        return '\n'.join(lines)
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'dual_grid_converted'] == True]
        stats['Dual Grid Converted'] = len(metadata)
        if self.min_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata[f'dual_grid_size'] <= self.max_active_voxels]
        stats[f'Active Voxels <= {self.max_active_voxels}'] = len(metadata)
        if self.max_num_faces is not None:
            metadata = metadata[metadata['num_faces'] <= self.max_num_faces]
            stats[f'Faces <= {self.max_num_faces}'] = len(metadata)
        return metadata, stats
    
    def read_mesh(self, root, instance):
        with open(os.path.join(root, f'{instance}.pickle'), 'rb') as f:
            dump = pickle.load(f)
        start = 0
        vertices = []
        faces = []
        for obj in dump['objects']:
            if obj['vertices'].size == 0 or obj['faces'].size == 0:
                continue
            vertices.append(obj['vertices'])
            faces.append(obj['faces'] + start)
            start += len(obj['vertices'])
        vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
        faces = torch.from_numpy(np.concatenate(faces, axis=0)).long()
        vertices_min = vertices.min(dim=0)[0]
        vertices_max = vertices.max(dim=0)[0]
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
        return {'mesh': [Mesh(vertices=vertices, faces=faces)]}
    
    def read_dual_grid(self, root, instance):
        coords, attr = o_voxel.io.read_vxz(os.path.join(root, f'{instance}.vxz'), num_threads=4)
        vertices = sp.SparseTensor(
            (attr['vertices'] / 255.0).float(),
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        intersected = vertices.replace(torch.cat([
            attr['intersected'] % 2,
            attr['intersected'] // 2 % 2,
            attr['intersected'] // 4 % 2,
        ], dim=-1).bool())
        return {'vertices': vertices, 'intersected': intersected}

    def get_instance(self, root, instance):
        mesh = self.read_mesh(root['mesh_dump'], instance)
        dual_grid = self.read_dual_grid(root['dual_grid'], instance)
        return {**mesh, **dual_grid}
    
    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['vertices'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], sp.SparseTensor):
                    pack[k] = sp.sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            
            packs.append(pack)
        
        if split_size is None:
            return packs[0]
        return packs
    