import os
import json
from typing import *
import numpy as np
import torch
from .. import models
from .components import StandardDatasetBase, ImageConditionedMixin
from ..modules.sparse import SparseTensor, sparse_cat
from ..representations import MeshWithVoxel
from ..utils.data_utils import load_balanced_group_indices
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics


class SLatPbrVisMixin:
    def __init__(
        self,
        *args,
        pretrained_pbr_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pbr_slat_dec = None
        self.pretrained_pbr_slat_dec = pretrained_pbr_slat_dec
        self.pbr_slat_dec_path = pbr_slat_dec_path
        self.pbr_slat_dec_ckpt = pbr_slat_dec_ckpt
        self.shape_slat_dec = None
        self.pretrained_shape_slat_dec = pretrained_shape_slat_dec
        self.shape_slat_dec_path = shape_slat_dec_path
        self.shape_slat_dec_ckpt = shape_slat_dec_ckpt
        
    def _loading_slat_dec(self):
        if self.pbr_slat_dec is not None and self.shape_slat_dec is not None:
            return
        if self.pbr_slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.pbr_slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.pbr_slat_dec_path, 'ckpts', f'decoder_{self.pbr_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_pbr_slat_dec)
        self.pbr_slat_dec = decoder.cuda().eval()

        if self.shape_slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.shape_slat_dec_path, 'config.json'), 'r'))
            cfg['models']['decoder']['args']['resolution'] = self.resolution
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.shape_slat_dec_path, 'ckpts', f'decoder_{self.shape_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_shape_slat_dec)
        self.shape_slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        del self.pbr_slat_dec
        self.pbr_slat_dec = None
        del self.shape_slat_dec
        self.shape_slat_dec = None
        
    @torch.no_grad()
    def decode_latent(self, z, shape_z, batch_size=4):
        self._loading_slat_dec()
        reps = []
        if self.shape_slat_normalization is not None:
            shape_z = shape_z * self.shape_slat_std.to(z.device) + self.shape_slat_mean.to(z.device)
        if self.pbr_slat_normalization is not None:
            z = z * self.pbr_slat_std.to(z.device) + self.pbr_slat_mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            mesh, subs = self.shape_slat_dec(shape_z[i:i+batch_size], return_subs=True)
            vox = self.pbr_slat_dec(z[i:i+batch_size], guide_subs=subs)
            reps.extend([
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / self.resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout = self.layout,
                )
                for m, v in zip(mesh, vox)
            ])
        self._delete_slat_dec()
        return reps
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        shape_z = sample['concat_cond'].cuda()
        z = sample['x_0'].cuda()
        reps = self.decode_latent(z, shape_z)
        
        # build camera
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)
        
        # render
        renderer = get_renderer(reps[0])
        images = {k: [] for k in self.layout}
        for representation in reps:
            image = {k: torch.zeros(3, 1024, 1024).cuda() for k in self.layout}
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, return_types=['attr'])
                for k in self.layout:
                    image[k][:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res[k]
            for k in self.layout:
                images[k].append(image[k])
        for k in self.layout:
            images[k] = torch.stack(images[k], dim=0)
        return images
    
    
class SLatPbr(SLatPbrVisMixin, StandardDatasetBase):
    """
    structured latent for sparse voxel pbr dataset
    
    Args:
        roots (str): path to the dataset
        latent_key (str): key of the latent to be used
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        resolution (int): resolution of decoded sparse voxel
        attrs (list): attributes to be decoded
        pretained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        full_pbr: bool = False,
        pbr_slat_normalization: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        attrs: list[str] = ['base_color', 'metallic', 'roughness', 'emissive', 'alpha'],
        pretrained_pbr_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):  
        self.resolution = resolution
        self.pbr_slat_normalization = pbr_slat_normalization
        self.shape_slat_normalization = shape_slat_normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.full_pbr = full_pbr
        self.value_range = (-1, 1)
        
        super().__init__(
            roots,
            pretrained_pbr_slat_dec=pretrained_pbr_slat_dec,
            pbr_slat_dec_path=pbr_slat_dec_path,
            pbr_slat_dec_ckpt=pbr_slat_dec_ckpt,
            pretrained_shape_slat_dec=pretrained_shape_slat_dec,
            shape_slat_dec_path=shape_slat_dec_path,
            shape_slat_dec_ckpt=shape_slat_dec_ckpt,
            **kwargs
        )
        
        self.loads = [self.metadata.loc[sha256, 'pbr_latent_tokens'] for _, sha256 in self.instances]
        
        if self.pbr_slat_normalization is not None:
            self.pbr_slat_mean = torch.tensor(self.pbr_slat_normalization['mean']).reshape(1, -1)
            self.pbr_slat_std = torch.tensor(self.pbr_slat_normalization['std']).reshape(1, -1)
        
        if self.shape_slat_normalization is not None:
            self.shape_slat_mean = torch.tensor(self.shape_slat_normalization['mean']).reshape(1, -1)
            self.shape_slat_std = torch.tensor(self.shape_slat_normalization['std']).reshape(1, -1)
        
        self.attrs = attrs
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]
            
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata['pbr_latent_encoded'] == True]
        stats['With PBR latent'] = len(metadata)
        metadata = metadata[metadata['shape_latent_encoded'] == True]
        stats['With shape latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['pbr_latent_tokens'] <= self.max_tokens]
        stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        if self.full_pbr:
            metadata = metadata[metadata['num_basecolor_tex'] > 0]
            metadata = metadata[metadata['num_metallic_tex'] > 0]
            metadata = metadata[metadata['num_roughness_tex'] > 0]
            stats['Full PBR'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        # PBR latent
        data = np.load(os.path.join(root['pbr_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.pbr_slat_normalization is not None:
            feats = (feats - self.pbr_slat_mean) / self.pbr_slat_std
        pbr_z = SparseTensor(feats, coords)
        
        # Shape latent
        data = np.load(os.path.join(root['shape_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.shape_slat_normalization is not None:
            feats = (feats - self.shape_slat_mean) / self.shape_slat_std
        shape_z = SparseTensor(feats, coords)
        
        assert torch.equal(shape_z.coords, pbr_z.coords), \
            f"Shape latent and PBR latent have different coordinates: {shape_z.coords.shape} vs {pbr_z.coords.shape}"
            
        return {
            'x_0': pbr_z,
            'concat_cond': shape_z,
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['x_0'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], SparseTensor):
                    pack[k] = sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            
            packs.append(pack)
        
        if split_size is None:
            return packs[0]
        return packs


class ImageConditionedSLatPbr(ImageConditionedMixin, SLatPbr):
    """
    Image conditioned structured latent dataset
    """
    pass
