from typing import Any, Dict, Tuple
import json
from abc import abstractmethod
import os
import warnings
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        try:
            self.roots = json.loads(roots)
            root_type = 'obj'
        except json.JSONDecodeError:
            self.roots = roots.split(',')
            root_type = 'list'
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        if root_type == 'obj':
            for key, root in self.roots.items():
                self._stats[key] = {}
                metadata = pd.DataFrame(columns=['sha256']).set_index('sha256')
                for _, r in root.items():
                    metadata = metadata.combine_first(pd.read_csv(os.path.join(r, 'metadata.csv')).set_index('sha256'))
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata.index.values])
                self.metadata = pd.concat([self.metadata, metadata])
        else:
            for root in self.roots:
                key = os.path.basename(root)
                self._stats[key] = {}
                metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata)
                self._stats[key].update(stats)
                self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
                metadata.set_index('sha256', inplace=True)
                self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        if len(self) == 0:
            raise IndexError('Cannot load from an empty dataset.')

        max_retries = min(10, len(self))
        root, instance = self.instances[index]
        last_error = None

        for _ in range(max_retries):
            try:
                return self.get_instance(root, instance)
            except Exception as e:
                last_error = e
                warnings.warn(f'Error loading {instance}: {e}')
                root, instance = self.instances[np.random.randint(0, len(self))]

        raise RuntimeError(
            f'Failed to load a valid instance after {max_retries} retries. '
            f'Last attempted instance: {instance}'
        ) from last_error
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append('  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)




def _load_condition_image(image_path: str, image_size: int) -> torch.Tensor:
    with Image.open(image_path) as image:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        alpha_np = np.asarray(image.getchannel(3))
        nz = alpha_np.nonzero()
        if len(nz[0]) == 0:
            crop_box = (0, 0, image.width, image.height)
        else:
            x0, y0 = nz[1].min(), nz[0].min()
            x1, y1 = nz[1].max(), nz[0].max()
            center_x = (x0 + x1) * 0.5
            center_y = (y0 + y1) * 0.5
            half_size = max(x1 - x0, y1 - y0) * 0.5
            crop_box = (
                int(center_x - half_size),
                int(center_y - half_size),
                int(center_x + half_size),
                int(center_y + half_size),
            )

        image = image.crop(crop_box).resize((image_size, image_size), Image.Resampling.LANCZOS)

        alpha = torch.from_numpy(np.asarray(image.getchannel(3))).float().div_(255.0)
        rgb = torch.from_numpy(np.asarray(image.convert('RGB'))).permute(2, 0, 1).float().div_(255.0)

    return rgb.mul_(alpha.unsqueeze(0))


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        view = np.random.randint(len(metadata['frames']))
        image_path = os.path.join(image_root, metadata['frames'][view]['file_path'])
        pack['cond'] = _load_condition_image(image_path, self.image_size)

        return pack


class MultiImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, max_image_cond_view = 4, **kwargs):
        self.image_size = image_size
        self.max_image_cond_view = max_image_cond_view
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        n_views = len(metadata['frames'])
        n_sample_views = np.random.randint(1, self.max_image_cond_view+1)

        assert n_views >= n_sample_views, f'Not enough views to sample {n_sample_views} unique images.'

        sampled_views = np.random.choice(n_views, size=n_sample_views, replace=False)

        cond_images = []
        for v in sampled_views:
            frame_info = metadata['frames'][v]
            image_path = os.path.join(image_root, frame_info['file_path'])
            cond_images.append(_load_condition_image(image_path, self.image_size))

        pack['cond'] = [torch.stack(cond_images, dim=0)]  # (V,3,H,W)
        return pack
