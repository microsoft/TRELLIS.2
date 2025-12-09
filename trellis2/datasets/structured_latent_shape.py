import os
import json
from typing import *
import numpy as np
import torch
from .. import models
from .components import ImageConditionedMixin
from ..modules.sparse import SparseTensor
from .structured_latent import SLatVisMixin, SLat
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics


class SLatShapeVisMixin(SLatVisMixin):
    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
            cfg['models']['decoder']['args']['resolution'] = self.resolution
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        self.slat_dec = decoder.cuda().eval()

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())
        
        # build camera
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)
        
        # render
        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['normal']
            images.append(image)
        images = torch.stack(images)
        return images
    
    
class SLatShape(SLatShapeVisMixin, SLat):
    """
    structured latent for shape generation
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the shape
        min_aesthetic_score (float): minimum aesthetic score
        max_tokens (int): maximum number of tokens
        latent_key (str): key of the latent to be used
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        super().__init__(
            roots,
            min_aesthetic_score=min_aesthetic_score,
            max_tokens=max_tokens,
            latent_key='shape_latent',
            normalization=normalization,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )
        self.resolution = resolution


class ImageConditionedSLatShape(ImageConditionedMixin, SLatShape):
    """
    Image conditioned structured latent for shape generation
    """
    pass
