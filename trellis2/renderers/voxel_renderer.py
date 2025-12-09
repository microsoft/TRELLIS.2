import torch
from easydict import EasyDict as edict
from ..representations import Voxel
from easydict import EasyDict as edict


class VoxelRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.rendering_options = edict({
            "resolution": None,
            "near": 0.1,
            "far": 10.0,
            "ssaa": 1,
        })
        self.rendering_options.update(rendering_options)
    
    def render(
            self,
            voxel: Voxel,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
        ) -> edict:
        """
        Render the gausssian.

        Args:
            voxel (Voxel): Voxel representation.
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
                depth (torch.Tensor): (H, W) rendered depth
                alpha (torch.Tensor): (H, W) rendered alpha
                ...
        """ 
        # lazy import
        if 'o_voxel' not in globals():
            import o_voxel
        renderer = o_voxel.rasterize.VoxelRenderer(self.rendering_options)
        positions = voxel.position
        attrs = voxel.attrs if colors_overwrite is None else colors_overwrite
        voxel_size = voxel.voxel_size
        
        # Render
        render_ret = renderer.render(positions, attrs, voxel_size, extrinsics, intrinsics)
        
        ret = {
            'depth': render_ret['depth'],
            'alpha': render_ret['alpha'],
        }
        if colors_overwrite is not None:
            ret['color'] = render_ret['attr']
        else:
            for k, s in voxel.layout.items():
                ret[k] = render_ret['attr'][s]
        
        return ret
