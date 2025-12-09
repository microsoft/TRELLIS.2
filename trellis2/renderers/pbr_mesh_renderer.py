from typing import *
import torch
from easydict import EasyDict as edict
import numpy as np
import utils3d
from ..representations.mesh import Mesh, MeshWithVoxel, MeshWithPbrMaterial, TextureFilterMode, AlphaMode, TextureWrapMode
import torch.nn.functional as F


def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -x, -y
    elif s == 1: rx, ry, rz = -torch.ones_like(x), x, -y
    elif s == 2: rx, ry, rz = x, y, torch.ones_like(x)
    elif s == 3: rx, ry, rz = x, -y, -torch.ones_like(x)
    elif s == 4: rx, ry, rz = x, torch.ones_like(x), -y
    elif s == 5: rx, ry, rz = -x, -torch.ones_like(x), -y
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map, res):
    if 'dr' not in globals():
        import nvdiffrast.torch as dr
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = F.normalize(cube_to_dir(s, gx, gy), dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap


class EnvMap:
    def __init__(self, image: torch.Tensor):
        self.image = image
        
    @property
    def _backend(self):
        if not hasattr(self, '_nvdiffrec_envlight'):
            if 'EnvironmentLight' not in globals():
                from nvdiffrec_render.light import EnvironmentLight
            cubemap = latlong_to_cubemap(self.image, [512, 512])
            self._nvdiffrec_envlight = EnvironmentLight(cubemap)
            self._nvdiffrec_envlight.build_mips()
        return self._nvdiffrec_envlight

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        return self._backend.shade(gb_pos, gb_normal, kd, ks, view_pos, specular)
    
    def sample(self, directions: torch.Tensor):
        if 'dr' not in globals():
            import nvdiffrast.torch as dr
        return dr.texture(
            self._backend.base.unsqueeze(0),
            directions.unsqueeze(0),
            boundary_mode='cube',
        )[0]
            

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def screen_space_ambient_occlusion(
    depth: torch.Tensor,
    normal: torch.Tensor,
    perspective: torch.Tensor,
    radius: float = 0.1,
    bias: float = 1e-6,
    samples: int = 64,
    intensity: float = 1.0,
) -> torch.Tensor:
    """
    Screen space ambient occlusion (SSAO)

    Args:
        depth (torch.Tensor): [H, W, 1] depth image
        normal (torch.Tensor): [H, W, 3] normal image
        perspective (torch.Tensor): [4, 4] camera projection matrix
        radius (float): radius of the SSAO kernel
        bias (float): bias to avoid self-occlusion
        samples (int): number of samples to use for the SSAO kernel
        intensity (float): intensity of the SSAO effect
    Returns:
        (torch.Tensor): [H, W, 1] SSAO image
    """
    device = depth.device
    H, W, _ = depth.shape
    
    fx = perspective[0, 0]
    fy = perspective[1, 1]
    cx = perspective[0, 2]
    cy = perspective[1, 2]
    
    y_grid, x_grid = torch.meshgrid(
        (torch.arange(H, device=device) + 0.5) / H * 2 - 1,
        (torch.arange(W, device=device) + 0.5) / W * 2 - 1,
        indexing='ij'
    )
    x_view = (x_grid.float() - cx) * depth[..., 0] / fx
    y_view = (y_grid.float() - cy) * depth[..., 0] / fy
    view_pos = torch.stack([x_view, y_view, depth[..., 0]], dim=-1) # [H, W, 3]
    
    depth_feat = depth.permute(2, 0, 1).unsqueeze(0)
    occlusion = torch.zeros((H, W), device=device)
    
    # start sampling
    for _ in range(samples):
        # sample normal distribution, if inside, flip the sign
        rnd_vec = torch.randn(H, W, 3, device=device)
        rnd_vec = F.normalize(rnd_vec, p=2, dim=-1)
        dot_val = torch.sum(rnd_vec * normal, dim=-1, keepdim=True)
        sample_dir = torch.sign(dot_val) * rnd_vec
        scale = torch.rand(H, W, 1, device=device)
        scale = scale * scale
        sample_pos = view_pos + sample_dir * radius * scale
        sample_z = sample_pos[..., 2]
        
        # project to screen space
        z_safe = torch.clamp(sample_pos[..., 2], min=1e-5)
        proj_u = (sample_pos[..., 0] * fx / z_safe) + cx
        proj_v = (sample_pos[..., 1] * fy / z_safe) + cy
        grid = torch.stack([proj_u, proj_v], dim=-1).unsqueeze(0)
        geo_z = F.grid_sample(depth_feat, grid, mode='nearest', padding_mode='border').squeeze()
        range_check = torch.abs(geo_z - sample_z) < radius
        is_occluded = (geo_z <= sample_z - bias) & range_check
        occlusion += is_occluded.float()
        
    f_occ = occlusion / samples * intensity
    f_occ = torch.clamp(f_occ, 0.0, 1.0)
    
    return f_occ.unsqueeze(-1)


def aces_tonemapping(x: torch.Tensor) -> torch.Tensor:
    """
    Applies ACES tone mapping curve to an HDR image tensor.
    Input:  x - HDR tensor, shape (..., 3), range [0, +inf)
    Output: LDR tensor, same shape, range [0, 1]
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    # Apply the ACES fitted curve
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    
    # Clamp to [0, 1] for display or saving
    return torch.clamp(mapped, 0.0, 1.0)


def gamma_correction(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """
    Applies gamma correction to an HDR image tensor.
    """
    return torch.clamp(x ** (1.0 / gamma), 0.0, 1.0)
    

class PbrMeshRenderer:
    """
    Renderer for the PBR mesh.

    Args:
        rendering_options (dict): Rendering options.
        """
    def __init__(self, rendering_options={}, device='cuda'):
        if 'dr' not in globals():
            import nvdiffrast.torch as dr
        
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "peel_layers": 8,
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device=device
        
    def render(
            self,
            mesh : Mesh,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            envmap : Union[EnvMap, Dict[str, EnvMap]],
            use_envmap_bg : bool = False,
            transformation : Optional[torch.Tensor] = None
        ) -> edict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            envmap (Union[EnvMap, Dict[str, EnvMap]]): environment map or a dictionary of environment maps
            use_envmap_bg (bool): whether to use envmap as background
            transformation (torch.Tensor): (4, 4) transformation matrix

        Returns:
            edict based on return_types containing:
                shaded (torch.Tensor): [3, H, W] shaded color image
                normal (torch.Tensor): [3, H, W] normal image
                base_color (torch.Tensor): [3, H, W] base color image
                metallic (torch.Tensor): [H, W] metallic image
                roughness (torch.Tensor): [H, W] roughness image
        """
        if 'dr' not in globals():
            import nvdiffrast.torch as dr
            
        if not isinstance(envmap, dict):
            envmap = {'' : envmap}
        num_envmaps = len(envmap)
            
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return edict(
                shaded=torch.full((4, resolution, resolution), 0.5, dtype=torch.float32, device=self.device),
            )
            
        rays_o, rays_d = utils3d.torch.get_image_rays(
            extrinsics, intrinsics, resolution * ssaa, resolution * ssaa
        )
        
        perspective = intrinsics_to_projection(intrinsics, near, far)
        
        full_proj = (perspective @ extrinsics).unsqueeze(0)
        extrinsics = extrinsics.unsqueeze(0)
        
        vertices = mesh.vertices.unsqueeze(0)
        vertices_orig = vertices.clone()
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        if transformation is not None:
            vertices_homo = torch.bmm(vertices_homo, transformation.unsqueeze(0).transpose(-1, -2))
            vertices = vertices_homo[..., :3].contiguous()
        vertices_camera = torch.bmm(vertices_homo, extrinsics.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces = mesh.faces
        
        v0 = vertices[0, mesh.faces[:, 0], :3]
        v1 = vertices[0, mesh.faces[:, 1], :3]
        v2 = vertices[0, mesh.faces[:, 2], :3]
        e0 = v1 - v0
        e1 = v2 - v0
        face_normal = torch.cross(e0, e1, dim=1)
        face_normal = F.normalize(face_normal, dim=1)
        
        out_dict = edict()
        shaded = torch.zeros((num_envmaps, resolution * ssaa, resolution * ssaa, 3), dtype=torch.float32, device=self.device)
        depth = torch.full((resolution * ssaa, resolution * ssaa, 1), 1e10, dtype=torch.float32, device=self.device)
        normal = torch.zeros((resolution * ssaa, resolution * ssaa, 3), dtype=torch.float32, device=self.device)
        max_w = torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device)
        alpha = torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device)
        with dr.DepthPeeler(self.glctx, vertices_clip, faces, (resolution * ssaa, resolution * ssaa)) as peeler:
            for _ in range(self.rendering_options["peel_layers"]):
                rast, rast_db = peeler.rasterize_next_layer()
                
                # Pos
                pos = dr.interpolate(vertices, rast, faces)[0][0]
                
                # Depth
                gb_depth = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces)[0][0]
                        
                # Normal
                gb_normal = dr.interpolate(face_normal.unsqueeze(0), rast, torch.arange(face_normal.shape[0], dtype=torch.int, device=self.device).unsqueeze(1).repeat(1, 3).contiguous())[0][0]
                gb_normal = torch.where(
                    torch.sum(gb_normal * (pos - rays_o), dim=-1, keepdim=True) > 0,
                    -gb_normal,
                    gb_normal
                )
                gb_cam_normal = (extrinsics[..., :3, :3].reshape(1, 1, 3, 3) @ gb_normal.unsqueeze(-1)).squeeze(-1)
                if _ == 0:
                    out_dict.normal = -gb_cam_normal * 0.5 + 0.5
                    mask = (rast[0, ..., -1:] > 0).float()
                    out_dict.mask = mask
                
                # PBR attributes
                if isinstance(mesh, MeshWithVoxel):
                    if 'grid_sample_3d' not in globals():
                        from flex_gemm.ops.grid_sample import grid_sample_3d
                    mask = rast[..., -1:] > 0
                    xyz = dr.interpolate(vertices_orig, rast, faces)[0]
                    xyz = ((xyz - mesh.origin) / mesh.voxel_size).reshape(1, -1, 3)
                    img = grid_sample_3d(
                        mesh.attrs,
                        torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1),
                        mesh.voxel_shape,
                        xyz,
                        mode='trilinear'
                    )
                    img = img.reshape(1, resolution * ssaa, resolution * ssaa, mesh.attrs.shape[-1]) * mask
                    gb_basecolor = img[0, ..., mesh.layout['base_color']]
                    gb_metallic = img[0, ..., mesh.layout['metallic']]
                    gb_roughness = img[0, ..., mesh.layout['roughness']]
                    gb_alpha = img[0, ..., mesh.layout['alpha']]
                elif isinstance(mesh, MeshWithPbrMaterial):
                    tri_id = rast[0, :, :, -1:]
                    mask = tri_id > 0
                    uv_coords = mesh.uv_coords.reshape(1, -1, 2)
                    texc, texd = dr.interpolate(
                        uv_coords,
                        rast,
                        torch.arange(mesh.uv_coords.shape[0] * 3, dtype=torch.int, device=self.device).reshape(-1, 3),
                        rast_db=rast_db,
                        diff_attrs='all'
                    )
                    # Fix problematic texture coordinates
                    texc = torch.nan_to_num(texc, nan=0.0, posinf=1e3, neginf=-1e3)
                    texc = torch.clamp(texc, min=-1e3, max=1e3)
                    texd = torch.nan_to_num(texd, nan=0.0, posinf=1e3, neginf=-1e3)
                    texd = torch.clamp(texd, min=-1e3, max=1e3)
                    mid = mesh.material_ids[(tri_id - 1).long()]
                    gb_basecolor = torch.zeros((resolution * ssaa, resolution * ssaa, 3), dtype=torch.float32, device=self.device)
                    gb_metallic = torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device)
                    gb_roughness = torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device)
                    gb_alpha = torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device)
                    for id, mat in enumerate(mesh.materials):
                        mat_mask = (mid == id).float() * mask.float()
                        mat_texc = texc * mat_mask
                        mat_texd = texd * mat_mask

                        if mat.base_color_texture is not None:
                            bc = dr.texture(
                                mat.base_color_texture.image.unsqueeze(0),
                                mat_texc,
                                mat_texd,
                                filter_mode='linear-mipmap-linear' if mat.base_color_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                boundary_mode='clamp' if mat.base_color_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                            )[0]
                            gb_basecolor += bc * mat.base_color_factor * mat_mask
                        else:
                            gb_basecolor += mat.base_color_factor * mat_mask
                            
                        if mat.metallic_texture is not None:
                            m = dr.texture(
                                mat.metallic_texture.image.unsqueeze(0),
                                mat_texc,
                                mat_texd,
                                filter_mode='linear-mipmap-linear' if mat.metallic_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                boundary_mode='clamp' if mat.metallic_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                            )[0]
                            gb_metallic += m * mat.metallic_factor * mat_mask
                        else:
                            gb_metallic += mat.metallic_factor * mat_mask

                        if mat.roughness_texture is not None:
                            r = dr.texture(
                                mat.roughness_texture.image.unsqueeze(0),
                                mat_texc,
                                mat_texd,
                                filter_mode='linear-mipmap-linear' if mat.roughness_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                boundary_mode='clamp' if mat.roughness_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                            )[0]
                            gb_roughness += r * mat.roughness_factor * mat_mask
                        else:
                            gb_roughness += mat.roughness_factor * mat_mask

                        if mat.alpha_mode == AlphaMode.OPAQUE:
                            gb_alpha += 1.0 * mat_mask
                        else:
                            if mat.alpha_texture is not None:
                                a = dr.texture(
                                    mat.alpha_texture.image.unsqueeze(0),
                                    mat_texc,
                                    mat_texd,
                                    filter_mode='linear-mipmap-linear' if mat.alpha_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                    boundary_mode='clamp' if mat.alpha_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                )[0]
                                if mat.alpha_mode == AlphaMode.MASK:
                                    gb_alpha += (a * mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                                elif mat.alpha_mode == AlphaMode.BLEND:
                                    gb_alpha += a * mat.alpha_factor * mat_mask
                            else:
                                if mat.alpha_mode == AlphaMode.MASK:
                                    gb_alpha += (mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                                elif mat.alpha_mode == AlphaMode.BLEND:
                                    gb_alpha += mat.alpha_factor * mat_mask
                if _ == 0:
                    out_dict.base_color = gb_basecolor
                    out_dict.metallic = gb_metallic
                    out_dict.roughness = gb_roughness
                    out_dict.alpha = gb_alpha
                    
                # Shading
                gb_basecolor = torch.clamp(gb_basecolor, 0.0, 1.0) ** 2.2
                gb_metallic = torch.clamp(gb_metallic, 0.0, 1.0)
                gb_roughness = torch.clamp(gb_roughness, 0.0, 1.0)
                gb_alpha = torch.clamp(gb_alpha, 0.0, 1.0)
                gb_orm = torch.cat([
                    torch.zeros_like(gb_metallic),
                    gb_roughness,
                    gb_metallic,
                ], dim=-1)
                gb_shaded = torch.stack([
                    e.shade(
                        pos.unsqueeze(0),
                        gb_normal.unsqueeze(0),
                        gb_basecolor.unsqueeze(0),
                        gb_orm.unsqueeze(0),
                        rays_o,
                        specular=True,
                    )[0]
                    for e in envmap.values()
                ], dim=0)
                
                # Compositing
                w = (1 - alpha) * gb_alpha
                depth = torch.where(w > max_w, gb_depth, depth)
                normal = torch.where(w > max_w, gb_cam_normal, normal)
                max_w = torch.maximum(max_w, w)
                shaded += w * gb_shaded
                alpha += w
        
        # Ambient occulusion
        f_occ = screen_space_ambient_occlusion(
            depth, normal, perspective, intensity=1.5
        )
        shaded *= (1 - f_occ)
        out_dict.clay = (1 - f_occ)
                
        # Background
        if use_envmap_bg:
            bg = torch.stack([e.sample(rays_d) for e in envmap.values()], dim=0)
            shaded += (1 - alpha) * bg
        
        for i, k in enumerate(envmap.keys()):
            shaded_key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[shaded_key] = shaded[i]
    
        # SSAA
        for k in out_dict.keys():
            if ssaa > 1:
                out_dict[k] = F.interpolate(out_dict[k].unsqueeze(0).permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
            else:
                out_dict[k] = out_dict[k].permute(2, 0, 1)
            out_dict[k] = out_dict[k].squeeze()
                
        # Post processing
        for k in envmap.keys():
            shaded_key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[shaded_key] = aces_tonemapping(out_dict[shaded_key])
            out_dict[shaded_key] = gamma_correction(out_dict[shaded_key])
            
        return out_dict
