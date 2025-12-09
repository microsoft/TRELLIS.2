from typing import *
import torch
from easydict import EasyDict as edict
from ..representations.mesh import Mesh, MeshWithVoxel, MeshWithPbrMaterial, TextureFilterMode, AlphaMode, TextureWrapMode
import torch.nn.functional as F


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
    

class MeshRenderer:
    """
    Renderer for the Mesh representation.

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
            "chunk_size": None,
            "antialias": True,
            "clamp_barycentric_coords": False,
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device=device
        
    def render(
            self,
            mesh : Mesh,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types = ["mask", "normal", "depth"],
            transformation : Optional[torch.Tensor] = None
        ) -> edict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            return_types (list): list of return types, can be "attr", "mask", "depth", "coord", "normal"

        Returns:
            edict based on return_types containing:
                attr (torch.Tensor): [C, H, W] rendered attr image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        if 'dr' not in globals():
            import nvdiffrast.torch as dr
            
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        chunk_size = self.rendering_options["chunk_size"]
        antialias = self.rendering_options["antialias"]
        clamp_barycentric_coords = self.rendering_options["clamp_barycentric_coords"]
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            ret_dict = edict()
            for type in return_types:
                if type == "mask" :
                    ret_dict[type] = torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device)
                elif type == "depth":
                    ret_dict[type] = torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device)
                elif type == "normal":
                    ret_dict[type] = torch.full((3, resolution, resolution), 0.5, dtype=torch.float32, device=self.device)
                elif type == "coord":
                    ret_dict[type] = torch.zeros((3, resolution, resolution), dtype=torch.float32, device=self.device)
                elif type == "attr":
                    if isinstance(mesh, MeshWithVoxel):
                        ret_dict[type] = torch.zeros((mesh.attrs.shape[-1], resolution, resolution), dtype=torch.float32, device=self.device)
                    else:
                        ret_dict[type] = torch.zeros((mesh.vertex_attrs.shape[-1], resolution, resolution), dtype=torch.float32, device=self.device)
            return ret_dict
        
        perspective = intrinsics_to_projection(intrinsics, near, far)
        
        full_proj = (perspective @ extrinsics).unsqueeze(0)
        extrinsics = extrinsics.unsqueeze(0)
        
        vertices = mesh.vertices.unsqueeze(0)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        if transformation is not None:
            vertices_homo = torch.bmm(vertices_homo, transformation.unsqueeze(0).transpose(-1, -2))
            vertices = vertices_homo[..., :3].contiguous()
        vertices_camera = torch.bmm(vertices_homo, extrinsics.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces = mesh.faces
        
        if 'normal' in return_types:
            v0 = vertices_camera[0, mesh.faces[:, 0], :3]
            v1 = vertices_camera[0, mesh.faces[:, 1], :3]
            v2 = vertices_camera[0, mesh.faces[:, 2], :3]
            e0 = v1 - v0
            e1 = v2 - v0
            face_normal = torch.cross(e0, e1, dim=1)
            face_normal = F.normalize(face_normal, dim=1)
            face_normal = torch.where(torch.sum(face_normal * v0, dim=1, keepdim=True) > 0, face_normal, -face_normal)
        
        out_dict = edict()
        if chunk_size is None:
            rast, rast_db = dr.rasterize(
                self.glctx, vertices_clip, faces, (resolution * ssaa, resolution * ssaa)
            )
            if clamp_barycentric_coords:
                rast[..., :2] = torch.clamp(rast[..., :2], 0, 1)
                rast[..., :2] /= torch.where(rast[..., :2].sum(dim=-1, keepdim=True) > 1, rast[..., :2].sum(dim=-1, keepdim=True), torch.ones_like(rast[..., :2]))
            for type in return_types:
                img = None
                if type == "mask" :
                    img = (rast[..., -1:] > 0).float()
                    if antialias: img = dr.antialias(img, rast, vertices_clip, faces)
                elif type == "depth":
                    img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces)[0]
                    if antialias: img = dr.antialias(img, rast, vertices_clip, faces)
                elif type == "normal" :
                    img = dr.interpolate(face_normal.unsqueeze(0), rast, torch.arange(face_normal.shape[0], dtype=torch.int, device=self.device).unsqueeze(1).repeat(1, 3).contiguous())[0]
                    if antialias: img = dr.antialias(img, rast, vertices_clip, faces)
                    img = (img + 1) / 2
                elif type == "coord":
                    img = dr.interpolate(vertices, rast, faces)[0]
                    if antialias: img = dr.antialias(img, rast, vertices_clip, faces)
                elif type == "attr":
                    if isinstance(mesh, MeshWithVoxel):
                        if 'grid_sample_3d' not in globals():
                            from flex_gemm.ops.grid_sample import grid_sample_3d
                        mask = rast[..., -1:] > 0
                        xyz = dr.interpolate(vertices, rast, faces)[0]
                        xyz = ((xyz - mesh.origin) / mesh.voxel_size).reshape(1, -1, 3)
                        img = grid_sample_3d(
                            mesh.attrs,
                            torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1),
                            mesh.voxel_shape,
                            xyz,
                            mode='trilinear'
                        )
                        img = img.reshape(1, resolution * ssaa, resolution * ssaa, mesh.attrs.shape[-1]) * mask
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
                        imgs = {
                            'base_color': torch.zeros((resolution * ssaa, resolution * ssaa, 3), dtype=torch.float32, device=self.device),
                            'metallic': torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device),
                            'roughness': torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device),
                            'alpha': torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device)
                        }
                        for id, mat in enumerate(mesh.materials):
                            mat_mask = (mid == id).float() * mask.float()
                            mat_texc = texc * mat_mask
                            mat_texd = texd * mat_mask

                            if mat.base_color_texture is not None:
                                base_color = dr.texture(
                                    mat.base_color_texture.image.unsqueeze(0),
                                    mat_texc,
                                    mat_texd,
                                    filter_mode='linear-mipmap-linear' if mat.base_color_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                    boundary_mode='clamp' if mat.base_color_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                )[0]
                                imgs['base_color'] += base_color * mat.base_color_factor * mat_mask
                            else:
                                imgs['base_color'] += mat.base_color_factor * mat_mask
                                
                            if mat.metallic_texture is not None:
                                metallic = dr.texture(
                                    mat.metallic_texture.image.unsqueeze(0),
                                    mat_texc,
                                    mat_texd,
                                    filter_mode='linear-mipmap-linear' if mat.metallic_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                    boundary_mode='clamp' if mat.metallic_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                )[0]
                                imgs['metallic'] += metallic * mat.metallic_factor * mat_mask
                            else:
                                imgs['metallic'] += mat.metallic_factor * mat_mask

                            if mat.roughness_texture is not None:
                                roughness = dr.texture(
                                    mat.roughness_texture.image.unsqueeze(0),
                                    mat_texc,
                                    mat_texd,
                                    filter_mode='linear-mipmap-linear' if mat.roughness_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                    boundary_mode='clamp' if mat.roughness_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                )[0]
                                imgs['roughness'] += roughness * mat.roughness_factor * mat_mask
                            else:
                                imgs['roughness'] += mat.roughness_factor * mat_mask

                            if mat.alpha_mode == AlphaMode.OPAQUE:
                                imgs['alpha'] += 1.0 * mat_mask
                            else:
                                if mat.alpha_texture is not None:
                                    alpha = dr.texture(
                                        mat.alpha_texture.image.unsqueeze(0),
                                        mat_texc,
                                        mat_texd,
                                        filter_mode='linear-mipmap-linear' if mat.alpha_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                        boundary_mode='clamp' if mat.alpha_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                    )[0]
                                    if mat.alpha_mode == AlphaMode.MASK:
                                        imgs['alpha'] += (alpha * mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                                    elif mat.alpha_mode == AlphaMode.BLEND:
                                        imgs['alpha'] += alpha * mat.alpha_factor * mat_mask
                                else:
                                    if mat.alpha_mode == AlphaMode.MASK:
                                        imgs['alpha'] += (mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                                    elif mat.alpha_mode == AlphaMode.BLEND:
                                        imgs['alpha'] += mat.alpha_factor * mat_mask
                    
                        img = torch.cat([imgs[name] for name in imgs.keys()], dim=-1).unsqueeze(0)
                    else:
                        img = dr.interpolate(mesh.vertex_attrs.unsqueeze(0), rast, faces)[0]
                        if antialias: img = dr.antialias(img, rast, vertices_clip, faces)
                        
                out_dict[type] = img
        else:
            z_buffer = torch.full((1, resolution * ssaa, resolution * ssaa), torch.inf, device=self.device, dtype=torch.float32)
            for i in range(0, faces.shape[0], chunk_size):
                faces_chunk = faces[i:i+chunk_size]
                rast, rast_db = dr.rasterize(
                    self.glctx, vertices_clip, faces_chunk, (resolution * ssaa, resolution * ssaa)
                )
                z_filter = torch.logical_and(
                    rast[..., 3] != 0,
                    rast[..., 2] < z_buffer
                )
                z_buffer[z_filter] = rast[z_filter][..., 2]
            
                for type in return_types:
                    img = None
                    if type == "mask" :
                        img = (rast[..., -1:] > 0).float()
                    elif type == "depth":
                        img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_chunk)[0]
                    elif type == "normal" :
                        face_normal_chunk = face_normal[i:i+chunk_size]
                        img = dr.interpolate(face_normal_chunk.unsqueeze(0), rast, torch.arange(face_normal_chunk.shape[0], dtype=torch.int, device=self.device).unsqueeze(1).repeat(1, 3).contiguous())[0]
                        img = (img + 1) / 2
                    elif type == "coord":
                        img = dr.interpolate(vertices, rast, faces_chunk)[0]
                    elif type == "attr":
                        if isinstance(mesh, MeshWithVoxel):
                            if 'grid_sample_3d' not in globals():
                                from flex_gemm.ops.grid_sample import grid_sample_3d
                            mask = rast[..., -1:] > 0
                            xyz = dr.interpolate(vertices, rast, faces_chunk)[0]
                            xyz = ((xyz - mesh.origin) / mesh.voxel_size).reshape(1, -1, 3)
                            img = grid_sample_3d(
                                mesh.attrs,
                                torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1),
                                mesh.voxel_shape,
                                xyz,
                                mode='trilinear'
                            )
                            img = img.reshape(1, resolution * ssaa, resolution * ssaa, mesh.attrs.shape[-1]) * mask
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
                            imgs = {
                                'base_color': torch.zeros((resolution * ssaa, resolution * ssaa, 3), dtype=torch.float32, device=self.device),
                                'metallic': torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device),
                                'roughness': torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device),
                                'alpha': torch.zeros((resolution * ssaa, resolution * ssaa, 1), dtype=torch.float32, device=self.device)
                            }
                            for id, mat in enumerate(mesh.materials):
                                mat_mask = (mid == id).float() * mask.float()
                                mat_texc = texc * mat_mask
                                mat_texd = texd * mat_mask

                                if mat.base_color_texture is not None:
                                    base_color = dr.texture(
                                        mat.base_color_texture.image.unsqueeze(0),
                                        mat_texc,
                                        mat_texd,
                                        filter_mode='linear-mipmap-linear' if mat.base_color_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                        boundary_mode='clamp' if mat.base_color_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                    )[0]
                                    imgs['base_color'] += base_color * mat.base_color_factor * mat_mask
                                else:
                                    imgs['base_color'] += mat.base_color_factor * mat_mask
                                    
                                if mat.metallic_texture is not None:
                                    metallic = dr.texture(
                                        mat.metallic_texture.image.unsqueeze(0),
                                        mat_texc,
                                        mat_texd,
                                        filter_mode='linear-mipmap-linear' if mat.metallic_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                        boundary_mode='clamp' if mat.metallic_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                    )[0]
                                    imgs['metallic'] += metallic * mat.metallic_factor * mat_mask
                                else:
                                    imgs['metallic'] += mat.metallic_factor * mat_mask

                                if mat.roughness_texture is not None:
                                    roughness = dr.texture(
                                        mat.roughness_texture.image.unsqueeze(0),
                                        mat_texc,
                                        mat_texd,
                                        filter_mode='linear-mipmap-linear' if mat.roughness_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                        boundary_mode='clamp' if mat.roughness_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                    )[0]
                                    imgs['roughness'] += roughness * mat.roughness_factor * mat_mask
                                else:
                                    imgs['roughness'] += mat.roughness_factor * mat_mask

                                if mat.alpha_mode == AlphaMode.OPAQUE:
                                    imgs['alpha'] += 1.0 * mat_mask
                                else:
                                    if mat.alpha_texture is not None:
                                        alpha = dr.texture(
                                            mat.alpha_texture.image.unsqueeze(0),
                                            mat_texc,
                                            mat_texd,
                                            filter_mode='linear-mipmap-linear' if mat.alpha_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                                            boundary_mode='clamp' if mat.alpha_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                                        )[0]
                                        if mat.alpha_mode == AlphaMode.MASK:
                                            imgs['alpha'] += (alpha * mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                                        elif mat.alpha_mode == AlphaMode.BLEND:
                                            imgs['alpha'] += alpha * mat.alpha_factor * mat_mask
                                    else:
                                        if mat.alpha_mode == AlphaMode.MASK:
                                            imgs['alpha'] += (mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                                        elif mat.alpha_mode == AlphaMode.BLEND:
                                            imgs['alpha'] += mat.alpha_factor * mat_mask
                        
                            img = torch.cat([imgs[name] for name in imgs.keys()], dim=-1).unsqueeze(0)
                        else:
                            img = dr.interpolate(mesh.vertex_attrs.unsqueeze(0), rast, faces_chunk)[0]
                            
                    if type not in out_dict:
                        out_dict[type] = img
                    else:
                        out_dict[type][z_filter] = img[z_filter]

        for type in return_types:
            img = out_dict[type]
            if ssaa > 1:
                img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                img = img.squeeze()
            else:
                img = img.permute(0, 3, 1, 2).squeeze()
            out_dict[type] = img

        if isinstance(mesh, (MeshWithVoxel, MeshWithPbrMaterial)) and 'attr' in return_types:
            for k, s in mesh.layout.items():
                out_dict[k] = out_dict['attr'][s]
            del out_dict['attr']
        
        return out_dict
