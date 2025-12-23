from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import trimesh
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
import o_voxel
import cumesh
import nvdiffrast.torch as dr
import cv2
import flex_gemm


class Trellis2TexturingPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'shape_slat_encoder',
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024'
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        tex_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
    ):
        if models is None:
            return
        super().__init__(models)
        self.tex_slat_sampler = tex_slat_sampler
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2TexturingPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])

        pipeline.low_vram = args.get('low_vram', True)
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'
        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Preprocess the input mesh.
        """
        vertices = mesh.vertices
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        tmp = vertices[:, 1].copy()
        vertices[:, 1] = -vertices[:, 2]
        vertices[:, 2] = tmp
        assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    def encode_shape_slat(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 1024,
    ) -> SparseTensor:
        """
        Encode the meshes to structured latent.

        Args:
            mesh (trimesh.Trimesh): The mesh to encode.
            resolution (int): The resolution of mesh
        
        Returns:
            SparseTensor: The encoded structured latent.
        """
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
            
        vertices = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
        ).to(self.device)
        intersected = vertices.replace(intersected).to(self.device)
            
        if self.low_vram:
            self.models['shape_slat_encoder'].to(self.device)
        shape_slat = self.models['shape_slat_encoder'](vertices, intersected)
        if self.low_vram:
            self.models['shape_slat_encoder'].cpu()
        return shape_slat

    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    def postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        pbr_voxel: SparseTensor,
        resolution: int = 1024,
        texture_size: int = 1024,
    ) -> trimesh.Trimesh:
        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals
        vertices_torch = torch.from_numpy(vertices).float().cuda()
        faces_torch = torch.from_numpy(faces).int().cuda()
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs_torch = torch.from_numpy(uvs).float().cuda()
        else:
            _cumesh = cumesh.CuMesh()
            _cumesh.init(vertices_torch, faces_torch)
            vertices_torch, faces_torch, uvs_torch, vmap = _cumesh.uv_unwrap(return_vmaps=True)
            vertices_torch = vertices_torch.cuda()
            faces_torch = faces_torch.cuda()
            uvs_torch = uvs_torch.cuda()
            vertices = vertices_torch.cpu().numpy()
            faces = faces_torch.cpu().numpy()
            uvs = uvs_torch.cpu().numpy()
            normals = normals[vmap.cpu().numpy()]
                
        # rasterize
        ctx = dr.RasterizeCudaContext()
        uvs_torch = torch.cat([uvs_torch * 2 - 1, torch.zeros_like(uvs_torch[:, :1]), torch.ones_like(uvs_torch[:, :1])], dim=-1).unsqueeze(0)
        rast, _ = dr.rasterize(
            ctx, uvs_torch, faces_torch,
            resolution=[texture_size, texture_size],
        )
        mask = rast[0, ..., 3] > 0
        pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch)[0][0]
        
        attrs = torch.zeros(texture_size, texture_size, pbr_voxel.shape[1], device=self.device)
        attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((pos[mask] + 0.5) * resolution).reshape(1, -1, 3),
            mode='trilinear',
        )
        
        # construct mesh
        mask = mask.cpu().numpy()
        base_color = np.clip(attrs[..., self.pbr_attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., self.pbr_attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., self.pbr_attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., self.pbr_attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        # extend
        mask = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask, 1, cv2.INPAINT_TELEA)[..., None]
        
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            doubleSided=True,
        )

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
        normals[:, 1], normals[:, 2] = normals[:, 2], -normals[:, 1]
        uvs[:, 1] = 1 - uvs[:, 1] # Flip UV V-coordinate
        
        textured_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
        )
        
        return textured_mesh
        
    
    @torch.no_grad()
    def run(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        resolution: int = 1024,
        texture_size: int = 2048,
    ) -> trimesh.Trimesh:
        """
        Run the pipeline.

        Args:
            mesh (trimesh.Trimesh): The mesh to texture.
            image (Image.Image): The image prompt.
            seed (int): The random seed.
            tex_slat_sampler_params (dict): Additional parameters for the texture latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        mesh = self.preprocess_mesh(mesh)
        torch.manual_seed(seed)
        cond = self.get_cond([image], 512) if resolution == 512 else self.get_cond([image], 1024)
        shape_slat = self.encode_shape_slat(mesh, resolution)
        tex_model = self.models['tex_slat_flow_model_512'] if resolution == 512 else self.models['tex_slat_flow_model_1024']
        tex_slat = self.sample_tex_slat(
            cond, tex_model,
            shape_slat, tex_slat_sampler_params
        )
        pbr_voxel = self.decode_tex_slat(tex_slat)
        out_mesh = self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size)
        return out_mesh
