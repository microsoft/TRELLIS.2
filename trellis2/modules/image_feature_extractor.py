from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image
from pathlib import Path


# Default location written by download_texture_models.py
_DINOV3_DEFAULT_PTH = (
    Path(__file__).parent.parent.parent
    / "models"
    / "dinov3-vitl16-pretrain-lvd1689m"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)


def _find_dinov3_pth(model_name: str) -> "Path | None":
    """Return a local .pth path for DINOv3, or None to fall back to HF."""
    import os
    env = os.environ.get("DINOV3_PTH_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p
    if model_name.endswith(".pth"):
        p = Path(model_name)
        if p.exists():
            return p
    if _DINOV3_DEFAULT_PTH.exists():
        return _DINOV3_DEFAULT_PTH
    return None


def _load_dinov3_from_pth(pth_path) -> "DINOv3ViTModel":
    """Load DINOv3ViTModel from a Meta-format .pth checkpoint."""
    from transformers import DINOv3ViTConfig
    print(f"  [DINOv3] loading from local .pth: {pth_path}")
    ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=True)

    hidden_size  = ckpt["patch_embed.proj.weight"].shape[0]
    num_blocks   = sum(1 for k in ckpt if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    intermediate = ckpt["blocks.0.mlp.fc1.weight"].shape[0]
    num_register = ckpt["storage_tokens"].shape[1]
    patch_size   = ckpt["patch_embed.proj.weight"].shape[2]
    num_heads    = hidden_size // 64  # standard ViT head_dim=64

    cfg = DINOv3ViTConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_blocks,
        num_attention_heads=num_heads,
        intermediate_size=intermediate,
        patch_size=patch_size,
        num_register_tokens=num_register,
        key_bias=True,
    )

    sd = {}
    sd["embeddings.cls_token"]               = ckpt["cls_token"]
    sd["embeddings.mask_token"]              = ckpt["mask_token"].unsqueeze(1)  # [1, D] -> [1, 1, D]
    sd["embeddings.register_tokens"]         = ckpt["storage_tokens"]
    sd["embeddings.patch_embeddings.weight"] = ckpt["patch_embed.proj.weight"]
    sd["embeddings.patch_embeddings.bias"]   = ckpt["patch_embed.proj.bias"]

    for i in range(num_blocks):
        s, d = f"blocks.{i}", f"layer.{i}"
        for sfx in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            sd[f"{d}.{sfx}"] = ckpt[f"{s}.{sfx}"]

        qkv_w = ckpt[f"{s}.attn.qkv.weight"]
        qkv_b = ckpt[f"{s}.attn.qkv.bias"]
        H = hidden_size
        sd[f"{d}.attention.q_proj.weight"] = qkv_w[:H]
        sd[f"{d}.attention.k_proj.weight"] = qkv_w[H:2*H]
        sd[f"{d}.attention.v_proj.weight"] = qkv_w[2*H:]
        sd[f"{d}.attention.q_proj.bias"]   = qkv_b[:H]
        sd[f"{d}.attention.k_proj.bias"]   = qkv_b[H:2*H]
        sd[f"{d}.attention.v_proj.bias"]   = qkv_b[2*H:]

        sd[f"{d}.attention.o_proj.weight"] = ckpt[f"{s}.attn.proj.weight"]
        sd[f"{d}.attention.o_proj.bias"]   = ckpt[f"{s}.attn.proj.bias"]
        sd[f"{d}.layer_scale1.lambda1"]    = ckpt[f"{s}.ls1.gamma"]
        sd[f"{d}.layer_scale2.lambda1"]    = ckpt[f"{s}.ls2.gamma"]
        sd[f"{d}.mlp.up_proj.weight"]      = ckpt[f"{s}.mlp.fc1.weight"]
        sd[f"{d}.mlp.up_proj.bias"]        = ckpt[f"{s}.mlp.fc1.bias"]
        sd[f"{d}.mlp.down_proj.weight"]    = ckpt[f"{s}.mlp.fc2.weight"]
        sd[f"{d}.mlp.down_proj.bias"]      = ckpt[f"{s}.mlp.fc2.bias"]

    model = DINOv3ViTModel(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [DINOv3] {len(missing)} missing keys (e.g. {missing[0]})")
    model.eval()
    return model


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        pth = _find_dinov3_pth(model_name)
        if pth is not None:
            self.model = _load_dinov3_from_pth(pth)
        else:
            self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.extract_features(image)
        return features
