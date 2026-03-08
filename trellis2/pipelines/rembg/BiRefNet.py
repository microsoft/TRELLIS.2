from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # Ensure compatibility with transformers 5.x before loading:
        # 1. Patch missing all_tied_weights_keys attribute
        # 2. Skip meta device init (BiRefNet's __init__ calls .item() which is incompatible)
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        try:
            birefnet_cls = get_class_from_dynamic_module(
                "birefnet.BiRefNet", model_name
            )
            if not hasattr(birefnet_cls, 'all_tied_weights_keys'):
                birefnet_cls.all_tied_weights_keys = {}

            @classmethod
            def _no_meta_init_context(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels):
                from transformers.modeling_utils import local_torch_dtype, init, apply_patches
                return [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights(), apply_patches()]
            birefnet_cls.get_init_context = _no_meta_init_context
        except Exception:
            pass

        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    