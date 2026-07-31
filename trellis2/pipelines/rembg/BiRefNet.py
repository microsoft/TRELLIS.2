from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
from ...model_revisions import RMBG_REPO, RMBG_REVISION


class BiRefNet:
    def __init__(
        self,
        model_name: str = RMBG_REPO,
        revision: Optional[str] = RMBG_REVISION,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._device = 'cpu'

    def to(self, device):
        self._device = device
        self.model.to(device)

    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')

    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
