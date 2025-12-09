from typing import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from transformers import AutoTokenizer, CLIPTextModel

from ....utils import dist_utils


class TextConditionedMixin:
    """
    Mixin for text-conditioned models.
    
    Args:
        text_cond_model: The text conditioning model.
    """
    def __init__(self, *args, text_cond_model: str = 'openai/clip-vit-large-patch14', **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cond_model_name = text_cond_model
        self.text_cond_model = None     # the model is init lazily
        
    def _init_text_cond_model(self):
        """
        Initialize the text conditioning model.
        """
        # load model
        with dist_utils.local_master_first():
            model = CLIPTextModel.from_pretrained(self.text_cond_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.text_cond_model_name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])
        
    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and isinstance(text[0], str), "TextConditionedMixin only supports list of strings as cond"
        if self.text_cond_model is None:
            self._init_text_cond_model()
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_text(cond)
        kwargs['neg_cond'] = self.text_cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_text(cond)
        kwargs['neg_cond'] = self.text_cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond
