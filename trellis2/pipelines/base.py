from typing import *
import torch
import torch.nn as nn
from .. import models
from ..model_revisions import revision_for_repo


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            revision = revision_for_repo(path, revision)
            config_file = hf_hub_download(
                path,
                config_file,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            is_external_model = v.count('/') >= 2
            model_path = v if is_external_model else f"{path}/{v}"
            _models[k] = models.from_pretrained(
                model_path,
                revision=None if is_external_model else revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        new_pipeline._pretrained_hub_kwargs = {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
