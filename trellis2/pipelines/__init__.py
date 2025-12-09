import importlib

__attributes = {
    "Trellis2ImageTo3DPipeline": "trellis2_image_to_3d",
    "Trellis2ImageTo3DCascadePipeline": "trellis2_image_to_3d_cascade",
    "Trellis2ImageToTexturePipeline": "trellis2_image_to_tex",
}

__submodules = ['samplers', 'rembg']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)


# For PyLance
if __name__ == '__main__':
    from . import samplers, rembg
    from .trellis_image_to_3d import TrellisImageTo3DPipeline
    from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from .trellis2_image_to_3d_cascade import Trellis2ImageTo3DCascadePipeline
    from .trellis2_image_to_tex import Trellis2ImageToTexturePipeline
