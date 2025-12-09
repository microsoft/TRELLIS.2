import importlib

__attributes = {
    'MeshRenderer': 'mesh_renderer',
    'VoxelRenderer': 'voxel_renderer',
    'PbrMeshRenderer': 'pbr_mesh_renderer',
    'EnvMap': 'pbr_mesh_renderer',
}

__submodules = []

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


# For Pylance
if __name__ == '__main__':
    from .mesh_renderer import MeshRenderer
    from .voxel_renderer import VoxelRenderer
    from .pbr_mesh_renderer import PbrMeshRenderer, EnvMap
    