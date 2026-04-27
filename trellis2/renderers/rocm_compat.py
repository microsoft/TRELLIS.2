"""
ROCm compatibility layer for TRELLIS.2.
Import this before using any renderer to automatically patch nvdiffrast imports.

Usage:
    import trellis2.renderers.rocm_compat  # auto-patches nvdiffrast
    # Then use renderers normally
"""
import sys
import importlib

def patch_nvdiffrast():
    """Replace nvdiffrast.torch with our ROCm adapter if nvdiffrast is not available."""
    try:
        import nvdiffrast.torch
        print("[rocm_compat] nvdiffrast available, using native implementation")
        return False
    except ImportError:
        pass

    # Import our adapter
    from trellis2.renderers.nvdiffrast_rocm_adapter import (
        RasterizeCudaContext, rasterize, interpolate, texture, antialias, DepthPeeler
    )

    # Create a fake nvdiffrast.torch module
    import types
    fake_dr = types.ModuleType('nvdiffrast.torch')
    fake_dr.RasterizeCudaContext = RasterizeCudaContext
    fake_dr.rasterize = rasterize
    fake_dr.interpolate = interpolate
    fake_dr.texture = texture
    fake_dr.antialias = antialias
    fake_dr.DepthPeeler = DepthPeeler

    # Also create fake nvdiffrast parent
    fake_nvdiffrast = types.ModuleType('nvdiffrast')
    fake_nvdiffrast.torch = fake_dr

    # Register in sys.modules so `import nvdiffrast.torch as dr` works
    sys.modules['nvdiffrast'] = fake_nvdiffrast
    sys.modules['nvdiffrast.torch'] = fake_dr

    print("[rocm_compat] Patched nvdiffrast with ROCm adapter")
    return True


def patch_nvdiffrec():
    """Replace nvdiffrec_render if not available."""
    try:
        import nvdiffrec_render
        print("[rocm_compat] nvdiffrec_render available, using native implementation")
        return False
    except ImportError:
        pass

    # Create minimal stubs
    import types
    import torch

    fake_render = types.ModuleType('nvdiffrec_render')

    class FakeEnvironmentLight:
        """Stub for nvdiffrec EnvironmentLight — PBR rendering won't work but won't crash."""
        def __init__(self, *args, **kwargs):
            print("[rocm_compat] Warning: EnvironmentLight is a stub (nvdiffrec not available)")

        def build_mips(self):
            pass

    fake_light = types.ModuleType('nvdiffrec_render.light')
    fake_light.EnvironmentLight = FakeEnvironmentLight

    fake_render.light = fake_light

    sys.modules['nvdiffrec_render'] = fake_render
    sys.modules['nvdiffrec_render.light'] = fake_light

    print("[rocm_compat] Patched nvdiffrec_render with stubs")
    return True


# Auto-patch on import
_patched_dr = patch_nvdiffrast()
_patched_rec = patch_nvdiffrec()
