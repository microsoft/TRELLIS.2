"""
Centralized backend resolution for trellis2-apple.

Resolves differentiable rasterization (dr), mesh processing (MeshBackend, BVH),
remeshing, and grid_sample once at import time. All other modules import from here.

Priority: Metal (mtldiffrast) > CUDA (nvdiffrast) > CPU fallback.
Mesh: cumesh (auto-selects Metal/CUDA internally).
"""
import platform
import os
from functools import lru_cache
import torch

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()
METAL_DISABLED = os.environ.get('TRELLIS_DISABLE_METAL', '0') == '1'

# Upstream microsoft/TRELLIS.2#169: fp16 GEMMs on non-CUDA backends accumulate
# in half precision, so decode-time decision logits near zero can flip sign
# versus fp32 and hard `> 0` thresholds punch scattered holes into the mesh.
FP32_DECODE_THRESHOLDS_DEFAULT = '0'


def fp32_decode_thresholds_enabled() -> bool:
    return os.environ.get(
        'TRELLIS_FP32_DECODE_THRESHOLDS', FP32_DECODE_THRESHOLDS_DEFAULT
    ) == '1'


BACKEND_ERRORS = {}

# ---------------------------------------------------------------------------
# Differentiable rasterizer  (dr)
# mtldiffrast (Metal) and nvdiffrast (CUDA) are separate packages
# ---------------------------------------------------------------------------
dr = None
_dr_backend = None

if not METAL_DISABLED:
    try:
        import mtldiffrast.torch as _mtldr
        dr = _mtldr
        _dr_backend = 'metal'
    except (ImportError, RuntimeError, OSError) as exc:
        BACKEND_ERRORS['mtldiffrast'] = str(exc)

if dr is None:
    try:
        import nvdiffrast.torch as _nvdr
        dr = _nvdr
        _dr_backend = 'cuda'
    except (ImportError, RuntimeError, OSError) as exc:
        BACKEND_ERRORS['nvdiffrast'] = str(exc)

HAS_DR = dr is not None


def RasterizeContext(device=None):
    """Create the appropriate rasterization context for the active backend."""
    if _dr_backend == 'metal':
        return dr.MtlRasterizeContext(device=device)
    elif _dr_backend == 'cuda':
        return dr.RasterizeCudaContext(device=device)
    raise RuntimeError("No differentiable rasterization backend available")

# ---------------------------------------------------------------------------
# Mesh processing  (MeshBackend, BVH, remesh)
# cumesh auto-selects Metal or CUDA backend internally
# ---------------------------------------------------------------------------
MeshBackend = None
BVH = None
remesh_narrow_band_dc = None
_mesh_backend = None

if not METAL_DISABLED or platform.system() != 'Darwin':
    try:
        import cumesh
        MeshBackend = cumesh.CuMesh
        BVH = cumesh.cuBVH
        remesh_narrow_band_dc = cumesh.remeshing.remesh_narrow_band_dc
        _mesh_backend = 'metal' if platform.system() == 'Darwin' else 'cuda'
    except (ImportError, RuntimeError, OSError) as exc:
        BACKEND_ERRORS['cumesh'] = str(exc)

HAS_MESH = MeshBackend is not None
HAS_REMESH = remesh_narrow_band_dc is not None

# ---------------------------------------------------------------------------
# CPU fallback mesh libraries
# ---------------------------------------------------------------------------
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    trimesh = None
    HAS_TRIMESH = False

try:
    import fast_simplification
    HAS_FAST_SIMPLIFICATION = True
except ImportError:
    fast_simplification = None
    HAS_FAST_SIMPLIFICATION = False

# ---------------------------------------------------------------------------
# Grid sample (flex_gemm on Metal/CUDA, F.grid_sample fallback)
# ---------------------------------------------------------------------------
_flex_grid_sample_3d = None
HAS_FLEX_GEMM = False

if not METAL_DISABLED or platform.system() != 'Darwin':
    try:
        from flex_gemm.ops.grid_sample import grid_sample_3d as _fgs3d
        _flex_grid_sample_3d = _fgs3d
        HAS_FLEX_GEMM = True
    except (ImportError, RuntimeError, OSError) as exc:
        BACKEND_ERRORS['flex_gemm'] = str(exc)

# ---------------------------------------------------------------------------
# Overall backend name
# ---------------------------------------------------------------------------
BACKEND = _dr_backend or _mesh_backend or ('cpu' if HAS_TRIMESH else None)
HAS_GPU = HAS_MPS or HAS_CUDA


def _probe_result(ok, error=None, **details):
    return {'ok': bool(ok), 'error': error, **details}


@lru_cache(maxsize=1)
def probe_metal_rasterizer():
    """Run a tiny UV raster instead of treating a successful import as proof."""
    if METAL_DISABLED or _dr_backend != 'metal':
        return _probe_result(False, 'Metal rasterizer is disabled or unavailable')
    try:
        positions = torch.tensor(
            [
                [-1.0, -1.0, 0.0, 1.0],
                [1.0, -1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [-1.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        triangles = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
        raster, _ = dr.rasterize(
            RasterizeContext(),
            positions,
            triangles,
            resolution=[8, 8],
        )
        covered = int((raster[..., 3] != 0).sum().item())
        ok = tuple(raster.shape) == (1, 8, 8, 4) and covered == 64
        return _probe_result(ok, None if ok else 'unexpected raster output', covered_pixels=covered)
    except Exception as exc:
        return _probe_result(False, f'{type(exc).__name__}: {exc}')


@lru_cache(maxsize=1)
def probe_metal_mesh_bvh():
    """Exercise both MtlMesh and an actual nearest-point MtlBVH query."""
    if METAL_DISABLED or _mesh_backend != 'metal' or not HAS_MPS:
        return _probe_result(False, 'Metal mesh/BVH backend is disabled or unavailable')
    try:
        vertices = torch.tensor(
            [
                [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
            device='mps',
        )
        faces = torch.tensor(
            [
                [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
            ],
            dtype=torch.int32,
            device='mps',
        )
        mesh = MeshBackend()
        mesh.init(vertices, faces)
        mesh_vertices, mesh_faces = mesh.read()
        bvh = BVH(vertices, faces)
        query = torch.tensor([[0.0, 0.0, 1.2]], dtype=torch.float32, device='mps')
        distance, face_id, uvw = bvh.unsigned_distance(query, return_uvw=True)
        torch.mps.synchronize()
        ok = (
            tuple(mesh_vertices.shape) == (8, 3)
            and tuple(mesh_faces.shape) == (12, 3)
            and bool(torch.isfinite(distance).all().item())
            and bool(torch.isfinite(uvw).all().item())
            and int(face_id.item()) >= 0
        )
        return _probe_result(
            ok,
            None if ok else 'unexpected mesh/BVH output',
            distance=float(distance.cpu().item()),
            face_id=int(face_id.cpu().item()),
        )
    except Exception as exc:
        return _probe_result(False, f'{type(exc).__name__}: {exc}')


def probe_metal_backends():
    return {
        'rasterizer': probe_metal_rasterizer(),
        'mesh_bvh': probe_metal_mesh_bvh(),
    }


def backend_report(run_probes=True):
    """Return a JSON-serializable snapshot used by setup and run metadata."""
    report = {
        'platform': platform.system(),
        'mps_available': HAS_MPS,
        'cuda_available': HAS_CUDA,
        'metal_disabled': METAL_DISABLED,
        'rasterizer': _dr_backend,
        'mesh': _mesh_backend,
        'flex_gemm': HAS_FLEX_GEMM,
        'trimesh': HAS_TRIMESH,
        'fast_simplification': HAS_FAST_SIMPLIFICATION,
        'errors': dict(BACKEND_ERRORS),
    }
    if run_probes and platform.system() == 'Darwin':
        report['capability_probes'] = probe_metal_backends()
    return report
