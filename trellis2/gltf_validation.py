"""Small glTF 2.0 validator for generated TRELLIS runtime outputs."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any

from PIL import Image

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def _load_glb(path: Path) -> tuple[dict[str, Any], bytes, bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError(f"truncated GLB: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ValueError(f"not a complete glTF 2.0 binary: {path}")
    document = None
    binary = b""
    offset = 12
    while offset < len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK:
            document = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
        elif chunk_type == BIN_CHUNK:
            binary = chunk
    if document is None:
        raise ValueError(f"missing GLB JSON chunk: {path}")
    return document, binary, payload


def _view(document: dict[str, Any], binary: bytes, index: int) -> bytes:
    view = document["bufferViews"][index]
    start = int(view.get("byteOffset", 0))
    return binary[start : start + int(view["byteLength"])]


def _image(document: dict[str, Any], binary: bytes, index: int) -> bytes:
    descriptor = document["images"][index]
    if "bufferView" in descriptor:
        return _view(document, binary, int(descriptor["bufferView"]))
    uri = descriptor.get("uri", "")
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    raise ValueError("all runtime textures must be embedded")


def _texture_payload(document: dict[str, Any], binary: bytes, info: dict[str, Any]) -> bytes:
    texture = document["textures"][int(info["index"])]
    return _image(document, binary, int(texture["source"]))


def inspect_glb(path: Path, *, require_pbr: bool) -> dict[str, Any]:
    document, binary, payload = _load_glb(path)
    if document.get("asset", {}).get("version") != "2.0":
        raise ValueError("asset.version must be 2.0")
    if document.get("animations") or document.get("cameras"):
        raise ValueError("static outputs must not contain animations or cameras")
    if document.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights"):
        raise ValueError("static outputs must not contain lights")

    vertices = 0
    triangles = 0
    bounds_min = [float("inf")] * 3
    bounds_max = [float("-inf")] * 3
    base_hashes = []
    mr_hashes = []
    primitive_count = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            primitive_count += 1
            if int(primitive.get("mode", 4)) != 4:
                raise ValueError("only TRIANGLES primitives are supported")
            attributes = primitive.get("attributes", {})
            required = ["POSITION", "NORMAL"] + (["TEXCOORD_0"] if require_pbr else [])
            missing = [name for name in required if name not in attributes]
            if missing:
                raise ValueError(f"primitive is missing attributes: {', '.join(missing)}")
            position = document["accessors"][int(attributes["POSITION"])]
            if not position.get("min") or not position.get("max"):
                raise ValueError("POSITION accessor has no bounds")
            vertices += int(position["count"])
            for axis in range(3):
                bounds_min[axis] = min(bounds_min[axis], float(position["min"][axis]))
                bounds_max[axis] = max(bounds_max[axis], float(position["max"][axis]))
            index_count = (
                int(document["accessors"][int(primitive["indices"])]["count"])
                if "indices" in primitive
                else int(position["count"])
            )
            if index_count <= 0 or index_count % 3:
                raise ValueError("invalid triangle index count")
            triangles += index_count // 3

            if require_pbr:
                if "material" not in primitive:
                    raise ValueError("PBR primitive has no material")
                material = document["materials"][int(primitive["material"])]
                pbr = material.get("pbrMetallicRoughness", {})
                for key, hashes in (
                    ("baseColorTexture", base_hashes),
                    ("metallicRoughnessTexture", mr_hashes),
                ):
                    if key not in pbr:
                        raise ValueError(f"PBR material is missing {key}")
                    texture = _texture_payload(document, binary, pbr[key])
                    hashes.append(hashlib.sha256(texture).hexdigest())
                with Image.open(io.BytesIO(_texture_payload(document, binary, pbr["baseColorTexture"]))) as image:
                    if "A" not in image.getbands():
                        raise ValueError("base-color texture is missing generated alpha")

    if primitive_count == 0 or vertices == 0 or triangles == 0:
        raise ValueError("GLB contains no non-empty mesh")
    dimensions = [bounds_max[index] - bounds_min[index] for index in range(3)]
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("GLB has empty bounds")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "vertices": vertices,
        "triangles": triangles,
        "primitives": primitive_count,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "dimensions": dimensions,
        "base_color_image_hashes": base_hashes,
        "metallic_roughness_image_hashes": mr_hashes,
    }


def validate_output_pair(raw_path: Path, candidate_path: Path) -> dict[str, Any]:
    raw = inspect_glb(raw_path, require_pbr=False)
    candidate = inspect_glb(candidate_path, require_pbr=True)
    for raw_size, candidate_size in zip(raw["dimensions"], candidate["dimensions"]):
        relative_delta = abs(raw_size - candidate_size) / max(raw_size, 1e-8)
        if relative_delta > 0.05:
            raise ValueError(
                "raw_full.glb and candidate_pbr.glb bounds differ by more than 5%"
            )
    return {"raw_full": raw, "candidate_pbr": candidate, "bounds_consistent": True}
