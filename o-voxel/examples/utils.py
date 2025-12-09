import os
import requests
import tarfile
import trimesh

HELMET_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/refs/heads/main/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb"
CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache")


def download_file(url, path):
    print(f"Downloading from {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved to {path}")


def get_helmet() -> trimesh.Trimesh:
    HELMET_PATH = os.path.join(CACHE_DIR, "helmet.glb")
    if not os.path.exists(HELMET_PATH):
        os.makedirs(CACHE_DIR, exist_ok=True)
        download_file(HELMET_URL, HELMET_PATH)
    return trimesh.load(HELMET_PATH)
