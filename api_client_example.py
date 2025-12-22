"""
Example client for the TRELLIS.2 3D Generation API.

Usage:
    python api_client_example.py <image_path> [--api-url http://localhost:8000]
"""

import argparse
import base64
import time
import requests
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """Read image file and encode to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def save_base64_to_file(base64_string: str, output_path: str):
    """Decode base64 string and save to file"""
    data = base64.b64decode(base64_string)
    with open(output_path, 'wb') as f:
        f.write(data)
    print(f"Saved: {output_path}")


def generate_3d(api_url: str, image_path: str, resolution: str = "1024"):
    """
    Submit image for 3D generation and wait for result.
    """
    print(f"Loading image: {image_path}")
    image_base64 = encode_image_to_base64(image_path)
    
    # Submit generation request
    print(f"Submitting to API: {api_url}/generate")
    response = requests.post(
        f"{api_url}/generate",
        json={
            "image_base64": image_base64,
            "resolution": resolution,
            "decimation_target": 500000,
            "texture_size": 2048,
        }
    )
    response.raise_for_status()
    result = response.json()
    
    job_id = result["job_id"]
    print(f"Job submitted! ID: {job_id}")
    
    # Poll for status - files are included directly in status response
    output_dir = Path(image_path).parent
    base_name = Path(image_path).stem
    mesh_saved = False
    
    while True:
        response = requests.get(f"{api_url}/status/{job_id}")
        response.raise_for_status()
        status = response.json()
        
        print(f"Status: {status['status']} - {status['progress']:.0f}% - {status['message']}")
        
        # Save mesh-only as soon as it's ready (before texturing completes)
        if status.get("mesh_only_ready") and status.get("mesh_only_base64") and not mesh_saved:
            mesh_path = output_dir / f"{base_name}_mesh_only.glb"
            save_base64_to_file(status["mesh_only_base64"], str(mesh_path))
            mesh_saved = True
            print("✓ Mesh-only GLB saved!")
        
        if status["status"] == "completed":
            # Save textured GLB
            if status.get("glb_base64"):
                glb_path = output_dir / f"{base_name}_output.glb"
                save_base64_to_file(status["glb_base64"], str(glb_path))
                print("✓ Textured GLB saved!")
            break
        elif status["status"] == "failed":
            print(f"Generation failed: {status.get('message', 'Unknown error')}")
            return None
        
        time.sleep(2)  # Poll every 2 seconds
    
    print("Done!")
    return job_id


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 API Client")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--resolution", default="1024", choices=["512", "1024", "1536"], help="Generation resolution")
    args = parser.parse_args()
    
    generate_3d(args.api_url, args.image_path, args.resolution)


if __name__ == "__main__":
    main()

