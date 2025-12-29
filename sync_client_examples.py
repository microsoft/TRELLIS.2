"""
Python client examples for TRELLIS2 FastAPI endpoints

This script demonstrates how to use the TRELLIS2 API for image-to-3D generation.
"""

import requests
import json
import time
from pathlib import Path
from typing import Optional


class TRELLIS2Client:
    """Client for interacting with TRELLIS2 API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> dict:
        """
        Check API health status
        
        Returns:
            Health status response
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def image_to_3d(
        self,
        image_path: str,
        seed: int = 42,
        randomize_seed: bool = False,
        preprocess_image: bool = True,
        generate_video: bool = False,
        generate_model: bool = True,
        pipeline_type: str = "512",
        sparse_structure_steps: int = 12,
        sparse_structure_guidance_strength: float = 7.5,
        sparse_structure_guidance_rescale: float = 0.7,
        sparse_structure_rescale_t: float = 5.0,
        shape_slat_steps: int = 12,
        shape_slat_guidance_strength: float = 7.5,
        shape_slat_guidance_rescale: float = 0.5,
        shape_slat_rescale_t: float = 3.0,
        tex_slat_steps: int = 12,
        tex_slat_guidance_strength: float = 1.0,
        tex_slat_guidance_rescale: float = 0.0,
        tex_slat_rescale_t: float = 3.0,
        decimation_target: int = 1000000,
        texture_size: int = 4096,
    ) -> dict:
        """
        Generate 3D model from image
        
        Args:
            image_path: Path to input image
            seed: Random seed for reproducibility
            randomize_seed: Whether to use a random seed
            preprocess_image: Whether to remove background and crop
            generate_video: Whether to generate preview video
            generate_model: Whether to generate GLB model
            pipeline_type: Quality/speed tradeoff (512, 1024, 1024_cascade, 1536_cascade)
            sparse_structure_steps: Sampling steps for sparse structure
            sparse_structure_guidance_strength: Guidance strength for sparse structure
            sparse_structure_guidance_rescale: Guidance rescale for sparse structure
            sparse_structure_rescale_t: Rescale T for sparse structure
            shape_slat_steps: Sampling steps for shape
            shape_slat_guidance_strength: Guidance strength for shape
            shape_slat_guidance_rescale: Guidance rescale for shape
            shape_slat_rescale_t: Rescale T for shape
            tex_slat_steps: Sampling steps for texture
            tex_slat_guidance_strength: Guidance strength for texture
            tex_slat_guidance_rescale: Guidance rescale for texture
            tex_slat_rescale_t: Rescale T for texture
            decimation_target: Target face count for decimation
            texture_size: Output texture resolution
        
        Returns:
            Generation response with URLs to generated assets
        """
        with open(image_path, 'rb') as f:
            files = {'image': (Path(image_path).name, f, 'image/png')}
            data = {
                'seed': seed,
                'randomize_seed': randomize_seed,
                'preprocess_image': preprocess_image,
                'generate_video': generate_video,
                'generate_model': generate_model,
                'pipeline_type': pipeline_type,
                'sparse_structure_steps': sparse_structure_steps,
                'sparse_structure_guidance_strength': sparse_structure_guidance_strength,
                'sparse_structure_guidance_rescale': sparse_structure_guidance_rescale,
                'sparse_structure_rescale_t': sparse_structure_rescale_t,
                'shape_slat_steps': shape_slat_steps,
                'shape_slat_guidance_strength': shape_slat_guidance_strength,
                'shape_slat_guidance_rescale': shape_slat_guidance_rescale,
                'shape_slat_rescale_t': shape_slat_rescale_t,
                'tex_slat_steps': tex_slat_steps,
                'tex_slat_guidance_strength': tex_slat_guidance_strength,
                'tex_slat_guidance_rescale': tex_slat_guidance_rescale,
                'tex_slat_rescale_t': tex_slat_rescale_t,
                'decimation_target': decimation_target,
                'texture_size': texture_size,
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/image-to-3d",
                files=files,
                data=data
            )
        
        response.raise_for_status()
        return response.json()
    
    def text_to_3d(
        self,
        prompt: str,
        negative_prompt: str = "",
        image_width: int = 1024,
        image_height: int = 1024,
        num_inference_steps: int = 9,
        seed: int = 42,
        randomize_seed: bool = False,
        preprocess_image: bool = True,
        generate_video: bool = False,
        generate_model: bool = True,
        pipeline_type: str = "512",
        sparse_structure_steps: int = 12,
        sparse_structure_guidance_strength: float = 7.5,
        sparse_structure_guidance_rescale: float = 0.7,
        sparse_structure_rescale_t: float = 5.0,
        shape_slat_steps: int = 12,
        shape_slat_guidance_strength: float = 7.5,
        shape_slat_guidance_rescale: float = 0.5,
        shape_slat_rescale_t: float = 3.0,
        tex_slat_steps: int = 12,
        tex_slat_guidance_strength: float = 1.0,
        tex_slat_guidance_rescale: float = 0.0,
        tex_slat_rescale_t: float = 3.0,
        decimation_target: int = 1000000,
        texture_size: int = 4096,
    ) -> dict:
        """
        Generate 3D model from text prompt
        
        First generates an image from text, then converts to 3D.
        
        Args:
            prompt: Text description of the desired 3D model
            negative_prompt: Things to avoid in the generation
            image_width: Width of generated image
            image_height: Height of generated image
            num_inference_steps: Steps for image generation (9 recommended)
            seed: Random seed for reproducibility
            randomize_seed: Whether to use a random seed
            preprocess_image: Whether to remove background and crop
            generate_video: Whether to generate preview video
            generate_model: Whether to generate GLB model
            pipeline_type: Quality/speed tradeoff (512, 1024, 1024_cascade, 1536_cascade)
            ... (other 3D parameters same as image_to_3d)
        
        Returns:
            Generation response with URLs to generated assets
        """
        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'image_width': image_width,
            'image_height': image_height,
            'num_inference_steps': num_inference_steps,
            'seed': seed,
            'randomize_seed': randomize_seed,
            'preprocess_image': preprocess_image,
            'generate_video': generate_video,
            'generate_model': generate_model,
            'pipeline_type': pipeline_type,
            'sparse_structure_steps': sparse_structure_steps,
            'sparse_structure_guidance_strength': sparse_structure_guidance_strength,
            'sparse_structure_guidance_rescale': sparse_structure_guidance_rescale,
            'sparse_structure_rescale_t': sparse_structure_rescale_t,
            'shape_slat_steps': shape_slat_steps,
            'shape_slat_guidance_strength': shape_slat_guidance_strength,
            'shape_slat_guidance_rescale': shape_slat_guidance_rescale,
            'shape_slat_rescale_t': shape_slat_rescale_t,
            'tex_slat_steps': tex_slat_steps,
            'tex_slat_guidance_strength': tex_slat_guidance_strength,
            'tex_slat_guidance_rescale': tex_slat_guidance_rescale,
            'tex_slat_rescale_t': tex_slat_rescale_t,
            'decimation_target': decimation_target,
            'texture_size': texture_size,
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/text-to-3d",
            data=data
        )
        
        response.raise_for_status()
        return response.json()
    
    def download_file(self, url: str, output_path: str) -> None:
        """
        Download a file from the API
        
        Args:
            url: URL path (e.g., /api/v1/outputs/job_id/output.glb)
            output_path: Local path to save the file
        """
        full_url = f"{self.base_url}{url}"
        response = requests.get(full_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def delete_job(self, job_id: str) -> dict:
        """
        Delete all outputs for a job
        
        Args:
            job_id: The job ID
        
        Returns:
            Delete response
        """
        response = requests.delete(f"{self.base_url}/api/v1/outputs/{job_id}")
        response.raise_for_status()
        return response.json()


# Example 1: Basic image-to-3D generation
def example_basic():
    """Basic example: Generate 3D model from image"""
    print("=" * 60)
    print("Example 1: Basic Image-to-3D Generation")
    print("=" * 60)
    
    client = TRELLIS2Client()
    
    # Check if API is healthy
    print("Checking API health...")
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Model Loaded: {health['model_loaded']}")
    print()
    
    # Generate 3D from image
    print("Generating 3D model from image...")
    image_path = "assets/example_image/T.png"
    
    result = client.image_to_3d(
        image_path=image_path,
        seed=42,
        generate_video=True,
        generate_model=True,
        pipeline_type="512"
    )
    
    print(f"Job ID: {result['job_id']}")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    # Download results
    output_dir = Path("client_outputs") / result['job_id']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if result['video_url']:
        print(f"\nDownloading video...")
        client.download_file(result['video_url'], str(output_dir / "output.mp4"))
        print(f"Saved to: {output_dir / 'output.mp4'}")
    
    if result['model_url']:
        print(f"\nDownloading GLB model...")
        client.download_file(result['model_url'], str(output_dir / "output.glb"))
        print(f"Saved to: {output_dir / 'output.glb'}")
    
    if result['preview_image_url']:
        print(f"\nDownloading preview image...")
        client.download_file(result['preview_image_url'], str(output_dir / "preview.png"))
        print(f"Saved to: {output_dir / 'preview.png'}")
    
    print("\n✓ Example 1 completed successfully!")
    print()


# Example 2: High-quality generation with custom parameters
def example_high_quality():
    """Advanced example: High-quality generation with custom parameters"""
    print("=" * 60)
    print("Example 2: High-Quality Generation")
    print("=" * 60)
    
    client = TRELLIS2Client()
    
    print("Generating high-quality 3D model...")
    image_path = "assets/example_image/T.png"
    
    result = client.image_to_3d(
        image_path=image_path,
        seed=123,
        pipeline_type="1536_cascade",  # Highest quality
        sparse_structure_steps=20,  # More steps for better quality
        shape_slat_steps=20,
        tex_slat_steps=20,
        decimation_target=2000000,  # More faces
        texture_size=8192,  # Higher texture resolution
        generate_video=True,
        generate_model=True,
    )
    
    print(f"Job ID: {result['job_id']}")
    print(f"Status: {result['status']}")
    
    # Download only the model
    output_dir = Path("client_outputs") / result['job_id']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if result['model_url']:
        print(f"\nDownloading high-quality GLB model...")
        client.download_file(result['model_url'], str(output_dir / "output_hq.glb"))
        print(f"Saved to: {output_dir / 'output_hq.glb'}")
    
    print("\n✓ Example 2 completed successfully!")
    print()


# Example 3: Fast generation for quick preview
def example_fast():
    """Fast generation example for quick preview"""
    print("=" * 60)
    print("Example 3: Fast Generation")
    print("=" * 60)
    
    client = TRELLIS2Client()
    
    print("Generating 3D model quickly...")
    image_path = "assets/example_image/T.png"
    
    result = client.image_to_3d(
        image_path=image_path,
        seed=456,
        pipeline_type="512",  # Fastest option
        sparse_structure_steps=8,  # Fewer steps
        shape_slat_steps=8,
        tex_slat_steps=8,
        generate_video=False,  # Skip video for speed
        generate_model=True,
    )
    
    print(f"Job ID: {result['job_id']}")
    print(f"Status: {result['status']}")
    
    # Download model
    output_dir = Path("client_outputs") / result['job_id']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if result['model_url']:
        print(f"\nDownloading GLB model...")
        client.download_file(result['model_url'], str(output_dir / "output_fast.glb"))
        print(f"Saved to: {output_dir / 'output_fast.glb'}")
    
    print("\n✓ Example 3 completed successfully!")
    print()


# Example 5: Text-to-3D generation
def example_text_to_3d():
    """Text-to-3D example"""
    print("=" * 60)
    print("Example 5: Text-to-3D Generation")
    print("=" * 60)
    
    client = TRELLIS2Client()
    
    print("Generating 3D model from text prompt...")
    prompt = "A cute robot toy, simple design, colorful, studio lighting, white background"
    
    result = client.text_to_3d(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted",
        image_width=1024,
        image_height=1024,
        num_inference_steps=9,
        seed=42,
        pipeline_type="512",
        generate_video=False,
        generate_model=True,
    )
    
    print(f"Job ID: {result['job_id']}")
    print(f"Status: {result['status']}")
    print(f"Prompt: {prompt}")
    
    # Download results
    output_dir = Path("client_outputs") / result['job_id']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if result['video_url']:
        print(f"\nDownloading video...")
        client.download_file(result['video_url'], str(output_dir / "output.mp4"))
        print(f"Saved to: {output_dir / 'output.mp4'}")
    
    if result['model_url']:
        print(f"\nDownloading GLB model...")
        client.download_file(result['model_url'], str(output_dir / "output.glb"))
        print(f"Saved to: {output_dir / 'output.glb'}")
    
    if result['preview_image_url']:
        print(f"\nDownloading preview image...")
        client.download_file(result['preview_image_url'], str(output_dir / "preview.png"))
        print(f"Saved to: {output_dir / 'preview.png'}")
    
    print("\n✓ Example 5 completed successfully!")
    print()


# Example 6: Batch processing multiple images
def example_batch():
    """Batch processing example"""
    print("=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)
    
    client = TRELLIS2Client()
    
    # List of images to process
    images = [
        "assets/example_image/T.png",
        # Add more images here
    ]
    
    jobs = []
    
    for i, image_path in enumerate(images):
        print(f"\nProcessing image {i+1}/{len(images)}: {image_path}")
        
        try:
            result = client.image_to_3d(
                image_path=image_path,
                seed=42 + i,
                pipeline_type="512",
                generate_video=False,  # Skip video for batch processing
                generate_model=True,
            )
            
            jobs.append({
                'image': image_path,
                'job_id': result['job_id'],
                'model_url': result['model_url']
            })
            
            print(f"  Job ID: {result['job_id']}")
            print(f"  Status: {result['status']}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Download all models
    print(f"\n\nDownloading {len(jobs)} models...")
    output_dir = Path("client_outputs") / "batch"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, job in enumerate(jobs):
        if job['model_url']:
            filename = f"model_{i+1}.glb"
            print(f"Downloading {filename}...")
            client.download_file(job['model_url'], str(output_dir / filename))
    
    print(f"\n✓ Example 4 completed successfully!")
    print(f"All models saved to: {output_dir}")
    print()


# Example 7: Error handling
def example_error_handling():
    """Example with error handling"""
    print("=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)
    
    client = TRELLIS2Client()
    
    try:
        # Try with invalid image
        result = client.image_to_3d(
            image_path="nonexistent.png",
            generate_model=True,
        )
    except FileNotFoundError as e:
        print(f"✓ Caught expected error: File not found")
    except Exception as e:
        print(f"✓ Caught error: {str(e)}")
    
    try:
        # Try with invalid pipeline type
        result = client.image_to_3d(
            image_path="assets/example_image/T.png",
            pipeline_type="invalid_type",
        )
    except requests.exceptions.HTTPError as e:
        print(f"✓ Caught expected error: Invalid pipeline type")
    except Exception as e:
        print(f"✓ Caught error: {str(e)}")
    
    print("\n✓ Example 7 completed successfully!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRELLIS2 API Client Examples")
    print("=" * 60 + "\n")
    
    # Check if server is running
    client = TRELLIS2Client()
    try:
        health = client.health_check()
        print(f"✓ API server is running and healthy")
        print(f"  Status: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}\n")
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Cannot connect to API server")
        print("  Please start the server first with: python api.py")
        print("  Or using: uvicorn api:app --host 0.0.0.0 --port 8000\n")
        exit(1)
    
    # Run examples
    try:
        # example_basic()
        # Uncomment to run other examples:
        # example_high_quality()
        # example_fast()
        example_text_to_3d()
        # example_batch()
        # example_error_handling()
    except Exception as e:
        print(f"\n✗ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()

