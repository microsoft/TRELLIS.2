#!/usr/bin/env python3
"""
Test client for the TRELLIS 2 API server
This script tests both image-to-3D and text-to-3D endpoints
"""

import os
import sys
import json
import time
import requests
import argparse
from pathlib import Path

# Configuration
DEFAULT_SERVER_URL = "http://localhost:6006"
DEFAULT_IMAGE_PATH = "assets/example_image/typical_creature_dragon.png"
DEFAULT_TEXT_PROMPT = "A majestic dragon with detailed scales, sharp claws, and wings"

def test_image_to_3d(server_url, image_path, verbose=False):
    """Test the image-to-3D endpoint"""
    endpoint = f"{server_url}/image_to_3d"

    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return

    # Prepare form data
    files = {
        "image": (os.path.basename(image_path), open(image_path, "rb"), "image/png")
    }

    form_data = {
        "seed": 42,
        "randomize_seed": False,
        "preprocess_image": True,
        "generate_video": True,
        "generate_model": True,
        "ss_sample_steps": 25,
        "ss_cfg_strength": 8.5,
        "slat_sample_steps": 25,
        "slat_cfg_strength": 4.0,
    }

    print(f"Sending image {image_path} to {endpoint}...")
    try:
        response = requests.post(endpoint, files=files, data=form_data)
        
        if response.status_code == 200:
            result = response.json()
            request_id = result.get("request_id")
            print(f"Request submitted successfully. Request ID: {request_id}")

            if verbose:
                print(json.dumps(result, indent=2))

            poll_task_status(server_url, request_id)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")


def test_text_to_3d(server_url, text_prompt, verbose=False):
    """Test the text-to-3D endpoint"""
    endpoint = f"{server_url}/text_to_3d"

    # Prepare form data
    form_data = {
        "text": text_prompt,
        "negative_text": "",
        "seed": 42,
        "randomize_seed": False,
        "image_width": 1024,
        "image_height": 1024,
        "num_inference_steps": 9,
        "preprocess_image": True,
        "generate_video": True,
        "generate_model": True,
        "ss_sample_steps": 30,
        "ss_cfg_strength": 7.5,
        "slat_sample_steps": 30,
        "slat_cfg_strength": 4.5,
    }

    print(f"Sending text prompt to {endpoint}...")
    print(f"Prompt: {text_prompt}")

    try:
        response = requests.post(endpoint, data=form_data)

        if response.status_code == 200:
            result = response.json()
            request_id = result.get("request_id")
            print(f"Request submitted successfully. Request ID: {request_id}")

            if verbose:
                print(json.dumps(result, indent=2))

            poll_task_status(server_url, request_id)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")


def poll_task_status(server_url, request_id, interval=5, max_attempts=60):
    """Poll for task status until completion or error"""
    endpoint = f"{server_url}/task/{request_id}"
    attempts = 0

    print(f"Polling for task status (request ID: {request_id})...")

    while attempts < max_attempts:
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                task = response.json()
                status = task.get("status")
                print(f"Status: {status}")

                if status == "complete":
                    print("Task completed successfully!")
                    output_files = task.get("output_files", [])
                    print(f"Output files: {output_files}")
                    return True
                elif status == "error":
                    print(f"Task failed with error: {task.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"Error checking status: {response.status_code}")
                
        except Exception as e:
            print(f"Error polling for status: {str(e)}")

        attempts += 1
        time.sleep(interval)

    print(f"Gave up after {max_attempts} attempts")
    return False

def check_server_status(server_url):
    """Check if the server is running"""
    try:
        response = requests.get(f"{server_url}/status", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def check_queue_status(server_url):
    """Check queue status"""
    try:
        response = requests.get(f"{server_url}/queue_status", timeout=5)
        if response.status_code == 200:
            print("\nQueue Status:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error checking queue: {response.status_code}")
    except Exception as e:
        print(f"Error checking queue: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test TRELLIS 2 API endpoints")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="Server URL")
    parser.add_argument(
        "--mode",
        choices=["image", "text", "both", "status", "queue"],
        default="both",
        help="Test mode: image-to-3D, text-to-3D, both, server status, or queue status",
    )
    parser.add_argument(
        "--image", default=DEFAULT_IMAGE_PATH, help="Path to image for image-to-3D test"
    )
    parser.add_argument(
        "--text", default=DEFAULT_TEXT_PROMPT, help="Text prompt for text-to-3D test"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")

    args = parser.parse_args()

    # Status check
    if args.mode == "status":
        print(f"Checking server status at {args.server}")
        if check_server_status(args.server):
            print("✅ Server is running")
        else:
            print("❌ Server is not responding")
        return
        
    # Queue check
    if args.mode == "queue":
        check_queue_status(args.server)
        return

    # Server check before other tests
    if not check_server_status(args.server):
        print(f"Error: Server at {args.server} is not responding")
        sys.exit(1)

    print(f"Testing TRELLIS 2 API at {args.server}")

    if args.mode in ["image", "both"]:
        # Check if default image exists relative to script or use provided absolute path
        img_path = args.image
        if not os.path.exists(img_path) and not os.path.isabs(img_path):
             # Try finding it in the project root if running from subdir
             project_root_img = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), img_path)
             if os.path.exists(project_root_img):
                 img_path = project_root_img
                 
        test_image_to_3d(args.server, img_path, args.verbose)

    if args.mode in ["text", "both"]:
        test_text_to_3d(args.server, args.text, args.verbose)

if __name__ == "__main__":
    main()
