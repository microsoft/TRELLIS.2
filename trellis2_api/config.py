"""
Configuration constants for the TRELLIS 2 API
"""
import os
from pathlib import Path

# Base configuration
BASE_IP = "http://127.0.0.1"
BASE_PORT = 6006
BASE_URL = f"{BASE_IP}:{BASE_PORT}"

# Directories
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
SERVER_ROOT = os.path.join(PROJECT_ROOT, "service")
INPUT_DIR = os.path.join(SERVER_ROOT, "input")
OUTPUT_DIR = os.path.join(SERVER_ROOT, "output")
DB_PATH = os.path.join(PACKAGE_ROOT, "trellis2.db")

# Create necessary directories
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Application settings
IP_HISTORY_LIMIT = 50
QUEUE_SIZE = 100

# Model configuration
MODEL_ID = "microsoft/TRELLIS.2-4B"
TEXT_TO_IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
