import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import uuid
import shutil
import logging
import asyncio
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .worker import start_worker
from .web_utils import (
    create_task,
    get_ip_output_dir,
    get_tasks_for_ip,
    get_task_by_id,
    TaskType,
    TaskStatus
)
from .config import BASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRELLIS2-API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background worker
    start_worker()
    yield
    # Shutdown logic if needed

app = FastAPI(title="TRELLIS 2 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_client_ip(request: Request):
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

@app.get("/status")
async def status():
    return {"status": "ok", "message": "TRELLIS 2 API server is running"}

@app.get("/my_requests")
async def get_my_requests_endpoint(request: Request):
    client_ip = get_client_ip(request)
    tasks = get_tasks_for_ip(client_ip)
    
    # Format response to match original API
    formatted_requests = []
    for task in tasks:
        output_files = []
        if task["status"] == TaskStatus.COMPLETE.value:
            for f in task["output_files"]:
                output_files.append(f"{BASE_URL}/output/{client_ip}/{task['request_id']}/{f}")
                
        formatted_requests.append({
            "request_id": task["request_id"],
            "task_type": task["task_type"],
            "status": task["status"],
            "start_time": task["start_time"],
            "finish_time": task["finish_time"],
            "output_files": output_files,
            "error": task["error"],
            "text": task["input_text"]
        })
        
    return {"ip_address": client_ip, "requests": formatted_requests}

@app.get("/output/{ip_address}/{request_id}/{filename}")
async def serve_file(ip_address: str, request_id: str, filename: str):
    output_dir = get_ip_output_dir(ip_address)
    file_path = os.path.join(output_dir, request_id, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path)

@app.post("/image_to_3d")
async def image_to_3d(
    request: Request,
    image: UploadFile = File(...),
    seed: int = Form(42),
    randomize_seed: bool = Form(False),
    preprocess_image: bool = Form(True),
    generate_video: bool = Form(True),
    generate_model: bool = Form(True),
    pipeline_type: str = Form("512"),
    ss_sample_steps: int = Form(12),
    ss_cfg_strength: float = Form(7.5),
    slat_sample_steps: int = Form(12),
    slat_cfg_strength: float = Form(3.5)
):
    client_ip = get_client_ip(request)
    request_id = str(uuid.uuid4())
    
    # Setup directories
    ip_dir = get_ip_output_dir(client_ip)
    request_dir = os.path.join(ip_dir, request_id)
    os.makedirs(request_dir, exist_ok=True)
    
    # Save input image
    input_filename = f"input_{image.filename}"
    input_path = os.path.join(request_dir, input_filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    # Map parameters
    params = {
        "seed": seed, "randomize_seed": randomize_seed,
        "preprocess_image": preprocess_image,
        "generate_video": generate_video,
        "generate_model": generate_model,
        "pipeline_type": pipeline_type,
        "sparse_structure_steps": ss_sample_steps,
        "sparse_structure_guidance_strength": ss_cfg_strength,
        "shape_slat_steps": slat_sample_steps,
        "shape_slat_guidance_strength": slat_cfg_strength,
        # Default other params for simplicity
    }
    
    create_task(
        request_id=request_id,
        task_type=TaskType.IMAGE_TO_3D,
        client_ip=client_ip,
        output_dir=request_dir,
        params=params,
        input_path=input_path
    )
    
    return {
        "request_id": request_id,
        "status": TaskStatus.QUEUED.value,
        "message": "Request queued successfully"
    }

@app.post("/text_to_3d")
async def text_to_3d(
    request: Request,
    text: str = Form(...),
    negative_text: str = Form(""),
    seed: int = Form(42),
    randomize_seed: bool = Form(False),
    # Image Gen Params
    image_width: int = Form(1024),
    image_height: int = Form(1024),
    num_inference_steps: int = Form(9),
    # 3D Params
    preprocess_image: bool = Form(True),
    generate_video: bool = Form(True),
    generate_model: bool = Form(True),
    pipeline_type: str = Form("512"),
    ss_sample_steps: int = Form(12),
    ss_cfg_strength: float = Form(7.5),
    slat_sample_steps: int = Form(12),
    slat_cfg_strength: float = Form(3.5)
):
    client_ip = get_client_ip(request)
    request_id = str(uuid.uuid4())
    
    # Setup directories
    ip_dir = get_ip_output_dir(client_ip)
    request_dir = os.path.join(ip_dir, request_id)
    os.makedirs(request_dir, exist_ok=True)
    
    # Map parameters
    params = {
        "prompt": text,
        "negative_prompt": negative_text,
        "image_width": image_width, "image_height": image_height,
        "num_inference_steps": num_inference_steps,
        "seed": seed, "randomize_seed": randomize_seed,
        "preprocess_image": preprocess_image,
        "generate_video": generate_video,
        "generate_model": generate_model,
        "pipeline_type": pipeline_type,
        "sparse_structure_steps": ss_sample_steps,
        "sparse_structure_guidance_strength": ss_cfg_strength,
        "shape_slat_steps": slat_sample_steps,
        "shape_slat_guidance_strength": slat_cfg_strength,
    }
    
    create_task(
        request_id=request_id,
        task_type=TaskType.TEXT_TO_3D,
        client_ip=client_ip,
        output_dir=request_dir,
        params=params,
        input_text=text
    )
    
    return {
        "request_id": request_id,
        "status": TaskStatus.QUEUED.value,
        "message": "Request queued successfully"
    }

@app.get("/queue_status")
async def queue_status():
    from .models import Session, Task
    session = Session()
    try:
        queue_length = session.query(Task).filter(Task.status == TaskStatus.QUEUED.value).count()
        processing = session.query(Task).filter(Task.status == TaskStatus.PROCESSING.value).all()
        return {
            "queue_length": queue_length,
            "processing_requests": [
                {
                    "request_id": t.request_id,
                    "start_time": t.start_time,
                    "worker_pid": t.worker_pid
                } for t in processing
            ]
        }
    finally:
        session.close()

@app.get("/task/{request_id}")
async def get_task_endpoint(request_id: str):
    task = get_task_by_id(request_id)
    if task:
        return task
    raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)
