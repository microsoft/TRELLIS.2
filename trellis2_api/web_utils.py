import os
from datetime import datetime
from sqlalchemy import desc
from .models import Session, Task, TaskStatus, TaskType
from .config import OUTPUT_DIR, IP_HISTORY_LIMIT

def get_ip_output_dir(ip_address):
    """Get the output directory for a specific IP address"""
    safe_ip = ip_address.replace(":", "_")
    ip_dir = os.path.join(OUTPUT_DIR, safe_ip)
    os.makedirs(ip_dir, exist_ok=True)
    return ip_dir

def create_task(request_id, task_type, client_ip, output_dir, params, input_path=None, input_text=None):
    """Create a new task in the database"""
    session = Session()
    try:
        task = Task(
            request_id=request_id,
            task_type=task_type,
            status=TaskStatus.QUEUED.value,
            client_ip=client_ip,
            request_output_dir=output_dir,
            input_path=input_path,
            input_text=input_text,
            params=params,
            created_at=datetime.now()
        )
        session.add(task)
        session.commit()
        return task
    finally:
        session.close()

def get_next_task():
    """Get the next queued task"""
    session = Session()
    try:
        task = session.query(Task).filter(
            Task.status == TaskStatus.QUEUED.value
        ).order_by(Task.created_at.asc()).first()
        
        if task:
            task.status = TaskStatus.PROCESSING.value
            task.start_time = datetime.now()
            task.worker_pid = os.getpid()
            session.commit()
            
            # Return a detached copy/dict so we can close the session
            return {
                "request_id": task.request_id,
                "task_type": task.task_type,
                "input_path": task.input_path,
                "input_text": task.input_text,
                "request_output_dir": task.request_output_dir,
                "params": task.params
            }
        return None
    finally:
        session.close()

def update_task_status(request_id, status, error=None, output_files=None):
    """Update task status and output files"""
    session = Session()
    try:
        task = session.query(Task).filter(Task.request_id == request_id).first()
        if task:
            task.status = status
            if status in [TaskStatus.COMPLETE.value, TaskStatus.ERROR.value]:
                task.finish_time = datetime.now()
            
            if error:
                task.error = error
                
            if output_files:
                task.video_filename = output_files.get("video")
                task.model_filename = output_files.get("model")
                task.preview_filename = output_files.get("preview")
                
            session.commit()
    finally:
        session.close()

def get_tasks_for_ip(client_ip):
    """Get recent tasks for an IP"""
    session = Session()
    try:
        tasks = session.query(Task).filter(
            Task.client_ip == client_ip
        ).order_by(desc(Task.created_at)).limit(IP_HISTORY_LIMIT).all()
        return [t.to_dict() for t in tasks]
    finally:
        session.close()

def get_task_by_id(request_id):
    """Get a specific task"""
    session = Session()
    try:
        task = session.query(Task).filter(Task.request_id == request_id).first()
        return task.to_dict() if task else None
    finally:
        session.close()
