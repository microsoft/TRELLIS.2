import enum
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Boolean,
    Float,
    create_engine,
    Enum,
    JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import DB_PATH

Base = declarative_base()

class TaskStatus(enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"

class TaskType(enum.Enum):
    IMAGE_TO_3D = "image_to_3d"
    TEXT_TO_3D = "text_to_3d"

class Task(Base):
    __tablename__ = "tasks"

    request_id = Column(String, primary_key=True)
    task_type = Column(Enum(TaskType), nullable=False)
    status = Column(String, nullable=False)
    client_ip = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    start_time = Column(DateTime)
    finish_time = Column(DateTime)
    
    # Inputs
    input_path = Column(String, nullable=True)      # For image input
    input_text = Column(String, nullable=True)      # For text input
    
    # Outputs
    request_output_dir = Column(String, nullable=False)
    video_filename = Column(String, nullable=True)
    model_filename = Column(String, nullable=True)
    preview_filename = Column(String, nullable=True)
    
    # Error tracking
    error = Column(String, nullable=True)
    worker_pid = Column(Integer, nullable=True)

    # Generation Parameters (stored as JSON for flexibility)
    params = Column(JSON, nullable=True)

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "task_type": self.task_type.value,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "finish_time": self.finish_time.isoformat() if self.finish_time else None,
            "input_text": self.input_text,
            "output_files": self._get_output_files(),
            "error": self.error,
            "worker_pid": self.worker_pid
        }
    
    def _get_output_files(self):
        files = []
        if self.status == TaskStatus.COMPLETE.value:
            if self.video_filename: files.append(self.video_filename)
            if self.model_filename: files.append(self.model_filename)
            if self.preview_filename: files.append(self.preview_filename)
        return files

# Database setup
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()
