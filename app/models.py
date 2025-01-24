from sqlalchemy import Column, String, Integer, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum as PyEnum

Base = declarative_base()

class TaskStatus(str, PyEnum):
    STARTING = "starting"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VideoTask(Base):
    __tablename__ = "video_tasks"

    id = Column(String, primary_key=True, index=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.STARTING)
    progress = Column(Integer, default=0)
    detail = Column(Text, nullable=True)
    output_file = Column(String, nullable=True)
