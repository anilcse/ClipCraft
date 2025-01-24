from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uuid
import os

from .database import init_db, get_db
from .models import VideoTask, TaskStatus
from .tasks import process_video_task
from sqlalchemy.orm import Session

app = FastAPI(title="Clipcraft")

# CORS (if you have a frontend or want external access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """
    Initialize the database on application startup.
    """
    init_db()

@app.post("/process-video/")
async def process_video(
    db: Session = get_db(),
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    start_time: str = Form("00:00:00"),  # e.g. "00:00:10"
    end_time: str = Form("00:00:10"),
    slow_motion: bool = Form(False)
):
    """
    Start a video processing task. Either a file upload OR a URL can be provided.
    The task will create a clipped segment from start_time to end_time and optionally apply slow-motion.
    """
    if not file and not video_url:
        raise HTTPException(status_code=400, detail="Must provide either a file or a video_url")

    # Create a unique task ID
    task_id = str(uuid.uuid4())

    # Save file if provided
    input_filepath = None
    if file:
        input_filepath = os.path.join("uploads", f"{task_id}_{file.filename}")
        with open(input_filepath, "wb") as buffer:
            buffer.write(await file.read())

    # Create a DB record
    video_task = VideoTask(
        id=task_id,
        status=TaskStatus.STARTING
    )
    db.add(video_task)
    db.commit()

    # Trigger Celery background task
    process_video_task.delay(
        task_id=task_id,
        local_file=input_filepath,
        video_url=video_url,
        start_time=start_time,
        end_time=end_time,
        slow_motion=slow_motion
    )

    return {"task_id": task_id, "message": "Video processing started."}

@app.get("/task-status/{task_id}")
def get_task_status(task_id: str, db: Session = get_db()):
    """
    Check the status of the background task in the database.
    """
    video_task = db.query(VideoTask).filter_by(id=task_id).first()
    if not video_task:
        raise HTTPException(status_code=404, detail="Task not found.")
    
    return {
        "task_id": video_task.id,
        "status": video_task.status,
        "progress": video_task.progress,
        "detail": video_task.detail,
        "output_file": video_task.output_file
    }

@app.get("/download/{task_id}")
def download_result(task_id: str, db: Session = get_db()):
    """
    Download the final output video if the task is completed.
    """
    video_task = db.query(VideoTask).filter_by(id=task_id).first()
    if not video_task or video_task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="Task not found or not completed yet.")

    if not video_task.output_file or not os.path.exists(video_task.output_file):
        raise HTTPException(status_code=404, detail="Output file not found.")
    
    filename = os.path.basename(video_task.output_file)
    return FileResponse(video_task.output_file, media_type="video/mp4", filename=filename)
