import os
import hashlib
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Form, Depends, Query, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import fcntl
from pydantic import BaseModel
import requests
from enum import Enum

# --- Database imports ---
from sqlalchemy import Column, Integer, String, Boolean, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker  # Updated import
from sqlalchemy.sql import text  # Add this import
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool
from sqlalchemy import exc

# --- ffmpeg-python ---
import ffmpeg

# ---------------------------------------------------------------------
# 1. DATABASE SETUP (SQLite)
# ---------------------------------------------------------------------

DATABASE_URL = "sqlite:///./tasks.db"

# Configure the engine with larger pool size and longer timeout
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=QueuePool,
    pool_size=20,  # Increase pool size
    max_overflow=30,  # Increase max overflow
    pool_timeout=60,  # Increase timeout
    pool_pre_ping=True  # Enable connection testing
)

# Add event listeners to handle connection issues
@event.listens_for(engine, 'connect')
def connect(dbapi_connection, connection_record):
    connection_record.info['pid'] = os.getpid()

@event.listens_for(engine, 'checkout')
def checkout(dbapi_connection, connection_record, connection_proxy):
    pid = os.getpid()
    if connection_record.info['pid'] != pid:
        connection_record.connection = connection_proxy.connection = None
        raise exc.DisconnectionError(
            "Connection record belongs to pid %s, "
            "attempting to check out in pid %s" %
            (connection_record.info['pid'], pid)
        )

# Update the session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Prevent expired object issues
)

Base = declarative_base()  # Updated usage

class VideoTask(Base):
    """SQLAlchemy model to store tasks."""
    __tablename__ = "videotasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    url = Column(String)
    status = Column(String)
    progress = Column(Integer)
    output_file = Column(String, nullable=True)
    audio_file = Column(String, nullable=True)
    slow_motion = Column(Boolean, default=False)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    completed_at = Column(String, nullable=True)

# Create tables if not exist
Base.metadata.create_all(bind=engine)

# Increase SQLite query limit
with engine.connect() as connection:
    connection.execute(text("PRAGMA journal_mode=WAL;"))  # Enable Write-Ahead Logging
    connection.execute(text("PRAGMA cache_size=-10000;"))  # Increase cache size
    connection.execute(text("PRAGMA busy_timeout=5000;"))  # Increase busy timeout
    connection.commit()  # Commit the changes

# ---------------------------------------------------------------------
# 2. FASTAPI APPLICATION SETUP
# ---------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets/audios", StaticFiles(directory="assets/audios"), name="audio_files")

# Make sure required directories exist
os.makedirs("inputs", exist_ok=True)
os.makedirs("segments", exist_ok=True)
os.makedirs("audios", exist_ok=True)  # Directory for uploaded audios
os.makedirs("locks", exist_ok=True)  # Directory for file-based locks
os.makedirs("assets/audios", exist_ok=True)  # Directory for static audio files

# Copy your test audio file to assets/audios if it doesn't exist
test_audio_file = "assets/audios/lost-in-dreams-abstract-chill-downtempo-cinematic-future-beats-270241.mp3"
if not os.path.exists(test_audio_file):
    # Create a placeholder file or copy from your source
    try:
        import shutil
        source_file = "path/to/your/source/audio.mp3"  # Update this path
        if os.path.exists(source_file):
            shutil.copy2(source_file, test_audio_file)
    except Exception as e:
        print(f"Warning: Could not setup test audio file: {e}")

# ---------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def get_db_session():
    """Helper to get a new SQLAlchemy session."""
    db = SessionLocal()
    try:
        # Test the connection
        db.execute(text("SELECT 1"))
        yield db
    except Exception as e:
        print(f"Database connection error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def get_video_id(url: str) -> str:
    """Generate a unique ID from the YouTube URL using an MD5 hash."""
    return hashlib.md5(url.encode()).hexdigest()

def get_unique_task_id(url: str) -> str:
    """Generate a unique task ID using the URL and a UUID."""
    return f"{hashlib.md5(url.encode()).hexdigest()}_{uuid.uuid4().hex}"

class FileLock:
    """Simple file-based lock."""
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.fd = None

    def acquire(self):
        """Acquire the lock."""
        self.fd = open(self.lock_file, "w")
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def release(self):
        """Release the lock."""
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None

def get_video_lock(video_id: str):
    """Get a lock for a specific video."""
    lock_file = os.path.join("locks", f"{video_id}.lock")
    os.makedirs("locks", exist_ok=True)
    return FileLock(lock_file)


def download_video(url: str, output_path: str) -> None:
    """
    Download YouTube video using yt-dlp into `output_path`.
    Raises HTTPException on failure.
    """
    import subprocess
    try:
        subprocess.run(
            ["yt-dlp", "--cookies", "./cookies.txt", "-f", "bestvideo+bestaudio", "--merge-output-format", "mp4", "-o", output_path, url],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to download video: {e}")

def get_video_fps(input_file: str) -> float:
    """
    Retrieve the FPS of a video using ffmpeg-python (probe).
    """
    try:
        probe = ffmpeg.probe(input_file)
        video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
        if not video_streams:
            raise ValueError("No video stream found.")
        fps_str = video_streams[0].get("r_frame_rate", "0/0")
        num, den = fps_str.split("/")
        return float(num) / float(den) if float(den) != 0 else 30.0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get video FPS: {e}")

def crop_video(input_file: str, output_file: str, start_time: str, end_time: str) -> None:
    """
    Crop the video (no forced FPS).
    We'll unify FPS at final merge.
    """
    if os.path.exists(output_file):
        os.remove(output_file)
    try:
        (
            ffmpeg
            .input(input_file, ss=start_time, to=end_time)
            .output(
                output_file,
                vcodec="libx264",
                acodec="aac",
                vsync="vfr"
            )
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to crop video: {e.stderr.decode()}")

def apply_slow_motion(input_file: str, output_file: str) -> None:
    """
    Apply slow-motion (2x) with setpts=2.0*PTS for video
    and atempo=0.5 for audio. No forced FPS yet.
    """
    if os.path.exists(output_file):
        os.remove(output_file)
    try:
        inp = ffmpeg.input(input_file)
        # Double the length (slower): setpts=2.0*PTS, audio: atempo=0.5
        v = inp.video.filter_("setpts", "2.0*PTS")
        a = inp.audio.filter_("atempo", 0.5)

        (
            ffmpeg
            .output(
                v,
                a,
                output_file,
                vcodec="libx264",
                acodec="aac",
                vsync="vfr"
            )
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply slow motion: {e.stderr.decode()}")

def merge_videos_with_original_fps(
    segment_paths: List[str],
    original_reference_file: str,
    output_file: str,
) -> None:
    """
    Merge segments using a filter-complex approach:
      - For each input segment [i], apply `fps=<orig_fps>` to the video
        so all segments match the original video's FPS.
      - Concatenate the video/audio streams.
    """
    if os.path.exists(output_file):
        os.remove(output_file)

    # 1) Determine the original video's FPS
    original_fps = get_video_fps(original_reference_file)

    # 2) For each segment, create an input and filter the video to original_fps
    video_audio_pairs = []
    for seg in segment_paths:
        inp = ffmpeg.input(seg)
        filtered_vid = inp.video.filter("fps", fps=original_fps)
        # keep the audio as-is
        aud = inp.audio
        video_audio_pairs.append((filtered_vid, aud))

    # Flatten (v, a) pairs for concat
    flatten_streams = []
    for (v, a) in video_audio_pairs:
        flatten_streams.extend([v, a])

    # The concat filter: n = number of segments
    n_segments = len(segment_paths)
    joined = ffmpeg.concat(*flatten_streams, v=1, a=1)

    # 3) Map them to final output
    out = (
        ffmpeg
        .output(
            joined,
            output_file,
            vcodec="libx264",
            acodec="aac",
            vsync="vfr",
        )
    )

    # 4) Execute the merge
    out.run(quiet=True)

def replace_audio(output_video: str, new_audio: str, final_output: str) -> None:
    """
    Replace the audio of the output video with the new audio.
    If the new audio is shorter, loop it to match the video length.
    """
    try:
        # Get video duration
        probe = ffmpeg.probe(output_video)
        video_duration = float(probe['format']['duration'])

        # Get audio duration
        probe = ffmpeg.probe(new_audio)
        audio_duration = float(probe['format']['duration'])

        # Calculate number of loops needed
        loops = int(video_duration // audio_duration) + 1

        # Create a temporary file with the looped audio
        looped_audio = os.path.join("audios", f"looped_{uuid.uuid4().hex}.m4a")  # Use .m4a for AAC
        os.makedirs("audios", exist_ok=True)

        # Loop the audio to match the video duration
        (
            ffmpeg
            .input(new_audio, stream_loop=loops)
            .output(looped_audio, t=video_duration, acodec="aac")
            .overwrite_output()
            .run(quiet=True)
        )

        # Replace audio in the video by mapping streams correctly
        video_input = ffmpeg.input(output_video)
        audio_input = ffmpeg.input(looped_audio)

        (
            ffmpeg
            .output(video_input.video, audio_input.audio, final_output, vcodec='copy', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )

        # Clean up temporary looped audio
        os.remove(looped_audio)

    except ffmpeg.Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to replace audio: {e.stderr.decode()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during audio replacement: {str(e)}")

# ---------------------------------------------------------------------
# 4. BACKGROUND TASK FOR VIDEO PROCESSING
# ---------------------------------------------------------------------
def process_video_task(
    task_id: str, 
    url: str, 
    crop_ranges: str, 
    audio_url: Optional[str] = None, 
    slow_motion: bool = False
):
    db = SessionLocal()
    try:
        task = db.query(VideoTask).filter(VideoTask.task_id == task_id).first()
        if not task:
            print(f"Task {task_id} not found")
            return

        try:
            # Update initial status
            task.status = TaskStatus.DOWNLOADING_INPUT_VIDEO.value
            task.progress = 10
            db.commit()

            # Process video
            video_id = get_video_id(url)
            input_path = os.path.join("inputs", f"{video_id}.mp4")

            if not os.path.exists(input_path):
                download_video(url, input_path)

            task.status = TaskStatus.PROCESSING_SEGMENTS.value
            task.progress = 20
            db.commit()

            # Process segments
            segments = crop_ranges.split()
            segment_paths = []
            segments_dir = os.path.join("segments", video_id)
            os.makedirs(segments_dir, exist_ok=True)

            for i, crange in enumerate(segments):
                start_time, end_time = crange.split("-")
                normal_seg = os.path.join(
                    segments_dir, f"{start_time}_{end_time}_normal.mp4"
                )

                crop_video(input_path, normal_seg, start_time, end_time)

                segment_paths.append(normal_seg)

                if slow_motion:
                    slow_seg = os.path.join(
                        segments_dir, f"{start_time}_{end_time}_slow.mp4"
                    )
                    apply_slow_motion(normal_seg, slow_seg)
                    segment_paths.append(slow_seg)

                # Update progress (roughly from 20% -> 70%)
                task.progress = int(20 + (i + 1) / len(segments) * 50)
                db.commit()

            # Merge segments
            task.status = TaskStatus.PROCESSING_OUTPUT.value
            db.commit()

            merged_output = os.path.join("segments", video_id, f"{video_id}_merged_output.mp4")
            merge_videos_with_original_fps(segment_paths, input_path, merged_output)

            # Handle audio if provided
            if audio_url:
                # Download and process audio
                audio_path = download_audio_from_url(audio_url)
                final_output = os.path.join("segments", video_id, f"{video_id}_final_output.mp4")
                replace_audio(merged_output, audio_path, final_output)
                os.remove(merged_output)  # Remove the merged output without new audio
                task.audio_file = audio_path  # Store audio file path
            else:
                final_output = merged_output

            # Update final status with completion time
            task.status = TaskStatus.COMPLETED.value
            task.progress = 100
            task.output_file = final_output
            task.completed_at = datetime.utcnow().isoformat()
            db.commit()

        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            task.status = TaskStatus.FAILED.value
            task.progress = 0
            task.completed_at = datetime.utcnow().isoformat()
            db.commit()
            raise

    except Exception as e:
        print(f"Error in process_video_task: {e}")
        if db:
            try:
                task = db.query(VideoTask).filter(VideoTask.task_id == task_id).first()
                if task:
                    task.status = TaskStatus.FAILED.value
                    task.progress = 0
                    task.completed_at = datetime.utcnow().isoformat()
                    db.commit()
            except:
                db.rollback()
        raise
    finally:
        if db:
            db.close()

# Add helper function for audio download
def download_audio_from_url(audio_url: str) -> str:
    try:
        # Remove leading slash if present
        if audio_url.startswith('/'):
            audio_url = audio_url[1:]
        
        # Check if file exists locally
        if os.path.exists(audio_url):
            return audio_url
            
        # If it's a remote URL, download it
        if audio_url.startswith(('http://', 'https://')):
            filename = os.path.basename(audio_url)
            local_path = os.path.join("audios", filename)
            
            response = requests.get(audio_url)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            return local_path
            
        return audio_url
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        raise

# ---------------------------------------------------------------------
# 5. API ENDPOINTS
# ---------------------------------------------------------------------
class VideoProcessRequest(BaseModel):
    url: str
    crop_ranges: str
    audio_url: Optional[str] = None
    slow_motion: bool = False
    user_id: str  # Add this field

# Add TaskStatus enum at the top of the file
class TaskStatus(str, Enum):
    DOWNLOADING_INPUT_VIDEO = 'DOWNLOADING_INPUT_VIDEO'
    PROCESSING_SEGMENTS = 'PROCESSING_SEGMENTS'
    PROCESSING_OUTPUT = 'PROCESSING_OUTPUT'
    FAILED = 'FAILED'
    COMPLETED = 'COMPLETED'

@app.post("/process-video")
async def process_video(
    request: VideoProcessRequest,
    background_tasks: BackgroundTasks
):
    try:
        task_id = str(uuid.uuid4())
        
        db = SessionLocal()
        try:
            task = VideoTask(
                task_id=task_id,
                user_id=request.user_id,
                url=request.url,
                status=TaskStatus.DOWNLOADING_INPUT_VIDEO.value,  # Use constant
                progress=0,
                audio_file=request.audio_url,
                slow_motion=request.slow_motion
            )
            db.add(task)
            db.commit()
            db.refresh(task)
        finally:
            db.close()

        background_tasks.add_task(
            process_video_task,
            task_id=task_id,
            url=request.url,
            crop_ranges=request.crop_ranges,
            audio_url=request.audio_url,
            slow_motion=request.slow_motion
        )

        return {"task_id": task_id}
    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process video: {str(e)}"
        )

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str, user_id: str = Query(...)):
    """
    Retrieve the status and progress of a specific task.
    - `task_id`: Unique identifier for the task
    - `user_id`: Unique identifier for the user
    """
    db = next(get_db_session())
    task_obj = db.query(VideoTask).filter(
        VideoTask.task_id == task_id,
        VideoTask.user_id == user_id  # Ensure task belongs to user
    ).first()
    if not task_obj:
        return JSONResponse(status_code=404, content={"message": "Task not found"})

    return {
        "status": task_obj.status,
        "task_id": task_id,
        "progress": task_obj.progress,
        "output_file": task_obj.output_file,
    }

@app.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str, user_id: str = Query(...)):
    """
    Cancel a specific task.
    - `task_id`: Unique identifier for the task
    - `user_id`: Unique identifier for the user
    """
    db = next(get_db_session())
    task_obj = db.query(VideoTask).filter(
        VideoTask.task_id == task_id,
        VideoTask.user_id == user_id
    ).first()
    if not task_obj or task_obj.status == TaskStatus.COMPLETED.value:
        return JSONResponse(status_code=404, content={"message": "Task not found or already completed"})

    task_obj.status = TaskStatus.FAILED.value
    task_obj.completed_at = datetime.utcnow().isoformat()
    db.commit()
    return JSONResponse({"message": "Task cancelled"})

@app.get("/download/{task_id}")
async def download_video_output(task_id: str, user_id: str = Query(...)):
    """
    Endpoint to download processed video.
    - `task_id`: Unique identifier for the task
    - `user_id`: Unique identifier for the user
    """
    db = next(get_db_session())
    task = db.query(VideoTask).filter(
        VideoTask.task_id == task_id,
        VideoTask.user_id == user_id
    ).first()
    if not task or task.status != TaskStatus.COMPLETED.value or not task.output_file:
        raise HTTPException(status_code=404, detail=task.output_file if task else "Task not found")
    if not os.path.exists(task.output_file):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        task.output_file,
        filename=os.path.basename(task.output_file),
        media_type="video/mp4"
    )

@app.get("/tasks")
async def get_all_tasks(user_id: str = Query(...)):
    """
    Retrieve all tasks associated with a user.
    - `user_id`: Unique identifier for the user
    """
    try:
        db = SessionLocal()
        tasks = db.query(VideoTask)\
            .filter(VideoTask.user_id == user_id)\
            .order_by(VideoTask.created_at.desc())\
            .all()
        return [
            {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "output_file": task.output_file,
                "created_at": task.created_at,
                "completed_at": task.completed_at
            }
            for task in tasks
        ]
    except Exception as e:
        print(f"Error getting tasks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tasks: {str(e)}"
        )
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Frontend template missing")

@app.get('/api/audio-options')
async def get_audio_options():
    """Get available audio options."""
    try:
        audio_folder = 'assets/audios'
        audio_files = []
        
        # Ensure the folder exists
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)
        
        # Get all audio files
        for file in os.listdir(audio_folder):
            if file.lower().endswith(('.mp3', '.wav', '.m4a')):
                file_path = os.path.join(audio_folder, file)
                try:
                    # Get file size
                    size = os.path.getsize(file_path)
                    # Format name
                    name = os.path.splitext(file)[0].replace('_', ' ').title()
                    
                    audio_files.append({
                        'file': f'/assets/audios/{file}',
                        'name': name,
                        'size': size,
                        'duration': '1:30'  # You can add actual duration if needed
                    })
                except Exception as e:
                    print(f"Error processing audio file {file}: {e}")
                    continue
        
        return audio_files
    except Exception as e:
        print(f"Error getting audio options: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get audio options: {str(e)}"
        )

# ---------------------------------------------------------------------
# 6. RUN THE APP
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
