import os
import hashlib
import re

from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Database imports ---
from sqlalchemy import Column, Integer, String, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- ffmpeg-python ---
import ffmpeg

# ---------------------------------------------------------------------
# 1. DATABASE SETUP
# ---------------------------------------------------------------------

DATABASE_URL = "sqlite:///./tasks.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class VideoTask(Base):
    """SQLAlchemy model to store tasks."""
    __tablename__ = "videotasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)  # e.g. MD5 hash
    url = Column(String)
    status = Column(String)
    progress = Column(Integer)
    output_file = Column(String, nullable=True)
    slow_motion = Column(Boolean, default=False)
    override_segments = Column(Boolean, default=False)

# Create tables if not exist
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------
# 2. FASTAPI APPLICATION SETUP
# ---------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Make sure required directories exist
os.makedirs("inputs", exist_ok=True)
os.makedirs("segments", exist_ok=True)

# ---------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def get_db_session():
    """Helper to get a new SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_video_id(url: str) -> str:
    """Generate a unique ID from the YouTube URL using an MD5 hash."""
    return hashlib.md5(url.encode()).hexdigest()

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
    This mimics a manual filter_complex:
      [0:v]fps=orig_fps[v0]; [1:v]fps=orig_fps[v1]; ... concat=...
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
            # Optionally add r=original_fps if you want the final container 
            # to explicitly declare that exact FPS:
            # r=original_fps
        )
    )

    # 4) Execute the merge
    out.run(quiet=True)

# ---------------------------------------------------------------------
# 4. BACKGROUND TASK FOR VIDEO PROCESSING
# ---------------------------------------------------------------------
def process_video_task(
    task_id: str, url: str, crop_ranges: str, slow_motion: bool, override_segments: bool
):
    """
    Steps:
      1) Download the original video (if missing).
      2) Crop or slow-mo segments individually (no forced FPS).
      3) Merge them with the original FPS, replicating the manual
         filter_complex approach that sets fps for each segment's video.
    """
    db = SessionLocal()
    try:
        task_obj = db.query(VideoTask).filter(VideoTask.task_id == task_id).first()
        if not task_obj:
            return

        # 1) Download
        task_obj.status = "Downloading video"
        task_obj.progress = 0
        db.commit()

        video_id = task_id
        input_video_path = os.path.join("inputs", f"{video_id}.mp4")

        if not os.path.exists(input_video_path):
            download_video(url, input_video_path)

        task_obj.status = "Video downloaded"
        task_obj.progress = 20
        db.commit()

        # 2) Crop / Slow-mo
        task_obj.status = "Cropping segments"
        db.commit()

        crop_list = crop_ranges.split()
        segment_paths = []
        segments_dir = os.path.join("segments", video_id)
        os.makedirs(segments_dir, exist_ok=True)

        for i, crange in enumerate(crop_list):
            start_time, end_time = crange.split("-")
            normal_seg = os.path.join(
                segments_dir, f"{start_time}_{end_time}_normal.mp4"
            )

            if override_segments or not os.path.exists(normal_seg):
                crop_video(input_video_path, normal_seg, start_time, end_time)

            segment_paths.append(normal_seg)

            if slow_motion:
                slow_seg = os.path.join(
                    segments_dir, f"{start_time}_{end_time}_slow.mp4"
                )
                if override_segments or not os.path.exists(slow_seg):
                    apply_slow_motion(normal_seg, slow_seg)
                segment_paths.append(slow_seg)

            # Update progress (roughly from 20% -> 70%)
            task_obj.progress = int(20 + (i + 1) / len(crop_list) * 50)
            db.commit()

        # 3) Merge using the original video's FPS
        task_obj.status = "Merging segments"
        db.commit()

        output_file = os.path.join("segments", video_id, f"{video_id}_merged_output.mp4")
        merge_videos_with_original_fps(segment_paths, input_video_path, output_file)

        task_obj.status = "Completed"
        task_obj.progress = 100
        task_obj.output_file = output_file
        db.commit()

    except Exception as e:
        task_obj.status = f"Failed: {str(e)}"
        db.commit()
    finally:
        db.close()

# ---------------------------------------------------------------------
# 5. API ENDPOINTS
# ---------------------------------------------------------------------
@app.post("/process-video")
async def process_video(
    url: str = Form(...),
    crop_ranges: str = Form(...),
    slow_motion: bool = Form(False),
    override_segments: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Start video processing task.
    - `override_segments`: if True, re-crops existing segments
    - `slow_motion`: if True, also produce a slow-motion copy of each segment
    """
    video_id = get_video_id(url)
    db = next(get_db_session())
    task_obj = db.query(VideoTask).filter(VideoTask.task_id == video_id).first()

    if task_obj and task_obj.status != "Completed":
        return JSONResponse(
            status_code=400,
            content={"message": "A task for this video is already in progress"}
        )

    if not task_obj:
        task_obj = VideoTask(
            task_id=video_id,
            url=url,
            status="Created",
            progress=0,
            slow_motion=slow_motion,
            override_segments=override_segments
        )
        db.add(task_obj)
        db.commit()
        db.refresh(task_obj)

    background_tasks.add_task(
        process_video_task,
        video_id, url, crop_ranges, slow_motion, override_segments
    )
    return JSONResponse({"task_id": video_id, "message": "Task started"})

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    db = next(get_db_session())
    task_obj = db.query(VideoTask).filter(VideoTask.task_id == task_id).first()
    if not task_obj:
        return JSONResponse(status_code=404, content={"message": "Task not found"})

    return {
        "status": task_obj.status,
        "progress": task_obj.progress,
        "output_file": task_obj.output_file,
    }

@app.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str):
    """
    Demonstration cancel endpoint. 
    Doesn't forcibly kill ffmpeg in a background thread,
    only updates DB status to 'Cancelled'.
    """
    db = next(get_db_session())
    task_obj = db.query(VideoTask).filter(VideoTask.task_id == task_id).first()
    if not task_obj or task_obj.status == "Completed":
        return JSONResponse(status_code=404, content={"message": "Task not found or already completed"})

    task_obj.status = "Cancelled"
    db.commit()
    return JSONResponse({"message": "Task cancelled"})

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Frontend template missing")

@app.get("/download/{task_id}")
async def download_video_output(task_id: str):
    """Endpoint to download processed video."""
    db = next(get_db_session())
    task = db.query(VideoTask).filter(VideoTask.task_id == task_id).first()
    if not task or task.status != "Completed" or not task.output_file:
        raise HTTPException(status_code=404, detail="File not found")
    if not os.path.exists(task.output_file):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        task.output_file,
        filename=os.path.basename(task.output_file),
        media_type="video/mp4"
    )

# ---------------------------------------------------------------------
# 6. RUN THE APP
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
