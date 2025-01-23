from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import hashlib
from typing import Dict
import re

app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory to store files
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# In-memory storage for task progress
task_progress: Dict[str, Dict[str, str]] = {}

def get_video_id(url: str) -> str:
    """Generate a unique ID from the YouTube URL using a hash."""
    return hashlib.md5(url.encode()).hexdigest()

def download_video(url: str, output_file: str) -> bool:
    """Download YouTube video using yt-dlp."""
    try:
        subprocess.run(
            ["yt-dlp", "-f", "bestvideo+bestaudio", "--merge-output-format", "mp4", "-o", output_file, url],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to download video: {e}")

def get_video_fps(input_file: str) -> float:
    """Get the FPS of a video using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", input_file,
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        # Extract FPS from ffmpeg output
        fps_match = re.search(r"(\d+(\.\d+)?)\s*fps", result.stderr)
        if fps_match:
            return float(fps_match.group(1))
        else:
            raise ValueError("FPS not found in ffmpeg output")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get video FPS: {e}")

def crop_video(input_file: str, output_file: str, start_time: str, end_time: str) -> bool:
    """Crop video using ffmpeg."""
    try:
        # Delete the output file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)

        subprocess.run(
            [
                "ffmpeg",
                "-i", input_file,
                "-ss", start_time,
                "-to", end_time,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-vsync", "vfr",  # Fix non-monotonic DTS issues
                output_file,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to crop video: {e}")

def apply_slow_motion(input_file: str, output_file: str) -> bool:
    """Apply slow-motion effect to a video using ffmpeg."""
    try:
        # Delete the output file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)

        subprocess.run(
            [
                "ffmpeg",
                "-i", input_file,
                "-vf", "setpts=2*PTS",  # Slow down by 2x
                "-af", "atempo=0.5",    # Slow down audio by 0.5x
                "-c:v", "libx264",
                "-c:a", "aac",
                "-vsync", "vfr",  # Fix non-monotonic DTS issues
                output_file,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply slow motion: {e}")

def merge_videos(input_files: list, output_file: str, fps: float) -> bool:
    """Merge videos using ffmpeg with the original FPS."""
    try:
        # Delete the output file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)

        # Create a text file with list of videos to merge
        with open("input.txt", "w") as f:
            for file in input_files:
                f.write(f"file '{file}'\n")

        subprocess.run(
            [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", "input.txt",
                "-r", str(fps),  # Use the original FPS
                "-c:v", "libx264",  # Re-encode video
                "-c:a", "aac",      # Re-encode audio
                "-vsync", "vfr",    # Fix non-monotonic DTS issues
                output_file,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge videos: {e}")
    finally:
        if os.path.exists("input.txt"):
            os.remove("input.txt")

def process_video_task(task_id: str, url: str, crop_ranges: str, slow_motion: bool):
    """Background task to process the video and update progress."""
    try:
        # Step 1: Download the video
        task_progress[task_id] = {"status": "Downloading video...", "download_progress": 0, "crop_progress": 0, "merge_progress": 0}
        input_file = os.path.join(TEMP_DIR, f"{task_id}.mp4")
        if not os.path.exists(input_file):
            if not download_video(url, input_file):
                task_progress[task_id] = {"status": "Failed to download video"}
                return

        task_progress[task_id]["download_progress"] = 100

        # Step 2: Get the original FPS of the video
        fps = get_video_fps(input_file)

        # Step 3: Crop the video into segments (single-threaded)
        crop_list = crop_ranges.split()
        segments = []
        task_progress[task_id]["status"] = "Cropping segments..."
        task_progress[task_id]["crop_progress"] = 0

        for i, crop_range in enumerate(crop_list):
            start_time, end_time = crop_range.split("-")

            # Create normal-speed segment
            normal_segment = os.path.join(TEMP_DIR, f"{task_id}_segment_{i}_normal.mp4")
            crop_video(input_file, normal_segment, start_time, end_time)
            segments.append(normal_segment)

            # Create slow-motion segment (if enabled)
            if slow_motion:
                slow_segment = os.path.join(TEMP_DIR, f"{task_id}_segment_{i}_slow.mp4")
                apply_slow_motion(normal_segment, slow_segment)
                segments.append(slow_segment)

            # Update progress
            task_progress[task_id]["crop_progress"] = int((i + 1) / len(crop_list) * 100)

        task_progress[task_id]["crop_progress"] = 100

        # Step 4: Merge the segments with the original FPS
        task_progress[task_id]["status"] = "Merging segments..."
        output_file = os.path.join(TEMP_DIR, f"{task_id}_output.mp4")
        if not merge_videos(segments, output_file, fps):
            task_progress[task_id] = {"status": "Failed to merge videos"}
            return

        task_progress[task_id]["merge_progress"] = 100

        # Step 5: Clean up temporary segment files
        for file in segments:
            if os.path.exists(file):
                os.remove(file)

        # Mark task as completed
        task_progress[task_id] = {"status": "Completed", "output_file": output_file}

    except Exception as e:
        task_progress[task_id] = {"status": f"Failed: {str(e)}"}

@app.post("/process-video/")
async def process_video(
    url: str = Form(...),
    crop_ranges: str = Form(...),
    slow_motion: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Start video processing task."""
    task_id = get_video_id(url)
    if task_id in task_progress and task_progress[task_id]["status"] != "Completed":
        return JSONResponse(status_code=400, content={"message": "Task already in progress"})

    # Initialize task progress
    task_progress[task_id] = {"status": "Starting...", "download_progress": 0, "crop_progress": 0, "merge_progress": 0}

    # Add task to background tasks
    background_tasks.add_task(process_video_task, task_id, url, crop_ranges, slow_motion)

    return JSONResponse(content={"task_id": task_id})

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a task."""
    if task_id not in task_progress:
        return JSONResponse(status_code=404, content={"message": "Task not found"})

    return JSONResponse(content=task_progress[task_id])

@app.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    if task_id not in task_progress or task_progress[task_id]["status"] == "Completed":
        return JSONResponse(status_code=404, content={"message": "Task not found or already completed"})

    # Update task status
    task_progress[task_id] = {"status": "Cancelled"}

    return JSONResponse(content={"message": "Task cancelled"})

# Run the backend
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)