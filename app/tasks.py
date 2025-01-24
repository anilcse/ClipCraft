import os
from celery import shared_task
from sqlalchemy.orm import Session

from .models import VideoTask, TaskStatus
from .database import SessionLocal
from .utils import download_remote_video, crop_and_slow_motion

@shared_task
def process_video_task(task_id: str, local_file: str, video_url: str,
                       start_time: str, end_time: str, slow_motion: bool):
    """
    Celery task to process a video. Either uses a local file or downloads from a URL.
    """
    db = SessionLocal()

    # 1. Load task
    video_task = db.query(VideoTask).filter_by(id=task_id).first()
    if not video_task:
        db.close()
        return

    # 2. Update status to DOWNLOADING or PROCESSING
    video_task.status = TaskStatus.DOWNLOADING if video_url else TaskStatus.PROCESSING
    db.commit()

    # 3. If a video_url is provided, download the video
    input_filepath = local_file
    if video_url:
        try:
            input_filepath = download_remote_video(task_id, video_url)
        except Exception as e:
            video_task.status = TaskStatus.FAILED
            video_task.detail = str(e)
            db.commit()
            db.close()
            return

    # 4. Actually do the cropping (and slow motion)
    video_task.status = TaskStatus.PROCESSING
    db.commit()

    output_filepath = os.path.join("uploads", f"{task_id}_output.mp4")
    try:
        crop_and_slow_motion(input_filepath, output_filepath, start_time, end_time, slow_motion)
    except Exception as e:
        video_task.status = TaskStatus.FAILED
        video_task.detail = str(e)
        db.commit()
        db.close()
        return

    # 5. Mark completed
    video_task.status = TaskStatus.COMPLETED
    video_task.progress = 100
    video_task.output_file = output_filepath
    db.commit()
    db.close()

    # (Optional) Clean up local_file if it was downloaded or not needed
    # os.remove(input_filepath)  # If you want to remove original files after processing
