import os
import ffmpeg
import requests

def download_remote_video(task_id: str, video_url: str) -> str:
    """
    Download the remote video from 'video_url' and return the local filepath.
    Simple requests-based download. 
    (For large files, consider streaming or more robust solutions.)
    """
    local_path = os.path.join("uploads", f"{task_id}_download.mp4")
    
    # Basic download logic
    r = requests.get(video_url, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path

def crop_and_slow_motion(input_file: str, output_file: str,
                         start_time: str, end_time: str, slow_motion: bool = False):
    """
    Crop the segment from start_time to end_time, then optionally apply slow motion.
    Uses ffmpeg-python to chain filters in one pass if desired.
    """
    # Prepare the input with the specified time range
    stream = ffmpeg.input(input_file, ss=start_time, to=end_time)

    # If slow_motion is True, 2x slower (video and audio)
    if slow_motion:
        # setpts=2*PTS => video is twice as long
        # atempo=0.5 => audio is half speed
        stream = stream.filter_("setpts", "2*PTS").filter_("atempo", "0.5")

    # Output with re-encoded video/audio
    stream = (
        stream
        .output(output_file, c_v="libx264", c_a="aac", vsync="vfr")
        .overwrite_output()
    )
    stream.run()
