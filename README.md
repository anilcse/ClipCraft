# Clipcraft

Clipcraft is a **generic** video processing backend built with FastAPI, Celery, and FFmpeg. 

## Features

- Upload or provide URL for a video.
- Extract a segment (start_time → end_time).
- Optionally apply slow-motion (2× slower).
- Background task processing with Celery.
- Download final processed video.

## Quickstart (Local)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
