# clipcraft/Dockerfile

FROM python:3.10-slim

# Install ffmpeg (adjust if needed)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Create a directory for the app
WORKDIR /clipcraft

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose FastAPI on port 8000
EXPOSE 8000

# Run FastAPI via Uvicorn (use a process manager like Gunicorn in production)
CMD ["uvicorn", "clipcraft.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
