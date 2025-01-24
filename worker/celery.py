# clipcraft/worker/celery.py

from celery import Celery

celery_app = Celery(
    "clipcraft",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Automatic discovery of tasks in our 'app.tasks' module
celery_app.autodiscover_tasks(["clipcraft.app"])
