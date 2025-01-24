from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from .models import Base

DATABASE_URL = "sqlite:///./clipcraft.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Create database tables if they don't exist.
    """
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_db():
    """
    Dependency for FastAPI routes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
