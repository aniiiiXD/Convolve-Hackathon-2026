from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# Use default local postgres URL if not set
# Defaulting to SQLite for robust local testing without auth issues
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medisync.db")

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Helper to create tables
    Base.metadata.create_all(bind=engine)
