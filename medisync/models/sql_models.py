from sqlalchemy import Column, String, Integer, DateTime
from medisync.core.db_sql import Base
import datetime
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    role = Column(String) # 'DOCTOR' or 'PATIENT'
    clinic_id = Column(String, index=True)
    password_hash = Column(String, nullable=True) # For real app, use bcrypt
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
