from sqlalchemy import Column, String, Integer, DateTime, Float, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
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

    @property
    def user_id(self):
        return self.username


class SearchQuery(Base):
    """Records every search query with metadata for learning"""
    __tablename__ = "search_queries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), index=True)
    clinic_id = Column(String, index=True)
    query_text_hash = Column(String, index=True)  # SHA256 hash for privacy
    query_type = Column(String)  # 'semantic', 'exact', 'hybrid'
    query_intent = Column(String, nullable=True)  # 'diagnosis', 'treatment', 'history'
    result_count = Column(Integer)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    session_id = Column(String, index=True)

    # Relationships
    interactions = relationship("ResultInteraction", back_populates="query", cascade="all, delete-orphan")
    outcomes = relationship("ClinicalOutcome", back_populates="query", cascade="all, delete-orphan")


class ResultInteraction(Base):
    """Tracks which results users clicked/used"""
    __tablename__ = "result_interactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String, ForeignKey("search_queries.id"), index=True)
    result_point_id = Column(String, index=True)  # Qdrant point ID
    result_rank = Column(Integer)  # Position in result list (1-indexed)
    result_score = Column(Float)  # Relevance score from Qdrant
    interaction_type = Column(String)  # 'view', 'click', 'use'
    dwell_time_seconds = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    query = relationship("SearchQuery", back_populates="interactions")


class ClinicalOutcome(Base):
    """Tracks clinical feedback after searches"""
    __tablename__ = "clinical_outcomes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String, ForeignKey("search_queries.id"), index=True)
    patient_id_hash = Column(String, index=True)  # Hashed for privacy
    clinic_id = Column(String, index=True)
    doctor_id = Column(String, ForeignKey("users.id"))
    outcome_type = Column(String)  # 'helpful', 'not_helpful', 'led_to_diagnosis'
    confidence_level = Column(Integer)  # 1-5 scale
    time_to_outcome_hours = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    query = relationship("SearchQuery", back_populates="outcomes")


class ModelTrainingBatch(Base):
    """Tracks data exports for model retraining"""
    __tablename__ = "model_training_batches"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_name = Column(String, unique=True, index=True)
    query_count = Column(Integer)
    date_range_start = Column(DateTime)
    date_range_end = Column(DateTime)
    export_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    training_status = Column(String)  # 'exported', 'training', 'completed', 'failed'
    model_metrics = Column(JSON, nullable=True)  # Store nDCG, MRR, etc.

    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
