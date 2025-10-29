import uuid
from sqlalchemy import Column, String, DateTime, func, JSON
from .db import Base

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    person_name = Column(String, nullable=False)
    # Store embedding as JSON array of floats
    embedding = Column(JSON, nullable=False)
    image_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())