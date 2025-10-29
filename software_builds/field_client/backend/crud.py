from typing import List, Optional
from sqlalchemy.orm import Session
from .models import FaceEmbedding

def create_face_embedding(
    db: Session,
    person_name: str,
    embedding: List[float],
    image_path: Optional[str] = None,
) -> FaceEmbedding:
    fe = FaceEmbedding(
        person_name=person_name,
        embedding=embedding,
        image_path=image_path,
    )
    db.add(fe)
    db.commit()
    db.refresh(fe)
    return fe

def get_face_embeddings(db: Session, skip: int = 0, limit: int = 50):
    return db.query(FaceEmbedding).offset(skip).limit(limit).all()

def get_face_embedding_by_id(db: Session, id: str) -> Optional[FaceEmbedding]:
    return db.query(FaceEmbedding).filter(FaceEmbedding.id == id).first()