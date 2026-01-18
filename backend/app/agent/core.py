from app.services.embedding import EmbeddingService
from app.services.qdrant_ops import COLLECTION_NAME
from app.core.database import client
from qdrant_client import models
import uuid
import time

class ClinicalAgent:
    def __init__(self, clinic_id: str, doctor_id: str):
        self.clinic_id = clinic_id
        self.doctor_id = doctor_id
        # Simple InMemory Memory of conversation
        self.stm = [] 

    def ingest_note(self, patient_id: str, text: str) -> str:
        """
        'Perception' phase: Encodes and stores a clinical observation.
        """
        # 1. Encode
        dense_vec = EmbeddingService.embed_dense(text)
        sparse_vec = EmbeddingService.embed_sparse(text)
        
        # 2. Store
        point_id = str(uuid.uuid4())
        payload = {
            "patient_id": patient_id,
            "clinic_id": self.clinic_id,
            "doctor_id": self.doctor_id,
            "text_content": text,
            "timestamp": time.time(),
            "type": "note"
        }

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense_text": dense_vec,
                        "sparse_code": models.SparseVector(
                            indices=sparse_vec["indices"], 
                            values=sparse_vec["values"]
                        )
                    },
                    payload=payload
                )
            ]
        )
        return point_id

    def ingest_image(self, patient_id: str, image_path: str) -> str:
        """
        'Perception' phase (Vision): Encodes and stores a medical image (X-ray, MRI).
        """
        try:
            # 1. Encode
            # If real image encoding fails (missing lib), we mock it for the demo
            image_vec = EmbeddingService.embed_image(image_path)
        except Exception as e:
            print(f"Warning: Image embedding failed ({e}). Using mock vector for detailed architecture test.")
            import numpy as np
            # Mock 512d CLIP vector
            image_vec = np.random.rand(512).tolist()

        # 2. Store
        point_id = str(uuid.uuid4())
        payload = {
            "patient_id": patient_id,
            "clinic_id": self.clinic_id,
            "doctor_id": self.doctor_id,
            "image_path": image_path,
            "timestamp": time.time(),
            "type": "image",
            "modality": "vision"
        }

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "image_clip": image_vec, # Named vector for vision
                        # We might also strictly require text vectors, but Qdrant allows sparse vectors.
                        # For now, we leave other named vectors empty or use placeholders if specific queries require them.
                    },
                    payload=payload
                )
            ]
        )
        return point_id

    def recall(self, query: str, limit: int = 5):
        """
        'Recall' phase: Hybrid RRF Search.
        """
        dense_q = EmbeddingService.embed_dense(query)
        sparse_q = EmbeddingService.embed_sparse(query)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_q, using="dense_text", limit=limit*2,
                    filter=models.Filter(must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id))])
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_q["indices"], values=sparse_q["values"]),
                    using="sparse_code", limit=limit*2,
                    filter=models.Filter(must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id))])
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        return results.points

    def reason(self, query: str, context_points) -> str:
        """
        'Reasoning' phase: Synthesize an answer.
        (MOCKED for Hackathon Speed - Placeholder for LLM)
        """
        context_text = "\n".join([f"- {p.payload['text_content']}" for p in context_points])
        
        # Mock Reasoning Logic
        return f"""
Based on the clinical history I've retrieved:

{context_text}

**Assessment:**
The patient seems to be exhibiting symptoms consistent with the query '{query}'. 
Recommended proceeding with standard diagnostic protocols.
(Note: This is an AI-generated synthesis).
"""
