from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.services.embedding import EmbeddingService
from app.services.qdrant_ops import COLLECTION_NAME
from app.core.database import client
from qdrant_client import models
import uuid
import json

router = APIRouter()

@router.post("/ingest")
async def ingest_clinical_record(
    patient_id: str = Form(...),
    clinic_id: str = Form(...),
    text_content: str = Form(...),
    # file: UploadFile = File(None) # Future: Handle file uploads for images
):
    """
    Ingests a clinical record (text) into Qdrant with Hybrid Embeddings.
    """
    try:
        # 1. Generate Embeddings (Multi-Vector)
        dense_vector = EmbeddingService.embed_dense(text_content)
        sparse_vector = EmbeddingService.embed_sparse(text_content)
        
        # 2. Prepare Point ID and Payload
        point_id = str(uuid.uuid4())
        payload = {
            "patient_id": patient_id,
            "clinic_id": clinic_id,
            "text_content": text_content,
            "data_type": "clinical_note",
            "modality": "text"
        }

        # 3. Upsert to Qdrant (Named Vectors)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense_text": dense_vector,
                        "sparse_code": models.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"]
                        )
                    },
                    payload=payload
                )
            ]
        )
        
        return {"status": "indexed", "point_id": point_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
