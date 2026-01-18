from fastapi import APIRouter, HTTPException
from app.models.api import SearchRequest, SearchResponse
from app.services.embedding import EmbeddingService
from app.services.qdrant_ops import COLLECTION_NAME
from app.core.database import client
from qdrant_client import models

router = APIRouter()

@router.post("/search", response_model=list[SearchResponse])
async def search_clinical_records(request: SearchRequest):
    """
    Performs Hybrid Search (Dense + Sparse) with RRF Fusion.
    Enforces Multitenancy via clinic_id.
    """
    try:
        # 1. Embed Query
        dense_query = EmbeddingService.embed_dense(request.query_text)
        sparse_query = EmbeddingService.embed_sparse(request.query_text)

        # 2. Build Prefetch Requests (Dense & Sparse)
        prefetch = [
            models.Prefetch(
                query=dense_query,
                using="dense_text",
                limit=request.limit * 2, # Fetch more for re-ranking
                filter=models.Filter(
                    must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=request.clinic_id))]
                )
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"]
                ),
                using="sparse_code",
                limit=request.limit * 2,
                filter=models.Filter(
                    must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=request.clinic_id))]
                )
            ),
        ]

        # 3. Execute Fusion Query (RRF)
        # We query points using the prefetch results and fuse them
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=request.limit
        )

        # 4. Format Response
        response = []
        for point in results.points:
            response.append(SearchResponse(
                score=point.score,
                text_content=point.payload.get("text_content"),
                metadata=point.payload
            ))
            
        return response

    except Exception as e:
        # Graceful error handling
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
