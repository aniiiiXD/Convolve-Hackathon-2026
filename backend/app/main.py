from fastapi import FastAPI
from app.api.endpoints import ingest, search
from app.services.qdrant_ops import initialize_collections

app = FastAPI(title="MediSync API", version="1.0")

# Initialize DB on startup
@app.on_event("startup")
async def startup_event():
    initialize_collections()

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])

@app.get("/")
async def root():
    return {"message": "MediSync API is running"}
