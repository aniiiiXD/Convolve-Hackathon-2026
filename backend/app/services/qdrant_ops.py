from qdrant_client import QdrantClient, models
from app.core.database import client

COLLECTION_NAME = "clinical_records"

def initialize_collections():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense_text": models.VectorParams(
                    size=768, 
                    distance=models.Distance.COSINE,
                    # Optimization: Binary Quantization to reduce memory usage 30x
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(
                            always_ram=True
                        )
                    )
                ),
                "image_clip": models.VectorParams(
                    size=512, 
                    distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                "sparse_code": models.SparseVectorParams(),
            }
        )
        print(f"Collection '{COLLECTION_NAME}' created with Hybrid Schema + Quantization.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def get_collection_info():
    return client.get_collection(COLLECTION_NAME)
