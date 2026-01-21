from qdrant_client import QdrantClient, models
from medisync.core_agents.database_agent import client

COLLECTION_NAME = "clinical_records"
FEEDBACK_COLLECTION = "feedback_analytics"
GLOBAL_INSIGHTS_COLLECTION = "global_medical_insights"

def initialize_collections():
    """Initialize all Qdrant collections"""
    _initialize_clinical_records()
    _initialize_feedback_analytics()
    _initialize_global_insights()


def _initialize_clinical_records():
    """Initialize main clinical records collection"""
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

    # Payload Indexes for efficient filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="clinic_id",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="patient_id",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print(f"Payload indexes ensured for '{COLLECTION_NAME}'.")


def _initialize_feedback_analytics():
    """Initialize feedback analytics collection for learning pipeline"""
    if not client.collection_exists(FEEDBACK_COLLECTION):
        client.create_collection(
            collection_name=FEEDBACK_COLLECTION,
            vectors_config={
                "query_dense": models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                ),
                "result_dense": models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                "query_sparse": models.SparseVectorParams(),
            }
        )
        print(f"Collection '{FEEDBACK_COLLECTION}' created for feedback analytics.")
    else:
        print(f"Collection '{FEEDBACK_COLLECTION}' already exists.")

    # Payload indexes for analytics queries
    client.create_payload_index(
        collection_name=FEEDBACK_COLLECTION,
        field_name="query_id",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=FEEDBACK_COLLECTION,
        field_name="clinic_id",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=FEEDBACK_COLLECTION,
        field_name="outcome_label",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print(f"Payload indexes ensured for '{FEEDBACK_COLLECTION}'.")


def _initialize_global_insights():
    """Initialize global medical insights collection (anonymized cross-clinic data)"""
    if not client.collection_exists(GLOBAL_INSIGHTS_COLLECTION):
        client.create_collection(
            collection_name=GLOBAL_INSIGHTS_COLLECTION,
            vectors_config={
                "insight_embedding": models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(
                            always_ram=True
                        )
                    )
                ),
            },
            sparse_vectors_config={
                "sparse_keywords": models.SparseVectorParams(),
            }
        )
        print(f"Collection '{GLOBAL_INSIGHTS_COLLECTION}' created for global insights.")
    else:
        print(f"Collection '{GLOBAL_INSIGHTS_COLLECTION}' already exists.")

    # Payload indexes for global insights queries
    client.create_payload_index(
        collection_name=GLOBAL_INSIGHTS_COLLECTION,
        field_name="insight_type",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=GLOBAL_INSIGHTS_COLLECTION,
        field_name="condition",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=GLOBAL_INSIGHTS_COLLECTION,
        field_name="treatment",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=GLOBAL_INSIGHTS_COLLECTION,
        field_name="sample_size",
        field_schema=models.PayloadSchemaType.INTEGER
    )
    print(f"Payload indexes ensured for '{GLOBAL_INSIGHTS_COLLECTION}'.")


def get_collection_info():
    return client.get_collection(COLLECTION_NAME)
