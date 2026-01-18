from qdrant_client import QdrantClient
from medisync.core.config import settings

# Initialize Qdrant Client (Shared Singleton)
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)
