import os
import logging
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY missing. Embeddings will fail unless set.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to init Gemini Client: {e}")
                self.client = None

    def get_dense_embedding(self, text: str) -> List[float]:
        """
        Generates a dense vector using Gemini API (text-embedding-004 or similar).
        """
        if not self.client:
             logger.error("Gemini Client not initialized.")
             return [0.0] * 768 # Fallback size for Gemini embeddings (usually 768)

        try:
            # Using the latest embedding model
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=768) 
            )
            # The structure depends on the library version, assuming result.embeddings[0].values based on docs
            # But the provided snippet says: 
            # [embedding_obj] = result.embeddings
            # listing = embedding_obj.values
            if hasattr(result, 'embeddings') and result.embeddings:
                 return result.embeddings[0].values
            return [0.0] * 768
            
        except Exception as e:
            logger.error(f"Gemini Embedding Failed: {e}")
            return [0.0] * 768

    def get_sparse_embedding(self, text: str):
        """
        Generates SPLADE sparse vector using FastEmbed (Local).
        Kept local as Gemini doesn't output sparse vectors directly commonly.
        """
        from fastembed import SparseTextEmbedding
        
        # Singleton-ish pattern
        if not hasattr(self, 'sparse_model'):
            self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
            
        embeddings = list(self.sparse_model.embed([text]))
        return embeddings[0]
