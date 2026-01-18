import os
import requests
from typing import List, Generator
from fastembed import TextEmbedding
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        # Fallback to local
        self.use_local = False
        if not self.hf_token or self.hf_token.startswith("hf_mock"):
            logger.warning("HF_TOKEN missing or mock. Using local FastEmbed.")
            self.use_local = True
            
        # Initialize local model if needed
        if self.use_local:
            try:
                self.local_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                self.local_model = None

    def get_dense_embedding(self, text: str) -> List[float]:
        """
        Generates a dense vector using HF API or local fallback.
        """
        if self.use_local or not self.hf_token:
             if self.local_model:
                return list(self.local_model.embed([text]))[0].tolist()
             else:
                return [0.0] * 384 # Mock fallback if everything fails

        try:
            payload = {"inputs": text, "options": {"wait_for_model": True}}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list):
                if isinstance(data[0], list):
                     return data[0]
                return data
            return data
            
        except Exception as e:
            logger.error(f"HF API Failed: {e}. Switching to local fallback.")
            self.use_local = True
            # Lazy load local model if strictly needed now
            if not hasattr(self, 'local_model') or self.local_model is None:
                self.local_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
            return self.get_dense_embedding(text)

    def get_sparse_embedding(self, text: str):
        """
        Generates SPLADE sparse vector using FastEmbed (Local).
        """
        from fastembed import SparseTextEmbedding
        
        # Singleton-ish pattern
        if not hasattr(self, 'sparse_model'):
            self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
            
        embeddings = list(self.sparse_model.embed([text]))
        return embeddings[0]
