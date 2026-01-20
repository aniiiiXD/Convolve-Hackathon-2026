import os
import logging
from typing import List, Optional
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

        # Fine-tuned model support
        self.use_finetuned = os.getenv("USE_FINETUNED_EMBEDDINGS", "false").lower() == "true"
        self.finetuned_model = None

        if self.use_finetuned:
            self._load_finetuned_model()

    def _load_finetuned_model(self):
        """Load fine-tuned embedding model from registry"""
        try:
            from medisync.models.model_registry import get_registry, ModelType, ModelStatus
            from sentence_transformers import SentenceTransformer

            registry = get_registry()

            # Get active fine-tuned model
            model_metadata = registry.get_model(
                model_type=ModelType.EMBEDDER,
                status=ModelStatus.ACTIVE
            )

            if not model_metadata:
                logger.warning(
                    "USE_FINETUNED_EMBEDDINGS=true but no active embedder found. "
                    "Falling back to Gemini."
                )
                self.use_finetuned = False
                return

            model_path = model_metadata['model_path']

            # Load SentenceTransformer model
            self.finetuned_model = SentenceTransformer(model_path)

            logger.info(
                f"Loaded fine-tuned embedder: {model_metadata['version']}"
            )

        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            self.use_finetuned = False

    def get_dense_embedding(self, text: str) -> List[float]:
        """
        Generates a dense vector using fine-tuned model (if enabled) or Gemini API.
        """
        # Try fine-tuned model first
        if self.use_finetuned and self.finetuned_model is not None:
            try:
                embedding = self.finetuned_model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Fine-tuned embedding failed: {e}, falling back to Gemini")
                # Fall through to Gemini

        # Fallback to Gemini
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
