from fastembed import TextEmbedding, SparseTextEmbedding
from typing import List, Dict, Any
import numpy as np

# Robust handling for ImageEmbedding
try:
    from fastembed import ImageEmbedding
    IMAGE_EMBEDDING_AVAILABLE = True
except ImportError:
    try:
        from fastembed.image import ImageEmbedding
        IMAGE_EMBEDDING_AVAILABLE = True
    except ImportError:
        try:
            from fastembed.vision import ImageEmbedding
            IMAGE_EMBEDDING_AVAILABLE = True
        except ImportError:
            try:
                # Last resort check for submodule
                from fastembed.image.image_embedding import ImageEmbedding
                IMAGE_EMBEDDING_AVAILABLE = True
            except ImportError:
                IMAGE_EMBEDDING_AVAILABLE = False
                print("Warning: ImageEmbedding not available. Install fastembed[image].")

# Dense Text Model: BAAI/bge-base-en (768 dims, matches collection)
dense_model = TextEmbedding(model_name="BAAI/bge-base-en")

# Sparse Model: prithivida/Splade_PP_en_v1 (FastEmbed supported)
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

# Image Model: CLIP (Using fastembed's clip-vit-b-32)
if IMAGE_EMBEDDING_AVAILABLE:
    try:
        image_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    except Exception as e:
        print(f"Failed to load ImageEmbedding model: {e}")
        image_model = None
        IMAGE_EMBEDDING_AVAILABLE = False
else:
    image_model = None

class EmbeddingService:
    @staticmethod
    def embed_dense(text: str) -> List[float]:
        # FastEmbed returns a generator, we take the first result
        embeddings = list(dense_model.embed([text]))
        return embeddings[0].tolist()

    @staticmethod
    def embed_sparse(text: str) -> Dict[int, float]:
        # Helper to convert SparseEmbedding result to Qdrant sparse format
        embeddings = list(sparse_model.embed([text]))
        user_vector = embeddings[0] # NamedTuple or similar with indices/values
        
        return {
            "indices": user_vector.indices.tolist(),
            "values": user_vector.values.tolist()
        }

    @staticmethod
    def embed_image(image_path_or_bytes) -> List[float]:
        if not IMAGE_EMBEDDING_AVAILABLE or image_model is None:
            raise ImportError("Image embedding is not available. Install fastembed[image]")
        # FastEmbed ImageEmbedding expects file paths or PIL images
        embeddings = list(image_model.embed([image_path_or_bytes]))
        return embeddings[0].tolist()
