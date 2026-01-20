from google import genai
from typing import List, Optional
import os

class EmbeddingService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding service with an optional API key.
        If no API key is provided, it will be read from the GEMINI_API_KEY environment variable.
        """
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-embedding-001"

    def get_embedding(self, text: str, task_type: str = "SEMANTIC_SIMILARITY", 
                     output_dimensionality: Optional[int] = 768) -> List[float]:
        """
        Get embedding for a single text input.
        
        Args:
            text: The input text to embed
            task_type: The type of task (e.g., "SEMANTIC_SIMILARITY", "RETRIEVAL_QUERY", etc.)
            output_dimensionality: The size of the output embedding vector (128-3072)
            
        Returns:
            List[float]: The embedding vector
        """
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={
                    "task_type": task_type,
                    "output_dimensionality": output_dimensionality
                } if output_dimensionality else {"task_type": task_type}
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

    def get_embeddings_batch(self, texts: List[str], task_type: str = "SEMANTIC_SIMILARITY",
                           output_dimensionality: Optional[int] = 768) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single batch.
        
        Args:
            texts: List of input texts to embed
            task_type: The type of task
            output_dimensionality: The size of the output embedding vectors
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config={
                    "task_type": task_type,
                    "output_dimensionality": output_dimensionality
                } if output_dimensionality else {"task_type": task_type}
            )
            return [embedding.values for embedding in result.embeddings]
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            raise

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        import numpy as np
        from numpy.linalg import norm
        
        dot_product = np.dot(embedding1, embedding2)
        norm_product = norm(embedding1) * norm(embedding2)
        return dot_product / norm_product if norm_product > 0 else 0.0

# Example usage
if __name__ == "__main__":
    # Initialize the service
    embedding_service = EmbeddingService()
    
    # Example: Get a single embedding
    text = "This is a test sentence."
    embedding = embedding_service.get_embedding(text)
    print(f"Embedding length: {len(embedding)}")
    
    # Example: Get multiple embeddings
    texts = ["First sentence", "Second sentence", "Third sentence"]
    embeddings = embedding_service.get_embeddings_batch(texts)
    print(f"Number of embeddings: {len(embeddings)}")
    
    # Example: Calculate similarity
    sim = embedding_service.cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first and second sentence: {sim:.4f}")