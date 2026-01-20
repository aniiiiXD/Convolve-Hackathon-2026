
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Qdrant in case verify_features relies on it global init
from unittest.mock import MagicMock
sys.modules['medisync.core.database'] = MagicMock()
sys.modules['medisync.services.qdrant_ops'] = MagicMock()

from medisync.services.embedding import EmbeddingService

def test_gemini_embedding():
    print("Testing Gemini Embedding Service...")
    
    # Check API Key
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not found in env. Test might fail or mock.")
    
    service = EmbeddingService()
    
    test_text = "Medical history of patient with diabetes."
    print(f"Embedding text: '{test_text}'")
    
    try:
        vec = service.get_dense_embedding(test_text)
        print(f"Vector received. Length: {len(vec)}")
        
        if len(vec) == 768:
            print("✓ SUCCESS: Vector dimension is 768 (expected for Gemini/configured).")
        elif len(vec) == 0:
            print("X FAILURE: Vector is empty.")
        else:
            print(f"? INFO: Vector dimension is {len(vec)}.")
            
        print(f"First 5 values: {vec[:5]}")
        
    except Exception as e:
        print(f"X CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini_embedding()
