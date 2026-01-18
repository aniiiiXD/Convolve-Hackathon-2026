"""
Ingestion Agent

Responsibilities:
- Handles PDFs, text notes, and images
- Runs OCR and chunking
- Generates embeddings via services
- Upserts data into Qdrant with metadata
"""

from typing import Dict, Any
from agents.adk_config import MediSyncAgent

class IngestionAgent(MediSyncAgent):
    """
    Responsible for bringing data into the system.
    """
    
    def __init__(self):
        super().__init__(name="IngestionAgent")
    
    def run(self, input_data: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """
        Ingests provided files or text.
        """
        files = input_data.get("files", [])
        text = input_data.get("text", "")
        
        self.log_activity("Ingesting", {"files_count": len(files), "text_len": len(text)})
        
        # 1. OCR & Chunking (delegated to services)
        # 2. Embedding Generation
        # 3. Upsert to Qdrant
        
        return {
            "status": "ingested",
            "chunks_created": 0  # Placeholder
        }
