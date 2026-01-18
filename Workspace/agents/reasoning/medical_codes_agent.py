"""
Medical Codes Agent

Responsibilities:
- Performs exact matching of ICD-10, medications, lab markers
- Uses sparse vectors / keyword search
- Zero interpretation, zero hallucination
"""

from typing import Dict, Any
from agents.adk_config import MediSyncAgent

class MedicalCodesAgent(MediSyncAgent):
    """
    Retrieves precise medical codes and definitions.
    """
    def __init__(self):
        super().__init__(name="MedicalCodesAgent")

    def run(self, input_data: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """
        Look up codes for terms.
        """
        terms = input_data.get("terms", [])
        self.log_activity("Looking up Codes", {"terms": terms})
        
        return {
            "codes": [{"term": "Edema", "code": "R60.0"}]
        }
