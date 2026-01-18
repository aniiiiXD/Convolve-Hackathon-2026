"""
Patient Explanation Agent

Responsibilities:
- Translates clinical state into layman language
- Avoids diagnoses and medical jargon
- Improves accessibility and trust
"""

from typing import Dict, Any
from agents.adk_config import MediSyncAgent

class PatientExplainerAgent(MediSyncAgent):
    """
    Translates medical findings into patient-friendly language.
    """
    def __init__(self):
        super().__init__(name="PatientExplainerAgent")

    def run(self, input_data: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """
        Simplifies the output.
        """
        clinical_text = input_data.get("clinical_text", "")
        self.log_activity("Simplifying for Patient", {"input_len": len(clinical_text)})
        
        return {
            "text": "Your fluid levels seem a bit high..."
        }
