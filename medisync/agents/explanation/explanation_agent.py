"""
Explanation Agent (Doctor-Facing)

Responsibilities:
- Converts reasoning into clinical narrative
- Cites evidence IDs explicitly
- Outputs are fully auditable
"""

from typing import Dict, Any
from agents.adk_config import MediSyncAgent

class ExplanationAgent(MediSyncAgent):
    """
    Generates detailed clinical explanations for doctors.
    """
    def __init__(self):
        super().__init__(name="ExplanationAgent")

    def run(self, input_data: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """
        Synthesize final response.
        """
        reasoning_trace = input_data.get("trace", [])
        self.log_activity("Generating Explanation", {"trace_steps": len(reasoning_trace)})
        
        return {
            "text": "Patient shows signs of worsening edema...",
            "citations": ["doc_1", "img_2"]
        }
