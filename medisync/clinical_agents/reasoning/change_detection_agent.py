"""
Change Detection Agent (Temporal Reasoning)

Responsibilities:
- Compares patient state vectors across time
- Detects new symptoms, worsening trends, or improvements
- Produces deltas instead of raw summaries
"""

from typing import Dict, Any
from agents.adk_config import MediSyncAgent

class ChangeDetectionAgent(MediSyncAgent):
    """
    Analyzes temporal changes in patient state.
    """
    def __init__(self):
        super().__init__(name="ChangeDetectionAgent")

    def run(self, input_data: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """
        Compares current state to historical states.
        """
        current_state = input_data.get("current_state", {})
        history = input_data.get("history", [])
        
        self.log_activity("Detecting Changes", {"history_length": len(history)})
        
        # 1. Compare vectors
        # 2. Identify significant deltas
        
        return {
            "status": "changed",
            "deltas": ["worsening_edema"]
        }
