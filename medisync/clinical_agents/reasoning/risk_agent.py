"""
Risk & Triage Agent

Responsibilities:
- Assigns urgency levels (low / medium / high)
- Flags patients needing review
- Suggests conservative next actions
- Never diagnoses
"""

from typing import Dict, Any
from agents.adk_config import MediSyncAgent

class RiskAgent(MediSyncAgent):
    """
    Assesses patient risk and urgency.
    """
    
    def __init__(self):
        super().__init__(name="RiskAgent")

    def run(self, input_data: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """
        Determines risk level based on patient state.
        """
        state = input_data.get("state", {})
        self.log_activity("Assessing Risk", {"state": state.get("state")})
        
        # 1. Check heuristics
        # 2. Consult Doctor Preferences (Loop 1)
        
        return {
            "urgency": "medium",
            "flag_review": True,
            "rationale": "Patient state is stable but history suggests monitoring."
        }
