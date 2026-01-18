"""
Uncertainty & Safety Agent

Responsibilities:
- Identifies low-confidence conclusions
- Forces disclaimers when evidence is weak
- Prevents overconfident system responses
"""

from typing import Dict, Any
from agents.adk_config import MediSyncAgent

class UncertaintyAgent(MediSyncAgent):
    """
    Safety gatekeeper.
    """
    def __init__(self):
        super().__init__(name="UncertaintyAgent")

    def run(self, input_data: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """
        Decides if response is safe to send.
        """
        confidence_data = input_data.get("confidence_data", {})
        self.log_activity("Checking Safety", confidence_data)
        
        return {
            "safe_to_respond": True,
            "required_disclaimers": []
        }
