"""
Orchestrator Agent

Responsibilities:
- Entry point for every user query
- Decides which agents to execute and in what order
- Maintains a query trace for auditing
- Does no medical reasoning
"""

from typing import Dict, Any, List
from agents.adk_config import MediSyncAgent

class OrchestratorAgent(MediSyncAgent):
    """
    The Orchestrator is the root agent in the ADK hierarchy.
    It receives the user query and delegates to specialist agents.
    """
    
    def __init__(self):
        super().__init__(name="Orchestrator")
        # In ADK, we might register sub-agents here
        self.active_trace = []

    def run(self, input_data: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """
        Main execution loop for user query.
        """
        query = input_data.get("query", "")
        self.log_activity("Received Query", {"query": query})
        
        # 1. Decision Logic (can be LLM-driven or rule-based)
        # For now, we route everything to Ingestion -> Reasoning -> Explanation
        
        plan = self._create_execution_plan(query)
        self.log_activity("Execution Plan Created", {"plan": plan})
        
        # 2. Execution (Mocking delegation)
        results = {}
        for step in plan:
            self.log_activity("Delegating", {"step": step})
            # In real ADK: results[step] = self.sub_agents[step].run(query)
        
        return {
            "status": "success",
            "trace": self.active_trace,
            "response": "Orchestration complete"
        }

    def _create_execution_plan(self, query: str) -> List[str]:
        """
        Determines the sequence of agents to call.
        """
        # Simple logical routing
        return ["Ingestion", "PatientState", "Risk", "Explanation"]
