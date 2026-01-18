from typing import Any, Dict, List, Optional
import os
from medisync.services.auth import User
from medisync.services.qdrant_ops import initialize_collections
from medisync.services.llm import LLMService

# Ensure collections exist on startup
initialize_collections()

class MediSyncAgent:
    def __init__(self, user: User):
        self.user = user
        self.llm = LLMService()
        self.context = {
             "user_id": user.username,
             "role": user.role,
             "clinic_id": user.clinic_id
        }
        print(f"[Agent] Initialized for User: {user.username} (Role: {user.role})")
        
    def run(self, input_text: str):
        """
        Base run method - to be overridden by specific agents.
        """
        raise NotImplementedError("Subclasses must implement run()")
