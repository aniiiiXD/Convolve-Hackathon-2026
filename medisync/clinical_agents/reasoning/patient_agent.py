from medisync.clinical_agents.base_clinical_agent import MediSyncAgent
from medisync.service_agents.gatekeeper_agent import User
from medisync.service_agents.encoding_agent import EmbeddingService
from medisync.service_agents.memory_ops_agent import COLLECTION_NAME, client
from medisync.service_agents.discovery_agent import DiscoveryService
from qdrant_client import models
import uuid
import time
import re

class PatientAgent(MediSyncAgent):
    def __init__(self, user: User):
        super().__init__(user)
        self.embedder = EmbeddingService()
        self.state = "IDLE"  # State tracking for multi-turn intents
        # Strict Check
        if user.role != "PATIENT":
            raise PermissionError("Only patients can initialize PatientAgent")

    def log_diary(self, text: str) -> str:
        """Logs a personal health diary entry."""
        dense_vec = self.embedder.get_dense_embedding(text)
        
        point_id = str(uuid.uuid4())
        payload = {
            "patient_id": self.user.user_id, # STRICTLY FORCE OWN ID
            "clinic_id": self.clinic_id,
            "text_content": text,
            "timestamp": time.time(),
            "type": "diary"
        }

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"dense_text": dense_vec},
                    payload=payload
                )
            ]
        )
        return point_id

    def get_my_history(self, limit: int = 10):
        """Retrieves ONLY this patient's history."""
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id)),
                models.FieldCondition(key="patient_id", match=models.MatchValue(value=self.user.user_id))
            ]
        )
        
        # Scroll newest first
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return results

    def get_health_insights(self):
        """
        Uses Recommendation API to find content similar to current user's history
        but from the broader clinic knowledge base (anonymized), effectively
        saying 'Patients like you also experienced...'
        """
        # 1. Get my recent vector IDs
        history = self.get_my_history(limit=3)
        if not history:
            return []
            
        positive_ids = [point.id for point in history]
        
        # 2. Recommend from CLINIC, but filter out MY own points to find external info?
        # Actually, for health insights, we want to find "What connects these symptoms?"
        # Let's search for similar records in the clinic.
        return DiscoveryService.recommend_similar(
            positive_ids=positive_ids,
            clinic_id=self.clinic_id,
            limit=3
        )

    def process_request(self, user_input: str):
        """ReAct Loop for Patient CLI."""
        yield ("THOUGHT", f"Processing: '{user_input}' (State: {self.state})...")
        time.sleep(0.3)

        # 1. Handle Active State
        if self.state == "AWAITING_LOG_CONTENT":
            yield ("ACTION", "Logging your response...")
            self.log_diary(user_input)
            self.state = "IDLE"
            yield ("ANSWER", "I've logged that in your personal health diary. Anything else?")
            return

        lower = user_input.lower()
        
        # 2. Intent Detection
        if any(w in lower for w in ["log", "diary", "record", "note"]):
            # Check if content is already provided (e.g., "log I have a headache")
            # Heuristic: if input length > 15 and has specific symptoms, assume it's one-shot.
            # But "log a symptom" is short.
            
            is_generic_trigger = user_input.strip().lower() in ["log", "log a symptom", "add diary", "record symptom", "i want to log"]
            
            if is_generic_trigger:
                self.state = "AWAITING_LOG_CONTENT"
                yield ("ANSWER", "What symptom or feeling would you like to log?")
            else:
                # One-shot logging
                yield ("ACTION", "Logging to Health Diary...")
                self.log_diary(user_input)
                yield ("ANSWER", "I've logged that in your personal health diary.")
            
        elif "history" in lower or "past" in lower:
            yield ("ACTION", "Fetching your history...")
            points = self.get_my_history()
            if points:
                yield ("RESULTS", points)
            else:
                yield ("ANSWER", "Your diary is empty.")
                
        elif "insight" in lower or "advice" in lower:
             yield ("THOUGHT", "Generating insights using Qdrant Recommendation API...")
             results = self.get_health_insights()
             if results:
                 yield ("RESULTS", results)
             else:
                 yield ("ANSWER", "No specific insights found yet. Keep logging symptoms!")
                 
        else:
            yield ("ANSWER", "I can 'log' symptoms, show 'history', or give 'insights'.")
