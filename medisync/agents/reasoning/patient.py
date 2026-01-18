from medisync.agents.adk_config import MediSyncAgent
from medisync.services.auth import User
from medisync.services.embedding import EmbeddingService
from medisync.services.qdrant_ops import COLLECTION_NAME, client
from medisync.services.discovery import DiscoveryService
from qdrant_client import models
import uuid
import time
import re

class PatientAgent(MediSyncAgent):
    def __init__(self, user: User):
        super().__init__(user)
        self.embedder = EmbeddingService()
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
        yield ("THOUGHT", f"Processing: '{user_input}'...")
        time.sleep(0.3)

        lower = user_input.lower()
        
        if any(w in lower for w in ["log", "diary", "hurt", "feel", "symptom"]):
            yield ("ACTION", "Logging to Health Diary...")
            self.log_diary(user_input)
            yield ("ANSWER", "I've logged that in your personal health diary.")
            
        elif "history" in lower or "past" in lower:
            yield ("ACTION", "Fetching your history...")
            points = self.get_my_history()
            if points:
                summary = "\n".join([f"- {p.payload.get('text_content')}" for p in points])
                yield ("ANSWER", f"Your recent entries:\n{summary}")
            else:
                yield ("ANSWER", "Your diary is empty.")
                
        elif "insight" in lower or "advice" in lower:
             yield ("THOUGHT", "Generating insights using Qdrant Recommendation API...")
             results = self.get_health_insights()
             if results:
                 # In a real app, we would summarize these. For now, list them.
                 summary = "\n".join([f"- Similar Case: {p.payload.get('text_content')}" for p in results])
                 yield ("ANSWER", f"Based on your symptoms, I found these similar patterns in our database:\n{summary}")
             else:
                 yield ("ANSWER", "No specific insights found yet. Keep logging symptoms!")
                 
        else:
            yield ("ANSWER", "I can 'log' symptoms, show 'history', or give 'insights'.")
