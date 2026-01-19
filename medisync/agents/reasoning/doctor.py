from medisync.agents.adk_config import MediSyncAgent
from medisync.services.auth import User
from medisync.services.embedding import EmbeddingService
from medisync.services.qdrant_ops import COLLECTION_NAME, client
from medisync.services.discovery import DiscoveryService
from qdrant_client import models
import uuid
import time
import re

class DoctorAgent(MediSyncAgent):
    def __init__(self, user: User):
        super().__init__(user)
        self.embedder = EmbeddingService()

    def ingest_note(self, patient_id: str, text: str) -> str:
        """Encodes and stores a clinical observation."""
        dense_vec = self.embedder.get_dense_embedding(text)
        sparse_vec = self.embedder.get_sparse_embedding(text)
        
        point_id = str(uuid.uuid4())
        payload = {
            "patient_id": patient_id,
            "clinic_id": self.clinic_id,
            "doctor_id": self.user.user_id,
            "text_content": text,
            "timestamp": time.time(),
            "type": "note"
        }

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense_text": dense_vec,
                        "sparse_code": models.SparseVector(
                            indices=sparse_vec.indices, 
                            values=sparse_vec.values
                        )
                    },
                    payload=payload
                )
            ]
        )
        return point_id

    def search_clinic(self, query: str, limit: int = 5):
        """Hybrid Search across the entire clinic."""
        dense_q = self.embedder.get_dense_embedding(query)
        sparse_q = self.embedder.get_sparse_embedding(query)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_q, using="dense_text", limit=limit*2,
                    filter=models.Filter(must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id))])
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_q.indices, values=sparse_q.values),
                    using="sparse_code", limit=limit*2,
                    filter=models.Filter(must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id))])
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        return results.points

    def discover_cases(self, target: str, context_positive: list[str], context_negative: list[str]):
        """Use Discovery API to find nuanced cases."""
        return DiscoveryService.discover_contextual(
            target_text=target,
            positive_texts=context_positive,
            negative_texts=context_negative,
            clinic_id=self.clinic_id
        )

    def process_request(self, user_input: str):
        """ReAct Loop for Doctor CLI."""
        yield ("THOUGHT", f"Analyzing input: '{user_input}'...")
        time.sleep(0.3)

        intent = "unknown"
        lower_input = user_input.lower()
        
        if any(w in lower_input for w in ["add", "save", "note", "record"]):
             intent = "ingest"
        elif "discover" in lower_input:
             intent = "discovery"
        elif any(w in lower_input for w in ["search", "find", "query"]):
             intent = "search"

        if intent == "ingest":
            # Extract Patient ID (Naive Regex)
            pid_match = re.search(r'(P-\d+|patient \w+)', user_input, re.IGNORECASE)
            patient_id = pid_match.group(0) if pid_match else "P-Unknown"
            
            yield ("ACTION", f"Ingesting note for {patient_id}...")
            point_id = self.ingest_note(patient_id, user_input)
            yield ("SYSTEM", f"✓ Saved (ID: {point_id})")
            yield ("ANSWER", f"Note recorded for **{patient_id}**.")

        elif intent == "discovery":
            # "Discover fractures context: diabetes, not: arm"
            # Simple parsing for demo
            parts = user_input.split("context:")
            target = parts[0].replace("discover", "").strip()
            context_part = parts[1] if len(parts) > 1 else ""
            
            yield ("THOUGHT", f"Discovery Search: Target='{target}', Context='{context_part}'")
            yield ("ACTION", "Calling Qdrant Discovery API...")
            
            # Mocking context parsing
            results = self.discover_cases(target, [context_part], [])
            
            yield ("RESULTS", results)
            # summary = "\n".join([f"- {p.payload.get('text_content')}" for p in results])
            # yield ("ANSWER", f"Discovery Results:\n{summary}")

        elif intent == "search":
            yield ("ACTION", f"Searching clinic records...")
            results = self.search_clinic(user_input)
            yield ("RESULTS", results)
            # summary = "\n".join([f"- {p.payload.get('text_content')}" for p in results])
            # yield ("ANSWER", f"Found:\n{summary}")
        
        else:
            yield ("ANSWER", "I can help you 'add a note', 'search', or 'discover' cases.")
