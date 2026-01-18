from app.services.embedding import EmbeddingService
from app.services.qdrant_ops import COLLECTION_NAME
from app.core.database import client
from qdrant_client import models
import uuid
import time

class ClinicalAgent:
    def __init__(self, clinic_id: str, doctor_id: str):
        self.clinic_id = clinic_id
        self.doctor_id = doctor_id
        # Simple InMemory Memory of conversation
        self.stm = [] 

    def ingest_note(self, patient_id: str, text: str) -> str:
        """
        'Perception' phase: Encodes and stores a clinical observation.
        """
        # 1. Encode
        dense_vec = EmbeddingService.embed_dense(text)
        sparse_vec = EmbeddingService.embed_sparse(text)
        
        # 2. Store
        point_id = str(uuid.uuid4())
        payload = {
            "patient_id": patient_id,
            "clinic_id": self.clinic_id,
            "doctor_id": self.doctor_id,
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
                            indices=sparse_vec["indices"], 
                            values=sparse_vec["values"]
                        )
                    },
                    payload=payload
                )
            ]
        )
        return point_id

    def ingest_image(self, patient_id: str, image_path: str) -> str:
        """
        'Perception' phase (Vision): Encodes and stores a medical image (X-ray, MRI).
        """
        try:
            # 1. Encode
            # If real image encoding fails (missing lib), we mock it for the demo
            image_vec = EmbeddingService.embed_image(image_path)
        except Exception as e:
            print(f"Warning: Image embedding failed ({e}). Using mock vector for detailed architecture test.")
            import numpy as np
            # Mock 512d CLIP vector
            image_vec = np.random.rand(512).tolist()

        # 2. Store
        point_id = str(uuid.uuid4())
        payload = {
            "patient_id": patient_id,
            "clinic_id": self.clinic_id,
            "doctor_id": self.doctor_id,
            "image_path": image_path,
            "timestamp": time.time(),
            "type": "image",
            "modality": "vision"
        }

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "image_clip": image_vec, # Named vector for vision
                        # We might also strictly require text vectors, but Qdrant allows sparse vectors.
                        # For now, we leave other named vectors empty or use placeholders if specific queries require them.
                    },
                    payload=payload
                )
            ]
        )
        return point_id

    def recall(self, query: str, limit: int = 5):
        """
        'Recall' phase: Hybrid RRF Search.
        """
        dense_q = EmbeddingService.embed_dense(query)
        sparse_q = EmbeddingService.embed_sparse(query)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_q, using="dense_text", limit=limit*2,
                    filter=models.Filter(must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id))])
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_q["indices"], values=sparse_q["values"]),
                    using="sparse_code", limit=limit*2,
                    filter=models.Filter(must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id))])
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        return results.points

    def process_request(self, user_input: str):
        """
        Simulates a ReAct (Reason + Act) loop.
        Yields tuples of (step_type, message) to visualizer the agent's brain.
        step_types: 'THOUGHT', 'ACTION', 'SYSTEM', 'ANSWER'
        """
        user_input_lower = user_input.lower()
        
        # 1. PERCEPTION & INTENT CLASSIFICATION
        yield ("THOUGHT", f"Analyzing input: '{user_input}'...")
        time.sleep(0.5) # Fake "processing" latency for dramatic effect

        intent = "unknown"
        if any(w in user_input_lower for w in ["add", "save", "note", "record", "remember", "ingest"]):
            intent = "ingest"
        elif any(w in user_input_lower for w in ["search", "find", "query", "what", "show", "recall", "history"]):
            intent = "recall"
        
        # 2. DECISION
        if intent == "ingest":
            yield ("THOUGHT", "Detected intent: [bold green]MEMORY INGESTION[/bold green]. Extracting entities...")
            
            # Simple heuristic extraction of patient ID
            import re
            pid_match = re.search(r'(P-\d+|patient \w+)', user_input, re.IGNORECASE)
            patient_id = pid_match.group(0) if pid_match else "P-Unknown"
            
            yield ("THOUGHT", f"Target Patient: {patient_id}. Strategy: Create dense + sparse embeddings.")
            yield ("ACTION", f"Calling [bold cyan]ingest_note(patient_id='{patient_id}')[/bold cyan]...")
            
            # Action
            point_id = self.ingest_note(patient_id, user_input)
            
            yield ("SYSTEM", f"✓ Saved content to Vector DB (ID: {point_id})")
            yield ("ANSWER", f"I have successfully memorized that note for **{patient_id}**.")

        elif intent == "recall":
            yield ("THOUGHT", "Detected intent: [bold magenta]KNOWLEDGE RETRIEVAL[/bold magenta].")
            yield ("THOUGHT", "Formulating Hybrid Search query (Dense + SPLADE)...")
            yield ("ACTION", f"Calling [bold cyan]recall('{user_input}')[/bold cyan]...")

            # Action
            results = self.recall(user_input, limit=3)
            
            yield ("SYSTEM", f"✓ Found {len(results)} relevant memories.")
            
            # Synthesis
            if results:
                summary = "\n".join([f"- {p.payload.get('text_content', 'Image')}" for p in results])
                yield ("ANSWER", f"Here is what I found:\n{summary}")
            else:
                yield ("ANSWER", "I searched my memory banks but found no matching records.")

        else:
            yield ("THOUGHT", "Intent unclear. Defaulting to general reasoning/chat.")
            yield ("ANSWER", "I'm listening. You can ask me to 'add a note' or 'search for a patient'.")

