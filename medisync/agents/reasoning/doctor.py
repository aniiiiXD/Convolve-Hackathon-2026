from medisync.agents.adk_config import MediSyncAgent
from medisync.services.auth import User
from medisync.services.embedding import EmbeddingService
from medisync.services.qdrant_ops import COLLECTION_NAME, client
from medisync.services.discovery import DiscoveryService
from medisync.services.feedback_middleware import FeedbackMiddleware
from medisync.services.global_insights import GlobalInsightsService
from medisync.models.reranker import get_reranker
from qdrant_client import models
import uuid
import time
import re
import os

class DoctorAgent(MediSyncAgent):
    def __init__(self, user: User):
        super().__init__(user)
        self.embedder = EmbeddingService()
        self.feedback_middleware = FeedbackMiddleware(
            enabled=os.getenv("FEEDBACK_ENABLED", "true").lower() == "true"
        )
        self.global_insights = GlobalInsightsService()
        self.use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
        self.reranker = get_reranker() if self.use_reranker else None

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
        """Hybrid Search across the entire clinic with optional re-ranking and feedback tracking."""
        dense_q = self.embedder.get_dense_embedding(query)
        sparse_q = self.embedder.get_sparse_embedding(query)

        # Stage 1: Fast retrieval (get more candidates if re-ranking is enabled)
        retrieval_limit = limit * 10 if self.use_reranker and self.reranker and self.reranker.is_available() else limit

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_q, using="dense_text", limit=retrieval_limit*2,
                    filter=models.Filter(must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id))])
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_q.indices, values=sparse_q.values),
                    using="sparse_code", limit=retrieval_limit*2,
                    filter=models.Filter(must=[models.FieldCondition(key="clinic_id", match=models.MatchValue(value=self.clinic_id))])
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=retrieval_limit
        )

        candidates = results.points

        # Stage 2: Re-ranking (if enabled)
        if self.use_reranker and self.reranker and self.reranker.is_available() and len(candidates) > limit:
            candidates = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=limit
            )
        else:
            candidates = candidates[:limit]

        # Track query with feedback middleware
        if self.feedback_middleware.enabled:
            try:
                from medisync.services.feedback_service import FeedbackService

                # Infer intent
                intent = self.feedback_middleware._infer_intent(query)

                # Record query
                query_id = FeedbackService.record_query(
                    user_id=self.user.id,
                    clinic_id=self.clinic_id,
                    query_text=query,
                    query_type="hybrid_reranked" if self.use_reranker else "hybrid",
                    query_intent=intent,
                    result_count=len(candidates),
                    session_id=self.feedback_middleware.session_id
                )

                self.feedback_middleware.current_query_id = query_id
                self.feedback_middleware.query_start_time = time.time()

                # Track result views automatically
                self.feedback_middleware.track_result_views(candidates, auto_log=True)

            except Exception as e:
                import logging
                logging.error(f"Error tracking search feedback: {e}")

        return candidates

    def discover_cases(self, target: str, context_positive: list[str], context_negative: list[str]):
        """Use Discovery API to find nuanced cases."""
        return DiscoveryService.discover_contextual(
            target_text=target,
            positive_texts=context_positive,
            negative_texts=context_negative,
            clinic_id=self.clinic_id
        )

    def get_clinical_recommendations(self, symptoms_text: str, limit: int = 5):
        """
        Finds similar cases to provide clinical decision support.
        This is effectively a semantic search for similar symptom patterns.
        """
        # We reuse search_clinic but conceptually this is for "recommendations"
        # In a real system, this might query a separate 'medical_knowledge' collection
        # or use Qdrant's 'recommend' API if we had a vector for the symptoms.

        # Here we do a focused search on the clinic's data to find precedents.
        return self.search_clinic(symptoms_text, limit=limit)

    def query_global_insights(self, query: str, limit: int = 5):
        """
        Query anonymized global medical insights for treatment decision support.

        Args:
            query: Medical query (e.g., "finger fracture treatment outcomes")
            limit: Maximum number of insights to return

        Returns:
            List of global insights
        """
        try:
            insights = self.global_insights.query_insights(
                query=query,
                user=self.user,
                limit=limit
            )
            return insights
        except PermissionError as e:
            import logging
            logging.warning(f"Access denied to global insights: {e}")
            return []
        except Exception as e:
            import logging
            logging.error(f"Error querying global insights: {e}")
            return []

    def record_result_click(self, result_point_id: str, result_rank: int, result_score: float):
        """
        Record that a user clicked/used a specific search result.

        Args:
            result_point_id: Qdrant point ID of the result
            result_rank: Position in result list (1-indexed)
            result_score: Relevance score
        """
        if self.feedback_middleware.enabled:
            self.feedback_middleware.record_result_interaction(
                result_point_id=result_point_id,
                result_rank=result_rank,
                result_score=result_score,
                interaction_type="click"
            )

    def record_clinical_outcome(self, patient_id: str, outcome_type: str, confidence_level: int):
        """
        Record clinical outcome feedback for the current query.

        Args:
            patient_id: Patient identifier
            outcome_type: Type of outcome (helpful, not_helpful, led_to_diagnosis)
            confidence_level: Confidence rating (1-5)
        """
        if self.feedback_middleware.enabled:
            self.feedback_middleware.record_clinical_outcome(
                patient_id=patient_id,
                doctor_id=self.user.id,
                outcome_type=outcome_type,
                confidence_level=confidence_level
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
        elif any(w in lower_input for w in ["global", "population", "cross-clinic", "insights"]):
             intent = "global_insights"
        elif any(w in lower_input for w in ["search", "find", "query"]):
             intent = "search"
        elif "history" in lower_input or "records" in lower_input:
             intent = "history"
        elif any(w in lower_input for w in ["recommend", "suggest", "advice", "what to do", "treatment"]):
             intent = "recommend"

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

        elif intent == "search":
            yield ("ACTION", f"Searching clinic records...")
            results = self.search_clinic(user_input)
            yield ("RESULTS", results)

        elif intent == "history":
            # Extract Patient ID
            pid_match = re.search(r'(P-\d+|patient \w+)', user_input, re.IGNORECASE)
            if pid_match:
                patient_id = pid_match.group(0)
                yield ("ACTION", f"Retrieving history for {patient_id}...")
                results = self.get_patient_history(patient_id)
                yield ("RESULTS", results)
            else:
                yield ("ANSWER", "Please specify which patient (e.g., 'history of P-101').")

        elif intent == "recommend":
            yield ("THOUGHT", "Fetching clinical recommendations based on similar past cases...")
            yield ("ACTION", f"Analyzing similar cases for: '{user_input}'")
            # Remove trigger words for better search
            query = user_input.replace("recommend", "").replace("suggest", "").strip()
            results = self.get_clinical_recommendations(query)
            if results:
                yield ("RESULTS", results)
                # yield ("ANSWER", "I found these similar cases that might suggest a diagnosis or treatment.")
            else:
                yield ("ANSWER", "No similar medical records found to base a recommendation on.")

        elif intent == "global_insights":
            yield ("THOUGHT", "Querying anonymized global medical insights...")
            yield ("ACTION", f"Searching cross-clinic data for: '{user_input}'")
            # Remove trigger words
            query = user_input.replace("global", "").replace("insights", "").replace("population", "").strip()
            insights = self.query_global_insights(query)
            if insights:
                yield ("GLOBAL_INSIGHTS", insights)
            else:
                yield ("ANSWER", "No global insights found for this query. Try searching for specific conditions or treatments.")

        else:
            yield ("ANSWER", "I can 'add note', 'search', 'discover', 'show history', 'recommend treatment', or query 'global insights'.")
