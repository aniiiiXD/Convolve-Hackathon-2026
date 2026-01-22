"""
MediSync Service Agents

Core service agents providing:
- Encoding (dense/sparse embeddings)
- Memory operations (Qdrant CRUD)
- Discovery (context-aware search)
- Feedback & Learning
- Global Insights
- Analytics

Advanced agents (v2):
- Advanced Retrieval Pipeline (4-stage: Sparse → Dense → RRF Fusion → Discovery)
- Differential Diagnosis (Discovery API powered)
- Insights Generator (comprehensive clinical insights)
"""

from medisync.service_agents.encoding_agent import EmbeddingService
from medisync.service_agents.memory_ops_agent import (
    COLLECTION_NAME,
    FEEDBACK_COLLECTION,
    GLOBAL_INSIGHTS_COLLECTION,
    initialize_collections
)
from medisync.service_agents.discovery_agent import DiscoveryService
from medisync.service_agents.gatekeeper_agent import User
from medisync.service_agents.learning_agent import FeedbackService
from medisync.service_agents.insights_agent import GlobalInsightsService

# Advanced agents
from medisync.service_agents.advanced_retrieval_agent import (
    AdvancedRetrievalPipeline,
    RetrievalResult,
    advanced_search
)
from medisync.service_agents.differential_diagnosis_agent import (
    DifferentialDiagnosisAgent,
    DifferentialResult,
    DiagnosticCandidate
)
from medisync.service_agents.insights_generator_agent import (
    InsightsGeneratorAgent,
    GeneratedInsight,
    InsightType
)

__all__ = [
    # Core
    'EmbeddingService',
    'COLLECTION_NAME',
    'FEEDBACK_COLLECTION',
    'GLOBAL_INSIGHTS_COLLECTION',
    'initialize_collections',
    'DiscoveryService',
    'User',
    'FeedbackService',
    'GlobalInsightsService',

    # Advanced
    'AdvancedRetrievalPipeline',
    'RetrievalResult',
    'advanced_search',
    'DifferentialDiagnosisAgent',
    'DifferentialResult',
    'DiagnosticCandidate',
    'InsightsGeneratorAgent',
    'GeneratedInsight',
    'InsightType'
]
