# MediSync System Architecture

**Qdrant Convolve 4.0 Pan-IIT Hackathon**

---

## Overview

MediSync is a multi-agent clinical decision support system built entirely on Qdrant's vector database capabilities. The architecture follows a modular agent-based design where each agent has specific clinical responsibilities.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                          │
│   Doctor Dashboard │ Patient Portal │ Diagnosis Tool │ Insights     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AUTHENTICATION LAYER                            │
│                    Gatekeeper Agent (RBAC)                          │
│              Doctor (Clinic Scope) │ Patient (Self Scope)           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  CLINICAL     │      │   SERVICE     │      │   CORE        │
│  AGENTS       │      │   AGENTS      │      │   AGENTS      │
│  - Doctor     │      │  - Retrieval  │      │  - Database   │
│  - Patient    │      │  - Diagnosis  │      │  - Config     │
│  - Vigilance  │      │  - Insights   │      │  - Privacy    │
│  - Evidence   │      │  - Encoding   │      │  - Registry   │
└───────────────┘      └───────────────┘      └───────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       QDRANT CLOUD                                   │
│  Collections: clinical_records │ feedback_analytics │ global_insights│
│  Features: Hybrid Search │ RRF Fusion │ Discovery API │ Filters     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
medisync/
├── core_agents/                 # Infrastructure
│   ├── config_agent.py          # Environment & settings
│   ├── database_agent.py        # Qdrant client connection
│   └── privacy_agent.py         # K-anonymity enforcement
│
├── service_agents/              # Core Services
│   ├── encoding_agent.py        # Gemini embeddings (dense + sparse)
│   ├── memory_ops_agent.py      # Collection CRUD operations
│   ├── discovery_agent.py       # Qdrant Discovery API wrapper
│   ├── gatekeeper_agent.py      # Authentication & RBAC
│   ├── advanced_retrieval_agent.py  # Multi-stage hybrid search
│   ├── differential_diagnosis_agent.py  # Diagnosis generation
│   ├── insights_generator_agent.py  # Clinical insights
│   └── learning_agent.py        # Feedback & analytics
│
├── clinical_agents/             # Clinical Intelligence
│   ├── reasoning/
│   │   ├── doctor_agent.py      # Doctor workflows
│   │   └── patient_agent.py     # Patient self-service
│   ├── autonomous/
│   │   ├── vigilance_agent.py   # Proactive monitoring
│   │   └── change_detection_agent.py  # State change tracking
│   └── explanation/
│       └── evidence_graph_agent.py  # Reasoning visualization
│
├── model_agents/                # ML Models
│   ├── ranking_agent.py         # Re-ranking with RRF
│   └── registry_agent.py        # Model version management
│
├── interface_agents/            # User Interfaces
│   ├── doctor_cli.py            # Doctor terminal UI
│   └── patient_cli.py           # Patient terminal UI
│
├── tests/                       # Test Suites
│   └── test_advanced_features.py
│
└── docs/                        # Documentation
    ├── architecture.md          # This file
    └── advanced_features.md     # Feature documentation
```

---

## Agent Responsibilities

### Core Agents

| Agent | File | Responsibility |
|-------|------|----------------|
| **Config** | `core_agents/config_agent.py` | Environment variables, feature flags |
| **Database** | `core_agents/database_agent.py` | Qdrant Cloud connection |
| **Privacy** | `core_agents/privacy_agent.py` | K-anonymity enforcement (K≥20) |

### Service Agents

| Agent | File | Responsibility |
|-------|------|----------------|
| **Encoding** | `service_agents/encoding_agent.py` | Dense (Gemini 768d) + Sparse (BM42) embeddings |
| **Memory Ops** | `service_agents/memory_ops_agent.py` | Collection initialization, CRUD |
| **Discovery** | `service_agents/discovery_agent.py` | Context-aware search |
| **Gatekeeper** | `service_agents/gatekeeper_agent.py` | Authentication, role-based access |
| **Advanced Retrieval** | `service_agents/advanced_retrieval_agent.py` | 4-stage hybrid search pipeline |
| **Differential Diagnosis** | `service_agents/differential_diagnosis_agent.py` | Diagnosis generation |
| **Insights Generator** | `service_agents/insights_generator_agent.py` | Temporal trends, risk patterns |
| **Learning** | `service_agents/learning_agent.py` | Feedback collection, analytics |

### Clinical Agents

| Agent | File | Responsibility |
|-------|------|----------------|
| **Doctor** | `clinical_agents/reasoning/doctor_agent.py` | Search, ingest, diagnose |
| **Patient** | `clinical_agents/reasoning/patient_agent.py` | View history, log symptoms |
| **Vigilance** | `clinical_agents/autonomous/vigilance_agent.py` | Critical alerts, monitoring |
| **Change Detection** | `clinical_agents/autonomous/change_detection_agent.py` | Track patient state changes |
| **Evidence Graph** | `clinical_agents/explanation/evidence_graph_agent.py` | Explainable reasoning |

---

## Data Flow

### Doctor Search Flow

```
Doctor Query: "chest pain diabetic patient"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    GATEKEEPER AGENT                          │
│  • Authenticate user                                         │
│  • Extract clinic_id for filtering                          │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ENCODING AGENT                            │
│  • Generate dense embedding (Gemini 768-dim)                │
│  • Generate sparse embedding (BM42)                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              ADVANCED RETRIEVAL PIPELINE                     │
│  Stage 1: Sparse prefetch (100 candidates)                  │
│  Stage 2: Dense prefetch (100 candidates)                   │
│  Stage 3: RRF fusion (combine & rank)                       │
│  Stage 4: Discovery refinement (optional)                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   QDRANT CLOUD                               │
│  Collection: clinical_records                               │
│  Filter: clinic_id = "Clinic-A"                             │
│  Named Vectors: dense_text, sparse_code                     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Ranked Results (Top K)
```

### Patient Privacy Flow

```
Patient Request: "Show my history"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    GATEKEEPER AGENT                          │
│  • Authenticate patient (P-101)                             │
│  • Set scope to PATIENT role                                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    PATIENT AGENT                             │
│  • Apply mandatory filter: patient_id = "P-101"            │
│  • Cannot access other patients' data                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   QDRANT CLOUD                               │
│  Filter: patient_id = "P-101" (enforced)                    │
│  Returns: Only this patient's records                        │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Patient's Own Records Only
```

---

## Qdrant Collections

### clinical_records

Primary collection for all medical data.

```python
{
    "name": "clinical_records",
    "vectors": {
        "dense_text": {
            "size": 768,           # Gemini embedding dimension
            "distance": "Cosine"
        },
        "sparse_code": {
            "type": "sparse"       # BM42/SPLADE vectors
        }
    },
    "payload_indices": [
        "patient_id",              # Patient isolation
        "clinic_id",               # Clinic isolation
        "record_type",             # note, lab, medication, etc.
        "timestamp"                # Temporal queries
    ]
}
```

### feedback_analytics

Stores user interactions for learning.

```python
{
    "name": "feedback_analytics",
    "vectors": {
        "dense_text": {"size": 768, "distance": "Cosine"}
    },
    "payload_indices": [
        "user_id",
        "query_id",
        "interaction_type",        # click, ignore, rate
        "timestamp"
    ]
}
```

### global_medical_insights

Anonymized global patterns (K-anonymity protected).

```python
{
    "name": "global_medical_insights",
    "vectors": {
        "dense_text": {"size": 768, "distance": "Cosine"}
    },
    "payload_indices": [
        "insight_type",
        "cohort_size",             # Must be >= 20
        "condition",
        "timestamp"
    ]
}
```

---

## Qdrant Features Used

### 1. Hybrid Search (Prefetch + RRF)

```python
# Single API call combining sparse and dense search
results = client.query_points(
    collection_name="clinical_records",
    prefetch=[
        Prefetch(query=sparse_vector, using="sparse_code", limit=100),
        Prefetch(query=dense_vector, using="dense_text", limit=100)
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10
)
```

### 2. Discovery API

```python
# Context-aware search with positive/negative examples
results = client.query_points(
    collection_name="clinical_records",
    query=DiscoverQuery(
        discover=DiscoverInput(
            target=query_embedding,
            context=[
                ContextPair(positive=pos_embedding, negative=neg_embedding)
            ]
        )
    ),
    using="dense_text",
    limit=10
)
```

### 3. Payload Filtering

```python
# Multi-level isolation
clinic_filter = Filter(must=[
    FieldCondition(key="clinic_id", match=MatchValue(value="Clinic-A")),
    FieldCondition(key="patient_id", match=MatchValue(value="P-101"))
])
```

### 4. Named Vectors

```python
# Separate vector spaces for different embedding types
vectors = {
    "dense_text": [0.1, 0.2, ...],    # 768-dim Gemini
    "sparse_code": SparseVector(       # Variable-length sparse
        indices=[1, 5, 100, 500],
        values=[0.8, 0.6, 0.4, 0.2]
    )
}
```

---

## Security Model

### Role-Based Access Control (RBAC)

```
┌─────────────────────────────────────────────────────────────┐
│                        ROLES                                 │
├─────────────────────────────────────────────────────────────┤
│  DOCTOR                                                      │
│  • Can search entire clinic's data                          │
│  • Can ingest new records                                   │
│  • Can run differential diagnosis                           │
│  • Cannot see other clinics                                 │
├─────────────────────────────────────────────────────────────┤
│  PATIENT                                                     │
│  • Can only see own records                                 │
│  • Can log symptoms                                         │
│  • Cannot see other patients                                │
│  • Cannot ingest clinical notes                             │
└─────────────────────────────────────────────────────────────┘
```

### Privacy Guarantees

| Feature | Implementation |
|---------|----------------|
| **Clinic Isolation** | All queries filtered by `clinic_id` |
| **Patient Isolation** | Patient queries filtered by `patient_id` |
| **K-Anonymity** | Global insights require cohort size ≥ 20 |
| **Audit Trail** | All queries logged with user context |
| **No PII in Embeddings** | Text content stored separately from vectors |

---

## Deployment

### Environment Variables

```env
# Required
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
GEMINI_API_KEY=your-gemini-key

# Optional
ENABLE_VIGILANCE=true
ENABLE_DISCOVERY_API=true
K_ANONYMITY_THRESHOLD=20
```

### Quick Start

```bash
# 1. Install dependencies
pip install qdrant-client python-dotenv rich pydantic-settings google-generativeai

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run tests
python3 test_all.py

# 4. Start CLI
python3 -m medisync.cli.doctor_cli
```

---

## Hackathon Requirements Mapping

| Requirement | Implementation |
|-------------|----------------|
| **Qdrant as primary database** | All data stored in Qdrant Cloud |
| **Hybrid search** | Prefetch chains + RRF fusion |
| **Discovery API** | Differential diagnosis, context search |
| **Memory** | Patient history, temporal analysis |
| **Recommendations** | Insights, alerts, workup suggestions |
| **Explainability** | Evidence graphs with citations |
| **Privacy** | K-anonymity, role-based isolation |
| **No external AI** | Qdrant native features only |
