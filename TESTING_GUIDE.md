# MediSync Testing Guide

Complete guide to test all components of the MediSync system.

## Prerequisites

### 1. Environment Setup

Create a `.env` file in the root directory:

```env
# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Gemini AI Configuration
GEMINI_API_KEY=your-gemini-api-key
```

### 2. Install Dependencies

```bash
# Backend
pip install qdrant-client python-dotenv rich pydantic-settings google-generativeai

# Frontend
cd frontend && npm install
```

---

## Quick Test (Recommended)

Run the comprehensive test suite that validates all 14 components:

```bash
python3 test_all.py
```

This tests:
| # | Component | Description |
|---|-----------|-------------|
| 1 | Qdrant Connection | Cloud database connectivity |
| 2 | Collections | Schema initialization |
| 3 | Embeddings | Gemini dense (768-dim) + sparse vectors |
| 4 | Authentication | Doctor/Patient login system |
| 5 | Ingestion | Clinical note storage |
| 6 | Hybrid Search | Sparse + Dense with RRF fusion |
| 7 | Discovery API | Context-aware retrieval |
| 8 | Privacy | Patient data isolation |
| 9 | Advanced Retrieval | Multi-stage pipeline |
| 10 | Insights | Clinical intelligence |
| 11 | Vigilance | Autonomous monitoring |
| 12 | Evidence Graph | Reasoning visualization |
| 13 | Diagnosis | Differential diagnosis |
| 14 | Reranker | Cross-encoder re-ranking |

---

## Testing Options

### Option A: Frontend Demo (No Backend Required)

The frontend includes mock data for demonstration:

```bash
cd frontend && npm run dev
```

Open http://localhost:3000 and test:
- `/` - Landing page with animations
- `/doctor` - Doctor dashboard with mock patients, alerts, insights
- `/doctor/diagnosis` - Enter symptoms, see mock differential diagnosis
- `/patient` - Patient portal with health score, vitals, medications

### Option B: CLI Testing (Requires Qdrant)

```bash
# Doctor CLI
python3 -m medisync.cli.doctor_cli

# Patient CLI
python3 -m medisync.cli.patient_cli
```

**Doctor CLI Commands:**
- Login: `Dr_Strange`
- Search: `search chest pain diabetes`
- Discover: `discover context: cardiac symptoms`
- Ingest: `ingest` (follow prompts)

**Patient CLI Commands:**
- Login: `P-101`
- Symptoms: `symptoms`
- History: `history`
- Insights: `insights`

### Option C: Unit Tests (Mocked)

```bash
cd medisync
python3 -m pytest tests/ -v
python3 -m pytest tests/test_advanced_features.py -v --cov=service_agents
```

---

## Component API Reference

### 1. Hybrid Search (Doctor Agent)

```python
from medisync.service_agents.gatekeeper_agent import AuthService
from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent

user = AuthService.login("Dr_Strange")
agent = DoctorAgent(user)

# Hybrid search with RRF fusion
results = agent.search_clinic("chest pain diabetes", limit=5)
for r in results:
    print(f"Score: {r.score:.3f} | {r.payload.get('text_content', '')[:80]}")
```

### 2. Discovery API (Context Search)

```python
from medisync.service_agents.gatekeeper_agent import AuthService
from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent

user = AuthService.login("Dr_Strange")
agent = DoctorAgent(user)

# Context-aware discovery
results = agent.discover_cases(
    target="cardiac patient",
    context_positive=["chest pain", "elevated troponin"],
    context_negative=["trauma"]
)
```

### 3. Advanced Retrieval Pipeline

```python
from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

pipeline = AdvancedRetrievalPipeline("Clinic-A")

# Multi-stage search (synchronous, returns tuple)
results, metrics = pipeline.search(
    query="patient with elevated blood pressure",
    limit=5,
    context_positive=["hypertension"],
    context_negative=["pediatric"]
)

print(f"Retrieved {len(results)} results in {sum(metrics.stage_timings.values())*1000:.0f}ms")
```

### 4. Differential Diagnosis

```python
from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

agent = DifferentialDiagnosisAgent("Clinic-A")

result = agent.generate_differential(
    symptoms="45-year-old male with crushing chest pain, diaphoresis",
    confirmed_findings=["elevated troponin", "ST depression"],
    ruled_out=["GERD", "musculoskeletal"]
)

print(f"Primary: {result.primary_diagnosis}")
print(f"Confidence: {result.primary_confidence:.0%}")
print(f"Red flags: {result.red_flags}")
```

### 5. Insights Generator

```python
from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent, InsightType
from medisync.service_agents.gatekeeper_agent import AuthService

user = AuthService.login("Dr_Strange")
agent = InsightsGeneratorAgent(user)

# Get available insight types
print([t.value for t in InsightType])
# ['temporal_trend', 'treatment_effectiveness', 'cohort_pattern',
#  'risk_indicator', 'anomaly', 'correlation', 'prediction', 'comparison']
```

### 6. Vigilance Agent

```python
from medisync.clinical_agents.autonomous.vigilance_agent import VigilanceAgentSync, AlertSeverity

agent = VigilanceAgentSync("Clinic-A")

# Check patient for alerts
alerts = agent.check_patient("P-101")
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.title}: {alert.message}")
```

### 7. Evidence Graph

```python
from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

agent = EvidenceGraphAgent()

graph = agent.generate_diagnostic_graph(
    patient_context="65 year old male with history of hypertension",
    symptoms=["chest pain", "shortness of breath", "fatigue"],
    evidence_records=[
        {"id": "1", "score": 0.85, "text_content": "Similar case with MI presentation"},
        {"id": "2", "score": 0.72, "text_content": "Chest pain resolved with nitroglycerin"}
    ],
    diagnosis_candidates=[
        {"diagnosis": "Acute Coronary Syndrome", "confidence": 0.8},
        {"diagnosis": "Heart Failure", "confidence": 0.6}
    ],
    recommendations=["Order stat troponin", "ECG", "Cardiology consult"]
)

# Output formats
print(graph.to_ascii())  # Terminal-friendly
print(graph.to_dot())    # Graphviz DOT format
print(graph.to_json())   # JSON export
```

### 8. Reranker

```python
from medisync.model_agents.ranking_agent import get_reranker

reranker = get_reranker()
print(f"Model: {reranker.reranker_model}")
print(f"Available: {reranker.is_available()}")

# Use with Qdrant hybrid search
results = reranker.rerank_with_qdrant(
    collection_name="clinical_records",
    query="chest pain",
    query_vector=dense_embedding,
    sparse_vector=sparse_embedding,
    initial_limit=50,
    top_k=5
)
```

### 9. Patient Agent (Privacy Isolated)

```python
from medisync.service_agents.gatekeeper_agent import AuthService
from medisync.clinical_agents.reasoning.patient_agent import PatientAgent

user = AuthService.login("P-101")
agent = PatientAgent(user)

# Get only this patient's records (privacy enforced)
history = agent.get_my_history()
print(f"Found {len(history)} records for patient {user.user_id}")
```

---

## Frontend Testing

### Development
```bash
cd frontend
npm run dev
# Open http://localhost:3000
```

### Production Build
```bash
npm run build
npm run start
```

### Type Checking
```bash
npm run lint
npx tsc --noEmit
```

---

## Troubleshooting

### "No module named 'medisync'"
```bash
cd /home/anixd/Documents/convovle
pip install -e .
# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "Qdrant connection failed"
- Verify `.env` has correct `QDRANT_URL` and `QDRANT_API_KEY`
- Check cluster status at https://cloud.qdrant.io

### "GEMINI_API_KEY not found"
- Add key to `.env`
- Get one at https://makersuite.google.com/app/apikey

### "Collection not found"
```python
# Initialize collections
from medisync.service_agents.memory_ops_agent import initialize_collections
initialize_collections()
```

### Frontend build errors
```bash
cd frontend
rm -rf node_modules .next
npm install
npm run build
```

---

## Test Data

Default test users in the system:

| User ID | Role | Clinic |
|---------|------|--------|
| Dr_Strange | DOCTOR | Clinic-A |
| P-101 | PATIENT | Clinic-A |
| P-102 | PATIENT | Clinic-A |
