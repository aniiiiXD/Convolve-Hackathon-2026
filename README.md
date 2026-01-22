# MediSync - Clinical AI Decision Support

**Qdrant Convolve 4.0 Pan-IIT Hackathon**

MediSync is a healthcare AI system providing clinical decision support through multi-stage hybrid retrieval, differential diagnosis, and autonomous monitoring - all powered by Qdrant's native features.

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- Qdrant Cloud account or local instance
- Gemini API key

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Gemini AI Configuration
GEMINI_API_KEY=your-gemini-api-key
```

### 3. Install Dependencies

```bash
# Backend
pip install qdrant-client python-dotenv rich pydantic-settings google-generativeai

# Frontend
cd frontend && npm install
```

### 4. Verify Installation

Run the comprehensive test suite to verify all 14 components:

```bash
python3 test_all.py
```

Expected output:
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Component          ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Qdrant Connection  │ ✓ PASS │
│ Collections        │ ✓ PASS │
│ Embeddings         │ ✓ PASS │
│ Authentication     │ ✓ PASS │
│ Ingestion          │ ✓ PASS │
│ Hybrid Search      │ ✓ PASS │
│ Discovery API      │ ✓ PASS │
│ Privacy            │ ✓ PASS │
│ Advanced Retrieval │ ✓ PASS │
│ Insights           │ ✓ PASS │
│ Vigilance          │ ✓ PASS │
│ Evidence Graph     │ ✓ PASS │
│ Diagnosis          │ ✓ PASS │
│ Reranker           │ ✓ PASS │
└────────────────────┴────────┘
Total: 14 passed, 0 failed
```

---

## User Interfaces

### Doctor Portal
```bash
python3 -m medisync.cli.doctor_cli
```
Login as `Dr_Strange` to access:
- **Hybrid Search**: Sparse + dense vector search with RRF fusion
- **Discovery API**: Context-aware case retrieval
- **Note Ingestion**: Add clinical notes to the system

### Patient Portal
```bash
python3 -m medisync.cli.patient_cli
```
Login as `P-101` to access:
- **Symptom Diary**: Log symptoms securely
- **Health History**: View personal medical records
- **Insights**: AI-powered health recommendations

### Frontend (Next.js)
```bash
cd frontend && npm run dev
```
Open http://localhost:3000 for the web interface:
- `/` - Landing page
- `/doctor` - Doctor dashboard with alerts, patients, insights
- `/doctor/diagnosis` - Differential diagnosis tool
- `/patient` - Patient health portal

---

## Core Features

### Multi-Stage Hybrid Retrieval
Single Qdrant API call that:
1. **Sparse Prefetch** - BM42/SPLADE keyword matching
2. **Dense Prefetch** - Gemini 768-dim semantic search
3. **RRF Fusion** - Reciprocal Rank Fusion for optimal ranking
4. **Discovery Refinement** - Context-aware result filtering

```python
from medisync.service_agents import AdvancedRetrievalPipeline

pipeline = AdvancedRetrievalPipeline("Clinic-A")
results, metrics = pipeline.search(
    query="patient with chest pain",
    limit=5,
    context_positive=["cardiac symptoms"],
    context_negative=["trauma"]
)
```

### Differential Diagnosis
Discovery API powered diagnosis ranking with confidence scores:

```python
from medisync.service_agents import DifferentialDiagnosisAgent

agent = DifferentialDiagnosisAgent("Clinic-A")
result = agent.generate_differential(
    symptoms="crushing chest pain, diaphoresis, radiation to left arm",
    confirmed_findings=["elevated troponin"],
    ruled_out=["GERD"]
)
```

### Insights Generator
Temporal trends, treatment effectiveness, and risk patterns:

```python
from medisync.service_agents import InsightsGeneratorAgent, InsightType
from medisync.service_agents.gatekeeper_agent import AuthService

user = AuthService.login("Dr_Strange")
agent = InsightsGeneratorAgent(user)
insights = agent.analyze_patient("P-101", [InsightType.TEMPORAL_TREND])
```

### Evidence Graphs
Visual reasoning chains from symptoms to diagnosis:

```python
from medisync.clinical_agents.explanation import EvidenceGraphAgent

agent = EvidenceGraphAgent()
graph = agent.generate_diagnostic_graph(
    patient_context="65yo male with hypertension",
    symptoms=["chest pain", "shortness of breath"],
    evidence_records=[{"score": 0.85, "text_content": "Similar MI case"}],
    diagnosis_candidates=[{"diagnosis": "ACS", "confidence": 0.8}],
    recommendations=["Order troponin", "ECG stat"]
)
print(graph.to_ascii())
```

### Autonomous Vigilance
Background monitoring for critical alerts:

```python
from medisync.clinical_agents.autonomous import VigilanceAgentSync

agent = VigilanceAgentSync("Clinic-A")
alerts = agent.check_patient("P-101")
```

---

## Architecture

```
medisync/
├── core_agents/           # Config, database, privacy
├── service_agents/        # Retrieval, diagnosis, insights
├── clinical_agents/       # Vigilance, evidence graphs, reasoning
├── model_agents/          # Re-ranker, registry
├── interface_agents/      # CLI interfaces
└── tests/                 # Test suites

frontend/
├── src/app/              # Next.js pages
├── src/components/       # React components
└── src/lib/              # API client, utilities
```

---

## Testing

```bash
# Comprehensive test (14 components)
python3 test_all.py

# Unit tests with mocks
python3 -m pytest medisync/tests/ -v

# Integration test
python3 verify.py

# Frontend
cd frontend && npm run build
```

---

## Privacy & Security

- **K-Anonymity (K≥20)**: Global insights require minimum cohort size
- **Clinic Isolation**: Doctors only see their clinic's data
- **Patient Isolation**: Patients only see their own records
- **No PII in Vectors**: Embeddings don't contain identifiable information

---

## Documentation

- [Architecture](medisync/docs/architecture.md)
- [Advanced Features](medisync/docs/advanced_features.md)
- [Testing Guide](TESTING_GUIDE.md)
