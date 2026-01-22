# MediSync - Clinical AI Decision Support

**Qdrant Convolve 4.0 Pan-IIT Hackathon**

MediSync is a healthcare AI system providing clinical decision support through multi-stage hybrid retrieval, differential diagnosis, and autonomous monitoring - all powered by Qdrant's native features.

---

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
pip install qdrant-client python-dotenv rich pydantic-settings google-generativeai fastembed

# Frontend
cd frontend && npm install
```

### 4. Run the Demo

```bash
# Interactive clinical demo (recommended for presentations)
python3 demo/conversation.py

# Quick verification (30 seconds)
python3 demo/quick_verify.py

# Full test suite
python3 demo/test_suite.py
```

---

## Demo Features

The interactive demo (`demo/conversation.py`) showcases **three clinical scenarios**:

| Scenario | Patient | Type | Condition |
|----------|---------|------|-----------|
| **A** | Tony Stark | Emergency | STEMI (Heart Attack) |
| **B** | Bruce Banner | Chronic | L4-L5 Disc Herniation |
| **C** | Peter Parker | Follow-up | Type 1 Diabetes Management |

### Demo Flow (12 Scenes)

```
Intro → Login → Alerts → Scenario Selection →
Patient Intake → Hybrid Search → Discovery API →
Differential Diagnosis → Evidence Graph (animated) →
Recommendations → Global Insights → Technical Deep-Dive → Summary
```

### Enhanced Evidence Graph

The demo features a **6-step animated evidence graph** showing AI reasoning:

1. **Patient Context** - Demographics and chief complaint
2. **Symptoms** - Presenting symptoms with visual indicators
3. **Evidence** - Labs, history, vitals with confidence scores
4. **AI Reasoning** - Pattern matching and rule-out logic
5. **Diagnosis Ranking** - Confidence bars for each differential
6. **Recommendations** - Prioritized clinical actions

See [demo/README.md](demo/README.md) for full demo documentation.

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

### Qdrant-Powered Capabilities

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Sparse (BM42) + Dense (Gemini 768d) + RRF Fusion |
| **Discovery API** | Context-aware search with positive/negative vectors |
| **Prefetch Chains** | Multi-stage retrieval in single API call |
| **Named Vectors** | `dense_text`, `sparse_code`, `image_clip` |
| **Payload Filters** | Clinic and patient-level data isolation |
| **Binary Quantization** | 30x memory optimization |

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

demo/
├── conversation.py        # Main interactive demo (3 scenarios)
├── quick_verify.py        # 30-second health check
├── test_suite.py          # Full 16-component test
├── run_all.py             # Demo runner utility
└── README.md              # Demo documentation

frontend/
├── src/app/              # Next.js pages
├── src/components/       # React components
└── src/lib/              # API client, utilities
```

---

## Privacy & Security

| Feature | Description |
|---------|-------------|
| **K-Anonymity** | K≥20 records, min 5 clinics for global insights |
| **Clinic Isolation** | Doctors only see their clinic's data |
| **Patient Isolation** | Patients only see their own records |
| **PII Removal** | Automatic detection of SSN, phone, email patterns |
| **Role-Based Access** | Doctor and Patient roles with different permissions |

---

## Testing

```bash
# Demo verification
python3 demo/quick_verify.py

# Full demo test suite (16 components)
python3 demo/test_suite.py

# Unit tests with mocks
python3 -m pytest medisync/tests/ -v

# Frontend build test
cd frontend && npm run build
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Demo Guide](demo/README.md) | Interactive demo documentation |
| [Testing Guide](TESTING_GUIDE.md) | Testing procedures |
| [Presentation](PRESENTATION.md) | Hackathon presentation notes |

---

## Key Differentiators

- **All Qdrant Native**: No external re-rankers or ColBERT - pure Qdrant APIs
- **Hybrid Search**: Combines keyword precision with semantic understanding
- **Discovery API**: Context-aware clinical reasoning
- **Privacy-First**: K-anonymity enables cross-clinic insights without exposing PII
- **Explainable AI**: Evidence graphs show complete reasoning chains

---

**MediSync** - Qdrant Convolve 4.0 Pan-IIT Hackathon
