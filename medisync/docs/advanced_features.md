# MediSync Advanced Features

**Qdrant Convolve 4.0 Pan-IIT Hackathon**

This document describes the advanced features implemented in MediSync, leveraging Qdrant's native capabilities for clinical decision support.

---

## Table of Contents

1. [Feature Overview](#feature-overview)
2. [Multi-Stage Hybrid Retrieval](#multi-stage-hybrid-retrieval)
3. [Discovery API Integration](#discovery-api-integration)
4. [Differential Diagnosis](#differential-diagnosis)
5. [Insights Generator](#insights-generator)
6. [Autonomous Vigilance](#autonomous-vigilance)
7. [Evidence Graphs](#evidence-graphs)
8. [Re-ranking Pipeline](#re-ranking-pipeline)
9. [Privacy & Security](#privacy--security)

---

## Feature Overview

| Feature | Qdrant Capability | Clinical Value |
|---------|-------------------|----------------|
| **Hybrid Search** | Prefetch + RRF Fusion | Combines keyword precision with semantic understanding |
| **Discovery API** | Context-aware search | Find similar cases while excluding ruled-out conditions |
| **Differential Diagnosis** | Discovery + LLM | Ranked diagnoses with confidence scores |
| **Insights Generator** | Aggregation queries | Temporal trends, treatment effectiveness |
| **Vigilance Agent** | Real-time queries | Proactive critical value alerts |
| **Evidence Graphs** | Citation linking | Explainable AI with audit trails |
| **Re-ranking** | Cross-encoder | Precision ranking for clinical relevance |

---

## Multi-Stage Hybrid Retrieval

**File:** `service_agents/advanced_retrieval_agent.py`

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SINGLE QDRANT API CALL                         │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Sparse Prefetch (BM42/SPLADE)                     │
│  └── Exact medical terminology matching (100 candidates)    │
│                          ↓                                  │
│  Stage 2: Dense Prefetch (Gemini 768-dim)                   │
│  └── Semantic similarity search (100 candidates)            │
│                          ↓                                  │
│  Stage 3: RRF Fusion (Reciprocal Rank Fusion)               │
│  └── Combines sparse + dense results optimally              │
│                          ↓                                  │
│  Stage 4: Discovery Refinement (Optional)                   │
│  └── Context-aware filtering with positive/negative examples│
└─────────────────────────────────────────────────────────────┘
```

### Usage

```python
from medisync.service_agents.advanced_retrieval_agent import AdvancedRetrievalPipeline

# Initialize with clinic ID
pipeline = AdvancedRetrievalPipeline("Clinic-A")

# Search returns (results, metrics) tuple
results, metrics = pipeline.search(
    query="patient with chest pain and elevated troponin",
    limit=10,
    context_positive=["cardiac symptoms", "ACS"],
    context_negative=["trauma", "musculoskeletal"]
)

# Access results
for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"Content: {r.payload.get('text_content', '')[:100]}")

# Access metrics
print(f"Total candidates: {metrics.total_candidates}")
print(f"Processing time: {sum(metrics.stage_timings.values())*1000:.0f}ms")
```

### Key Benefits

- **Single API Call**: All stages execute in one Qdrant query
- **No External Dependencies**: Replaced ColBERT with native Qdrant features
- **Configurable**: Enable/disable discovery refinement per query

---

## Discovery API Integration

**File:** `service_agents/discovery_agent.py`

Qdrant's Discovery API enables context-aware search using positive and negative examples.

### How It Works

```
Query: "Find cardiac cases"
  +
Positive Context: ["chest pain", "elevated troponin"]
  +
Negative Context: ["trauma", "pediatric"]
  =
Results biased toward cardiac cases, away from trauma/pediatric
```

### Usage

```python
from medisync.service_agents.gatekeeper_agent import AuthService
from medisync.clinical_agents.reasoning.doctor_agent import DoctorAgent

user = AuthService.login("Dr_Strange")
agent = DoctorAgent(user)

# Context-aware case discovery
results = agent.discover_cases(
    target="cardiac patient",
    context_positive=["chest pain", "elevated troponin", "ST changes"],
    context_negative=["trauma", "pediatric", "anxiety"]
)

for r in results:
    print(f"Patient: {r.payload.get('patient_id')}")
    print(f"Relevance: {r.score:.2f}")
```

### Clinical Applications

| Use Case | Positive Examples | Negative Examples |
|----------|-------------------|-------------------|
| ACS Workup | chest pain, troponin elevation | GERD, anxiety |
| Sepsis Screen | fever, tachycardia, WBC elevation | viral URI |
| Stroke Alert | neurological deficit, sudden onset | migraine, vertigo |

---

## Differential Diagnosis

**File:** `service_agents/differential_diagnosis_agent.py`

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    INPUT: Symptoms                          │
│  "45M crushing chest pain, diaphoresis, arm radiation"     │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│              DISCOVERY API SEARCH                           │
│  • Target: Symptom embedding                                │
│  • Positive: High-confidence confirmed diagnoses            │
│  • Negative: Ruled-out conditions                           │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│              CANDIDATE EXTRACTION                           │
│  • Extract diagnoses from similar cases                     │
│  • Calculate confidence based on match frequency            │
│  • Identify supporting evidence                             │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│                    OUTPUT                                   │
│  • Primary diagnosis + confidence                           │
│  • Alternative diagnoses ranked                             │
│  • Red flags for immediate attention                        │
│  • Recommended workup                                       │
└────────────────────────────────────────────────────────────┘
```

### Usage

```python
from medisync.service_agents.differential_diagnosis_agent import DifferentialDiagnosisAgent

agent = DifferentialDiagnosisAgent("Clinic-A")

result = agent.generate_differential(
    symptoms="45-year-old male with crushing substernal chest pain, diaphoresis, radiation to left arm",
    confirmed_findings=["elevated troponin", "ST depression"],
    ruled_out=["musculoskeletal", "GERD", "anxiety"]
)

print(f"Primary: {result.primary_diagnosis}")
print(f"Confidence: {result.primary_confidence:.0%}")
print(f"Red Flags: {result.red_flags}")
print(f"Workup: {result.recommended_workup}")

for alt in result.alternatives:
    print(f"  - {alt.diagnosis}: {alt.confidence:.0%}")
```

### Output Structure

```python
@dataclass
class DifferentialResult:
    primary_diagnosis: str
    primary_confidence: float      # 0.0 - 1.0
    alternatives: List[DiagnosticCandidate]
    red_flags: List[str]          # Urgent findings
    recommended_workup: List[str] # Suggested tests
    reasoning: str                # Explanation
    evidence_ids: List[str]       # Qdrant point IDs
```

---

## Insights Generator

**File:** `service_agents/insights_generator_agent.py`

### Insight Types

| Type | Description | Example |
|------|-------------|---------|
| `TEMPORAL_TREND` | Changes over time | "HbA1c increased 1.2% over 6 months" |
| `TREATMENT_EFFECTIVENESS` | Treatment outcomes | "Metformin reduced glucose 23%" |
| `COHORT_PATTERN` | Population patterns | "Similar patients have 40% readmission rate" |
| `RISK_INDICATOR` | Risk factors | "3 risk factors for cardiovascular disease" |
| `ANOMALY` | Unusual findings | "Potassium 6.8 - critical high" |
| `CORRELATION` | Variable relationships | "BP correlates with sodium intake (r=0.7)" |
| `PREDICTION` | Future projections | "80% probability of event in 30 days" |
| `COMPARISON` | Benchmarking | "Recovery 20% faster than average" |

### Usage

```python
from medisync.service_agents.insights_generator_agent import InsightsGeneratorAgent, InsightType
from medisync.service_agents.gatekeeper_agent import AuthService

user = AuthService.login("Dr_Strange")
agent = InsightsGeneratorAgent(user)

# Get all insight types
print([t.value for t in InsightType])

# Generate specific insights
insights = agent.analyze_patient(
    patient_id="P-101",
    insight_types=[InsightType.TEMPORAL_TREND, InsightType.RISK_INDICATOR]
)
```

### Privacy Protection

- **K-Anonymity (K≥20)**: Cohort comparisons require minimum 20 patients
- **No PII**: Patient identifiers never included in insight text
- **Audit Trail**: All insights linked to evidence IDs

---

## Autonomous Vigilance

**File:** `clinical_agents/autonomous/vigilance_agent.py`

### Alert Types

| Alert | Severity | Trigger |
|-------|----------|---------|
| `CRITICAL_VALUE` | CRITICAL | K+ > 6.5, Na+ < 120, etc. |
| `DETERIORATION` | HIGH | Worsening trend detected |
| `DRUG_INTERACTION` | HIGH | Dangerous medication combination |
| `MISSED_FOLLOWUP` | MEDIUM | Overdue appointment |
| `TREND_ALERT` | MEDIUM | Gradual worsening over time |
| `PREVENTIVE_CARE` | LOW | Due for screening/vaccination |

### Usage

```python
from medisync.clinical_agents.autonomous.vigilance_agent import VigilanceAgentSync, AlertSeverity

# Initialize with clinic ID
agent = VigilanceAgentSync("Clinic-A")

# Check patient for alerts
alerts = agent.check_patient("P-101")

for alert in alerts:
    print(f"[{alert.severity.value.upper()}] {alert.title}")
    print(f"  {alert.message}")
    if alert.severity == AlertSeverity.CRITICAL:
        print("  ⚠️ IMMEDIATE ACTION REQUIRED")
```

### Alert Severities

```python
class AlertSeverity(Enum):
    INFO = "info"        # Informational
    WARNING = "warning"  # Monitor closely
    HIGH = "high"        # Urgent attention
    CRITICAL = "critical" # Immediate action
```

---

## Evidence Graphs

**File:** `clinical_agents/explanation/evidence_graph_agent.py`

### Purpose

Visual reasoning chains showing how evidence leads to clinical conclusions.

### Graph Structure

```
         ┌─────────────────────────────────────┐
         │          DIAGNOSIS                  │
         │    "Acute Coronary Syndrome"        │
         │         Confidence: 87%             │
         └─────────────────┬───────────────────┘
                           │ LEADS_TO
       ┌───────────────────┼───────────────────┐
       ↓                   ↓                   ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  REASONING  │    │  REASONING  │    │  REASONING  │
│ "Troponin   │    │ "ST changes │    │ "Classic    │
│  elevation" │    │  on ECG"    │    │  symptoms"  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │ SUPPORTS         │ SUPPORTS         │ SUPPORTS
       ↓                   ↓                   ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  EVIDENCE   │    │  EVIDENCE   │    │  EVIDENCE   │
│ "Lab 01/22" │    │ "ECG 01/22" │    │ "Note 01/22"│
└─────────────┘    └─────────────┘    └─────────────┘
```

### Usage

```python
from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent

agent = EvidenceGraphAgent()

graph = agent.generate_diagnostic_graph(
    patient_context="65 year old male with history of hypertension",
    symptoms=["chest pain", "shortness of breath", "diaphoresis"],
    evidence_records=[
        {"id": "1", "score": 0.85, "text_content": "Troponin I elevated to 2.4 ng/mL"},
        {"id": "2", "score": 0.78, "text_content": "ECG shows ST elevation in V1-V4"},
        {"id": "3", "score": 0.72, "text_content": "Crushing substernal chest pain"}
    ],
    diagnosis_candidates=[
        {"diagnosis": "Acute Coronary Syndrome", "confidence": 0.87},
        {"diagnosis": "Unstable Angina", "confidence": 0.65},
        {"diagnosis": "Aortic Dissection", "confidence": 0.25}
    ],
    recommendations=[
        "Activate cath lab",
        "Aspirin 325mg",
        "Heparin bolus"
    ]
)

# Export formats
print(graph.to_ascii())   # Terminal display
print(graph.to_dot())     # GraphViz visualization
print(graph.to_json())    # JSON for web frontend
```

### Export Formats

| Format | Use Case |
|--------|----------|
| `to_ascii()` | Terminal/CLI display |
| `to_dot()` | GraphViz visualization |
| `to_json()` | Web frontend, API response |
| `to_dict()` | Python processing |

---

## Re-ranking Pipeline

**File:** `model_agents/ranking_agent.py`

### Architecture

Uses Qdrant's native hybrid search with RRF fusion for optimal ranking.

```python
from medisync.model_agents.ranking_agent import get_reranker

reranker = get_reranker()

print(f"Model: {reranker.reranker_model}")
print(f"Available: {reranker.is_available()}")

# Re-rank with hybrid search
results = reranker.rerank_with_qdrant(
    collection_name="clinical_records",
    query="chest pain management",
    query_vector=dense_embedding,      # 768-dim Gemini
    sparse_vector=sparse_embedding,    # BM42/SPLADE
    initial_limit=50,                  # Candidates
    top_k=5                           # Final results
)
```

---

## Privacy & Security

### Multi-Level Isolation

```
┌─────────────────────────────────────────────────────────┐
│                    CLINIC LEVEL                          │
│  Doctors only see data from their assigned clinic       │
│  Filter: clinic_id = user.clinic_id                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   PATIENT LEVEL                          │
│  Patients only see their own records                     │
│  Filter: patient_id = user.user_id                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  K-ANONYMITY (K≥20)                      │
│  Global insights require minimum cohort size             │
│  Protects individual patient privacy                     │
└─────────────────────────────────────────────────────────┘
```

### Implementation

```python
# Clinic isolation (Doctor Agent)
clinic_filter = models.Filter(must=[
    models.FieldCondition(
        key="clinic_id",
        match=models.MatchValue(value=self.clinic_id)
    )
])

# Patient isolation (Patient Agent)
patient_filter = models.Filter(must=[
    models.FieldCondition(
        key="patient_id",
        match=models.MatchValue(value=self.user.user_id)
    )
])

# K-Anonymity check
if cohort_size < 20:
    raise PrivacyError("Insufficient cohort size for comparison")
```

---

## Hackathon Requirements Mapping

| Requirement | MediSync Implementation |
|-------------|------------------------|
| **Qdrant as primary search** | All agents use Qdrant Cloud exclusively |
| **Hybrid search** | Prefetch chains + RRF fusion |
| **Discovery API** | Differential diagnosis, context-aware search |
| **Memory** | Patient history, temporal trends |
| **Recommendations** | Insights, vigilance alerts, workup suggestions |
| **Explainability** | Evidence graphs with full citations |
| **Privacy** | K-anonymity, clinic/patient isolation |
| **No external AI dependencies** | Removed ColBERT, uses Qdrant native |
