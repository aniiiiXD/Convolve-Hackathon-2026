# MediSync - Presentation Guide

**Qdrant Convolve 4.0 Pan-IIT Hackathon**

---

## Elevator Pitch (30 seconds)

> "MediSync is a clinical AI decision support system that transforms how doctors diagnose and monitor patients. Using Qdrant's native hybrid search and Discovery API, we combine keyword precision with semantic understanding in a single API call. Our system generates differential diagnoses, proactive alerts, and explainable evidence graphs - all while maintaining strict patient privacy through K-anonymity. No external AI dependencies, just Qdrant."

---

## Key Differentiators

| Feature | Traditional Approach | MediSync with Qdrant |
|---------|---------------------|----------------------|
| **Search** | Keyword OR semantic | Hybrid (RRF fusion in single call) |
| **Context** | None | Discovery API with +/- examples |
| **Diagnosis** | Rule-based | Evidence-based with confidence scores |
| **Explainability** | Black box | Full evidence graph with citations |
| **Privacy** | Per-query filtering | Built-in K-anonymity (K≥20) |
| **Dependencies** | ColBERT, external models | Qdrant native only |

---

## Demo Flow (5-7 minutes)

### 1. Run Test Suite (1 min)

```bash
python3 test_all.py
```

Show 14/14 tests passing:
- Qdrant connection
- Hybrid search
- Discovery API
- Privacy isolation
- All agents working

### 2. Doctor CLI Demo (2 min)

```bash
python3 -m medisync.cli.doctor_cli
```

**Login:** `Dr_Strange`

**Demo commands:**
```
search chest pain diabetic patient
```
→ Show hybrid search results with scores

```
discover context: cardiac symptoms positive: troponin elevation negative: trauma
```
→ Show Discovery API in action

### 3. Frontend Demo (2-3 min)

```bash
cd frontend && npm run dev
```

**Show:**
1. **Landing page** (`/`) - Medical Modernist design, ECG animations
2. **Doctor Dashboard** (`/doctor`) - Patient list, alerts, insights
3. **Diagnosis Tool** (`/doctor/diagnosis`) - Enter symptoms, see differential
4. **Patient Portal** (`/patient`) - Health score, vitals, privacy

### 4. Code Walkthrough (1-2 min)

Show key Qdrant integration:

**Hybrid Search (single API call):**
```python
# service_agents/advanced_retrieval_agent.py
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

**Discovery API:**
```python
# service_agents/discovery_agent.py
results = client.query_points(
    query=DiscoverQuery(
        discover=DiscoverInput(
            target=symptom_embedding,
            context=[
                ContextPair(positive=cardiac_embedding, negative=trauma_embedding)
            ]
        )
    )
)
```

---

## Technical Highlights

### 1. Multi-Stage Hybrid Retrieval

```
┌─────────────────────────────────────────────────┐
│           SINGLE QDRANT API CALL                │
├─────────────────────────────────────────────────┤
│  Sparse Prefetch (BM42) → 100 candidates        │
│            ↓                                    │
│  Dense Prefetch (Gemini 768d) → 100 candidates  │
│            ↓                                    │
│  RRF Fusion → Optimal ranking                   │
│            ↓                                    │
│  Discovery Refinement → Context-aware results   │
└─────────────────────────────────────────────────┘
```

### 2. Qdrant Features Used

| Feature | Implementation |
|---------|----------------|
| **Prefetch** | Multi-stage retrieval chains |
| **RRF Fusion** | Reciprocal Rank Fusion for hybrid |
| **Discovery API** | Context-aware diagnosis |
| **Named Vectors** | Separate dense/sparse spaces |
| **Payload Filters** | Privacy isolation |
| **Sparse Vectors** | BM42 keyword matching |

### 3. Privacy Architecture

```
Clinic Isolation: clinic_id filter on all queries
        ↓
Patient Isolation: patient_id filter for patients
        ↓
K-Anonymity: Global insights require K≥20
        ↓
Audit Trail: All queries logged
```

---

## Slides Content

### Slide 1: Title
**MediSync: Clinical AI Decision Support**
- Qdrant Convolve 4.0 Pan-IIT Hackathon
- Team: [Your names]

### Slide 2: Problem
- Doctors need rapid access to similar cases
- Current systems: keyword OR semantic, not both
- No context-aware search ("find cardiac, exclude trauma")
- Black-box AI recommendations

### Slide 3: Solution
**MediSync leverages Qdrant's native features:**
- Hybrid search (Prefetch + RRF) in single API call
- Discovery API for context-aware diagnosis
- Evidence graphs for explainability
- K-anonymity for privacy

### Slide 4: Architecture
```
[Insert architecture diagram from architecture.md]
```

### Slide 5: Key Features
1. **Multi-Stage Hybrid Retrieval** - Sparse + Dense + RRF
2. **Differential Diagnosis** - Discovery API powered
3. **Autonomous Vigilance** - Proactive critical alerts
4. **Evidence Graphs** - Explainable reasoning chains
5. **Privacy** - K-anonymity, role-based isolation

### Slide 6: Demo
[Live demo screenshots/video]

### Slide 7: Technical Deep Dive
```python
# Single API call for hybrid search
client.query_points(
    prefetch=[sparse_prefetch, dense_prefetch],
    query=FusionQuery(fusion=Fusion.RRF)
)
```

### Slide 8: Results
- 14/14 tests passing
- All Qdrant features integrated
- No external AI dependencies
- Production-grade frontend

### Slide 9: Future Roadmap
- FastAPI REST server
- Real-time WebSocket alerts
- Multi-modal (images, PDFs)
- Fine-tuned medical embeddings

### Slide 10: Thank You
- GitHub: [repo URL]
- Demo: [demo URL]
- Questions?

---

## Q&A Preparation

### "How does hybrid search work?"
> We use Qdrant's prefetch chains with RRF fusion. First, we run sparse (BM42) and dense (Gemini) searches as prefetches, then fuse results using Reciprocal Rank Fusion - all in a single API call. This combines keyword precision with semantic understanding.

### "What's the Discovery API doing?"
> The Discovery API lets us bias search results toward positive examples and away from negative examples. For diagnosis, we embed confirmed symptoms as positives and ruled-out conditions as negatives. This gives context-aware results without complex re-ranking logic.

### "How do you handle privacy?"
> Three levels: (1) Clinic isolation - doctors only see their clinic's data via payload filters, (2) Patient isolation - patients only see their own records, (3) K-anonymity - global insights require minimum 20 patients to prevent re-identification.

### "Why no external AI models?"
> Hackathon requirement, but also better architecture. Qdrant's native RRF fusion and Discovery API replace what would typically require external re-rankers like ColBERT. Fewer dependencies = easier deployment.

### "What embeddings do you use?"
> Gemini embeddings for dense vectors (768 dimensions) and BM42/SPLADE for sparse vectors. Both are generated by our Encoding Agent and stored as named vectors in Qdrant.

### "How scalable is this?"
> Qdrant Cloud handles the heavy lifting. Our architecture is stateless - agents are pure functions that query Qdrant. We tested with simulated clinic data and all 14 components pass.

---

## Files to Have Open

1. `test_all.py` - Show passing tests
2. `medisync/service_agents/advanced_retrieval_agent.py` - Hybrid search code
3. `medisync/service_agents/discovery_agent.py` - Discovery API code
4. `frontend/src/app/doctor/page.tsx` - Dashboard UI
5. `README.md` - Overview

---

## Checklist Before Presentation

- [ ] `.env` configured with valid API keys
- [ ] `python3 test_all.py` passes 14/14
- [ ] Frontend builds: `cd frontend && npm run build`
- [ ] Terminal size readable for audience
- [ ] Browser bookmarks for frontend pages
- [ ] Code editor with files pre-opened
- [ ] Network connection stable (Qdrant Cloud, Gemini API)
