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
| **Explainability** | Black box | Animated evidence graphs |
| **Privacy** | Per-query filtering | Built-in K-anonymity (K≥20) |
| **Dependencies** | ColBERT, external models | Qdrant native only |

---

## Demo Flow (5-7 minutes)

### Recommended: Interactive Demo

```bash
python3 demo/conversation.py
```

This showcases **three clinical scenarios** with an enhanced evidence graph:

| Scenario | Patient | Type | What It Shows |
|----------|---------|------|---------------|
| **A** | Tony Stark | Emergency | STEMI - High urgency workflow |
| **B** | Bruce Banner | Chronic | Back pain - Routine differential |
| **C** | Peter Parker | Follow-up | Diabetes - Medication adjustment |

### Demo Scenes (12 total)

1. **Intro** - MediSync overview
2. **Login** - Dr. Strange authentication
3. **Alerts** - Vigilance system showing today's patients
4. **Scenario Selection** - Choose A, B, or C
5. **Patient Intake** - Vitals, symptoms, history, labs
6. **Hybrid Search** - Sparse + Dense + RRF in action
7. **Discovery API** - Context-aware with +/- vectors
8. **Differential Diagnosis** - AI-generated with confidence bars
9. **Evidence Graph** - 6-step animated reasoning chain
10. **Recommendations** - Prioritized clinical actions
11. **Global Insights** - K-anonymized cross-clinic data
12. **Technical Deep-Dive** - Named vectors, code examples

### Suggested Demo Order for Judges

1. **Quick Verify** (30 sec) - Show system works
   ```bash
   python3 demo/quick_verify.py
   ```

2. **Scenario A: Tony Stark** (3 min) - Emergency, high drama
   - Shows urgency, critical alerts, STEMI workflow

3. **Scenario C: Peter Parker** (2 min) - Routine care
   - Shows system handles everyday cases too

4. **Highlight Technical Scene** (1 min)
   - Named vectors architecture
   - Qdrant code examples

---

## Enhanced Evidence Graph

The demo features a **6-step animated evidence graph**:

```
Step 1: Patient Context
    ╔═══════════════════════════════════════╗
    ║  PATIENT                              ║
    ║  Tony Stark, 65yo Male                ║
    ╚═══════════════════════════════════════╝
              │
              ▼
Step 2: Presenting Symptoms
    ┌─────────────────────────────────────────┐
    │  ● Crushing substernal chest pain       │
    │  ● Radiation to left arm and jaw        │
    └─────────────────────────────────────────┘
              │
              ▼
Step 3: Clinical Evidence
    ╔═══════════════════════════════════════════════╗
    ║  ✓ Labs: Troponin I: 2.4 ng/mL       [0.95]  ║
    ║  ✓ History: Hypertension (10 years)  [0.85]  ║
    ╚═══════════════════════════════════════════════╝
              │
              ▼
Step 4: AI Reasoning
    ┌─────────────────────────────────────────────┐
    │  ◆ Symptom pattern matches known cases      │
    │  ◆ Evidence strongly supports primary Dx    │
    └─────────────────────────────────────────────┘
              │
              ▼
Step 5: Diagnosis Ranking (with confidence bars)
    ╔═══════════════════════════════════════════════════════╗
    ║  #1 STEMI                      ██████████████████░░ 94% ║
    ║  #2 Unstable Angina            ████████████░░░░░░░░ 60% ║
    ╚═══════════════════════════════════════════════════════╝
              │
              ▼
Step 6: Recommendations
    ┌─────────────────────────────────────────────┐
    │  ⚡ ACTIVATE CARDIAC CATH LAB - STEMI ALERT │
    │  !  Aspirin 325mg chewed                    │
    └─────────────────────────────────────────────┘
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
| **Named Vectors** | `dense_text`, `sparse_code`, `image_clip` |
| **Payload Filters** | Privacy isolation |
| **Sparse Vectors** | BM42 keyword matching |

### 3. Privacy Architecture

```
Clinic Isolation: clinic_id filter on all queries
        ↓
Patient Isolation: patient_id filter for patients
        ↓
K-Anonymity: Global insights require K≥20, min 5 clinics
        ↓
PII Removal: Automatic SSN, phone, email detection
```

---

## Three Patient Scenarios

### Scenario A: Tony Stark (Emergency)

| Field | Value |
|-------|-------|
| **Condition** | ST-Elevation Myocardial Infarction (STEMI) |
| **Urgency** | CRITICAL |
| **Key Labs** | Troponin I: 2.4 ng/mL (60x normal) |
| **Primary Dx** | STEMI (94% confidence) |
| **Action** | Activate Cardiac Cath Lab |

**Why this scenario?** Shows emergency workflow, high-stakes decision making, critical alerts.

### Scenario B: Bruce Banner (Chronic)

| Field | Value |
|-------|-------|
| **Condition** | L4-L5 Disc Herniation with Radiculopathy |
| **Urgency** | ROUTINE |
| **Key Imaging** | X-ray: Mild disc space narrowing |
| **Primary Dx** | Disc Herniation (75% confidence) |
| **Action** | MRI, Physical Therapy referral |

**Why this scenario?** Shows chronic pain workup, differential diagnosis process, imaging recommendations.

### Scenario C: Peter Parker (Follow-up)

| Field | Value |
|-------|-------|
| **Condition** | Type 1 Diabetes with hypoglycemia |
| **Urgency** | ROUTINE |
| **Key Labs** | HbA1c: 7.8% (target <7.0%) |
| **Primary Dx** | Insulin-to-Carb Ratio Mismatch (70% confidence) |
| **Action** | Adjust insulin ratio, enable Exercise Mode |

**Why this scenario?** Shows routine care, medication adjustment, lifestyle optimization.

---

## Q&A Preparation

### "How does hybrid search work?"
> We use Qdrant's prefetch chains with RRF fusion. First, we run sparse (BM42) and dense (Gemini) searches as prefetches, then fuse results using Reciprocal Rank Fusion - all in a single API call. This combines keyword precision with semantic understanding.

### "What's the Discovery API doing?"
> The Discovery API lets us bias search results toward positive examples and away from negative examples. For diagnosis, we embed confirmed symptoms as positives and ruled-out conditions as negatives. This gives context-aware results without complex re-ranking logic.

### "How do you handle privacy?"
> Three levels: (1) Clinic isolation - doctors only see their clinic's data via payload filters, (2) Patient isolation - patients only see their own records, (3) K-anonymity - global insights require minimum 20 patients from at least 5 clinics to prevent re-identification.

### "Why no external AI models?"
> Qdrant's native RRF fusion and Discovery API replace what would typically require external re-rankers like ColBERT. Fewer dependencies = easier deployment.

### "What's the evidence graph?"
> The evidence graph is our explainable AI feature. It shows the complete reasoning chain from patient symptoms → clinical evidence → AI reasoning → diagnosis ranking → recommendations. It's animated step-by-step so you can see exactly how the system reached its conclusion.

### "What embeddings do you use?"
> Gemini embeddings for dense vectors (768 dimensions) and BM42 for sparse vectors. Both stored as named vectors in Qdrant.

---

## Files to Have Open

1. `demo/conversation.py` - Main demo script
2. `demo/README.md` - Demo documentation
3. `medisync/service_agents/advanced_retrieval_agent.py` - Hybrid search code
4. `medisync/service_agents/discovery_agent.py` - Discovery API code
5. `README.md` - Project overview

---

## Checklist Before Presentation

- [ ] `.env` configured with valid API keys
- [ ] `python3 demo/quick_verify.py` passes
- [ ] `python3 demo/conversation.py` runs without errors
- [ ] Terminal size readable for audience
- [ ] Practice running through all three scenarios
- [ ] Network connection stable (Qdrant Cloud, Gemini API)
- [ ] Know which scenario to demo for different time constraints:
  - **2 min**: Scenario A only (emergency)
  - **5 min**: Scenarios A + C (emergency + follow-up)
  - **7 min**: All three scenarios

---

## Quick Commands Reference

```bash
# Main interactive demo
python3 demo/conversation.py

# Quick verification
python3 demo/quick_verify.py

# Full test suite
python3 demo/test_suite.py

# Demo runner
python3 demo/run_all.py demo
```

---

**MediSync** - Qdrant Convolve 4.0 Pan-IIT Hackathon
