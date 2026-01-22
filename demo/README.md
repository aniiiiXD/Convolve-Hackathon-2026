# MediSync Demo Suite

**Qdrant Convolve 4.0 Pan-IIT Hackathon**

A comprehensive clinical AI demonstration showcasing Qdrant's advanced vector search capabilities for healthcare applications.

---

## Quick Start

```bash
# Run the main interactive demo
python3 demo/conversation.py

# Quick 30-second health check
python3 demo/quick_verify.py

# Full test suite (16 tests)
python3 demo/test_suite.py
```

---

## Demo Files

| File | Description |
|------|-------------|
| `conversation.py` | **Main Demo** - Interactive clinical scenarios with all features |
| `quick_verify.py` | Fast 30-second system verification |
| `test_suite.py` | Full 16-component test suite |
| `run_all.py` | Demo runner utility |

---

## Three Clinical Scenarios

The demo features three distinct patient scenarios to showcase different clinical workflows:

| Option | Patient | Type | Urgency | Chief Complaint |
|--------|---------|------|---------|-----------------|
| **A** | Tony Stark | Emergency | CRITICAL | Crushing chest pain (STEMI) |
| **B** | Bruce Banner | Chronic | ROUTINE | Lower back pain, radiating to leg |
| **C** | Peter Parker | Follow-up | ROUTINE | Diabetes management, hypoglycemia |

### Scenario A: Tony Stark (Emergency)
- **Condition**: ST-Elevation Myocardial Infarction (STEMI)
- **Demonstrates**: Urgent care workflow, high-confidence diagnosis, critical recommendations
- **Key Features**: Cardiac catheterization alert, door-to-balloon time tracking

### Scenario B: Bruce Banner (Chronic)
- **Condition**: L4-L5 Disc Herniation with Radiculopathy
- **Demonstrates**: Chronic pain workup, differential diagnosis, imaging recommendations
- **Key Features**: MRI ordering, physical therapy referral, ergonomic assessment

### Scenario C: Peter Parker (Follow-up)
- **Condition**: Type 1 Diabetes with recurrent hypoglycemia
- **Demonstrates**: Routine follow-up, medication adjustment, lifestyle optimization
- **Key Features**: Insulin pump adjustments, CGM optimization, pattern analysis

---

## Demo Flow (12 Scenes)

```
┌─────────────────────────────────────────────────────────────┐
│  1. Intro          →  MediSync overview & features          │
│  2. Login          →  Dr. Strange authentication            │
│  3. Alerts         →  Vigilance system - today's patients   │
│  4. Selection      →  Choose scenario (A/B/C)               │
├─────────────────────────────────────────────────────────────┤
│  5. Patient Intake →  Vitals, symptoms, history, labs       │
│  6. Hybrid Search  →  Sparse + Dense + RRF Fusion           │
│  7. Discovery API  →  Context-aware search (+/- vectors)    │
│  8. Differential   →  AI-generated diagnoses with scores    │
│  9. Evidence Graph →  Animated explainable AI reasoning     │
│ 10. Recommendations→  Prioritized clinical actions          │
├─────────────────────────────────────────────────────────────┤
│ 11. Global Insights→  K-anonymized cross-clinic data        │
│ 12. Technical      →  Named vectors, code examples          │
│ 13. Summary        →  Feature recap                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Enhanced Evidence Graph

The evidence graph features a 6-step animated visualization showing the AI reasoning chain:

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
    │  ● Diaphoresis (profuse sweating)       │
    └─────────────────────────────────────────┘
                      │
                      ▼
Step 3: Clinical Evidence
    ╔═══════════════════════════════════════════════╗
    ║  ✓ Labs: Troponin I: 2.4 ng/mL       [0.95]  ║
    ║  ✓ History: Hypertension (10 years)  [0.85]  ║
    ║  ✓ Vitals: BP: 88/56 mmHg            [0.80]  ║
    ╚═══════════════════════════════════════════════╝
                      │
                      ▼
Step 4: AI Reasoning
    ┌─────────────────────────────────────────────┐
    │  ◆ Symptom pattern matches known cases      │
    │  ◆ Evidence strongly supports primary Dx    │
    │  ◆ Negative findings rule out alternatives  │
    └─────────────────────────────────────────────┘
                      │
                      ▼
Step 5: Diagnosis Ranking (with confidence bars)
    ╔═══════════════════════════════════════════════════════╗
    ║  #1 STEMI                      ██████████████████░░ 94% ║
    ║  #2 Unstable Angina            ████████████░░░░░░░░ 60% ║
    ║  #3 Aortic Dissection          ████░░░░░░░░░░░░░░░░ 20% ║
    ╚═══════════════════════════════════════════════════════╝
                      │
                      ▼
Step 6: Recommendations
    ┌─────────────────────────────────────────────┐
    │  ⚡ ACTIVATE CARDIAC CATH LAB - STEMI ALERT │
    │  !  Aspirin 325mg chewed                    │
    │  !  Heparin 60 units/kg IV bolus            │
    └─────────────────────────────────────────────┘
```

---

## Features Demonstrated

### Qdrant Features
| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Sparse (BM42) + Dense (Gemini 768d) + RRF Fusion |
| **Discovery API** | Context-aware search with positive/negative vectors |
| **Prefetch Chains** | Multi-stage retrieval pipeline |
| **Named Vectors** | `dense_text` (768d), `sparse_code` (variable), `image_clip` (512d) |
| **Payload Filters** | Clinic-level and patient-level data isolation |
| **Binary Quantization** | 30x memory optimization |

### Clinical AI
| Feature | Description |
|---------|-------------|
| **Multi-Scenario Support** | Emergency, Chronic, and Follow-up workflows |
| **Differential Diagnosis** | AI-generated with confidence scoring |
| **Evidence Graphs** | Animated explainable AI visualization |
| **Vigilance Monitoring** | Proactive patient alerts |
| **Change Detection** | Temporal patient state tracking |

### Privacy & Security
| Feature | Description |
|---------|-------------|
| **K-Anonymity** | K≥20 records, min 5 clinics required |
| **Role-Based Access** | Doctor and Patient roles with isolation |
| **PII Removal** | Automatic detection of SSN, phone, email patterns |
| **Cross-Clinic Insights** | Anonymized aggregated data sharing |

---

## Demo Runner

```bash
# Show available options
python3 demo/run_all.py

# Run main demo (recommended)
python3 demo/run_all.py demo

# Quick health check
python3 demo/run_all.py verify

# Full test suite
python3 demo/run_all.py test
```

---

## Requirements

- Python 3.10+
- Qdrant Cloud connection (or local instance)
- Gemini API key (`GEMINI_API_KEY` environment variable)
- Required packages: `rich`, `qdrant-client`, `fastembed`, `google-genai`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Qdrant connection fails | Run `python3 demo/quick_verify.py` to diagnose |
| Missing API key | Set `export GEMINI_API_KEY=your_key` |
| Sparse model error | Cache will auto-download on first run |
| Collections missing | Run test suite first to initialize |

---

## For Hackathon Presentation

**Recommended demo order:**

1. **Quick Verify** (30 sec) - Show system is working
2. **Main Demo - Scenario A** (Tony Stark) - High drama emergency case
3. **Main Demo - Scenario C** (Peter Parker) - Show routine care capability
4. **Highlight Technical Deep-Dive** - For judges to see Qdrant integration

**Key talking points:**
- All search features use **Qdrant native APIs** (no external re-rankers)
- **Hybrid search** combines keyword precision with semantic understanding
- **Discovery API** enables context-aware clinical reasoning
- **K-anonymity** enables privacy-preserving cross-clinic insights

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MediSync Demo                          │
├─────────────────────────────────────────────────────────────┤
│  conversation.py                                            │
│  ├── SCENARIOS (A, B, C)         # Patient data structures  │
│  ├── Utility Functions           # UI helpers               │
│  ├── Intro & Login Scenes        # Authentication           │
│  ├── Scenario Selection          # Patient picker           │
│  ├── Patient Scenes              # Generic, data-driven     │
│  ├── Evidence Graph              # Animated visualization   │
│  ├── Global Insights             # K-anonymity demo         │
│  └── Technical Deep-Dive         # Code examples            │
├─────────────────────────────────────────────────────────────┤
│  Qdrant Collections                                         │
│  ├── clinical_records            # Per-clinic PHI data      │
│  ├── feedback_analytics          # Hashed query logs        │
│  └── global_medical_insights     # K-anonymized insights    │
└─────────────────────────────────────────────────────────────┘
```

---

**MediSync** - Qdrant Convolve 4.0 Pan-IIT Hackathon
