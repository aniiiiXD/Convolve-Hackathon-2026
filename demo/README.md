# MediSync Demo Suite

Qdrant Convolve 4.0 Pan-IIT Hackathon

## Quick Start

```bash
# From project root
cd /path/to/convovle

# Run the main demo (recommended for presentations)
python3 demo/conversation.py

# Quick verification (30 seconds)
python3 demo/quick_verify.py

# Full test suite (16 tests)
python3 demo/test_suite.py
```

## Demo Files

| File | Description |
|------|-------------|
| `conversation.py` | **Main Demo** - Interactive clinical scenario with ALL features |
| `quick_verify.py` | Fast 30-second system verification |
| `test_suite.py` | Full 16-component test suite |
| `run_all.py` | Demo runner utility |

## Main Demo Scenes (conversation.py)

The main demo walks through a complete clinical workflow:

1. **Authentication** - Doctor login with role-based access
2. **Vigilance Alerts** - Proactive patient monitoring
3. **Patient Intake** - New ER admission scenario
4. **Hybrid Search** - Sparse + Dense + RRF Fusion
5. **Discovery API** - Context-aware search with +/- vectors
6. **Differential Diagnosis** - AI-generated with confidence scores
7. **Evidence Graph** - Explainable AI visualization
8. **Clinical Recommendations** - Treatment protocols
9. **Global Insights** - Cross-clinic K-anonymized data sharing
10. **Technical Deep-Dive** - Named vectors, code examples

## Features Demonstrated

### Qdrant Features
- Hybrid Search (Sparse BM42 + Dense Gemini + RRF Fusion)
- Discovery API (Context-aware search with positive/negative vectors)
- Prefetch Chains (Multi-stage retrieval pipeline)
- Named Vectors (dense_text, sparse_code, image_clip)
- Payload Filters (Clinic + Patient isolation)
- Binary Quantization (30x memory optimization)

### Clinical AI
- Differential Diagnosis Generation
- Evidence Graphs (Explainable AI)
- Vigilance Monitoring (Proactive alerts)
- Change Detection (Temporal patient state tracking)
- Similar Case Retrieval

### Privacy & Security
- Role-based access control (Doctor/Patient)
- Clinic-level data isolation
- K-anonymity (K>=20, min_clinics>=5)
- PII removal (SSN, phone, email patterns)
- Cross-clinic anonymized insights

## Demo Runner

```bash
python3 demo/run_all.py demo      # Main demo (recommended)
python3 demo/run_all.py verify    # Quick health check
python3 demo/run_all.py test      # Full test suite
```

## Troubleshooting

If demos fail:

1. Check Qdrant connection: `python3 demo/quick_verify.py`
2. Check API keys: `echo $GEMINI_API_KEY`
3. Check collections exist: Run test suite first
