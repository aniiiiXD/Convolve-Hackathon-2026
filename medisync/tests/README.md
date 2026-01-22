# MediSync Test Suite

## Quick Start

```bash
# From project root - run comprehensive test
python3 test_all.py

# Unit tests with mocks
cd medisync && python3 -m pytest tests/ -v
```

---

## Test Files

| File | Description | Requires Qdrant |
|------|-------------|-----------------|
| `test_all.py` (root) | Comprehensive 14-component test | Yes |
| `test_advanced_features.py` | Unit tests with mocks | No |

---

## test_all.py (Comprehensive Test)

Tests all 14 MediSync components against live Qdrant:

| # | Component | What's Tested |
|---|-----------|---------------|
| 1 | Qdrant Connection | Cloud connectivity |
| 2 | Collections | Schema initialization |
| 3 | Embeddings | Gemini dense + sparse vectors |
| 4 | Authentication | Doctor/Patient login |
| 5 | Ingestion | Note storage |
| 6 | Hybrid Search | RRF fusion |
| 7 | Discovery API | Context-aware search |
| 8 | Privacy | Patient isolation |
| 9 | Advanced Retrieval | Multi-stage pipeline |
| 10 | Insights | Clinical intelligence |
| 11 | Vigilance | Alert monitoring |
| 12 | Evidence Graph | Reasoning visualization |
| 13 | Diagnosis | Differential diagnosis |
| 14 | Reranker | Cross-encoder ranking |

### Running

```bash
cd /home/anixd/Documents/convovle
python3 test_all.py
```

### Expected Output

```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Component          ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Qdrant Connection  │ ✓ PASS │
│ Collections        │ ✓ PASS │
│ ...                │ ...    │
└────────────────────┴────────┘
Total: 14 passed, 0 failed
```

---

## test_advanced_features.py (Unit Tests)

Mocked tests that don't require Qdrant connection:

```bash
cd medisync
python3 -m pytest tests/test_advanced_features.py -v

# With coverage
python3 -m pytest tests/test_advanced_features.py --cov=service_agents
```

### Test Classes

- `TestAdvancedRetrievalPipeline` - Prefetch + RRF fusion
- `TestDifferentialDiagnosisAgent` - Discovery API diagnosis
- `TestInsightsGeneratorAgent` - Clinical insights
- `TestVigilanceAgent` - Alert monitoring
- `TestChangeDetectionAgent` - State changes
- `TestEvidenceGraphAgent` - Reasoning chains

---

## Requirements

### For test_all.py (live tests)

```bash
# Environment variables required
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-key
GEMINI_API_KEY=your-gemini-key
```

### For unit tests (mocked)

```bash
pip install pytest pytest-asyncio pytest-cov
```

---

## Test Data

Default test users:

| User ID | Role | Clinic |
|---------|------|--------|
| `Dr_Strange` | DOCTOR | Clinic-A |
| `P-101` | PATIENT | Clinic-A |
| `P-102` | PATIENT | Clinic-A |

---

## Troubleshooting

### "No module named 'medisync'"

```bash
cd /home/anixd/Documents/convovle
pip install -e .
# Or
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "Qdrant connection failed"

- Check `.env` has valid `QDRANT_URL` and `QDRANT_API_KEY`
- Verify cluster is running at https://cloud.qdrant.io

### "GEMINI_API_KEY not found"

- Add key to `.env`
- Get one at https://makersuite.google.com/app/apikey
