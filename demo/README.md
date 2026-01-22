# MediSync Demo Suite

Qdrant Convolve 4.0 Pan-IIT Hackathon

## Quick Start

```bash
# From project root
cd /path/to/convovle

# Run quick verification (30 seconds)
python3 demo/quick_verify.py

# Run full test suite (16 tests)
python3 demo/test_suite.py

# Run interactive conversation demo (for presentations)
python3 demo/conversation.py
```

## Available Demos

| Demo | File | Description |
|------|------|-------------|
| **Quick Verify** | `quick_verify.py` | Fast 30-second verification |
| **Test Suite** | `test_suite.py` | Full 16-component test |
| **Conversation** | `conversation.py` | Interactive clinical scenario |
| **Evidence Graph** | `evidence_graph.py` | Explainable AI visualization |
| **Hybrid Search** | `hybrid_search_demo.py` | Sparse + Dense + RRF demo |
| **Discovery API** | `discovery_demo.py` | Context-aware search |
| **Global Insights** | `global_insights_demo.py` | K-anonymity & cross-clinic sharing |

## Demo Runner

Use the unified runner to execute demos:

```bash
# Show menu
python3 demo/run_all.py

# Run specific demo
python3 demo/run_all.py hybrid
python3 demo/run_all.py discovery
python3 demo/run_all.py conversation

# Run all demos in sequence
python3 demo/run_all.py all
```

## For Hackathon Presentation

Recommended order for live demo:

1. **Quick Verify** - Show system is working
2. **Conversation** - Interactive clinical scenario (main demo)
3. **Evidence Graph** - Show explainable AI
4. **Hybrid Search** - Technical deep-dive
5. **Global Insights** - Privacy & cross-clinic sharing

## Qdrant Features Demonstrated

- **Hybrid Search**: Sparse (BM42) + Dense (Gemini 768d) + RRF Fusion
- **Discovery API**: Context-aware search with positive/negative vectors
- **Prefetch Chains**: Multi-stage retrieval pipeline
- **Named Vectors**: Separate embedding spaces per use case
- **Payload Filters**: Clinic and patient isolation
- **K-Anonymity**: 5-clinic threshold for global insights

## Files Generated

Some demos create output files:

- `evidence_graph.dot` - GraphViz visualization
- `evidence_graph.json` - Frontend data

## Troubleshooting

If demos fail:

1. Check Qdrant connection: `python3 demo/quick_verify.py`
2. Check API keys: `echo $GEMINI_API_KEY`
3. Check collections exist: Run test suite first
