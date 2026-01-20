# MediSync Learning System - Quick Start Guide

## Prerequisites

- Python 3.10+
- PostgreSQL or SQLite
- Qdrant vector database
- Gemini API key
- 8GB+ RAM (16GB recommended for model training)
- GPU optional but recommended for training

## Installation

### 1. Install Dependencies

```bash
# Install learning system dependencies
pip install -r learning_requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import sentence_transformers; print('Sentence Transformers: OK')"
```

### 2. Set Environment Variables

Add to your `.env` file:

```bash
# Existing
DATABASE_URL=sqlite:///./medisync.db
QDRANT_URL=http://localhost:6333
GEMINI_API_KEY=your_gemini_api_key_here

# New: Learning System
USE_FINETUNED_EMBEDDINGS=false
USE_RERANKER=false
FEEDBACK_ENABLED=true
GLOBAL_INSIGHTS_MIN_SAMPLE_SIZE=20
GLOBAL_INSIGHTS_MIN_CLINICS=5
```

### 3. Initialize Database

```bash
# Create new SQL tables
python -c "
from medisync.core.db_sql import init_db
init_db()
print('Database initialized')
"
```

### 4. Initialize Qdrant Collections

```bash
# Create feedback_analytics and global_medical_insights collections
python -c "
from medisync.services.qdrant_ops import initialize_collections
initialize_collections()
print('Qdrant collections initialized')
"
```

## Basic Usage

### Collect Feedback

```python
from medisync.services.feedback_service import FeedbackService

# Record a search
query_id = FeedbackService.record_query(
    user_id="doctor1",
    clinic_id="clinic1",
    query_text="chest pain patient history",
    query_type="hybrid",
    result_count=5
)

# Record interaction
FeedbackService.record_interaction(
    query_id=query_id,
    result_point_id="point123",
    result_rank=1,
    result_score=0.92,
    interaction_type="click"
)

# Record outcome
FeedbackService.record_outcome(
    query_id=query_id,
    patient_id="patient456",
    clinic_id="clinic1",
    doctor_id="doctor1",
    outcome_type="helpful",
    confidence_level=5
)

print("Feedback recorded successfully!")
```

### Check Feedback Statistics

```python
from medisync.services.feedback_service import FeedbackService

stats = FeedbackService.get_query_statistics(days=30)
print(f"Total queries: {stats['total_queries']}")
print(f"Click-through rate: {stats['click_through_rate']}%")
```

## Training Workflow

### Step 1: Collect Data (Minimum 1,000 queries)

Run your application normally with `FEEDBACK_ENABLED=true`. The system will automatically log:
- All searches
- Result clicks
- Clinical outcomes

### Step 2: Export Training Data

```bash
# Export last 30 days of feedback
python medisync/training/data_processor.py \
    --days 30 \
    --output-dir ./training_data \
    --batch-name "batch_2026_01_21"

# Output:
# - training_data/embedding_batch_*_train.jsonl
# - training_data/embedding_batch_*_val.jsonl
# - training_data/embedding_batch_*_test.jsonl
# - training_data/reranker_batch_*_train.jsonl
# - training_data/reranker_batch_*_val.jsonl
# - training_data/reranker_batch_*_test.jsonl
```

### Step 3: Train Embedding Model

```bash
# Train fine-tuned embeddings (3-5 hours on GPU)
python medisync/training/embedding_trainer.py \
    --train-file ./training_data/embedding_*_train.jsonl \
    --val-file ./training_data/embedding_*_val.jsonl \
    --epochs 3 \
    --batch-size 16 \
    --learning-rate 2e-5

# Model saved to: ./models/embeddings/embedder-YYYYMMDD-HHMMSS
```

### Step 4: Train Re-Ranker

```bash
# Train re-ranker (5-8 hours on GPU)
python medisync/training/reranker_trainer.py \
    --train-file ./training_data/reranker_*_train.jsonl \
    --val-file ./training_data/reranker_*_val.jsonl \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 2e-5

# Model saved to: ./models/rerankers/reranker-YYYYMMDD-HHMMSS
```

### Step 5: Evaluate Models

```bash
# Evaluate embedder
python medisync/training/evaluator.py \
    --model-type embedder \
    --model-path ./models/embeddings/embedder-20260121-140000 \
    --test-file ./training_data/embedding_*_test.jsonl

# Evaluate re-ranker
python medisync/training/evaluator.py \
    --model-type reranker \
    --model-path ./models/rerankers/reranker-20260121-150000 \
    --test-file ./training_data/reranker_*_test.jsonl
```

### Step 6: Promote to Production

```python
from medisync.models.model_registry import get_registry, ModelType, ModelStatus

registry = get_registry()

# Promote embedder
registry.update_status(
    model_type=ModelType.EMBEDDER,
    version="embedder-20260121-140000",
    status=ModelStatus.ACTIVE
)

# Promote re-ranker
registry.update_status(
    model_type=ModelType.RERANKER,
    version="reranker-20260121-150000",
    status=ModelStatus.ACTIVE
)

print("Models promoted to active!")
```

### Step 7: Enable in Production

Update `.env`:
```bash
USE_FINETUNED_EMBEDDINGS=true
USE_RERANKER=true
```

Restart your application to load the new models.

## Global Insights

### Extract Medical Entities

```python
from medisync.services.medical_entity_extractor import MedicalEntityExtractor

extractor = MedicalEntityExtractor()

entities = extractor.extract_entities(
    "45yo female with type 2 diabetes, treated with metformin 500mg twice daily, "
    "good glycemic control achieved after 3 months"
)

print(entities)
# {
#   "condition": "type_2_diabetes",
#   "treatment": "metformin",
#   "outcome": "stable",
#   "duration_days": 90,
#   "age_bracket": "40-50",
#   "gender": "female"
# }
```

### Query Global Insights

```python
from medisync.services.global_insights import GlobalInsightsService
from medisync.services.auth import User

service = GlobalInsightsService()

# Create authorized user
doctor = User(
    id="doc123",
    username="dr_smith",
    role="DOCTOR",
    clinic_id="clinic1"
)

# Query insights
insights = service.query_insights(
    query="diabetes treatment outcomes",
    user=doctor,
    limit=5
)

for insight in insights:
    print(f"\n{insight['condition']} → {insight['treatment']}")
    print(f"  Sample size: {insight['sample_size']}")
    print(f"  Clinic count: {insight['clinic_count']}")
    print(f"  Statistics: {insight.get('statistics', {})}")
```

### Search by Condition

```python
# Find all insights for a specific condition
diabetes_insights = service.search_by_condition(
    condition="diabetes",
    user=doctor,
    limit=10
)

print(f"Found {len(diabetes_insights)} insights for diabetes")
```

## Privacy Compliance

### Verify K-Anonymity

```python
from medisync.core.privacy import PrivacyValidator, PrivacyFilter

# Example records
records = [
    {"condition": "fracture", "treatment": "cast", "clinic_id": "c1"},
    {"condition": "fracture", "treatment": "cast", "clinic_id": "c2"},
    # ... more records
]

# Apply K-anonymity filter
filtered = PrivacyFilter.apply_k_anonymity(
    records=records,
    k=20,
    min_clinics=5,
    grouping_keys=['condition', 'treatment']
)

print(f"Records after K-anonymity: {len(filtered)}")
```

### Audit for PII

```python
from medisync.core.privacy import PrivacyFilter

text = "Patient John Doe (SSN: 123-45-6789) called from 555-123-4567"

# Audit
pii_matches = PrivacyFilter.audit_for_pii(text)
print("PII found:", pii_matches)

# Remove PII
sanitized = PrivacyFilter.remove_pii(text)
print("Sanitized:", sanitized)
# Output: "Patient John Doe (SSN: [SSN]) called from [PHONE]"
```

## Monitoring

### Check Model Registry

```python
from medisync.models.model_registry import get_registry, ModelType

registry = get_registry()

# List all embedders
embedders = registry.list_models(ModelType.EMBEDDER)
for model in embedders:
    print(f"{model['version']}: {model['status']} (nDCG@5: {model['metrics'].get('ndcg@5', 'N/A')})")

# Get active model
active_reranker = registry.get_model(ModelType.RERANKER)
print(f"Active re-ranker: {active_reranker['version']}")
```

### View Feedback Statistics

```python
from medisync.services.feedback_service import FeedbackService

# Last 7 days
stats_7d = FeedbackService.get_query_statistics(days=7)
print("Last 7 days:")
print(f"  Queries: {stats_7d['total_queries']}")
print(f"  CTR: {stats_7d['click_through_rate']}%")

# Last 30 days
stats_30d = FeedbackService.get_query_statistics(days=30)
print("\nLast 30 days:")
print(f"  Queries: {stats_30d['total_queries']}")
print(f"  CTR: {stats_30d['click_through_rate']}%")
```

### Global Insights Statistics

```python
from medisync.services.global_insights import GlobalInsightsService

service = GlobalInsightsService()
stats = service.get_statistics(user=doctor)

print(f"Total insights: {stats['total_insights']}")
print(f"Total samples: {stats['total_samples']}")
print(f"Average sample size: {stats['avg_sample_size']}")
print(f"Unique conditions: {stats['unique_conditions']}")
print(f"Unique treatments: {stats['unique_treatments']}")
```

## Troubleshooting

### Models Not Loading

```python
from medisync.models.reranker import get_reranker

reranker = get_reranker()
if not reranker.is_available():
    print("Re-ranker not available. Check:")
    print("1. Model is registered in registry")
    print("2. Model has ACTIVE status")
    print("3. Model files exist at specified path")
```

### No Training Data

```bash
# Check feedback statistics
python -c "
from medisync.services.feedback_service import FeedbackService
stats = FeedbackService.get_query_statistics(days=30)
print(f'Queries: {stats[\"total_queries\"]}')
print('Need at least 1,000 queries with interactions')
"
```

### GPU Not Detected

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training will use CPU (slower)")
```

## Next Steps

1. **Integrate FeedbackMiddleware** into your agents
2. **Collect 1,000+ queries** before first training
3. **Train initial models** and evaluate
4. **Deploy in shadow mode** to test
5. **Run A/B tests** to validate improvement
6. **Set up automated retraining** (weekly/monthly)

## Support

- Documentation: `LEARNING_SYSTEM_IMPLEMENTATION.md`
- Logs: Check `medisync/logs/` directory
- Issues: Review error logs and stack traces

## Performance Tips

### Training
- Use GPU for 5-10x speedup
- Start with small batches on CPU to test
- Monitor GPU memory usage
- Use mixed precision (FP16) for faster training

### Inference
- Batch scoring for re-ranker (8-16 documents at once)
- Cache embeddings when possible
- Use quantization for faster inference
- Profile latency and optimize bottlenecks

### Storage
- Archive old training batches after 6 months
- Cleanup old model versions (keep last 5)
- Monitor Qdrant collection sizes
- Regular database maintenance

## Security Checklist

- [ ] Gemini API key secured in environment variable
- [ ] Database credentials not in code
- [ ] Access control enforced (DOCTOR/SYSTEM only for global insights)
- [ ] PII hashing enabled for all feedback
- [ ] K-anonymity validation passing (K≥20, ≥5 clinics)
- [ ] Audit logs enabled
- [ ] HTTPS for API endpoints
- [ ] Regular security updates

## License

Copyright © 2026 MediSync. All rights reserved.
