# Quick Start Guide - Intensive Test

## Fastest Way to Run

```bash
# Option 1: Direct Python execution
python3 medisync/tests/test_intensive_conversation.py

# Option 2: Using the runner script (interactive menu)
./medisync/tests/run_tests.sh

# Option 3: Using the runner script (direct)
./medisync/tests/run_tests.sh --intensive
```

## What You'll See

The test simulates a complete 6-week patient treatment journey with:

- **20+ Patient dialogues**: Logging symptoms, checking history, seeking insights
- **21+ Doctor dialogues**: Clinical notes, searches, case discovery
- **Real-time output**: Formatted with emojis and colored panels
- **Complete statistics**: Summary report at the end

## Expected Duration

- **Total time**: 2-3 minutes (includes 1-second pauses between actions)
- **Total actions**: 41+ interactions
- **Data created**: 41+ records in Qdrant

## Sample Output

```
═══ PHASE 1: Initial Visit (Day 1) ═══

🧑 Patient (Dialogue #1):
   I've been feeling pain in my right index finger for 3 days...
   💭 Processing: 'I've been feeling pain...'
   ⚡ Logging to Health Diary...
   ╭─────────────────────────────────╮
   │ I've logged that in your diary  │
   ╰─────────────────────────────────╯

👨‍⚕️  Doctor (Dialogue #1):
   add note P-001 Patient presents with finger pain...
   💭 Analyzing input: 'add note P-001...'
   ⚡ Ingesting note for P-001...
   🖥️  ✓ Saved (ID: abc123...)
   ╭──────────────────────────────────╮
   │ Note recorded for **P-001**     │
   ╰──────────────────────────────────╯
```

## Success Criteria

✅ All 41+ actions complete without errors
✅ 21 doctor notes ingested to Qdrant
✅ 20 patient diary entries logged
✅ Multiple searches return results
✅ 100% success rate

## Common Issues

**HuggingFace API Error (Expected):**
```
ERROR: HF API Failed: 410 Client Error. Switching to local fallback.
```
This is normal! The system automatically uses local embeddings.

**First Run:**
Downloads ~532MB model (one-time only)

**Slow execution:**
First few embeddings take longer due to model loading

## Test Output Files

After running, check:
- SQLite database: `medisync.db` (user records)
- Qdrant cloud: Clinical records in `clinical_records` collection

## Cleanup (Optional)

Test users remain in the database:
- Username: `dr_test_intensive` (DOCTOR)
- Username: `patient_test_intensive` (PATIENT)
- Clinic: `TEST-CLINIC-001`

To remove, delete from SQLite manually or rerun with different usernames.
