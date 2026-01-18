# MediSync Testing Guide

This document outlines the testing strategy for the MediSync system, including database setup, environment configuration, and full end-to-end workflows.

## 1. Prerequisites

### Environment Variables
Ensure your `.env` file contains:
```env
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="..."

# For Prod-Grade Embeddings
HF_TOKEN="hf_..."

# For LLM Brain
GEMINI_API_KEY="AIza..."

# Database (Default local postgres)
DATABASE_URL="postgresql://postgres@localhost/medisync_db"
```

### Database Setup
We use PostgreSQL.
1. Create the database:
   ```bash
   createdb medisync_db
   ```
   *(If you don't have postgres user, change the URL to match a user that exists, e.g. `postgresql://anixd@localhost/medisync_db`)*

2. Initialize Tables:
   The system auto-initializes tables on the first run of `db_sql.py` or the test script.

## 2. Automated Robust Verification
We have a unified test script that:
1. Resets the database (optional).
2. Registers Users (Doctor + Patient).
3. Simulates a clinical note ingestion (using HF embeddings).
4. Simulates a patient checking history.
5. Simulates an LLM Diagnosis Assist workflow.

Run it:
```bash
python3 tests/test_robust.py
```

## 3. Manual Workflow walkthroughs

### Scenario A: New Patient Onboarding
**Actor**: Dr. Strange (Doctor)
**Goal**: Record initial consultation.

1. **Login**:
   ```bash
   python3 medisync/cli/doctor_cli.py
   # Enter "Dr_Strange"
   ```
   *Note: If using DB, you might need to register first or use the seeded users.*

2. **Ingest**:
   > "New patient P-101. Complains of chronic migraines. Prescribed Sumatriptan."

3. **Verify**:
   The system uses Hugging Face to embed this note and Qdrant to store it.

### Scenario B: Patient Access
**Actor**: P-101 (Patient)
**Goal**: Recall what the doctor said.

1. **Login**:
   ```bash
   python3 medisync/cli/patient_cli.py
   # Enter "P-101"
   ```

2. **History**:
   > "Show my history"

   *Expected*: Should see the "chronic migraines" note entered by Dr. Strange.

### Scenario C: Diagnosis Assist (Gemini)
**Actor**: Dr. Strange
**Goal**: Get AI opinion.

1. **Discovery**:
   > "Analyze P-101's condition based on history"

   *System Action*:
   1. Retrieves P-101's notes from Qdrant.
   2. Sends text to Gemini Pro.
   3. Returns synthesized analysis.
