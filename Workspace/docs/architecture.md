# System Architecture

## 1. Agent Responsibilities

### 1.1 Orchestrator Agent
**File:** `agents/orchestration/orchestrator_agent.py`
- **Entry point** for every user query.
- Decides which agents to execute and in what order.
- Maintains a query trace for auditing.
- **Does no medical reasoning.**

### 1.2 Ingestion Agent
**File:** `agents/ingestion/ingestion_agent.py`
- Handles PDFs, text notes, and images.
- Runs OCR and chunking.
- Generates embeddings via services.
- Upserts data into Qdrant with metadata.

### 1.3 Patient State Agent (Core Intelligence)
**File:** `agents/reasoning/patient_state_agent.py`
- Synthesizes all evidence into a single patient state.
- Classifies state (stable / deteriorating / improving).
- Outputs state vectors with evidence pointers.
- Central object used by downstream agents.

### 1.4 Change Detection Agent (Temporal Reasoning)
**File:** `agents/reasoning/change_detection_agent.py`
- Compares patient state vectors across time.
- Detects new symptoms, worsening trends, or improvements.
- Produces deltas instead of raw summaries.

### 1.5 Risk & Triage Agent
**File:** `agents/reasoning/risk_agent.py`
- Assigns urgency levels (low / medium / high).
- Flags patients needing review.
- Suggests conservative next actions.
- **Never diagnoses.**

### 1.6 Medical Codes Agent
**File:** `agents/reasoning/medical_codes_agent.py`
- Performs exact matching of ICD‑10, medications, lab markers.
- Uses sparse vectors / keyword search.
- Zero interpretation, zero hallucination.

### 1.7 Imaging Evidence Agent
**File:** `agents/reasoning/imaging_evidence_agent.py`
- Performs semantic similarity over medical images.
- Confirms or contradicts text‑based findings.
- Supports longitudinal image comparison.

### 1.8 Evidence Curator Agent
**File:** `agents/validation/evidence_curator_agent.py`
- Filters noisy or redundant chunks.
- Prioritizes recent, high‑confidence, multi‑modal evidence.
- Improves explanation quality and safety.

### 1.9 Validator Agent
**File:** `agents/validation/validator_agent.py`
- Cross‑checks outputs from all reasoning agents.
- Detects contradictions and agreement levels.
- Produces an agreement / consistency score.

### 1.10 Uncertainty & Safety Agent
**File:** `agents/validation/uncertainty_agent.py`
- Identifies low‑confidence conclusions.
- Forces disclaimers when evidence is weak.
- Prevents overconfident system responses.

### 1.11 Explanation Agent (Doctor‑Facing)
**File:** `agents/explanation/explanation_agent.py`
- Converts reasoning into clinical narrative.
- Cites evidence IDs explicitly.
- Outputs are fully auditable.

### 1.12 Patient Explanation Agent
**File:** `agents/explanation/patient_explainer_agent.py`
- Translates clinical state into layman language.
- Avoids diagnoses and medical jargon.
- Improves accessibility and trust.

## 2. Services
- **Qdrant Services** (`services/qdrant_services.py`): Hybrid search, discovery, filters.
- **Qdrant Multimodal** (`services/qdrant_multimodal.py`): Named vectors & multimodal helpers.
- **Embedding Service** (`services/embedding_service.py`): Dense / sparse / image embeddings.
- **OCR Service** (`services/ocr_service.py`): PDF & image text extraction.
- **LLM Service** (`services/llm_service.py`): LLM calls (summarization only).
- **Auth Context** (`services/auth_context.py`): Doctor ID / clinic ID context.

## 3. Feedback Loops (System Intelligence)

A feedback loop is a **past decision stored in Qdrant influencing a future agent decision**.

### Loop 1: Doctor Behavior → System Adaptation
**Mechanism:** Doctors ignoring alerts or marking results as useful trains the system preferences.

1. **Event:** Doctor ignores an alert, clicks a result, or marks utility.
2. **Recording Agent:** `Doctor Preference Agent` (in `orchestration/`)
3. **Qdrant Storage:** Collection `doctor_memory`
   ```json
   {
     "doctor_id": "D-14",
     "event": "ignored_alert",
     "alert_type": "low_risk_edema",
     "timestamp": "2026-01-18T19:40Z"
   }
   ```
4. **Reading Agent:** `Risk Agent`, `Orchestrator Agent`
5. **Behavior Change:** Raises thresholds for ignored alerts; deprioritizes similar future cases.
6. **Code Mapping:**
   - `/server/routes/feedback.py` → writes event
   - `doctor_preference_agent.py` → aggregates
   - `risk_agent.py` → reads preferences before scoring

### Loop 2: Temporal Patient Memory → Clinical Confidence
**Mechanism:** Longitudinal data validation strengthens or weakens confidence in current decisions.

1. **Event:** New data arrives (lab, note, image)
2. **Recording Agent:** `Patient State Agent`
3. **Qdrant Storage:** Collection `patient_state` (Append-only)
   ```json
   {
     "patient_id": "P-90210",
     "state": "deteriorating",
     "drivers": ["fluid retention"],
     "timestamp": "2026-01-18"
   }
   ```
4. **Reading Agent:** `Change Detection Agent`
5. **Behavior Change:**
   - Trend confirms deterioration → **Confidence ↑**
   - Oscillating data → **Uncertainty ↑**
6. **Code Mapping:**
   - `patient_state_agent.py` → upserts
   - `change_detection_agent.py` → compares last N states

### Loop 3: Evidence Agreement → Trust / Uncertainty
**Mechanism:** Cross-modal consistency checks determine system self-trust.

1. **Event:** Agents produce conflicting or agreeing outputs (e.g., text vs. image).
2. **Recording Agent:** `Validator Agent`
3. **Qdrant Storage:** Collection `confidence_graph`
   ```json
   {
     "query_id": "Q-81",
     "agreement_score": 0.42,
     "conflict": ["text_vs_image"]
   }
   ```
4. **Reading Agent:** `Uncertainty Agent`, `Explanation Agent`
5. **Behavior Change:** Adds disclaimers to explanations; Risk Agent downgrades urgency.
6. **Code Mapping:**
   - `validator_agent.py` → writes agreement
   - `uncertainty_agent.py` → enforces safety language

### Loop 4: Evidence Quality → Explanation Quality
**Mechanism:** Curation of evidence leads to more precise and safer explanations.

1. **Event:** Evidence is curated (redundancy removed, high-confidence retained).
2. **Recording Agent:** `Evidence Curator Agent`
3. **Qdrant Storage:** Collection `agent_workspace`
   ```json
   {
     "query_id": "Q-81",
     "curated_evidence_ids": ["v12", "v19", "img_7"]
   }
   ```
4. **Reading Agent:** `Explanation Agent`
5. **Behavior Change:** Generates shorter, clearer explanations with better citations.
6. **Code Mapping:**
   - `evidence_curator_agent.py` → filters
   - `explanation_agent.py` → consumes only curated IDs

### Loop 5: Outcome Awareness (Optional / Simulated)
**Mechanism:** Future outcomes validate past risk assessments (Simulated for Hackathon).

1. **Event:** Follow-up visit shows improvement or lack thereof.
2. **Recording Agent:** `Change Detection Agent`
3. **Qdrant Storage:** Collection `decision_outcomes`
   ```json
   {
     "decision_id": "R-22",
     "outcome": "improved",
     "time_to_improve": "5 days"
   }
   ```
4. **Reading Agent:** `Risk Agent`
5. **Behavior Change:** Reinforces similar future decisions; adjusts internal trust scores.

## 4. Key Design Principle
> **Agents represent clinical responsibilities, not data modalities.**

This ensures the system is interpretable, safe, and aligned with real healthcare workflows.

## 5. Mapping to Hackathon Requirements
- **Search:** Hybrid multimodal retrieval inside agents.
- **Memory:** Persistent evolving patient & doctor state in Qdrant.
- **Recommendations:** Risk‑aware, context‑aware decision support.
- **Qdrant:** Acts as the shared memory, coordination bus, and audit log for all agents.
