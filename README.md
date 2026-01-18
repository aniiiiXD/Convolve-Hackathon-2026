# MediSync Workspace (Gen 2)

This is the fused codebase combining the original backend logic with the Agent Workspace architecture. It features a unified service layer and dedicated CLIs for Doctors and Patients.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Qdrant (running locally or cloud)

### 2. Installation
```bash
pip install qdrant-client fastembed python-dotenv rich pydantic-settings
```

### 3. Configuration
Ensure you have a `.env` file in the root or parent directory with:
```env
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="your-key-if-any"
```

---

## 🖥️ User Interfaces

### 🩺 Doctor Portal (Clinical Agent)
Login as **Dr_Strange** to access the Doctor Portal.
- **Features**: Ingest Notes, Hybrid Search, Discovery API.
```bash
python3 medisync/cli/doctor_cli.py
```
*Tip: Try "Discover cases context: diabetes"*

### 👤 Patient Companion
Login as **P-101** to access your personal health companion.
- **Features**: Symptom Diary, Personal History, Health Insights (Recommendations).
```bash
python3 medisync/cli/patient_cli.py
```
*Tip: Try "Any insights?" after logging symptoms.*

---

## ✅ Automated Verification
Run the verification script to test the entire pipeline (Ingestion -> Search -> Privacy Isolation).
```bash
python3 verify.py
```

## 📂 Directory Structure
- **`medisync/core/`**: Configuration & Database connections.
- **`medisync/services/`**: Feature logic (Qdrant, Auth, Embeddings).
- **`medisync/agents/`**: Reasoning agents (Doctor, Patient).
- **`medisync/cli/`**: Terminal user interfaces.
