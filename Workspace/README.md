# MediSync Workspace

This directory contains the source code for the MediSync agent system, built using the **Google Agent Development Kit (ADK)** framework pattern.

## Overview
MediSync is a multi-agent clinical decision support system designed to assist doctors by synthesizing patient data, identifying risks, and explaining reasoning with cited evidence.

The system is organized into specialized agent groups:
- **Orchestration**: Directs the flow of execution.
- **Ingestion**: Handles data intake (OCR, Embeddings).
- **Reasoning**: Core clinical logic (Patient State, Risk, Medical Codes).
- **Validation**: Ensures safety and consistency.
- **Explanation**: Generates user-facing outputs.

## Prerequisites
- **Python 3.10+**
- **Qdrant**: Vector database (needs to be running, e.g., via Docker)

## Installation

1. **Install Dependencies**
   ```bash
   pip install google-adk qdrant-client
   ```
   *Note: Since the system currently uses a mock ADK config (`agents/adk_config.py`), you can run it without the actual `google-adk` package if testing the structure, but a real deployment requires it.*

2. **Environment Setup**
   Ensure your Qdrant instance is accessible and set environment variables:
   ```bash
   export QDRANT_HOST="localhost"
   export QDRANT_PORT=6333
   ```

## Directory Structure
```text
Workspace/
├── agents/             # ADK Agent implementations
│   ├── adk_config.py   # Shared ADK configuration & Base Class
│   ├── orchestration/  # OrchestratorAgent
│   ├── ingestion/      # IngestionAgent
│   ├── reasoning/      # PatientState, Risk, etc.
│   ├── validation/     # Validator, Uncertainty, Curator
│   └── explanation/    # Explanation agents
├── services/           # Helper services (Qdrant, LLM, OCR)
├── server/             # API Server (FastAPI)
└── docs/               # Architecture & Design docs
```

## Usage

### Running the Orchestrator (Python)
You can instantiate and run the Orchestrator agent directly from Python:

```python
import sys
import os

# Ensure Workspace is in python path
sys.path.append(os.getcwd())

from agents.orchestration.orchestrator_agent import OrchestratorAgent

# Initialize
orchestrator = OrchestratorAgent()

# Run a query
query_payload = {
    "query": "Analyze patient P-90210 for signs of deterioration."
}

result = orchestrator.run(query_payload)

print(f"Status: {result['status']}")
print(f"Response: {result['response']}")
print(f"Trace: {result['trace']}")
```

### Running the API Server
To start the FastAPI server:
```bash
cd server
uvicorn main:app --reload
```

## Architecture
For a deep dive into the system architecture, agent responsibilities, and feedback loops, see [docs/architecture.md](docs/architecture.md).
