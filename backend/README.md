# MediSync: Clinical Intelligence Agent

> **"Not just a search engine, but a reasoning engine for clinical data."**

MediSync is a multimodal medical agent designed to ingest, understand, and reason across heterogenous patient data (Clinical Notes, Imaging, ICD Codes). It uses a sophisticated **Hybrid Retrieval (Dense + Sparse/SPLADE + Vision)** memory system built on Qdrant.

## 🧠 Agentic Architecture

Unlike traditional microservices, MediSync is built as an **Agentic Workflow**:

1.  **Perception (Ingestion)**:
    *   **tools/ingest**: converting raw text/images into a semantic latent space using 3 different neural models (BGE-Base, SPLADE-PP, CLIP).
    *   **memory/qdrant**: Storing named vectors with Binary Quantization for extreme performance.

2.  **Recall (Retrieval)**:
    *   **tools/search**: Reciprocal Rank Fusion (RRF) to combine semantic understanding with keyword precision.
    *   **Dynamic Filtering**: Strictly enforces Multi-tenancy boundaries (`clinic_id`).

3.  **Reasoning (Coming Soon)**:
    *   Synthesizes retrieved context to answer complex clinical queries ("Identify patients at risk of...")

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install the agent dependencies
pip install -r requirements.txt
pip install fastembed[image] qdrant-client
```

Ensure your `.env` contains your Memory (Qdrant) credentials:
```
QDRANT_URL=...
QDRANT_API_KEY=...
```

### 2. Run the Agent Server

```bash
cd backend
uvicorn app.main:app --reload
```

### 3. Verify Capability

We have provided a self-testing script to verify the agent's memory systems:

```bash
python verify_backend.py
```

## 🛠 API Capabilities

| Tool | Endpoint | Description |
| :--- | :--- | :--- |
| **Ingest Tool** | `POST /api/v1/ingest` | Memorize a clinical record (creates Dense + Sparse + Metadata). |
| **Recall Tool** | `POST /api/v1/search` | perform Hybrid RRF search to find relevant context. |

## 🏗 Tech Stack

*   **Brain**: Python 3.12 (FastAPI)
*   **Memory**: Qdrant (Hybrid Search, Multitenancy)
*   **Encoders**: FastEmbed (On-device inference)
    *   *Dense*: `BAAI/bge-base-en`
    *   *Sparse*: `prithivida/Splade_PP_en_v1`
    *   *Vision*: `Qdrant/clip-ViT-B-32-vision`
