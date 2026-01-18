---
name: qdrant-docs
description: >
  Complete Qdrant vector database documentation and reference. Use when working with Qdrant
  for vector search, semantic search, or building AI applications with embeddings. Covers
  core concepts, architecture, deployment options, scaling, optimization, integration guides,
  and advanced features. Use whenever the user asks about Qdrant setup, configuration,
  querying, optimization, or troubleshooting.
---

# Qdrant Documentation

This skill provides comprehensive Qdrant vector database documentation to help you build, deploy, and optimize vector search applications.

## What is Qdrant?

Qdrant is an AI-native vector database and semantic search engine for extracting meaningful information from unstructured data using embeddings.

## When to Use This Skill

Use this skill when:
- Setting up or configuring Qdrant (local or cloud)
- Working with collections, points, vectors, or payloads
- Implementing vector search or hybrid search
- Optimizing performance (indexing, quantization, sharding)
- Scaling Qdrant (distributed deployment, replication)
- Integrating Qdrant with applications (Python, JavaScript, etc.)
- Troubleshooting Qdrant issues
- Understanding advanced features (MUVERA, ACORN, multi-vector search)

## Documentation Structure

The complete documentation is in `references/qdrant-documentation.txt` with:
- 39 pages covering all Qdrant topics
- Core concepts, architecture, deployment options
- API references and code examples
- Performance optimization guides
- Integration tutorials

## Quick Reference

### Common Tasks

**Creating a collection:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
```

**Inserting points:**
```python
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 384 dimensions
            payload={"text": "sample document"}
        )
    ]
)
```

**Searching:**
```python
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],
    limit=5
)
```

## How to Use

1. **For specific topics:** Read `references/qdrant-documentation.txt` and search for relevant sections
2. **For overview:** Check the table of contents in `references/qdrant-toc.txt`
3. **For quick lookup:** Use grep/search to find specific terms in the documentation

## Key Topics Covered

- **Deployment:** OSS, Managed Cloud, Hybrid Cloud, Private Cloud
- **Core Concepts:** Collections, Points, Vectors, Payloads, Indexing
- **Search:** Similarity search, Hybrid retrieval, Filtering, Multi-stage queries
- **Optimization:** Memory management, Quantization, HNSW tuning
- **Scaling:** Sharding, Replication, Distributed deployment
- **Advanced:** Multi-vector search, MUVERA, ACORN, Strict mode
- **Integration:** Client libraries, FastEmbed, Cloud Inference

## Reference Files

- `references/qdrant-documentation.txt` - Complete formatted documentation (10,529 lines)
- `references/qdrant-toc.txt` - Table of contents with page titles and URLs

Read these files as needed to answer specific questions about Qdrant functionality, configuration, or best practices.
