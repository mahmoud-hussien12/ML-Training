## Project: Retrieval-Augmented Generation (RAG) – From Scratch

This project demonstrates a **production-oriented Retrieval-Augmented Generation (RAG) system**, built step by step with a strong focus on **engineering clarity, modularity, and real-world trade-offs**.

The goal is to go beyond “toy RAG demos” and show how RAG systems are **designed, reasoned about, and operated** in practice.

---

## 🎯 Current Scope (Week 1)

### ✅ Document Ingestion & Indexing Pipeline

Implemented a complete ingestion pipeline that transforms raw documents into a searchable vector index.

### Pipeline Overview

```
Raw documents (.txt)
      ↓
Document Loader
      ↓
Chunking (overlapping windows)
      ↓
Embedding
      ↓
FAISS Vector Store
```

---

## 🧱 Components Implemented

### 1️⃣ Document Loading

* Loads raw text documents from disk
* Attaches basic metadata (source filename)
* Designed to be easily extended to PDFs, HTML, or databases

**Why it matters:**
Clear separation between data sources and downstream processing improves maintainability and extensibility.

---

### 2️⃣ Chunking Strategy

* Sliding-window chunking with configurable:

  * `chunk_size`
  * `overlap`
* Preserves context across chunk boundaries
* Chunk by separator (\n) to get accurate match if we know the information is line separated

**Why it matters:**
Chunking directly affects retrieval quality. Overlap reduces semantic loss when relevant information spans multiple chunks.

---

### 3️⃣ Embedding Abstraction

* Clean embedder interface
* Current implementation uses OpenAI embeddings
* Designed to support swapping:

  * OpenAI → open-source models
  * Cloud → local inference

**Why it matters:**
Embedding abstraction prevents vendor lock-in and supports cost/performance experimentation.

---

### 4️⃣ FAISS Vector Store

* Uses FAISS for efficient similarity search
* Stores embeddings alongside rich metadata:

  * source
  * chunk_id
  * original text
* Supports:

  * incremental additions
  * persistence to disk

**Why it matters:**
Vector databases are core infrastructure in RAG systems. Explicit control over indexing and metadata enables filtering, debugging, and monitoring.

---

### 5️⃣ Ingestion Orchestration

* End-to-end ingestion flow:

  * load → chunk → embed → index → persist
* Single entry point for indexing new data
* Designed for repeatable, idempotent runs

---

## Retrieval Tuning

Implemented a configurable retrieval layer with support for:

 * Top-k tuning to balance recall vs precision
 * Score thresholds to filter weak semantic matches
 * Metadata-based filtering (e.g. source-level constraints)

Retrieval behavior can be tuned without retraining embeddings, enabling
fast iteration and safer RAG behavior in production systems.

---

## RAG Query Pipeline

Implemented a full Retrieval-Augmented Generation pipeline:

 * Query embedding
 * Tuned retrieval (top-k, thresholds, metadata filters)
 * Explicit prompt construction with retrieved context
 * Deterministic LLM generation (temperature=0)

The pipeline is fully inspectable, enabling debugging of retrieval quality,
prompt grounding, and answer faithfulness.

---

## Reranking Layer

Added a cross-encoder reranker to improve retrieval precision.

Pipeline:
FAISS (fast recall) → Cross-encoder (precision) → LLM

Benefits:
- More relevant context
- Reduced hallucination risk
- Better answer quality

Trade-off:
- Increased latency due to additional model inference
 
---

## 🧠 Design Principles

* **Explicit abstractions** over magic frameworks
* **Configurable chunking and embedding**
* **Inspectability** (metadata retained at every step)
* **Production-first mindset**, not notebook demos

---

## 🛠 Tech Stack (So Far)

* Python
* FAISS
* OpenAI Embeddings
* Modular project structure (clean separation of concerns)

---

## Evaluation

Implemented basic evaluation for both retrieval and answer quality.

### Retrieval Evaluation
- Keyword-based recall
- Measures whether relevant concepts appear in retrieved chunks

### Answer Evaluation
- Simple overlap scoring with expected answers
- Used as a lightweight proxy for correctness

These evaluations allow isolating issues between retrieval and generation.

Future improvements:
- LLM-based evaluation (faithfulness, relevance)
- Human evaluation
- Benchmark datasets

---

## 📌 Why This Project Matters

Many roles mention RAG, vector databases, and LLMs without depth.
This project demonstrates **how those systems are actually built**, not just used.

---
