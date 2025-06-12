# 🔍 Hybrid RAG QA Assistant

This project is a Retrieval-Augmented Generation (RAG) question-answering system that combines keyword-based and semantic search to generate accurate technical answers with inline citations. The final assistant includes a live Gradio interface, citation evaluation, and a short report.

---

## 📚 Dataset

Two JSON datasets were used:

- `top_ai_questions.json`
- `top_datascience_questions.json`

Each record contains a technical question (`title`), the full body of the post (`body`), and a source URL (`link`).

---

## 🧠 Methodology

### 🔹 1. Retrieval

- **BM25**: Lexical keyword search via `rank_bm25`
- **MiniLM**: Dense semantic search using Sentence Transformers (`all-MiniLM-L6-v2`) and FAISS

### 🔹 2. Fusion

- **Reciprocal Rank Fusion (RRF)** combines top-k BM25 and semantic results for balanced recall and precision.

### 🔹 3. Generation

- **LLM**: `mistralai/Mistral-7B-Instruct-v0.1` via Hugging Face Transformers
- **Prompt**: Top 5 fused passages are passed as context, with inline citations
- **Output**: Factual, concise answer generated with source links included

### 🔹 4. UI

- **Gradio App**: Live demo for user queries, integrated with full RAG pipeline

---

## 📊 Evaluation

| Metric     | BM25   | MiniLM |
|------------|--------|--------|
| MAP@10     | 0.733  | 0.833  |
| MRR@10     | 0.733  | 0.833  |
| nDCG@10    | 0.796  | 0.877  |

### 🔎 Manual QA Evaluation (10 Questions)

All answers were factually correct, citation-aligned, and free of hallucinations.

---

## 🚀 How to Run

### 1. Setup (Colab Recommended)

```bash
pip install -q transformers accelerate bitsandbytes faiss-cpu rank_bm25 sentence-transformers gradio
