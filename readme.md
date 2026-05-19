# 🧠 DocMind RAG

> A minimal, production-ready **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain + ChromaDB** — designed for technical interview portfolios.

---

## ✨ Features

- 📄 **PDF & Text Ingestion** — auto-detects file type
- ✂️ **Smart Chunking** — recursive splitter with overlap & metadata preservation
- 🔍 **Semantic Search** — local sentence-transformer embeddings (22 MB, CPU-friendly)
- 🤖 **Local LLM** — runs on Ollama, zero API cost, fully offline
- 🔗 **Modern LCEL** — LangChain Expression Language composition
- 🧪 **Tested** — pytest suite with mocked LLM for CI

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Pull a lightweight model:
  ```bash
  ollama pull llama3.2
  ```

### 2. Install

```bash
git clone https://github.com/yourusername/docmind-rag.git
cd docmind-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env if you want to change models or chunk sizes
```

### 4. Run Demo

```bash
python -m src.pipeline data/sample_docs/demo.txt
```

Then ask questions interactively:
```
> What is RAG?
🤖 Retrieval-Augmented Generation (RAG) is a technique...

> quit
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 🏗️ Architecture

```
Document → Loader → Chunker → Embeddings → ChromaDB → Retriever → LLM → Answer
```

| Component | File | Responsibility |
|-----------|------|----------------|
| Config | `src/config.py` | `.env`-driven settings |
| Loader | `src/document_loader.py` | PDF / text parsing |
| Chunker | `src/chunker.py` | Recursive text splitting |
| Vector Store | `src/vector_store.py` | ChromaDB + HF embeddings |
| Retriever | `src/retriever.py` | Top-k similarity search |
| Generator | `src/generator.py` | LCEL chain + Ollama LLM |
| Pipeline | `src/pipeline.py` | End-to-end orchestration |

---

## 🛠️ Tech Stack

- **LangChain** (v0.3+) — orchestration & LCEL
- **ChromaDB** — persistent vector storage
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) — local embeddings
- **Ollama** — local LLM inference
- **Pydantic Settings** — configuration management
- **pytest** — testing

---

## 📁 Project Structure

```
docmind-rag/
├── data/sample_docs/      # Demo documents
├── src/
│   ├── config.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── .env.example
├── requirements.txt
├── readme.md
└── plan.md
```

---

## 🎯 Interview Talking Points

1. **Why ChromaDB?** — Zero-config, persistent, runs locally. Perfect for demos and prototypes.
2. **Why local embeddings?** — `all-MiniLM-L6-v2` is tiny, fast, and competitive on MTEB benchmarks.
3. **Why LCEL?** — Declarative composition, streaming support, and easy observability.
4. **Extensibility** — Swap Ollama for OpenRouter / GPT-4 by changing one line in `.env`.

---

## 📄 License

MIT
