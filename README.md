# DocuMind RAG

**Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain + ChromaDB** 

---

## Features

- **PDF & Text Ingestion** — auto-detects file type
- **Smart Chunking** — recursive splitter with overlap & metadata preservation
- **Semantic Search** — local sentence-transformer embeddings (22 MB, CPU-friendly)
- **Local LLM** — runs on Ollama, zero API cost, fully offline
- **Modern LCEL** — LangChain Expression Language composition
- **Tested** — pytest suite with mocked LLM for CI

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Pull a lightweight model:
  ```bash
  ollama pull llama3.2
  ```

### 2. Install

```bash
git clone https://github.com/EliaszDev/DocuMindRAG.git
cd DocuMindRAG
python -m venv venv
source venv/bin/activate
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
Retrieval-Augmented Generation (RAG) is a technique...

> quit
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Architecture

```
Document -> Loader -> Chunker -> Embeddings -> ChromaDB -> Retriever -> LLM -> Answer
```

| Component | File | Responsibility |
|-----------|------|----------------|
| Config | `src/config.py` | `.env`-driven pydantic settings |
| Loader | `src/document_loader.py` | PDF / text parsing with auto-detection |
| Chunker | `src/chunker.py` | Recursive text splitting with metadata |
| Vector Store | `src/vector_store.py` | ChromaDB + HuggingFace embeddings |
| Retriever | `src/retriever.py` | Top-k similarity search |
| Generator | `src/generator.py` | LCEL chain + Ollama LLM |
| Pipeline | `src/pipeline.py` | End-to-end orchestration + CLI |

---

## Tech Stack

- **LangChain** (v0.3+) — orchestration & LCEL chains
- **ChromaDB** — persistent vector storage
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) — local embeddings
- **Ollama** — local LLM inference
- **Pydantic Settings** — configuration management
- **pytest** — unit & integration tests

---

## Project Structure

```
DocuMindRAG/
├── data/sample_docs/         # Demo documents
├── src/
│   ├── __init__.py
│   ├── config.py             # Pydantic settings
│   ├── document_loader.py    # PDF & text loaders
│   ├── chunker.py            # Recursive text splitter
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── retriever.py          # Semantic search
│   ├── generator.py          # LCEL + Ollama chain
│   └── pipeline.py           # End-to-end orchestrator
├── tests/
│   ├── test_chunker.py       # Chunk size & overlap tests
│   ├── test_retriever.py     # Top-k retrieval tests
│   └── test_pipeline.py      # E2E tests with mocked LLM
├── .env.example              # Config template
├── requirements.txt          # Dependencies
├── plan.md                   # Implementation plan
└── readme.md                 # This file
```

---

## Configuration

All tunables live in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector DB storage path |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `LLM_MODEL` | `llama3.2` | Ollama model name |
| `CHUNK_SIZE` | `500` | Max chars per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `4` | Retrieved chunks per query |

---

## Usage (Library)

```python
from pathlib import Path
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest a document
pipeline.ingest(Path("my_document.pdf"))

# Ask a question
answer = pipeline.ask("What are the main findings?")
print(answer)

# Reset the vector store
pipeline.reset()
```

---

## Interview Talking Points

1. **Why ChromaDB?** Zero-config, persistent, runs locally. Ideal for demos and prototypes.
2. **Why local embeddings?** `all-MiniLM-L6-v2` is 22 MB, fast on CPU, and competitive on MTEB benchmarks.
3. **Why LCEL?** Declarative composition, streaming support, and easy observability.
4. **Extensibility** Swap Ollama for OpenRouter / GPT-4 by changing one line in `.env`.

---

## License

MIT
