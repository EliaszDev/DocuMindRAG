"""End-to-end pipeline tests with mocked LLM."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.pipeline import RAGPipeline
from src.vector_store import ChromaVectorStore


def test_pipeline_ingest_and_ask():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_dir=tmpdir, collection_name="test_pipeline")
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Mocked answer"
        pipeline = RAGPipeline(vector_store=store, generator=mock_gen)

        doc_path = Path(tmpdir) / "demo.txt"
        doc_path.write_text("RAG stands for Retrieval-Augmented Generation.")
        pipeline.ingest(doc_path)

        answer = pipeline.ask("What does RAG stand for?")
        assert answer == "Mocked answer"
        # Verify generator was called with retrieved context
        call_args = mock_gen.generate.call_args
        context_docs = call_args[0][0]
        assert any("Retrieval-Augmented" in d.page_content for d in context_docs)


def test_pipeline_reset():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_dir=tmpdir, collection_name="test_reset")
        pipeline = RAGPipeline(vector_store=store)
        doc_path = Path(tmpdir) / "demo.txt"
        doc_path.write_text("Some content here.")
        pipeline.ingest(doc_path)
        pipeline.reset()
        results = pipeline.retriever.retrieve("content")
        assert len(results) == 0
