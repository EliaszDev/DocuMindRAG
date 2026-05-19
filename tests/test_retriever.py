"""Tests for semantic retrieval."""

import tempfile
from pathlib import Path

from langchain_core.documents import Document

from src.retriever import SemanticRetriever
from src.vector_store import ChromaVectorStore


def test_retriever_finds_relevant_chunks():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_dir=tmpdir, collection_name="test_retriever")
        docs = [
            Document(page_content="Cats are small carnivorous mammals."),
            Document(page_content="Dogs are domesticated mammals."),
            Document(page_content="The sky is blue because of Rayleigh scattering."),
        ]
        store.add_documents(docs)
        retriever = SemanticRetriever(store, k=2)
        results = retriever.retrieve("Tell me about felines")
        assert len(results) == 2
        assert any("cat" in r.page_content.lower() for r in results)


def test_retriever_respects_top_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_dir=tmpdir, collection_name="test_topk")
        docs = [Document(page_content=f"Document number {i}") for i in range(10)]
        store.add_documents(docs)
        retriever = SemanticRetriever(store, k=3)
        results = retriever.retrieve("number")
        assert len(results) == 3
