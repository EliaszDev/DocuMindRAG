"""Tests for the text chunker."""

from langchain_core.documents import Document

from src.chunker import TextChunker
from src.config import settings


def test_chunker_splits_long_text():
    text = "word " * 200  # 1000 chars
    docs = [Document(page_content=text)]
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.split(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 100 + 10  # allow small overshoot


def test_chunker_preserves_metadata():
    docs = [Document(page_content="Hello world", metadata={"source": "demo.txt"})]
    chunker = TextChunker()
    chunks = chunker.split(docs)
    assert all(chunk.metadata.get("source") == "demo.txt" for chunk in chunks)


def test_chunker_overlap():
    text = "a b c d e f g h i j"
    docs = [Document(page_content=text)]
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.split(docs)
    assert len(chunks) >= 2
    # Adjacent chunks should share some content
    assert any(word in chunks[0].page_content for word in chunks[1].page_content.split())
