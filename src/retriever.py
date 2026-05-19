"""Semantic retriever backed by ChromaDB."""

from langchain_core.documents import Document

from src.vector_store import ChromaVectorStore


class SemanticRetriever:
    """Top-k similarity search over the vector store."""

    def __init__(self, vector_store: ChromaVectorStore, k: int = 4):
        self.retriever = vector_store.as_retriever(k=k)

    def retrieve(self, query: str) -> list[Document]:
        """Return the k most relevant documents for the query."""
        return self.retriever.invoke(query)
