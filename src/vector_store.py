"""ChromaDB vector store wrapper with HuggingFace embeddings."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


class ChromaVectorStore:
    """Persistent ChromaDB collection with local sentence-transformer embeddings."""

    def __init__(
        self,
        persist_dir: str | Path = settings.chroma_persist_dir,
        collection_name: str = "docmind",
        embed_model: str = settings.embed_model,
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir),
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Add or upsert documents into the collection."""
        self.vectorstore.add_documents(documents)

    def as_retriever(self, k: int = settings.top_k):
        """Return a LangChain retriever for similarity search."""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        self.vectorstore.reset_collection()
