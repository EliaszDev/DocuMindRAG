"""Document loading utilities for PDF and plain text."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentLoader:
    """Load PDF and text files into LangChain Document objects."""

    @staticmethod
    def load_pdf(path: Path) -> list[Document]:
        """Load a PDF file and return a list of Documents (one per page)."""
        loader = PyPDFLoader(str(path))
        return loader.load()

    @staticmethod
    def load_text(path: Path) -> list[Document]:
        """Load a plain text file and return a single Document."""
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    @classmethod
    def load(cls, path: Path) -> list[Document]:
        """Auto-detect file type and load."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return cls.load_pdf(path)
        if suffix in (".txt", ".md", ".rst"):
            return cls.load_text(path)
        raise ValueError(f"Unsupported file type: {suffix}")
