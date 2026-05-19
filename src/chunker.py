"""Text chunking with recursive character splitting."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


class TextChunker:
    """Split documents into overlapping chunks while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents into smaller chunks."""
        return self.splitter.split_documents(documents)
