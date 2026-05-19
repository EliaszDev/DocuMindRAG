"""End-to-end RAG pipeline orchestrator."""

from pathlib import Path

from src.chunker import TextChunker
from src.document_loader import DocumentLoader
from src.generator import LCELGenerator
from src.retriever import SemanticRetriever
from src.vector_store import ChromaVectorStore


class RAGPipeline:
    """Orchestrates ingestion, storage, retrieval, and generation."""

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        generator: LCELGenerator | None = None,
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.vector_store = vector_store or ChromaVectorStore()
        self.retriever = SemanticRetriever(self.vector_store)
        self.generator = generator or LCELGenerator()

    def ingest(self, file_path: Path) -> None:
        """Load, chunk, and store a document."""
        docs = self.loader.load(file_path)
        chunks = self.chunker.split(docs)
        self.vector_store.add_documents(chunks)

    def ask(self, question: str) -> str:
        """Retrieve relevant chunks and generate an answer."""
        context = self.retriever.retrieve(question)
        return self.generator.generate(context, question)

    def reset(self) -> None:
        """Wipe the vector database."""
        self.vector_store.clear_collection()


if __name__ == "__main__":
    import sys

    from src.config import settings

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <path_to_document>")
        sys.exit(1)

    doc_path = Path(sys.argv[1])
    pipeline = RAGPipeline()
    print(f"Ingesting {doc_path} ...")
    pipeline.ingest(doc_path)
    print("Ready for questions. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip().lower() in ("quit", "exit", "q"):
            break
        answer = pipeline.ask(query)
        print(f"🤖 {answer}\n")
