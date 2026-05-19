"""LCEL chain for answer generation using Ollama."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from src.config import settings


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use only the provided context to answer the question. "
    "If the context does not contain the answer, say 'I don't know based on the provided documents.'."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class LCELGenerator:
    """LangChain Expression Language generator backed by a local Ollama model."""

    def __init__(self, model: str = settings.llm_model):
        self.llm = ChatOllama(model=model, temperature=0.0)
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | PROMPT_TEMPLATE
            | self.llm
            | StrOutputParser()
        )

    def generate(self, context: list[Document], question: str) -> str:
        """Generate an answer from retrieved documents and a user question."""
        formatted = _format_docs(context)
        return self.chain.invoke({"context": formatted, "question": question})
