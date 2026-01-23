from __future__ import annotations

from typing import Iterable, List

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.rag.config import OPENAI_CHAT_MODEL
from app.rag.vector_store import get_vector_store


def _collect_documents(messages: Iterable[BaseMessage]) -> List[Document]:
    docs: List[Document] = []
    for message in messages:
        artifact = getattr(message, "artifact", None)
        if artifact:
            for item in artifact:
                if isinstance(item, Document):
                    docs.append(item)
    return docs


def build_rag_agent(k: int = 4):
    vector_store = get_vector_store()

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=k)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_context]
    system_prompt = (
        "You are a helpful assistant. Use the retrieval tool to answer the user. "
        "If the answer is not in the retrieved context, say you don't know."
    )

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    agent = create_agent(llm, tools, system_prompt=system_prompt)
    return agent


def answer_question(question: str, k: int = 4) -> tuple[str, List[Document]]:
    agent = build_rag_agent(k=k)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    messages = result.get("messages", [])
    answer = messages[-1].content if messages else ""
    sources = _collect_documents(messages)
    return answer, sources
