# retrieve + prompt + generate

from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from .generate import QwenGenerator
from .retriever import HybridRetriever


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: QwenGenerator,
        generator_config: dict[str, Any],
    ):
        self.retriever = retriever
        self.generator = generator or QwenGenerator(generator_config)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", generator_config["system_prompt"]),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )
        self.chain = self._build_chain()
        self.top_k = 10

    def _build_chain(self) -> Runnable[str, dict[str, Any]]:
        answer = (
            RunnableLambda(
                lambda x: {"context": format_docs(x["docs"]), "question": x["question"]}
            )
            | self.prompt
            | self.generator.as_runnable()
        )
        return (
            RunnableParallel(
                docs=(
                    self.retriever.as_runnable()
                    | RunnableLambda(lambda docs: docs[: self.top_k])
                ),
                question=RunnablePassthrough(),
            )
            | RunnablePassthrough.assign(answer=answer)
            | RunnableLambda(lambda x: {"answer": x["answer"], "sources": x["docs"]})
        )

    def run(self, query: str) -> dict[str, Any]:
        """Returns {"answer": str, "sources": list[Document]}."""
        return self.chain.invoke(query)
