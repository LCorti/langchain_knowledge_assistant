from typing import Any

import faiss
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import CrossEncoder

from .index import load_bm25_index, load_faiss_index, prepare_indexes


def get_bm25_retriever(config: dict) -> BM25Retriever:
    retriever = load_bm25_index(config)
    retriever.k = config["retriever_params"]["k"]
    return retriever


def get_faiss_retriever(config: dict) -> VectorStoreRetriever:
    store = load_faiss_index(config)
    if faiss.get_num_gpus() > 0:
        print("Moving FAISS to GPU 0.")
        resources = faiss.StandardGpuResources()
        store.index = faiss.index_cpu_to_gpu(resources, 0, store.index)
        store._gpu_resources = resources
    else:
        print("Using FAISS on CPU.")

    params = config["retriever_params"]
    search_type = params["search_type"]
    search_kwargs = {"k": params["k"]}
    if search_type == "mmr":
        search_kwargs.update(
            fetch_k=params["fetch_k"], lambda_mult=params["lambda_mult"]
        )
    elif search_type == "similarity_score_threshold":
        if config["index_params"]["distance_strategy"] == "MAX_INNER_PRODUCT" and not (
            config["embeddings"]["norm_embs"] or config["index_params"]["normalize_L2"]
        ):
            raise ValueError(
                "Threshold search with inner products requires normalized embeddings."
            )
        search_kwargs["score_threshold"] = params["score_threshold"]
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


class HybridRetriever:
    """Combines BM25 (sparse retrieval) and FAISS (dense retrieval)."""

    def __init__(
        self,
        bm25_config: dict[str, Any],
        faiss_config: dict[str, Any],
        retriever_config: dict[str, Any],
        reranker_config: dict[str, Any],
    ):
        if not 1 <= reranker_config["top_n"] <= reranker_config["rerank_k"]:
            raise ValueError("Require 1 <= top_n <= rerank_k.")
        if reranker_config["reranker_batch_size"] < 1:
            raise ValueError("reranker_batch_size must be positive.")
        self.bm25_config = bm25_config
        self.faiss_config = faiss_config
        self.retriever_config = retriever_config
        self._retriever: EnsembleRetriever | None = None

        self.reranker_model = reranker_config["reranker_model"]
        self.reranker_device = reranker_config["reranker_device"]
        self.rerank_k = reranker_config["rerank_k"]
        self.top_n = reranker_config["top_n"]
        self.reranker_batch_size = reranker_config["reranker_batch_size"]
        self._reranker: CrossEncoder | None = None

    def _load(self) -> None:
        if self._retriever is not None:
            return

        # Prepare indexes in case they are missing.
        prepare_indexes(self.bm25_config, self.faiss_config)
        self._retriever = EnsembleRetriever(
            retrievers=[
                get_bm25_retriever(self.bm25_config),
                get_faiss_retriever(self.faiss_config),
            ],
            weights=self.retriever_config["fusion"]["weights"],
            c=self.retriever_config["fusion"]["c"],
            id_key="doc_id",
        )

    def retrieve(self, query: str) -> list[Document]:
        self._load()
        documents = self._retriever.invoke(query)
        return self.rerank(query, documents)

    def retrieve_stages(self, query: str) -> dict[str, list[Document]]:
        self._load()
        bm25 = self._retriever.retrievers[0].invoke(query)
        faiss = self._retriever.retrievers[1].invoke(query)
        fusion = self._retriever.weighted_reciprocal_rank([bm25, faiss])
        stages = {"bm25": bm25, "faiss": faiss, "fusion": fusion}
        if self.reranker_model is not None:
            stages["reranker_candidates"] = fusion[: self.rerank_k]
            stages["reranker"] = self.rerank(query, fusion)
        return stages

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        # Preserve existing behavior when reranking is disabled.
        if self.reranker_model is None or not documents:
            return documents

        if self._reranker is None:
            self._reranker = CrossEncoder(
                self.reranker_model,
                device=self.reranker_device,
                max_length=8192,
            )

        candidates = documents[: self.rerank_k]
        pairs = [(query, doc.page_content) for doc in candidates]

        scores = self._reranker.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [doc for doc, _ in ranked[: self.top_n]]

    def as_runnable(self) -> Runnable[str, list[Document]]:
        return RunnableLambda(self.retrieve)
