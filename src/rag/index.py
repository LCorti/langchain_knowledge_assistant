import json
import pickle
import re

import faiss
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import resolve_project_path
from .data import load_source_corpus

# Normalization of volatile trace strings
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")
_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_LINE = re.compile(r"\bline \d+\b", re.IGNORECASE)

# Versions first (optional leading "v"), then any word-character run
_TOKEN = re.compile(r"v?\d+(?:\.\d+)+|\w+", re.UNICODE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def normalize(text: str) -> str:
    text = _UUID.sub(" ", text)
    text = _HEX.sub(" ", text)
    text = _LINE.sub("line", text)
    return text


def code_aware_preprocessing(text: str) -> list[str]:
    """
    Custom tokenization for BM25
    - Normalisation takes care of UUIDs, HEX, and line numbers.
    - _TOKEN matches version numbers and words delimited by punctuation
        langchain_core.prompts -> 'langchain_core' 'prompts'
    - matches are taken in lowercase but camelCase one are split
    """

    tokens = []

    for identifier in _TOKEN.findall(normalize(text)):
        normalized = identifier.lower()
        tokens.append(normalized)

        # Preserve the full identifier and also expose its component words.
        parts = [
            part.lower()
            for part in _CAMEL_BOUNDARY.sub("_", identifier).split("_")
            if part
        ]
        if parts != [normalized]:
            tokens.extend(parts)

    return tokens


def bm25_index_exists(config: dict) -> bool:
    return resolve_project_path(config["main"]["persist_path"]).is_file()


def faiss_index_exists(config: dict) -> bool:
    path = resolve_project_path(config["main"]["persist_path"])
    return all((path / name).is_file() for name in ("index.faiss", "index.pkl"))


def build_bm25_index(docs: list[Document], config: dict) -> BM25Retriever:
    bm25 = BM25Retriever.from_documents(
        documents=docs,
        bm25_params=config["index_params"],
        preprocess_func=code_aware_preprocessing,
        **config["retriever_params"],
    )
    path = resolve_project_path(config["main"]["persist_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(bm25, stream)
    print(f"BM25 index created at {path}")
    return bm25


def load_bm25_index(config: dict) -> BM25Retriever:
    """Load only indexes created locally."""
    path = resolve_project_path(config["main"]["persist_path"])
    if not bm25_index_exists(config):
        raise FileNotFoundError(
            f"BM25 index not found at {path}. "
            "Prepare indexes with scripts/build_index.py first."
        )
    with path.open("rb") as stream:
        return pickle.load(stream)


def _get_embeddings(config: dict) -> HuggingFaceEmbeddings:
    settings = config["embeddings"]
    return HuggingFaceEmbeddings(
        model_name=settings["emb_model"],
        model_kwargs={"device": settings["device"]},
        encode_kwargs={
            "normalize_embeddings": settings["norm_embs"],
            "batch_size": settings["batch_size"],
        },
        show_progress=True,
    )


def _normalized_inner_product_relevance(score: float) -> float:
    # Unit-vector dot products are cosine similarities in [-1, 1]. Map them
    # monotonically to [0, 1]; LangChain's default IP conversion reverses positives.
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _faiss_options(config: dict) -> dict:
    params = config["index_params"]
    options = {
        "distance_strategy": DistanceStrategy[params["distance_strategy"]],
        "normalize_L2": params["normalize_L2"],
    }
    if params["distance_strategy"] == "MAX_INNER_PRODUCT" and (
        config["embeddings"]["norm_embs"] or params["normalize_L2"]
    ):
        options["relevance_score_fn"] = _normalized_inner_product_relevance
    return options


def _faiss_build_settings(config: dict) -> dict:
    return {
        "version": 1,
        "emb_model": config["embeddings"]["emb_model"],
        "norm_embs": config["embeddings"]["norm_embs"],
        "distance_strategy": config["index_params"]["distance_strategy"],
        "normalize_L2": config["index_params"]["normalize_L2"],
    }


def build_faiss_index(docs: list[Document], config: dict) -> FAISS:
    store = FAISS.from_documents(
        documents=docs,
        embedding=_get_embeddings(config),
        **_faiss_options(config),
    )
    path = resolve_project_path(config["main"]["persist_path"])
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    (path / "build_config.json").write_text(
        json.dumps(_faiss_build_settings(config), indent=2) + "\n"
    )
    print(f"FAISS vector store created at {path}")
    return store


def load_faiss_index(config: dict) -> FAISS:
    """Load a trusted local store with the same distance settings used to build it."""
    path = resolve_project_path(config["main"]["persist_path"])
    if not faiss_index_exists(config):
        raise FileNotFoundError(
            f"FAISS index missing or incomplete at {path}. "
            "Prepare indexes with scripts/build_index.py first."
        )
    store = FAISS.load_local(
        str(path),
        embeddings=_get_embeddings(config),
        allow_dangerous_deserialization=True,  # Only load trusted, locally built indexes.
        **_faiss_options(config),
    )
    expected_metric = (
        faiss.METRIC_INNER_PRODUCT
        if config["index_params"]["distance_strategy"] == "MAX_INNER_PRODUCT"
        else faiss.METRIC_L2
    )
    if store.index.metric_type != expected_metric:
        raise ValueError(
            f"FAISS distance metric differs from {path}. Rebuild the index."
        )
    return store


def prepare_indexes(bm25_config: dict, faiss_config: dict) -> None:
    """Build missing indexes from one corpus load; load existing ones."""
    build_sparse = not bm25_index_exists(bm25_config)
    build_dense = not faiss_index_exists(faiss_config)
    if not build_sparse and not build_dense:
        print("Both indexes already exist.")
        return

    corpus = load_source_corpus()
    if build_sparse:
        build_bm25_index(corpus, bm25_config)
    if build_dense:
        build_faiss_index(corpus, faiss_config)
