import logging
import os

from datasets import disable_progress_bars
from transformers.utils import logging as hf_logging

from .config import (
    get_bm25_config,
    get_faiss_config,
    get_generator_config,
    get_reranker_config,
    get_retriever_config,
)
from .generate import QwenGenerator
from .pipeline import RAGPipeline
from .retriever import HybridRetriever

"""Set logging levels to keep cli clean."""
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

for name in (
    "transformers",
    "huggingface_hub",
    "sentence_transformers",
    "langchain",
    "langchain_core",
    "langchain_community",
):
    logging.getLogger(name).setLevel(logging.ERROR)

disable_progress_bars()

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
"""== == == == == == == == == == == =="""


def main() -> None:
    bm25_config = get_bm25_config()
    faiss_config = get_faiss_config()
    generator_config = get_generator_config()
    retriever_config = get_retriever_config()
    reranker_config = get_reranker_config()

    print("Loading retrievers...")
    retriever = HybridRetriever(
        bm25_config, faiss_config, retriever_config, reranker_config
    )
    generator = QwenGenerator(generator_config)
    pipeline = RAGPipeline(retriever, generator, generator_config)
    run_cli_loop(pipeline)


def run_cli_loop(pipeline) -> None:
    print("Retriever ready.")
    print("\n")
    print("=" * 50)
    print("Ask your question about LangChain:")
    print("=" * 50)

    while True:
        try:
            user_input = input("> ").strip()
        except KeyboardInterrupt, EOFError:
            break

        if user_input == "/bye":
            break

        if not user_input:
            continue

        print("\n")
        response = pipeline.run(user_input)
        print("\nAgent:")
        print(f"{response['answer']}\n")
        print("\tSources:")
        for i, doc in enumerate(response["sources"]):
            print(f"\t[{i + 1}]: {doc.metadata.get('doc_id')}")
        print("\n")

    print("Bye!\n")


if __name__ == "__main__":
    main()
