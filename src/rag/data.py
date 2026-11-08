from datasets import load_dataset
from langchain_core.documents import Document


def load_source_corpus(
    corpus_path: str = "freshstack/corpus-oct-2024",
    corpus_name: str = "langchain",
    corpus_split: str = "train",
) -> list[Document]:
    """Load the pre-chunked corpus, preserving chunk IDs and source metadata."""
    print(f"Loading corpus {corpus_path} -- {corpus_name} -- {corpus_split}")
    corpus = load_dataset(path=corpus_path, name=corpus_name, split=corpus_split)
    required = {"_id", "text", "metadata"}
    missing = required - set(corpus.column_names)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")
    return [
        Document(
            page_content=row["text"],
            metadata={"doc_id": row["_id"], "chunk_metadata": row["metadata"]},
        )
        for row in corpus
    ]
