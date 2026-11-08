from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BM25_CONFIG_PATH = Path("config/bm25_config.yaml")
FAISS_CONFIG_PATH = Path("config/faiss_config.yaml")
GENERATOR_CONFIG_PATH = Path("config/generator_config.yaml")
RETRIEVER_CONFIG_PATH = Path("config/retriever_config.yaml")
RERANKER_CONFIG_PATH = Path("config/reranker_config.yaml")
EVALUATION_CONFIG_PATH = Path("config/evaluation.yaml")
RETRIEVAL_DIAGNOSTICS_CONFIG_PATH = Path("config/retrieval_diagnostics.yaml")
GENERATION_EVALUATION_CONFIG_PATH = Path("config/generation_evaluation.yaml")


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the checkout, not the working directory."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_yaml_config(cfg_path: str | Path) -> dict:
    full_path = resolve_project_path(cfg_path)
    with full_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a configuration mapping at {full_path}")
    return config


def get_bm25_config(cfg_path: str | Path = BM25_CONFIG_PATH) -> dict:
    config = _load_yaml_config(cfg_path)
    config["main"]["persist_path"] = str(
        resolve_project_path(config["main"]["persist_path"])
    )
    return config


def get_faiss_config(cfg_path: str | Path = FAISS_CONFIG_PATH) -> dict:
    config = _load_yaml_config(cfg_path)
    config["main"]["persist_path"] = str(
        resolve_project_path(config["main"]["persist_path"])
    )
    return config


def get_retriever_config(cfg_path: str | Path = RETRIEVER_CONFIG_PATH) -> dict:
    config = _load_yaml_config(cfg_path)
    return config


def get_generator_config(cfg_path: str | Path = GENERATOR_CONFIG_PATH) -> dict:
    config = _load_yaml_config(cfg_path)
    return config


def get_reranker_config(cfg_path: str | Path = RERANKER_CONFIG_PATH) -> dict:
    config = _load_yaml_config(cfg_path)
    return config


def get_evaluation_config(cfg_path: str | Path = EVALUATION_CONFIG_PATH) -> dict:
    config = _load_yaml_config(cfg_path)
    for key in ("cache_dir", "results_dir"):
        config[key] = str(resolve_project_path(config[key]))
    return config


def get_generation_evaluation_config(
    cfg_path: str | Path = GENERATION_EVALUATION_CONFIG_PATH,
) -> dict:
    """Load evaluation settings; citation-only runs do not need a query cache."""
    config = _load_yaml_config(cfg_path)
    config["results_dir"] = str(resolve_project_path(config["results_dir"]))
    if "cache_dir" in config:
        config["cache_dir"] = str(resolve_project_path(config["cache_dir"]))
    return config


def get_retrieval_diagnostics_config(
    cfg_path: str | Path = RETRIEVAL_DIAGNOSTICS_CONFIG_PATH,
) -> dict:
    config = _load_yaml_config(cfg_path)
    for key in ("cache_dir", "results_dir"):
        config[key] = str(resolve_project_path(config[key]))
    return config
