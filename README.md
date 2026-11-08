# LangChain Knowledge Assistant

This project sets up a prototype for a knowledge assistant on technical quetsions related to LangChain.

The repository includes
- RAG pipeline on pre-chunked FreshStack passages, retrieval, reranking, and answer generation using Qwen3.5-9B
- an interactive CLI
- retrieval diagnostics
- held-out retrieval evaluation (answer similarity, nugget coverage, and citation support)

## Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Interactive use](#interactive-use)
- [Repository structure](#repository-structure)
- [Configuration and indexes](#configuration-and-indexes)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Limitations and improvements](#limitations-and-improvements)

## Architecture

```text
Question
  ├─ BM25: code-aware tokenization, up to 100 passages
  └─ FAISS: dense similarity search, up to 100 passages
       ↓
Weighted reciprocal-rank fusion (BM25/dense weights 0.5/0.5, c=60)
       ↓
Cross-encoder reranking of the first 100 fused passages → top 50
       ↓
First 10 passages, numbered [1] … [10]
       ↓
Qwen answer + source document IDs
```

These are the interactive CLI defaults, not fixed settings for every evaluation.

| Component | Default |
|---|---|
| Sparse retrieval | BM25 with code-aware preprocessing |
| Dense embeddings | `Alibaba-NLP/gte-modernbert-base`, normalized embeddings |
| Vector search | FAISS, maximum inner product (cosine ranking for normalized vectors) |
| Reranker | `Alibaba-NLP/gte-reranker-modernbert-base` |
| Answer generator | `Qwen/Qwen3.5-9B`, bfloat16, automatic device placement |
| Source corpus | [`freshstack/corpus-oct-2024`](https://huggingface.co/datasets/freshstack/corpus-oct-2024), `langchain`, upstream `train` split |
| Evaluation queries | [`freshstack/queries-oct-2024`](https://huggingface.co/datasets/freshstack/queries-oct-2024), `langchain`, upstream `test` split |

- A hybrid retriever was chosen to capture semantics (sparse) as well as meaning (dense).
- Given that the source corpus contains a mix of natural language, Python code, LangChain code, and stack traces, this prototype implements a custom preprocessing function to tokenize texts for BM25 such that code-related sequences (e.g., package names, camel-casing, etc.) are decently handled.
- The corpus already contains chunks; ingestion preserves their IDs and metadata.
- Generation defaults to sampling (`temperature: 0.7`), thinking disabled, and at most 1,024 new tokens.
- The prompt requests citations immediately after factual sentences, short claim groups, and code examples.

## Setup

### Requirements

- Python **3.14 or newer**, as declared in `pyproject.toml`.
- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- A CUDA-capable NVIDIA GPU for the supplied configuration. The dependency set includes `faiss-gpu`; Linux PyTorch packages use the CUDA 13.0 wheel index. Embeddings, reranking, and answer similarity default to `cuda`.
- The prototype was built using 1x NVIDIA RTX PRO 6000 Blackwell.

*This is not a CPU-only project/installation*.

Run commands below from the repository root:

```sh
uv sync
```

- The first use of each model or dataset can download it.
- Once the required assets are cached, inference and judging run locally
- No hosted generation or judge API in this prototype.

### Optional custom Hugging Face cache

To keep model/dataset downloads under the checkout, you can set `HF_HOME` via the following helper which creates this directory within this project and adds or replaces `HF_HOME` in the `~/.bashrc` profile.

```sh
bash make_hf_cache.sh
```

## Optional authentication

The configured assets are public, so authentication is optional. The regular `knowledge-assistant` command does not load `.env`. If you want the repository's `.env`-loading wrapper, put `HF_TOKEN` in the ignored `.env` file and use:

```sh
uv run python main_cli.py
```

That wrapper logs in when a token is present, then starts the same CLI.

## Interactive use

```sh
uv run knowledge-assistant
```

Enter a LangChain question at the `>` prompt. Each response includes the answer and a numbered list of source `doc_id` values matching the context citation order.

Use `/bye`, Ctrl-C, or end-of-input to exit.

The first question builds any missing BM25/FAISS indexes and loads the retrieval and generation models, so it can be substantially slower than subsequent questions. Each question is independent: earlier turns are not included in the prompt.

## Repository structure

```text
.
├── README.md
├── pyproject.toml                 # Dependencies and knowledge-assistant entry point
├── uv.lock                       # Dependency lockfile
├── .python-version               # Local Python version selection
├── main_cli.py                   # Optional .env/authentication wrapper
├── make_hf_cache.sh               # Optional cache-location shell-profile helper
├── config/
│   ├── bm25_config.yaml           # Sparse index and retrieval depth
│   ├── faiss_config.yaml          # Embeddings, vector index, search settings
│   ├── retriever_config.yaml      # Rank-fusion weights and constant
│   ├── reranker_config.yaml       # Interactive reranker and candidate/output depths
│   ├── generator_config.yaml      # Qwen model, decoding, and answer prompt
│   ├── evaluation.yaml            # Held-out retrieval evaluation
│   ├── retrieval_diagnostics.yaml # Dev split and named retrieval experiments
│   └── generation_evaluation.yaml # Answer/nugget/citation evaluation settings
├── src/rag/
│   ├── cli.py                    # Interactive input/output loop
│   ├── config.py                 # YAML loading and checkout-relative paths
│   ├── data.py                   # FreshStack chunks → LangChain Documents
│   ├── index.py                  # BM25/FAISS persistence and preprocessing
│   ├── retriever.py              # Hybrid retrieval, stage capture, reranking
│   ├── generate.py               # Qwen loading and LangChain Runnable adapter
│   ├── pipeline.py               # Retrieve → numbered context → answer + sources
│   └── evaluation/
│       ├── prepare_queries.py    # Reproducible local dev/test split cache
│       ├── retrieval.py          # Final-ranking retrieval evaluation
│       ├── retrieval_diagnostics.py # Per-stage comparisons and examples
│       ├── metrics.py            # Retrieval metric implementations
│       ├── generation.py         # Answer-quality and citation-only runners
│       ├── answer_similarity.py  # Jina paired answer/gold cosine similarity
│       ├── answer_stats.py       # Compiles eval for answer quality
│       ├── nugget_judge.py       # Gold-nugget coverage judging
│       ├── citation_parsing.py   # Factual-unit/citation extraction and validation
│       ├── citation_stats.py     # Compiles eval for citation support
│       ├── citation_judge.py     # Cited-passage support judging
│       └── citation_metrics.py   # Deterministic citation metrics
├── tests/                        # unittest tests with fixtures/mocked model calls
├── data/                         # Generated indexes and cached query splits (ignored)
├── hf_cache/                     # Optional model/dataset cache (ignored)
└── results/                      # Evaluation outputs
```

> `data/`, `hf_cache/`, and run outputs are runtime artifacts.

## Configuration and indexes

Configuration and artifact paths handled by `src/rag/config.py` resolve relative to the checkout.

- **Retrieval:** loads the BM25, FAISS, retriever, and reranker YAMLs. Fusion weights are ordered `[BM25, FAISS]`. `rerank_k` limits the fused candidate pool; `top_n` limits the reranked output and must satisfy `1 <= top_n <= rerank_k`. The answer pipeline then uses at most 10 passages.
- **Index creation:** `HybridRetriever` prepares missing indexes automatically. Defaults are `data/bm25_index` and `data/faiss_vs/` (`index.faiss`, `index.pkl`, and build metadata).
- **Index reuse:** existing files are reused, not rebuilt on every invocation. If you change BM25 preprocessing/index parameters or the dense embedding model, normalization, or distance settings, use a fresh `main.persist_path` or explicitly remove and rebuild the old index. Compatibility with all possible settings is not guaranteed! For instance, the FAISS loader mainly checks its distance metric.
- **Pickling indexes:** BM25 and FAISS index loading use/rely on pickle-backed deserialization. For this project, security boundaries are ignored but, still something to keep in mind.

## Evaluation

Separate measurements:
- Retrieval metrics measure which evidence was found.
- Answer similarity and nugget coverage measure the generated answer against gold references.
- Citation support measures whether the answer's own cited evidence supports its claims.

### 1. Prepare shared query splits

```sh
uv run python -m src.rag.evaluation.prepare_queries \
  --config config/evaluation.yaml
```

The supplied evaluation YAMLs uses `dev_size: 0.3` to shuffle and partition the **upstream test split** into approximately 30% local `dev` and 70% local `test`. These are project-level subsets. `dev` is used for tuning, while `test` is reserved for the final evaluation.

The local splits are cached in `data/langchain_queries/`. Configuration settings are stored so, for a different eval coniguration, explicitly pass `--overwrite` to recreate the query splits (the rest is left untouched).

### 2. Tune retrieval on `dev`

Use both options with the supplied diagnostics YAML:

```sh
uv run python -m src.rag.evaluation.retrieval_diagnostics \
  --config config/retrieval_diagnostics.yaml \
  --experiments config/retrieval_diagnostics.yaml
```

- `--config` supplies split/cache provenance, cutoffs, and results location.
- `--experiments` supplies recursive overrides in `defaults` and named `configurations`, plus example sampling settings.
- Without `--experiments`, the runner evaluates one baseline from the project retrieval configs. Without `--config`, it loads `config/evaluation.yaml`, which **currently selects test, not dev**.
- `--output-dir` must name a directory that does not yet exist.

The evaluated stages are `bm25`, `faiss`, `fusion`, `reranker_candidates`, and `reranker`.
Setting `reranker_model: null` omits the final two. Each query/configuration is retrieved once, then the captured rankings are used for scoring and examples.
The supplied named configurations compare similarity search, MMR, and equal-weight fusion; with current defaults, `similarity` and `equal_weight_fusion` are identical.
> The current `.yaml` configs contain the final configuration values.

Each `results/retrieval_diagnostics_<split>_<timestamp>/` contains:

- `summary.csv`: aggregate metrics per configuration and stage.
- `<configuration>.json`: effective settings, query IDs, aggregate/per-query scores, returned-document counts, rankings through the largest cutoff, missed relevant IDs, and sampled query-document pairs.

Examples include question/document text, rank, relevance, and supported nugget IDs. `examples_per_kind` caps each category per stage/configuration; sampling is deterministic across queries. **Good** means relevant to at least one nugget; **bad** means explicitly non-relevant and never positively judged for that query; **unjudged** is separate, not a known negative. Text is limited by `text_chars`.

### 3. Evaluate held-out retrieval

After freezing settings based on dev diagnostics:

```sh
uv run python -m src.rag.evaluation.retrieval \
  --config config/evaluation.yaml
```

The runner requires `split: test` (default). Its procedure is:

1. Load index/search settings from the BM25 and FAISS project YAMLs, but fusion and reranking settings from the evaluation YAML. Set final reranker `top_n` to `max(cutoffs)`; the defaults retrieve 100 hits per component, rerank the first 100 fused hits, and return up to 50.
2. Build binary relevance judgments from nuggets' relevant/non-relevant corpus IDs. Positive judgments win conflicts; query-level relevance is the union across nuggets, while nugget-level judgments retain aspect information.
3. Retrieve for each joined `query_title` and `query_text`. Require document IDs, deduplicate by ID while retaining first rank, and keep the largest cutoff. Decreasing synthetic scores preserve order; they are not confidence scores.
5. Macro-average metrics over all queries, including empty result lists as zeros, then print and save the aggregate results with effective settings.

Output is `results/retrieval_test.json`, or the corresponding file under the configured `results_dir`. It includes query count, cutoffs, effective settings, and aggregate metrics, but no per-query rankings. Repeated runs **overwrite this file**; use a different results directory to retain multiple evaluations.

#### Retrieval metrics

Defaults: `k = [1, 3, 5, 10, 20, 50]`. The implementations in `src/rag/evaluation/metrics.py` are taken from [FreshStack's retrieval metrics](https://github.com/fresh-stack/freshstack/blob/main/freshstack/retrieval/metrics.py).

See also the [FreshStack paper](https://arxiv.org/abs/2504.13128).

| Metric | Interpretation |
|---|---|
| `Recall@k` | Fraction of query-relevant documents retrieved in the first k hits |
| `NDCG@k` | Ranking quality using binary query-level relevance (`pytrec_eval`) |
| `Coverage@k` | Fraction of gold nuggets supported by at least one retrieved relevant document |
| `alpha-nDCG@k` | Nugget-aware diversified ranking quality (`pyndeval`); emitted only for k ≤ 20 |
| `Hole@k` | First-k hits absent from the evaluated query set's annotation universe, divided by k |

Hole is **not** the per-query unjudged fraction: a document annotated for another evaluated query counts as annotated. It divides by the requested k even for short rankings, so an empty ranking's zero Hole is not evidence of good retrieval.

### 4. Evaluate answer quality

```sh
uv run python -m src.rag.evaluation.generation \
  --config config/generation_evaluation.yaml
```

The runner reuses project retrieval/generator settings, with its own reranker overrides: currently a 200-passage candidate pool, compared with 100 in the CLI. Both supply only the first 10 final passages to Qwen.

Two independent measurements are reported:

- **`jina_cosine_similarity`:** paired generated-answer/gold-`answer_text` cosine similarity with normalized embeddings. Scores are within [-1, 1]. Original code/citations remain in the text, empty answers score zero, and long inputs are truncated to `max_length`.
  - The configured similarity model is `jinaai/jina-embeddings-v5-text-small` on CUDA, with batch size 8 and `max_length: 8192`. The adapter in `answer_similarity.py` uses `trust_remote_code=True` and calls `.encode(..., task="text-matching")`.
  - This model was chosen since it is specifically tuned to provide similarity scores, among other downstream tasks.
- **`nugget_coverage`:** fraction of gold nuggets judged `fully_covered`, averaged equally over queries. `nugget_coverage_micro` pools nuggets across queries. `partially_covered`, `absent`, and `contradicted` rates are also reported; partial coverage earns no fractional credit, and contradiction takes precedence.

Each gold nugget is evaluated independently in a separate call to a judge (same Qwen used for generation, greedy decoding, thinking disabled). Every call sees only the question, generated answer, and the single nugget being assessed—not source passages or the full gold answer. Verdicts across all gold nuggets are then aggregated into answer coverage.Non-absent verdicts require an exact answer excerpt. The supplied settings allow one repair retry and set `invalid_as_absent: true`: exhausted validation failures become flagged `absent` judgments with zero coverage. Setting this to false makes those failures raise instead. Runtime/model-generation errors still propagate.

An LLM judge is used here to provide a flexible way to interpret the alignment between final answer and individual nuggets. Compared to the retrieval step, when semantics and meaning are fused, here I decided to focus on meaning.
Extractions of nuggets and code literals are included later, when evaluating the judgements.

Each `results/generation_<split>_<timestamp>/` contains:

- `settings.json`: effective settings and selected query IDs.
- `answers.jsonl`: checkpointed questions, generated answers, gold answers/nuggets, and the exact supplied source order/text; each answer is saved before scoring.
- `per_query.jsonl`: completed scores and nugget labels, excerpts, explanations, raw responses, and validation-failure details where applicable.
- `summary.json`: aggregate metrics and label counts, written only on success.

To rescore without retrieval or answer generation:

```sh
uv run python -m src.rag.evaluation.generation \
  --config config/generation_evaluation.yaml \
  --answers results/<previous_run>/answers.jsonl
```

Selected queries must all be present; question, gold answer, and nuggets are checked against the configured query split. 

### 5. Evaluate citation support from saved answers

> This mode does **NOT** rerun retrieval, answer generation, similarity, or nugget judging.

```sh
uv run python -m src.rag.evaluation.generation \
  --citations-only \
  --answers results/<previous_run>/answers.jsonl
```

- Here,  `--answers` is required and `--overwrite` is rejected.
- Each input record requires `query_id`, `question`, `answer`, and `sources`.
- Sources need consecutive 1-based `citation` numbers and the exact passage `text` supplied during generation.
- Missing evidence is an error, never reconstructed by retrieval. An explicitly empty source list is allowed, but references to it are invalid.

The Qwen judge performs two separate tasks with greedy decoding and thinking disabled:

**(1) Extract factual units and references.**
- Identify exact answer spans, attached citations, and abstentions.
- Retain uncited factual units and code; exclude nonfactual headings/filler.
- Recognize `[1]`, `[1][3]`, and `[1, 3]`, but not code indexing or Markdown links.
- Local validation checks literal, ordered, nonoverlapping spans and citation provenance.
- Claims come from those original spans with citation markers removed, not unchecked model paraphrases.

**(2) Judge cited support.** 
- For each unit with valid referencesudge its valid cited passages jointly.
- Labels are `fully_supported`, `partially_supported`, `contradicted` (takes precedence), and `insufficient_evidence`.
- Support/conflict verdicts require literal source excerpts; topical overlap alone is insufficient.

- Extraction validation failures are recorded with zero per-answer metrics.
- Invalid support judgments retain claims in denominators but earn no support or contradiction
credit.
- These failures are recorded explicitly rather than silently treated as successful judgments. Runtime/model-generation errors still stop the run.

| Metric | Definition |
|---|---|
| `citation_validity` | Valid reference occurrences / extracted reference occurrences |
| `cited_claim_support` | Fully supported units with no invalid references / cited units |
| `supported_citation_coverage` | Same credited supported units / all factual units |
| `contradiction_rate` | Units contradicted by their cited evidence / all factual units |

- Partial support gets no fractional credit. References count by occurrence for validity, but cited passages are deduplicated for judging. For instance, for `[1][99]` when only source 1 exists, validity is 1/2: the valid subset is judged, but the unit receives no support/coverage credit even if fully supported.
- A contradiction in that subset still counts. Uncited/all-invalid units have no semantic verdict and an explicit skip reason. Validity covers references in extracted factual units, not all answer text.
- Macro metrics average all answers equally; `*_micro` metrics pool reference/claim counts. Abstentions, empty answers, and validation failures are counted separately.
- Each `results/citations_<split>_<timestamp>/` contains `settings.json`, a copy of selected `answers.jsonl` records, checkpointed `per_query.jsonl` with extraction, verdict, excerpt, error, and metric details, and `summary.json` on success.
-  The split label comes from configuration; citation-only mode does not load that split. Runtime is roughly one extraction per nonempty answer plus one support call per validly cited unit, with additional calls if retries are enabled.

## Tests
> **Note**: These tests have been autonomously generated by GPT-6.1 Sol, despite instructions against it. Looking at some CoTs, they seemed related to GPT's latent verifications, rather than proper unit testing. Included for the sake of completeness.

The tests use `unittest`, synthetic data, and mocked model calls; they do not require model downloads or hosted services. They still require the project's Python dependencies, including importable FAISS/PyTorch components.

```sh
PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_*.py' -v
```

Coverage includes corpus/index setup, retrieval stages and splits, generation, answer similarity, nugget judging, citation parsing/judging, and citation metrics. Passing mocked tests does not establish compatibility with downloaded model code or successful execution on a particular GPU.

# Evaluation Results

## Retrieval Performance

> See `results/retrieval_diagnostics` for measures on `dev` set.
> See `results/retrieval_test` for measures on `test` set.

The retrieval diagnostic in `results/retrieval_diagnostics_dev_20261106T203022_442898Z` show the following
- Based on the `dev` set, hybrid setup works better than either sparse-only or dense-only retrievers with scores like `Recall@50` going from 30.88% to 36.75% without reranker and 32.52% with it.
- Similarity search performed better compared to MMR across the board.
- The re-ranker seems to hurt the final performance, both during `dev` tuning and held-out `test` evaluation. Performance dips when using a reranker range from ~2% points to ~8% points, which is significant.
  - This warrants further tuning as reranking is often used to boost performance, as in Table 4 of the FreshStack paper. I used a reranker I was familiar with but it is probably not a good fit for this data.
- While Recall is generally weak not exceedeing 0.4 (even before reranking), measured `Recall@50` is not excessively different from the reference measures from the original FreshStack paper (Table 2; inference setting), considering no query rewriting is done.
  - Another thing that emerged when tuning on the `dev` set is that BM25 may lead to higher recall compared to FAISS. I think this is due to the code-aware tokenisation leading to better "direct matches" with elements from the query.
- `Coverage@20` follows the same trend.
- `alpha-nDCG@10` is weaker compared to the reference metrics.
- It should be noted that in the original paper, many higher scores comes from the fusion of 4 retrievers and when those systems are tested when the golden answers are used as query (oracle setting). Still, the scores from the current configuration are not amazing per se.
  - As far as the `LangChain` split is concerned, the 4-way fusion of the original paper is the only configuration that exceeds the 0.4 boundary for Recall.

## Answer Quality

> See `results/generation_test` for raw measures on answer quality.
> Run `uv run python -m src.rag.evaluation.answer_stats --run-dir ./results/generation_test_20261107T193759_498949Z/` for a dedicated report.

- Cosine similarity between golden answer and Qwen-generated answers appears on the higher side. But, this is just a baseline and not evidence of correctness.
  - From personal experience, I have used other embedding models which report high scores on average but that are not really able to discern information and content, resulting in scores collapse in a narrow range (e.g., 0.85+).
  - In this case, the specificity of the topic and shared vocabulary may be the root causes.
- Strict nugget coverage (macro) lies around 23%, which is definitely on the low side. Partial coverage bumps this to ~30%.
- There is a high rate of queries with 0 nugget coverage (~58%). Note that this was done asking the judge to look for the exact substring. This, in turn, also penalises the nugget coverage scores above.
- Refusals (based on regex) are between 32.4% and 50% of the requests, depending on the phrasing from the model. These may be due to the weak candidate recall from the retrievers+reranker.

Some examples of queries with high similarity but zero coverage
```
76633711 -- cos=0.890
  nugget: The split_text function in RecursiveCharacterTextSplitter selects a separator from the provided list, defaulting to char
  answer: Based on the provided context, I do not know the specific reason why your `RecursiveCharacterTextSplitter` output is `['a\nbcefg', 'hij\nk']
77570745 -- cos=0.886
  nugget: Use `mapChatMessagesToStoredMessages` to serialize LangChain messages.
  answer: Based on the provided context, you can deserialize the serialized messages using the `load` or `loads` functions from `langchain_core.load`.
76451997 -- cos=0.885
  nugget: Set a `max_tokens_limit` when creating the `ConversationalRetrievalChain` object to manage token size.
  answer: Based on the provided context, there is no specific information or code example detailing how to reduce the size of tokens being passed to O
78213463 -- cos=0.894
  nugget: Implement a method to maintain conversation context without resending all past messages, such as "ConversationSummaryMem
  answer: Based on the provided context, the standard and documented method to continue a conversation with the ChatGPT API is to include all previous
78352032 -- cos=0.884
  nugget: The distance function is defined in the vector database (Solr) when the vector field is set up and cannot be changed dir
  answer: I do not know how to change the distance metric directly in `similarity_search` or `as_retriever` for a `Solr` vector store based on the pro
```

## Citation Support of Generated Answers

> See `results/citations_test` for raw measures on citation support.
> Run `uv run python -m src.rag.evaluation.citation_stats --run-dir ./results/citations_test_20261107T210239_061244Z/` for a dedicated report.

- About 61.3% claims from the answers were scored but to failed extraction or model abstension. Depending on whether we restrict the analysis, the following metrics are obtained, which potentially reflect issues in the pipeline rather than the quality of the answers.

| Metric | All | Scored-only |
|---|---|---|
| `citation_validity` | 0.5839 | 0.9416 |
| `cited_claim_support` | 0.1403 | 0.2289 |
| `supported_citation_coverage` | 0.0785 | 0.1281 |
| `contradiction_rate` | 0.0066 | 0.0107 |

- Across all responses, there are about 50% of the passages provided as context being *cited*. The rest has invalid judgements, substrings not matching, or citation number-mismatches, leading to lower scores.
- Among the sources cited
  - 24.5% support final answers
  - 8% partially support the final answers
  - 34.5% constitute insufficient evidence
  - 0.16% are flagged as contradictions
  - 37%, instead, are considered invalid (e.g., when the judgement could not be parsed)
- Reference validity is surprisingly (maybe suspiciously) high so either the model the model does not cite non-existent citations, or something in my current setup is not catching other errors.

Some examples of contradicting claims
```
query=75952371 claim_id=2 cites=[2]
  CLAIM : Python examples showing how to set the model name using the `model_name` attribute (e.g., `new OpenAI({ modelName: "gpt-3.5-turbo-instruct" })`)
  QUOTE : const model = new OpenAI({ modelName: "gpt-3.5-turbo-instruct" });
  REASON: The cited passage shows a JavaScript/TypeScript example using `new OpenAI({ modelName: "gpt-3.5-turbo-instruct" })`, which directly contradicts the claim that Python examples use the `model_name` attribute. The code uses `modelName` (camelCase), not `model_name` (snake_case), and the syntax is clearly for a JavaScript environment (import/export, async/await), not Python.

query=76244938 claim_id=9 cites=[2]
  CLAIM : The `StructuredChatAgent` is the recommended agent for structured tools, as noted in the deprecation warning for `StructuredChatAgent` which suggests using the `createStructuredChatAgent` method instead .
  QUOTE : @deprecated Use the {@link https://api.js.langchain.com/functions/langchain.agents.createStructuredChatAgent.html | createStructuredChatAgent method instead}.
  REASON: The cited passage explicitly marks the `StructuredChatAgent` class as deprecated and recommends using the `createStructuredChatAgent` method instead, which contradicts the claim that `StructuredChatAgent` is the recommended agent.
```

# Limitations and improvements

- **Single-turn only:** no conversation state, streaming interface, or service API.
- **Snapshot knowledge:** the October 2024 corpus may not match current LangChain APIs. The code generated in the answers may not be usable and/or up to date.
- **Retrieval bottlenecks:** relevant material can be missed during initial search, fusion truncation, reranking, or the final 10-passage context selection. The current setup was tuned on the queries `dev` set but a more extensive configuration sweep would be beneficial.
- **No aspectual retrieval:** Nuggets from the `dev` set were not used to tune the prototype via query expansion or rewriting to cover different aspects of the query. Nuggets were considered for evaluation purposes, when only the query is provided as input to the RAG pipeline.
- **Uncalibrated model judgments:** Qwen generates answers and judges both nugget coverage and citation support. Family-level biases and model self-preference can affect scores. That is, a Qwen-judge may prefer Qwen-generated answers. An independent judge or human evaluators can help audit the correcteness of the results.
- **Extraction is not a completeness proof:** valid literal claim spans do not establish that every factual claim was extracted or attributed correctly.
- **Reproducibility and operational readiness:** sampled answers, unpinned upstream revisions, large local models, and trusted pickle indexes need attention before reliable automated use.
- **Parsers:** the project currently uses a mix of regex and LLM-as-a-judge to verify citation format and judge their use within the final response. These sub-modules have not been fully fine-tuned to handle regex edge-cases, nor to squeeze the maximum out of the LLM judge. Although, being a 9B model, its judging capabilities may be on the lower end of the spectrum.
  - As an example, here is a citation judgement where the judge is at fault, and seemingly confabulates the use of Claude instead of GPT-3.
    ```
    query=78115486 claim_id=1 cites=[1]
      CLAIM : The provided context describes: *   A solution using **Azure OpenAI Service** for embeddings and **GPT-3** for answering questions .
      QUOTE : This solution uses Azure OpenAI Service for creating embeddings vectors from documents. For answering the question of a user, it retrieves the most relevant document and then uses GPT-3 to extract the matching answer for the question.
      REASON: The cited passage explicitly describes a solution using Azure OpenAI Service for embeddings and GPT-3 for answering questions. This directly contradicts the claim which states the context describes using Azure OpenAI Service for embeddings and GPT-3, but the user's question and context clearly involve Anthropic Claude, not GPT-3. The claim incorrectly identifies the LLM used in the user's scenario.
    ```

# AI usage
GPT-6.1 Sol (Medium effort) was used within Zed Agent, helping with the following:
- Verification of internal LangChain pipeline
- Brainstorming and coding of evaluation scripts
- Integration of FreshStack metrics
- Unit test generation
- README
