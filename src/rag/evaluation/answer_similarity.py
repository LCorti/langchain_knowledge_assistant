"""Paired answer similarity using Jina's symmetric text-matching adapter."""

import math

import torch


class JinaAnswerSimilarity:
    def __init__(
        self,
        model_id: str = "jinaai/jina-embeddings-v5-text-small",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 8192,
        revision: str | None = None,
    ):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        # if type(max_length) is not int or not 1 <= max_length <= 8192:
        #     raise ValueError("max_length must be between 1 and 8192")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.revision = revision
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModel

            # Jina v3 implements its adapters and encode API in repository code.
            self._model = (
                AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    revision=self.revision,
                )
                .to(self.device)
                .eval()
            )
        return self._model

    def score(self, answers: list[str], references: list[str]) -> list[float]:
        """Return paired cosine similarities, not BERTScore or probabilities.

        Original text, including code and citation markers, is preserved. Empty
        answers receive zero; references must be nonempty. The encoder truncates
        inputs at max_length tokens.
        """
        if len(answers) != len(references):
            raise ValueError("answers and references must have equal lengths")
        if any(not isinstance(text, str) for text in answers + references):
            raise ValueError("answers and references must contain strings")
        if any(not text.strip() for text in references):
            raise ValueError("references must be nonempty")
        scores = [0.0] * len(answers)
        indices = [i for i, answer in enumerate(answers) if answer.strip()]
        if not indices:
            return scores

        import numpy as np

        texts = [answers[i] for i in indices] + [references[i] for i in indices]
        model = self._load()
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            with torch.inference_mode():
                out = model.encode(
                    texts=batch,
                    task="text-matching",
                    max_length=self.max_length,
                )

            if isinstance(out, torch.Tensor):
                out = out.float().cpu().numpy()
            chunks.append(np.asarray(out, dtype=np.float64))
        vectors = np.concatenate(chunks, axis=0)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Jina returned an unexpected embedding shape")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (
            not np.isfinite(vectors).all()
            or not np.isfinite(norms).all()
            or (norms == 0).any()
        ):
            raise ValueError("Jina returned nonfinite or zero-length embeddings")
        vectors = vectors / norms
        count = len(indices)
        for j, i in enumerate(indices):
            score = float(np.dot(vectors[j], vectors[count + j]))
            if not math.isfinite(score):
                raise ValueError("Nonfinite answer similarity")
            scores[i] = max(-1.0, min(1.0, score))
        return scores
