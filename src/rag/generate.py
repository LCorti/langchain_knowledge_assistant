# prompt, LLM, citation parsing, abstention

from collections.abc import Sequence
from typing import Any

import torch
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

_ROLE = {"human": "user", "ai": "assistant", "system": "system"}


def _to_chat_dicts(messages: Sequence[BaseMessage | dict]) -> list[dict]:
    """Convert LangChain messages (or dicts) to chat-template dicts."""
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
        else:
            out.append({"role": _ROLE.get(m.type, m.type), "content": m.content})
    return out


def strip_thinking(text: str) -> str:
    """Drop the chain of thought."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


class QwenGenerator:
    """Loads Qwen3.5-9B once and generates answers from chat messages."""

    def __init__(self, generator_config: dict[str, Any]):
        self.model_id = generator_config["model_id"]
        self.enable_thinking = generator_config["enable_thinking"]
        self.max_new_tokens = generator_config["max_new_tokens"]
        self.gen_kwargs = {
            "do_sample": generator_config["temperature"] > 0,
            "temperature": generator_config["temperature"],
            "top_p": generator_config["top_p"],
            "top_k": generator_config["top_k"],
        }
        self.device_map = "auto"
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=self.device_map
        ).eval()

    def generate(
        self,
        messages: Sequence[BaseMessage | dict],
        *,
        enable_thinking: bool | None = None,
        max_new_tokens: int | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Generate with call-local overrides, leaving answer settings unchanged.

        A supplied generation_kwargs replaces (rather than merges with) gen_kwargs.
        None uses the corresponding instance setting.
        """
        gen_kwargs = dict(
            self.gen_kwargs if generation_kwargs is None else generation_kwargs
        )
        self._load()
        inputs = self._tokenizer.apply_chat_template(
            _to_chat_dicts(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=(
                self.enable_thinking if enable_thinking is None else enable_thinking
            ),
            return_tensors="pt",
            return_dict=True,
        ).to(next(self._model.parameters()).device)

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=(
                    self.max_new_tokens if max_new_tokens is None else max_new_tokens
                ),
                **gen_kwargs,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return strip_thinking(text)

    def as_runnable(self) -> Runnable[Any, str]:
        """LangChain Runnable: PromptValue | list of messages -> answer string."""

        def _run(x: PromptValue | Sequence[BaseMessage | dict]) -> str:
            msgs = x.to_messages() if isinstance(x, PromptValue) else x
            return self.generate(msgs)

        return RunnableLambda(_run)
