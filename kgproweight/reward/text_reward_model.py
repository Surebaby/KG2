"""Text reward model — two interchangeable back-ends.

Bug-fix #2 in :doc:`docs/refactor_notes`. Previously the legacy
``TextRewardHead`` was an *untrained* ``nn.Linear`` that produced random
rewards. Here we expose two concrete back-ends and a thin dispatcher
``TextRewardModel`` that the rest of the codebase imports.

1. ``RearagPromptScorer`` — uses the externally-trained ReaRAG-9B model as
   a step-level prompt scorer. We feed the concatenated trajectory prefix
   plus the candidate step into ReaRAG and read the *answer-token logprob*
   sum, which we normalise into ``[-1, 1]``.

2. ``LlamaTextRewardHead`` — a Llama-3-8B (or any HF causal LM) with a
   linear scalar head on top, fine-tuned on silver data's text-quality
   labels. Used as a fallback when ReaRAG-9B is not available.

The dispatcher only needs the ``score_step(prompt, step_text) -> float``
contract.
"""

from __future__ import annotations

from typing import Any, List, Optional

from kgproweight.utils.logging import get_logger
from kgproweight.utils.paths import model_path

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backend 1: ReaRAG-9B prompt scorer
# ---------------------------------------------------------------------------

class RearagPromptScorer:
    """Score a single ``(prompt, step_text)`` pair by summing the step's logprobs.

    The scorer expects an already-loaded HuggingFace ``AutoModelForCausalLM``
    + tokenizer. Constructing instances is the responsibility of
    :func:`build_text_reward_model`.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", max_length: int = 4096) -> None:
        import torch  # local import; this class is GPU-only

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @classmethod
    def from_pretrained(cls, model_id_or_path: str, device: str = "cuda", dtype: str = "bf16") -> "RearagPromptScorer":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return cls(model=model, tokenizer=tokenizer, device=device)

    def score_step(self, prompt: str, step_text: str) -> float:
        torch = self.torch
        if not step_text.strip():
            return 0.0

        # Tokenise prompt and step; we score the logprobs assigned to the step tokens.
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        step_ids = self.tokenizer(step_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        # Truncate from the left of the prompt if too long.
        total_len = prompt_ids.size(0) + step_ids.size(0)
        if total_len > self.max_length:
            overflow = total_len - self.max_length
            prompt_ids = prompt_ids[overflow:]

        input_ids = torch.cat([prompt_ids, step_ids]).unsqueeze(0).to(self.device)
        labels = input_ids.clone()
        # Mask the prompt portion so loss is computed only over the step tokens.
        labels[:, : prompt_ids.size(0)] = -100

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=labels)
            # outputs.loss is mean NLL over step tokens.
            nll = outputs.loss.item()

        # Convert NLL → reward ∈ [-1, 1]. A typical strongly-likely span has NLL ~1.5;
        # extremely unlikely text trends toward 5+. We map via a smooth saturating fn:
        # reward = tanh( (2.5 - nll) / 1.5 )
        import math

        return math.tanh((2.5 - nll) / 1.5)


# ---------------------------------------------------------------------------
# Backend 2: Llama-3-8B + linear reward head (fallback, trainable)
# ---------------------------------------------------------------------------

class LlamaTextRewardHead:
    """Llama-3-8B causal LM + linear scalar head trained on silver labels.

    Training data: 0/1 text-quality labels derived from silver acceptance
    (accepted trajectories → +1, rejected → -1). Trained briefly in
    :mod:`kgproweight.training.phase2_prm` so that the PPO loop always has
    a calibrated text reward signal.

    At inference, the score is ``tanh(linear(last_hidden_state))``.
    """

    def __init__(self, base_model, tokenizer, head, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.head = head
        self.device = device
        self.base_model.eval()
        self.head.eval()

    @classmethod
    def from_pretrained(
        cls,
        head_path: str,
        base_model_id: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bf16",
    ) -> "LlamaTextRewardHead":
        import torch
        from transformers import AutoModel, AutoTokenizer
        import torch.nn as nn

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        base_id = base_model_id or model_path("llama3-8B-instruct")
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModel.from_pretrained(base_id, torch_dtype=torch_dtype, device_map=device)
        hidden = base.config.hidden_size
        head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh()).to(device=device, dtype=torch_dtype)
        if head_path:
            sd = torch.load(head_path, map_location=device, weights_only=True)
            head.load_state_dict(sd)
        for p in base.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = False
        return cls(base_model=base, tokenizer=tokenizer, head=head, device=device)

    def score_step(self, prompt: str, step_text: str) -> float:
        torch = self.torch
        text = (prompt + "\n\n" + step_text).strip()
        if not text:
            return 0.0
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        with torch.no_grad():
            out = self.base_model(**enc, output_hidden_states=False)
            last_hidden = out.last_hidden_state[:, -1, :]
            score = self.head(last_hidden).squeeze(-1).float().item()
        return float(score)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TextRewardModel:
    """Uniform front-end exposed to the rest of the codebase.

    Construction usually goes through :func:`build_text_reward_model` which
    consults the runtime config to pick the appropriate backend.
    """

    def __init__(self, backend: Any, name: str) -> None:
        self.backend = backend
        self.name = name

    @property
    def is_dummy(self) -> bool:
        return isinstance(self.backend, _DummyTextReward)

    def score_step(self, prompt: str, step_text: str) -> float:
        return float(self.backend.score_step(prompt, step_text))

    def score_steps(self, prompts: List[str], step_texts: List[str]) -> List[float]:
        return [self.score_step(p, s) for p, s in zip(prompts, step_texts)]


class _DummyTextReward:
    """Zero-everywhere reward; used when the user really wants to disable R_Text."""

    def score_step(self, prompt: str, step_text: str) -> float:  # noqa: ARG002
        return 0.0


def build_text_reward_model(
    backend: str = "auto",
    rearag_path: Optional[str] = None,
    fallback_head_path: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bf16",
) -> TextRewardModel:
    """Construct a :class:`TextRewardModel` according to the ``backend`` choice.

    ``backend``:
      * ``"rearag"``  — must succeed, raises otherwise.
      * ``"llama_head"`` — must succeed.
      * ``"auto"``  — try rearag, then llama_head, then dummy with a warning.
      * ``"dummy"`` — always returns 0.0 (only useful for diagnostics).
    """
    chosen = backend.lower()
    if chosen == "dummy":
        return TextRewardModel(_DummyTextReward(), name="dummy")

    if chosen in ("rearag", "auto"):
        path = rearag_path or model_path("rearag")
        try:
            backend_obj = RearagPromptScorer.from_pretrained(path, device=device, dtype=dtype)
            logger.info("Text reward backend: ReaRAG-9B prompt scorer at %s", path)
            return TextRewardModel(backend_obj, name="rearag")
        except Exception as exc:
            logger.warning("Failed to load ReaRAG-9B (%s); will try Llama head fallback.", exc)
            if chosen == "rearag":
                raise

    if chosen in ("llama_head", "auto"):
        if fallback_head_path is None:
            logger.warning(
                "No fallback_head_path; falling back to dummy text reward. "
                "Train a head via `kgproweight.training.phase2_prm` or set "
                "`text_reward_fallback_path` in the config."
            )
            if chosen == "llama_head":
                raise FileNotFoundError("text_reward_fallback_path is required for llama_head backend.")
            return TextRewardModel(_DummyTextReward(), name="dummy")
        try:
            backend_obj = LlamaTextRewardHead.from_pretrained(
                fallback_head_path, device=device, dtype=dtype
            )
            logger.info("Text reward backend: Llama-3-8B + head at %s", fallback_head_path)
            return TextRewardModel(backend_obj, name="llama_head")
        except Exception as exc:
            logger.warning("Failed to load Llama head (%s); using dummy text reward.", exc)
            if chosen == "llama_head":
                raise
            return TextRewardModel(_DummyTextReward(), name="dummy")

    raise ValueError(f"Unknown text reward backend: {backend!r}")
