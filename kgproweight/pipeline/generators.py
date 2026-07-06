"""Generator factories used by the inference pipelines.

The default path on Pro 6000 (96 GB) is plain bf16 — no 4-bit quantisation,
no LoRA at inference. A QLoRA path is preserved for 24 GB cards (4090,
3090) where the user wants to evaluate the trained LoRA without merging.
"""

from __future__ import annotations

from typing import Any, Optional

from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import get_logger
from kgproweight.utils.paths import model_path

logger = get_logger(__name__)


def build_generator(
    config: Any,
    lora_path: Optional[str] = None,
    dtype: str = "bf16",
):
    """Return a FlashRAG-style generator wrapping a HF causal LM.

    Parameters
    ----------
    config:
        ``flashrag.config.Config`` object.
    lora_path:
        Optional PEFT adapter to merge or attach for inference.
    dtype:
        ``"bf16"`` (default), ``"fp16"``, or ``"fp32"``.
    """
    setup_flashrag()
    from flashrag.utils import get_generator

    framework = config["framework"] if "framework" in config else "hf"
    if framework == "vllm":
        logger.warning("vLLM generator selected; LoRA inference falls back to the HF generator.")
        return get_generator(config)

    # HF path: optionally attach LoRA before returning.
    base_gen = get_generator(config)
    if lora_path is None:
        return base_gen

    try:
        import torch  # noqa: F401
        from peft import PeftModel
    except ImportError:
        logger.warning("peft not installed; ignoring lora_path=%s", lora_path)
        return base_gen

    inner_model = getattr(base_gen, "model", None)
    if inner_model is None:
        logger.warning("Generator does not expose .model; cannot attach LoRA.")
        return base_gen

    logger.info("Attaching LoRA adapter from %s", lora_path)
    inner_model = PeftModel.from_pretrained(inner_model, lora_path)
    inner_model.eval()
    base_gen.model = inner_model
    return base_gen


def build_qlora_inference_generator(
    model_id_or_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    device: str = "cuda",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
):
    """4-bit QLoRA inference helper for 24 GB cards.

    Returns an object with a ``generate(prompts)`` method matching the
    minimal interface used by the FlashRAG pipelines.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_id = model_id_or_path or model_path("llama3-8B-instruct")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb_config, device_map=device
    )
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    class _Gen:
        def __init__(self) -> None:
            self.tokenizer = tokenizer
            self.model = model
            self.device = device

        def generate(self, prompts):
            outs = []
            for p in prompts:
                enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=4096).to(device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                outs.append(tokenizer.decode(gen[0][enc.input_ids.size(1) :], skip_special_tokens=True))
            return outs

    return _Gen()
