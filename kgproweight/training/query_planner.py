"""Answer-free learned query-planner SFT utilities.

The model sees only dataset identity and question text.  Supervision targets
contain relation/dependency structure but never gold answers or evidence tails.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from kgproweight.utils.paths import model_path
from kgproweight.utils.seed import set_seed


PLANNER_SYSTEM_PROMPT = """You are a multi-hop query planner. Return exactly one JSON object and nothing else.
Do not answer the question. Do not invent intermediate entities or values.
For 2wikimultihopqa, emit anchors and relation steps with Wikidata PIDs and dependency slots.
For musique, emit ordered answer-free subquery templates and dependencies."""


@dataclass(frozen=True)
class PlannerTrainConfig:
    experiment_id: str
    train_path: str
    dev_path: str
    output_dir: str
    base_model: str = "llama3-8B-instruct"
    train_per_dataset: int = 4000
    dev_per_dataset: int = 300
    seed: int = 42
    max_seq_length: int = 768
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 1.0e-4
    warmup_ratio: float = 0.03
    epochs: float = 1.0
    max_steps: int = -1
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    load_in_4bit: bool = True
    dtype: str = "bf16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def _read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def planner_messages(record: Mapping[str, Any], *, include_target: bool) -> list[Dict[str, str]]:
    dataset = str(record["dataset"])
    target_type = str(record["target_type"])
    # Zero-shot transfer onto a target type the dataset was not supervised with:
    # add a target-type-conditional hint, so the canonical dataset target types
    # (2Wiki=relation_graph, MuSiQue=subquery_graph) keep their learned prompt
    # text unchanged.  relation_graph transfer explicitly requests hop_N slots.
    if target_type == "relation_graph" and dataset in {"hotpotqa", "musique"}:
        hotpot_hint = (
            "Use anchors and relation steps with Wikidata PIDs, hop_N output "
            "slots and hop_N dependencies.\n"
        )
    elif target_type == "subquery_graph" and dataset == "hotpotqa":
        hotpot_hint = "Use an answer-free subquery graph with entity >> relation operators.\n"
    else:
        hotpot_hint = ""
    user = (
        f"Dataset: {dataset}\n"
        f"Question: {str(record['question']).strip()}\n"
        f"Required target type: {target_type}\n"
        + hotpot_hint
        + "Produce the answer-free query plan JSON."
    )
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if include_target:
        target = json.dumps(record["target"], ensure_ascii=False, separators=(",", ":"))
        messages.append({"role": "assistant", "content": target})
    return messages


def balanced_sample(
    path: str | Path,
    *,
    per_dataset: int,
    seed: int,
) -> list[Dict[str, Any]]:
    if per_dataset <= 0:
        raise ValueError("per_dataset must be positive")
    grouped: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        grouped[str(row["dataset"])].append(row)
    if set(grouped) != {"2wikimultihopqa", "musique"}:
        raise ValueError(f"unexpected planner datasets: {sorted(grouped)}")
    selected: list[Dict[str, Any]] = []
    for offset, dataset in enumerate(sorted(grouped)):
        candidates = sorted(grouped[dataset], key=lambda row: str(row["question_key"]))
        if len(candidates) < per_dataset:
            raise ValueError(f"{dataset} has {len(candidates)} records, need {per_dataset}")
        selected.extend(random.Random(seed + offset).sample(candidates, per_dataset))
    random.Random(seed).shuffle(selected)
    return selected


def sample_identity(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = [str(row["question_key"]) for row in records]
    return {
        "n": len(keys),
        "by_dataset": dict(Counter(str(row["dataset"]) for row in records)),
        "question_key_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }


def encode_record(record: Mapping[str, Any], tokenizer, *, max_seq_length: int) -> Dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        planner_messages(record, include_target=False),
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        planner_messages(record, include_target=True),
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"chat-template prompt is not a prefix for {record['question_key']}")
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"record exceeds max_seq_length={max_seq_length}: "
            f"{record['question_key']} has {len(full_ids)} tokens"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError(f"no supervised tokens for {record['question_key']}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
        "supervised_length": len(full_ids) - len(prompt_ids),
    }


def encode_records(records: Sequence[Mapping[str, Any]], tokenizer, *, max_seq_length: int):
    import datasets

    rows = [encode_record(record, tokenizer, max_seq_length=max_seq_length) for record in records]
    return datasets.Dataset.from_list(rows)


def length_summary(dataset) -> Dict[str, Any]:
    lengths = sorted(int(value) for value in dataset["length"])
    supervised = sorted(int(value) for value in dataset["supervised_length"])

    def percentile(values: Sequence[int], fraction: float) -> int:
        return values[min(len(values) - 1, int((len(values) - 1) * fraction))]

    return {
        "n": len(lengths),
        "total_tokens": sum(lengths),
        "length": {
            "min": lengths[0], "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95), "max": lengths[-1],
        },
        "supervised_length": {
            "min": supervised[0], "p50": percentile(supervised, 0.50),
            "p95": percentile(supervised, 0.95), "max": supervised[-1],
        },
    }


def load_smoke_config(path: str | Path, *, output_override: str | None = None, max_steps: int | None = None) -> PlannerTrainConfig:
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(document["data"]["split_root"])
    sampling, model, training = document["data"]["sampling"], document["model"], document["training"]
    return PlannerTrainConfig(
        experiment_id=str(document["experiment"]["id"]),
        train_path=str(root / document["data"]["train_file"]),
        dev_path=str(root / document["data"]["dev_file"]),
        output_dir=str(output_override or training["output_dir"]),
        base_model=str(model["base_model"]),
        train_per_dataset=int(sampling["train_per_dataset"]),
        dev_per_dataset=int(sampling["dev_per_dataset"]),
        seed=int(training["seed"]),
        max_seq_length=int(training["max_seq_length"]),
        batch_size=int(training["per_device_train_batch_size"]),
        grad_accum=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training["warmup_ratio"]),
        epochs=float(training["num_train_epochs"]),
        max_steps=int(max_steps if max_steps is not None else -1),
        eval_steps=int(training["eval_steps"]),
        save_steps=int(training["save_steps"]),
        load_in_4bit=bool(model["load_in_4bit"]),
        dtype=str(model["dtype"]),
        lora_r=int(model["lora_r"]),
        lora_alpha=int(model["lora_alpha"]),
        lora_dropout=float(model["lora_dropout"]),
        target_modules=tuple(model["target_modules"]),
    )


def prepare_data(cfg: PlannerTrainConfig, tokenizer) -> tuple[Any, Any, Dict[str, Any]]:
    train_records = balanced_sample(
        cfg.train_path, per_dataset=cfg.train_per_dataset, seed=cfg.seed
    )
    dev_records = balanced_sample(
        cfg.dev_path, per_dataset=cfg.dev_per_dataset, seed=cfg.seed
    )
    train_dataset = encode_records(train_records, tokenizer, max_seq_length=cfg.max_seq_length)
    dev_dataset = encode_records(dev_records, tokenizer, max_seq_length=cfg.max_seq_length)
    report = {
        "train_sample": sample_identity(train_records),
        "dev_sample": sample_identity(dev_records),
        "train_lengths": length_summary(train_dataset),
        "dev_lengths": length_summary(dev_dataset),
    }
    return train_dataset, dev_dataset, report


def run_query_planner_sft(cfg: PlannerTrainConfig, *, probe: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("query planner SFT requires a CUDA-visible process")
    if cfg.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("active GPU does not support bf16")
    if "confirmation" in Path(cfg.train_path).name or "confirmation" in Path(cfg.dev_path).name:
        raise ValueError("confirmation data must not enter planner training/dev")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    set_seed(cfg.seed)
    base_id = model_path(cfg.base_model)
    out_dir, experiment_id = prepare_new_run_dir(
        cfg.output_dir,
        experiment_id=(f"{cfg.experiment_id}-PROBE{cfg.max_steps}" if probe else cfg.experiment_id),
        extra={
            "phase": "query_planner_sft_probe" if probe else "query_planner_sft_smoke",
            "config": cfg.__dict__,
            "input_artifacts": {
                "train": artifact_identity(cfg.train_path),
                "dev": artifact_identity(cfg.dev_path),
                "base_model": artifact_identity(base_id),
            },
        },
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset, dev_dataset, data_report = prepare_data(cfg, tokenizer)
    (out_dir / "data_report.json").write_text(
        json.dumps(data_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map={"": 0},
    )
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.enable_input_require_grads()

    evaluation_enabled = not probe
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.dtype == "bf16",
        fp16=cfg.dtype == "fp16",
        logging_steps=1 if probe else cfg.logging_steps,
        logging_first_step=True,
        eval_strategy="no" if probe else "steps",
        eval_steps=cfg.eval_steps,
        save_strategy="no" if probe else "steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        seed=cfg.seed,
        data_seed=cfg.seed,
        group_by_length=True,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding="longest", label_pad_token_id=-100
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset if evaluation_enabled else None,
        data_collator=collator,
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start
    loss_path = out_dir / "training_history.jsonl"
    with loss_path.open("w", encoding="utf-8") as fh:
        for row in trainer.state.log_history:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    metrics = {
        "elapsed_seconds": elapsed,
        "global_steps": trainer.state.global_step,
        "seconds_per_optimizer_step": elapsed / max(1, trainer.state.global_step),
        "train_loss": result.training_loss,
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }
    (out_dir / "throughput.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        out_dir,
        status="COMPLETE",
        extra={
            "phase": "query_planner_sft_probe" if probe else "query_planner_sft_smoke",
            "experiment_id": experiment_id,
            "config": cfg.__dict__,
            "data_report": data_report,
            "throughput": metrics,
            "output_artifacts": {
                "final": artifact_identity(final_dir),
                "history": artifact_identity(loss_path),
            },
        },
    )
    return {"output_dir": str(out_dir), "final": str(final_dir), **metrics}
