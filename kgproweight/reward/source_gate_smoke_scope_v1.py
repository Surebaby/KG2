"""Fail-closed execution scope for one authorized, complete A-smoke600.

This module validates authority/configuration and performs no reward arithmetic,
gate fitting, model loading, label reading, or automatic training expansion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from kgproweight.utils.paths import project_root, model_path

SCHEMA = "source-credit-v2-bounded-a-smoke600-scope-v1"
SCOPE_FIELD = "execution_scope"
PARENT_GATE_SHA256 = "8f657059c9ec7b2750db44fedae0679da4aaf1d4d65d24d72b761e507b229971"
UTILITY_REPORT_SHA256 = "a469aa370567ea3020e9af6c4cee5bafd10f2e2472281e54f295d9222c4c7516"
PARENT_MASK_SHA256 = "9c065a8a768073ed3f5e293178ace4e1f47d2327645b955b8091fa2cd792e5d3"
RECOVERY_PROTOCOL_SHA256 = "286e89606fa574e5b2f856c052602f08be65583d4b5d0c4ef6bcb39bbbd2533a"
MODEL_AUTHORITY_SHA256 = "e877116411d20a062c4e8a7929c08154c51df0544cdc0c312ac583adb9bab26b"
PARENT_CONFIG_SCIENTIFIC_SHA256 = "bbad29f56bfd4267a9d2fbc47b512f5bfe8e5f86127d035914bf11fac0a1a288"
MANUAL_AUTHORIZATION_SHA256 = "8b5fe05e368a59817482241da45dc5f25e521101f568dab5b87bf988da259fd7"
SMOKE_SCHEDULE_SHA256 = "9ca30133f6547edc49701df1bc9827bfae0298918c6ba229c9669b4ebe123f1c"
ALLOWED_GATE_FIELDS = {"payload_sha256", "experiment_id", "training_clearance", "independent_confirmation_clearance",
                       "ppo_launch_clearance", "training_clearance_scope", "source_credit_mask", SCOPE_FIELD}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def portable_path(path, *, root=None):
    root = Path(root or project_root()).absolute()
    path = Path(path)
    if path.is_absolute():
        try: path = path.relative_to(root)
        except ValueError as exc: raise ValueError("probe artifact outside project root") from exc
    require(bool(path.parts) and ".." not in path.parts and str(path) not in ("", "."), "invalid portable probe path")
    return path.as_posix()


def bound_path(ref, *, root=None):
    root = Path(root or project_root())
    require(isinstance(ref, Mapping) and isinstance(ref.get("path"), str) and not Path(ref["path"]).is_absolute(),
            "probe bindings must be project-relative")
    path = root / portable_path(ref["path"], root=root)
    require(path.is_file() and file_sha(path) == ref.get("sha256")
            and ("bytes" not in ref or path.stat().st_size == ref["bytes"]), "probe bound asset changed")
    return path


def config_identity(cfg, *, root=None):
    """Bind every field of the actual CLI-forwarded training dataclass."""
    from kgproweight.training.phase3_ppo import Phase3PPOConfig
    require(is_dataclass(cfg) and isinstance(cfg, Phase3PPOConfig), "complete real Phase3PPOConfig required for scoped gate")
    root = Path(root or project_root()).absolute()
    def normalize(value):
        if isinstance(value, Path): value = str(value)
        if isinstance(value, str) and Path(value).is_absolute(): return portable_path(value, root=root)
        if isinstance(value, dict): return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [normalize(v) for v in value]
        return value
    return normalize(asdict(cfg))


def read_scope(data, artifact_path, *, root=None):
    require(isinstance(data.get(SCOPE_FIELD), Mapping), "scoped child lacks an explicit bound scope")
    scope_path = bound_path(data[SCOPE_FIELD], root=root)
    scope = json.loads(scope_path.read_text())
    require(scope.get("schema_version") == SCHEMA and scope.get("scope") == "complete_A_smoke600_only",
            "unsupported or unbounded gate execution scope")
    require(scope.get("full_ppo_clearance") is False and scope.get("matched600_clearance") is False,
            "manual A-smoke600 cannot authorize matched controls/full")
    require(portable_path(artifact_path, root=root) == scope["child_gate_path"], "scoped gate loaded from an unregistered path")
    require(scope.get("manual_A_smoke600_clearance") is True, "explicit manual A-smoke600 clearance required")
    return scope


def validate_smoke_scope(data, cfg, artifact_path, mask, *, root=None):
    root = Path(root or project_root())
    require(root.resolve() == Path(__file__).resolve().parents[2], "probe project root must be the executing code checkout")
    require(Path.cwd().resolve() == root.resolve(), "probe must execute from its bound project root")
    require(cfg is not None and artifact_path is not None, "scoped probe child requires exact runtime configuration")
    scope = read_scope(data, artifact_path, root=root)
    actual = config_identity(cfg, root=root)
    required = {"total_steps": 600, "batch_size": 4, "rollouts_per_prompt": 4,
                "source_gate_mode": "learned", "source_gate_credit_version": "v2",
                "source_gate_format_version": "v2", "source_gated_reward_version": "v1",
                "answer_format_reward_version": "v2", "runtime_contract_version": "v2",
                "mixed_outcome_reward": True, "mixed_text_reward": True, "proofkg_process_reward": True,
                "proofkg_process_version": "v2_3", "log_with": "tensorboard", "save_every_steps":200,
                "health_guard_after_steps":200,"health_guard_window":15,"health_guard_min_valid_rate":.7,
                "health_guard_max_length_capped_frac":.2,"health_guard_max_mean_kl":10.0}
    require(all(actual.get(k) == v and type(actual.get(k)) is type(v) for k, v in required.items()),
            "scoped child permits only complete A/600 trajectories/K4/batch4")
    require(digest({k:v for k,v in actual.items() if k not in ("output_dir", "source_gate_calibration_path")}) == PARENT_CONFIG_SCIENTIFIC_SHA256,
            "bounded scope cannot change the original scientific runtime configuration")
    require(actual == scope["runtime_config"] and digest(actual) == scope["runtime_config_sha256"],
            "actual complete runtime config differs from frozen probe")
    require(actual["source_gate_calibration_path"] == scope["child_gate_path"], "runtime selected another gate")
    refs = scope["bindings"]
    for field in ("silver_path", "question_kg_records_path", "rollout_sampling_weights_path", "sft_replay_silver_path"):
        require(field in refs and actual[field] == refs[field]["path"], "runtime data/replay identity is not bound")
    paths = {key: bound_path(ref, root=root) for key, ref in refs.items()}
    require(refs["parent_gate"]["sha256"] == PARENT_GATE_SHA256 and refs["utility_report"]["sha256"] == UTILITY_REPORT_SHA256
            and refs["parent_mask"]["sha256"] == PARENT_MASK_SHA256
            and refs["recovery_protocol"]["sha256"] == RECOVERY_PROTOCOL_SHA256
            and refs["model_authority"]["sha256"] == MODEL_AUTHORITY_SHA256, "probe authority is not the frozen accepted confirmation")
    require(refs["manual_authorization"]["sha256"] == MANUAL_AUTHORIZATION_SHA256
            and refs["schedule"]["sha256"] == SMOKE_SCHEDULE_SHA256, "manual authority or fixed600 schedule differs")
    authorization = json.loads(paths["manual_authorization"].read_text())
    require(authorization.get("schema_version") == "ppo-a-smoke600-human-authorization-v1"
            and authorization.get("authorized_complete_A_smoke600") is True
            and authorization.get("trajectory_limit") == 600
            and authorization.get("start_from_original_strong_sft") is True
            and authorization.get("use_probe_checkpoint_as_initialization") is False
            and authorization.get("automatic_matched600_clearance") is False
            and authorization.get("full12000_clearance") is False
            and authorization.get("automatic_restart_or_expansion") is False
            and authorization.get("fresh_confirmation_health_status") == "FAIL", "manual budget or original SFT authority changed")
    require(authorization["guard"] == {"after_trajectories":200,"window_ppo_batches":15,"min_valid_rate":.7,
            "max_length_capped_frac":.2,"max_mean_kl":10.0,"nonfinite_immediate":True}, "original smoke cost guard differs")
    for key, ref in authorization["evidence"].items():
        require(refs[key] == ref, "probe evidence or parent smoke config not bound to manual authority")
    lineage = json.loads(paths["probe_independent_lineage"].read_text())
    require(json.loads(paths["probe_training_manifest"].read_text())["status"] == "COMPLETE"
            and lineage["status"] == "PASS_ENGINEERING_LINEAGE_ONLY"
            and json.loads(paths["probe_independent_parameters_events"].read_text())["status"] == "PASS_ENGINEERING_PROBE12_NOT_PERFORMANCE_EVIDENCE",
            "complete probe and independent engineering evidence required")
    require(refs["probe_scope"] == lineage["inputs"]["scope"], "probe scope differs from accepted lineage")
    probe_scope = json.loads(paths["probe_scope"].read_text())
    for key in ("silver_path", "question_kg_records_path", "sft_replay_silver_path", "rollout_sampling_weights_path"):
        require(refs[key] == probe_scope["bindings"][key], "training data changed after accepted probe")
    report = json.loads(paths["utility_report"].read_text())
    manifest = json.loads(paths["utility_manifest"].read_text())
    require(manifest["outputs"]["report.json"]["sha256"] == UTILITY_REPORT_SHA256
            and report.get("independent_utility_status") == "PASS" and report.get("engineering_probe_eligibility") is True
            and report.get("health_status") == "FAIL" and report.get("overall_status") == "FAIL"
            and report["decision"].get("matched600_investment_clearance") is False
            and report["decision"].get("full_ppo_auto_launch") is False
            and report["integrity"].get("status") == "PASS", "utility/health/engineering scope changed")
    require(scope["confirmation_status"] == {"independent_utility": "PASS", "health": "FAIL", "overall": "FAIL"},
            "health failure must remain explicit")
    require(report["recovery"]["protocol"]["sha256"] == refs["recovery_protocol"]["sha256"], "confirmation recovery lineage differs")
    parent = json.loads(paths["parent_gate"].read_text())
    require(all(parent.get(k) is False for k in ("training_clearance", "independent_confirmation_clearance", "ppo_launch_clearance")),
            "parent gate flags must remain false")
    require({k:v for k,v in data.items() if k not in ALLOWED_GATE_FIELDS} ==
            {k:v for k,v in parent.items() if k not in ALLOWED_GATE_FIELDS}, "scoped child changed alpha/normalization/scientific parent fields")
    require(data.get("training_clearance") is True and data.get("independent_confirmation_clearance") is True
            and data.get("ppo_launch_clearance") is False
            and data.get("training_clearance_scope") == "complete_A_smoke600_only", "scoped child flags contradict limited permission")
    require(data["source_credit_mask"]["path"] == refs["parent_mask"]["path"]
            and {k:v for k,v in data["source_credit_mask"].items() if k != "path"} ==
                {k:v for k,v in parent["source_credit_mask"].items() if k != "path"}
            and mask.manifest_sha256 == PARENT_MASK_SHA256
            and mask.payload_sha256 == parent["source_credit_mask"]["payload_sha256"], "original training800/671 mask must be retained")
    counts = Counter((e["original_m_graph"], e["status"]) for e in mask._entries.values())
    require(counts == {(1,"PASS"):671,(1,"UNVERIFIED"):100,(1,"FAIL"):29,(0,"UNVERIFIED"):30}, "training mask population changed")
    require(actual["fixed_rollout_schedule_path"] == refs["schedule"]["path"], "runtime schedule differs")
    rows = [json.loads(line) for line in paths["schedule"].read_text().splitlines() if line.strip()]
    require(len(rows) == 600 and [x["rollout_index"] for x in rows] == list(range(1,601)), "fixed schedule must have exactly600 trajectories")
    for start in range(0,600,4):
        require(len({(x["dataset"], x["qid"]) for x in rows[start:start+4]}) == 1, "schedule violates K4 groups")
    require(Counter(x["dataset"] for x in rows) == {"hotpotqa":200,"2wikimultihopqa":200,"musique":200}, "fixed three-domain probe schedule differs")
    required_code = {"kgproweight/reward/source_gate_smoke_scope_v1.py", "kgproweight/reward/source_credit_gate_v2.py",
                     "kgproweight/training/phase3_ppo.py", "scripts/train/phase3_ppo.py", "scripts/train/_split_args.py",
                     "scripts/prepare/resolve_phase3_ppo_runtime_config.py", "scripts/prepare/freeze_source_credit_v2_smoke600_scope_v1.py",
                     "scripts/sourcegate_python.sh", "scripts/train/supervise_scoped_smoke600_v1.py",
                     refs["config"]["path"], refs["parent_config"]["path"]}
    required_code |= {p.relative_to(root).as_posix() for p in (root / "kgproweight").rglob("*.py")}
    require(required_code <= set(scope["code_bindings"]), "probe enforcement code closure incomplete")
    for name, ref in scope["code_bindings"].items():
        require(ref["path"] == name, "code logical path mismatch")
        bound_path(ref, root=root)
    authority_path = paths["model_authority"]
    authority = json.loads(authority_path.read_text())
    require(scope["models"] == authority["models"], "model identity declarations differ from frozen input authority")
    for role in ("base_model", "rearag_model", "policy_tokenizer"):
        info = scope["models"][role]
        allowed = set(info["files"])
        if role == "policy_tokenizer": allowed |= {"adapter_model.safetensors", "adapter_config.json", "training_args.bin"}
        if role == "rearag_model": allowed.add("configuration.json")  # ModelScope metadata, never an HF loader input.
        model_dir = root / info["path"]
        runtime_files = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*")
                         if p.is_file() and not any(part.startswith(".") for part in p.relative_to(model_dir).parts)
                         and p.suffix in {".json", ".bin", ".safetensors", ".model", ".py", ".txt", ".pt", ".pth"}}
        require(runtime_files <= allowed, "unbound model/tokenizer loading file can override frozen weights")
        for name, frozen in info["files"].items():
            ref = {"path": portable_path(Path(info["path"]) / name, root=root), "sha256": frozen["sha256"], "bytes": frozen["bytes"]}
            bound_path(ref, root=root)
    require(actual["sft_checkpoint"] == portable_path(scope["models"]["policy_tokenizer"]["path"], root=root), "SFT checkpoint differs from bound policy tokenizer")
    for name, filename in (("policy", "adapter_model.safetensors"), ("policy_config", "adapter_config.json")):
        ref = refs["sft:" + filename]
        require(ref["path"] == str(Path(actual["sft_checkpoint"]) / filename)
                and ref["sha256"] == authority["source_bindings"][name]["sha256"], "SFT/ref adapter identity differs")
    require(scope["limits"] == {"trajectories":600,"prompt_groups":150,"rollouts_per_prompt":4,"ppo_batches":150,
                                "automatic_resume":False,"automatic_restart_or_expansion":False}, "bounded execution limits changed")
    return {"scope": "complete_A_smoke600_only", "runtime_config_sha256": digest(actual), "trajectory_limit": 600,
            "utility_report_sha256": UTILITY_REPORT_SHA256, "health_status": "FAIL", "matched600_clearance": False, "manual_A_smoke600_clearance": True}


def validate_smoke_execution_paths(gate, cfg):
    """Verify the actual model loaders' paths before any CUDA allocation."""
    if "execution_scope" not in gate.artifact:
        return
    root = project_root()
    scoped = read_scope(gate.artifact, cfg.source_gate_calibration_path, root=root)
    adapter = json.loads((root / cfg.sft_checkpoint / "adapter_config.json").read_text())
    actual_base = Path(adapter["base_model_name_or_path"])
    if not actual_base.is_absolute(): actual_base = root / actual_base
    expected_base = root / scoped["models"]["base_model"]["path"]
    expected_rearag = root / scoped["models"]["rearag_model"]["path"]
    require(actual_base.is_dir() and actual_base.resolve() == expected_base.resolve(),
            "runtime adapter base model path is missing or differs from bound base")
    require(Path(model_path("rearag")).resolve() == expected_rearag.resolve(), "runtime ReaRAG environment points to an unbound model")
