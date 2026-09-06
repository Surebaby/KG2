"""Explicit dispatch for independently frozen probe12 and manual A-smoke600.

Legacy heuristic scope descriptions do not confer or select a bounded scope.
Any referenced v2 training-clearance claim is checked before reward dispatch,
including a child whose scope markers were stripped and payload re-signed.
"""
from pathlib import Path
import json

from kgproweight.reward import source_gate_probe_scope_v1 as probe
from kgproweight.reward import source_gate_smoke_scope_v1 as smoke


def module_for_scope(data, *, root=None):
    path = probe.bound_path(data.get("execution_scope"), root=root)
    scoped = json.loads(path.read_text())
    pair = (scoped.get("schema_version"), scoped.get("scope"))
    choices = {(probe.SCHEMA, "complete_A_probe12_only"): probe,
               (smoke.SCHEMA, "complete_A_smoke600_only"): smoke}
    if pair not in choices:
        raise ValueError("unsupported bounded execution scope")
    return choices[pair]


def validate_bounded_scope(data, cfg, artifact_path, mask):
    if cfg is None or artifact_path is None:
        raise ValueError("scoped child requires exact runtime configuration")
    module = module_for_scope(data)
    function = module.validate_probe_scope if module is probe else module.validate_smoke_scope
    return function(data, cfg, artifact_path, mask)


def validate_bounded_execution_paths(gate, cfg):
    if "execution_scope" not in gate.artifact:
        return
    module = module_for_scope(gate.artifact)
    function = module.validate_probe_execution_paths if module is probe else module.validate_smoke_execution_paths
    function(gate, cfg)


def load_referenced_bounded_before_dispatch(cfg):
    path = Path(cfg.source_gate_calibration_path or "")
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    from kgproweight.reward.source_credit_gate_v2 import ARTIFACT_SCHEMA, SourceCreditGateV2
    relevant = ("execution_scope" in data
                or data.get("training_clearance_scope") in {"complete_A_probe12_only", "complete_A_smoke600_only"}
                or (data.get("schema_version") == ARTIFACT_SCHEMA and data.get("training_clearance") is True))
    if not relevant:
        return None
    gate = SourceCreditGateV2.load(path, runtime_config=cfg)
    validate_bounded_execution_paths(gate, cfg)
    return gate
