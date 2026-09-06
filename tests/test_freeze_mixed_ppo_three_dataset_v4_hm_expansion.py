from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from scripts.prepare import freeze_mixed_ppo_three_dataset_v4_hm_expansion as hm
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    HOTPOT_TARGET_CELLS,
    MUSIQUE_TARGET_HOPS,
    _identity,
)
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import sha256_file
from scripts.prepare.materialize_mixed3_v4_expansion_retrieval import (
    _frozen_counts,
    _read_jsonl,
    _resolve_bound_file,
    _validate_requests,
)


def _identity_rows(
    dataset: str,
    strata: dict[str, int],
    *,
    prefix: str,
) -> list[dict]:
    rows: list[dict] = []
    serial = 0
    for stratum, count in strata.items():
        for _ in range(count):
            raw = {
                "id": f"{prefix}-{serial}",
                "question": f"Which marker belongs to {prefix} item {serial}?",
            }
            rows.append(
                _identity(
                    raw,
                    dataset=dataset,
                    route=f"{dataset}_outcome",
                    question_type=stratum.split("/", 1)[0],
                    stratum=stratum,
                    source_role="retained_parent",
                )
            )
            serial += 1
    return rows


def _inputs(tmp_path: Path, *, parent_schema: str = hm.EXPECTED_PARENT_SCHEMA):
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "schema_version": parent_schema,
                "status": hm.EXPECTED_PARENT_STATUS,
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    hotpot_raw = tmp_path / "hotpot.jsonl"
    musique_raw = tmp_path / "musique.jsonl"
    replay = tmp_path / "replay.jsonl"
    protected = tmp_path / "protected.jsonl"
    for path in (hotpot_raw, musique_raw, replay, protected):
        path.write_text("", encoding="utf-8")
    return parent, hotpot_raw, musique_raw, replay, (protected,)


def test_freezes_hm_requests_without_proofkg_or_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _inputs(tmp_path)
    parent_population = [
        *({"dataset": "hotpotqa"} for _ in range(600)),
        *({"dataset": "musique"} for _ in range(599)),
    ]
    hotpot = _identity_rows(
        "hotpotqa", HOTPOT_TARGET_CELLS, prefix="hotpot-selected"
    )
    musique = _identity_rows(
        "musique", MUSIQUE_TARGET_HOPS, prefix="musique-selected"
    )
    hotpot_new = hotpot[:417]
    musique_new = musique[:401]
    hotpot_reserve = _identity_rows(
        "hotpotqa",
        {stratum: 1 for stratum in HOTPOT_TARGET_CELLS},
        prefix="hotpot-reserve",
    )
    musique_reserve = _identity_rows(
        "musique",
        {stratum: 1 for stratum in MUSIQUE_TARGET_HOPS},
        prefix="musique-reserve",
    )

    monkeypatch.setattr(hm, "_load_protocol_output", lambda protocol, name: parent_population)
    monkeypatch.setattr(
        hm,
        "build_hotpot_population",
        lambda *args, **kwargs: (
            hotpot,
            hotpot_new,
            hotpot_reserve,
            {"retained_parent": 583, "new_retrieval": 417},
        ),
    )
    monkeypatch.setattr(
        hm,
        "build_musique_population",
        lambda *args, **kwargs: (
            musique,
            musique_new,
            musique_reserve,
            {"retained_parent": 599, "new_retrieval": 401},
        ),
    )

    def fake_manifest(directory, extra=None, *, status="COMPLETE"):
        target = Path(directory) / "manifest.json"
        target.write_text(
            json.dumps({"status": status, "run": dict(extra or {})}),
            encoding="utf-8",
        )
        return target

    monkeypatch.setattr(hm, "dump_manifest", fake_manifest)
    output_dir = tmp_path / "hm-freeze"
    report = hm.freeze_hm_expansion(
        parent_protocol_path=paths[0],
        hotpot_raw_path=paths[1],
        musique_raw_path=paths[2],
        replay_path=paths[3],
        protected_paths=paths[4],
        output_dir=output_dir,
        reserve_per_stratum=1,
        experiment_id="TEST-HM-V4-PREREG",
    )

    assert report["status"] == hm.STATUS
    assert report["population"]["resolved_unique_by_dataset"] == {
        "hotpotqa": 1000,
        "musique": 1000,
    }
    assert report["population"]["retrieval_requests_by_dataset"] == {
        "hotpotqa": 417,
        "musique": 401,
    }
    assert report["intended_final_v4"]["two_wiki"]["status"] == (
        "UNRESOLVED_NOT_BOUND"
    )
    assert report["intended_final_v4"]["finalization_ready"] is False
    assert report["intended_final_v4"]["training_started"] is False
    assert all(report["gates"].values())
    assert "proof800" not in report["outputs"]

    protocol_path = output_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    request_identity = protocol["outputs"]["retrieval_requests"]
    request_path = _resolve_bound_file(request_identity, label="retrieval_requests")
    requests = _read_jsonl(request_path)
    assert len(requests) == 818
    assert Counter(row["dataset"] for row in requests) == Counter(
        {"hotpotqa": 417, "musique": 401}
    )
    assert all(row["gold_access"] is False for row in requests)
    assert _frozen_counts(protocol) == {"hotpotqa": 417, "musique": 401}
    validated = _validate_requests(requests, _frozen_counts(protocol))
    assert len(validated["hotpotqa"]) == 417
    assert len(validated["musique"]) == 401
    for name, identity in protocol["outputs"].items():
        assert sha256_file(Path(identity["path"])) == identity["sha256"], name


def test_refuses_overwrite_before_reading_inputs(tmp_path: Path):
    output_dir = tmp_path / "already-frozen"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        hm.freeze_hm_expansion(
            parent_protocol_path=tmp_path / "missing-parent",
            hotpot_raw_path=tmp_path / "missing-hotpot",
            musique_raw_path=tmp_path / "missing-musique",
            replay_path=tmp_path / "missing-replay",
            protected_paths=(),
            output_dir=output_dir,
        )


def test_rejects_unexpected_parent_protocol_before_selection(tmp_path: Path):
    paths = _inputs(tmp_path, parent_schema="wrong-parent-schema")
    with pytest.raises(ValueError, match="unexpected parent schema"):
        hm.freeze_hm_expansion(
            parent_protocol_path=paths[0],
            hotpot_raw_path=paths[1],
            musique_raw_path=paths[2],
            replay_path=paths[3],
            protected_paths=paths[4],
            output_dir=tmp_path / "out",
        )
