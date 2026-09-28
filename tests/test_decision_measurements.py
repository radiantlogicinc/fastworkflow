"""The decision-model wording is the wording that was measured.

``tests/fixtures/decision_measurements.json`` records the offline measurements
behind the finish check's and the search router's published precision and
recall: the calibration they were made under, each metric with its n and 95%
interval, and the sha256 of every evidence file (pre-registration, labelling
rules, scripts, labels, keys, results). The evidence itself is not in the repo;
the manifest names where it is kept.

It also records ``wording.sha256``: the digest of every question template both
features send, rendered with fixed arguments. This test renders the shipped
templates again and compares. A changed question fails here even when the
golden request bodies (``test_decision_provider``) have been regenerated to
match, because a new wording is an unmeasured one: change the digest only
together with a new measurement recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from fastworkflow.observation_offloading import finish_check, search_router
from fastworkflow.observation_offloading.decision import OneOf, YesNo

FIXTURE = Path(__file__).parent / "fixtures" / "decision_measurements.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _question(question: Any) -> dict[str, Any]:
    if isinstance(question, YesNo):
        return {"instructions": question.instructions, "true": question.true, "false": question.false}
    assert isinstance(question, OneOf), type(question)
    # Choice order is part of what is asked.
    return {"instructions": question.instructions, "choices": [[k, v] for k, v in question.choices.items()]}


def rendered_wording(arguments: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every measured question template, rendered with *arguments*."""
    a = arguments
    router = search_router._questions({})
    return {
        "finish_check.action": _question(finish_check._action_question(a["k"])),
        "finish_check.applies": _question(finish_check._applies_question(a["k"], a["name"], a["kind"])),
        "finish_check.part_applies": _question(
            finish_check._part_applies_question(a["k"], a["j"], a["part"], a["name"], a["kind"])),
        "finish_check.exec": _question(finish_check._exec_question(a["step"], a["part"], a["name"])),
        "finish_check.exec_without_part": _question(finish_check._exec_question(a["step"], None, a["name"])),
        "finish_check.any": _question(finish_check._any_question(a["k"], a["step"])),
        "search_router.wants": _question(router["wants"]),
        "search_router.for_report": _question(router["for_report"]),
    }


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def wording_digests(arguments: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """The whole wording's digest, and one per template so a failure names the template."""
    rendered = rendered_wording(arguments)
    return _digest(rendered), {name: _digest(q) for name, q in rendered.items()}


def _manifest() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_shipped_question_wording_is_the_measured_wording():
    wording = _manifest()["wording"]
    whole, per_template = wording_digests(wording["arguments"])
    changed = sorted(name for name, digest in per_template.items()
                     if wording["templates"].get(name) != digest)
    assert not changed, (
        f"question wording changed for {changed}: the published precision and recall were measured "
        "with the old wording. Record a new measurement in tests/fixtures/decision_measurements.json "
        "before updating its digests.")
    assert sorted(wording["templates"]) == sorted(per_template)
    assert whole == wording["sha256"]


def test_the_manifest_matches_the_calibration_and_thresholds_that_ship():
    manifest = _manifest()
    check, router = manifest["finish_check"], manifest["search_router"]
    assert check["calibration"] == finish_check.CALIBRATION
    assert check["model"] == finish_check.DEFAULT_MODEL
    assert check["thresholds"] == {"FLAG_MIN": finish_check.FLAG_MIN, "ASK_MIN": finish_check.ASK_MIN}
    assert router["model"] == search_router.DEFAULT_ROUTER_MODEL
    assert router["thresholds"] == {"ROUTE_ALL_ROWS_MIN": search_router.ROUTE_ALL_ROWS_MIN}


def test_every_metric_states_its_n_and_interval():
    manifest = _manifest()
    metrics = [m for feature in ("finish_check", "search_router") for m in manifest[feature]["metrics"]]
    assert metrics
    for metric in metrics:
        assert isinstance(metric["n"], int) and metric["n"] > 0, metric
        low, high = metric["ci95"]
        assert 0.0 <= low <= metric["value"] <= high <= 1.0, metric
        assert metric["interval_method"], metric


def test_every_evidence_file_has_a_digest_and_a_location():
    manifest = _manifest()
    for feature in ("finish_check", "search_router"):
        groups = manifest[feature]["evidence"]
        assert groups, feature
        for group in groups:
            assert group["archive_location"], feature
            assert group["files"], feature
            for entry in group["files"]:
                assert entry["role"] and entry["path"], entry
                assert SHA256_RE.match(entry["sha256"]), entry
        roles = {entry["role"] for group in groups for entry in group["files"]}
        assert {"script", "labels", "key", "results"} <= roles, (feature, roles)
    assert any(entry["role"] == "pre_registration"
               for group in manifest["finish_check"]["evidence"] for entry in group["files"])
