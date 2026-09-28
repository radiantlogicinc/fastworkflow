"""Explicit trace enrichment registry and trajectory-manifest integration."""
from __future__ import annotations

import json

from fastworkflow import tracing
from fastworkflow.observability import enrichment
from fastworkflow.observation_offloading.manifest import (
    install_span_policy,
    uninstall_span_policy,
)


def test_registration_is_ordered_and_multiple_enrichers_coexist():
    calls = []
    tokens = []
    try:
        tokens.append(enrichment._register(
            "test-z", lambda raw: (calls.append("z") or
                                    enrichment.AttributeEnrichment({"z": raw["plain"]}))))
        tokens.append(enrichment._register(
            "test-a", lambda raw: (calls.append("a") or
                                    enrichment.AttributeEnrichment({"a": "added"}))))
        capped = tracing._capped({"plain": "value"})
        assert calls == ["a", "z"]
        assert capped == {"plain": "value", "a": "added", "z": "value"}
    finally:
        for token in tokens:
            enrichment._remove(token)


def test_registration_never_overwrites_and_removal_is_token_scoped():
    first = enrichment._register(
        "test-stable-key", lambda raw: enrichment.AttributeEnrichment({"first": True}))
    try:
        try:
            enrichment._register(
                "test-stable-key",
                lambda raw: enrichment.AttributeEnrichment({"second": True}))
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate registration replaced the first enricher")
        assert enrichment._remove(first) is True
        second = enrichment._register(
            "test-stable-key", lambda raw: enrichment.AttributeEnrichment({"second": True}))
        try:
            assert enrichment._remove(first) is False
            assert tracing._capped({"plain": "value"})["second"] is True
        finally:
            enrichment._remove(second)
    finally:
        enrichment._remove(first)


def test_failure_is_safe_and_does_not_suppress_other_enrichers():
    def broken(raw):
        raise RuntimeError("fixture failure")

    broken_token = enrichment._register("test-broken", broken)
    good_token = enrichment._register(
        "test-good", lambda raw: enrichment.AttributeEnrichment({"survived": True}))
    try:
        assert tracing._capped({"plain": "value"}) == {
            "plain": "value", "survived": True}
    finally:
        enrichment._remove(broken_token)
        enrichment._remove(good_token)


def test_additions_do_not_overwrite_original_or_each_other():
    first = enrichment._register(
        "test-collision-a",
        lambda raw: enrichment.AttributeEnrichment({"plain": "changed", "shared": "a"}))
    second = enrichment._register(
        "test-collision-b",
        lambda raw: enrichment.AttributeEnrichment(
            {"shared": "b"}, {"messages": "shared"}))
    try:
        old_limit = tracing.MAX_ATTR_BYTES
        tracing.MAX_ATTR_BYTES = 4
        capped = tracing._capped({"plain": "original", "messages": "long message"})
        assert capped["plain"]["truncated"] is True
        assert capped["shared"] == "a"
        # The second enricher did not own the colliding addition, so it cannot
        # blank the source and point at the first enricher's value.
        assert capped["messages"]["value"] == "long"
        assert "replaced_by" not in capped["messages"]
    finally:
        tracing.MAX_ATTR_BYTES = old_limit
        enrichment._remove(first)
        enrichment._remove(second)


def test_replacement_copies_a_caller_owned_envelope():
    original = {"truncated": True, "value": "caller", "original_length": 6}
    token = enrichment._register(
        "test-envelope-copy", lambda raw: enrichment.AttributeEnrichment(
            {"summary": {"complete": True}}, {"envelope": "summary"}))
    try:
        capped = tracing._capped({"envelope": original})
        assert capped["envelope"]["value"] == ""
        assert capped["envelope"]["replaced_by"] == "summary"
        assert original == {
            "truncated": True, "value": "caller", "original_length": 6}
    finally:
        enrichment._remove(token)


def test_manifest_activation_is_explicit_idempotent_and_preserves_exact_capping():
    uninstall_span_policy()
    observation = "evidence row\n" * 20
    messages = json.dumps([{
        "role": "user",
        "content": f"[[ ## observation_0 ## ]]\n{observation}\n[[ ## answer ## ]]\ndone",
    }])
    old_limit = tracing.MAX_ATTR_BYTES
    tracing.MAX_ATTR_BYTES = 32
    try:
        before = tracing._capped({"messages": messages, "plain": None})
        assert "trajectory_manifest" not in before
        install_span_policy()
        install_span_policy()
        after = tracing._capped({
            "messages": messages, "plain": None,
            "trajectory_manifest": {"contract": "caller-value"},
        })
        stored = after["messages"]
        assert stored["truncated"] is True
        assert stored["original_length"] == len(messages.encode("utf-8"))
        assert stored["value"] == ""
        assert stored["replaced_by"] == "trajectory_manifest"
        assert after["plain"] is None
        manifest = after["trajectory_manifest"]
        assert manifest["contract"] == enrichment.TRAJECTORY_MANIFEST_CONTRACT
        assert manifest["observation_count"] == 1
        # Structured manifest additions intentionally retain their complete rows.
        assert isinstance(manifest, dict)
        assert manifest["observations"][0]["utf8_bytes"] > tracing.MAX_ATTR_BYTES
    finally:
        tracing.MAX_ATTR_BYTES = old_limit
        uninstall_span_policy()


def test_nonmessage_and_unparseable_attributes_are_unchanged():
    install_span_policy()
    try:
        assert tracing._capped(None) == {}
        assert tracing._capped({"messages": None, "count": 3}) == {
            "messages": None, "count": 3}
        assert tracing._capped({"messages": "not a trajectory"}) == {
            "messages": "not a trajectory"}
    finally:
        uninstall_span_policy()
