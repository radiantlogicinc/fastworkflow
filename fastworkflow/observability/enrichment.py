"""Internal span-attribute enrichment before records reach a trace sink.

Registration is process-wide and explicit. Callers activate built-in enrichers
at their existing runtime boundary; constructing another agent is idempotent,
and closing one agent does not remove enrichment needed by another.

This is an internal observability seam, not a plugin API. Enrichers inspect raw
attributes but return structured additions. Original attributes are capped
first; additions remain uncapped to preserve the trajectory-manifest contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import re
from threading import RLock
from typing import Any, Callable, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttributeEnrichment:
    """Additions and named replacements produced from uncapped attributes."""

    additions: Mapping[str, Any] = field(default_factory=dict)
    replace_truncated: Mapping[str, str] = field(default_factory=dict)
    replace_existing: frozenset[str] = frozenset()


AttributeEnricher = Callable[[Mapping[str, Any]], AttributeEnrichment | None]


@dataclass(frozen=True)
class _RegistrationToken:
    key: str
    identity: int


_lock = RLock()
_enrichers: dict[str, tuple[_RegistrationToken, AttributeEnricher]] = {}
_next_identity = 0
TRAJECTORY_MANIFEST_CONTRACT = "fastworkflow-trajectory-manifest/1"
_DSPY_OBS_RE = re.compile(
    r"\[\[ ## (observation_(\d+)) ## \]\]\s*(.*?)(?=\n\[\[ ## |\Z)", re.S
)


def _register(key: str, enricher: AttributeEnricher) -> _RegistrationToken:
    """Register one stable key without replacing an existing enricher."""
    if not key:
        raise ValueError("enricher key must not be empty")
    global _next_identity
    with _lock:
        if key in _enrichers:
            raise ValueError(f"span attribute enricher already registered: {key}")
        _next_identity += 1
        token = _RegistrationToken(key, _next_identity)
        _enrichers[key] = (token, enricher)
        return token


def _register_once(key: str, enricher: AttributeEnricher) -> _RegistrationToken:
    """Idempotently register the same built-in callback under one stable key."""
    with _lock:
        current = _enrichers.get(key)
        if current is not None:
            if current[1] is not enricher:
                raise ValueError(f"span attribute enricher already registered: {key}")
            return current[0]
        return _register(key, enricher)


def _remove(token: _RegistrationToken) -> bool:
    """Remove only the registration represented by *token*."""
    with _lock:
        current = _enrichers.get(token.key)
        if current is None or current[0] != token:
            return False
        del _enrichers[token.key]
        return True


def _apply(raw: Mapping[str, Any], capped: MutableMapping[str, Any]) -> None:
    """Apply a deterministic registry snapshot without failing tracing."""
    with _lock:
        snapshot = [(key, registered[1])
                    for key, registered in sorted(_enrichers.items())]
    claimed = set(capped)
    for key, enricher in snapshot:
        try:
            result = enricher(raw)
            if result is None:
                continue
            accepted = set()
            for name, value in result.additions.items():
                if name in claimed and name not in result.replace_existing:
                    logger.warning(
                        "span attribute enricher %s skipped existing attribute %s",
                        key, name)
                    continue
                capped[name] = value
                claimed.add(name)
                accepted.add(name)
            for name, replacement in result.replace_truncated.items():
                if replacement not in accepted:
                    continue
                stored = capped.get(name)
                if isinstance(stored, dict) and stored.get("truncated"):
                    # cap_attr_value passes non-string objects through. Copy an
                    # envelope before mutation so caller-owned mappings remain raw.
                    stored = dict(stored)
                    capped[name] = stored
                    stored["value"] = ""
                    stored["replaced_by"] = replacement
        except Exception as error:  # noqa: BLE001 - tracing must never fail a turn
            logger.warning("span attribute enricher %s failed: %r", key, error)


def observation_row(key: str, text: str) -> dict[str, Any]:
    """One observation slot of the prompt, described without carrying it.

    Two digests, because the slot and the evidence are not the same bytes.
    ``sha256`` is the PROMPT SLOT exactly as the model received
    it, header and all, and is what a reader has to hash to prove what was
    sent. ``response_sha256`` is the command response inside that slot --
    ``canonical_response`` takes the offloading package's handle line and its
    escape back off -- and is what ``fw.agent.step`` recorded, because that span
    closes before the completion hook annotates. Comparing the first with the
    second is comparing non-equivalent bytes, and it reported unchanged resident
    evidence as mismatched.

    ``alias`` is read from either line the offloading package prints, not from
    the offload label alone: since the handle line became unconditional the
    inline case IS the normal case, and it was the one reporting no alias.
    ``alias_source`` says which line named it, so a resident observation and a
    pointer to one are still told apart by the row rather than by inference.

    One normalisation, used by every field that reads the text's shape: the
    alias, the kind and the response were previously read off differently
    normalised copies, so a slot with leading whitespace could report an alias
    and call itself text in the same row.
    """
    # Lazy by design: tracing imports this stdlib-only registry during core
    # startup. Importing the observation_offloading package here at module load
    # would execute its agent compatibility exports and cycle back to tracing.
    from fastworkflow.observation_offloading.labels import (  # noqa: PLC0415
        canonical_response, is_offload_label, observation_alias,
    )

    probe = text.lstrip()
    alias, source = observation_alias(probe)
    response = canonical_response(probe)
    encoded = text.encode("utf-8")
    return {
        "key": key, "alias": alias, "alias_source": source,
        "kind": "label" if is_offload_label(probe) else "text",
        "chars": len(text), "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "response_sha256": (None if response is None else
                            hashlib.sha256(response.encode("utf-8")).hexdigest()),
    }


def _message_blobs(messages_json: str) -> tuple[list[str], int]:
    try:
        parsed = json.loads(messages_json)
    except (TypeError, ValueError):
        return [str(messages_json or "")], 0
    if not isinstance(parsed, list):
        return [messages_json], 0
    blobs: list[str] = []
    system_prompt_chars = 0
    for message in parsed:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            text = "\n".join(parts)
        if message.get("role") == "system" and system_prompt_chars == 0:
            system_prompt_chars = len(text)
        if text:
            blobs.append(text)
    return blobs, system_prompt_chars


def manifest_from_messages_json(messages_json: str) -> Optional[dict[str, Any]]:
    if not isinstance(messages_json, str) or not messages_json:
        return None
    blobs, system_prompt_chars = _message_blobs(messages_json)
    by_index: dict[int, dict[str, Any]] = {}
    for blob in blobs:
        for match in _DSPY_OBS_RE.finditer(blob):
            by_index[int(match.group(2))] = observation_row(match.group(1), match.group(3))
    if not by_index:
        return None
    observations = [by_index[index] for index in sorted(by_index)]
    return {
        "contract": TRAJECTORY_MANIFEST_CONTRACT,
        "system_prompt_chars": system_prompt_chars,
        "messages_original_length": len(messages_json.encode("utf-8")),
        "observation_count": len(observations), "observations": observations,
    }


def _comparable_digests(row: Mapping[str, Any]) -> set[str]:
    """Digests of this row that a step's own evidence can honestly be equal to.

    The canonical response first -- that is the raw tool return the step span
    recorded -- and the prompt-slot digest too, because a slot this enricher
    never annotated (an older recording, a non-execute tool) has only that one
    and the two are then the same bytes anyway.
    """
    return {
        value
        for value in (row.get("response_sha256"), row.get("sha256"))
        if isinstance(value, str) and value
    }


def classify_against_steps(
    manifest: Mapping[str, Any],
    step_sha256_by_index: Mapping[int, str],
) -> dict[str, list[int]]:
    """Each step's recorded observation digest against the manifest's rows.

    ``step_sha256_by_index`` is the digest of the RAW tool return from
    ``fw.agent.step`` -- recorded before the completion hook prints the handle
    line -- so residency is decided against the response inside the slot, not
    against the annotated slot.

    ``mismatched`` therefore means the evidence genuinely differs, and it still
    can: a rehydrated listing carries its own response plus the stored rows
    behind its result handle, which is an intentional transformation of the
    slot and not the step's bytes. A rehydrated offload label, whose archived
    response comes back whole, is resident -- the transformation is the header,
    and the header is no longer counted against it.
    """
    by_index: dict[int, dict[str, Any]] = {}
    for row in manifest.get("observations") or []:
        key = str(row.get("key") or "")
        if not key.startswith("observation_"):
            continue
        try:
            index = int(key.removeprefix("observation_"))
        except ValueError:
            continue
        by_index[index] = row
    resident: list[int] = []
    labelled: list[int] = []
    absent: list[int] = []
    mismatched: list[int] = []
    for index, digest in sorted(step_sha256_by_index.items()):
        row = by_index.get(index)
        if row is None:
            absent.append(index)
        elif row.get("kind") == "label":
            labelled.append(index)
        elif digest in _comparable_digests(row):
            resident.append(index)
        else:
            mismatched.append(index)
    return {
        "resident": resident,
        "labelled": labelled,
        "absent": absent,
        "mismatched": mismatched,
    }


def _trajectory_manifest_enrichment(attributes: Mapping[str, Any]
                                    ) -> Optional[AttributeEnrichment]:
    messages = attributes.get("messages")
    if not isinstance(messages, str):
        return None
    manifest = manifest_from_messages_json(messages)
    if not manifest:
        return None
    return AttributeEnrichment(
        additions={"trajectory_manifest": manifest},
        replace_truncated={"messages": "trajectory_manifest"},
        # This name is the built-in manifest contract. Preserve the previous
        # span policy, which recomputed it from raw messages and replaced a
        # caller-provided value.
        replace_existing=frozenset({"trajectory_manifest"}))


_manifest_registration: Optional[_RegistrationToken] = None


def install_trajectory_manifest_enrichment() -> None:
    global _manifest_registration
    with _lock:
        if _manifest_registration is None:
            _manifest_registration = _register_once(
                "observation-trajectory-manifest", _trajectory_manifest_enrichment)


def uninstall_trajectory_manifest_enrichment() -> None:
    global _manifest_registration
    with _lock:
        if _manifest_registration is not None:
            _remove(_manifest_registration)
            _manifest_registration = None
