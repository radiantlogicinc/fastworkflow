"""The exact prompt an LLM call received, stored as content-addressed pieces.

An ``fw.llm.call`` span records the wire ``messages`` as one attribute, and an
attribute over ``tracing.MAX_ATTR_BYTES`` is cut. Agent prompts pass that bound
routinely -- they repeat the whole trajectory on every step -- so the record of
what the model was shown stopped at the cut, and the trajectory-manifest
enricher then dropped even the prefix.

This module splits such a prompt into pieces: each message's text is cut at
the DSPy field markers (``[[ ## name ## ]]``), so the system prompt, the user
query and every trajectory field is one piece. Pieces are keyed by the sha256
of their text, which is what makes them cheap to keep: step N+1's prompt is
step N's plus a few new pieces, and a piece already stored for the turn is
not stored again.

What the span keeps is ``prompt_slots_ref``: the messages with every text
replaced by the ordered list of its piece digests, and the sha256 of the
messages as recorded. ``rebuild`` puts the text back and says whether the
result hashes to that digest, so a reader is shown the prompt as sent, or is
told exactly which pieces it cannot be shown.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Optional

from fastworkflow import tracing
from fastworkflow.observability import enrichment

PROMPT_SLOTS_CONTRACT = "fastworkflow-prompt-slots/1"
REF_ATTRIBUTE = "prompt_slots_ref"
#: The marker a text is replaced by inside the stored template.
SLOTS_KEY = "__fw_prompt_slots__"
#: The template is stored uncapped on the span (enrichment additions are), so
#: it gets a bound of its own. Messages carry roles and texts; a template over
#: this holds something else large (inline image data), and is not recorded.
MAX_TEMPLATE_BYTES = tracing.MAX_ATTR_BYTES

_FIELD_BOUNDARY = re.compile(r"(?=\[\[ ## )")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_text(text: str) -> list[str]:
    """Cut *text* in front of every DSPy field marker; the pieces join back to it."""
    return [piece for piece in _FIELD_BOUNDARY.split(text) if piece]


def _slotted(text: str, texts: dict[str, str]) -> dict[str, list[str]]:
    digests = []
    for piece in split_text(text):
        digest = _digest(piece)
        texts.setdefault(digest, piece)
        digests.append(digest)
    return {SLOTS_KEY: digests}


def _template_of(message: Any, texts: dict[str, str]) -> Any:
    if not isinstance(message, dict):
        return message
    template = dict(message)
    content = message.get("content")
    if isinstance(content, str):
        template["content"] = _slotted(content, texts)
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part = {**part, "text": _slotted(part["text"], texts)}
            parts.append(part)
        template["content"] = parts
    return template


def build(messages_json: str) -> Optional[tuple[dict[str, Any], dict[str, str]]]:
    """``(ref, texts)`` for a recorded ``messages`` value, or None.

    None when the value is not a JSON list of messages, holds no text, or
    leaves a template too large to keep on the span.
    """
    if not isinstance(messages_json, str) or not messages_json:
        return None
    try:
        parsed = json.loads(messages_json)
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    texts: dict[str, str] = {}
    template = [_template_of(message, texts) for message in parsed]
    if not texts:
        return None
    if len(json.dumps(template, ensure_ascii=False).encode("utf-8")) > MAX_TEMPLATE_BYTES:
        return None
    ref = {
        "contract": PROMPT_SLOTS_CONTRACT,
        "messages_sha256": _digest(messages_json),
        "messages_bytes": len(messages_json.encode("utf-8")),
        "slot_count": len(texts),
        "template": template,
    }
    return ref, texts


def slot_digests(ref: Mapping[str, Any]) -> list[str]:
    """Every piece digest *ref* names, once each, in first-use order."""
    found: dict[str, None] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if SLOTS_KEY in value and isinstance(value[SLOTS_KEY], list):
                found.update((str(d), None) for d in value[SLOTS_KEY])
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(ref.get("template"))
    return list(found)


def rebuild(ref: Mapping[str, Any], stored: Mapping[str, str]) -> dict[str, Any]:
    """The messages *ref* describes, from the stored piece texts.

    ``stored`` maps a digest to the text kept for it. A piece kept under a
    capture policy that redacted or withheld it holds text that no longer
    hashes to its digest; it is shown as kept and listed in ``altered``. A
    piece with no row at all is shown as a placeholder and listed in
    ``missing``. ``verified`` is True only when the rebuilt messages hash to
    the recorded digest, i.e. this is byte for byte what the model was sent.
    """
    missing: list[str] = []
    altered: list[str] = []

    def text_for(digest: str) -> str:
        text = stored.get(digest)
        if text is None:
            if digest not in missing:
                missing.append(digest)
            return f"[piece {digest[:12]} not stored]"
        if _digest(text) != digest and digest not in altered:
            altered.append(digest)
        return text

    def fill(value: Any) -> Any:
        if isinstance(value, dict):
            if SLOTS_KEY in value and isinstance(value[SLOTS_KEY], list):
                return "".join(text_for(str(d)) for d in value[SLOTS_KEY])
            return {key: fill(child) for key, child in value.items()}
        if isinstance(value, list):
            return [fill(child) for child in value]
        return value

    messages = fill(ref.get("template") or [])
    rebuilt_digest = _digest(json.dumps(messages, ensure_ascii=False))
    return {
        "messages": messages,
        "verified": rebuilt_digest == ref.get("messages_sha256"),
        "messages_sha256": ref.get("messages_sha256"),
        "messages_bytes": ref.get("messages_bytes"),
        "missing": missing,
        "altered": altered,
    }


def _prompt_slot_enrichment(
    attributes: Mapping[str, Any],
) -> Optional[enrichment.AttributeEnrichment]:
    messages = attributes.get("messages")
    if not isinstance(messages, str):
        return None
    if len(messages.encode("utf-8")) <= tracing.MAX_ATTR_BYTES:
        return None
    built = build(messages)
    if built is None:
        return None
    ref, texts = built
    return enrichment.AttributeEnrichment(
        additions={REF_ATTRIBUTE: ref, tracing.ATTR_PROMPT_SLOT_TEXTS: texts},
    )


_registration: Optional[Any] = None


def install_prompt_slot_enrichment() -> None:
    """Record over-cap prompts as pieces. Idempotent; process-wide."""
    global _registration
    _registration = enrichment.register_builtin("prompt-slots", _prompt_slot_enrichment)


def uninstall_prompt_slot_enrichment() -> None:
    global _registration
    if _registration is not None:
        enrichment.unregister(_registration)
        _registration = None
