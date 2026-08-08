"""One injective mapping from an identifier to a filesystem name.

Shared rather than copied because two stores key durable records by channel id —
`checkpoint_store` (one directory per channel) and `session_state_store` (one
pending blob per channel) — and a second encoder only has to differ by one
character class to put two channels back on one path. That is not a
hypothetical: the pending store had its own folding rule, which collapsed
``tenant/user-1`` and ``tenant_user-1`` onto one file and could hand a user a
stranger's suspended conversation (fix-7hn). It lives here so the two
derivations cannot drift apart again.

Nothing in here imports from `fastworkflow`, so it stays importable from any
layer that needs a storage key.
"""

from __future__ import annotations

import hashlib

# NAME_MAX is 255 bytes almost everywhere; percent-encoding can triple a name.
# 200 leaves room for the ".{64 hex}" tail and for temp-name suffixes.
_MAX_NAME_LEN = 200

# Lowercase only. Uppercase is escaped because a case-insensitive volume folds
# `Tenant` onto `tenant`, and a fold is an alias the record-level identity check
# would only ever report *after* one channel had already overwritten the other.
# `.` is escaped too, which reserves it as an unambiguous delimiter for the
# oversized-id tail hash and rules out the `.` and `..` directory names.
_UNRESERVED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")


def encode_path_component(raw: str) -> str:
    """Percent-encode ``raw`` into an injective, filesystem-safe ASCII name.

    Injective because decoding is unambiguous: `%` is itself always escaped, so
    every `%` in the output opens an escape, and every other output character
    stands for itself. Hence ``tenant/a`` -> ``tenant%2Fa``, ``tenant_a`` ->
    ``tenant_a`` and ``tenant%2Fa`` -> ``tenant%252Fa`` are three names.

    Injective *after case folding* as well, because escapes emit uppercase hex
    and nothing else emits a capital: two outputs that fold together must have
    identical `%` positions and therefore be identical.

    Oversized ids fall back to ``<prefix>.<sha256 of the raw id>``. That is
    collision-*resistant* rather than injective, which is why the raw id is also
    stored in the record and compared on read: an astronomically unlikely
    collision quarantines instead of cross-serving state.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError("path component must be a non-empty str")

    encoded = "".join(
        chr(byte) if chr(byte) in _UNRESERVED else f"%{byte:02X}"
        for byte in raw.encode("utf-8")
    )
    if len(encoded) <= _MAX_NAME_LEN:
        return encoded

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    prefix = _trim_partial_escape(encoded[: _MAX_NAME_LEN - len(digest) - 1])
    return f"{prefix}.{digest}"


def _trim_partial_escape(text: str) -> str:
    """Drop a truncated `%XX` so the readable prefix stays decodable."""
    if text.endswith("%"):
        return text[:-1]
    return text[:-2] if len(text) >= 2 and text[-2] == "%" else text
