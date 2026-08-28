"""MetricsSink protocol and default sinks (turn_result_design_final.md §11 [X7]).

Minimal v1 surface: a counter/histogram protocol, a no-op default, and a
log-emitting fallback. Core metric names emitted today:

- ``fw_turns_total{status}``           — one increment per finalized turn
- ``fw_turn_duration_seconds{status}`` — end-to-end logical-turn duration

The remaining §11 names (``fw_record_write_failures_total``, ...) are claimed
by this catalog and land with the writers that produce them (Phase 2+).

Stdlib-only by design (imported by core runtime modules).
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class MetricsSink(Protocol):
    """Counter/histogram sink. Implementations must never raise to callers."""

    def increment(self, name: str, value: int = 1, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...


class NoOpMetricsSink:
    """Default sink: metrics structurally present, nothing recorded."""

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        pass

    def observe(self, name: str, value: float, **labels: str) -> None:
        pass


class LoggingMetricsSink:
    """Log-emitting fallback: each metric event becomes one debug log line."""

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        logger.debug(f"metric increment {name}{labels or ''} +{value}")

    def observe(self, name: str, value: float, **labels: str) -> None:
        logger.debug(f"metric observe {name}{labels or ''} = {value}")


def safe_increment(sink: MetricsSink, name: str, value: int = 1, **labels: str) -> None:
    """Increment, swallowing sink failures (a broken sink never fails a turn)."""
    try:
        sink.increment(name, value, **labels)
    except Exception as exc:
        logger.warning(f"MetricsSink.increment({name}) failed: {exc!r}")


def safe_observe(sink: MetricsSink, name: str, value: float, **labels: str) -> None:
    """Observe, swallowing sink failures (a broken sink never fails a turn)."""
    try:
        sink.observe(name, value, **labels)
    except Exception as exc:
        logger.warning(f"MetricsSink.observe({name}) failed: {exc!r}")
