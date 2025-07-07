from __future__ import annotations

"""Utility for displaying a startup progress bar/spinner.

The logic is intentionally isolated in this tiny helper so that the rest of
*fastworkflow* can remain completely UI-agnostic.  All public methods are
safe no-ops when the progress bar has not been initialised, so library code
may freely call them regardless of whether an interactive CLI is being
used.
"""

from typing import Optional

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)


class _NullProgress:
    """A stand-in that swallows all method calls (used when no TTY)."""

    def update(self, *_, **__):
        pass

    def stop(self):
        pass


class StartupProgress:
    """Singleton-style helper wrapping *rich*'s *Progress* widget."""

    _progress: Optional[Progress] = None
    _task_id: Optional[int] = None
    _total: int = 0
    _completed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def begin(cls, total: int = 100) -> "StartupProgress":
        """Initialise and display the progress bar.

        If *begin* is called twice we simply ignore the second call and keep
        returning the same singleton so that library code can be agnostic
        about initialisation order.
        """
        if cls._progress is not None:
            return cls

        # Construct the Rich Progress bar – we keep it minimal so it works
        # nicely even on narrow terminals.
        try:
            cls._progress = Progress(
                SpinnerColumn(),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TextColumn("{task.fields[message]}")
            )
            cls._task_id = cls._progress.add_task("startup", total=total, message="")
            cls._progress.start()
        except Exception:
            # If stdout is not a TTY, fall back to a NullProgress so that all
            # subsequent calls are cheap no-ops.
            cls._progress = _NullProgress()  # type: ignore
            cls._task_id = None
        cls._total = total
        cls._completed = 0
        return cls

    @classmethod
    def add_total(cls, delta: int) -> None:
        """Increase the *total* number of expected steps."""
        if cls._progress is None or cls._task_id is None:
            return
        cls._total += delta
        cls._progress.update(cls._task_id, total=cls._total)

    @classmethod
    def advance(cls, message: str = "", step: int = 1) -> None:
        """Advance the bar and optionally show a new *message*."""
        if cls._progress is None or cls._task_id is None:
            return
        cls._completed += step
        cls._progress.update(cls._task_id, advance=step, message=message)

    @classmethod
    def end(cls) -> None:
        """Mark the bar as complete and remove it from the display."""
        if cls._progress is None or cls._task_id is None:
            return
        # Ensure we finish at 100 % even if the caller forgot some steps.
        remaining = cls._total - cls._completed
        if remaining > 0:
            cls._progress.update(cls._task_id, advance=remaining, message="Ready")
        cls._progress.stop()
        cls._progress = None
        cls._task_id = None
        cls._total = 0
        cls._completed = 0 