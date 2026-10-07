"""Package-private mixin of chatbot feedback-note routes, moved verbatim from server.py."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from fastworkflow.observability import control, feedback
from fastworkflow.observability.comparison import InvalidExecutionRef, PassSelector
from fastworkflow.observability.store import (
    IncompatibleObservabilityDB,
    ObservabilityStore,
)


class _FeedbackRoutes:
    def _post_feedback_note(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        if not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        self._handle_feedback_notes(query, body)

    def _handle_feedback_notes(self, query, body=None):
        """Recorded review notes: GET lists a turn's, POST appends one.

        The same route serves a person in the composer and a coding agent
        posting over HTTP. Only `provenance` distinguishes them, and neither
        gets to skip the category, the subcategory, or the validation of the
        evidence its anchors name.
        """
        q = lambda key: (query.get(key) or [None])[0]  # noqa: E731
        writing = body is not None
        if writing and not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        turn_key = q("turn_key")
        if not turn_key and writing and isinstance(body.get("ref"), dict):
            turn_key = body["ref"].get("turn_key")
        if not turn_key:
            self._error(400, "turn_key is required")
            return
        try:
            store = self.chatbot.open_store()
            if store is None or store.get_turn(turn_key) is None:
                self._error(404, "turn not found")
                return
            if writing:
                self._append_feedback_note(store, turn_key, body)
            self._send_json(
                {
                    "feedback": feedback.present(store.list_human_feedback(turn_key)),
                },
                status=201 if writing else 200,
            )
        except (IncompatibleObservabilityDB, control.ControlUnavailable) as exc:
            self._error(409, str(exc))
        except sqlite3.OperationalError as exc:
            status = control.write_refusal_status(exc)
            if status is None:
                raise
            self._error(status, str(exc))
        except (feedback.FeedbackError, InvalidExecutionRef) as exc:
            self._error(400, str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            self._error(400, str(exc))

    _FEEDBACK_WRITE_REQUIRED = {
        "target_kind", "span_ids", "target_label", "provenance",
        "comment", "category", "subcategory",
    }
    _FEEDBACK_WRITE_ALLOWED = _FEEDBACK_WRITE_REQUIRED | {
        "ref", "pass_selector", "paired",
    }

    def _feedback_writer(self, store):
        """Where a new note lands: the live evidence database.

        A file this process cannot write is refused rather than diverted:
        there is no second file for a comment to land in, so a note that
        cannot be appended is reported as not recorded (§2.5).
        """
        if not os.access(store.db_path, os.W_OK):
            raise control.ControlUnavailable(
                f"the live evidence database {store.db_path} is read-only for "
                "this process, so the comment was not recorded"
            )
        return ObservabilityStore.open_for_annotation(store.db_path)

    def _append_feedback_note(self, store, turn_key, body):
        """Validate one posted note and append it where it is allowed to go.

        The body's `ref` is optional scope on the PRIMARY side; its
        experiment/task/attempt are checked against the turn row rather than
        believed. `paired` names the second execution of a comparison, which
        is authorized here (`_feedback_source_for`) rather than trusted from
        the request.
        """
        if not self._FEEDBACK_WRITE_REQUIRED <= set(body) or not set(body) <= self._FEEDBACK_WRITE_ALLOWED:
            raise ValueError(
                "provide target_kind, span_ids, target_label, provenance, "
                "category, subcategory and comment; optionally ref, "
                "pass_selector and paired"
            )
        return self._record_feedback_note(store, turn_key, body)

    def _record_feedback_note(self, store, turn_key, body):
        """The write itself, against the live evidence database."""
        writable = self._feedback_writer(store)
        identity = writable.store_identity()
        primary_raw = dict(body.get("ref") or {})
        primary_raw.update({
            "store_id": primary_raw.get("store_id") or identity,
            "turn_key": turn_key,
            "target_kind": body["target_kind"],
            "span_ids": body["span_ids"],
            "target_label": body["target_label"],
        })
        primary = feedback.FeedbackTarget.from_mapping(primary_raw)
        sources = {identity: writable}
        paired = paired_selector = None
        if body.get("paired") is not None:
            if not isinstance(body["paired"], dict):
                raise ValueError("paired must be an object")
            paired = feedback.FeedbackTarget.from_mapping(body["paired"])
            sources[paired.store_id] = self._feedback_source_for(
                paired, writable, identity
            )
            paired_selector = self._pass_selector(body["paired"].get("pass_selector"))
        return feedback.record_feedback(
            writable,
            primary=primary,
            comment=body["comment"],
            category=body["category"],
            subcategory=body["subcategory"],
            provenance=body["provenance"],
            sources=sources,
            paired=paired,
            primary_pass_selector=self._pass_selector(body.get("pass_selector")),
            paired_pass_selector=paired_selector,
        )

    @staticmethod
    def _feedback_source_for(target, writable, identity):
        """The store a paired reference names, or a refusal.

        One authorized answer: the database being written to. A store id it
        does not report is not resolved by searching the disk.
        """
        if target.store_id == identity:
            return writable
        raise feedback.FeedbackError(
            f"store {target.store_id!r} was not authorized for this write; "
            "only the workflow's live database is readable"
        )

    @staticmethod
    def _pass_selector(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("pass_selector must be an object")
        try:
            return PassSelector(
                pass_id=str(value.get("pass_id") or ""),
                attribute_key=value.get("attribute_key"),
                attribute_value=value.get("attribute_value"),
                root_span_ids=frozenset(value.get("root_span_ids") or ()),
                span_ids=frozenset(value.get("span_ids") or ()),
                exclude_span_ids=frozenset(value.get("exclude_span_ids") or ()),
            )
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"invalid pass_selector: {exc}") from exc

    def _handle_task_feedback(self, store, query):
        """Every authorized comment on one task, across attempts.

        No hidden default filter: without query parameters this answers the
        whole authorized record for the task, including comparison comments
        anchored on the other side of a pair and task-level summaries. The
        filters below narrow it only when a reader asks.
        """
        q = lambda name: (query.get(name) or [None])[0]  # noqa: E731
        experiment_id, task_id = q("experiment"), q("task")
        if not experiment_id or not task_id:
            self._error(400, "experiment and task are required")
            return
        identity = store.store_identity()
        sources = {identity: store} if identity else {}
        try:
            page = feedback.consolidate_task_feedback(
                sources,
                experiment_id=experiment_id,
                task_id=task_id,
                category=q("category"),
                subcategory=q("subcategory"),
                provenance=q("provenance"),
                target_kind=q("target_kind"),
                component=q("component"),
                attempt=self._int(q("attempt"), None) if q("attempt") is not None else None,
                limit=self._int(q("limit"), 100),
                offset=self._int(q("offset"), 0),
            )
        except IncompatibleObservabilityDB as exc:
            self._error(409, str(exc))
            return
        except (feedback.FeedbackError, ValueError, TypeError, KeyError) as exc:
            self._error(400, str(exc))
            return
        payload = page.as_dict()
        payload["feedback"] = feedback.present(payload["feedback"])
        self._send_json(payload)
