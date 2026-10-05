"""Package-private mixin of chatbot feedback-note routes, moved verbatim from server.py."""

from __future__ import annotations

import contextlib
import os
from typing import Any

from fastworkflow.observability import feedback, feedback_sidecar
from fastworkflow.observability.comparison import InvalidExecutionRef, PassSelector
from fastworkflow.observability.store import (
    IncompatibleObservabilityDB,
    ObservabilityStore,
)
from fastworkflow.observability.workspace import (
    UnknownLogicalExperiment,
    UnknownWorkspaceStore,
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
            workspace = self.chatbot.workspace
            if workspace is not None:
                with workspace.registry.open(q("store_id") or "") as store:
                    if store.get_turn(turn_key) is None:
                        self._error(404, "turn not found")
                        return
                    # Sealed or not, workspace evidence is never appended to.
                    # The note goes to the annotation sidecar beside it, and
                    # the read is the union, so a reader cannot tell which
                    # file a comment came out of (fix-9eg.19.1). A GET goes
                    # through `reader_for`, which creates nothing: listing
                    # comments must not be what puts a control file beside
                    # somebody's sealed archive.
                    if writing:
                        writable = feedback_sidecar.AnnotatedEvidence.for_writing(store)
                        self._append_feedback_note(
                            writable, turn_key, body, writable=writable
                        )
                        reader = writable
                    else:
                        reader = feedback_sidecar.reader_for(store)
                    self._send_json(
                        {
                            "feedback": feedback.present(
                                reader.list_human_feedback(turn_key)
                            ),
                            "read_only": True,
                            "annotated": True,
                        },
                        status=201 if writing else 200,
                    )
                return
            store = self.chatbot.open_store()
            if store is None or store.get_turn(turn_key) is None:
                self._error(404, "turn not found")
                return
            reader = self._feedback_reader(store)
            if writing:
                self._append_feedback_note(store, turn_key, body)
                reader = self._feedback_reader(store)
            self._send_json(
                {
                    "feedback": feedback.present(reader.list_human_feedback(turn_key)),
                    "read_only": False,
                    "annotated": reader is not store,
                },
                status=201 if writing else 200,
            )
        except (IncompatibleObservabilityDB, UnknownWorkspaceStore,
                feedback_sidecar.FeedbackSidecarError) as exc:
            self._error(409, str(exc))
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
        """Where a new note lands: the evidence database, or a sidecar.

        Appending to the evidence is the ordinary case and stays the default.
        A file this process cannot write, which is what a read-only or sealed
        archive on disk looks like from here, must not be appended to, and
        that is not a reason to refuse somebody's comment.

        It routes to `feedback_sidecar`, a mutable control file beside the
        evidence. Workspace mode never reaches this — it is unconditionally
        annotated, because "writable on disk" is not permission to break a
        seal somebody attested to.
        """
        if not os.access(store.db_path, os.W_OK):
            return feedback_sidecar.AnnotatedEvidence.for_writing(store)
        return ObservabilityStore.open_for_annotation(store.db_path)

    @staticmethod
    def _feedback_reader(store):
        """The store plus its annotation sidecar, if one was ever written.

        A store with no sidecar file reads exactly as before, and reading does
        not create one: `reader_for` opens an existing sidecar read-only and
        otherwise hands back the evidence store untouched.
        """
        return feedback_sidecar.reader_for(store)

    def _append_feedback_note(self, store, turn_key, body, writable=None):
        """Validate one posted note and append it where it is allowed to go.

        The body's `ref` is optional scope on the PRIMARY side; its
        experiment/task/attempt are checked against the turn row rather than
        believed. `paired` names the second execution of a comparison and may
        live in another workspace store, which is resolved and authorized
        here (`_feedback_source_for`) rather than trusted from the request.
        """
        if not self._FEEDBACK_WRITE_REQUIRED <= set(body) or not set(body) <= self._FEEDBACK_WRITE_ALLOWED:
            raise ValueError(
                "provide target_kind, span_ids, target_label, provenance, "
                "category, subcategory and comment; optionally ref, "
                "pass_selector and paired"
            )
        with contextlib.ExitStack() as stack:
            return self._record_feedback_note(store, turn_key, body, writable, stack)

    def _record_feedback_note(self, store, turn_key, body, writable, stack):
        """The write itself, with the paired side's store held open.

        The stack is what lets a comparison comment name evidence in ANOTHER
        archive: the second store is leased from the workspace registry for
        the length of this write, verified on the way in, and released here
        rather than being kept by the anchor it authorized.
        """
        if writable is None:
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
                paired, writable, identity, stack
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

    def _feedback_source_for(self, target, writable, identity, stack=None):
        """The store a paired reference names, or a refusal.

        Two authorized answers and no third: the database being written to,
        and another store the loaded workspace manifest DECLARES by evidence
        identity. A store id nobody declared is not resolved by searching the
        disk.
        """
        if target.store_id == identity:
            return writable
        workspace = self.chatbot.workspace
        if workspace is not None:
            # The manifest already declares each store's `store_identity()`,
            # and `registry.open` re-verifies the archive against it, so the
            # translation from the identity an ExecutionRef carries to the
            # manifest's own store id is a lookup rather than a search of the
            # disk. Leasing it through the registry is what keeps the paired
            # side inside the same authorization as every other workspace
            # read: an undeclared identity raises `UnknownWorkspaceStore`.
            other_id = workspace.registry.store_id_for_identity(target.store_id)
            if stack is None:  # pragma: no cover - callers pass one
                stack = contextlib.ExitStack()
            other = stack.enter_context(workspace.registry.open(other_id))
            # Read through the sidecar reader so the paired side's own
            # annotations are visible to anything that reads back from the
            # authorized set, and creating nothing if it has none.
            return feedback_sidecar.reader_for(other)
        raise feedback.FeedbackError(
            f"store {target.store_id!r} was not authorized for this write; "
            "only the workflow's live database and the workspace's sealed "
            "archives are readable"
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

    def _handle_workspace_task_feedback(self, workspace, q):
        """The task Feedback view over a workspace, scoped by its manifest.

        The store-aware twin of `/api/task-feedback`. The scope is not a
        `store_id` the caller supplies but the segments the manifest already
        declares for the logical experiment, which is what makes a comparison
        comment visible from BOTH of its tasks: the note lives in the sidecar
        beside the left-hand archive, and the right-hand task's view finds it
        because that archive is one of the experiment's own segments. Asking
        the reader to know which archive somebody happened to write in would
        make the read depend on where the comment landed.

        Each segment is queried under its own `local_experiment_id`; the page
        still reports the logical id the reader asked about.
        """
        experiment_id, task_id = q("experiment"), q("task")
        if not experiment_id or not task_id:
            self._error(400, "experiment and task are required")
            return
        try:
            with contextlib.ExitStack() as stack:
                sources, locals_ = {}, []
                for segment in workspace.segments(experiment_id):
                    store = stack.enter_context(
                        workspace.registry.open(segment["store_id"])
                    )
                    identity = store.store_identity() or segment["store_id"]
                    sources[identity] = feedback_sidecar.reader_for(store)
                    if segment["local_experiment_id"] not in locals_:
                        locals_.append(segment["local_experiment_id"])
                page = feedback.consolidate_task_feedback(
                    sources,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    local_experiment_ids=locals_,
                    category=q("category"),
                    subcategory=q("subcategory"),
                    provenance=q("provenance"),
                    target_kind=q("target_kind"),
                    component=q("component"),
                    attempt=(
                        self._int(q("attempt"), None)
                        if q("attempt") is not None
                        else None
                    ),
                    limit=self._int(q("limit"), 100),
                    offset=self._int(q("offset"), 0),
                )
        except (UnknownLogicalExperiment, UnknownWorkspaceStore,
                IncompatibleObservabilityDB,
                feedback_sidecar.FeedbackSidecarError) as exc:
            self._error(409, str(exc))
            return
        except (feedback.FeedbackError, ValueError, TypeError, KeyError) as exc:
            self._error(400, str(exc))
            return
        payload = page.as_dict()
        payload["feedback"] = feedback.present(payload["feedback"])
        payload["read_only"] = True
        self._send_json(payload)

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
        sources = {}
        identity = store.store_identity()
        if identity:
            # Each source is read through `_feedback_reader`, so a store whose
            # evidence could not be appended to still contributes the notes
            # recorded in its sidecar. They merge into one list under one store
            # id and deduplicate on the same `feedback_uid`.
            sources[identity] = self._feedback_reader(store)
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
        except (IncompatibleObservabilityDB, UnknownWorkspaceStore,
                feedback_sidecar.FeedbackSidecarError) as exc:
            self._error(409, str(exc))
            return
        except (feedback.FeedbackError, ValueError, TypeError, KeyError) as exc:
            self._error(400, str(exc))
            return
        payload = page.as_dict()
        payload["feedback"] = feedback.present(payload["feedback"])
        self._send_json(payload)
