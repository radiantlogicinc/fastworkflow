"""What the FastAPI feedback routes are allowed to read, write and resolve.

Two defects, both found by adversarial review of the feedback surface and both
about the routes trusting something that is not a boundary.

fix-bnym — CHANNEL OWNERSHIP. The observability store is per WORKFLOW: every
channel this server serves appends its evidence to the same SQLite file. The
feedback routes treated presence in that file as permission, so
``GET /feedback`` answered another channel's turn with its private notes and
``POST /post_feedback`` appended to it — while ``GET /turns`` on the very same
key correctly returned 404. Ownership is now checked on every turn a note
names, primary and paired, before any evidence question is asked.

fix-jxkk — RECORDED PASSES. ``pass_id`` has been on the wire for both anchors
since they landed, and nothing ever supplied the ``PassSelector`` that makes
one resolvable, so ``validate_target`` refused every reference naming a real
recorded pass exactly as it refused imaginary ones. The selector is now
discovered from the anchored turn's own ``fw.pass`` spans — discovered, never
described by the request, so a caller names a pass and cannot assert
membership for activity the span tree does not place in it.

Everything here runs against the real FastAPI app, real JWTs and a real
observability store. Turns are recorded two ways on purpose: through the real
turn lifecycle (which is what proves the stored ``channel_id`` a real turn
writes is the one the ownership check compares against), and by seeding rows
and spans directly (which is how a second channel's evidence and a
pass-stamped trace get into the store without a second server).

No model call is made by any test in this file.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow import tracing

# The hermetic server fixtures. test_fastapi_service's fixtures skip when the
# repo's private env files are absent; these write their own from the shipped
# template, so this file runs anywhere.
from tests.test_fastapi_streaming_lifecycle import (  # noqa: F401
    app_module,
    env_files,
    hello_world_workflow_path,
)


ALICE = "feedback-scope-alice"
BOB = "feedback-scope-bob"


def _initialize(client: TestClient, channel_id: str) -> dict[str, str]:
    resp = client.post("/initialize", json={"channel_id": channel_id})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _runtime(app_module, channel_id: str):
    return asyncio.run(app_module.session_manager.get_session(channel_id))


def _record_real_turn(runtime, message: str) -> str:
    """Drive one real logical turn and return the key it recorded under.

    Begin, append, finalize — the same chokepoint a served turn goes through,
    so the ``channel_id`` in the stored row is written by the production path
    rather than by this test. Nothing here reaches the NLU pipeline or a model.
    """
    ctx = runtime.execution_context
    ctx._begin_turn(message)
    ctx.append_conversation_turn(message, None)
    ctx._build_turn_result(
        fastworkflow.CommandOutput(
            command_name="",
            command_response=fastworkflow.CommandResponse(response="ok"),
        )
    )
    ctx.trace_sink.flush()
    return ctx.last_completed_turn_key


def _turn_row(turn_key: str, channel_id: str) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": None,
        "ordinal": None,
        "user_message": "do the thing",
        "refined_user_message": None,
        "entry_workflow_name": "feedback-scope-test",
        "entry_context": "global",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": "done",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-19T00:00:00+00:00",
        "completed_at": "2026-09-19T00:00:02+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": None,
        "task_id": None,
        "attempt": None,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps({}),
    }


def _pass_span(
    span_id: str,
    turn_key: str,
    channel_id: str,
    pass_id: str | None,
    start_ns: int,
    parent_span_id: str | None = None,
) -> tracing.Span:
    """One dispatch, stamped with the pass that made it as a producer stamps it.

    ``pass_id=None`` stamps nothing, which is how a span that belongs to no
    pass — or one that inherits its pass from an ancestor — is recorded.
    """
    attributes: dict = {
        tracing.ATTR_COMMAND_CALL_ID: f"call-{span_id}",
        "success": True,
    }
    if pass_id is not None:
        attributes[tracing.ATTR_PASS] = pass_id
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=parent_span_id,
        name=tracing.SPAN_COMMAND_EXECUTE,
        kind=tracing.KIND_INTERNAL,
        channel_id=channel_id,
        command_name="add_two_numbers",
        context="global",
        start_ns=start_ns,
        end_ns=start_ns + 1_000_000,
        status=tracing.STATUS_OK,
        attributes=attributes,
    )


def _seed(store, turn_key: str, channel_id: str, spans: list | None = None) -> str:
    """Put one recorded turn (and optionally its spans) into the real store."""
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn, _turn_row(turn_key, channel_id), [], store._store_redactor()
        )
        if spans:
            store.upsert_span_rows(conn, spans, store._store_redactor())
        conn.commit()
    return turn_key


def _note(turn_key: str, comment: str, **overrides) -> dict:
    body = {
        "turn_key": turn_key,
        "target_kind": "turn",
        "target_label": "Turn",
        "comment": comment,
        "category": "conclusions",
        "subcategory": "what_went_right",
        "provenance": "coding_agent",
    }
    body.update(overrides)
    return body


@pytest.fixture
def two_channels(app_module):
    """Alice and Bob on one server, sharing one evidence store.

    Yields the client, both channels' headers, and the store — which is the
    SAME object for both, because that sharing is the whole hazard.
    """
    client = TestClient(app_module.app)
    alice = _initialize(client, ALICE)
    bob = _initialize(client, BOB)
    store = _runtime(app_module, ALICE).observability_store
    assert store is not None, "these tests are about a real evidence store"
    assert store is _runtime(app_module, BOB).observability_store
    return client, alice, bob, store


# ----------------------------------------------------------------------
# fix-bnym: channel ownership
# ----------------------------------------------------------------------


def test_a_real_turn_stays_readable_and_writable_by_its_own_channel(
    app_module, two_channels
):
    """The check must pass what the production write path records.

    Comparing a request's JWT channel against a stored column is only safe if
    the column holds what the turn actually wrote, so this records through the
    real turn lifecycle rather than seeding a row. A historical turn — one
    recorded earlier, no longer the runtime's last — stays usable too: the
    fix narrows WHO may comment, not WHEN.
    """
    client, alice, _bob, _store = two_channels
    runtime = _runtime(app_module, ALICE)
    historical = _record_real_turn(runtime, "the first exchange")
    latest = _record_real_turn(runtime, "the second exchange")
    assert historical != latest

    for turn_key, comment in ((historical, "about the first"), (latest, "about the second")):
        written = client.post("/post_feedback", headers=alice, json=_note(turn_key, comment))
        assert written.status_code == 201, written.text
        listed = client.get("/feedback", headers=alice, params={"turn_key": turn_key})
        assert listed.status_code == 200, listed.text
        assert [row["comment"] for row in listed.json()["feedback"]] == [comment]


def test_reading_another_channels_feedback_is_a_404(two_channels):
    """GET /feedback used to hand Bob Alice's private note verbatim."""
    client, alice, bob, store = two_channels
    turn_key = _seed(store, "scope-alice-private", ALICE)
    assert client.post(
        "/post_feedback", headers=alice, json=_note(turn_key, "Alice private review")
    ).status_code == 201

    leaked = client.get("/feedback", headers=bob, params={"turn_key": turn_key})
    assert leaked.status_code == 404, leaked.text
    # Not one character of the note, and no confirmation that the key resolves.
    assert "Alice private review" not in leaked.text
    assert leaked.json()["detail"] == f"Turn not found: {turn_key}"


def test_writing_on_another_channels_turn_is_a_404_and_records_nothing(two_channels):
    """POST used to return 201 and append Bob's note to Alice's turn."""
    client, alice, bob, store = two_channels
    turn_key = _seed(store, "scope-alice-target", ALICE)
    assert client.post(
        "/post_feedback", headers=alice, json=_note(turn_key, "Alice private review")
    ).status_code == 201

    forged = client.post(
        "/post_feedback", headers=bob, json=_note(turn_key, "Bob wrote on Alice's turn")
    )
    assert forged.status_code == 404, forged.text
    # A refused write returns no prior comments: the refusal is the whole body.
    assert "Alice private review" not in forged.text
    assert set(forged.json()) == {"detail"}

    owner_view = client.get("/feedback", headers=alice, params={"turn_key": turn_key})
    assert [row["comment"] for row in owner_view.json()["feedback"]] == [
        "Alice private review"
    ]


def test_a_foreign_turn_is_indistinguishable_from_an_unknown_one(two_channels):
    """Same status and same detail, or the route is an existence oracle."""
    client, _alice, bob, store = two_channels
    foreign = _seed(store, "scope-foreign", ALICE)
    unknown = "scope-never-recorded"

    reads = [
        client.get("/feedback", headers=bob, params={"turn_key": key})
        for key in (foreign, unknown)
    ]
    writes = [
        client.post("/post_feedback", headers=bob, json=_note(key, "probe"))
        for key in (foreign, unknown)
    ]
    for pair in (reads, writes):
        assert [r.status_code for r in pair] == [404, 404], [r.text for r in pair]
    # The keys differ, so only the shape of the detail can be compared.
    for response, key in zip(reads + writes, [foreign, unknown] * 2):
        assert response.json()["detail"] == f"Turn not found: {key}"


def test_the_404_is_the_one_get_turns_already_returns(two_channels):
    """Same status and same detail as the route that had it right all along.

    Asserted against the live ``GET /turns`` rather than against a copy of its
    string, so the three refusals cannot drift apart: if /turns ever changes
    how it declines a foreign key, this fails until the feedback routes follow.
    """
    client, _alice, bob, store = two_channels
    turn_key = _seed(store, "scope-404-shape", ALICE)

    canonical = client.get(f"/turns/{turn_key}", headers=bob)
    read = client.get("/feedback", headers=bob, params={"turn_key": turn_key})
    write = client.post("/post_feedback", headers=bob, json=_note(turn_key, "probe"))

    assert canonical.status_code == 404, canonical.text
    for response in (read, write):
        assert response.status_code == canonical.status_code
        assert response.json()["detail"] == canonical.json()["detail"]


def test_a_padded_turn_key_cannot_slip_past_ownership(two_channels):
    """Authorization and evidence must agree on ONE spelling of a key.

    ``ExecutionRef`` strips its turn keys, so the key the write lands under is
    not necessarily the key the request typed. Checking ownership against the
    raw text would authorize `" alice-turn "` (no such row -> would have to
    fall one way or the other) and then write to `"alice-turn"`. The check runs
    on the normalized reference, so both halves see the same string.
    """
    client, alice, bob, store = two_channels
    turn_key = _seed(store, "scope-padded", ALICE)
    padded = f"  {turn_key}  "

    forged = client.post("/post_feedback", headers=bob, json=_note(padded, "probe"))
    assert forged.status_code == 404, forged.text
    assert forged.json()["detail"] == f"Turn not found: {turn_key}"

    # And the owner's padded key is accepted, landing on the canonical row —
    # which is what makes the paragraph above a real hazard rather than a
    # theoretical one.
    written = client.post("/post_feedback", headers=alice, json=_note(padded, "mine"))
    assert written.status_code == 201, written.text
    listed = client.get("/feedback", headers=alice, params={"turn_key": turn_key})
    assert [row["comment"] for row in listed.json()["feedback"]] == ["mine"]


def test_a_paired_anchor_naming_another_store_is_refused(two_channels):
    """A reference cannot reach a database this channel was not authorized for.

    Refused as a bad request rather than a 404: the complaint is about the
    store the caller named, and it says nothing about whether any turn exists.
    """
    client, alice, _bob, store = two_channels
    mine = _seed(store, "scope-store-mine", ALICE)
    theirs = _seed(store, "scope-store-theirs", BOB)

    refused = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            mine,
            "reaching for another database",
            paired={
                "turn_key": theirs,
                "target_label": "Other",
                "target_kind": "turn",
                "store_id": "some-other-evidence-store",
            },
        ),
    )
    assert refused.status_code == 400, refused.text
    assert "some-other-evidence-store" in refused.json()["detail"]
    assert client.get(
        "/feedback", headers=alice, params={"turn_key": mine}
    ).json()["feedback"] == []


def test_a_paired_anchor_in_another_channel_is_refused(two_channels):
    """The second side of a comparison is evidence too, and is checked as such.

    A note whose primary is the caller's own turn would otherwise carry a
    frozen reference to somebody else's — validated against their evidence,
    and reachable forever from the row.
    """
    client, alice, _bob, store = two_channels
    mine = _seed(store, "scope-pair-mine", ALICE)
    theirs = _seed(store, "scope-pair-theirs", BOB)

    refused = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            mine,
            "mine beat theirs",
            paired={"turn_key": theirs, "target_label": "Other", "target_kind": "turn"},
        ),
    )
    assert refused.status_code == 404, refused.text
    assert refused.json()["detail"] == f"Turn not found: {theirs}"

    listed = client.get("/feedback", headers=alice, params={"turn_key": mine})
    assert listed.json()["feedback"] == []


def test_a_paired_anchor_in_the_same_channel_is_recorded(two_channels):
    """Ownership narrows the pair; it does not abolish comparisons."""
    client, alice, _bob, store = two_channels
    left = _seed(store, "scope-pair-left", ALICE)
    right = _seed(store, "scope-pair-right", ALICE)

    written = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            left,
            "the left run explained itself and the right one did not",
            paired={"turn_key": right, "target_label": "Right", "target_kind": "turn"},
        ),
    )
    assert written.status_code == 201, written.text

    listed = client.get("/feedback", headers=alice, params={"turn_key": left})
    row = listed.json()["feedback"][-1]
    anchors = row["anchors"]
    assert anchors["primary"]["turn_key"] == left
    assert anchors["paired"]["turn_key"] == right
    assert anchors["pair_key"]


# ----------------------------------------------------------------------
# fix-jxkk: pass selectors resolved from recorded evidence
# ----------------------------------------------------------------------


def _seed_two_pass_turn(store, turn_key: str, channel_id: str) -> str:
    """One turn, two stamped passes, plus a child that inherits and an orphan.

    ``<key>-teacher-child`` stamps nothing of its own: it hangs off the teacher
    root, so the only thing that puts it in the teacher pass is the ancestry
    walk. ``<key>-orphan`` hangs off nothing and stamps nothing, so it is in no
    pass at all. Both are here so a membership check can be shown to be reading
    the span tree rather than just comparing stamps.
    """
    return _seed(
        store,
        turn_key,
        channel_id,
        [
            _pass_span(f"{turn_key}-teacher", turn_key, channel_id, "teacher", 1_700_000_000_000_000_000),
            _pass_span(f"{turn_key}-student", turn_key, channel_id, "student", 1_700_000_000_001_000_000),
            _pass_span(
                f"{turn_key}-teacher-child",
                turn_key,
                channel_id,
                None,
                1_700_000_000_002_000_000,
                parent_span_id=f"{turn_key}-teacher",
            ),
            _pass_span(f"{turn_key}-orphan", turn_key, channel_id, None, 1_700_000_000_003_000_000),
        ],
    )


def test_a_recorded_primary_pass_is_accepted(two_channels):
    """`pass_id=teacher` on a turn whose spans stamp teacher used to be a 400."""
    client, alice, _bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-primary", ALICE)

    written = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(turn_key, "the teacher pass reasoned it out", pass_id="teacher"),
    )
    assert written.status_code == 201, written.text

    row = client.get(
        "/feedback", headers=alice, params={"turn_key": turn_key}
    ).json()["feedback"][-1]
    # The pass is frozen into the anchor, so the note is about the pass rather
    # than about whichever pass a later reader assumes.
    assert row["anchors"]["primary"]["ref"]["pass_id"] == "teacher"


def test_a_recorded_paired_pass_is_accepted(two_channels):
    """Teacher against student within one turn: two references, one trace."""
    client, alice, _bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-paired", ALICE)

    written = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            turn_key,
            "the student pass dropped the teacher's second step",
            pass_id="student",
            paired={
                "turn_key": turn_key,
                "target_label": "Teacher",
                "target_kind": "turn",
                "pass_id": "teacher",
            },
        ),
    )
    assert written.status_code == 201, written.text

    anchors = client.get(
        "/feedback", headers=alice, params={"turn_key": turn_key}
    ).json()["feedback"][-1]["anchors"]
    assert anchors["primary"]["ref"]["pass_id"] == "student"
    assert anchors["paired"]["ref"]["pass_id"] == "teacher"
    # Two passes of one turn are two different executions, which is exactly
    # what makes them comparable at all.
    assert anchors["primary"]["ref"]["ref_id"] != anchors["paired"]["ref"]["ref_id"]


def test_an_unrecorded_pass_is_refused_rather_than_relabelled(two_channels):
    """A pass the trace does not stamp must not borrow a neighbour's spans."""
    client, alice, _bob, store = two_channels
    turn_key = _seed(
        store,
        "scope-pass-teacher-only",
        ALICE,
        [_pass_span("scope-pass-only-span", "scope-pass-teacher-only", ALICE, "teacher", 1_700_000_000_000_000_000)],
    )

    refused = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(turn_key, "about a pass that never ran", pass_id="student"),
    )
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert "'student'" in detail
    # It says what the turn DOES record, so the caller can correct itself
    # without guessing — and it plainly did not silently answer as teacher.
    assert "teacher" in detail

    listed = client.get("/feedback", headers=alice, params={"turn_key": turn_key})
    assert listed.json()["feedback"] == []


def test_a_pass_on_a_turn_that_records_none_is_refused(two_channels):
    """An ordinary single-pass turn stamps nothing, and says so."""
    client, alice, _bob, store = two_channels
    turn_key = _seed(store, "scope-pass-none", ALICE)

    refused = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(turn_key, "which pass?", pass_id="teacher"),
    )
    assert refused.status_code == 400, refused.text
    assert "none" in refused.json()["detail"]


def test_a_paired_pass_is_checked_against_its_own_turns_evidence(two_channels):
    """The paired side resolves its own selector, not the primary's."""
    client, alice, _bob, store = two_channels
    stamped = _seed_two_pass_turn(store, "scope-pass-stamped", ALICE)
    plain = _seed(store, "scope-pass-plain", ALICE)

    refused = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            stamped,
            "compared against a pass the other turn never recorded",
            pass_id="teacher",
            paired={
                "turn_key": plain,
                "target_label": "Other",
                "target_kind": "turn",
                "pass_id": "teacher",
            },
        ),
    )
    assert refused.status_code == 400, refused.text
    assert "scope-pass-plain" in refused.json()["detail"]
    assert client.get(
        "/feedback", headers=alice, params={"turn_key": stamped}
    ).json()["feedback"] == []


# ----------------------------------------------------------------------
# fix-jxkk, second finding: the named spans must be IN the named pass
# ----------------------------------------------------------------------
#
# `feedback.validate_target` asks two questions about a pass-scoped component
# target and neither of them is this one. It checks that the span ids are
# recorded ON THE TURN, and it calls `PassSelector.resolve_against`, which only
# asks whether the selector matches SOME span of the turn. A turn holding both
# a teacher pass and a student pass answers yes to both for either pass, so a
# caller could anchor to the student's span while declaring `pass_id=teacher`
# and have the store freeze that claim into the row forever. Membership is
# resolved here, from the span tree, with `comparison._pass_id_for` — the same
# function the projection attributes steps with.


def test_a_span_from_another_pass_is_refused_on_the_primary(two_channels):
    """`pass_id=teacher` + the student's span is a forged attribution."""
    client, alice, _bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-mix-primary", ALICE)

    refused = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            turn_key,
            "blaming the teacher for the student's step",
            target_kind="span",
            span_ids=[f"{turn_key}-student"],
            pass_id="teacher",
        ),
    )
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert f"{turn_key}-student" in detail
    assert "'teacher'" in detail

    assert client.get(
        "/feedback", headers=alice, params={"turn_key": turn_key}
    ).json()["feedback"] == []


def test_a_span_from_another_pass_is_refused_on_the_paired_side(two_channels):
    """The paired anchor is frozen into the row too, so it is checked too."""
    client, alice, _bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-mix-paired", ALICE)

    refused = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            turn_key,
            "a real teacher anchor against a forged student one",
            target_kind="span",
            span_ids=[f"{turn_key}-teacher"],
            pass_id="teacher",
            paired={
                "turn_key": turn_key,
                "target_label": "Student step",
                "target_kind": "span",
                "span_ids": [f"{turn_key}-teacher"],
                "pass_id": "student",
            },
        ),
    )
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert f"{turn_key}-teacher" in detail
    assert "'student'" in detail

    assert client.get(
        "/feedback", headers=alice, params={"turn_key": turn_key}
    ).json()["feedback"] == []


def test_a_span_in_no_pass_at_all_is_refused(two_channels):
    """An unstamped orphan belongs to neither pass, not to whichever is asked."""
    client, alice, _bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-orphan", ALICE)

    for pass_id in ("teacher", "student"):
        refused = client.post(
            "/post_feedback",
            headers=alice,
            json=_note(
                turn_key,
                f"claiming the orphan for {pass_id}",
                target_kind="span",
                span_ids=[f"{turn_key}-orphan"],
                pass_id=pass_id,
            ),
        )
        assert refused.status_code == 400, refused.text
        assert f"{turn_key}-orphan" in refused.json()["detail"]


def test_a_spans_own_pass_is_accepted_including_by_inheritance(two_channels):
    """The check must not over-reject: membership is the span TREE, not the stamp.

    `<key>-teacher-child` carries no `fw.pass` of its own and is in the teacher
    pass solely because its parent is — exactly how a producer that opens one
    span per pass records everything that pass did. A membership check that
    compared stamps instead of walking ancestry would refuse this, and would
    make component feedback unusable on every real distillation trace.
    """
    client, alice, _bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-inherited", ALICE)

    written = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            turn_key,
            "the teacher's own step and the one it nested",
            target_kind="span",
            span_ids=[f"{turn_key}-teacher", f"{turn_key}-teacher-child"],
            pass_id="teacher",
        ),
    )
    assert written.status_code == 201, written.text

    row = client.get(
        "/feedback", headers=alice, params={"turn_key": turn_key}
    ).json()["feedback"][-1]
    assert row["anchors"]["primary"]["ref"]["pass_id"] == "teacher"
    assert sorted(row["anchors"]["primary"]["span_ids"]) == [
        f"{turn_key}-teacher",
        f"{turn_key}-teacher-child",
    ]


def test_a_span_that_is_not_on_the_turn_still_fails_as_unrecorded(two_channels):
    """The membership check must not swallow the "not recorded here" refusal.

    A span id the turn does not hold is a different mistake from one that is
    held by the other pass, and the caller needs to be able to tell them apart.
    """
    client, alice, _bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-absent-span", ALICE)

    refused = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            turn_key,
            "a span from nowhere",
            target_kind="span",
            span_ids=["span-that-was-never-recorded"],
            pass_id="teacher",
        ),
    )
    assert refused.status_code == 400, refused.text
    assert "not recorded on turn" in refused.json()["detail"]


def test_component_feedback_without_a_pass_is_unaffected(two_channels):
    """No pass named, nothing to be a member of. The ordinary case still works."""
    client, alice, _bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-absent-passid", ALICE)

    written = client.post(
        "/post_feedback",
        headers=alice,
        json=_note(
            turn_key,
            "a remark on one step of the whole turn",
            target_kind="span",
            span_ids=[f"{turn_key}-student"],
        ),
    )
    assert written.status_code == 201, written.text
    row = client.get(
        "/feedback", headers=alice, params={"turn_key": turn_key}
    ).json()["feedback"][-1]
    assert row["anchors"]["primary"]["ref"]["pass_id"] is None


def test_ownership_is_decided_before_the_pass_is_described(two_channels):
    """A foreign turn must not leak which passes it recorded.

    The pass refusal names the turn's recorded passes, which is useful to an
    owner and is evidence disclosure to anybody else — so the 404 has to come
    first, and a wrong pass on a foreign turn has to look like an unknown key.
    """
    client, _alice, bob, store = two_channels
    turn_key = _seed_two_pass_turn(store, "scope-pass-foreign", ALICE)

    refused = client.post(
        "/post_feedback",
        headers=bob,
        json=_note(turn_key, "probing", pass_id="student"),
    )
    assert refused.status_code == 404, refused.text
    assert refused.json()["detail"] == f"Turn not found: {turn_key}"
    assert "teacher" not in refused.text
