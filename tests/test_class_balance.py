"""Integration tests for the reserved-class budget (R7.2).

Per the repo testing rules these are integration tests against real components: the
real `fastworkflow.train.class_balance` module, a real `RoutingDefinition` and
`CommandContextModel` for the real `tests/todo_list_workflow`. No Mock fixtures and
no patching of fastWorkflow components. Utterance lists that appear inline are test
*inputs*; the system under test is the real selection logic.

R7.3 weighting and its torch-backed tests were removed after the experiment measured
no corrected benefit. Weighting is intentionally not shipped; production keeps the
native unweighted HuggingFace loss path.
"""

import os

import pytest

from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.nlu_labels import WILDCARD_LABEL
from fastworkflow.train.class_balance import (
    coverage_floor_of,
    group_ancestor_utterances,
    reserved_candidate_counts,
    reserved_class_budget,
    select_reserved_rows,
)

# R7.3's optional class-weight schemes, encoded-label tensor ordering, unit-loss
# equivalence, and gradient-scaling checks belonged only to the measured-null branch.
# The fixed R7.2 ratio is production policy rather than environment configuration;
# tests pass it explicitly, and only the escalation class is budgeted.


TODO_LIST_WORKFLOW = os.path.join(os.path.dirname(__file__), "todo_list_workflow")


# ---------------------------------------------------------------------------
# Real-workflow fixtures. No LLM: a command's hand-written `plain_utterances`
# stand in for the generated set, which is exactly the earliest slice of what
# `_get_utterances` returns ([command_name] + seeds + synthetic).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def todo_routing_definition() -> RoutingDefinition:
    return RoutingDefinition.build(TODO_LIST_WORKFLOW)


@pytest.fixture(scope="module")
def todo_command_directory() -> CommandDirectory:
    return CommandDirectory.load(TODO_LIST_WORKFLOW)


@pytest.fixture(scope="module")
def todo_ancestor_cache(todo_routing_definition, todo_command_directory) -> dict:
    """The shape `cache_ancestor_utterances` leaves behind: cache[ctx][cmd] = rows."""
    cache: dict[str, dict[str, list[str]]] = {}
    for context_name in todo_routing_definition.contexts:
        per_command: dict[str, list[str]] = {}
        for command_name in todo_routing_definition.contexts[context_name]:
            todo_command_directory.ensure_command_hydrated(command_name)
            metadata = todo_command_directory.get_utterance_metadata(command_name)
            if metadata is None:
                continue
            rows = [command_name] + list(metadata.plain_utterances or [])
            per_command[command_name] = rows
        if per_command:
            cache[context_name] = per_command
    return cache


# ---------------------------------------------------------------------------
# R7.2 — budget derivation
# ---------------------------------------------------------------------------


def test_budget_is_the_cost_invariant_when_it_exceeds_the_floor():
    """At ratio 1.0 the reserved classes may match, not exceed, the real rows."""
    assert reserved_class_budget(own_row_count=200, coverage_floor=33, ratio=1.0) == 200
    assert reserved_class_budget(own_row_count=200, coverage_floor=33, ratio=0.5) == 100
    assert reserved_class_budget(own_row_count=200, coverage_floor=33, ratio=3.0) == 600


def test_coverage_floor_wins_over_the_cost_budget():
    """Cost is a preference; one row per ancestor command is a requirement."""
    # 33 ancestor intents against a context contributing only 20 real rows: the cost
    # invariant would allow 20, which would silently make 13 ancestor commands
    # unreachable by escalation from here.
    assert reserved_class_budget(own_row_count=20, coverage_floor=33, ratio=1.0) == 33
    assert reserved_class_budget(own_row_count=0, coverage_floor=33, ratio=1.0) == 33


def test_budget_is_never_negative():
    assert reserved_class_budget(own_row_count=-5, coverage_floor=-5, ratio=1.0) == 0


# ---------------------------------------------------------------------------
# R7.2 — round-robin selection (decision D4)
# ---------------------------------------------------------------------------


@pytest.fixture
def lopsided_groups() -> dict[str, dict[str, list[str]]]:
    """One verbose command against several terse ones — the F8 shape, in miniature."""
    return {
        "Deep": {
            "Deep/verbose": [f"verbose-{i}" for i in range(200)],
            "Deep/terse": ["terse-0", "terse-1"],
        },
        "Shallow": {
            "Shallow/only": ["only-0", "only-1", "only-2"],
        },
    }


def test_every_source_survives_even_below_the_floor(lopsided_groups):
    """The coverage pass ignores the budget: a source never loses its last row."""
    selected = select_reserved_rows(lopsided_groups, budget=1)
    # Sorted (group, source) order: Deep/terse, Deep/verbose, then Shallow/only.
    assert selected == ["terse-0", "verbose-0", "only-0"]
    assert len(selected) == coverage_floor_of(lopsided_groups) == 3


def test_verbose_source_cannot_crowd_out_the_others(lopsided_groups):
    """200 rows from one command must not consume a budget of 6."""
    selected = select_reserved_rows(lopsided_groups, budget=6)
    assert len(selected) == 6
    verbose = [row for row in selected if row.startswith("verbose-")]
    # Equal share per source: 2 each, not 4/1/1 and certainly not 6/0/0.
    assert len(verbose) == 2
    assert set(selected) == {
        "verbose-0", "verbose-1", "terse-0", "terse-1", "only-0", "only-1",
    }


def test_selection_is_earliest_first_within_a_source(lopsided_groups):
    """Earliest-first keeps hand-written seeds and discards the synthetic tail."""
    selected = select_reserved_rows(lopsided_groups, budget=12)
    verbose = [row for row in selected if row.startswith("verbose-")]
    assert verbose == [f"verbose-{i}" for i in range(len(verbose))]


def test_exhausted_sources_do_not_stall_the_fill(lopsided_groups):
    """A budget larger than the terse sources must still be spent, on what is left."""
    selected = select_reserved_rows(lopsided_groups, budget=30)
    assert len(selected) == 30
    assert {"terse-0", "terse-1", "only-0", "only-1", "only-2"} <= set(selected)


def test_budget_above_supply_returns_everything(lopsided_groups):
    selected = select_reserved_rows(lopsided_groups, budget=10**6)
    assert len(selected) == 200 + 2 + 3


def test_selection_is_deterministic(lopsided_groups):
    """Two calls, and a call over a differently-ordered dict, must agree."""
    first = select_reserved_rows(lopsided_groups, budget=9)
    second = select_reserved_rows(lopsided_groups, budget=9)
    reordered = {
        "Shallow": lopsided_groups["Shallow"],
        "Deep": {
            "Deep/terse": lopsided_groups["Deep"]["Deep/terse"],
            "Deep/verbose": lopsided_groups["Deep"]["Deep/verbose"],
        },
    }
    third = select_reserved_rows(reordered, budget=9)
    assert first == second == third


def test_excluded_rows_never_enter_the_escalation_class(lopsided_groups):
    """An utterance valid in the local context must not also mean "ask my parent"."""
    selected = select_reserved_rows(
        lopsided_groups, budget=6, exclude={"verbose-0", "only-0"}
    )
    assert "verbose-0" not in selected
    assert "only-0" not in selected
    # Coverage is preserved by moving to the next row, not by dropping the source.
    assert "verbose-1" in selected
    assert "only-1" in selected


def test_always_include_rows_come_first_and_are_never_dropped(lopsided_groups):
    selected = select_reserved_rows(
        lopsided_groups, budget=1, always_include=[WILDCARD_LABEL]
    )
    assert selected[0] == WILDCARD_LABEL
    assert set(selected) >= {"verbose-0", "terse-0", "only-0"}


def test_selection_deduplicates_across_sources():
    """The same utterance under two ancestor commands must train one row, not two."""
    groups = {
        "A": {"A/one": ["shared", "a-1"], "A/two": ["shared", "b-1"]},
    }
    selected = select_reserved_rows(groups, budget=10)
    assert selected.count("shared") == 1
    assert set(selected) == {"shared", "a-1", "b-1"}


def test_candidate_counts_keep_raw_and_deduplicated_denominators_distinct():
    groups = {
        "A": {"A/one": ["shared", "a-1"], "A/two": ["shared", "b-1"]},
        "B": {"B/one": ["local", "b-2"]},
    }

    raw_count, deduplicated_count = reserved_candidate_counts(
        groups, exclude={"local"}
    )

    assert raw_count == 6
    assert deduplicated_count == 4


@pytest.mark.parametrize("groups", [{}, {"A": {}}, {"A": {}, "B": {}}])
def test_candidate_counts_accept_empty_inputs(groups):
    assert reserved_candidate_counts(groups) == (0, 0)


def test_candidate_counts_keep_raw_total_when_every_row_is_excluded():
    groups = {
        "A": {"A/one": ["shared", "a-1"], "A/two": ["shared", "b-1"]},
    }

    raw_count, deduplicated_count = reserved_candidate_counts(
        groups, exclude={"shared", "a-1", "b-1"}
    )

    assert raw_count == 4
    assert deduplicated_count == 0


# ---------------------------------------------------------------------------
# R7.2 — against the real todo_list_workflow hierarchy
# ---------------------------------------------------------------------------


def test_real_ancestor_chain_is_grouped_by_context_and_command(
    todo_routing_definition, todo_ancestor_cache
):
    """`TodoItem` inherits from `TodoList` inherits from `TodoListManager`."""
    ancestors = todo_routing_definition.context_model.get_ancestor_contexts("TodoItem")
    assert "TodoList" in ancestors, f"unexpected ancestor chain: {ancestors}"

    grouped = group_ancestor_utterances(
        ancestors, todo_ancestor_cache, skip_labels={WILDCARD_LABEL}
    )
    assert grouped, "no ancestor utterances grouped for TodoItem"
    assert set(grouped) <= set(ancestors)
    for context_name, by_command in grouped.items():
        assert by_command, f"{context_name} grouped to an empty command map"
        for command_name, rows in by_command.items():
            assert rows, f"{command_name} grouped to zero rows"
            assert command_name.split("/")[-1] != WILDCARD_LABEL


def test_real_hierarchy_keeps_every_ancestor_intent_under_a_tight_budget(
    todo_routing_definition, todo_ancestor_cache
):
    """The coverage claim of R7.2, on a real hierarchy: no intent is dropped."""
    ancestors = todo_routing_definition.context_model.get_ancestor_contexts("TodoItem")
    grouped = group_ancestor_utterances(
        ancestors, todo_ancestor_cache, skip_labels={WILDCARD_LABEL}
    )
    floor = coverage_floor_of(grouped)
    assert floor > 1, "the fixture must offer more than one ancestor command"

    # A budget of 1 is below the floor by construction; every ancestor command must
    # still be represented by at least one row.
    selected = set(select_reserved_rows(grouped, budget=1))
    for by_command in grouped.values():
        for command_name, rows in by_command.items():
            assert selected & set(rows), (
                f"ancestor command {command_name} lost every row and is now "
                "unreachable by escalation from TodoItem"
            )


def test_real_hierarchy_capped_selection_never_exceeds_the_budget(
    todo_routing_definition, todo_ancestor_cache
):
    ancestors = todo_routing_definition.context_model.get_ancestor_contexts("TodoItem")
    grouped = group_ancestor_utterances(
        ancestors, todo_ancestor_cache, skip_labels={WILDCARD_LABEL}
    )
    floor = coverage_floor_of(grouped)
    budget = reserved_class_budget(own_row_count=1000, coverage_floor=floor, ratio=1.0)
    selected = select_reserved_rows(grouped, budget=budget)
    assert floor <= len(selected) <= budget


def test_missing_ancestor_context_is_skipped_not_fatal(todo_ancestor_cache):
    grouped = group_ancestor_utterances(
        ["TodoList", "NoSuchContext"], todo_ancestor_cache
    )
    assert "NoSuchContext" not in grouped


# ---------------------------------------------------------------------------
# The contract the trainer relies on when it calls the above
# ---------------------------------------------------------------------------


def test_escalation_rows_never_collide_with_the_context_own_rows(
    todo_routing_definition, todo_ancestor_cache
):
    """An utterance must never be trained under both a real command and `wildcard`.

    Before R7.2 the trainer built escalation rows from `net_ancestor_utterances`, a set
    difference that had already removed everything the context's own commands claim. The
    budgeted selection reads from the ancestor cache instead, which has NOT had that
    subtraction applied, so the exclusion has to be passed explicitly -- and if it is
    dropped, the same string is emitted once under its real command label and again under
    WILDCARD_LABEL. Nothing downstream rejects that: it trains, and it teaches the
    classifier that a locally-valid utterance means "ask my parent instead".

    This test exists because that is a silent corruption. It reproduces the trainer's own
    call shape rather than testing `exclude` in the abstract.
    """
    ancestors = todo_routing_definition.context_model.get_ancestor_contexts("TodoItem")
    grouped = group_ancestor_utterances(
        ancestors, todo_ancestor_cache, skip_labels={WILDCARD_LABEL}
    )
    assert grouped, "fixture must supply ancestor rows for this to mean anything"

    # Stand in for the rows TodoItem's own commands contributed, by claiming rows that
    # genuinely exist in the ancestor chain -- an overlap the trainer really does see,
    # because a command inherited from a base class appears in both label spaces.
    all_ancestor_rows = sorted(
        {row for by_command in grouped.values() for rows in by_command.values()
         for row in rows}
    )
    context_utterances = set(all_ancestor_rows[: max(1, len(all_ancestor_rows) // 3)])

    selected = select_reserved_rows(
        grouped,
        budget=len(all_ancestor_rows),
        always_include=(),
        exclude=context_utterances,
    )

    collisions = context_utterances & set(selected)
    assert not collisions, (
        "these utterances would be trained under two labels at once: "
        f"{sorted(collisions)[:5]}"
    )

    # And the exclusion must not be achieved by returning nothing at all.
    assert selected, "every escalation row was excluded; the class would vanish"
