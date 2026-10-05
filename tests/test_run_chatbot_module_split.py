"""The chatbot server re-exports helpers that moved out of server.py.

Phase A of fix-hzux.11 moved pure functions into sibling modules and left
``ChatbotServer`` importing those same objects. A re-export that copied the
function, or a name that disappeared, would break callers that still import
from ``fastworkflow.run_chatbot.server``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from fastworkflow.observability import comparison, diagnosis, turn_derivations
from fastworkflow.run_chatbot import (
    http_common,
    provenance,
    selection_api,
    selection_workspace,
    server,
    turn_annotations,
    workflow_discovery,
)

TURN_ANNOTATION_REEXPORTS = (
    "LOW_CONFIDENCE_DEFAULT_MARGIN",
    "TRAINING_RUN_DEFAULT_LIMIT",
    "TRAINING_RUN_MAX_LIMIT",
    "_wire_bool",
    "_wire_number",
    "_workspace_segment_verdicts",
    "annotate_attempt_rows",
    "annotate_projected_attempts",
    "annotate_turn_detail",
    "annotate_turn_diagnosis",
    "annotate_turn_rows",
    "annotate_workspace_attempts",
    "cost_rollup",
    "count_llm_calls_cut_at_limit",
    "diagnostic_store_id",
    "evidence_verdict",
    "execution_ledger",
    "is_low_confidence",
    "llm_call_cost",
    "llm_call_cut_at_limit",
    "merge_cost_rollups",
    "turn_decision_signals",
    "turn_query_from_params",
    "turn_span_stamps",
)

PROVENANCE_REEXPORTS = (
    "benchmark_pin_check",
    "experiment_provenance",
    "provenance_differences",
)

WORKFLOW_DISCOVERY_REEXPORTS = (
    "_looks_like_workflow",
    "_rel_under",
    "_workflow_is_trained",
    "browse_directories",
    "invalidate_workflow_candidate_walk_cache",
    "list_workflow_candidates",
)


def _assert_same(
    names: tuple[str, ...], home: object, facade: object = server
) -> None:
    for name in names:
        assert getattr(facade, name) is getattr(home, name)


# fix-cnoc moved the pure span derivations out of turn_annotations into
# observability/turn_derivations.py (and `cost_rollup` into comparison, beside
# the `usage_rollup` it delegates to); turn_annotations re-exports every one.
TURN_DERIVATION_NAMES = (
    "CONSEQUENCE_ORDER",
    "LOW_CONFIDENCE_DEFAULT_MARGIN",
    "SIGNAL_TOPK_MARGIN",
    "SPAN_AGENT_TOOL_CALL",
    "SPAN_ASK_USER",
    "SPAN_COMMAND_EXECUTE",
    "SPAN_LLM_CALL",
    "SPAN_NLU_INTENT",
    "_exact_int",
    "_finite_number",
    "_mapping_attr",
    "_span_attributes",
    "_text_or_none",
    "count_llm_calls_cut_at_limit",
    "execution_ledger",
    "is_low_confidence",
    "llm_call_cost",
    "llm_call_cut_at_limit",
    "merge_cost_rollups",
    "turn_decision_signals",
)

COMPARISON_TURN_DERIVATION_NAMES = ("cost_rollup",)

DEFAULT_PROJECTION_FUNCTIONS = (
    comparison.default_ledger_projection,
    comparison.default_cost_rollup,
    diagnosis.default_decision_signals,
    diagnosis.default_low_confidence_test,
)


SELECTION_WORKSPACE_REEXPORTS = (
    "_ManifestScopedReader",
    "_WorkspaceNames",
    "_refuse_ambiguous_attempts",
    "_with_turn_refs",
    "_workspace_archive",
    "_workspace_attempts",
    "_workspace_consistency",
    "_workspace_get",
    "_workspace_selected_rows",
    "_workspace_selected_runs",
    "_workspace_side",
    "handle_workspace_get",
)


HTTP_COMMON_REEXPORTS = (
    "STORE_UNAVAILABLE",
    "run_clear_conversations",
)

# Every method of _ChatbotRequestHandler before the phase-B mixin split.
# Pinned literally so a dropped or renamed method fails here even when a
# same-named method on BaseHTTPRequestHandler would still resolve.
PRE_SPLIT_HANDLER_METHODS = (
    "log_message",
    "_redacted_path",
    "_begin_request",
    "_end_request",
    "_report_internal_error",
    "_read_json_body",
    "_send",
    "_send_json",
    "_error",
    "_is_loopback_authority",
    "_host_origin_allowed",
    "_token_valid",
    "_gate",
    "_review_capability",
    "do_GET",
    "_handle_get",
    "_handle_trace_navigation",
    "_refuse_write",
    "_dispatch_write",
    "do_POST",
    "do_PUT",
    "do_DELETE",
    "do_PATCH",
    "_post_feedback_note",
    "_post_benchmark_record",
    "_post_selection",
    "_post_experiment_setup",
    "_post_review_capture",
    "_post_review_assignment",
    "_post_benchmark_version",
    "_post_select_workspace",
    "_post_select_workflow",
    "_post_configure_env",
    "_post_train",
    "_post_clear_conversations",
    "_put_benchmark_analysis",
    "_delete_selection",
    "_delete_benchmark_experiment",
    "_patch_registration",
    "_patch_experiment",
    "_handle_process_log",
    "_handle_api",
    "_training_limit",
    "_handle_training_history",
    "_handle_review_assignment",
    "_handle_review_evidence",
    "_search_turns_response",
    "_handle_workspace",
    "_handle_setup",
    "_benchmark_workflow_path",
    "_handle_selection_api",
    "_handle_navigation",
    "_send_navigation",
    "_handle_feedback_notes",
    "_feedback_writer",
    "_workspace_live",
    "_sealed_evidence",
    "_append_feedback_note",
    "_record_feedback_note",
    "_feedback_source_for",
    "_pass_selector",
    "_handle_workspace_task_feedback",
    "_handle_task_feedback",
    "_handle_benchmark_registration",
    "_handle_registration_patch",
    "_benchmark_experiments",
    "_handle_benchmarks",
    "_handle_benchmark_post",
    "_analysis_payload_from_body",
    "_handle_benchmark_analysis_put",
    "_handle_experiments",
    "_handle_experiment_patch",
    "_serve_artifact",
    "_float_or_none",
    "_int",
)

# Mixin classes, in MRO order, each ahead of BaseHTTPRequestHandler.
HANDLER_MIXIN_NAMES = (
    "_ReviewRoutes",
    "_WorkspaceRoutes",
    "_ExperimentRoutes",
    "_BenchmarkRoutes",
    "_FeedbackRoutes",
    "_NavigationRoutes",
    "_ControlPlaneRoutes",
)

MIXIN_MODULES = (
    "handler_benchmark.py",
    "handler_control.py",
    "handler_experiment.py",
    "handler_feedback.py",
    "handler_navigation.py",
    "handler_review.py",
    "handler_workspace.py",
)


def _handler_mixins() -> tuple[type, ...]:
    by_name = {cls.__name__: cls for cls in server._ChatbotRequestHandler.__mro__}
    return tuple(by_name[name] for name in HANDLER_MIXIN_NAMES)


def _defined_callables(cls: type) -> dict[str, object]:
    found = {}
    for name, value in cls.__dict__.items():
        if name.startswith("__"):
            continue
        if isinstance(value, (staticmethod, classmethod)) or callable(value):
            found[name] = value
    return found


def test_server_reexports_are_the_moved_objects():
    _assert_same(TURN_ANNOTATION_REEXPORTS, turn_annotations)
    _assert_same(PROVENANCE_REEXPORTS, provenance)
    _assert_same(WORKFLOW_DISCOVERY_REEXPORTS, workflow_discovery)
    _assert_same(HTTP_COMMON_REEXPORTS, http_common)


def test_turn_derivations_have_one_canonical_home_and_old_paths_are_the_same_objects():
    for name in TURN_DERIVATION_NAMES:
        value = getattr(turn_derivations, name)
        if callable(value):
            assert value.__module__ == turn_derivations.__name__, name
    assert comparison.cost_rollup.__module__ == comparison.__name__
    _assert_same(TURN_DERIVATION_NAMES, turn_derivations, turn_annotations)
    _assert_same(COMPARISON_TURN_DERIVATION_NAMES, comparison, turn_annotations)
    via_server = tuple(n for n in TURN_ANNOTATION_REEXPORTS if n in TURN_DERIVATION_NAMES)
    _assert_same(via_server, turn_derivations)
    _assert_same(COMPARISON_TURN_DERIVATION_NAMES, comparison)


def test_default_projections_return_the_canonical_functions():
    assert comparison.default_ledger_projection() is turn_derivations.execution_ledger
    assert comparison.default_cost_rollup() is comparison.cost_rollup
    assert diagnosis.default_decision_signals() is turn_derivations.turn_decision_signals
    assert diagnosis.default_low_confidence_test() is turn_derivations.is_low_confidence


def test_default_projections_import_nothing_at_call_time():
    """The old defaults imported from the debug-UI package inside the function
    to dodge an import cycle; the layering now makes that unnecessary."""
    for function in DEFAULT_PROJECTION_FUNCTIONS:
        tree = ast.parse(inspect.getsource(function))
        nested = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert nested == [], function.__qualname__


def test_turn_derivations_import_nothing_from_fastworkflow():
    """Stdlib-only is what keeps it below comparison and diagnosis."""
    tree = ast.parse(Path(turn_derivations.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("fastworkflow"), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("fastworkflow"), alias.name


def test_pre_split_handler_methods_resolve_on_the_composed_class():
    """Phase B moved handler methods onto mixins; the composed class still has each one."""
    owners = {server._ChatbotRequestHandler.__name__, *HANDLER_MIXIN_NAMES}
    for name in PRE_SPLIT_HANDLER_METHODS:
        attr = getattr(server._ChatbotRequestHandler, name)
        func = getattr(attr, "__func__", attr)
        owner = func.__qualname__.split(".", 1)[0]
        assert owner in owners, name


def test_mixin_method_names_are_unique_and_mixins_define_no_init():
    seen: dict[str, str] = {}
    for mixin in _handler_mixins():
        assert "__init__" not in mixin.__dict__
        for name in _defined_callables(mixin):
            assert name not in seen, f"{name} defined on {seen[name]} and {mixin.__name__}"
            seen[name] = mixin.__name__
    overlap = set(_defined_callables(server._ChatbotRequestHandler)) & set(seen)
    assert overlap == set()


def test_handler_mro_places_mixins_before_base_http_request_handler():
    names = [cls.__name__ for cls in server._ChatbotRequestHandler.__mro__]
    assert names[: 1 + len(HANDLER_MIXIN_NAMES)] == [
        "_ChatbotRequestHandler",
        *HANDLER_MIXIN_NAMES,
    ]
    assert names[1 + len(HANDLER_MIXIN_NAMES)] == "BaseHTTPRequestHandler"


def test_mixin_modules_do_not_import_server():
    """Mixins must not import server.py; that import would be a cycle."""
    package = Path(server.__file__).resolve().parent
    for name in (*MIXIN_MODULES, "http_common.py"):
        tree = ast.parse((package / name).read_text())
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert module != "fastworkflow.run_chatbot.server", name
                assert not module.endswith(".run_chatbot.server"), name


def test_selection_api_reexports_are_the_moved_workspace_objects():
    """Workspace archive GETs moved out of selection_api and stay the same objects."""
    _assert_same(
        SELECTION_WORKSPACE_REEXPORTS, selection_workspace, selection_api
    )


def test_selection_workspace_does_not_import_server():
    """selection_workspace must not import server.py; that import would be a cycle."""
    package = Path(selection_workspace.__file__).resolve().parent
    tree = ast.parse((package / "selection_workspace.py").read_text())
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        for module in modules:
            assert module != "fastworkflow.run_chatbot.server"
            assert not module.endswith(".run_chatbot.server")
