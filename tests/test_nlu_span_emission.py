"""v2 NLU spans + structured diagnosis attributes (bead fix-kw7.15, D3 as amended).

Real components throughout (testing_rules.mdc): the trained
fastworkflow/examples/hello_world workflow drives the REAL intent-detection
matching layers (exact prefix, fuzzy pre-match, TinyBERT/DistilBERT
classifier) and the real XML-regex parameter extraction; messaging_app_4's
``set_current_user`` drives the real ``db_lookup`` fuzzy matcher. The only
test-authored piece is RecordingTraceSink — a real implementation of the
TraceSink protocol (the pluggable seam the design defines).

No LLM is called anywhere here: the classifier path is local models, the
extraction path is the agent-mode XML regex, and db_lookup/validation run at
the validate_parameters level.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


HELLO_WORLD = str(
    Path(__file__).parent.parent / "fastworkflow" / "examples" / "hello_world"
)
MESSAGING_APP = str(
    Path(__file__).parent.parent / "fastworkflow" / "examples" / "messaging_app_4"
)


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    env = dotenv_values("fastworkflow/examples/fastworkflow.env")
    fastworkflow.init(dict(env))
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> None:
        pass

    def record_conversation_label(self, *args) -> None:
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [s for s in self.spans if s.name == name]


@pytest.fixture
def hello_ctx():
    if not Path(HELLO_WORLD, "___command_info").is_dir():
        pytest.skip("hello_world is not trained on this machine")
    sink = RecordingTraceSink()
    wf = fastworkflow.Workflow.create(
        HELLO_WORLD,
        workflow_id_str=f"nluspan-{uuid.uuid4().hex}",
        workflow_context={"run_as_agent": True},
    )
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    ctx.bind_app_workflow(wf)
    ctx._begin_turn("nlu span test")
    ctx.push_active_workflow(wf)
    yield ctx, sink
    ctx.pop_active_workflow()


class TestIntentSpan:
    def test_exact_prefix_layer(self, hello_ctx):
        ctx, sink = hello_ctx
        out = CommandExecutor.invoke_command(
            ctx,
            "add_two_numbers <first_num>5</first_num><second_num>3</second_num>",
        )
        assert out.success
        span = sink.named(tracing.SPAN_NLU_INTENT)[-1]
        assert span.status == "ok"
        assert span.attributes["matcher_layer"] == "exact_prefix"
        assert span.attributes["command_name"] == "add_two_numbers"
        assert span.attributes["resolved"] is True
        assert span.attributes["ambiguous"] is False
        assert span.attributes["stage"] == "INTENT_DETECTION"
        # Nested under the enclosing fw.command.execute span via the stack.
        assert span.parent_span_id is not None

    def test_fuzzy_prematch_layer(self, hello_ctx):
        ctx, sink = hello_ctx
        CommandExecutor.invoke_command(ctx, "what can i do")
        span = sink.named(tracing.SPAN_NLU_INTENT)[-1]
        assert span.attributes["matcher_layer"] == "fuzzy_prematch"
        assert span.attributes["command_name"] == "IntentDetection/what_can_i_do"
        assert span.attributes["is_cme_command"] is True

    def test_classifier_layer_carries_confidence_and_threshold(self, hello_ctx):
        ctx, sink = hello_ctx
        # A phrasing no prefix/fuzzy layer claims: the classifier must decide,
        # and its numbers must land on the span whether or not it is confident.
        CommandExecutor.invoke_command(
            ctx, "please compute the total of five plus three"
        )
        spans = [
            s
            for s in sink.named(tracing.SPAN_NLU_INTENT)
            if s.attributes.get("matcher_layer") == "classifier"
        ]
        assert spans, "classifier layer never ran"
        classifier = spans[0].attributes["classifier"]
        assert 0.0 <= classifier["confidence"] <= 1.0
        assert 0.0 <= classifier["ambiguous_threshold"] <= 1.0
        assert classifier["model_tier"] in ("tiny", "large")
        assert isinstance(classifier["topk_labels"], list)
        assert classifier["confident"] == (
            classifier["confidence"] > classifier["ambiguous_threshold"]
        )
        # An unconfident prediction is an ambiguity with a candidate set.
        if not classifier["confident"]:
            assert spans[0].attributes["ambiguous"] is True
            assert spans[0].attributes["candidates"] == classifier["topk_labels"]


class TestParamExtractionSpan:
    def test_xml_extraction_valid(self, hello_ctx):
        ctx, sink = hello_ctx
        CommandExecutor.invoke_command(
            ctx,
            "add_two_numbers <first_num>5</first_num><second_num>3</second_num>",
        )
        span = sink.named(tracing.SPAN_NLU_PARAM_EXTRACTION)[-1]
        assert span.status == "ok"
        assert span.command_name == "add_two_numbers"
        assert span.attributes["extraction_method"] == "xml_regex"
        assert span.attributes["retry_round"] is False
        assert span.attributes["parameters_valid"] is True
        assert span.attributes["validation_hook"]["ran"] is True
        assert span.attributes["validation_hook"]["is_valid"] is True


class TestStructuredValidationDiagnostics:
    """validate_parameters(diagnostics=...) fills structured outcomes —
    exercised at the validation seam with REAL command classes so no LLM or
    trained model is needed."""

    def _validator_for(self, workflow, command_name):
        from fastworkflow.utils.signatures import InputForParamExtraction

        return InputForParamExtraction.create(workflow, command_name, "")

    def test_missing_fields_are_structured(self):
        from fastworkflow.command_routing import RoutingRegistry

        wf = fastworkflow.Workflow.create(
            HELLO_WORLD, workflow_id_str=f"nluspan-miss-{uuid.uuid4().hex}"
        )
        routing = RoutingRegistry.get_definition(wf.folderpath)
        params_class = routing.get_command_class(
            "add_two_numbers", fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
        )
        params = params_class.model_construct(first_num=5.0, second_num="NOT_FOUND")
        validator = self._validator_for(wf, "add_two_numbers")
        diagnostics: dict = {}
        is_valid, _msg, _sugg, _fields = validator.validate_parameters(
            wf, "add_two_numbers", params, diagnostics=diagnostics
        )
        assert is_valid is False
        assert diagnostics["missing_fields"] == ["second_num"]
        assert diagnostics["invalid_fields"] == []

    def test_db_lookup_events_are_structured(self):
        from fastworkflow.command_routing import RoutingRegistry

        wf = fastworkflow.Workflow.create(
            MESSAGING_APP, workflow_id_str=f"nluspan-db-{uuid.uuid4().hex}"
        )
        routing = RoutingRegistry.get_definition(wf.folderpath)
        # The app's own startup action: attach the root ChatRoom context.
        set_root = routing.get_command_class(
            "set_root_context", fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE
        )()
        set_root(wf, "set root context")
        add_user = routing.get_command_class(
            "ChatRoom/add_user", fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE
        )()
        add_params = routing.get_command_class(
            "ChatRoom/add_user", fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
        )
        wf.command_context_for_response_generation = wf.root_command_context
        add_user(wf, "add Alice", add_params(user_name="Alice", is_premium_user=False))
        add_user(wf, "add Bob", add_params(user_name="Bob Smith", is_premium_user=True))

        params_class = routing.get_command_class(
            "ChatRoom/set_current_user", fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
        )
        validator = self._validator_for(wf, "ChatRoom/set_current_user")

        # One-edit typo on a long-enough name: the real fuzzy matcher
        # auto-applies the correction, and the event records the rewrite.
        diagnostics: dict = {}
        params = params_class.model_construct(user_name="Bob Smyth")
        is_valid, _m, _s, _f = validator.validate_parameters(
            wf, "ChatRoom/set_current_user", params, diagnostics=diagnostics
        )
        assert is_valid is True
        (event,) = diagnostics["db_lookup"]
        assert event["field"] == "user_name"
        assert event["input_value"] == "Bob Smyth"
        assert event["outcome"] == "applied"
        assert event["corrected_value"] == "Bob Smith"
        assert event["corrected"] is True

        # A far-off value: rejected with structured suggestions.
        diagnostics = {}
        params = params_class.model_construct(user_name="Alicia")
        is_valid, _m, _s, _f = validator.validate_parameters(
            wf, "ChatRoom/set_current_user", params, diagnostics=diagnostics
        )
        (event,) = diagnostics["db_lookup"]
        if event["outcome"] == "rejected":
            assert is_valid is False
            assert event["suggestions"]
            assert diagnostics["invalid_fields"] == ["user_name"]
        else:
            # The matcher considered it close enough to apply — still a
            # structured, truthful record.
            assert event["outcome"] == "applied"

    def test_diagnostics_none_changes_nothing(self):
        from fastworkflow.command_routing import RoutingRegistry

        wf = fastworkflow.Workflow.create(
            HELLO_WORLD, workflow_id_str=f"nluspan-none-{uuid.uuid4().hex}"
        )
        routing = RoutingRegistry.get_definition(wf.folderpath)
        params_class = routing.get_command_class(
            "add_two_numbers", fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
        )
        params = params_class.model_construct(first_num=1.0, second_num=2.0)
        validator = self._validator_for(wf, "add_two_numbers")
        is_valid, _m, _s, _f = validator.validate_parameters(
            wf, "add_two_numbers", params
        )
        assert is_valid is True


class TestContextMutations:
    def test_root_span_records_context_diff(self):
        sink = RecordingTraceSink()
        wf = fastworkflow.Workflow.create(
            str(Path(__file__).parent / "todo_list_workflow"),
            workflow_id_str=f"nluspan-ctx-{uuid.uuid4().hex}",
            workflow_context={"kept": "same", "to_change": "before", "to_remove": 1},
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
        ctx.bind_app_workflow(wf)
        ctx._begin_turn("mutate context")

        workflow_context = wf.context
        workflow_context["added_key"] = {"answer": 42}
        workflow_context["to_change"] = "after"
        del workflow_context["to_remove"]
        wf.context = workflow_context

        ctx.finalize_turn_for_observability(
            fastworkflow.CommandOutput(
                command_name="noop",
                command_response=fastworkflow.CommandResponse(response="done"),
            )
        )
        closes = [
            s for s in sink.named(tracing.SPAN_TURN) if s.end_ns is not None
        ]
        assert closes, "root span never closed"
        mutations = closes[-1].attributes["context_mutations"]
        assert "added_key" in mutations["added"]
        assert mutations["changed"]["to_change"] == {
            "from": "'before'",
            "to": "'after'",
        }
        assert mutations["removed"] == ["to_remove"]
        assert "kept" not in mutations.get("changed", {})

    def test_no_mutations_yields_none(self):
        sink = RecordingTraceSink()
        wf = fastworkflow.Workflow.create(
            str(Path(__file__).parent / "todo_list_workflow"),
            workflow_id_str=f"nluspan-ctx2-{uuid.uuid4().hex}",
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
        ctx.bind_app_workflow(wf)
        ctx._begin_turn("no mutation")
        ctx.finalize_turn_for_observability(
            fastworkflow.CommandOutput(
                command_name="noop",
                command_response=fastworkflow.CommandResponse(response="done"),
            )
        )
        closes = [s for s in sink.named(tracing.SPAN_TURN) if s.end_ns is not None]
        assert closes[-1].attributes["context_mutations"] is None


class TestHostScope:
    def test_no_host_means_no_emission_and_no_failure(self):
        # Deep sites outside any host_scope must silently no-op.
        assert tracing.current_host() is None
        span = tracing.start_span(
            tracing.current_host(), tracing.SPAN_NLU_INTENT, attributes={"x": 1}
        )
        assert span is None
        tracing.end_span(None, span)  # no-op, no raise

    def test_scope_binds_and_restores(self):
        sentinel = object()
        with tracing.host_scope(sentinel):
            assert tracing.current_host() is sentinel
            with tracing.host_scope(None):
                assert tracing.current_host() is None
            assert tracing.current_host() is sentinel
        assert tracing.current_host() is None
