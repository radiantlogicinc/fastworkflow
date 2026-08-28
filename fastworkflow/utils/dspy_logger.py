"""
DSPy Logging Utilities
Author: Dhar Rawal

Works with DSPy to log forward calls and their results, using a custom handler function.
Works with typed predictors too!
"""
import functools
import json
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Optional, Tuple

import dspy
from dspy.utils.callback import BaseCallback
from pydantic import BaseModel

from fastworkflow import tracing


CAPTURE_MODULE_DECORATOR = "module_decorator"
CAPTURE_DSPY_API = "dspy_api"

_active_observability_host: ContextVar[Any] = ContextVar(
    "fastworkflow_dspy_observability_host", default=None
)
_module_frames: ContextVar[Optional[list]] = ContextVar(
    "fastworkflow_dspy_module_frames", default=None
)


class _ModuleFrame:
    """One DSPy module invocation and the LLM spans opened inside it.

    Frames are what make the drill-down structured rather than raw: the LM
    callback sees only the wire request and the completion text, while the
    module's parsed ``Prediction`` — where ChainOfThought's ``reasoning``
    field actually lives — is only visible when the module returns.
    """

    __slots__ = ("name", "source", "inputs", "call_id", "spans")

    def __init__(
        self,
        name: str,
        source: str,
        inputs: Any = None,
        call_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self.source = source
        self.inputs = inputs
        self.call_id = call_id
        self.spans: list[tuple[Any, Any]] = []


def _push_module_frame(frame: _ModuleFrame) -> Optional[_ModuleFrame]:
    """Enter a module frame, or no-op when no turn is being observed."""
    frames = _module_frames.get()
    if frames is None:
        return None
    frames.append(frame)
    return frame


def _close_module_frame(
    frame: Optional[_ModuleFrame],
    outputs: Any,
    exception: Optional[BaseException] = None,
) -> None:
    """Leave a module frame, back-filling its LLM spans with the module result."""
    if frame is None:
        return
    frames = _module_frames.get()
    if frames and frame in frames:
        del frames[frames.index(frame):]
    if not frame.spans:
        return

    attributes: dict[str, Any] = {}
    if outputs is not None:
        attributes["module_output"] = _json_text(outputs)
        if reasoning := _reasoning_from(outputs):
            attributes["reasoning"] = _json_text(reasoning)
    if exception is not None:
        attributes["module_exception"] = repr(exception)
    if not attributes:
        return

    for host, span in frame.spans:
        # The span already carries its end time; re-emitting it is the
        # store's idempotent upsert, not a second span.
        merged = dict(attributes)
        if "reasoning" in span.attributes:
            merged.pop("reasoning", None)
        tracing.end_span(host, span, status=span.status, attributes=merged, close=False)


def _module_chain(frames: list) -> str:
    """The enclosing module path, without the repetition that comes from
    seeing the same module twice — once via DSPy's ``__call__`` callback and
    once via the decorator on its ``forward``."""
    names: list[str] = []
    for frame in frames:
        if not names or names[-1] != frame.name:
            names.append(frame.name)
    return " > ".join(names)


@contextmanager
def observe_dspy_host(host: Any):
    """Bind DSPy calls in this execution context to a fastWorkflow trace host.

    The trace callback is registered through ``dspy.context`` rather than on
    individual LM instances so that every DSPy call in the turn is covered —
    including module-level events, which DSPy only dispatches to callbacks
    registered in settings.

    With no live sink (FW_OBSERVABILITY=0, or the no-op default) this binds
    nothing: the callback would otherwise JSON-project every prompt on every
    LM call before discovering there is nowhere to send it — a per-call cost
    the disabled path must not pay. One cheap check per turn instead.
    """
    if tracing.get_sink(host) is None:
        yield
        return
    host_token = _active_observability_host.set(host)
    frames_token = _module_frames.set([])
    callback = observability_callback()
    active = list(dspy.settings.get("callbacks", None) or [])
    try:
        if callback in active:
            yield
        else:
            with dspy.context(callbacks=active + [callback]):
                yield
    finally:
        _module_frames.reset(frames_token)
        _active_observability_host.reset(host_token)


def observe_dspy_calls(func: Callable) -> Callable:
    """Decorator that binds a host method's DSPy calls to its active turn."""

    @functools.wraps(func)
    def wrapper(host: Any, *args: Any, **kwargs: Any) -> Any:
        with observe_dspy_host(host):
            return func(host, *args, **kwargs)

    return wrapper


class DSPyProgramLog(BaseModel):
    """DSPy Program Log"""

    dspy_program_class: str
    dspy_input_args: Tuple[Any, ...] = ()
    dspy_input_kwargs: Dict[str, Any] = {}
    dspy_completions_dict: Dict[str, Any] = {}
    # dspy_module_logs: list[DSPyModuleLog] = []


class DSPyForward:  # pylint: disable=too-few-public-methods
    """DSPy Forward Interceptor"""

    # class variable for custom handler
    save_dspyprogramlog_func: Optional[Callable[[DSPyProgramLog], None]] = None

    @classmethod
    def intercept(cls, func: Callable) -> Callable:
        """
        Decorator to log forward calls and their results, using a custom handler function.
        Using __call__(...) enables the class itself to be used as a decorator.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            dspy_program_log: DSPyProgramLog = DSPyProgramLog(dspy_program_class=func.__qualname__.split(".")[-2])
            dspy_program_log.dspy_input_args = args[1:] if args else ()
            dspy_program_log.dspy_input_kwargs = kwargs

            frame = _push_module_frame(
                _ModuleFrame(
                    name=dspy_program_log.dspy_program_class,
                    source=CAPTURE_MODULE_DECORATOR,
                    inputs={
                        "args": dspy_program_log.dspy_input_args,
                        "kwargs": dspy_program_log.dspy_input_kwargs,
                    },
                )
            )
            result: Optional[dspy.Prediction] = None
            failure: Optional[BaseException] = None
            try:
                result = func(*args, **kwargs)

                completions = getattr(result, "completions", None)
                if completions:
                    dspy_program_log.dspy_completions_dict = (
                        completions._completions  # pylint: disable=protected-access
                    )
                else:
                    dspy_program_log.dspy_completions_dict = {}

                if DSPyForward.save_dspyprogramlog_func:
                    DSPyForward.save_dspyprogramlog_func(  # pylint: disable=abstract-class-instantiated, not-callable
                        dspy_program_log
                    )

                return result
            except BaseException as exc:
                failure = exc
                raise
            finally:
                _close_module_frame(frame, result, failure)

        return wrapper


def _is_secret_key(key: str) -> bool:
    """Whether a payload key names a credential.

    ``*_tokens`` is deliberately excluded: ``prompt_tokens`` and its siblings
    are usage counts, and redacting them costs the developer the cost signal
    while protecting nothing.
    """
    upper = str(key).upper()
    if any(marker in upper for marker in ("API_KEY", "SECRET", "PASSWORD")):
        return True
    return upper == "TOKEN" or (upper.endswith("_TOKEN") and not upper.endswith("_TOKENS"))


def _plain_value(value: Any, depth: int = 0) -> Any:
    """Best-effort JSON-safe projection for DSPy callback payloads."""
    if depth > 8:
        return {"truncated": True, "type": type(value).__name__}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _is_secret_key(key) else _plain_value(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item, depth + 1) for item in value]
    # dspy.Prediction/Example project through toDict(); without it they fall
    # through to vars() and surface as an unreadable {"_store": ...} envelope.
    for projector in ("toDict", "to_dict"):
        method = getattr(value, projector, None)
        if callable(method):
            try:
                return _plain_value(method(), depth + 1)
            except Exception:
                pass
    if hasattr(value, "model_dump"):
        try:
            return _plain_value(value.model_dump(mode="python"), depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _plain_value(vars(value), depth + 1)
        except Exception:
            pass
    return repr(value)


def _json_text(value: Any) -> str:
    return json.dumps(_plain_value(value), ensure_ascii=False, default=repr)


_REASONING_KEYS = (
    "reasoning",  # ChainOfThought's output field
    "reasoning_content",  # provider-native extended thinking
    "thinking",
    "next_thought",  # fastWorkflowReAct's per-step thought
)


def _reasoning_from(value: Any) -> Any:
    """Collect stated reasoning fields without inventing reasoning."""
    value = _plain_value(value)
    found = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in _REASONING_KEYS and child:
                    found.append(child)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    if not found:
        return None
    return found[0] if len(found) == 1 else found


class DSPyObservabilityCallback(BaseCallback):
    """Emit one ``fw.llm.call`` span per DSPy LM invocation.

    Identity comes from module logging when a ``DSPyForward.intercept``-decorated
    module is on the stack, and from DSPy's own module callbacks otherwise, so
    built-in Predict/ChainOfThought modules remain fully inspectable.

    The request and completion are read from the callback payloads rather than
    from ``lm.history``: run_fastapi_mcp runs with ``disable_history=True``
    (a memory bound, see run_fastapi_mcp/server_memory.py), which leaves that
    history permanently empty in the process the chatbot actually traces.
    History is still harvested opportunistically for usage and cost, which the
    callback payloads do not carry.
    """

    def __init__(self) -> None:
        self._calls: dict[str, tuple[Any, Any, Any, int]] = {}
        self._calls_lock = threading.Lock()

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        if _active_observability_host.get() is None:
            return
        _push_module_frame(
            _ModuleFrame(
                name=type(instance).__qualname__,
                source=CAPTURE_DSPY_API,
                inputs=inputs,
                call_id=call_id,
            )
        )

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        frames = _module_frames.get()
        if not frames:
            return
        frame = next(
            (item for item in reversed(frames) if item.call_id == call_id), None
        )
        _close_module_frame(frame, outputs, exception)

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        host = _active_observability_host.get()
        if host is None:
            return
        with self._calls_lock:
            if call_id in self._calls:
                return  # registered both in settings and on the LM instance

        frames = list(_module_frames.get() or [])
        innermost = frames[-1] if frames else None
        # Module logging first, DSPy's own module identity as the fallback.
        identity = next(
            (item for item in reversed(frames) if item.source == CAPTURE_MODULE_DECORATOR),
            innermost,
        )
        payload = dict(inputs or {})
        messages = payload.pop("messages", None)
        prompt = payload.pop("prompt", None)
        payload.pop("items", None)
        payload.pop("request", None)

        attributes: dict[str, Any] = {
            "module": identity.name if identity else type(instance).__qualname__,
            "capture_source": identity.source if identity else CAPTURE_DSPY_API,
            "model": getattr(instance, "model", None),
            "messages": _json_text(messages),
        }
        if prompt is not None:
            attributes["prompt"] = _json_text(prompt)
        if payload:
            attributes["call_kwargs"] = _json_text(payload)
        if chain := _module_chain(frames):
            attributes["module_chain"] = chain
        if identity is not None and identity.inputs:
            attributes["module_input"] = _json_text(identity.inputs)

        span = tracing.start_span(
            host,
            tracing.SPAN_LLM_CALL,
            kind=tracing.KIND_LLM,
            attributes=attributes,
        )
        if span is None:
            return
        if innermost is not None:
            innermost.spans.append((host, span))
        # Remember the entry that was newest BEFORE this call, by identity.
        # Length growth cannot answer "did a new entry land": a bounded history
        # (the chatbot's server caps it at one entry per LM) trims as it
        # appends, so the length is constant and every call after the first
        # would look like it produced nothing.
        history = getattr(instance, "history", None) or []
        previous_entry = history[-1] if history else None
        with self._calls_lock:
            self._calls[call_id] = (host, span, instance, previous_entry)

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        with self._calls_lock:
            call = self._calls.pop(call_id, None)
        if call is None:
            return

        host, span, instance, previous_entry = call
        history = list(getattr(instance, "history", None) or [])
        newest = history[-1] if history else None
        entry = newest if newest is not None and newest is not previous_entry else None
        attributes: dict[str, Any] = {"output": _json_text(outputs)}
        if isinstance(entry, dict):
            attributes.update(
                {
                    "usage": _json_text(entry.get("usage")),
                    "cost": entry.get("cost"),
                    "history_uuid": entry.get("uuid"),
                    "response_model": entry.get("response_model"),
                    "cache_hit": bool(
                        getattr(entry.get("response"), "cache_hit", False)
                    ),
                    "provider_response": _json_text(entry.get("response")),
                }
            )
        elif entry is not None:
            attributes["provider_response"] = _json_text(entry)
        else:
            attributes["usage_capture"] = (
                "unavailable: DSPy history is disabled in this process"
            )
        response = entry.get("response") if isinstance(entry, dict) else None
        reasoning = _reasoning_from([outputs, response])
        if reasoning is not None:
            attributes["reasoning"] = _json_text(reasoning)
        if exception is not None:
            attributes["exception"] = repr(exception)
        tracing.end_span(
            host,
            span,
            status=tracing.STATUS_ERROR if exception is not None else tracing.STATUS_OK,
            attributes=attributes,
        )


_OBSERVABILITY_CALLBACK = DSPyObservabilityCallback()


def observability_callback() -> DSPyObservabilityCallback:
    """The process-wide stateless DSPy callback used by fastWorkflow LMs."""
    return _OBSERVABILITY_CALLBACK


class DSPyLogger:
    """DSPy Logger"""

    def __init__(self):
        pass

    def __enter__(self) -> "DSPyLogger":
        DSPyForward.save_dspyprogramlog_func = self
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        DSPyForward.save_dspyprogramlog_func = None

    def __call__(dspy_program_log: DSPyProgramLog) -> None:
        """Default handler to save the dspy program log"""
        # args_str = ', '.join([repr(a) for a in dspy_program_log.dspy_input_args] +
        #                      [f"{k}={v!r}" for k, v in dspy_program_log.dspy_input_kwargs.items()])
        # print(args_str)
        print(f"{dspy_program_log.dspy_program_class}")
        print(f"{dspy_program_log.dspy_input_args}")
        print(f"{dspy_program_log.dspy_input_kwargs}")
        print(f"{json.dumps(dspy_program_log.dspy_completions_dict)}")


class DSPyRotatingFileLogger(DSPyLogger):
    """DSPy Rotating File Logger Singleton with Asynchronous Writes"""
    # configurable parameters
    max_file_size = 1024 * 1024  # 1MB
    backup_count = 5

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_file_path: str):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super().__new__(cls)
                    cls._instance.__init__(log_file_path)
        return cls._instance

    def __init__(self, log_file_path: str):
        if hasattr(self, 'logger'):  # Prevent re-initialization
            return

        # Create a logger
        self.logger = logging.getLogger("dspy_log")
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers to prevent console output
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create a rotating file handler
        self.handler = RotatingFileHandler(log_file_path, 
                                           maxBytes=DSPyRotatingFileLogger.max_file_size, 
                                           backupCount=DSPyRotatingFileLogger.backup_count)
        self.handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(self.handler)

        # Prevent propagation to the root logger
        self.logger.propagate = False

    def __call__(self, dspy_program_log: DSPyProgramLog) -> None:
        """Log the dspy program log asynchronously"""
        log_message = dspy_program_log.model_dump_json()
        threading.Thread(target=self._log_to_file, args=(log_message,)).start()

    def _log_to_file(self, log_message: str) -> None:
        """Write log message to file"""
        self.logger.info(log_message)

def _how_to_use():
    """
    how to use:
    Use @DSPyForward.intercept to decorate the forward function of your dspy program
    Call the forward function in the context of the DSPyLogger or DSPyRotatingFileLogger (if you want to log to a rotating file)
    The dspy logger object will get called with an instance of DSPyProgramLog
    """

    class BasicQA(dspy.Module):
        """DSPy Module for testing DSPyLogger"""

        def __init__(self):
            super().__init__()

            self.generate_answer = dspy.Predict("topic, question -> answer")

        gpt3_turbo = dspy.LM(model="openai/gpt-3.5-turbo", api_key="<YOUR_API_KEY>")

        @DSPyForward.intercept
        def forward(self, topic, question):
            """forward pass"""
            with dspy.context(lm=BasicQA.gpt3_turbo):
                return self.generate_answer(topic=topic, question=question)

    get_answer = BasicQA()
    # If you just want to log to the console, use DSPyLogger
    with DSPyLogger:
        _ = get_answer("geography quiz", question="What is the capital of France?")

    # If you want to log to a file, use DSPyFileLogger
    with DSPyRotatingFileLogger("dspy_logs.jsonl"):
        _ = get_answer("geography quiz", question="What is the capital of France?")

    # This will print:
    # BasicQA
    # ('geography quiz',)
    # {'question': 'What is the capital of France?'}
    # {"answer": ["Topic: geography quiz\nQuestion: What is the capital of France?\nAnswer: Paris"]}


if __name__ == "__main__":
    _how_to_use()
