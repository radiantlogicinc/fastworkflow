from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ProbabilisticConfig:
    max_retries: int = 2
    score_threshold: Optional[float] = None


def _get_field_post_condition(field) -> Optional[str]:
    # Pydantic v2 stores extra in json_schema_extra; be defensive
    try:
        extra = getattr(field, "json_schema_extra", None) or {}
        return extra.get("post_conditions")
    except Exception:
        return None


def validate_post_conditions(input_obj: Any, output_obj: Any) -> tuple[bool, Optional[str]]:
    """
    Validate post-conditions defined on the Pydantic Output model's fields.

    A field can specify a string expression under json_schema_extra["post_conditions"].
    The expression is evaluated with variables: input, output, value (the field value).

    Returns (True, None) if all pass, otherwise (False, message).
    """
    model_fields = getattr(output_obj.__class__, "model_fields", {})
    for name, field in model_fields.items():
        expr = _get_field_post_condition(field)
        if not expr:
            continue
        value = getattr(output_obj, name, None)
        # Very small, sandboxed eval context
        context = {"input": input_obj, "output": output_obj, "value": value}
        try:
            ok = bool(eval(expr, {"__builtins__": {}}, context))  # noqa: S307
        except Exception as e:  # expression error -> fail fast
            return False, f"Post-condition for field '{name}' raised error: {e}"
        if not ok:
            return False, f"Post-condition failed for field '{name}': {expr}"
    return True, None


def score_output(scoring_func: Optional[Callable[[Any], float] | str], output_obj: Any) -> Optional[float]:
    """
    Minimal scoring: if scoring_func is a callable, call it; if it's a string,
    this function returns None (reserved for future NL-based scoring via LLMs/DSPy).
    """
    if scoring_func is None:
        return None
    if callable(scoring_func):
        try:
            return float(scoring_func(output_obj))
        except Exception:
            return None
    # NL-based scoring (str) not implemented here to keep dependencies optional
    return None


def run_probabilistic(
    generate_fn: Callable[[], Any],
    input_obj: Any,
    scoring_func: Optional[Callable[[Any], float] | str] = None,
    config: Optional[ProbabilisticConfig] = None,
) -> Any:
    """
    Run a generation function with post-condition checks and optional scoring/retries.
    The generate_fn must return a Pydantic Output model instance.
    """
    cfg = config or ProbabilisticConfig()

    last_output = None
    last_error: Optional[str] = None
    for _ in range(max(1, cfg.max_retries + 1)):
        output = generate_fn()
        last_output = output

        ok, err = validate_post_conditions(input_obj, output)
        if not ok:
            last_error = err
            continue

        if cfg.score_threshold is not None:
            score = score_output(scoring_func, output)
            if score is None or score < cfg.score_threshold:
                last_error = f"Score below threshold: {score} < {cfg.score_threshold}"
                continue

        # success
        return output

    # Exhausted retries; return last output; callers may choose to fallback
    return last_output