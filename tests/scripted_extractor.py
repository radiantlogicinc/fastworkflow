"""Script the shared parameter extractor's model in a test (fix-48np).

A planning command's typed call takes its arguments from the shared extractor
(``planning_commands.extracted_arguments``), which reads its LM with ``dspy_utils.get_lm("LLM_PARAM_EXTRACTION",
...)`` rather than from the caller's dspy context. A test that scripts the planner's ``DummyLM`` scripts the
extractor's answers here; every other role still gets its own LM.
"""

from __future__ import annotations

import dspy
from dspy.utils import DummyLM


class ScriptedExtractor(DummyLM):
    """Answers the extractor's calls in order (JSON, as the extractor's adapter reads them) and keeps each
    request in ``prompts`` (a list, so a copy of the LM shares it)."""

    def __init__(self, answers):
        super().__init__(list(answers), adapter=dspy.JSONAdapter())
        self.prompts: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def script_extractor(monkeypatch, *answers: dict) -> ScriptedExtractor:
    """Make the shared extractor answer *answers* in order (each a dict of the planning command's fields; a
    ``reasoning`` is added when missing). Returns the scripted LM."""
    from fastworkflow.utils import signatures

    lm = ScriptedExtractor([{"reasoning": "scripted", **answer} for answer in answers])
    original = signatures.dspy_utils.get_lm

    def get_lm(name, *args, **kwargs):
        return lm if name == "LLM_PARAM_EXTRACTION" else original(name, *args, **kwargs)

    monkeypatch.setattr(signatures.dspy_utils, "get_lm", get_lm)
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    return lm
