"""The framework sentinels are module constants, not env settings.

NOT_FOUND, INVALID and the error-message templates are fixed in fastworkflow.utils.signatures.
An env file that still sets one of their old keys has no effect, and the shipped env template
no longer carries them.
"""
import os
import re

import fastworkflow
from fastworkflow.utils import signatures
from fastworkflow._workflows.command_metadata_extraction import parameter_extraction

ENV_TEMPLATE = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "fastworkflow.env")
OLD_KEYS = ("NOT_FOUND", "INVALID", "MISSING_INFORMATION_ERRMSG",
            "INVALID_INFORMATION_ERRMSG", "PARAMETER_EXTRACTION_ERROR_MSG")


def test_the_sentinels_have_their_fixed_values():
    assert signatures.NOT_FOUND == "NOT_FOUND"
    assert signatures.INVALID == "INVALID"
    assert signatures.MISSING_INFORMATION_ERRMSG == "Missing parameter values: "
    assert signatures.INVALID_INFORMATION_ERRMSG == "Invalid parameter values: "
    assert signatures.PARAMETER_EXTRACTION_ERROR_MSG == "Error in parameter extraction: {error}"


def test_parameter_extraction_uses_the_same_objects():
    assert parameter_extraction.NOT_FOUND is signatures.NOT_FOUND
    assert parameter_extraction.MISSING_INFORMATION_ERRMSG is signatures.MISSING_INFORMATION_ERRMSG
    assert parameter_extraction.PARAMETER_EXTRACTION_ERROR_MSG is signatures.PARAMETER_EXTRACTION_ERROR_MSG


def test_an_env_setting_of_an_old_key_is_ignored(monkeypatch):
    monkeypatch.setenv("NOT_FOUND", "MISSING_FROM_ENV")
    monkeypatch.setenv("MISSING_INFORMATION_ERRMSG", "from env: ")
    assert signatures.NOT_FOUND == "NOT_FOUND"
    assert signatures.MISSING_INFORMATION_ERRMSG == "Missing parameter values: "


def test_the_env_template_no_longer_carries_the_old_keys_and_names_fw_turn_max_steps():
    with open(ENV_TEMPLATE, encoding="utf-8") as fh:
        text = fh.read()
    for key in OLD_KEYS:
        assert not re.search(rf"^{key}\s*=", text, re.M), key
    assert re.search(r"^# FW_TURN_MAX_STEPS=400$", text, re.M)
