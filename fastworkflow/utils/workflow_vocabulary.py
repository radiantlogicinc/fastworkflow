"""The workflow's own vocabulary: the words a free-text argument must not merely echo.

A planner that fills a free-text argument can write a word that names the kind of record
rather than a record (``author='author'``), a field name (``title='title'``), a declared
choice of another input (``'book'``), or a pronoun (``'that one'``). Those words are the
workflow's own vocabulary. This module derives that set from the workflow's command
metadata. Nothing here is hand-listed except the closed list of English pronouns.

Two layers:

- ``derive_vocabulary`` is pure: it maps a ``WorkflowMetadata`` (command names, context names,
  input field names, declared choice values) to the set of words.
- ``write_workflow_vocabulary`` is the train-side builder. It stores the derived set in
  ``<workflow>/___command_info/workflow_vocabulary.json`` at the top level. The file is
  workflow-scoped and NOT versioned: like ``command_directory.json`` it derives from command
  source, not from a trained model, so it carries the ``source_fingerprint`` of that source.
- ``load_workflow_vocabulary`` is the runtime reader. It reads the stored file when its
  fingerprint matches the live source. Otherwise it derives the vocabulary in memory from the
  metadata and never writes into ``___command_info``. That fallback is what serves every
  workflow that has not been trained since this file was introduced.

Kind selectors. A ``str`` field that names the kinds it accepts (``resource_type``: "Which kind
of thing access is being asked about. One of: application, group, permission, repository,
resource") may take a kind word as its value; the kind-word screen does not refuse one of the
field's listed options. A field is a kind selector when its description lists kind words (an
enumeration "One of: a, b, c" / "e.g. x, y", or at least two kind words it names), or when its
labeled training examples (``<command>_param_labeled.json``) use kind words as its value. The
options are those listed words and example values. Kind words are the workflow's context names
(the record kinds its commands are grouped under), not the wider vocabulary (command-name parts
such as "find" and "list") nor the declared choice values (such as "add" or "login").

Fingerprint: the stored file is fresh only while the command source fingerprint and a digest
of the labeled-example files both match; the labeled files are rewritten by train, so they are
re-checked on each load rather than trusted.

The stored file (``FORMAT_VERSION`` 2)::

    {
      "format": 2,
      "source_fingerprint": "<compute_commands_source_fingerprint>",
      "labeled_fingerprint": "<digest of the *_param_labeled.json files>",
      "words": ["author", "book", "book_name", "book_uid", ...],   # sorted, lowercase
      "pronouns": ["he", "her", "it", "that", ...],                # sorted, lowercase
      "kind_words": ["account", "application", ...],               # sorted, lowercase
      "kind_selectors": {"Directory/who_has_access_to": {"resource_type": ["application", ...]}}
    }
"""
from __future__ import annotations

import enum
import hashlib
import json
import os
import re
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Literal, Mapping, get_args, get_origin

from fastworkflow.command_directory import CommandDirectory, compute_commands_source_fingerprint
from fastworkflow.utils import python_utils
from fastworkflow.utils.logging import logger

FORMAT_VERSION = 2
VOCABULARY_FILENAME = "workflow_vocabulary.json"
COMMAND_INFO_FOLDERNAME = "___command_info"
LABELED_SUFFIX = "_param_labeled.json"

#: English pronouns and demonstratives: closed, not derived from any workflow.
PRONOUNS = frozenset({
    "her", "him", "his", "they", "them", "their", "this", "that", "it", "these", "those",
    "she", "he", "its", "one",
})

#: Suffixes that turn a record-kind word into a field-name form (``right_name``, ``finding_uid``).
RECORD_SUFFIXES = ("name", "uid", "id", "code", "label")

#: The refusal of a free-text value that is only a kind word or field name of the workflow.
KIND_WORD_WHY = "is a kind word of this workflow, not a value"

#: An explicit enumeration in a field description: "One of: a, b, c", "one of a, b or c", "e.g. x, y".
_ENUMERATION = re.compile(
    r"(?i:\bone of|\be\.g\.?)\s*:?\s*"
    r"(?P<items>[a-z][a-z0-9_]*(?:\s*(?:,|\bor\b|\band\b)\s*[a-z][a-z0-9_]*)*)")
_ITEM_SEPARATOR = re.compile(r"\s*(?:,|\bor\b|\band\b)\s*")
_DESCRIPTION_WORD = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class WorkflowMetadata:
    """What the derivation reads from a workflow's command metadata. Core commands are excluded."""

    command_names: frozenset[str]   # qualified names: "find_member" or "Context/find_member"
    context_names: frozenset[str]
    field_names: frozenset[str]     # input fields of every command and skill
    choice_values: frozenset[str]   # declared Literal / Enum values of those fields
    #: ``(command, field, description)`` for each input field of a command or skill.
    field_descriptions: frozenset[tuple[str, str, str]] = frozenset()
    #: ``(command, field, value)`` for each string value a labeled example gives an input field.
    labeled_values: frozenset[tuple[str, str, str]] = frozenset()


@dataclass(frozen=True)
class WorkflowVocabulary:
    words: frozenset[str]
    pronouns: frozenset[str]
    source_fingerprint: str
    origin: str                     # "file" (read from the stored file) or "derived" (in memory)
    labeled_fingerprint: str = ""
    kind_words: frozenset[str] = frozenset()
    #: ``{qualified command: {field: options}}``: the kind-selector fields and their listed values.
    kind_selectors: Mapping[str, Mapping[str, frozenset[str]]] = dataclass_field(default_factory=dict)

    def has_word(self, text: str) -> bool:
        """Whether *text*, case-insensitively, is one of the workflow's words."""
        return text.strip().lower() in self.words

    def kind_options(self, command: str | None, field_name: str) -> frozenset[str]:
        """The values a kind-selector field lists (empty when the field is not one)."""
        return self.kind_selectors.get(command or "", {}).get(field_name, frozenset())


def is_pronoun_phrase(text: str) -> bool:
    """"it", "her", "that one": every word of the value is an English pronoun or demonstrative."""
    words = text.lower().split()
    return bool(words) and all(w in PRONOUNS for w in words)


def is_kind_word(text: str, field_name: str, vocabulary: WorkflowVocabulary | None,
                 options: frozenset[str] = frozenset()) -> bool:
    """The value is the input's own name (*field_name*), or a word of the workflow's vocabulary.

    A value the field lists as one of its kind-selector *options* is not a kind word here: the
    field names the kinds it takes.
    """
    word = text.lower()
    if word in options:
        return False
    return word == field_name.lower() or (vocabulary is not None and vocabulary.has_word(text))


def description_kind_options(description: str, kind_words: frozenset[str]) -> frozenset[str]:
    """The kind-selector options a field description lists; empty when it lists none.

    An explicit enumeration ("One of: a, b, c", "e.g. x, y") lists its items. Without one, the
    description lists the kind words it names, when it names at least two of them.
    """
    text = description or ""
    match = _ENUMERATION.search(text)
    if match:
        items = {word for word in _ITEM_SEPARATOR.split(match["items"]) if word}
        return frozenset(items) if len(items) >= 2 else frozenset()
    named = {word for word in _DESCRIPTION_WORD.findall(text.lower()) if word in kind_words}
    return frozenset(named) if len(named) >= 2 else frozenset()


def kind_words_of(metadata: WorkflowMetadata) -> frozenset[str]:
    """The workflow's kind words for kind selectors, lowercase: its context names, the record
    kinds its commands are grouped under (``group``, ``account``). Declared choice values are not
    used: a choice such as ``add``, ``login`` or ``every`` is not a record kind, and admitting it
    lets an unrelated field (``query``) accept it. A workflow whose commands have no contexts (skills
    only) has no kind words, so its kind selectors come from explicit enumerations alone."""
    words = {word.strip().lower() for word in metadata.context_names}
    return frozenset(word for word in words if word and word != "none")


def derive_kind_selectors(metadata: WorkflowMetadata) -> dict[str, dict[str, frozenset[str]]]:
    """``{command: {field: options}}`` for the workflow's kind-selector fields.

    Source (1): the field's description lists kind words (``description_kind_options``).
    Source (2): the field's labeled examples give kind words as its value (those values).
    Both sources contribute options to the same field; a field with neither is not listed.
    """
    kinds = kind_words_of(metadata)
    options: dict[tuple[str, str], set[str]] = {}
    for command, field_name, description in metadata.field_descriptions:
        listed = description_kind_options(description, kinds)
        if listed:
            options.setdefault((command, field_name), set()).update(listed)
    for command, field_name, value in metadata.labeled_values:
        word = value.strip().lower()
        if word in kinds:
            options.setdefault((command, field_name), set()).add(word)
    selectors: dict[str, dict[str, frozenset[str]]] = {}
    for (command, field_name), words in sorted(options.items()):
        selectors.setdefault(command, {})[field_name] = frozenset(words)
    return selectors


def _record_stem(field: str) -> str:
    """``finding`` for ``finding_uid``; ``author`` for ``author``; "" for a bare suffix (``name``)."""
    for suffix in RECORD_SUFFIXES:
        if field.endswith(f"_{suffix}") and len(field) > len(suffix) + 1:
            return field[: -(len(suffix) + 1)]
    return "" if field in RECORD_SUFFIXES else field


def derive_vocabulary(metadata: WorkflowMetadata) -> frozenset[str]:
    """The workflow's words, lowercase. Pure: the same metadata always gives the same set.

    Contains the context names, the declared choice values, the command names and the parts
    of their names, the field names (snake_case and spaced), the stems of the field names
    (``finding_uid`` gives ``finding``), and each record word compounded with the record
    suffixes (``finding_uid``, ``right_name``). Pronouns are not included: they are stored
    beside the words.
    """
    record_words = set(metadata.context_names) | set(metadata.choice_values)
    for field in metadata.field_names:
        stem = _record_stem(field)
        if stem:
            record_words.add(stem)

    words: set[str] = set(record_words)
    words.update(metadata.command_names)
    for command in metadata.command_names:
        words.update(command.rpartition("/")[2].split("_"))
    words.update(metadata.field_names)
    words.update(field.replace("_", " ") for field in metadata.field_names)
    for record in record_words:
        words.update(f"{record}_{suffix}" for suffix in RECORD_SUFFIXES)
    return frozenset(word.lower() for word in words if word)


def _choices(annotation: Any) -> set[str]:
    """The declared string choices of a field annotation: ``Literal`` values and ``Enum`` values,
    looking through ``Optional``, ``Union`` and containers."""
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {member.value for member in annotation if isinstance(member.value, str)}
    if get_origin(annotation) is Literal:
        return {value for value in get_args(annotation) if isinstance(value, str)}
    found: set[str] = set()
    for argument in get_args(annotation):
        found |= _choices(argument)
    return found


def _input_fields(directory: CommandDirectory, name: str) -> dict[str, tuple[Any, str]]:
    """``{field: (annotation, description)}`` of the command's or skill's ``Signature.Input``;
    {} when it has none."""
    try:
        directory.ensure_command_hydrated(name)
    except Exception as error:  # noqa: BLE001 - one unreadable command must not stop the derivation
        logger.warning(f"workflow vocabulary: cannot read the inputs of {name}: {type(error).__name__}: {error}")
        return {}
    metadata = directory.map_command_2_metadata[name]
    if not metadata.command_parameters_class or not metadata.parameter_extraction_signature_module_path:
        return {}
    module = python_utils.get_module(
        metadata.parameter_extraction_signature_module_path,
        metadata.workflow_folderpath or directory.workflow_folderpath,
    )
    target: Any = module
    for part in metadata.command_parameters_class.split("."):
        target = getattr(target, part, None)
    model_fields = getattr(target, "model_fields", None) or {}
    return {field: (info.annotation, info.description or "") for field, info in model_fields.items()}


def _labeled_examples(folder: Path, command: str) -> list[dict[str, Any]]:
    """The ``fields`` of each valid example in the command's labeled file; [] when it has none.
    The file is named by the command's bare name (``DirectoryExplorer/browse_catalog`` gives
    ``browse_catalog_param_labeled.json``), as the trainset reader names it."""
    path = folder / COMMAND_INFO_FOLDERNAME / f"{command.rpartition('/')[2]}{LABELED_SUFFIX}"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning(f"workflow vocabulary: {path} is unreadable ({error}); its examples are not read")
        return []
    examples = payload.get("valid_examples") if isinstance(payload, dict) else None
    return [example["fields"] for example in examples or []
            if isinstance(example, dict) and isinstance(example.get("fields"), dict)]


def labeled_fingerprint(workflow_folderpath: str) -> str:
    """A digest of the workflow's labeled-example files (names and contents). Does not create
    the folder; a workflow without labeled files gets the digest of nothing."""
    digest = hashlib.sha256()
    for path in sorted((Path(workflow_folderpath) / COMMAND_INFO_FOLDERNAME).glob(f"*{LABELED_SUFFIX}")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def gather_metadata(workflow_folderpath: str) -> WorkflowMetadata:
    """The derivation's input, read from the workflow's command metadata (core commands excluded)."""
    folder = Path(workflow_folderpath).resolve()
    directory = CommandDirectory.load(str(folder))
    core = set(directory.core_command_names)
    commands: set[str] = set()
    contexts: set[str] = set()
    fields: set[str] = set()
    choices: set[str] = set()
    descriptions: set[tuple[str, str, str]] = set()
    labeled: set[tuple[str, str, str]] = set()
    for name in directory.map_command_2_metadata:
        if name in core:
            continue
        commands.add(name)
        context = name.rpartition("/")[0]
        if context:
            contexts.add(context)
        inputs = _input_fields(directory, name)
        for field, (annotation, description) in inputs.items():
            fields.add(field)
            choices |= _choices(annotation)
            descriptions.add((name, field, description))
        for example in _labeled_examples(folder, name):
            for field, value in example.items():
                if field in inputs and isinstance(value, str):
                    labeled.add((name, field, value))
    return WorkflowMetadata(
        command_names=frozenset(commands),
        context_names=frozenset(contexts),
        field_names=frozenset(fields),
        choice_values=frozenset(choices),
        field_descriptions=frozenset(descriptions),
        labeled_values=frozenset(labeled),
    )


_IN_MEMORY: dict[str, WorkflowVocabulary] = {}


def vocabulary_path(workflow_folderpath: str) -> Path:
    return Path(workflow_folderpath) / COMMAND_INFO_FOLDERNAME / VOCABULARY_FILENAME


def _derived(folder: str, fingerprint: str, labeled: str) -> WorkflowVocabulary:
    """The vocabulary derived from the workflow's metadata (the in-memory form, also what is stored)."""
    metadata = gather_metadata(folder)
    return WorkflowVocabulary(
        words=derive_vocabulary(metadata),
        pronouns=PRONOUNS,
        source_fingerprint=fingerprint,
        origin="derived",
        labeled_fingerprint=labeled,
        kind_words=kind_words_of(metadata),
        kind_selectors=derive_kind_selectors(metadata),
    )


def write_workflow_vocabulary(workflow_folderpath: str) -> Path:
    """Derive the workflow's vocabulary and store it atomically at the top of ``___command_info``.

    Called by ``fastworkflow train``. The fingerprints are taken before the metadata is read, so
    a source or labeled-file edit made during the write leaves a stale file rather than a
    wrongly fresh one.
    """
    folder = str(Path(workflow_folderpath).resolve())
    fingerprint = compute_commands_source_fingerprint(folder)
    labeled = labeled_fingerprint(folder)
    vocabulary = _derived(folder, fingerprint, labeled)
    payload = {
        "format": FORMAT_VERSION,
        "source_fingerprint": fingerprint,
        "labeled_fingerprint": labeled,
        "words": sorted(vocabulary.words),
        "pronouns": sorted(vocabulary.pronouns),
        "kind_words": sorted(vocabulary.kind_words),
        "kind_selectors": {
            command: {field_name: sorted(options) for field_name, options in fields.items()}
            for command, fields in sorted(vocabulary.kind_selectors.items())
        },
    }
    target = vocabulary_path(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{VOCABULARY_FILENAME}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _IN_MEMORY.pop(folder, None)
    return target


def _string_list(value: Any) -> frozenset[str] | None:
    """*value* as a frozenset of strings, or None when it is not a list of strings."""
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return frozenset(value)
    return None


def _selectors_from_json(value: Any) -> dict[str, dict[str, frozenset[str]]] | None:
    """The stored ``kind_selectors`` as ``{command: {field: options}}``, or None when malformed."""
    if not isinstance(value, dict):
        return None
    selectors: dict[str, dict[str, frozenset[str]]] = {}
    for command, fields in value.items():
        if not isinstance(fields, dict):
            return None
        for field_name, options in fields.items():
            listed = _string_list(options)
            if listed is None:
                return None
            selectors.setdefault(command, {})[field_name] = listed
    return selectors


def _read_stored(folder: str, fingerprint: str, labeled: str) -> WorkflowVocabulary | None:
    """The stored vocabulary when the file exists, is this format and matches both fingerprints."""
    path = vocabulary_path(folder)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.debug(f"workflow vocabulary: {path} is unreadable ({error}); deriving in memory")
        return None
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_VERSION:
        logger.debug(f"workflow vocabulary: {path} has another format; deriving in memory")
        return None
    if payload.get("source_fingerprint") != fingerprint or payload.get("labeled_fingerprint") != labeled:
        logger.debug(f"workflow vocabulary: {path} is stale (source or labeled examples changed); "
                     "deriving in memory")
        return None
    words = _string_list(payload.get("words"))
    pronouns = _string_list(payload.get("pronouns"))
    kinds = _string_list(payload.get("kind_words"))
    selectors = _selectors_from_json(payload.get("kind_selectors"))
    if words is None or pronouns is None or kinds is None or selectors is None:
        logger.debug(f"workflow vocabulary: {path} is malformed; deriving in memory")
        return None
    return WorkflowVocabulary(words, pronouns, fingerprint, origin="file", labeled_fingerprint=labeled,
                              kind_words=kinds, kind_selectors=selectors)


def load_workflow_vocabulary(workflow_folderpath: str) -> WorkflowVocabulary:
    """The workflow's vocabulary: the stored file when fresh, else derived in memory.

    Never writes. The result is cached per workflow path for the process and recomputed only
    when the command sources' or the labeled examples' fingerprint changes. Logs (debug) which
    path was taken when the cache is filled.
    """
    folder = str(Path(workflow_folderpath).resolve())
    fingerprint = compute_commands_source_fingerprint(folder)
    labeled = labeled_fingerprint(folder)
    cached = _IN_MEMORY.get(folder)
    if cached is not None and cached.source_fingerprint == fingerprint and cached.labeled_fingerprint == labeled:
        return cached
    vocabulary = _read_stored(folder, fingerprint, labeled)
    if vocabulary is None:
        vocabulary = _derived(folder, fingerprint, labeled)
        logger.debug(f"workflow vocabulary for {folder}: derived in memory "
                     f"(no fresh {VOCABULARY_FILENAME}); {len(vocabulary.words)} words")
    else:
        logger.debug(f"workflow vocabulary for {folder}: read {VOCABULARY_FILENAME} "
                     f"({len(vocabulary.words)} words)")
    _IN_MEMORY[folder] = vocabulary
    return vocabulary
