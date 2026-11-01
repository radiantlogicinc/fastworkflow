"""Phase 9 step 5a (fix-3rsq.9.8, design 2.50): each head records its not-here samples.

A head's sparse "not here" class takes one hand-written seed of every foreign command (up to
``NOT_HERE_SEEDS``, or 8 under sparse_8). The sample for a head is chosen ONCE, when the command
first enters the head's foreign set, recorded in the head's training signature, and reused on later
trains while its seed text still exists. So:

* a seed edit of command C dirties only C's own heads, and a foreign head whose recorded sample
  moved (or was edited away) is dirty too, and only that head;
* adding or removing a command dirties every head;
* ``--regenerate-utterances`` chooses every sample again.

The planner's per-head decision is ``dirty_heads_from_signatures``, which is pure over two signatures
and the label maps; the rules are tested on hand-built signatures. The real-workflow tests copy
``fastworkflow/examples/messaging_app_4`` into ``tmp_path`` and run the real planner. Nothing here
fits a model: the one train() call is a ``stop_before_fit`` glue run whose only fake is the LLM
boundary (generation), the same seam tests/test_lookup_head_scope.py uses.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    _stable_rank,
    choose_not_here_sample,
    experiment_reserved_class_arm,
    resolve_not_here_sample,
)
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import selective_training as st
from fastworkflow.train import utterance_cache
from fastworkflow.train.determinism import (
    ProvenanceRecorder,
    UtteranceProvenance,
    set_provenance_recorder,
)
from fastworkflow.train.selective_training import (
    CommandFingerprint,
    ContextSignature,
    TrainingSignature,
    dirty_heads_from_signatures,
)

REPO = Path(__file__).resolve().parent.parent
MESSAGING = REPO / "fastworkflow" / "examples" / "messaging_app_4"
SEED = 42


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


# ----------------------------------------------------------------- hand-built workflow (planner)

#: Same shape as tests/test_selective_training.py: a head's OWN label space, base-resolved.
CONTEXT_COMMANDS = {
    "Workspace": {"Workspace/bulk_decide", "Workspace/list_all"},
    "ReviewTicket": {"ReviewTicket/certify_approve"},
    "Account": {"Account/close"},
    "AdminAccount": {"Account/close", "AdminAccount/force_close"},
}

SEEDS = {
    "Workspace/bulk_decide": [f"bulk decide phrasing {index}" for index in range(6)],
    "Workspace/list_all": ["list everything", "show all workspaces"],
    "ReviewTicket/certify_approve": [f"certify ticket variant {index}" for index in range(5)],
    "Account/close": [f"close account variant {index}" for index in range(4)],
    "AdminAccount/force_close": ["force close this account"],
}


def _foreign(context_name: str, commands) -> list[str]:
    return sorted(set(commands) - CONTEXT_COMMANDS[context_name])


def _samples(seeds: dict, count: int = 1, recorded: dict | None = None) -> dict:
    """Each head's samples through the real resolver, reusing *recorded* where it still holds."""
    samples = {}
    for context_name in CONTEXT_COMMANDS:
        samples[context_name] = {
            command: resolve_not_here_sample(
                context_name, command, seeds[command], count,
                ((recorded or {}).get(context_name) or {}).get(command))
            for command in _foreign(context_name, seeds) if seeds.get(command)
        }
    return samples


def _signature(seeds: dict, samples: dict, contexts: dict | None = None) -> TrainingSignature:
    contexts = CONTEXT_COMMANDS if contexts is None else contexts
    fingerprints = {
        command: CommandFingerprint(
            command_name=command,
            source_path=f"/workflow/_commands/{command}.py",
            source_sha256=hashlib.sha256(f"source {command}".encode()).hexdigest(),
            seed_utterances_sha256=hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        )
        for command, texts in seeds.items()
    }
    context_signatures = {
        name: ContextSignature(
            context_name=name,
            label_space=sorted(own),
            reserved_class_arm="sparse_1",
            not_here_samples=samples.get(name),
        )
        for name, own in contexts.items()
    }
    return TrainingSignature(command_fingerprints=fingerprints, contexts=context_signatures)


def _heads_dirtied(previous: TrainingSignature, current: TrainingSignature, contexts=None) -> dict:
    _dirty, reasons = dirty_heads_from_signatures(
        previous, current, CONTEXT_COMMANDS if contexts is None else contexts)
    return reasons


def _edit(seeds: dict, command: str, *, add: str | None = None, replace: tuple | None = None) -> dict:
    edited = {name: list(texts) for name, texts in seeds.items()}
    if add is not None:
        edited[command].append(add)
    if replace is not None:
        old, new = replace
        edited[command] = [new if text == old else text for text in edited[command]]
    return edited


# ----------------------------------------------------------------- the choice itself (pure)

def test_the_choice_is_deterministic_and_drawn_from_the_seeds():
    seeds = [f"utterance number {index}" for index in range(10)]
    first = choose_not_here_sample("Workspace", "Workspace/bulk_decide", seeds, 1)
    assert first == choose_not_here_sample("Workspace", "Workspace/bulk_decide", seeds, 1)
    assert len(first) == 1 and set(first) <= set(seeds)


def test_the_choice_does_not_depend_on_the_order_of_the_seed_list():
    seeds = [f"utterance number {index}" for index in range(12)]
    shuffled = list(seeds)
    random.Random(7).shuffle(shuffled)
    assert choose_not_here_sample("Workspace", "A/one", seeds, 3) == \
        choose_not_here_sample("Workspace", "A/one", shuffled, 3)


def test_the_stable_rank_is_sha256_over_head_command_and_seed():
    expected = hashlib.sha256("\x00".join(("H", "A/one", "text")).encode("utf-8")).hexdigest()
    assert _stable_rank("H", "A/one", "text") == expected


def test_the_choice_is_the_same_under_another_python_hash_seed(tmp_path):
    """Never Python's salted hash: two interpreters with different PYTHONHASHSEED agree."""
    code = (
        "from fastworkflow.model_pipeline_training import choose_not_here_sample as c;"
        "print(c('Workspace', 'Workspace/bulk_decide', [f'utterance number {i}' for i in range(20)], 3))"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed}, timeout=300, check=True)
        outputs.add(completed.stdout.strip().splitlines()[-1])
    in_process = str(choose_not_here_sample(
        "Workspace", "Workspace/bulk_decide", [f"utterance number {i}" for i in range(20)], 3))
    assert outputs == {in_process}


def test_a_recorded_sample_is_kept_while_its_text_survives_even_if_a_lower_ranked_seed_arrives():
    seeds = ["recorded phrase", "another phrase"]
    recorded = choose_not_here_sample("H", "C/cmd", seeds, 1)
    lower = next(
        f"new phrase {index}" for index in range(100_000)
        if _stable_rank("H", "C/cmd", f"new phrase {index}") < _stable_rank("H", "C/cmd", recorded[0])
    )
    assert choose_not_here_sample("H", "C/cmd", seeds + [lower], 1) == [lower]  # a fresh choice moves
    assert resolve_not_here_sample("H", "C/cmd", seeds + [lower], 1, recorded) == recorded


def test_a_recorded_seed_edited_away_is_chosen_again_for_that_command():
    seeds = ["recorded phrase", "another phrase", "third phrase"]
    recorded = choose_not_here_sample("H", "C/cmd", seeds, 1)
    survivors = [text for text in seeds if text != recorded[0]]
    rechosen = resolve_not_here_sample("H", "C/cmd", survivors, 1, recorded)
    assert rechosen == choose_not_here_sample("H", "C/cmd", survivors, 1)
    assert rechosen != recorded and rechosen[0] in survivors


def test_no_recorded_sample_means_a_fresh_choice():
    seeds = [f"phrase {index}" for index in range(5)]
    assert resolve_not_here_sample("H", "C/cmd", seeds, 1, None) == \
        choose_not_here_sample("H", "C/cmd", seeds, 1)
    assert resolve_not_here_sample("H", "C/cmd", seeds, 1, []) == \
        choose_not_here_sample("H", "C/cmd", seeds, 1)


def test_sparse_8_records_up_to_eight_distinct_samples_and_all_of_a_short_list():
    many = [f"phrase {index}" for index in range(20)]
    eight = choose_not_here_sample("H", "C/cmd", many, 8)
    assert len(eight) == 8 and len(set(eight)) == 8 and set(eight) <= set(many)
    few = ["one", "two", "three"]
    assert sorted(choose_not_here_sample("H", "C/cmd", few, 8)) == sorted(few)
    assert resolve_not_here_sample("H", "C/cmd", few, 8, few) == few


def test_the_record_round_trips_through_the_signature_json():
    signature = _signature(SEEDS, _samples(SEEDS))
    restored = TrainingSignature.model_validate(json.loads(signature.model_dump_json()))
    assert restored == signature
    assert restored.contexts["Workspace"].not_here_samples == _samples(SEEDS)["Workspace"]


# ----------------------------------------------------------------- the planner rules (hand-built)

def test_a_seed_edit_dirties_only_the_heads_that_label_the_command():
    previous = _signature(SEEDS, _samples(SEEDS))
    edited = _edit(SEEDS, "Workspace/bulk_decide", add="a phrase no head recorded")
    current = _signature(edited, _samples(edited, recorded=previous_samples(previous)))

    reasons = _heads_dirtied(previous, current)

    assert set(reasons) == {"Workspace"}
    assert reasons["Workspace"] == ["label space contains changed command(s): Workspace/bulk_decide"]


def test_a_seed_edit_to_a_command_with_two_own_heads_dirties_both_of_them_only():
    previous = _signature(SEEDS, _samples(SEEDS))
    edited = _edit(SEEDS, "Account/close", add="a new close phrase")
    current = _signature(edited, _samples(edited, recorded=previous_samples(previous)))

    assert set(_heads_dirtied(previous, current)) == {"Account", "AdminAccount"}


def test_an_edit_of_a_recorded_sample_dirties_exactly_the_heads_that_recorded_it():
    previous = _signature(SEEDS, _samples(SEEDS))
    recorded_by = {
        name: previous.contexts[name].not_here_samples["Workspace/bulk_decide"][0]
        for name in ("ReviewTicket", "Account", "AdminAccount")
    }
    # The text ReviewTicket recorded for bulk_decide. Heads that chose a different text must hold.
    old = recorded_by["ReviewTicket"]
    heads_recording_old = {
        name for name, text in recorded_by.items() if text == old}
    assert "ReviewTicket" in heads_recording_old
    assert len(set(recorded_by.values())) >= 2, "the heads must not all record one text"

    edited = _edit(SEEDS, "Workspace/bulk_decide", replace=(old, "reworded bulk decide phrase"))
    current = _signature(edited, _samples(edited, recorded=previous_samples(previous)))
    reasons = _heads_dirtied(previous, current)

    assert set(reasons) == {"Workspace"} | heads_recording_old
    assert set(CONTEXT_COMMANDS) - set(reasons), "a head that recorded another text must carry"
    for name in heads_recording_old:
        assert current.contexts[name].not_here_samples["Workspace/bulk_decide"] != [old]
    for name in set(CONTEXT_COMMANDS) - set(reasons):
        assert current.contexts[name].not_here_samples["Workspace/bulk_decide"] == [
            previous.contexts[name].not_here_samples["Workspace/bulk_decide"][0]]


def test_an_added_command_dirties_every_head():
    previous = _signature(SEEDS, _samples(SEEDS))
    grown_contexts = {name: set(own) for name, own in CONTEXT_COMMANDS.items()}
    grown_contexts["Account"].add("Account/unlock")
    grown_seeds = dict(SEEDS, **{"Account/unlock": ["unlock the account"]})
    current = _signature(grown_seeds, _samples(grown_seeds, recorded=previous_samples(previous)),
                         contexts=grown_contexts)

    reasons = _heads_dirtied(previous, current, contexts=grown_contexts)

    assert set(reasons) == set(CONTEXT_COMMANDS)
    for name in CONTEXT_COMMANDS:
        assert any("every head is retrained" in reason for reason in reasons[name])


def test_a_removed_command_dirties_every_head():
    previous = _signature(SEEDS, _samples(SEEDS))
    shrunk_seeds = {name: texts for name, texts in SEEDS.items() if name != "AdminAccount/force_close"}
    shrunk_contexts = {name: set(own) for name, own in CONTEXT_COMMANDS.items()}
    shrunk_contexts["AdminAccount"].discard("AdminAccount/force_close")
    current = _signature(shrunk_seeds, _samples(shrunk_seeds, recorded=previous_samples(previous)),
                         contexts=shrunk_contexts)

    assert set(_heads_dirtied(previous, current, contexts=shrunk_contexts)) == set(CONTEXT_COMMANDS)


def test_an_unchanged_workflow_dirties_no_head():
    previous = _signature(SEEDS, _samples(SEEDS))
    current = _signature(SEEDS, _samples(SEEDS, recorded=previous_samples(previous)))
    assert _heads_dirtied(previous, current) == {}


def test_a_baseline_without_recorded_samples_dirties_every_head_once():
    """Old artifacts: the first train chooses and records the samples, a full retrain once."""
    previous = _signature(SEEDS, {})
    for context_signature in previous.contexts.values():
        context_signature.not_here_samples = None
    current = _signature(SEEDS, _samples(SEEDS))

    reasons = _heads_dirtied(previous, current)

    assert set(reasons) == set(CONTEXT_COMMANDS)
    for name in CONTEXT_COMMANDS:
        assert any("no recorded not-here samples" in reason for reason in reasons[name])


def previous_samples(signature: TrainingSignature) -> dict:
    return {name: context.not_here_samples for name, context in signature.contexts.items()}


# ----------------------------------------------------------------- real workflow, real planner

def _copy(tmp_path: Path, name: str) -> str:
    destination = tmp_path / name
    shutil.copytree(MESSAGING, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    return str(destination)


def _rebuild(workflow: str) -> None:
    RoutingRegistry.clear_registry()


def _seeded_commands(workflow: str) -> dict[str, tuple[str, list[str]]]:
    """command name -> (its source file, its hand-written seeds), read from the real registry."""
    RoutingRegistry.clear_registry()
    cmd_dir = RoutingRegistry.get_definition(workflow, load_cached=False).command_directory
    found = {}
    # Core commands (go_up and the rest) are foreign candidates too, so they are read as well.
    for name in list(cmd_dir.get_execution_commands()) + list(cmd_dir.core_command_names):
        try:
            cmd_dir.ensure_command_hydrated(name)
        except Exception:  # noqa: BLE001 - a command that will not hydrate has no seeds to read
            continue
        metadata = cmd_dir.get_utterance_metadata(name)
        if metadata is None or not metadata.plain_utterances:
            continue
        found[name] = (metadata.generated_utterances_module_filepath, list(metadata.plain_utterances))
    return found


def _publish_baseline(workflow: str, *, with_contributions: bool = False) -> str:
    """Publish a complete-looking sparse_1 version with placeholder files. Nothing is trained."""
    RoutingRegistry.clear_registry()
    root = Path(workflow, "___command_info", utterance_cache.CACHE_DIRNAME)
    root.mkdir(parents=True, exist_ok=True)
    contexts = sorted(st.contexts_for_training(workflow))
    context_commands, _ = st.build_context_maps(workflow, contexts)
    for commands in context_commands.values():
        for command in commands:
            Path(root, f"{utterance_cache.slugify(command)}.baseline.json").write_text(
                json.dumps({"entries": {}}), encoding="utf-8")
    version = av.new_version_id()
    for context_name in contexts:
        folder = av.version_dir(workflow, version) / av.context_folder_name(context_name)
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
            (folder / artifact).touch()
    signature, _unresolved = st.compute_training_signature(
        workflow, contexts, seed=SEED, reserved_class_arm="sparse_1")
    fields = {}
    if with_contributions:
        fields = st.context_contribution_manifest_fields(signature, st._full_retrain(contexts, "baseline"))
    av.write_manifest(workflow, version, seed=SEED, **fields)
    st.save_training_signature(workflow, version, signature)
    av.publish_version(workflow, version)
    return version


def _plan(workflow: str, version: str, cache_mode: str = utterance_cache.MODE_REUSE):
    RoutingRegistry.clear_registry()
    return st.compute_training_plan(
        workflow, sorted(st.contexts_for_training(workflow)),
        changed_only=True, seed=SEED, carry_forward_from=version,
        cache_mode=cache_mode, reserved_class_arm="sparse_1")


def _baseline_samples(workflow: str, version: str) -> dict:
    signature = st.load_training_signature(workflow, version)
    assert signature is not None
    return {name: context.not_here_samples for name, context in signature.contexts.items()}


def _heads_with_foreign(workflow: str, command: str) -> tuple[set[str], set[str]]:
    """(the heads that label *command*, the heads that take it as a foreign command)."""
    context_commands, _ = st.build_context_maps(workflow, sorted(st.contexts_for_training(workflow)))
    own = {name for name, commands in context_commands.items() if command in commands}
    return own, set(context_commands) - own


def test_choosing_is_deterministic_and_recorded_in_the_signature(tmp_path):
    workflow = _copy(tmp_path, "rs_choose")
    contexts = sorted(st.contexts_for_training(workflow))
    first, _ = st.compute_training_signature(workflow, contexts, seed=SEED, reserved_class_arm="sparse_1")
    second, _ = st.compute_training_signature(workflow, contexts, seed=SEED, reserved_class_arm="sparse_1")

    seeds = _seeded_commands(workflow)
    own_by_context, _ = st.build_context_maps(workflow, contexts)
    for name in contexts:
        recorded = first.contexts[name].not_here_samples
        assert recorded is not None and recorded == second.contexts[name].not_here_samples
        for command, sample in recorded.items():
            assert sample == choose_not_here_sample(name, command, seeds[command][1], 1)
            assert command not in own_by_context[name]


def test_a_later_train_reuses_the_recorded_samples(tmp_path):
    workflow = _copy(tmp_path, "rs_reuse")
    version = _publish_baseline(workflow)
    plan, signature = _plan(workflow, version)

    assert plan.contexts_to_train == []
    assert {name: context.not_here_samples for name, context in signature.contexts.items()} == \
        _baseline_samples(workflow, version)


def test_an_edited_recorded_seed_is_chosen_again_for_that_command_only(tmp_path):
    """Exactly the heads that recorded the edited text, and the command's own heads, retrain."""
    workflow = _copy(tmp_path, "rs_edited")
    version = _publish_baseline(workflow)
    recorded = _baseline_samples(workflow, version)
    seeded = _seeded_commands(workflow)

    # A command whose recorded sample some foreign head holds, and whose text is unique in its file.
    target = None
    for command in sorted(seeded):
        own, foreign = _heads_with_foreign(workflow, command)
        recorders = {h: recorded[h][command][0] for h in foreign if recorded[h].get(command)}
        for text in set(recorders.values()):
            path = seeded[command][0]
            if Path(path).read_text(encoding="utf-8").count(f'"{text}"') == 1 and len(seeded[command][1]) >= 2:
                target = (command, text, own, {h for h, t in recorders.items() if t == text})
                break
        if target:
            break
    assert target is not None, "messaging_app_4 has a command whose recorded sample can be edited"
    command, old, own, recording_heads = target
    path = seeded[command][0]
    Path(path).write_text(Path(path).read_text(encoding="utf-8").replace(
        f'"{old}"', '"reworded for the planner test"', 1), encoding="utf-8")
    _rebuild(workflow)

    plan, signature = _plan(workflow, version)

    expected = own | recording_heads
    assert set(plan.contexts_to_train) == expected
    assert set(plan.contexts_carried_forward) == set(st.contexts_for_training(workflow)) - expected
    for name in st.contexts_for_training(workflow):
        now = signature.contexts[name].not_here_samples or {}
        before = recorded[name] or {}
        moved = {c for c in set(now) | set(before) if now.get(c) != before.get(c)}
        assert moved <= {command}, f"{name} re-chose {moved}, not only {command}"
        if name in recording_heads:
            assert now[command] != [old] and now[command][0] != old


def test_regenerate_utterances_chooses_every_sample_again(tmp_path):
    """The flag reaches the planner as cache_mode; a non-reuse mode ignores recorded samples."""
    workflow = _copy(tmp_path, "rs_regenerate")
    version = _publish_baseline(workflow)
    recorded = _baseline_samples(workflow, version)
    command = "ChatRoom/list_users"
    own, foreign = _heads_with_foreign(workflow, command)
    head = sorted(foreign)[0]
    old = recorded[head][command][0]

    # A seed that ranks below the recorded one for this head: a fresh choice moves to it.
    new = next(
        f"fresh phrase {index}" for index in range(100_000)
        if _stable_rank(head, command, f"fresh phrase {index}") < _stable_rank(head, command, old)
    )
    path = _seeded_commands(workflow)[command][0]
    Path(path).write_text(Path(path).read_text(encoding="utf-8").replace(
        f'"{old}"', f'"{old}",\n        "{new}"', 1), encoding="utf-8")
    _rebuild(workflow)

    reuse_plan, reuse_signature = _plan(workflow, version)
    assert reuse_signature.contexts[head].not_here_samples[command] == [old]
    assert head not in reuse_plan.contexts_to_train

    regen_plan, regen_signature = _plan(workflow, version, cache_mode=utterance_cache.MODE_REGENERATE)
    assert regen_plan.is_full_retrain
    assert regen_signature.contexts[head].not_here_samples[command] == [new]


def test_an_artifact_without_recorded_samples_is_retrained_once(tmp_path):
    workflow = _copy(tmp_path, "rs_old_artifact")
    version = _publish_baseline(workflow)
    path = st.signature_path(workflow, version)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["format_version"] = 1  # what a pre-9.8 build wrote
    for context in payload["contexts"].values():
        context.pop("not_here_samples", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    plan, signature = _plan(workflow, version)

    assert plan.is_full_retrain
    assert any("format_version" in reason for reason in plan.global_reasons)
    assert all(context.not_here_samples is not None for context in signature.contexts.values()
               if context.reserved_class_arm == "sparse_1")


def test_the_publish_check_verifies_the_recorded_samples(tmp_path):
    workflow = _copy(tmp_path, "rs_publish_check")
    version = _publish_baseline(workflow, with_contributions=True)
    assert av.verify_version_consistency(workflow, version).problems == []

    manifest_path = av.version_dir(workflow, version) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    block = manifest[av.CONTEXT_CONTRIBUTIONS_KEY]
    context_name = next(name for name in sorted(block) if block[name].get(av.NOT_HERE_SAMPLES_KEY))
    command = sorted(block[context_name][av.NOT_HERE_SAMPLES_KEY])[0]
    block[context_name][av.NOT_HERE_SAMPLES_KEY][command] = ["a sample nobody recorded"]
    manifest_path.write_text(json.dumps(manifest))

    problems = av.verify_version_consistency(workflow, version).problems
    assert any("not-here samples" in problem for problem in problems), problems


def _glue_train(tmp_path, monkeypatch, name: str, reuse=None) -> dict:
    """The real train(), stopped before any fit, with only the generation LLM boundary replaced."""
    from fastworkflow.model_pipeline_training import train
    from fastworkflow.train import generate_synthetic
    import fastworkflow.model_pipeline_training as mpt

    workflow_path = _copy(tmp_path, name)

    def deterministic_generation(seed_utterances, command_name, **_kwargs):
        rows = [f"{command_name} phrase {index}" for index in range(6)]
        personas = {row: f"persona-{index % 3}" for index, row in enumerate(rows)}
        return rows, UtteranceProvenance(
            command_name=command_name, seed=SEED,
            persona_ids=sorted(set(personas.values())), utterance_personas=personas,
            generator_config={"personas_per_batch": 1},
            seed_utterance_count=len(seed_utterances),
            generated_count=len(rows), final_count=len(rows))

    monkeypatch.setattr(generate_synthetic, "generate_diverse_utterances_with_provenance",
                        deterministic_generation)
    recorder = ProvenanceRecorder(workflow_path)
    set_provenance_recorder(recorder)
    mpt._HOLDOUT_EXPORT = {}
    workflow = fastworkflow.Workflow.create(workflow_path, workflow_id_str=f"not-here-{name}")
    try:
        with experiment_reserved_class_arm("sparse_1"):
            with contextlib.redirect_stdout(io.StringIO()):
                return train(workflow, stop_before_fit=True, not_here_samples=reuse)
    finally:
        mpt._HOLDOUT_EXPORT = None
        set_provenance_recorder(None)
        workflow.close()
        RoutingRegistry.clear_registry()


def test_train_records_the_samples_it_used_and_reuses_a_recorded_map(tmp_path, monkeypatch):
    first = _glue_train(tmp_path, monkeypatch, "rs_glue_first")
    assert first, "the trainer recorded no not-here samples"
    seeds = _seeded_commands(str(tmp_path / "rs_glue_first"))
    for context_name, samples in first.items():
        for command, sample in samples.items():
            assert sample == choose_not_here_sample(context_name, command, seeds[command][1], 1)

    again = _glue_train(tmp_path, monkeypatch, "rs_glue_again", reuse=first)
    assert again == first

    victim_context, victim_command = next(
        (ctx, cmd) for ctx, samples in first.items() for cmd in samples)
    tampered = {ctx: {cmd: list(s) for cmd, s in samples.items()} for ctx, samples in first.items()}
    tampered[victim_context][victim_command] = ["a seed that no longer exists"]
    rechosen = _glue_train(tmp_path, monkeypatch, "rs_glue_rechosen", reuse=tampered)
    assert rechosen[victim_context][victim_command] == first[victim_context][victim_command]
    assert {ctx: {c: s for c, s in v.items() if (ctx, c) != (victim_context, victim_command)}
            for ctx, v in rechosen.items()} == \
        {ctx: {c: s for c, s in v.items() if (ctx, c) != (victim_context, victim_command)}
         for ctx, v in first.items()}
