"""Scoped archive and label integration, plus opt-in real DSPy provider tests."""
import hashlib
import inspect
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

import fastworkflow
from fastworkflow.observation_offloading.archive import (
    REDACTION_ENV,
    REDACTION_OFF,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.labels import (
    offload_label, label_alias, is_offload_label,
    alias_line)
from fastworkflow import context_budget, tracing
from fastworkflow.observation_offloading.search import (
    SEARCH_MODEL_ENV,
    search_memory,
)
from fastworkflow.observation_offloading.state import (
    reset_observation_state,
    snapshot_events,
)
from fastworkflow.utils.logging import logger


class ObservationSearch(unittest.TestCase):
    def setUp(self):
        reset_observation_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('channel', 'turn')

    def persist(self, alias, text):
        step_index = int(alias[1:])
        self.archive.persist(self.scope, alias=alias, command_name='show_owners', step_index=step_index,
                             text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def search(self, question, alias, **kwargs):
        return search_memory(question, alias, scope=self.scope, selected_archive=self.archive, **kwargs)

    def test_alias_is_required_and_validated_before_model_call(self):
        self.assertIs(inspect.signature(search_memory).parameters['alias'].default, inspect.Parameter.empty)
        for alias in ['', 'O', 'O1 O2', 'O-1', 'S1', 'O1; O2']:
            with self.assertRaises(ValueError):
                self.search('Who?', alias)
        self.assertIn('no matching offloaded handle O0', self.search('Who?', 'O0'))

    def test_no_fallback_to_another_handle_or_turn(self):
        self.persist('O1', 'Secret from another observation')
        self.assertIn('no matching offloaded handle O2', self.search('Who?', 'O2'))
        other = RuntimeHandleScope('channel', 'another-turn')
        result = search_memory('Who?', 'O1', scope=other, selected_archive=self.archive)
        self.assertIn('no matching offloaded handle O1', result)

    def test_label_uses_command_argument_and_authored_description(self):
        label = offload_label(alias='O11', command_name='show_owners limit=100',
                             response='payload', description='identity UIDs and holder names')
        self.assertEqual(label, 'Offloaded observation O11 returned by show_owners limit=100. '
                                'It contains identity UIDs and holder names. '
                                'Normally restored for the final answer.')
        self.assertTrue(is_offload_label(label))
        self.assertEqual(label_alias(label), 'O11')

    def test_a_label_in_the_earlier_wording_is_still_recognised(self):
        # A trajectory recorded before the wording changed must still resume.
        legacy = ('Use search_memory tool to search inside Observation O9 returned by '
                  'show_owners. It was offloaded to memory and contains holder rows.')
        self.assertTrue(is_offload_label(legacy))
        self.assertEqual(label_alias(legacy), 'O9')
        # And the older label that carried the whole restore promise.
        promise = ('Offloaded observation O9 returned by show_owners. It contains holder '
                   'rows. It is restored in full when the final answer is written, so search '
                   'it with search_memory only for a value you need for your next step.')
        self.assertTrue(is_offload_label(promise))
        self.assertEqual(label_alias(promise), 'O9')

    def test_small_observations_and_long_command_arguments_never_expand(self):
        for turn, (text, command) in enumerate([("Context is now '*'", 'reset_context'), ('x'*5000, 'query '+'é'*6000)]):
            # One observation per alias per scope: each case is its own turn.
            scope = RuntimeHandleScope('channel', f'turn-{turn}')
            trajectory = {'tool_name_0': 'execute_workflow_query', 'tool_args_0': {'command': command}, 'observation_0': alias_line('O0') + text}
            # min_offload_saving_bytes=0 removes the 1 KB floor entirely, so the
            # only thing left to refuse these is the swap itself being a loss.
            decisions = compact_trajectory(trajectory, step_index=0, min_offload_saving_bytes=0,
                recent_observations_protected=0, packed_target_tokens=1,
                scope=scope, selected_archive=self.archive)
            self.assertEqual(trajectory['observation_0'], alias_line('O0') + text)
            self.assertEqual(decisions[0]['reason'], 'below_min_saving')
            self.assertLess(decisions[0]['offload_saving_bytes'], 0)
            # The observation stays inline AND is searchable: keeping it in the
            # prompt is a residency decision, not an availability one (A2).
            self.assertEqual(self.archive.get(scope, 'O0')['text'], text)

    @unittest.skipUnless(os.environ.get('FW_TEST_OBSERVATION_SEARCH_LIVE') == '1', 'requires configured observation-search provider')
    def test_broad_question_returns_bounded_summary_not_truncated_table(self):
        text = '477 holder(s).\n' + '\n'.join(f'{i:032x} Person {i}' for i in range(477))
        self.persist('O0', text)
        import dspy
        with dspy.context(disable_history=True):
            answer = self.search('Give every identity_uid and label in this result.', 'O0')
        self.assertIn('477', answer)
        self.assertLess(len(answer), 2000)
        self.assertEqual(snapshot_events()[-1]['status'], 'answered')
        self.assertGreater(snapshot_events()[-1]['usage']['completion_tokens'], 0)

    @unittest.skipUnless(os.environ.get('FW_TEST_OBSERVATION_SEARCH_LIVE') == '1', 'requires configured observation-search provider')
    def test_full_observation_reasoning_and_scope_with_real_dspy(self):
        self.persist('O0', 'Directory data\n'+'unrelated row\n'*1500+'\nAlisha Ochoa identity_uid=c062a2718f5148a84d081358a2b082b1 account_uid=account-123\n')
        self.persist('O0', 'Alisha Ochoa account_uid=WRONG-OTHER-OBSERVATION')
        answer = self.search('What is Alisha Ochoa account UID, not her identity UID?', 'O0')
        self.assertIn('account-123', answer)
        self.assertNotIn('WRONG-OTHER-OBSERVATION', answer)
        event = snapshot_events()[-1]
        self.assertEqual(event['status'], 'answered')
        self.assertGreater(event['observation_bytes'], 12000)


#: A model registered with litellm for these tests, whose window does not depend
#: on which litellm table (bundled or downloaded) the process loaded.
WIDE_TEST_MODEL = 'openai/fw-test-wide-search-model'
WIDE_TEST_WINDOW = 1_000_000


def register_wide_model() -> str:
    litellm.register_model({WIDE_TEST_MODEL: {
        'max_input_tokens': WIDE_TEST_WINDOW, 'max_tokens': 32_768, 'litellm_provider': 'openai',
        'mode': 'chat', 'input_cost_per_token': 0.0, 'output_cost_per_token': 0.0}})
    return WIDE_TEST_MODEL


class SearchModelRole(unittest.TestCase):
    """CORE-8: a deployment that never declared the search role still searches.

    ``get_lm`` is stubbed, so no provider, credential or network is reached.
    """

    def setUp(self) -> None:
        reset_observation_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('channel', 'turn')
        # Longer than SHORT_OBSERVATION_BYTES, so the search asks get_lm for a model.
        text = 'holder rows\n' + 'x' * 300
        self.archive.persist(self.scope, alias='O0', command_name='show_owners', step_index=0, text=text,
                             text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def selected_role(self, env):
        """The (model_env, key_env) pair ``search_memory`` asks ``get_lm`` for."""
        asked: dict = {}

        def get_lm(model_env, key_env, **_kwargs):
            asked['model_env'] = model_env
            asked['key_env'] = key_env
            raise RuntimeError('no provider in this test')

        with patch.dict(os.environ, env), \
                patch.dict('fastworkflow._env_vars', {}, clear=True), \
                patch('fastworkflow.observation_offloading.search.get_lm', get_lm):
            result = search_memory('Who?', 'O0', scope=self.scope,
                                   selected_archive=self.archive)
        asked['result'] = result
        return asked

    def test_an_undeclared_search_role_falls_back_to_the_agents_model(self):
        asked = self.selected_role({SEARCH_MODEL_ENV: ''})
        self.assertEqual(asked['model_env'], context_budget.AGENT_MODEL_ENV)
        self.assertEqual(asked['key_env'], 'LITELLM_API_KEY_AGENT')
        # And the diagnostic names the role that actually failed, not one the
        # deployment never set.
        self.assertIn('Check LLM_AGENT and LITELLM_API_KEY_AGENT', asked['result'])

    def test_a_declared_search_role_is_used_unchanged(self):
        asked = self.selected_role({SEARCH_MODEL_ENV: 'vendor/search-model'})
        self.assertEqual(asked['model_env'], SEARCH_MODEL_ENV)
        self.assertEqual(asked['key_env'], 'LITELLM_API_KEY_OBSERVATION_SEARCH')
        self.assertIn(
            'Check LLM_OBSERVATION_SEARCH and LITELLM_API_KEY_OBSERVATION_SEARCH',
            asked['result'])

    def test_the_recorded_subject_reaches_the_search_model(self):
        seen: dict = {}
        lm = SimpleNamespace(history=[{'usage': {'completion_tokens': 7}, 'cost': 0.0}],
                             model='fixture-lm')
        text = 'holder rows\n' + 'x' * 300
        self.archive.persist(self.scope, alias='O1', command_name='show_owners', step_index=1, text=text,
                             text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             context_clause='Account 28c5aeb5 Jane Roe')

        def predict(_signature):
            def call(question, subject, observation, **_):
                seen['subject'] = subject
                return SimpleNamespace(answer='ok')
            return call

        with patch('fastworkflow.observation_offloading.search.get_lm', return_value=lm), \
                patch('fastworkflow.observation_offloading.search.dspy') as fake_dspy:
            fake_dspy.Predict.side_effect = predict
            search_memory('Whose holders?', 'O1', scope=self.scope, selected_archive=self.archive)
        self.assertIn('Account 28c5aeb5 Jane Roe', seen['subject'])

    def test_the_archived_observation_is_sent_to_the_model_in_full(self):
        seen: dict = {}
        lm = SimpleNamespace(history=[{'usage': {'completion_tokens': 7}, 'cost': 0.0}],
                             model='fixture-lm')
        large = 'holder rows\n' + ('x' * 30_000)

        def predict(_signature):
            def call(question, subject, observation, **_):
                seen['observation'] = observation
                return SimpleNamespace(answer='ok')
            return call

        self.archive.persist(self.scope, alias='O1', command_name='show_owners', step_index=1, text=large,
                             text_sha256=hashlib.sha256(large.encode()).hexdigest())
        with patch('fastworkflow.observation_offloading.search.get_lm', return_value=lm), \
                patch('fastworkflow.observation_offloading.search.dspy') as fake_dspy:
            fake_dspy.Predict.side_effect = predict
            search_memory('Which holder?', 'O1', scope=self.scope, selected_archive=self.archive)
        self.assertEqual(seen['observation'], large)


class ArchiveAndSubjectCache(unittest.TestCase):
    """The archive's stored byte counts, and the subject cache kept per alias."""

    def setUp(self) -> None:
        reset_observation_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('channel', 'turn')

    def persist(self, alias, command, text, context_clause=None):
        step_index = int(alias[1:])
        self.archive.persist(self.scope, alias=alias, command_name=command, step_index=step_index, text=text,
                             text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             context_clause=context_clause)

    def test_a_context_is_read_back_from_the_archive_once_recorded(self):
        self.persist('O0', 'show_owners', 'rows\n' + 'x' * 400)
        self.assertIsNone(self.archive.get(self.scope, 'O0')['context_clause'])
        self.persist('O1', 'show_owners', 'rows\n' + 'x' * 400, context_clause='Account 1')
        self.assertEqual(self.archive.get(self.scope, 'O1')['context_clause'], 'Account 1')

