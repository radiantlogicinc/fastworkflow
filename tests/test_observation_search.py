"""Scoped archive and label integration, plus opt-in real DSPy provider tests."""
import hashlib
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastworkflow.observation_offloading.agent import current_search_reasoning
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.continuation import replan_trajectory_skeleton
from fastworkflow.observation_offloading.labels import offload_label, label_alias, is_offload_label, alias_line
from fastworkflow import context_budget, tracing
from fastworkflow.observation_offloading.search import (
    DEFAULT_PAGE_BYTES,
    EVIDENCE_PREFIX_NOTICE,
    SEARCH_MEMORY_MAX_PAGES,
    SEARCH_MODEL_ENV,
    SEARCH_OBSERVATION,
    NO_NARROWING,
    SHORT_OBSERVATION_BYTES,
    SHORT_OBSERVATION_MARK,
    bounded_evidence,
    narrowing_inputs,
    completion_was_truncated,
    is_bounded_evidence_observation,
    is_context_window_error,
    is_over_window_observation,
    search_answer_max_bytes_from_env,
    search_memory,
    search_observation_max_bytes,
    search_window_tokens,
)
from fastworkflow.observation_offloading.listing import ROWS_SERVED_MARK, parse_table, served_rows
from fastworkflow.observation_offloading.search_router import (
    ROUTER_ENV,
    ROUTER_KEY_ENV,
    SearchRouter,
    router_for_workflow,
)
from fastworkflow.observation_offloading.state import reset_runtime_state, snapshot_events


class ObservationSearch(unittest.TestCase):
    def setUp(self):
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 1, 'turn')

    def persist(self, alias, text):
        self.archive.persist(self.scope, alias=alias, offload_order=int(alias[1:]),
                             command_name='show_holders', step_index=int(alias[1:])-1,
                             text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def search(self, question, alias, **kwargs):
        return search_memory(question, alias, scope=self.scope, selected_archive=self.archive, **kwargs)

    def test_truncated_provider_response_is_not_an_evidence_answer(self):
        self.assertTrue(completion_was_truncated({'response': {'choices': [{'finish_reason': 'length'}]}}))
        self.assertTrue(completion_was_truncated({'usage': {'completion_tokens': 2048}}))
        self.assertFalse(completion_was_truncated({'response': {'choices': [{'finish_reason': 'stop'}]}, 'usage': {'completion_tokens': 50}}))

    def test_alias_is_required_and_validated_before_model_call(self):
        self.assertIs(inspect.signature(search_memory).parameters['alias'].default, inspect.Parameter.empty)
        for alias in ['', 'O0', 'O1 O2', 'O-1', 'S1', 'O1; O2']:
            with self.assertRaises(ValueError):
                self.search('Who?', alias)

    def test_no_fallback_to_another_handle_or_turn(self):
        self.persist('O1', 'Secret from another observation')
        self.assertIn('no matching offloaded handle O2', self.search('Who?', 'O2'))
        other = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 2, 'another-turn')
        result = search_memory('Who?', 'O1', scope=other, selected_archive=self.archive)
        self.assertIn('no matching offloaded handle O1', result)

    def test_label_uses_command_argument_and_authored_description(self):
        label = offload_label(alias='O12', command_name='show_holders limit=100',
                             response='payload', description='identity UIDs and holder names')
        self.assertEqual(label, 'Offloaded observation O12 returned by show_holders limit=100. '
                                'It contains identity UIDs and holder names. It is restored in full '
                                'when the final answer is written, so search it with search_memory '
                                'only for a value you need for your next step.')
        self.assertTrue(is_offload_label(label))
        self.assertEqual(label_alias(label), 'O12')

    def test_a_label_in_the_earlier_wording_is_still_recognised(self):
        # A trajectory recorded before the wording changed must still resume.
        legacy = ('Use search_memory tool to search inside Observation O9 returned by '
                  'show_holders. It was offloaded to memory and contains holder rows.')
        self.assertTrue(is_offload_label(legacy))
        self.assertEqual(label_alias(legacy), 'O9')

    def test_small_observations_and_long_command_arguments_never_expand(self):
        for turn, (text, command) in enumerate([("Context is now '*'", 'reset_context'), ('x'*5000, 'query '+'é'*6000)]):
            # One observation per alias per scope: each case is its own turn.
            scope = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 1, f'turn-{turn}')
            trajectory = {'tool_name_0': 'execute_workflow_query', 'tool_args_0': {'command': command}, 'observation_0': text}
            # min_offload_saving_bytes=0 removes the 1 KB floor entirely, so the
            # only thing left to refuse these is the swap itself being a loss.
            decisions = compact_trajectory(trajectory, min_offload_saving_bytes=0,
                recent_observations_protected=0, packed_target_tokens=1,
                scope=scope, selected_archive=self.archive)
            self.assertEqual(trajectory['observation_0'], alias_line('O1') + text)
            self.assertEqual(decisions[0]['reason'], 'below_min_saving')
            self.assertLess(decisions[0]['offload_saving_bytes'], 0)
            skeleton, _ = replan_trajectory_skeleton(trajectory, scope=scope, selected_archive=self.archive)
            # An inline copy in the replan skeleton keeps the same printed handle.
            self.assertEqual(skeleton['observation_0'], alias_line('O1') + text)
            # The observation stays inline AND is searchable: keeping it in the
            # prompt is a residency decision, not an availability one (A2).
            self.assertEqual(self.archive.get(scope, 'O1')['text'], text)

    def test_replan_pointer_is_persisted_and_small_text_stays_inline(self):
        trajectory = {'tool_name_0': 'execute_workflow_query', 'tool_args_0': {'command': 'show_holders'}, 'observation_0': 'holder rows\n'+'x'*9000,
                      'tool_name_1': 'execute_workflow_query', 'tool_args_1': {'command': 'reset_context'}, 'observation_1': "Context is now '*'"}
        skeleton, _ = replan_trajectory_skeleton(trajectory, greedy_max_bytes=1000, scope=self.scope, selected_archive=self.archive)
        self.assertTrue(is_offload_label(skeleton['observation_0']))
        self.assertEqual(self.archive.get(self.scope, 'O1')['text'], trajectory['observation_0'])
        self.assertEqual(skeleton['observation_1'], trajectory['observation_1'])

    def test_replan_archive_failure_preserves_original_evidence(self):
        # A real SQLite failure: the database path names a directory.
        self.archive.db_path = self.tmp.name
        text = 'holder rows\n' + 'x'*9000
        skeleton, metadata = replan_trajectory_skeleton(
            {'tool_name_0': 'execute_workflow_query', 'tool_args_0': {'command': 'show_holders'}, 'observation_0': text},
            greedy_max_bytes=1000, scope=self.scope, selected_archive=self.archive)
        self.assertEqual(skeleton['observation_0'], text)
        self.assertEqual(metadata['persistence_failures'], ['O1'])
        self.assertTrue(metadata['over_target'])

    def test_replan_keeps_irreducible_non_command_evidence_without_aborting(self):
        text = 'available command metadata\n' + 'x'*30000
        skeleton, metadata = replan_trajectory_skeleton(
            {'tool_name_0': 'what_can_i_do', 'observation_0': text},
            scope=self.scope, selected_archive=self.archive)
        self.assertEqual(skeleton['observation_0'], text)
        self.assertTrue(metadata['over_target'])
        self.assertIsNone(self.archive.get(self.scope, 'O1'))

    def test_reasoning_is_current_step_even_after_resume_or_truncation(self):
        trajectory = {'tool_name_8': 'search_memory', 'thought_8': 'stale thought',
                      'tool_name_20': 'search_memory', 'thought_20': 'Need the account UID, not identity UID'}
        agent = SimpleNamespace(current_trajectory=trajectory)
        self.assertEqual(current_search_reasoning(agent), trajectory['thought_20'])
        trajectory['tool_name_21'] = 'execute_workflow_query'
        self.assertEqual(current_search_reasoning(agent), '')

    @unittest.skipUnless(os.environ.get('FW_TEST_OBSERVATION_SEARCH_LIVE') == '1', 'requires configured observation-search provider')
    def test_broad_question_returns_bounded_summary_not_truncated_table(self):
        text = '477 holder(s).\n' + '\n'.join(f'{i:032x} Person {i}' for i in range(477))
        self.persist('O1', text)
        import dspy
        with dspy.context(disable_history=True):
            answer = self.search('Give every identity_uid and label in this result.', 'O1')
        self.assertIn('477', answer)
        self.assertLess(len(answer), 2000)
        self.assertEqual(snapshot_events()[-1]['status'], 'answered')
        self.assertGreater(snapshot_events()[-1]['usage']['completion_tokens'], 0)

    @unittest.skipUnless(os.environ.get('FW_TEST_OBSERVATION_SEARCH_LIVE') == '1', 'requires configured observation-search provider')
    def test_full_observation_reasoning_and_scope_with_real_dspy(self):
        self.persist('O1', 'Directory data\n'+'unrelated row\n'*1500+'\nAlisha Ochoa identity_uid=c062a2718f5148a84d081358a2b082b1 account_uid=account-123\n')
        self.persist('O2', 'Alisha Ochoa account_uid=WRONG-OTHER-OBSERVATION')
        answer = self.search('What is her UID?', 'O1', reasoning='I need Alisha Ochoa account UID, not her identity UID')
        self.assertIn('account-123', answer)
        self.assertNotIn('WRONG-OTHER-OBSERVATION', answer)
        event = snapshot_events()[-1]
        self.assertEqual(event['status'], 'answered')
        self.assertGreater(event['observation_bytes'], 12000)
        self.assertIn('account UID', event['reasoning'])


class ContextWindowExceededError(Exception):
    """The litellm class, by name only: the detector matches the chain, not the import."""


class SearchInputBound(unittest.TestCase):
    """What one search_memory call may hand the search model.

    Every test here is offline. The predictor is a stub that records the
    ``observation`` it was constructed with, so the byte bound is proved by
    measuring the input, not by trusting a provider to refuse it.
    """

    def setUp(self) -> None:
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 1, 'turn')

    def persist(self, text, alias='O1', command='show_holders'):
        self.archive.persist(self.scope, alias=alias, offload_order=int(alias[1:]),
                             command_name=command, step_index=int(alias[1:]) - 1,
                             text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def run_search(self, *, answer='ok', alias='O1', error=None, question='Which rows mention value?'):
        """search_memory against a stub predictor that captures its input."""
        seen: dict = {}
        lm = SimpleNamespace(history=[{'usage': {'completion_tokens': 7}, 'cost': 0.0}],
                             model='fixture-lm')

        def predict(_signature):
            def call(question, subject, observation, **_):
                seen['observation'] = observation
                seen['subject'] = subject
                seen['subject_bytes'] = len(subject.encode('utf-8'))
                seen['observation_bytes'] = len(observation.encode('utf-8'))
                if error is not None:
                    raise error
                return SimpleNamespace(answer=answer)
            return call

        with patch('fastworkflow.observation_offloading.search.get_lm', return_value=lm), \
                patch('fastworkflow.observation_offloading.search.dspy') as fake_dspy:
            fake_dspy.Predict.side_effect = predict
            seen['result'] = search_memory(question, alias, scope=self.scope,
                                           selected_archive=self.archive)
        seen['event'] = [e for e in snapshot_events() if e['kind'] == 'search_memory'][-1]
        return seen

    # -- the budget ----------------------------------------------------------

    def test_the_budget_is_a_quarter_of_the_search_models_window(self):
        # Since 2026-09-27 the bound is a quarter of the window, no longer the
        # former page geometry (DEFAULT_PAGE_BYTES x SEARCH_MEMORY_MAX_PAGES).
        self.assertEqual(SEARCH_OBSERVATION.reference_bytes,
                         context_budget.REFERENCE_WINDOW_TOKENS * 4 // 4)
        self.assertEqual(SEARCH_OBSERVATION.reference_bytes, 131_072)
        self.assertGreater(SEARCH_OBSERVATION.reference_bytes,
                           DEFAULT_PAGE_BYTES * SEARCH_MEMORY_MAX_PAGES)
        # And it is a fraction of a window, so it moves with the model.
        self.assertEqual(SEARCH_OBSERVATION.bytes_for(2 * context_budget.REFERENCE_WINDOW_TOKENS),
                         2 * SEARCH_OBSERVATION.reference_bytes)
        self.assertEqual(SEARCH_OBSERVATION.floor, DEFAULT_PAGE_BYTES)

    def test_the_bound_comes_from_the_search_models_own_window(self):
        env = {'FW_MODEL_CONTEXT_TOKENS': '', SEARCH_MODEL_ENV: 'vendor/wide-search-model'}
        windows = {'vendor/wide-search-model': 4 * context_budget.REFERENCE_WINDOW_TOKENS}
        with patch.dict(os.environ, env), patch.dict('fastworkflow._env_vars', {}, clear=True), \
                patch.object(context_budget, '_model_window_tokens', windows.get):
            tokens, source = search_window_tokens()
            self.assertEqual(tokens, 4 * context_budget.REFERENCE_WINDOW_TOKENS)
            self.assertIn('vendor/wide-search-model', source)
            self.assertEqual(search_observation_max_bytes(),
                             4 * SEARCH_OBSERVATION.reference_bytes)

    def test_the_bound_has_no_tuning_override(self):
        # The bound has no override of its own: the search model's window is
        # the only input, and FW_MODEL_CONTEXT_TOKENS is how it is corrected.
        self.assertIsNone(SEARCH_OBSERVATION.override_env)
        base = {'FW_MODEL_CONTEXT_TOKENS': '', SEARCH_MODEL_ENV: ''}
        with patch.dict(os.environ, base), \
                patch.dict('fastworkflow._env_vars', {}, clear=True):
            self.assertEqual(search_observation_max_bytes(), SEARCH_OBSERVATION.reference_bytes)
        with patch.dict(os.environ, {**base, 'FW_MODEL_CONTEXT_TOKENS': str(
                    2 * context_budget.REFERENCE_WINDOW_TOKENS)}), \
                patch.dict('fastworkflow._env_vars', {}, clear=True):
            self.assertEqual(search_observation_max_bytes(), 2 * SEARCH_OBSERVATION.reference_bytes)

    def test_the_page_is_a_hard_byte_bound_and_a_prefix(self):
        # text_page ends just after the newline that can sit AT the budget;
        # bounded_evidence never reports more bytes than it was given.
        for width in (1, 2, 3, 11):
            with self.subTest(width=width):
                text = (('x' * (width - 1)) + '\n') * (8_192 // width + 10)
                page = bounded_evidence(text, 4_096)
                self.assertLessEqual(page['shown_bytes'], 4_096)
                self.assertEqual(page['shown_bytes'], len(page['text'].encode('utf-8')))
                self.assertTrue(text.startswith(page['text']))
                self.assertTrue(page['bounded'])
                self.assertEqual(page['total_bytes'], len(text.encode('utf-8')))

    def test_a_heading_before_one_long_line_still_spends_the_whole_budget(self):
        # One text_page call ends at the LAST newline in its window, so a
        # 17-byte heading followed by an unbroken 30 KB line would be read as
        # 17 bytes and the search answered from the heading.
        text = 'holder uid label\n' + 'x' * 30_000
        page = bounded_evidence(text, 12_288)
        self.assertEqual(page['shown_bytes'], 12_288)
        self.assertTrue(text.startswith(page['text']))

    # -- the input the model actually gets -----------------------------------

    def test_an_oversized_observation_is_cut_before_the_model_is_called(self):
        # The measured worst case, byte for byte: 40,000 short rows, 440,000 bytes.
        text = 'row  value\n' * 40_000
        self.assertEqual(len(text.encode('utf-8')), 440_000)
        self.persist(text)
        budget = search_observation_max_bytes()
        seen = self.run_search()
        event = seen['event']
        # The budget bounds the EVIDENCE. What the model is handed is that read
        # plus the truncation disclosure, appended after the budget.
        evidence, _, notice = seen['observation'].partition('\n[TRUNCATED:')
        self.assertLessEqual(event['observation_sent_bytes'], budget)
        # Not a token of the budget wasted either: a whole page is still read.
        self.assertGreater(event['observation_sent_bytes'], budget - DEFAULT_PAGE_BYTES)
        self.assertTrue(text.startswith(evidence))
        self.assertEqual(len(evidence.encode('utf-8')), event['observation_sent_bytes'])
        self.assertTrue(notice)
        self.assertEqual(event['observation_bytes'], 440_000)
        self.assertTrue(event['observation_bounded'])
        self.assertEqual(event['observation_max_bytes'], budget)
        self.assertEqual(event['status'], 'answered')

    def test_the_answer_states_the_omission(self):
        text = 'row  value\n' * 40_000
        self.persist(text)
        seen = self.run_search(answer='rows 1-3 mention value')
        # The evidence read, not the model input: the input also carries the
        # truncation disclosure appended after the budget.
        result, shown = seen['result'], seen['event']['observation_sent_bytes']
        self.assertTrue(is_bounded_evidence_observation(result))
        self.assertTrue(result.startswith('Observation O1 (tier=sqlite, bounded):\n'), result)
        self.assertIn('rows 1-3 mention value', result)
        self.assertIn(f'answered from the first {shown:,} of 440,000 UTF-8 bytes of O1', result)
        self.assertIn(f'{440_000 - shown:,} bytes were NOT read', result)
        # The absence inference a partial read would invite is denied outright.
        self.assertIn('nothing missing from the answer is thereby absent from O1', result)
        # And the action offered is one that can actually reach the other bytes.
        self.assertIn('re-run show_holders', result)
        self.assertNotIn('call search_memory on O1 again with a narrower question', result)

    def test_a_bounded_read_and_a_bounded_answer_share_the_one_budget(self):
        self.persist('row  value\n' * 40_000)
        answer = '\n'.join(f'{index:032x} Person {index}' for index in range(400))
        seen = self.run_search(answer=answer)
        result = seen['result']
        self.assertLessEqual(len(result.encode('utf-8')), search_answer_max_bytes_from_env())
        self.assertIn('BOUNDED ANSWER', result)
        self.assertIn('BOUNDED EVIDENCE', result)
        self.assertTrue(result.rstrip().endswith(']'))

    def test_an_observation_under_the_bound_is_passed_through_unchanged(self):
        text = 'holder rows\n' + 'x' * 4_000
        self.assertLess(len(text.encode('utf-8')), search_observation_max_bytes())
        self.persist(text)
        seen = self.run_search(answer='Cooper holds it.')
        # Byte-identical input: nothing is re-paged, re-joined or re-encoded.
        self.assertEqual(seen['observation'], text)
        # Byte-identical output: exactly what an unbounded search returned.
        self.assertEqual(seen['result'], 'Observation O1 (tier=sqlite):\nCooper holds it.')
        self.assertNotIn('BOUNDED', seen['result'])
        self.assertFalse(seen['event']['observation_bounded'])
        self.assertEqual(seen['event']['observation_sent_bytes'],
                         seen['event']['observation_bytes'])

    # -- the overflow that used to be a bare failure string -------------------

    def test_a_context_window_error_is_a_typed_actionable_outcome(self):
        self.persist('row  value\n' * 40_000)
        seen = self.run_search(error=ContextWindowExceededError('prompt of 90000 tokens'))
        result = seen['result']
        self.assertTrue(is_over_window_observation(result), result)
        # Not the generic failure string, which named the class and nothing else.
        self.assertNotIn('failed (ContextWindowExceededError)', result)
        self.assertIn('do not retry it unchanged', result)
        self.assertIn('Re-run show_holders', result)
        self.assertIn('FW_MODEL_CONTEXT_TOKENS', result)
        self.assertIn(SEARCH_MODEL_ENV, result)
        self.assertIn(f'{search_observation_max_bytes():,}-byte bound', result)
        event = seen['event']
        self.assertEqual(event['status'], 'over_window')
        self.assertEqual(event['reason'], 'context_window_exceeded')
        self.assertEqual(event['error'], 'ContextWindowExceededError')

    def test_the_overflow_is_recognised_by_class_chain_or_by_words(self):
        class Subclass(ContextWindowExceededError):
            pass
        self.assertTrue(is_context_window_error(Subclass('x')))
        self.assertTrue(is_context_window_error(
            RuntimeError("This model's maximum context length is 8192 tokens")))
        self.assertTrue(is_context_window_error(ValueError('code: context_length_exceeded')))
        self.assertFalse(is_context_window_error(RuntimeError('connection reset by peer')))

    def test_an_unrelated_provider_failure_is_still_the_generic_failure(self):
        # Longer than SHORT_OBSERVATION_BYTES, so the provider is actually called.
        self.persist('holder rows\n' + 'x' * 300)
        seen = self.run_search(error=RuntimeError('connection reset by peer'))
        self.assertFalse(is_over_window_observation(seen['result']))
        self.assertIn('search of O1 failed (RuntimeError)', seen['result'])
        # The provider message itself is never printed back.
        self.assertNotIn('connection reset', seen['result'])
        self.assertEqual(seen['event']['status'], 'error')

    # -- what the search model is told about a partial read -------------------

    def test_the_prefix_is_disclosed_to_the_model_only_when_it_is_one(self):
        """DOC-5: the model must not read a cut list as the whole list.

        ``bounded_evidence_marking`` denies the absence inference to the AGENT,
        after the answer exists. This is the other end: the model that writes
        the answer is told in band that it is holding a prefix, and by how much,
        so a row it was never shown is not reported as missing.
        """
        cut = 'row  value\n' * 40_000
        self.persist(cut)
        seen = self.run_search()
        omitted = 440_000 - seen['event']['observation_sent_bytes']
        self.assertIn(EVIDENCE_PREFIX_NOTICE.format(omitted=omitted), seen['observation'])
        self.assertTrue(seen['event']['observation_bounded'])

        # An observation that fits is handed over byte-identical: no notice, and
        # nothing that could make a complete read look partial.
        reset_runtime_state()
        whole = 'holder rows\n' + 'x' * 4_000
        self.assertLess(len(whole.encode('utf-8')), search_observation_max_bytes())
        self.persist(whole, alias='O2')
        unbounded = self.run_search(alias='O2')
        self.assertEqual(unbounded['observation'], whole)
        self.assertNotIn('TRUNCATED', unbounded['observation'])
        self.assertFalse(unbounded['event']['observation_bounded'])


class SearchModelRole(unittest.TestCase):
    """CORE-8: a deployment that never declared the search role still searches.

    The window half already falls back to the agent's model
    (``search_window_tokens``), so refusing to call the agent's model left the
    two halves disagreeing: evidence sized for a model the search would not use.
    ``get_lm`` is stubbed, so no provider, credential or network is reached.
    """

    def setUp(self) -> None:
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 1, 'turn')
        # Longer than SHORT_OBSERVATION_BYTES, so the search asks get_lm for a model.
        text = 'holder rows\n' + 'x' * 300
        self.archive.persist(self.scope, alias='O1', offload_order=1,
                             command_name='show_holders', step_index=0, text=text,
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
            result = search_memory('Who?', 'O1', scope=self.scope,
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


class ShortObservationsAndServedRows(unittest.TestCase):
    """fix-cj7t / fix-xg1a: what search_memory answers WITHOUT the search model.

    A short observation is returned verbatim with the turn's better-matching
    handles; the rows of a listing a search asks for in full are copied, never
    paraphrased. ``get_lm`` raises in every test here, so any model call fails
    the test.
    """

    LISTING = ("3 holder(s); shown=3, remaining=0, complete=true.\n"
               "Each line below is `identity_uid  label`.\n"
               "identity_uid  label\n"
               + "\n".join(f"{index:032x}  Person {index}" for index in range(3)) + "\n")

    def setUp(self) -> None:
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 1, 'turn')

    def persist(self, alias, command, text):
        self.archive.persist(self.scope, alias=alias, offload_order=int(alias[1:]),
                             command_name=command, step_index=int(alias[1:]) - 1, text=text,
                             text_sha256=hashlib.sha256(text.encode()).hexdigest())

    class _DecisionClient:
        """Stands in for the decision-model client: fixed answers, records what it was sent."""

        def __init__(self, choice='all_rows', p_all_rows=0.97, error=None):
            self.choice, self.p_all_rows, self.error = choice, p_all_rows, error
            self.sent = []

        def system_one(self, *, state, questions):
            self.sent.append(state)
            if self.error is not None:
                raise self.error
            return SimpleNamespace(
                answers={'wants': SimpleNamespace(choice=self.choice,
                                                  probabilities={'all_rows': self.p_all_rows}),
                         'for_report': SimpleNamespace(noul=0.8)},
                usage=None)

    def search(self, question, alias, router=None):
        def no_model(*_args, **_kwargs):
            raise AssertionError('the search model must not be called')

        with patch('fastworkflow.observation_offloading.search.get_lm', no_model):
            result = search_memory(question, alias, scope=self.scope,
                                   selected_archive=self.archive, router=router)
        return result, [e for e in snapshot_events() if e['kind'] == 'search_memory'][-1]

    def test_a_short_observation_is_returned_verbatim_with_better_handles(self):
        self.persist('O1', 'list_entitlements', 'rows\n' + 'x' * (SHORT_OBSERVATION_BYTES + 1))
        self.persist('O2', 'go_up', "Context is now 'DirectoryExplorer'")
        result, event = self.search('List the entitlements for Heidi Turner', 'O2')
        self.assertTrue(result.startswith(SHORT_OBSERVATION_MARK))
        self.assertIn("Context is now 'DirectoryExplorer'", result)
        self.assertIn('O1 (list_entitlements', result)
        self.assertEqual(event['status'], 'short_verbatim')
        self.assertEqual(event['related'], ['O1'])

    def test_a_short_observation_with_no_better_match_says_so(self):
        self.persist('O2', 'go_up', "Context is now 'DirectoryExplorer'")
        result, _ = self.search('What remediation actions exist?', 'O2')
        self.assertIn('No other observation in this turn matches', result)

    def test_parse_table_reads_the_listing_by_its_shape(self):
        table = parse_table(self.LISTING)
        self.assertEqual(table['columns'], 'identity_uid  label')
        self.assertEqual(len(table['rows']), 3)
        self.assertEqual(table['preamble'][0], '3 holder(s); shown=3, remaining=0, complete=true.')
        self.assertIsNone(parse_table('{"total": 1, "types": {"a": "String"}}'))

    def test_served_rows_copy_the_rows_and_say_when_they_are_cut(self):
        table = parse_table(self.LISTING)
        whole, shown, total = served_rows('O3', table, 3_000)
        self.assertEqual((shown, total), (3, 3))
        for row in table['rows']:
            self.assertIn(row, whole)
        self.assertNotIn('NOT shown', whole)
        one_row = None
        for budget in range(100, len(whole.encode()) + 1):
            text, shown, _ = served_rows('O3', table, budget)
            if shown:
                self.assertLessEqual(len(text.encode()), budget)
            if shown == 1 and one_row is None:
                one_row = text
        self.assertIsNotNone(one_row)
        self.assertIn('rows 2-3 are NOT shown here', one_row)
        self.assertIn('restored in full when the final answer is written', one_row)

    def test_an_all_rows_route_is_served_by_code(self):
        self.persist('O3', 'show_holders', self.LISTING + 'x' * SHORT_OBSERVATION_BYTES)
        router = SearchRouter(self._DecisionClient(), questions={})
        result, event = self.search('Who are the holders?', 'O3', router=router)
        self.assertIn(ROWS_SERVED_MARK, result)
        self.assertIn(f"{0:032x}  Person 0", result)
        self.assertEqual(event['status'], 'rows_served')
        self.assertEqual(event['listing_shape'], 'aligned')
        self.assertEqual((event['rows_shown'], event['rows_total']), (3, 3))
        self.assertEqual(event['router']['choice'], 'all_rows')
        self.assertIn('latency_ms', event['router'])
        self.assertIn('latency_ms', event)

    def test_a_failing_router_leaves_the_search_to_the_model(self):
        self.persist('O3', 'show_holders', self.LISTING + 'x' * SHORT_OBSERVATION_BYTES)
        router = SearchRouter(self._DecisionClient(error=TimeoutError()), questions={})
        route = router.route('Who are the holders?', '', self.LISTING)
        self.assertEqual(route['error'], 'TimeoutError')
        self.assertFalse(router.wants_all_rows(route))

    def test_what_the_router_is_sent_passes_the_capture_policy(self):
        client = self._DecisionClient()
        secret = 'Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz123456'
        SearchRouter(client, questions={}).route('Who holds it?', f'I saw {secret}',
                                                 self.LISTING)
        sent = client.sent[0]
        self.assertEqual(set(sent), {'question', 'agent_reasoning', 'observation_start'})
        self.assertEqual(len(sent['observation_start'].splitlines()), 4)
        self.assertNotIn('sk-abcdefghijklmnopqrstuvwxyz123456', sent['agent_reasoning'])
        self.assertIn('[REDACTED]', sent['agent_reasoning'])

    class _RecordingSink:
        def __init__(self):
            self.spans = []

        def emit_span(self, span):
            self.spans.append(span)

        def emit_turn_record(self, record):
            pass

        def record_conversation_label(self, *args):
            pass

    def test_a_routing_call_is_an_fw_search_route_span(self):
        sink = self._RecordingSink()
        host = SimpleNamespace(trace_sink=sink, current_turn_key='turn-1')
        router = SearchRouter(self._DecisionClient(), questions={}, model='jev-test')
        router.route('Who are the holders of the secret list?', 'my reasoning', self.LISTING,
                     host=host)
        spans = [s for s in sink.spans if s.name == tracing.SPAN_SEARCH_ROUTE]
        self.assertEqual(len(spans), 1)
        span = spans[-1]
        self.assertEqual(span.kind, tracing.KIND_LLM)
        self.assertEqual(span.status, tracing.STATUS_OK)
        self.assertEqual(span.attributes['model'], 'jev-test')
        self.assertEqual(span.attributes['choice'], 'all_rows')
        self.assertIn('latency_ms', span.attributes)
        # The verdict, never the request: nothing of the question or evidence.
        self.assertNotIn('secret list', json.dumps(span.attributes, default=str))
        self.assertNotIn('Person 0', json.dumps(span.attributes, default=str))

    def test_a_failed_routing_call_is_an_error_span_and_no_host_means_no_span(self):
        sink = self._RecordingSink()
        host = SimpleNamespace(trace_sink=sink, current_turn_key='turn-1')
        router = SearchRouter(self._DecisionClient(error=TimeoutError()), questions={})
        router.route('Who?', '', self.LISTING, host=host)
        failed = [s for s in sink.spans if s.name == tracing.SPAN_SEARCH_ROUTE][-1]
        self.assertEqual(failed.status, tracing.STATUS_ERROR)
        self.assertEqual(failed.attributes['error_type'], 'TimeoutError')
        before = len(sink.spans)
        router.route('Who?', '', self.LISTING)
        self.assertEqual(len(sink.spans), before)

    def test_routing_is_opt_in(self):
        base = {ROUTER_ENV: '', ROUTER_KEY_ENV: 'a-key-present-for-something-else'}
        with patch.dict(os.environ, base), patch.dict('fastworkflow._env_vars', {}, clear=True):
            self.assertIsNone(router_for_workflow('opt-in-check'))
        with patch.dict(os.environ, {ROUTER_ENV: 'jev', ROUTER_KEY_ENV: ''}), \
                patch.dict('fastworkflow._env_vars', {}, clear=True):
            self.assertIsNone(router_for_workflow('opt-in-check'))

    def test_an_ambiguous_or_miscounted_listing_is_refused(self):
        split_label = ("identity_uid  label\naaa  Alice\nbbb  Bob  (contractor)  x\nccc  Carol\n")
        self.assertIsNone(parse_table(split_label))
        miscounted = "5 holder(s); shown=5\nidentity_uid  label\naaa  Alice\nbbb  Bob\n"
        self.assertIsNone(parse_table(miscounted))

    def test_markdown_and_tabbed_listings_are_read(self):
        md = parse_table("| uid | name |\n|---|---|\n| a | Alice |\n| b | Bob |\nafter\n")
        self.assertEqual((md['shape'], len(md['rows'])), ('markdown', 2))
        tabbed = parse_table("uid\tname\na\tAlice\nb\tBob\n")
        self.assertEqual((tabbed['shape'], len(tabbed['rows'])), ('tabbed', 2))
        self.assertIsNone(parse_table("| uid | name |\n|---|---|\n| a | Alice | extra |\n"))

    def test_only_optional_non_selecting_inputs_are_offered_for_narrowing(self):
        inputs = [
            {'name': 'filter', 'type': 'typing.Optional[str]', 'description': 'narrow to a name'},
            {'name': 'identity_uid', 'type': 'typing.Optional[str]', 'description': 'open one',
             'available_from': "['list_identities']"},
            {'name': 'account_uid', 'type': "<class 'str'>", 'description': 'required'},
        ]
        self.assertEqual(narrowing_inputs('list_identities', lambda _c: inputs),
                         'filter: narrow to a name')
        self.assertEqual(narrowing_inputs('x', None), NO_NARROWING)
        self.assertEqual(narrowing_inputs('x', lambda _c: 1 / 0), NO_NARROWING)
