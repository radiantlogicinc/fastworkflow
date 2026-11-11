"""Deterministic guards on the plan-review reader's edits, from recorded review exchanges.

Each ``RECORDED`` case is a real exchange from an internal benchmark (passes 8 and 9, observability DB): the
sub-task proposal shown, the user's reply word for word, and the edits the reader model returned for it (the
``ProposedEdit`` list of the ``fw.llm.call`` span). The tests run them through the real ``_proposed_edits_to_engine``
and ``apply_edits``; no model is called and nothing is mocked. Goals are shortened; inputs, step order and replies are
as recorded. Before the guards, ``r4_5_08_pass9`` lost three steps and a user who asked to remove the risk and
whole-list settings had whole steps dropped.
"""
from __future__ import annotations

import pytest

from fastworkflow import review_engine as re_
from fastworkflow.review_engine import EditRefused, Verdict, apply_edits
from fastworkflow.review_proposals import (
    Means, ProposedEdit, SubTask, SubTaskInput, SubTaskProposal, _proposed_edits_to_engine, merge_subtask_input_edit)

RECORDED = {'r4_5_08_pass9': {'id': 99,
                   'steps': [('Quarterly packet for Henderson Ramirez and Lewis',
                              'unit=Henderson Ramirez and Lewis Ltd 1, with_login_only=true',
                              'planning:department_quarterly_review'),
                             ('Audit Practice Fusion_Nurse Practitioner across ',
                              'right=Practice Fusion_Nurse Practitioner, include_risk=true',
                              'planning:cross_system_privilege_audit'),
                             ('Audit Allscripts Sunrise_Medical Records Special',
                              'right=Allscripts Sunrise_Medical Records Specialist',
                              'planning:cross_system_privilege_audit'),
                             ("Work the finding of the control 'Active identiti",
                              'control=Active identities who are contractors without a departure date, '
                              'family=identity, open_first_person=true, whole_list=true',
                              'planning:control_failure_investigation'),
                             ('Franklin Smith has asked for the right DrChrono_',
                              'right=DrChrono_Medical Records Specialist 1, person=Franklin Smith',
                              'planning:access_request_triage')],
                   'reply': 'Mostly right. In step 2 I did not ask for risk, so drop it. In step 4 I only want the '
                            'first identity the control names, not the whole list. Step 1 should just cover the first '
                            'three people on the roster; drop the extra "second person" sentence.',
                   'edits': [{'op': 'drop', 'index': 2, 'indexes': [3]},
                             {'op': 'drop', 'index': 5},
                             {'op': 'change',
                              'index': 1,
                              'field': 'inputs',
                              'value': 'unit=Henderson Ramirez and Lewis Ltd 1,with_login_only=true; means: '
                                       'planning:department_quarterly_review; desired outcome: not stated'},
                             {'op': 'drop', 'index': 4}]},
 'p8_r1_5_08': {'id': 56,
                'steps': [('Quarterly packet for Henderson Ramirez and Lewis',
                           'unit=Henderson Ramirez and Lewis Ltd 1, with_login_only=true',
                           'planning:department_quarterly_review'),
                          ('Audit Practice Fusion_Nurse Practitioner across ',
                           'right=Practice Fusion_Nurse Practitioner, include_risk=true',
                           'planning:cross_system_privilege_audit'),
                          ('Audit Allscripts Sunrise_Medical Records Special',
                           'right=Allscripts Sunrise_Medical Records Specialist',
                           'planning:cross_system_privilege_audit'),
                          ("Work the finding of the control 'Active identiti",
                           'control=Active identities who are contractors without a departure date, family=identity, '
                           'open_first_person=true, whole_list=true',
                           'planning:control_failure_investigation'),
                          ('Franklin Smith has asked for the right DrChrono_',
                           'right=DrChrono_Medical Records Specialist 1, person=Franklin Smith',
                           'planning:access_request_triage')],
                'reply': 'Mostly right. In step 2 I did not ask for risk, so drop the risk setting. In step 4 I only '
                         'want the first identity the finding names, not the whole list, so drop the whole-list '
                         'setting.',
                'edits': [{'op': 'drop', 'index': 2, 'indexes': [1], 'field': 'inputs', 'value': 'risk'},
                          {'op': 'drop', 'index': 4, 'indexes': [3], 'field': 'inputs', 'value': 'whole_list'}]},
 'p8_brandon': {'id': 143,
                'steps': [('Audit Active Directory_Cloud Administrator where',
                           'right=Active Directory_Cloud Administrator',
                           'planning:cross_system_privilege_audit'),
                          ('Audit Active Directory_Compliance Officer wherev',
                           'right=Active Directory_Compliance Officer',
                           'planning:cross_system_privilege_audit'),
                          ("Alan Cooper: down to Alan Cooper's logins, right",
                           'name=Alan Cooper, kind=person',
                           'planning:inspect_entity'),
                          ("Alisha Ochoa: down to Alisha Ochoa's logins, rig",
                           'name=Alisha Ochoa, kind=person',
                           'planning:inspect_entity'),
                          ("Anna Garcia: down to Anna Garcia's logins, right",
                           'name=Anna Garcia, kind=person',
                           'planning:inspect_entity'),
                          ("Barbara Sanchez: down to Barbara Sanchez's login",
                           'name=Barbara Sanchez, kind=person',
                           'planning:inspect_entity'),
                          ("Brandon Miller: down to Brandon Miller's logins,",
                           'name=Miller, kind=person',
                           'planning:inspect_entity'),
                          ("Work the open finding of the control 'Active con",
                           'control=Active contractor identities whom manager left, family=identity, '
                           'open_first_person=true, whole_list=true',
                           'planning:control_failure_investigation'),
                          ('Christopher Hubbard leaves on Friday, so Christo',
                           'person=Christopher Hubbard',
                           'planning:leaver_offboarding_sweep')],
                'reply': "In step 7 the name should be 'Brandon Miller', not just Miller. In step 8 I only want the "
                         'first identity the finding names, so drop the whole-list setting. The rest is fine.',
                'edits': [{'op': 'change', 'index': 7, 'field': 'name', 'value': 'Brandon Miller'},
                          {'op': 'drop', 'index': 8, 'indexes': [8]}]},
 'p8_wrong_index_drop': {'id': 145,
                         'steps': [('Aaron Garrison is leaving: what Aaron Garrison c',
                                    'person=Aaron Garrison',
                                    'planning:leaver_offboarding_sweep'),
                                   ('Alexander Herrera is leaving: what Alexander Her',
                                    'person=Alexander Herrera',
                                    'planning:leaver_offboarding_sweep'),
                                   ('David Farrell is leaving: what David Farrell can',
                                    'person=David Farrell',
                                    'planning:leaver_offboarding_sweep'),
                                   ('Amy Walsh is leaving: what Amy Walsh can reach, ',
                                    'person=Amy Walsh',
                                    'planning:leaver_offboarding_sweep'),
                                   ('April Watts is leaving: what April Watts can rea',
                                    'person=April Watts',
                                    'planning:leaver_offboarding_sweep'),
                                   ('Quarterly review of Gardner Robinson and Lawrenc',
                                    'unit=Gardner Robinson and Lawrence LLC 2, with_login_only=true',
                                    'planning:department_quarterly_review'),
                                   ("Work the finding for the control 'Identities who",
                                    'control=Identities who are contractors set as inactive or whose departure date '
                                    'has passed and owning active accounts, family=identity, open_first_person=true, '
                                    'whole_list=true',
                                    'planning:control_failure_investigation')],
                         'reply': 'Steps 1 to 6 are fine. In step 7 I only want the first identity the finding names, '
                                  'not the whole list, so please drop the whole-list setting.',
                         'edits': [{'op': 'drop', 'index': 1, 'value': 'whole_list'}]},
 'p9_swapped_steps': {'id': 64,
                      'steps': [('Quarterly packet for Henderson Ramirez and Lewis',
                                 'unit=Henderson Ramirez and Lewis Ltd 1, with_login_only=true',
                                 'planning:department_quarterly_review'),
                                ('Audit Practice Fusion_Nurse Practitioner across ',
                                 'right=Practice Fusion_Nurse Practitioner, include_risk=true',
                                 'planning:cross_system_privilege_audit'),
                                ('Audit Allscripts Sunrise_Medical Records Special',
                                 'right=Allscripts Sunrise_Medical Records Specialist',
                                 'planning:cross_system_privilege_audit'),
                                ("Work the finding of the control 'Active identiti",
                                 'control=Active identities who are contractors without a departure date, '
                                 'family=identity, open_first_person=true, whole_list=true',
                                 'planning:control_failure_investigation'),
                                ('Franklin Smith has asked for the right DrChrono_',
                                 'right=DrChrono_Medical Records Specialist 1, person=Franklin Smith',
                                 'planning:access_request_triage')],
                      'reply': 'In step 2 I did not ask for risk scores, so leave those out. In step 4 I only want the '
                               'first identity the finding names, not the whole list of identities.',
                      'edits': [{'op': 'change',
                                 'index': 1,
                                 'field': 'inputs',
                                 'value': 'right=Practice Fusion_Nurse Practitioner,include_risk=false'},
                                {'op': 'change',
                                 'index': 3,
                                 'field': 'inputs',
                                 'value': 'control=Active identities who are contractors without a departure '
                                          'date,family=identity,open_first_person=true,whole_list=false'}]},
 'p8_unit_other_step': {'id': 91,
                        'steps': [('Mcclain Miller and Henderson and Sons 1 is being',
                                   'unit=Henderson and Sons 1, scope=every',
                                   'planning:department_roster_walk'),
                                  ('Angelica Schneider is one of the two who do not ',
                                   'person=Angelica Schneider',
                                   'planning:leaver_offboarding_sweep'),
                                  ('Christopher Walker is one of the two who do not ',
                                   'person=Christopher Walker',
                                   'planning:leaver_offboarding_sweep')],
                        'reply': "In step 1 the unit name is cut short. It should be the full 'Mcclain Miller and "
                                 "Henderson and Sons 1'. Steps 2 and 3 are fine.",
                        'edits': [{'op': 'change',
                                   'index': 2,
                                   'field': 'inputs',
                                   'value': 'unit=Henderson and Sons 1'}]},
 'drop_goes_to_named_step': {'id': 68,
                             'steps': [("Which controls does 'Henderson Ramirez and Lewis",
                                        'name=Henderson Ramirez and Lewis and Sons, kind=unit, fact=controls',
                                        'planning:record_fact'),
                                       ('Which controls does Lewis and Sons trip?',
                                        'name=Lewis and Sons, kind=unit, fact=controls',
                                        'planning:record_fact')],
                             'reply': 'Step 2 should not be there; please drop step 2. Step 1 is right.',
                             'edits': [{'op': 'drop', 'index': 1, 'indexes': [2], 'field': 'goal'}]},
 'explicit_drop_stays': {'id': 161,
                         'steps': [("Which controls does 'Henderson Ramirez and Lewis",
                                    'name=Henderson Ramirez and Lewis and Sons, kind=person, fact=controls',
                                    'planning:record_fact'),
                                   ('Which controls does Lewis and Sons trip?',
                                    'name=Lewis and Sons, kind=unit, fact=controls',
                                    'planning:record_fact')],
                         'reply': "Step 1 is right, but remove step 2. 'Henderson Ramirez and Lewis and Sons' is one "
                                  'name, so there should only be the one step.',
                         'edits': [{'op': 'drop', 'index': 2, 'indexes': [2], 'field': 'true', 'value': 'true'}]},
 'explicit_drop_one': {'id': 107,
                       'steps': [('Explain why the control on the group Allscripts ',
                                  'control=Allscripts Sunrise_Clinical Manager, control_is=name, tags=true',
                                  'planning:explain_finding'),
                                 ('Show the findings the group Allscripts Sunrise_C',
                                  'name=Allscripts Sunrise_Clinical Manager, kind=group',
                                  'planning:record_findings_remediation')],
                       'reply': 'Remove step 1; I never asked why a control was flagged. Keep step 2 as shown.',
                       'edits': [{'op': 'drop', 'index': 1}]},
 'explicit_multi_drop': {'id': 167,
                         'steps': [("HR has sent me five names for this month's offbo",
                                    'unit=Gardner Robinson and Lawrence LLC 2, with_login_only=true',
                                    'planning:department_quarterly_review'),
                                   ("HR has sent me five names for this month's offbo",
                                    'unit=Gardner Robinson and Lawrence LLC 2, with_login_only=true',
                                    'planning:department_quarterly_review'),
                                   ("HR has sent me five names for this month's offbo",
                                    'unit=Gardner Robinson and Lawrence LLC 2, people=5, with_login_only=true',
                                    'planning:department_quarterly_review'),
                                   ("HR has sent me five names for this month's offbo",
                                    'unit=Gardner Robinson and Lawrence LLC 2, with_login_only=true',
                                    'planning:department_quarterly_review'),
                                   ("HR has sent me five names for this month's offbo",
                                    'unit=Gardner Robinson and Lawrence LLC 2, with_login_only=true',
                                    'planning:department_quarterly_review'),
                                   ('Aaron Garrison: what Aaron Garrison can reach, t',
                                    'person=Aaron Garrison',
                                    'planning:leaver_offboarding_sweep'),
                                   ('Alexander Herrera: what Alexander Herrera can re',
                                    'person=Alexander Herrera',
                                    'planning:leaver_offboarding_sweep'),
                                   ('David Farrell: what David Farrell can reach, the',
                                    'person=David Farrell',
                                    'planning:leaver_offboarding_sweep'),
                                   ('Amy Walsh: what Amy Walsh can reach, their first',
                                    'person=Amy Walsh',
                                    'planning:leaver_offboarding_sweep'),
                                   ('April Watts: what April Watts can reach, their f',
                                    'person=April Watts',
                                    'planning:leaver_offboarding_sweep'),
                                   ("Work the finding for the control 'Identities who",
                                    'control=Identities who are contractors set as inactive or whose departure date '
                                    'has passed and owning active accounts, family=identity, open_first_person=true, '
                                    'whole_list=true',
                                    'planning:control_failure_investigation')],
                         'reply': "Still wrong: the unit review of 'Gardner Robinson and Lawrence LLC 2' appears five "
                                  'times. Keep only the first one, drop the next four unit reviews, and it should '
                                  'cover three people, not five. Offboarding steps and the control step are right.',
                         'edits': [{'op': 'drop', 'index': 2},
                                   {'op': 'drop', 'index': 3},
                                   {'op': 'drop', 'index': 4},
                                   {'op': 'drop', 'index': 5},
                                   {'op': 'change', 'index': 1, 'field': 'inputs', 'value': 'people=3'}]},
 'drop_and_rename': {'id': 106,
                     'steps': [('Which controls does Henderson Ramirez trip?',
                                'name=Henderson Ramirez, kind=right, fact=controls',
                                'planning:record_fact'),
                               ('Which controls does Lewis and Sons trip?',
                                'name=Lewis and Sons, kind=unit, fact=controls',
                                'planning:record_fact')],
                     'reply': "Remove step 2, and change step 1's name to 'Henderson Ramirez and Lewis and Sons' in "
                              'full.',
                     'edits': [{'op': 'drop', 'index': 2},
                               {'op': 'change',
                                'index': 1,
                                'field': 'goal',
                                'value': 'Which controls does Henderson Ramirez and Lewis and Sons trip?'}]},
 'unasked_drop_refused': {'id': 27,
                          'steps': [('AzureDevOps_Administrator is on my privileged-gr',
                                     'name=AzureDevOps_Administrator, kind=group',
                                     'planning:inspect_entity'),
                                    ('AzureDevOps_Administrator is on my privileged-gr',
                                     'name=AzureDevOps_Administrator, kind=group, fact=controls',
                                     'planning:record_fact'),
                                    ('AzureDevOps_Administrator is on my privileged-gr',
                                     'name=AzureDevOps_Administrator, kind=group',
                                     'planning:inspect_entity')],
                          'reply': 'Step 3 just repeats step 1. Change step 3 to list the members of the group '
                                   'AzureDevOps_Administrator, meaning who is in it.',
                          'edits': [{'op': 'drop', 'index': 2}]},
 'negated_quote': {'id': 94,
                   'steps': [('Audit GitLab_Infrastructure Manager wherever the',
                              'right=GitLab_Infrastructure Manager',
                              'planning:cross_system_privilege_audit'),
                             ('Audit GitLab_Backup Administrator wherever they ',
                              'right=GitLab_Backup Administrator',
                              'planning:cross_system_privilege_audit'),
                             ("Work the open finding of the control 'Active ide",
                              'control=Active identities who are contractors with past ending date and no active '
                              'account, family=identity, open_first_person=true, whole_list=true',
                              'planning:control_failure_investigation'),
                             ('Adam Fitzgerald is leaving the unit: what Adam F',
                              'person=Adam Fitzgerald',
                              'planning:leaver_offboarding_sweep'),
                             ('Elizabeth Calderon is leaving the unit: what Eli',
                              'person=Elizabeth Calderon',
                              'planning:leaver_offboarding_sweep'),
                             ('Walk IT Operations end to end — every identity o',
                              'unit=IT Operations end to end, scope=every',
                              'planning:department_roster_walk'),
                             ('Show the GitLab_Technical Leads collection: who ',
                              'name=GitLab_Technical Leads, kind=group',
                              'planning:inspect_entity')],
                   'reply': "Everything is fine except step 6: the unit name should be just 'IT Operations', without "
                            "'end to end'.",
                   'edits': [{'op': 'change',
                              'index': 6,
                              'indexes': [6],
                              'field': 'inputs',
                              'value': 'unit=IT Operations, scope=every'}]},
 'current_wrong_quote': {'id': 96,
                         'steps': [('Audit GitLab_Infrastructure Manager wherever the',
                                    'right=GitLab_Infrastructure Manager',
                                    'planning:cross_system_privilege_audit'),
                                   ('Audit GitLab_Backup Administrator wherever they ',
                                    'right=GitLab_Backup Administrator',
                                    'planning:cross_system_privilege_audit'),
                                   ("Work the open finding of the control 'Active ide",
                                    'control=Active identities who are contractors with past ending date and no active '
                                    'account, family=identity, open_first_person=true, whole_list=true',
                                    'planning:control_failure_investigation'),
                                   ('Adam Fitzgerald is leaving the unit: what Adam F',
                                    'person=Adam Fitzgerald',
                                    'planning:leaver_offboarding_sweep'),
                                   ('Elizabeth Calderon is leaving the unit: what Eli',
                                    'person=Elizabeth Calderon',
                                    'planning:leaver_offboarding_sweep'),
                                   ('Walk IT Operations end to end — every identity o',
                                    'unit=end to end, scope=every',
                                    'planning:department_roster_walk'),
                                   ('Show the GitLab_Technical Leads collection: who ',
                                    'name=GitLab_Technical Leads, kind=group',
                                    'planning:inspect_entity')],
                         'reply': "Step 6 now shows the unit as 'end to end'. The unit should be 'IT Operations'. "
                                  'Everything else is fine.',
                         'edits': [{'op': 'change',
                                    'index': 6,
                                    'field': 'inputs',
                                    'value': 'unit=IT Operations, scope=every'}]},
 'negated_quote_two': {'id': 192,
                       'steps': [('Walk Blake and Sons and Sons and Sons: every per',
                                  'unit=Sons, scope=every',
                                  'planning:department_roster_walk'),
                                 ('Ian Morales has asked for the right Confluence_S',
                                  'right=Confluence_System Analyst 2, person=Ian Morales',
                                  'planning:access_request_triage')],
                       'reply': "In step 1 the unit name is wrong. It should be the unit 'Blake and Sons and Sons', "
                                "not 'Blake and Sons and Sons and Sons' or just Sons. Step 2 is fine.",
                       'edits': [{'op': 'change',
                                  'index': 1,
                                  'field': 'unit',
                                  'value': 'Blake and Sons and Sons',
                                  'parts': ['Sub-tasks:\n'
                                            '1. Walk Blake and Sons and Sons: every person on the roster down to their '
                                            'logins, rights and collections. (inputs: unit=Blake and Sons and Sons, '
                                            'scope=every; means: planning:department_roster_walk); desired outcome: '
                                            'not stated']}]},
 'said_literally': {'id': 200,
                    'steps': [('Who owns GitLab', 'name=GitLab, kind=application, fact=owner', 'planning:record_fact'),
                              ('which rights does it publish. How sensitive is G',
                               'application=GitLab_Infrastructure Manager, compare_rights=true',
                               'planning:application_recertification')],
                    'reply': "In step 2, change the application to GitLab and the right to 'GitLab_Infrastructure "
                             "Manager'.",
                    'edits': [{'op': 'change', 'index': 2, 'field': 'application', 'value': 'GitLab'},
                              {'op': 'change',
                               'index': 2,
                               'field': 'right',
                               'value': 'GitLab_Infrastructure Manager'}]},
 'clear_none': {'id': 15,
                'steps': [('I have just taken over the tenant you are co',
                           'kind=control, inputs.kind=control',
                           'planning:list_records')],
                'reply': 'The step still shows kind control. Please clear that kind setting entirely so it covers '
                         'every kind of record, not just controls.',
                'edits': [{'op': 'change', 'index': 1, 'field': 'kind', 'value': 'None'}]},
 'kind_change': {'id': 12,
                 'steps': [('Tag AzureDevOps_Contractors as `privileged`.',
                            'name=AzureDevOps_Contractors, kind=login, tag=privileged, action=add',
                            'planning:tag_record')],
                 'reply': 'AzureDevOps_Contractors is a group, not a login. Please change the kind to group; the tag '
                          'and the add are right.',
                 'edits': [{'op': 'change', 'index': 1, 'field': 'kind', 'value': 'group'}]},
 'split': {'id': 201,
           'steps': [('Recertify GitLab: its owner.', 'application=GitLab', 'planning:application_recertification'),
                     ('Recertify GitLab: the logins on it. Recertify Gi',
                      'application=GitLab',
                      'planning:application_recertification'),
                     ('Recertify GitLab: the GitLab_Administrators grou',
                      'application=GitLab, group=GitLab_Administrators',
                      'planning:application_recertification'),
                     ("Recertify GitLab: the application's risk.",
                      'application=GitLab',
                      'planning:application_recertification')],
           'reply': 'Step 2 merges two things. Split it: one step for the logins on GitLab, and a separate step for '
                    'the rights it publishes with the first opened on its sensitivity. The rest is fine.',
           'edits': [{'op': 'split',
                      'index': 2,
                      'parts': ['Recertify GitLab: the logins on it.',
                                'Recertify GitLab: the rights it publishes with the first opened on its '
                                'sensitivity.']}]}}


def _proposal(case: dict) -> SubTaskProposal:
    items = []
    for goal, inputs, means in case["steps"]:
        pieces = [SubTaskInput.from_text(text) for text in inputs.split(", ") if text] if inputs else []
        items.append(SubTask(goal=goal, inputs=pieces, means=Means.from_text(means)))
    return SubTaskProposal(items=items)


def _apply(name: str) -> SubTaskProposal:
    case = RECORDED[name]
    proposal = _proposal(case)
    edits = _proposed_edits_to_engine(
        proposal, [ProposedEdit(**each) for each in case["edits"]], reply_text=case["reply"])
    return apply_edits(proposal, edits)


def _inputs(proposal: SubTaskProposal, step: int) -> dict[str, str]:
    return {each.name: each.value for each in proposal.items[step - 1].inputs if each.name}


def test_the_recorded_r4_reply_clears_two_settings_and_loses_no_step():
    # benchmark pass 9, task 5.08, round 4 (span d59f2737...): the reader returned op=drop on steps 2, 4 and 5.
    applied = _apply("r4_5_08_pass9")
    assert len(applied.items) == 5
    assert "include_risk" not in _inputs(applied, 2)
    assert _inputs(applied, 2)["right"] == "Practice Fusion_Nurse Practitioner"
    assert "whole_list" not in _inputs(applied, 4)
    assert _inputs(applied, 4)["open_first_person"] == "true"
    assert _inputs(applied, 5)["person"] == "Franklin Smith"


def test_a_reader_drop_that_names_a_setting_clears_the_setting_not_the_step():
    # pass 8, task 5.08, round 1: "drop the risk setting" and "drop the whole-list setting" read as drop steps 2 and 4.
    applied = _apply("p8_r1_5_08")
    assert len(applied.items) == 5
    assert "include_risk" not in _inputs(applied, 2)
    assert "whole_list" not in _inputs(applied, 4)


def test_a_setting_cleared_in_one_step_and_a_name_fixed_in_another_are_both_applied():
    applied = _apply("p8_brandon")
    assert len(applied.items) == 9
    assert _inputs(applied, 7)["name"] == "Brandon Miller"
    assert "whole_list" not in _inputs(applied, 8)


def test_a_drop_the_reply_aims_at_a_setting_of_a_later_step_does_not_remove_step_one():
    applied = _apply("p8_wrong_index_drop")
    assert len(applied.items) == 7
    assert _inputs(applied, 1)["person"] == "Aaron Garrison"
    assert "whole_list" not in _inputs(applied, 7)


def test_input_changes_the_reader_put_on_the_wrong_steps_go_to_the_steps_the_reply_names():
    applied = _apply("p9_swapped_steps")
    assert "include_risk" not in _inputs(applied, 1) and "right" not in _inputs(applied, 1)
    assert _inputs(applied, 2)["include_risk"] == "false"
    assert "control" not in _inputs(applied, 3)
    assert _inputs(applied, 4)["whole_list"] == "false"


def test_a_unit_fix_on_the_wrong_step_goes_to_the_step_that_has_the_unit():
    applied = _apply("p8_unit_other_step")
    assert _inputs(applied, 1)["unit"] == "Mcclain Miller and Henderson and Sons 1"
    assert "unit" not in _inputs(applied, 2)


def test_a_drop_of_the_wrong_step_goes_to_the_step_the_reply_asks_to_remove():
    applied = _apply("drop_goes_to_named_step")
    assert len(applied.items) == 1
    assert _inputs(applied, 1)["name"] == "Henderson Ramirez and Lewis and Sons"


def test_a_drop_the_reply_does_not_ask_for_is_refused_with_a_question():
    with pytest.raises(EditRefused, match="did not ask"):
        _apply("unasked_drop_refused")


@pytest.mark.parametrize("name,steps_left", [
    ("explicit_drop_stays", 1), ("explicit_drop_one", 1), ("explicit_multi_drop", 7), ("drop_and_rename", 1),
])
def test_drops_the_reply_asks_for_still_apply(name, steps_left):
    assert len(_apply(name).items) == steps_left


def test_a_name_the_reply_rules_out_is_not_put_back():
    assert _inputs(_apply("negated_quote"), 6)["unit"] == "IT Operations"
    assert _inputs(_apply("current_wrong_quote"), 6)["unit"] == "IT Operations"
    assert _inputs(_apply("negated_quote_two"), 1)["unit"] == "Blake and Sons and Sons"


def test_a_value_the_user_wrote_is_not_replaced_by_a_longer_quoted_name():
    applied = _apply("said_literally")
    assert _inputs(applied, 2)["application"] == "GitLab"
    assert _inputs(applied, 2)["right"] == "GitLab_Infrastructure Manager"


def test_the_reader_saying_none_for_an_input_clears_that_input():
    assert "kind" not in _inputs(_apply("clear_none"), 1)


def test_ordinary_recorded_corrections_are_unchanged():
    assert _inputs(_apply("kind_change"), 1)["kind"] == "group"
    assert len(_apply("split").items) == 5


def test_name_equals_none_clears_one_input_and_inputs_none_clears_all():
    existing = [SubTaskInput(name="unit", value="X"), SubTaskInput(name="scope", value="every")]
    assert [each.name for each in merge_subtask_input_edit(existing, "scope=none")] == ["unit"]
    assert merge_subtask_input_edit(existing, "none") == []
    assert [each.value for each in merge_subtask_input_edit(existing, "unit=Y")] == ["Y", "every"]


@pytest.mark.parametrize("reply", [
    "Yes, go ahead, but do not run any of the actions.",     # pass 9 round 2 (the reader read it as "no edit")
    "Yes, go ahead, but do not apply anything.",
    "yes, run that, and apply no corrections",
    "Yes, go ahead. Just do not apply any corrections.",
])
def test_a_yes_with_a_no_effect_qualifier_is_accepted(reply):
    assert re_._affirmative_accept_cue(reply)
    proposal = SubTaskProposal(items=[SubTask(goal="Which remediation actions exist for Ann?")])
    decision = re_.ProposalReplyPolicy().decide(proposal, reply)
    assert decision.verdict == Verdict.CONFIRM


@pytest.mark.parametrize("reply", [
    "yes, but do not run step 2",
    "Yes, go ahead, but do not run any of them.",
    "Yes, but also add a step that lists the rights.",
    "yes but don't change step 3",
    "Yes, Confluence is right, but the step must specifically check for orphaned accounts on it.",
])
def test_a_yes_that_asks_for_an_edit_is_not_accepted(reply):
    assert not re_._affirmative_accept_cue(reply)
