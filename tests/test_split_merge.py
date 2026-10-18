"""split_merge.merge_subtasks: the deterministic clean-up after the splitter's model."""
import re

from fastworkflow.split_merge import merge_subtasks, request_subjects, subjects_in


# ------------------------------------------------------------------ subjects
CASE_IDS = (re.compile(r"\bcase_[A-Za-z0-9]+"),)


def test_subjects_are_names_quotes_ticks_and_declared_identifiers_not_sentence_openers():
    q = ("The control 'Stale cases whose owner left' names Nora Banks. "
         "Then tag `on-leave` on case_X7kQ2Lm9PqR4sT. Open the NORTHFEED feed.")
    assert set(request_subjects(q, CASE_IDS)) == {
        "Stale cases whose owner left", "Nora Banks", "on-leave",
        "case_X7kQ2Lm9PqR4sT", "NORTHFEED"}


def test_an_identifier_is_a_subject_only_when_the_workflow_declares_its_pattern():
    q = "Explain case_X7kQ2Lm9PqR4sT and case_B4nW8Rt2Yy6uPq."
    assert request_subjects(q) == []
    assert set(request_subjects(q, CASE_IDS)) == {"case_X7kQ2Lm9PqR4sT", "case_B4nW8Rt2Yy6uPq"}
    split = ["Explain case_X7kQ2Lm9PqR4sT.", "Explain case_B4nW8Rt2Yy6uPq."]
    assert merge_subtasks(q, split) == [q]             # no subjects: one job
    assert merge_subtasks(q, split, CASE_IDS) == split  # two declared identifiers: two jobs


def test_a_chain_ending_on_a_company_suffix_is_one_name():
    assert request_subjects("How risky is Hart Mills and Grey Ltd as a unit?") == [
        "Hart Mills and Grey Ltd"]
    assert request_subjects("Which controls does Reed Ortiz and Nolan and Sons trip?") == [
        "Reed Ortiz and Nolan and Sons"]
    assert set(request_subjects("Compare Ivan Moss and Ella Nolan.")) == {
        "Ivan Moss", "Ella Nolan"}


def test_weekdays_months_and_sentence_openers_are_not_names():
    assert request_subjects("Leo Grant leaves on Friday. Show me everything he can reach.") == [
        "Leo Grant"]
    assert request_subjects("April Lowe is a contractor.") == ["April Lowe"]
    assert request_subjects("Who is on the IT Support roster?") == ["IT Support"]


def test_a_part_of_a_long_name_stands_for_it_unless_another_name_shares_it():
    known = request_subjects("How risky is Hart Mills and Grey Ltd as a unit?")
    assert subjects_in("How risky is Hart Mills?", known) == {"hart mills and grey ltd"}
    assert subjects_in("How risky is Grey Ltd?", known) == {"hart mills and grey ltd"}
    known = ["Ruby Fox", "Ruby Kerr"]
    assert subjects_in("Show Ruby's logins.", known) == {"ruby"}


# ------------------------------------------------------------------ rule 3: no new subject
def test_a_follow_up_about_the_same_record_is_rejoined_to_the_whole_request():
    q = "How risky is Zoe Hale, and which findings are driving that?"
    assert merge_subtasks(q, ["How risky is Zoe Hale?", "Which findings are driving that?"]) == [q]
    q = "What is BuildOps_Technical Lead nested under, and what does it contain?"
    assert merge_subtasks(q, ["What is BuildOps_Technical Lead nested under?",
                              "What does BuildOps_Technical Lead contain?"]) == [q]


def test_a_trailing_instruction_is_rejoined_and_the_qualifier_restored():
    q = ("Owen Pike is supposed to have a separate admin login. Find the accounts on file under that "
         "name. The matches are enough.")
    assert merge_subtasks(q, ["Find the accounts on file under Owen Pike.", "The matches are enough."]) == [q]
    q = ("April Lowe is an active contractor. Show me the findings on April Lowe and the corrections that "
         "would close them. Do not apply anything.")
    assert merge_subtasks(q, ["Show me the findings on April Lowe.",
                              "Show me the corrections that would close them for April Lowe."]) == [q]


def test_a_business_unit_split_at_its_and_is_rejoined():
    q = "Which controls does Reed Ortiz and Nolan and Sons trip?"
    assert merge_subtasks(q, ["Which controls does Reed Ortiz trip?",
                              "Which controls does Nolan and Sons trip?"]) == [q]


def test_a_part_naming_its_own_person_stays_separate():
    q = ("Audit the right Core Gateway_Network Administrator: holders, the publishing system and the feed. "
         "While you are on it, confirm whether Iris Vance and Lena Marsh are among its holders.")
    split = ["Audit the right Core Gateway_Network Administrator: holders, the publishing system and the feed.",
             "Confirm whether Iris Vance is among its holders.",
             "Confirm whether Lena Marsh is among its holders."]
    assert merge_subtasks(q, split) == split


def test_a_subject_less_part_joins_the_part_before_it_only():
    q = ("Work the control 'Cases without owner' through. Then put Nora Banks through an "
         "offboarding sweep. Apply none of them.")
    assert merge_subtasks(q, ["Work the control 'Cases without owner' through.",
                              "Put Nora Banks through an offboarding sweep.",
                              "Apply none of them."]) == [
        "Work the control 'Cases without owner' through.",
        "Put Nora Banks through an offboarding sweep. Apply none of them."]


def test_different_subject_sets_are_not_merged_even_when_one_holds_the_other():
    q = ("PayHub recertification: the PayHub_Administrators group and the application's risk. "
         "Then list the PayHub logins that no person owns.")
    split = ["Recertify PayHub: the PayHub_Administrators group and the application's risk.",
             "List the PayHub logins that no person owns."]
    assert merge_subtasks(q, split) == split


# ------------------------------------------------------------------ rules 1, 2: meta lines, contained duplicates
def test_meta_lines_and_contained_duplicates_are_dropped():
    q = ("The control 'Cases without owner' names Nora Banks. Work the finding through. Then "
         "sweep Nora Banks.")
    assert merge_subtasks(q, ["Work the finding of the control 'Cases without owner' through.",
                              "Sweep Nora Banks.",
                              "(No tag action is requested; no sub-task is generated for a tag.)"]) == [
        "Work the finding of the control 'Cases without owner' through.", "Sweep Nora Banks."]
    q = "Leo Grant leaves on Friday: what he can reach and the corrections on offer. Apply none."
    assert merge_subtasks(q, ["What Leo Grant can reach and the corrections on offer, applying none.",
                              "What Leo Grant can reach and the corrections on offer."]) == [q]


# ------------------------------------------------------------------ rule 4: comparisons
def test_a_comparison_split_per_record_is_one_subtask_in_the_requests_words():
    q = ("List the logins nobody owns on NORTHFEED and on SOUTHFEED, with a count for each. Confirm whether the "
         "HR feed carries any logins at all. Then open the riskier of NORTHFEED and SOUTHFEED and show me the "
         "controls it trips.")
    split = ["List the logins nobody owns on NORTHFEED, with a count.",
             "List the logins nobody owns on SOUTHFEED, with a count.",
             "Confirm whether the HR feed carries any logins at all.",
             "Open NORTHFEED and show me the controls it trips.",
             "Open SOUTHFEED and show me the controls it trips."]
    assert merge_subtasks(q, split) == split[:3] + [
        "Open the riskier of NORTHFEED and SOUTHFEED and show me the controls it trips."]


def test_a_comparison_whose_names_come_a_sentence_earlier_takes_both_sentences():
    q = ("Ivan Moss and Ella Nolan each hold about ten rights. Compare them side by side: logins, "
         "rights and risk.")
    assert merge_subtasks(q, ["Compare Ivan Moss side by side: logins, rights and risk.",
                              "Compare Ella Nolan side by side: logins, rights and risk."]) == [q]


def test_per_person_jobs_without_a_comparison_stay_split():
    q = "Run the offboarding sweep for Ann Lee, Bo Chan and Cy Diaz."
    split = ["Run the offboarding sweep for Ann Lee.", "Run the offboarding sweep for Bo Chan.",
             "Run the offboarding sweep for Cy Diaz."]
    assert merge_subtasks(q, split) == split


# ------------------------------------------------------------------ edges
def test_fewer_than_two_subtasks_are_returned_unchanged():
    assert merge_subtasks("q", []) == []
    assert merge_subtasks("q", ["only one"]) == ["only one"]


# ------------------------------------------------------------------ rule 3: a name extending another name
def test_a_name_extending_another_with_a_back_reference_joins_its_group():
    q = ("Who owns PayHub, which rights does it publish, and how sensitive is PayHub_Infrastructure Manager "
         "among them?")
    assert merge_subtasks(q, ["Who owns PayHub?", "Which rights does PayHub publish?",
                              "How sensitive is PayHub_Infrastructure Manager among them?"]) == [q]


def test_a_child_name_without_a_back_reference_stays_separate():
    q = "Who owns PayHub? Separately, how sensitive is PayHub_Infrastructure Manager?"
    split = ["Who owns PayHub?", "How sensitive is PayHub_Infrastructure Manager?"]
    assert merge_subtasks(q, split) == split


def test_a_name_with_a_shared_prefix_but_no_separator_is_not_related():
    q = "Who owns Pay, and how sensitive is PayHub_Admin among them?"
    split = ["Who owns Pay?", "How sensitive is PayHub_Admin among them?"]
    assert merge_subtasks(q, split) == split


def test_the_joined_child_name_keeps_a_later_parent_only_job_apart():
    # 4.04's shape: the child group is part of the recertification; the later PayHub-only job is not
    q = ("PayHub recertification: owner, logins and rights, the PayHub_Administrators group with its members, "
         "and the application's risk. Then list the PayHub logins that no person owns.")
    split = ["Recertify PayHub: owner, logins and rights.",
             "Show the PayHub_Administrators group with its members, and the application's risk.",
             "List the PayHub logins that no person owns."]
    assert merge_subtasks(q, split) == [
        "Recertify PayHub: owner, logins and rights. Show the PayHub_Administrators group with its members, "
        "and the application's risk.",
        "List the PayHub logins that no person owns."]


# ------------------------------------------------------------------ rule 5: limits travel with the sub-tasks
def test_a_limit_pointing_back_goes_to_the_job_it_qualifies_only():
    q = ("The control 'Cases without owner' names Nora Banks. Work the finding through: why it "
         "was flagged. Then put Nora Banks through an offboarding sweep: what she can reach and the "
         "corrections available. Apply none of them.")
    assert merge_subtasks(q, [
        "Work the finding of the control 'Cases without owner': why it was flagged.",
        "Nora Banks leaves: what Nora Banks can reach and the corrections available, without "
        "applying any."]) == [
        "Work the finding of the control 'Cases without owner': why it was flagged.",
        "Nora Banks leaves: what Nora Banks can reach and the corrections available, without "
        "applying any. Apply none of them."]


def test_a_limit_pointing_at_nothing_goes_to_every_subtask():
    q = "Show me the logins of Ann Lee, Bo Chan and Cy Diaz. I do not need the full population."
    assert merge_subtasks(q, ["Show me the logins of Ann Lee.", "Show me the logins of Bo Chan.",
                              "Show me the logins of Cy Diaz."]) == [
        "Show me the logins of Ann Lee. I do not need the full population.",
        "Show me the logins of Bo Chan. I do not need the full population.",
        "Show me the logins of Cy Diaz. I do not need the full population."]


def test_a_limit_already_carried_is_not_repeated_and_a_dropped_one_is_given_back():
    q = "Sweep Ann Lee. Sweep Bo Chan. Do not apply anything."
    assert merge_subtasks(q, ["Sweep Ann Lee. Do not apply anything.", "Sweep Bo Chan."]) == [
        "Sweep Ann Lee. Do not apply anything.", "Sweep Bo Chan. Do not apply anything."]


def test_a_limit_clause_stays_with_its_own_sentence_job():
    q = "Show Ann Lee's findings, without opening any of them. Then sweep Bo Chan."
    assert merge_subtasks(q, ["Show Ann Lee's findings.", "Sweep Bo Chan."]) == [
        "Show Ann Lee's findings, without opening any of them.", "Sweep Bo Chan."]


def test_a_limit_naming_a_subject_goes_only_to_subtasks_naming_it():
    q = "Sweep Ann Lee. Sweep Bo Chan. Do not touch Bo Chan's tags."
    assert merge_subtasks(q, ["Sweep Ann Lee.", "Sweep Bo Chan."]) == [
        "Sweep Ann Lee.", "Sweep Bo Chan. Do not touch Bo Chan's tags."]


def test_reminders_and_questions_are_not_limits():
    q = "Sweep Ann Lee. Sweep Bo Chan. Don't forget to tag Cy Diaz."
    split = ["Sweep Ann Lee.", "Sweep Bo Chan.", "Tag Cy Diaz."]
    assert merge_subtasks(q, split) == split


def test_a_limit_clause_in_a_sentence_no_subtask_carries_goes_to_every_subtask():
    # the 2.05 shape (owner decision: every per-person sub-task carries the limit)
    q = ("Before the clean-up, check whether Ann Lee, Bo Chan and Cy Diaz each have access to Ledger. "
         "Answer per person; I do not need the full population.")
    assert merge_subtasks(q, ["Check whether Ann Lee has access to Ledger.",
                              "Check whether Bo Chan has access to Ledger.",
                              "Check whether Cy Diaz has access to Ledger."]) == [
        "Check whether Ann Lee has access to Ledger. I do not need the full population.",
        "Check whether Bo Chan has access to Ledger. I do not need the full population.",
        "Check whether Cy Diaz has access to Ledger. I do not need the full population."]


def test_a_check_on_a_change_naming_part_of_its_subjects_joins_the_change():
    q = ("Put `on-leave` on Mia Stone. Show me the tags on the record before and after, "
         "so I can see the change landed and nothing else moved.")
    for check in ("Show me the tags on Mia Stone before and after.",
                  "Show me the tags on Mia Stone's record before and after, so I can see the change landed."):
        assert merge_subtasks(q, ["Put `on-leave` on Mia Stone.", check]) == [q]



def test_a_name_extension_joins_when_the_request_sentence_points_back_though_the_model_dropped_it():
    q = "Who owns PayHub, which rights does it publish, and how sensitive is PayHub_Infrastructure Manager among them?"
    assert merge_subtasks(q, ["Who owns PayHub?", "Which rights does PayHub publish?",
                              "How sensitive is PayHub_Infrastructure Manager?"]) == [q]


# ------------------------------------------------------------------ a fragment of a request name
def test_a_sub_task_naming_a_fragment_of_one_request_name_is_that_name():
    q = "Give me the relational picture of Hart and Sons and Sons."
    split = ["Give me the relational picture of Hart and Sons.", "Give me the relational picture of Sons."]
    assert merge_subtasks(q, split) == [q]


# ------------------------------------------------------------------ invented values
def test_a_sub_task_quoting_a_value_the_request_never_gives_is_dropped():
    q = "Put Ann Lee through an offboarding sweep. Then open Bo Chan on his logins."
    split = ["Put Ann Lee through an offboarding sweep.", "Apply the tag 'offboarding' on Ann Lee.",
             "Open Bo Chan on his logins."]
    assert merge_subtasks(q, split) == ["Put Ann Lee through an offboarding sweep.", "Open Bo Chan on his logins."]


def test_a_quoted_value_the_request_gives_quoted_or_as_a_name_is_kept():
    q = "Tag Ann Lee `vip`. Then open Bo Chan on his logins."
    assert merge_subtasks(q, ["Tag 'Ann Lee' `vip`.", "Open Bo Chan on his logins."]) == [
        "Tag 'Ann Lee' `vip`.", "Open Bo Chan on his logins."]


# ------------------------------------------------------------------ rule 6: a dropped job is put back
def test_a_request_sentence_whose_record_no_sub_task_names_is_put_back_in_place():
    q = "List the logins nobody owns on NORTHFEED. Confirm whether the HR feed carries any logins. Then open Bo Chan."
    split = ["List the logins nobody owns on NORTHFEED.", "Open Bo Chan."]
    assert merge_subtasks(q, split) == [
        "List the logins nobody owns on NORTHFEED.", "Confirm whether the HR feed carries any logins.", "Open Bo Chan."]


# ------------------------------------------------------------------ referents (chosen by the caller)
def test_the_unnamed_subjects_are_the_request_subjects_a_sub_task_leaves_out():
    from fastworkflow.split_merge import unnamed_subjects
    q = "Ann Lee leaves on Friday. Then tag the identity `on-leave`."
    assert unnamed_subjects(q, "Tag the identity `on-leave`.") == ["Ann Lee"]


def test_a_referent_replaces_the_one_pointing_phrase_or_is_appended():
    from fastworkflow.split_merge import with_referent
    assert with_referent("Raj Patel from that unit is leaving.", "Hart Mills Ltd") == \
        "Raj Patel from Hart Mills Ltd is leaving."
    assert with_referent("Confirm whether Raj Patel is among its holders.", "Payroll_Approver") == \
        "Confirm whether Raj Patel is among Payroll_Approver's holders."
    # two pointing phrases: which one means the subject is unknown, so it is appended
    assert with_referent("Triage that request: who holds it, its risk, and that unit's roster.", "Hart Mills Ltd") == \
        "Triage that request: who holds it, its risk, and that unit's roster (for Hart Mills Ltd)."
    # "those rights" is plural: not a pointer to one record
    assert with_referent("Compare them on the systems those rights belong to.", "Payroll_Approver") == \
        "Compare them on the systems those rights belong to (for Payroll_Approver)."
    assert with_referent("Open Ann Lee.", "Ann Lee") == "Open Ann Lee."
