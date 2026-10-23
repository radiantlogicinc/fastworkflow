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
    q = "Ann Lee leaves on Friday. Then tag the customer `on-leave`."
    assert unnamed_subjects(q, "Tag the customer `on-leave`.") == ["Ann Lee"]


def test_a_record_listed_with_the_one_a_sub_task_names_is_not_its_unnamed_subject():
    from fastworkflow.split_merge import unnamed_subjects
    q = ("Northwind Ltd is closing. Walk its whole roster: every person on it. Ann Lee and Bob Kerr leave at the end "
         "of the month, so each of them down to their first login listed. And Cid Moss has asked for the right "
         "Payroll_Admin: what it exposes and who holds it.")
    ann = "Ann Lee leaves at the end of the month, so each of them down to their first login listed."
    assert unnamed_subjects(q, ann) == ["Northwind Ltd", "Payroll_Admin", "Cid Moss"]
    assert "Ann Lee" not in unnamed_subjects(q, ann.replace("Ann Lee", "Bob Kerr"))
    # a sub-task naming neither keeps both as candidates
    assert {"Ann Lee", "Bob Kerr"} <= set(unnamed_subjects(q, "Walk the whole roster of Northwind Ltd."))


def test_a_referent_replaces_the_one_pointing_phrase_or_is_appended():
    from fastworkflow.split_merge import with_referent
    assert with_referent("Raj Patel from that team is leaving.", "Hart Mills Ltd") == \
        "Raj Patel from Hart Mills Ltd is leaving."
    assert with_referent("Confirm whether Raj Patel is among its holders.", "Payroll_Approver") == \
        "Confirm whether Raj Patel is among Payroll_Approver's holders."
    # two pointing phrases: which one means the subject is unknown, so it is appended
    assert with_referent("Triage that request: who holds it, its risk, and that team's roster.", "Hart Mills Ltd") == \
        "Triage that request: who holds it, its risk, and that team's roster (for Hart Mills Ltd)."
    # "those rights" is plural: not a pointer to one record
    assert with_referent("Compare them on the systems those rights belong to.", "Payroll_Approver") == \
        "Compare them on the systems those rights belong to (for Payroll_Approver)."
    assert with_referent("Open Ann Lee.", "Ann Lee") == "Open Ann Lee."


def test_a_definite_description_of_the_referent_gets_the_record_written_after_it():
    from fastworkflow.split_merge import with_referent
    text = "Tag the customer `on-leave` and show me its tags before and after."
    # the caller found that "the customer" stands for the record: written in there, and the one "its" too
    assert with_referent(text, "Ann Lee", "the customer") == \
        "Tag the customer Ann Lee `on-leave` and show me Ann Lee's tags before and after."
    assert with_referent("Tag the customer `on-leave`.", "Ann Lee", "the customer") == \
        "Tag the customer Ann Lee `on-leave`."
    # several pointing phrases: the description names the record, so nothing is appended
    assert with_referent("Show me the customer's tags, its risk and its logins.", "Ann Lee", "the customer") == \
        "Show me the customer Ann Lee's tags, its risk and its logins."
    # without a description the old behaviour holds
    assert with_referent("Tag the customer `on-leave`.", "Ann Lee") == "Tag the customer `on-leave` (for Ann Lee)."


def test_definite_descriptions_leave_out_plurals_common_words_of_phrases_and_named_ones():
    from fastworkflow.split_merge import definite_descriptions
    assert definite_descriptions("Tag the customer `on-leave` and show me its tags.") == ["the customer"]
    assert definite_descriptions("Show me the customer's tags.") == ["the customer"]
    assert definite_descriptions(
        "Open the riskier of NORTHFEED and SOUTHFEED and show me the controls it trips.") == []
    assert definite_descriptions("Open the unit Payroll and the first login listed.") == []
    assert definite_descriptions("Show me the full list.") == []


# ------------------------------------------------------------------ rule 6a: a sentence's subjects stay with it
def test_a_subject_cut_from_its_sentence_is_restored_though_another_sub_task_names_it():
    q = ("Feed review. List the logins nobody owns on NORTHFEED and on SOUTHFEED, with a count for each. "
         "Confirm whether the HR feed carries any logins at all. "
         "Then open the riskier of NORTHFEED and SOUTHFEED and show me the controls it trips.")
    split = ["List the logins nobody owns on NORTHFEED, with a count for each.",
             "Confirm whether the HR feed carries any logins at all.",
             "Open the riskier of NORTHFEED and SOUTHFEED and show me the controls it trips."]
    # the sentence comes back whole, then rule 7 cuts it one sub-task per feed ("for each")
    assert merge_subtasks(q, split) == [
        "List the logins nobody owns on NORTHFEED, with a count for each.",
        "List the logins nobody owns on SOUTHFEED, with a count for each.",
        "Confirm whether the HR feed carries any logins at all.",
        "Open the riskier of NORTHFEED and SOUTHFEED and show me the controls it trips."]


def test_a_sentence_split_per_subject_or_kept_whole_is_left_alone():
    q = ("Feed review. List the logins nobody owns on NORTHFEED and on SOUTHFEED, with a count for each. "
         "Then open the riskier of NORTHFEED and SOUTHFEED and show me the controls it trips.")
    split = ["List the logins nobody owns on NORTHFEED, with a count for each.",
             "List the logins nobody owns on SOUTHFEED, with a count for each.",
             "Open the riskier of NORTHFEED and SOUTHFEED and show me the controls it trips."]
    assert merge_subtasks(q, split) == split


# ------------------------------------------------------------------ rule 3a: statements are context
def test_a_statement_subtask_is_context_for_the_job_after_it():
    q = "Ann Lee's login is a standard account. Remove the `admin` tag from it."
    assert merge_subtasks(q, ["Ann Lee's login is a standard account.",
                              "Remove the `admin` tag from Ann Lee's login."]) == [q]


def test_a_dropped_statement_is_written_before_the_job_pointing_back_to_its_record():
    q = ("I am closing out Field Sales for the year. Walk the whole unit: every person on that roster. "
         "Then explain the controls 'Stale badges' and 'Shared logins': why each was flagged.")
    split = ["Walk the whole unit: every person on that roster.",
             "Explain the control 'Stale badges': why it was flagged.",
             "Explain the control 'Shared logins': why it was flagged."]
    assert merge_subtasks(q, split) == [
        "I am closing out Field Sales for the year. Walk the whole unit: every person on that roster.",
        "Explain the control 'Stale badges': why it was flagged.",
        "Explain the control 'Shared logins': why it was flagged."]


#: An asking sentence naming the unit, then a walk of "it" that names no record, then two leavers.
RESTRUCTURE = ("Field Sales and Sons 1 is being restructured and I need the access picture before it happens. "
               "Walk the whole roster - every identity on it, down to logins, rights and collections. "
               "Ann Lee and Bob Kerr leave at the end of the month, so each of them down to their first login.")
LEAVERS = ["Ann Lee leaves at the end of the month: down to Ann Lee's first login.",
           "Bob Kerr leaves at the end of the month: down to Bob Kerr's first login."]


def test_an_asking_sentence_whose_record_no_subtask_names_is_written_before_the_recordless_job_after_it():
    split = ["Walk the whole roster - every identity on it, down to logins, rights and collections."] + LEAVERS
    assert merge_subtasks(RESTRUCTURE, split) == [
        "Field Sales and Sons 1 is being restructured and I need the access picture before it happens. "
        "Walk the whole roster - every identity on it, down to logins, rights and collections."] + LEAVERS


def test_an_asking_sentence_is_not_written_before_jobs_that_name_a_record():
    # the next sentence's sub-tasks name records of their own: the sentence is put back as its own job
    q = ("Field Sales and Sons 1 is being restructured and I need the access picture before it happens. "
         "Ann Lee and Bob Kerr leave at the end of the month, so each of them down to their first login.")
    assert merge_subtasks(q, LEAVERS) == [
        "Field Sales and Sons 1 is being restructured and I need the access picture before it happens."] + LEAVERS
    # a sub-task names the sentence's record: nothing is written in
    split = ["Walk the whole Field Sales and Sons 1 roster - every identity on it, down to logins, rights "
             "and collections."] + LEAVERS
    assert merge_subtasks(RESTRUCTURE, split) == split


def test_statements_and_asks():
    from fastworkflow.split_merge import _asks, _is_statement
    for statement in ("Ann Lee leaves at the end of the week.", "I am closing out Field Sales for the year.",
                      "Field Sales and Sons 1 is being folded into another unit."):
        assert _is_statement(statement)
    for ask in ("Audit the right X across the systems it appears in.", "I want everything Ann Lee can reach.",
                "Ann Lee leaves on Friday: what Ann Lee can reach.", "Quarterly review of Field Sales: the unit.",
                "Ann Lee has asked for X, so triage that request.", "Who holds X?"):
        assert not _is_statement(ask)


# ------------------------------------------------------------------ rule 3b and 6: wrong and dropped records
def test_a_subtask_putting_another_sentences_record_in_a_job_is_dropped():
    q = ("I need the month-end picture of both BuildOps admin rights and of Field Sales together. "
         "Audit BuildOps_Release Manager and BuildOps_Backup Administrator wherever they appear, each with the "
         "system that publishes it. Then walk Field Sales end to end: every identity on the roster.")
    split = ["Audit BuildOps admin rights wherever they appear, including the system that publishes it.",
             "Audit Field Sales wherever it appears, including the system that publishes it.",
             "Walk Field Sales end to end: every identity on the roster."]
    assert merge_subtasks(q, split) == [
        "Audit BuildOps_Release Manager wherever they appear, each with the system that publishes it.",
        "Audit BuildOps_Backup Administrator wherever they appear, each with the system that publishes it.",
        "Walk Field Sales end to end: every identity on the roster."]


def test_an_asked_job_whose_record_only_other_jobs_name_is_put_back():
    q = ("Audit the right Core Gateway_Network Administrator across the systems it appears in: holders, the "
         "publishing system and the feed. While you are on it, confirm whether Iris Vance and Lena Marsh are "
         "among its holders.")
    split = ["Confirm whether Iris Vance is among the holders of the right Core Gateway_Network Administrator.",
             "Confirm whether Lena Marsh is among the holders of the right Core Gateway_Network Administrator."]
    assert merge_subtasks(q, split) == [
        "Audit the right Core Gateway_Network Administrator across the systems it appears in: holders, the "
        "publishing system and the feed."] + split


def test_a_part_of_a_name_counts_only_written_as_a_name():
    known = ["BuildOps_Backup Administrator", "Field Sales"]
    assert subjects_in("Audit the administrator rights.", known) == frozenset()
    assert subjects_in("Audit Administrator wherever it appears.", known) == {"buildops_backup administrator"}


# ------------------------------------------------------------------ rule 7: one job per record
def test_the_same_thing_asked_of_each_record_of_a_list_is_one_subtask_per_record():
    q = ("Three people need the same look - Ann Lee, Bo Chan and Cy Dale - so I want each of them down to "
         "their logins. Then explain the control 'Stale badges'.")
    split = ["Three people need the same look - Ann Lee, Bo Chan and Cy Dale - so I want each of them down to "
             "their logins.", "Explain the control 'Stale badges'."]
    assert merge_subtasks(q, split) == [
        f"Three people need the same look - {name} - so I want {name} down to their logins."
        for name in ("Ann Lee", "Bo Chan", "Cy Dale")] + ["Explain the control 'Stale badges'."]


def test_a_comparison_or_a_list_without_each_stays_one_subtask():
    q = "Compare Ann Lee and Bo Chan side by side, each on logins. Then explain the control 'Stale badges'."
    split = ["Compare Ann Lee and Bo Chan side by side, each on logins.", "Explain the control 'Stale badges'."]
    assert merge_subtasks(q, split) == split
    q = "Show the logins Ann Lee and Bo Chan share. Then explain the control 'Stale badges'."
    split = ["Show the logins Ann Lee and Bo Chan share.", "Explain the control 'Stale badges'."]
    assert merge_subtasks(q, split) == split


def test_a_list_of_quoted_names_is_cut_too():
    q = "Explain the controls 'Stale badges' and 'Shared logins': why each was flagged. Then walk Field Sales."
    split = ["Explain the controls 'Stale badges' and 'Shared logins': why each was flagged.", "Walk Field Sales."]
    assert merge_subtasks(q, split) == ["Explain the controls 'Stale badges': why each was flagged.",
                                        "Explain the controls 'Shared logins': why each was flagged.",
                                        "Walk Field Sales."]


def test_a_long_name_dense_request_merges_in_well_under_a_second_per_thousand_words():
    """fix-ajir.7: every request subject is tested against the same text; without the caches of
    ``_norm`` and the word patterns that was quadratic (about 5 s at 2,000 words, 24 s at 8,000)."""
    import random
    import time

    rng = random.Random(1)
    words = ("Ann Lee and Bob Ray then the group Payroll, its members; each of them 'Ops Team' "
             "\"North\" also so but (x) [y] - Finance_Admin RL-1234 it they").split()
    q = " ".join(rng.choice(words) for _ in range(2000))
    split = [q[i:i + len(q) // 5] for i in range(0, len(q), len(q) // 5)][:5]
    start = time.perf_counter()
    merge_subtasks(q, split)
    assert time.perf_counter() - start < 2.5
