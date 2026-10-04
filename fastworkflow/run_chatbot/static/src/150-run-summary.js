/* -- a summary over the runs a reader ticked -------------------------- */

/* Which runs are ticked, for ONE task read from ONE source.

   The key carries the source as well as the task, so switching either drops
   the ticks rather than carrying "attempt 3" into a task, or an archive, that
   means something else by it. `generation` rises on every tick and every
   refresh: a response whose generation, source or navigation token no longer
   matches is dropped instead of rendered under a question nobody asked.

   The Runs view's own navigation token is used AS PASSED IN. Claiming a new
   one here would stale the task list and the consistency band that share it,
   and this panel is part of that render rather than a navigation of its
   own. */
var selectedRuns = { key: null, chosen: {}, generation: 0, mode: null,
                     population: null };

/* Which "check for run changes" click owns its own request. Separate from the
   drill-down claim below: checking is not navigating, so it must not cancel a
   drill-down and a drill-down must not cancel it. */
var selectedRunsCheck = 0;

/* Which request owns the POPULATION NOTICE, and whether that owner has
   answered yet.

   Two different requests report on the same population into the same box: the
   explicit check, and the drill-down validation that carries the baseline
   along. Ownership of the notice therefore cannot live in either one's own
   clock -- with a clock each, an older validation answering after a newer
   check had painted "the runs recorded for this task have changed" overwrote
   it with that request's older "membership below is unchanged", and the reader
   was told the population had not moved by evidence read before it did.

   So ONE monotonic claim orders every write to this box -- the pending line,
   the answer, a refusal and a failure alike -- while navigation ownership
   stays where it was (`selectedRunsClick`), because a check is not a
   navigation and must not cancel a drill-down the reader asked for.

   `pending` says the box currently holds an unanswered "checking…" line. Its
   writer's answer may since have been superseded, so whoever takes the claim
   is responsible for clearing it rather than leaving a question on screen that
   nothing will ever answer. */
var populationNotice = { claim: 0, pending: false };

function claimPopulationNotice() {
  populationNotice.claim += 1;
  return populationNotice.claim;
}

/* A "checking…" line, from the request that currently owns the box. */
function pendingPopulationNotice(box, claim, text) {
  if (!box || claim !== populationNotice.claim) { return false; }
  clear(box);
  box.appendChild(el("div", "sub", text));
  populationNotice.pending = true;
  return true;
}

/* The answer, painted only if this request still owns the box. */
function writePopulationNotice(box, claim, check) {
  if (!box || claim !== populationNotice.claim) { return false; }
  populationNotice.pending = false;
  renderPopulationNotice(box, check);
  return true;
}

/* A refusal or a failure. Ordered by the same claim as an answer, because a
   stale error overwriting a newer answer is the same defect. */
function writePopulationProblem(box, claim, message) {
  if (!box || claim !== populationNotice.claim) { return false; }
  populationNotice.pending = false;
  clear(box);
  box.appendChild(el("div", "chipMarker unknown", message));
  return true;
}

/* This request will write nothing after all. Any unanswered "checking…" line
   belongs to a write that is now superseded, so it goes rather than standing
   over a question nothing will answer; an answer already on screen is left
   exactly as it is. */
function abandonPopulationNotice(box, claim) {
  if (!box || claim !== populationNotice.claim || !populationNotice.pending) {
    return false;
  }
  populationNotice.pending = false;
  clear(box);
  return true;
}

/* A new question about the population: the box is emptied and every write
   still in flight is ordered before this. */
function resetPopulationNotice(box) {
  claimPopulationNotice();
  populationNotice.pending = false;
  if (box) { clear(box); }
}

/* Request safety, matching the route: it refuses more than this rather than
   summarizing some of them, so the page says so before asking. */
var SELECTED_RUNS_MAX = 20;

/* Which drill-down click owns the navigation. Monotonic, and separate from
   the summary's generation: two contributors clicked while both validations
   are in flight must not both navigate, and the one that wins has to be the
   one the reader clicked LAST -- not whichever check happened to answer
   first. Claimed on the click, checked before the navigation. */
var selectedRunsClick = 0;

function selectedRunsKey(experimentId, taskId) {
  return String(benchmarkExperimentSource) + "\u001f" + experimentId
    + "\u001f" + taskId;
}

function resetSelectedRuns(experimentId, taskId) {
  var key = selectedRunsKey(experimentId, taskId);
  if (selectedRuns.key === key) { return; }
  selectedRuns = { key: key, chosen: {}, generation: 0, mode: null,
                   population: null };
}

function selectedRunAttempts() {
  return Object.keys(selectedRuns.chosen)
    .filter(function (attempt) { return selectedRuns.chosen[attempt]; })
    .map(Number)
    .sort(function (a, b) { return a - b; });
}

/* The population a result was made over, as the page may echo it back.

   Two opaque values the server published plus the ids it listed. Neither
   opaque value is believed as a description of anything: the server re-derives
   both from its own metadata and compares, and refuses a baseline taken under
   another task, source or archive. Nothing here names a store. */
function rememberPopulation(data) {
  var population = data.task_population;
  selectedRuns.population = !population ? null : {
    scope: population.population_scope,
    digest: population.population_digest,
    rule: population.selection_rule,
    members: population.finished_attempts || [],
    memberCount: population.finished,
    unfinished: population.unfinished_attempts || [],
    unfinishedComplete: population.unfinished_listing_complete !== false
  };
}

function populationBaselineQuery() {
  var base = selectedRuns.population;
  if (!base) { return ""; }
  var parts = ["population_scope=" + encodeURIComponent(base.scope),
               "expect_population=" + encodeURIComponent(base.digest),
               "population_rule=" + encodeURIComponent(base.rule),
               "baseline_member_count=" + encodeURIComponent(base.memberCount),
               "baseline_unfinished_complete="
                 + (base.unfinishedComplete ? "true" : "false")];
  base.members.forEach(function (attempt) {
    parts.push("baseline_member=" + encodeURIComponent(attempt));
  });
  base.unfinished.forEach(function (attempt) {
    parts.push("baseline_unfinished=" + encodeURIComponent(attempt));
  });
  return parts.join("&");
}

function attemptList(ids) {
  return (ids || []).map(function (id) { return "attempt " + id; }).join(", ");
}

/* What the population check answered, painted where it stays.

   Membership is NOT touched here. A newly eligible run is news about the task,
   not a change to any run this summary counted, so it is said plainly and the
   result stays exactly as it was until somebody refreshes. */
function renderPopulationNotice(box, check) {
  if (!box) { return; }
  clear(box);
  if (!check) { return; }
  if (check.changed === null || check.changed === undefined) {
    box.appendChild(el("div", "sub",
      "No baseline was available, so nothing is claimed about whether the "
      + "runs recorded for this task have changed."));
    return;
  }
  if (!check.changed) {
    box.appendChild(el("div", "sub",
      "Checked just now: this task still records " + fmtCount(check.recorded)
      + " run(s), " + fmtCount(check.finished) + " of them finished, so the "
      + "membership below is unchanged. This read the runs, not what they "
      + "recorded: opening a contributor still re-checks that run's evidence."));
    return;
  }
  var lines = [];
  if (check.newly_finished_count) {
    lines.push(fmtCount(check.newly_finished_count) + " run(s) that were "
      + "still going have finished (" + attemptList(check.newly_finished) + ")");
  }
  if (check.newly_recorded_finished_count) {
    lines.push(fmtCount(check.newly_recorded_finished_count) + " finished "
      + "run(s) have been recorded since ("
      + attemptList(check.newly_recorded_finished) + ")");
  }
  if (check.newly_recorded_unfinished_count) {
    lines.push(fmtCount(check.newly_recorded_unfinished_count) + " unfinished "
      + "run(s) have been recorded since ("
      + attemptList(check.newly_recorded_unfinished) + ")");
  }
  if (!check.detail_complete && check.added_members_count) {
    lines.push(fmtCount(check.added_members_count) + " finished run(s) are "
      + "eligible now that were not (" + attemptList(check.added_members) + ")");
  }
  if (check.removed_finished_count) {
    lines.push(fmtCount(check.removed_finished_count) + " run(s) this summary "
      + "counted are no longer recorded ("
      + attemptList(check.removed_finished) + ")");
  }
  if (check.removed_unfinished_count) {
    lines.push(fmtCount(check.removed_unfinished_count) + " unfinished run(s) "
      + "are no longer recorded (" + attemptList(check.removed_unfinished) + ")");
  }
  if (check.lost_eligibility_count) {
    lines.push(fmtCount(check.lost_eligibility_count) + " run(s) this summary "
      + "counted no longer read as finished ("
      + attemptList(check.lost_eligibility) + ")");
  }
  if (check.undetailed || !lines.length) {
    lines.push("the runs recorded for this task have changed in a way the "
      + "listed attempts do not account for");
  }
  box.appendChild(el("div", "chipMarker unknown",
    "The runs recorded for this task have changed: " + lines.join("; ")
    + ". Nothing below has been replaced and no run it counted is claimed to "
    + "have changed. Refresh to summarize the runs as they are now."));
}

function selectedRunsMember(data, attempt) {
  var found = null;
  (data.members || []).forEach(function (member) {
    if (Number(member.attempt) === Number(attempt)) { found = member; }
  });
  return found;
}

/* One member's source, resolved from its OWN reference.

   An archive has two names and they are not interchangeable: the reference
   carries the evidence identity, and every workspace route is addressed by
   the manifest's name. Both travel on the member, and `pairReadScope` picks
   the one its route needs. */
function selectedRunsSide(data, attempt, experimentId) {
  var member = selectedRunsMember(data, attempt);
  return {
    which: "attempt " + attempt,
    run: member || {},
    projection: {},
    storeId: (member && member.store_id) || null,
    manifestStoreId: (member && member.manifest_store_id) || null,
    experimentId: experimentId
  };
}

/* Re-check ONE member before opening it, and refuse to open it if what it
   records is no longer what was counted.

   The route re-projects that run and compares the digest the summary
   published for it, so an outcome, a duration, a cost or an answer edited in
   place -- with every identifier unchanged -- is disclosed here rather than
   quietly standing in for the evidence the totals were made of. Optimistic:
   it detects a change, it does not prevent one. */
function validateSelectedMember(data, attempt, experimentId, taskId, note, nav,
                                then, notice) {
  var claim = ++selectedRunsClick;
  var member = selectedRunsMember(data, attempt);
  if (!member || !member.evidence_digest) { then(); return; }
  var generation = selectedRuns.generation;
  var source = selectedRuns.key;
  var query = "?attempt=" + attempt + "&expect_member="
    + encodeURIComponent(attempt + ":" + member.evidence_digest);
  /* The population rides along on the SAME request, answered from the same
     metadata read. It never re-resolves the membership on screen: this
     validates the one fixed member and reports drift beside it. */
  var baseline = populationBaselineQuery();
  if (baseline) { query += "&" + baseline; }
  /* Claimed at REQUEST time, and only when a baseline actually rides along:
     the answer will report on the population, so it takes its turn in the same
     queue as the explicit check. Without a baseline nothing is claimed,
     because nothing about the population will be said and silently blanking a
     valid notice is not a report. */
  var noticeClaim = baseline ? claimPopulationNotice() : 0;
  if (note) {
    note.textContent = "checking that this run still records what was counted…";
  }
  selectionRead(
    taskSelectionPath(experimentId, taskId, "/selected-runs/validation" + query)
  ).then(function (check) {
    if (expNavStale(nav) || generation !== selectedRuns.generation
        || source !== selectedRuns.key || claim !== selectedRunsClick) {
      /* Four ways to be overtaken: the reader navigated, ticked a different
         selection, changed source, or clicked something else. The last one
         is why this is checked at all -- a slow check of the FIRST click
         answering after a second click would otherwise open the run the
         reader has already clicked past. */
      if (note) {
        note.textContent = "The page moved on before that check answered, so "
          + "nothing was opened.";
      }
      abandonPopulationNotice(notice, noticeClaim);
      return;
    }
    if (check.unavailable) {
      if (note) { note.textContent = check.error; }
      abandonPopulationNotice(notice, noticeClaim);
      return;
    }
    /* Before the staleness verdict, as it was: a member that no longer records
       what was counted is a separate answer from whether the task's runs have
       moved, and the reader is told both. */
    writePopulationNotice(notice, noticeClaim, check.population_check);
    if (check.stale) {
      if (note) {
        note.textContent = (check.changed && check.changed.length)
          ? "This run no longer records what these totals were computed from, "
            + "so it was not opened. Refresh the summary to count it as it is "
            + "now."
          : "This run is no longer part of the selection the totals were "
            + "computed over, so it was not opened.";
      }
      return;
    }
    if (note) { note.textContent = ""; }
    then();
  }).catch(function (e) {
    if (note) { note.textContent = e.message; }
    /* The failure is reported beside the run that was clicked. It says nothing
       about the population, so it does not write into that notice -- but the
       claim it took has to be given up, or a "checking…" line whose answer this
       request superseded would stay on screen. */
    abandonPopulationNotice(notice, noticeClaim);
  });
}

/* The exact members, each saying what it put into the totals -- and what it
   did not. A run that finished having recorded nothing is HERE, in the
   population, contributing no dispatch and saying so: leaving it out would
   quietly shrink the denominator of everything above. */
function renderSelectedRunsMembers(box, data, experimentId, taskId, label, nav,
                                   notice) {
  var list = el("div");
  list.setAttribute("data-selected-members", "");
  (data.members || []).forEach(function (member) {
    var row = el("div", "sub");
    row.setAttribute("data-selected-member", String(member.attempt));
    row.appendChild(el("span", null,
      "attempt " + member.attempt + " · "
      + (member.outcome || member.execution_status || "unfinished")
      + " · " + fmtCount(member.dispatches) + " dispatch(es) counted · "
      + fmtCount(member.turns_read) + " turn(s) read"));
    if (member.is_best) { row.appendChild(el("span", "pill ok", "Best run")); }
    if (!member.execution_ref) {
      row.appendChild(el("span", "chipMarker unknown",
        "finished with no recorded turns, so it is in this population and in "
        + "none of these counts"));
    }
    if (member.turns_unreadable) {
      row.appendChild(el("span", "chipMarker unknown",
        fmtCount(member.turns_unreadable) + " turn(s) it names could not be "
        + "read"));
    }
    var note = el("div", "sub", "");
    if (member.execution_ref) {
      var compare = el("button", "evidenceLink", member.is_best
        ? "Compare with another attempt" : "Compare with the best run");
      compare.type = "button";
      compare.addEventListener("click", function () {
        validateSelectedMember(data, member.attempt, experimentId, taskId, note,
                               nav, function () {
          taskCompare.left = member.is_best ? String(member.attempt) : null;
          taskCompare.right = member.is_best ? null : String(member.attempt);
          taskCompare.rightExperiment = experimentId;
          /* A member is a WHOLE run, so the pair opens whole. The picker
             keeps its pass selection per task, and inheriting a teacher/
             student scope from an earlier comparison would silently compare
             part of the run this panel counted all of. */
          taskCompare.leftPass = null;
          taskCompare.rightPass = null;
          taskView = "compare";
          showExperimentTask(experimentId, taskId, label);
        }, notice);
      });
      row.appendChild(compare);
    }
    /* The existing task Feedback view, unchanged: comments are written about
       runs and pairs, and this summary does not invent an object to comment
       on. */
    var comment = el("button", "evidenceLink", "Comment");
    comment.type = "button";
    comment.addEventListener("click", function () {
      taskView = "feedback";
      showExperimentTask(experimentId, taskId, label);
    });
    row.appendChild(comment);
    row.appendChild(note);
    list.appendChild(row);
  });
  box.appendChild(list);
}

/* Requested, included, and every run that is neither -- named, not counted.

   Nothing here is a rate: "3 of 5" is two numbers and stays two numbers. */
function appendSelectedRunsPopulation(box, data) {
  var population = data.population || {};
  var line = el("div", "sub");
  line.setAttribute("data-selected-population", "");
  line.textContent = fmtCount(population.requested || 0)
    + " run(s) asked for · " + fmtCount(population.included || 0)
    + " in these counts · " + fmtCount(population.with_readable_evidence || 0)
    + " with readable evidence";
  box.appendChild(line);
  (population.excluded_runs || []).forEach(function (row) {
    box.appendChild(el("div", "chipMarker unknown",
      "attempt " + row.attempt + " is not in these counts: " + row.detail));
  });
  (population.missing_evidence || []).forEach(function (attempt) {
    box.appendChild(el("div", "chipMarker unknown",
      "attempt " + attempt + " finished having recorded nothing, so it is in "
      + "this population and in none of these counts; how much it would have "
      + "contributed is unknown"));
  });
}

/* What this TASK has, beside what this summary is over.

   Counts only, each one off the response. "Planned" appears only when a
   per-task run plan is actually recorded: an experiment-wide declared total is
   not a plan for this task and is never shown as one. */
function appendTaskPopulation(box, data) {
  var population = data.task_population;
  if (!population) { return; }
  var line = el("div", "sub");
  line.setAttribute("data-task-population", "");
  var text = "This task: " + fmtCount(population.recorded)
    + " run(s) recorded · " + fmtCount(population.finished) + " finished · "
    + fmtCount(population.unfinished) + " still running";
  if (population.planned !== null && population.planned !== undefined) {
    text += " · " + fmtCount(population.planned) + " planned for this task";
  }
  line.textContent = text;
  box.appendChild(line);
  if ((population.planned === null || population.planned === undefined)
      && population.selection_rule === "all_finished") {
    box.appendChild(el("div", "sub", population.planned_note || ""));
  }
}

function renderSelectedRunsResult(body, data, experimentId, taskId, label, ctx,
                                  nav, notice) {
  var summary = data.command_summary || {};
  var coverage = summary.coverage || {};
  var rule = (data.scope || {}).selection_rule;
  /* Zero finished runs is an ANSWER. No counts, no cost, no fabricated zero:
     a task that has finished nothing has spent nothing only in the sense that
     nothing was recorded, and saying "$0.00" would be a claim. */
  if (!data.member_count && rule === "all_finished") {
    appendTaskPopulation(body, data);
    body.appendChild(el("div", "empty",
      "This task has recorded no finished run yet, so there is nothing to "
      + "summarize. That is the answer rather than a missing one: no counts "
      + "and no cost are shown because none were recorded."));
    return;
  }
  body.appendChild(el("div", "sub", commandHeadline(summary)));
  appendTaskPopulation(body, data);
  appendSelectedRunsPopulation(body, data);
  appendCommandCoverage(body, summary, {
    headline: fmtCount(coverage.runs_included || 0)
      + " run(s) pooled: " + fmtCount(coverage.turns_examined || 0)
      + " recorded turn(s) examined across them, counted in "
      + (summary.unit || "dispatches") + " · " + summary.metric_version
      + ". Every figure below is over the dispatches themselves, so a median "
      + "is the median of the durations and not an average of the runs.",
    subject: "the selected runs",
    where: "in the selected runs"
  });
  if (coverage.population_complete === false) {
    body.appendChild(el("div", "chipMarker unknown",
      "These totals cover the observed evidence of the runs that are in them; "
      + "the population above says what that is a part of."));
  }
  /* The same words the rest of the page uses for money, including the ones
     for "calls were made and none recorded a cost", which is not zero. */
  var cost = data.cost || {};
  var money = fmtCostAmount(cost);
  body.appendChild(el("div", "sub",
    "Recorded LLM cost: "
    + (money || "no LLM call was observed in the selected runs")
    + ". This is what the evidence charged, not an allocation: no share of it "
    + "is attributed to a command or a dispatch."
    + (cost.population_complete === false
        ? " " + fmtCount(cost.members_without_readable_evidence)
          + " selected run(s) recorded no readable evidence, so this covers "
          + "the calls that were observed rather than the selection."
        : "")));
  var outcomes = data.run_outcomes || {};
  var tally = [];
  Object.keys(outcomes.by_outcome || {}).sort().forEach(function (key) {
    tally.push(fmtCount(outcomes.by_outcome[key]) + " " + key);
  });
  body.appendChild(el("div", "sub",
    "Run outcomes: " + (tally.length ? tally.join(", ") : "none recorded")
    + " · " + fmtCount(outcomes.runs_with_failed_dispatches || 0)
    + " run(s) contain a failed dispatch, which is a different count from a "
    + "failed run and is not derived from it."));
  renderSelectedRunsMembers(body, data, experimentId, taskId, label, nav,
                            notice);
  var groups = summary.groups || [];
  if (!groups.length) {
    body.appendChild(el("div", "empty",
      "No command dispatch was observed in the selected runs."));
    return;
  }
  /* The pair view's own renderers, with the source resolved per contributor
     and the drill-down gated on the member still recording what was counted.
     Nothing is recomputed here: every number came off the response. */
  var pooled = {
    experimentId: ctx.experimentId, taskId: ctx.taskId, storeId: ctx.storeId,
    comparison: null,
    beforeOpen: function (contributor, note, open) {
      validateSelectedMember(data, contributor.attempt, experimentId, taskId,
                             note, nav, open, notice);
    }
  };
  var resolve = function (contributor) {
    return selectedRunsSide(data, contributor.attempt, experimentId);
  };
  var wrap = el("div", "ledgerWrap");
  var table = el("table", "ledger");
  var head = el("tr");
  ["command", "dispatches", "succeeded", "failed", "not recorded",
   "record vs span", "inclusive min / median / max", "timed"]
    .forEach(function (column) { head.appendChild(el("th", null, column)); });
  table.appendChild(head);
  groups.forEach(function (group) {
    commandGroupRows(group, resolve, pooled).forEach(function (row) {
      table.appendChild(row);
    });
  });
  wrap.appendChild(table);
  body.appendChild(wrap);
}

/* The panel: tick runs above, summarize them here.

   One endpoint, the same one a coding agent calls, and no arithmetic on this
   side of it. Membership is fixed to what the server resolved when the
   summary was made: a later tick does not edit the answer on screen, it
   discards it, because a figure re-labelled with a population it was not
   computed over is worse than no figure. */
function renderSelectedRunsPanel(card, experimentId, taskId, label, runs, nav) {
  var ctx = { experimentId: experimentId, taskId: taskId,
              storeId: runs.store_id || null };
  var panel = el("div", "card");
  panel.setAttribute("data-selected-runs", "");
  panel.appendChild(el("h2", null, "Summary of selected runs"));
  panel.appendChild(el("div", "sub",
    "Tick the finished runs above and summarize them together, or summarize "
    + "every finished run of this task. The dispatches of the member runs are "
    + "pooled, the runs that are not in the counts are named, and nothing here "
    + "ranks them or picks one."));
  var controls = el("div", "runDecision");
  var summarize = el("button", "primary", "Summarize the selected runs");
  summarize.type = "button";
  summarize.setAttribute("data-selected-runs-submit", "");
  /* The rule, resolved by the server from this task's attempt metadata. The
     page never builds this population out of the rows it happens to show. */
  var allFinished = el("button", null, "Summarize all finished runs");
  allFinished.type = "button";
  allFinished.setAttribute("data-selected-runs-all", "");
  var refresh = el("button", null, "Refresh");
  refresh.type = "button";
  refresh.setAttribute("data-selected-runs-refresh", "");
  refresh.hidden = true;
  var recheck = el("button", null, "Check for run changes");
  recheck.type = "button";
  recheck.setAttribute("data-selected-runs-check", "");
  recheck.hidden = true;
  var count = el("span", "sub", "");
  count.setAttribute("data-selected-runs-count", "");
  controls.appendChild(summarize);
  controls.appendChild(allFinished);
  controls.appendChild(refresh);
  controls.appendChild(recheck);
  controls.appendChild(count);
  panel.appendChild(controls);
  /* Its own region, above the result and outside it: a notice painted into a
     drill-down row leaves with the view that drill-down opens, and a result
     with no members has no row to paint into. */
  var notice = el("div");
  notice.setAttribute("data-selected-runs-notice", "");
  panel.appendChild(notice);
  var body = el("div");
  body.setAttribute("data-selected-runs-body", "");
  panel.appendChild(body);
  card.appendChild(panel);

  function load(rule) {
    var wholeTask = rule === "all_finished";
    var attempts = selectedRunAttempts();
    if (!wholeTask && (!attempts.length || attempts.length > SELECTED_RUNS_MAX)) {
      return;
    }
    var generation = ++selectedRuns.generation;
    var source = selectedRuns.key;
    selectedRuns.mode = wholeTask ? "all_finished" : "explicit";
    selectedRuns.population = null;
    recheck.hidden = true;
    resetPopulationNotice(notice);
    clear(body);
    body.appendChild(el("div", "empty", wholeTask
      ? "resolving every finished run of this task…"
      : "summarizing " + fmtCount(attempts.length) + " run(s)…"));
    var query = wholeTask
      ? "scope=all_finished"
      : attempts.map(function (attempt) {
          return "attempt=" + encodeURIComponent(attempt);
        }).join("&");
    selectionRead(
      taskSelectionPath(experimentId, taskId, "/selected-runs?" + query)
    ).then(function (data) {
      /* Three guards, for the three things that can overtake this answer: the
         reader navigated, ticked something else, or changed source. */
      if (expNavStale(nav) || generation !== selectedRuns.generation
          || source !== selectedRuns.key) { return; }
      clear(body);
      if (data.unavailable) {
        selectionNote(body, "These runs cannot be summarized together.",
                      data.error);
        return;
      }
      rememberPopulation(data);
      refresh.hidden = false;
      recheck.hidden = !selectedRuns.population;
      renderSelectedRunsResult(body, data, experimentId, taskId, label, ctx,
                               nav, notice);
    }).catch(function (e) {
      if (expNavStale(nav) || generation !== selectedRuns.generation
          || source !== selectedRuns.key) { return; }
      clear(body);
      /* The route's own refusal, which says how many runs there are and why
         none of them was summarized. */
      body.appendChild(el("div", "empty", "Not summarized: " + e.message));
    });
  }

  /* Metadata only: it names no attempt, so nothing is re-projected and no
     unselected run's turns are read. The result on screen is left alone --
     Refresh replaces membership and this never does. */
  function checkRunChanges() {
    var baseline = populationBaselineQuery();
    if (!baseline) { return; }
    var claim = ++selectedRunsCheck;
    /* The shared claim, taken at request time beside this button's own: the
       button's clock drops an older CHECK, and the notice claim orders this
       write against every other writer of the same box. */
    var noticeClaim = claimPopulationNotice();
    var generation = selectedRuns.generation;
    var source = selectedRuns.key;
    pendingPopulationNotice(notice, noticeClaim,
      "checking whether the runs recorded for this task have changed…");
    selectionRead(taskSelectionPath(
      experimentId, taskId, "/selected-runs/validation?" + baseline
    )).then(function (answer) {
      if (expNavStale(nav) || generation !== selectedRuns.generation
          || source !== selectedRuns.key || claim !== selectedRunsCheck) {
        abandonPopulationNotice(notice, noticeClaim);
        return;
      }
      if (answer.unavailable) {
        writePopulationProblem(notice, noticeClaim, answer.error);
        return;
      }
      writePopulationNotice(notice, noticeClaim, answer.population_check);
    }).catch(function (e) {
      if (expNavStale(nav) || generation !== selectedRuns.generation
          || source !== selectedRuns.key || claim !== selectedRunsCheck) {
        abandonPopulationNotice(notice, noticeClaim);
        return;
      }
      writePopulationProblem(notice, noticeClaim, e.message);
    });
  }

  summarize.addEventListener("click", function () { load("explicit"); });
  allFinished.addEventListener("click", function () { load("all_finished"); });
  refresh.addEventListener("click", function () {
    load(selectedRuns.mode || "explicit");
  });
  recheck.addEventListener("click", checkRunChanges);

  return function update() {
    var attempts = selectedRunAttempts();
    var tooMany = attempts.length > SELECTED_RUNS_MAX;
    count.textContent = !attempts.length
      ? "No run is selected yet."
      : (tooMany
          ? (fmtCount(attempts.length) + " run(s) selected, and at most "
             + SELECTED_RUNS_MAX + " can be summarized in one request. "
             + "Nothing is sampled, so untick some rather than being shown "
             + "part of what you asked for.")
          : (fmtCount(attempts.length) + " run(s) selected"));
    summarize.disabled = !attempts.length || tooMany;
    /* A different selection is a different question, so any answer on screen
       stops being an answer to it, and any request still in flight stops
       being one anybody asked for. */
    selectedRuns.generation++;
    selectedRuns.mode = null;
    selectedRuns.population = null;
    refresh.hidden = true;
    recheck.hidden = true;
    resetPopulationNotice(notice);
    clear(body);
  };
}

/* The pointer says WHAT was decided and WHEN; who decided it and why is in
   the append-only decision log, which the panel below opens on request. */
function bestRunProvenance(pointer) {
  return "Recorded by a " + (pointer.decision || "decision")
    + (pointer.decided_at ? " at " + pointer.decided_at : "")
    + ". Who decided it and any reason are in the decisions below.";
}

/* What can be done with one attempt, and what visibly cannot.

   Only a FINISHED attempt is selectable: a run still going has no outcome to
   prefer. An attempt that finished with no recorded turns stays selectable --
   it really did finish and a reviewer may mean it -- but its comparison is
   disabled with the evidence's own words for why. */
function renderAttemptActions(actions, experimentId, taskId, label, row, runs,
                              decisions) {
  turnFindEntryButton(actions,
    { experiment: experimentId, task: taskId, attempt: row.attempt },
    "Find problems in this attempt",
    "Search the turns this attempt recorded.");
  if (row.comparable) {
    var compare = el("button", null, row.is_best
      ? "Compare with another attempt" : "Compare with the best run");
    compare.type = "button";
    compare.addEventListener("click", function () {
      taskCompare.left = row.is_best ? String(row.attempt) : null;
      taskCompare.right = row.is_best ? null : String(row.attempt);
      taskCompare.rightExperiment = experimentId;
      taskView = "compare";
      showExperimentTask(experimentId, taskId, label);
    });
    actions.appendChild(compare);
  } else if (row.comparable === false) {
    var refused = el("button", null, "Compare");
    refused.type = "button";
    refused.disabled = true;
    refused.title = row.evidence_label || "This attempt recorded no turns.";
    actions.appendChild(refused);
    actions.appendChild(el("span", "sub",
      row.evidence_label || "No recorded turns, so there is nothing to open."));
  }
  if (!decisions) { return; }
  var reload = function () { showExperimentTask(experimentId, taskId, label); };
  if (row.selectable && !row.is_best) {
    var choose = el("button", "primary", "Use as best run");
    choose.type = "button";
    choose.addEventListener("click", function () {
      openBestRunDecision(actions, experimentId, taskId, row, runs, reload);
    });
    actions.appendChild(choose);
  } else if (row.selectable === false) {
    actions.appendChild(el("span", "sub",
      "Not finished, so it cannot be the best run yet."));
  }
  if (row.is_best) {
    var clear_ = el("button", null, "Clear the best run");
    clear_.type = "button";
    clear_.addEventListener("click", function () {
      openBestRunDecision(actions, experimentId, taskId, row, runs, reload,
                          "clear");
    });
    actions.appendChild(clear_);
  }
}

/* One prompt for all three best-run decisions. Each carries the CAS token the
   list was rendered from, so a best run somebody else moved meanwhile is
   reported rather than overwritten. */
function openBestRunDecision(actions, experimentId, taskId, row, runs, reload,
                             kind) {
  var box = el("div", "card");
  var heading = kind === "clear"
    ? "Clear the best run of this task"
    : "Use attempt " + row.attempt + " as the best run";
  box.appendChild(el("h2", null, heading));
  box.appendChild(el("div", "sub",
    "This is a decision about this task only. It does not touch the "
    + "experiment winner, and it does not change any comment already written: "
    + "a comparison comment names the runs it was written about."));
  var reason = reasonField(box, "Reason");
  var status = el("div", "sub", "");
  var send = el("button", "primary",
    kind === "clear" ? "Clear it" : "Record this best run");
  send.type = "button";
  send.addEventListener("click", function () {
    send.disabled = true;
    status.textContent = "recording…";
    var path = taskSelectionPath(experimentId, taskId, "/best-run");
    var request = kind === "clear"
      ? selectionWrite(path, "DELETE", decisionBody({
          expected_selection_id: runs.expected_selection_id || undefined,
          reason: reason.value.trim() || undefined
        }))
      : selectionWrite(path, "POST", decisionBody({
          attempt: row.attempt,
          expected_selection_id: runs.expected_selection_id || undefined,
          reason: reason.value.trim() || undefined
        }));
    request.then(function (result) {
      if (result.ok) { showNotice(heading + ": recorded"); reload(); return; }
      send.disabled = false;
      status.textContent = "";
      renderSelectionRefusal(box, result, reload);
    }).catch(function (e) {
      send.disabled = false;
      status.textContent = e.message;
    });
  });
  var undecided = el("button", null, "Record that I looked and did not choose");
  undecided.type = "button";
  undecided.addEventListener("click", function () {
    undecided.disabled = true;
    status.textContent = "recording…";
    selectionWrite(
      taskSelectionPath(experimentId, taskId, "/best-run/undecided"), "POST",
      decisionBody({
        expected_selection_id: runs.expected_selection_id || undefined,
        candidate_attempt: row.attempt,
        reason: reason.value.trim() || undefined
      })
    ).then(function (result) {
      if (result.ok) { showNotice("Recorded as undecided"); reload(); return; }
      undecided.disabled = false;
      status.textContent = "";
      renderSelectionRefusal(box, result, reload);
    });
  });
  var row_ = el("div", "runDecision");
  row_.appendChild(send);
  row_.appendChild(undecided);
  row_.appendChild(status);
  box.appendChild(row_);
  actions.appendChild(box);
  send.focus();
}

/* A refused decision, in the words of whatever refused it.

   A 409 carrying `stale` is the one a reader has to act on: the pointer moved
   while this page was open, so nothing was overwritten and the reply names
   what is current. The explanation and the re-read are together because
   "somebody else chose meanwhile" is only useful next to the new state. */
function renderSelectionRefusal(container, result, reload) {
  var data = result.data || {};
  if (data.stale) {
    var box = selectionNote(container,
      "Nobody's decision was overwritten.",
      "This selection moved while the page was open, so the write was "
      + "refused instead of being applied on top of it.");
    if (data.current_experiment_id) {
      box.appendChild(el("div", null,
        "It now names experiment " + data.current_experiment_id + "."));
    }
    box.appendChild(el("div", "sub",
      "read as " + (data.expected_selection_id || "—")
      + ", now " + (data.current_selection_id || "nothing")));
    var again = el("button", "primary", "Re-read and decide again");
    again.type = "button";
    again.addEventListener("click", function () { reload(); });
    box.appendChild(again);
    return;
  }
  selectionNote(container, "That was refused.",
    data.error || ("Request failed (" + result.status + ")."));
}

function renderBestRunHistory(card, experimentId, taskId, nav) {
  var box = el("details", "card");
  box.appendChild(el("summary", null, "Best-run decisions"));
  card.appendChild(box);
  box.addEventListener("toggle", function () {
    if (!box.open || box.dataset.loaded) { return; }
    box.dataset.loaded = "1";
    selectionRead(taskSelectionPath(experimentId, taskId, "/runs/history?limit=50"))
      .then(function (data) {
        if (expNavStale(nav)) { return; }
        if (data.unavailable) {
          box.appendChild(el("div", "sub", data.error));
          return;
        }
        renderDecisionHistory(box, data.history || [],
                              "No best-run decision has been recorded.");
      }).catch(function (e) {
        box.dataset.loaded = "";
        box.appendChild(el("div", "sub", e.message));
      });
  });
}

/* Decisions, oldest facts intact.

   The append-only log is the stale-decision record too. A refused write is
   reported to the person who attempted it and never lands here; what DOES land
   is what each decision was taken against — the pointer it replaced and the
   pointer it produced — so "kept the winner" can be read as a judgement about
   a specific experiment rather than about whatever holds the contest now. */
function renderDecisionHistory(container, history, emptyText) {
  if (!history.length) {
    container.appendChild(el("div", "sub", emptyText));
    return;
  }
  history.forEach(function (row) {
    var item = el("div", "feedbackHistoryItem");
    var title = el("div", "title");
    title.appendChild(el("span", null, row.decision || "recorded"));
    /* "initial" is the automatic first-experiment election: nobody judged it,
       and the row must not read as though somebody did. */
    if (row.automatic || row.decision === "initial") {
      title.appendChild(el("span", "pill", "automatic"));
    }
    item.appendChild(title);
    var who = (row.created_at || row.decided_at || "")
      + " · " + (row.actor || "—")
      + (row.actor_kind ? " (" + feedbackProvenanceLabel(row.actor_kind) + ")" : "");
    item.appendChild(el("div", "sub", who));
    if (row.candidate_experiment_id) {
      item.appendChild(el("div", "sub",
        "candidate " + row.candidate_experiment_id
        + (row.candidate_attempt === null || row.candidate_attempt === undefined
            ? "" : " attempt " + row.candidate_attempt)));
    }
    var was = decisionPointerText(row, "previous");
    var now = decisionPointerText(row, "new");
    item.appendChild(el("div", "sub",
      "decided against " + (was || "nothing recorded")
      + " → " + (now || "nothing recorded")));
    var said = row.rationale || row.reason;
    if (said) { item.appendChild(el("pre", "json", said)); }
    container.appendChild(item);
  });
}

/* One side of a decision's before/after, in whichever scope it was taken: a
   winner decision names an experiment, a best-run decision an attempt. */
function decisionPointerText(row, side) {
  var experiment = row[side + "_experiment_id"];
  var task = row[side + "_task_id"];
  var attempt = row[side + "_attempt"];
  if (!experiment && (attempt === null || attempt === undefined)) { return ""; }
  var text = experiment || "this experiment";
  if (task) { text += " · " + task; }
  if (attempt !== null && attempt !== undefined) { text += " · attempt " + attempt; }
  return text;
}

