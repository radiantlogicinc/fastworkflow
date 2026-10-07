/* -- the experiment winner ------------------------------------------- */

/* The contest this experiment is in, and the decision controls for it.

   Rendered on both the recorded-experiment page and the registration page,
   because "which experiment is this workflow's current answer" is the same
   question whether or not the run has finished -- the first experiment of a
   contest wins at creation, before it has run anything.

   The badge and the STATUS are deliberately two things. Being the winner is a
   pointer somebody (or the automatic first-experiment rule) set; whether the
   experiment completed, is still running or is invalid is its own recorded
   fact, and collapsing them would let a pointer read as a pass. */
function renderWinnerPanel(container, experimentId, reload) {
  var card = el("div", "card");
  sectionHeader(card, "Winner of this contest");
  var body = el("div");
  card.appendChild(body);
  container.appendChild(card);
  body.appendChild(el("div", "sub", "reading the contest…"));
  selectionRead(selectionPath(experimentId, "/winner")).then(function (data) {
    clear(body);
    if (data.unavailable) {
      selectionNote(body, "This experiment is not in this workflow's contest.", data.error);
      return;
    }
    var winner = data.winner;
    var badges = el("div", "diffToolbar");
    badges.appendChild(el("span", data.is_winner ? "pill ok" : "pill",
                          data.is_winner ? "Winner" : "Not the winner"));
    if (data.automatic && data.is_winner) {
      badges.appendChild(el("span", "pill", "automatic — first experiment"));
    }
    if (winner && winner.experiment && winner.experiment.status) {
      badges.appendChild(expStatusPill(winner.experiment.status));
    }
    body.appendChild(badges);
    /* The one distinction a reader cannot recover from the badge: this is a
       pointer, and it is not the per-task best run. */
    body.appendChild(el("div", "sub",
      "This workflow's current answer for the benchmark — a pointer, not a "
      + "score. Task best runs are chosen separately and never move it."));
    if (data.automatic && data.is_winner) {
      body.appendChild(el("div", "sub",
        "Held since it was created, before anything ran."));
    }
    var kv = el("dl", "kv");
    function pair(k, v) {
      kv.appendChild(el("dt", null, k));
      kv.appendChild(el("dd", null,
        v === null || v === undefined || v === "" ? "—" : String(v)));
    }
    pair("current winner", winner ? winner.experiment_id : "nobody");
    pair("recorded status", winner && winner.experiment
      ? winner.experiment.status : "not recorded");
    pair("how it was decided", winner
      ? (winner.decision || "—")
        + (winner.decided_at ? " at " + winner.decided_at : "")
      : "—");
    pair("experiments in this contest", (data.members || []).length);
    pair("runs per task", data.runs_per_task);
    body.appendChild(kv);
    /* The pointer is a fact even when the run behind it cannot be read. Saying
       so beats an empty status that would read as "no status was recorded". */
    if (winner && winner.experiment_resolved === false) {
      body.appendChild(el("div", "sub",
        "The winner is recorded, but experiment " + winner.experiment_id
        + " has no recorded run in this workflow yet, so its current state is "
        + "unknown rather than absent."));
    }
    if (winner && winner.experiment_id !== experimentId) {
      var goto_ = el("button", "evidenceLink",
                     "Open the winning experiment's records");
      goto_.type = "button";
      goto_.addEventListener("click", function () {
        openBenchmarkExecution(winner.experiment_id);
      });
      body.appendChild(goto_);
    }
    renderWinnerDecision(body, experimentId, data, reload);
    renderWinnerHistory(body, experimentId);
  }).catch(function (e) {
    clear(body);
    body.appendChild(el("div", "sub", "Could not read the contest: " + e.message));
  });
}

/* Promote, keep or stay undecided, with an optional reason and no rubric.

   All three carry the selection id this panel was rendered from, which is what
   the contest requires: a "keep" recorded against a winner that was replaced
   while somebody was reading is a judgement about a different experiment than
   the one it would be filed under. A contest with no winner yet (experiments
   recorded before contests were kept in the live database) offers promotion
   alone, recorded against "nobody", which makes this experiment its first. */
function renderWinnerDecision(body, experimentId, data, reload) {
  var box = el("div");
  box.appendChild(el("h2", null, "Decide"));
  if (!data.expected_selection_id) {
    box.appendChild(el("div", "sub",
      "This contest has no winner yet. Promoting this experiment makes it the first."));
  }
  var reason = reasonField(box, "Reason (optional)");
  var status = el("div", "sub", "");
  var row = el("div", "runDecision");
  [["promote", "Promote this experiment", "primary"],
   ["keep", "Keep the current winner", null],
   ["undecided", "Leave it undecided", null]].filter(function (choice) {
    return data.expected_selection_id || choice[0] === "promote";
  }).forEach(function (choice) {
    var button = el("button", choice[2], choice[1]);
    button.type = "button";
    if (choice[0] === "promote" && data.is_winner) {
      button.disabled = true;
      button.title = "This experiment already holds the contest.";
    }
    button.addEventListener("click", function () {
      status.textContent = "recording…";
      selectionWrite(selectionPath(experimentId, "/winner/decisions"), "POST",
        decisionBody({
          decision: choice[0],
          expected_selection_id: data.expected_selection_id || null,
          /* Named on every decision, not only on a promotion: without it the
             history can say a winner was kept but not which challenger was
             turned down. */
          candidate_experiment_id: experimentId,
          rationale: reason.value.trim() || undefined
        })
      ).then(function (result) {
        if (result.ok) {
          showNotice("Decision recorded: " + choice[0]);
          if (reload) { reload(); }
          return;
        }
        status.textContent = "";
        renderSelectionRefusal(box, result, reload || function () {});
      }).catch(function (e) { status.textContent = e.message; });
    });
    row.appendChild(button);
  });
  row.appendChild(status);
  box.appendChild(row);
  body.appendChild(box);
}

function renderWinnerHistory(body, experimentId) {
  var box = el("details", "card");
  box.appendChild(el("summary", null, "Decisions in this contest"));
  body.appendChild(box);
  box.addEventListener("toggle", function () {
    if (!box.open || box.dataset.loaded) { return; }
    box.dataset.loaded = "1";
    selectionRead(selectionPath(experimentId, "/winner/history?limit=50"))
      .then(function (data) {
        if (data.unavailable) {
          box.appendChild(el("div", "sub", data.error));
          return;
        }
        renderDecisionHistory(box, data.history || [],
                              "No decision has been recorded in this contest.");
      }).catch(function (e) {
        box.dataset.loaded = "";
        box.appendChild(el("div", "sub", e.message));
      });
  });
}

/* -- running the same setup again (fix-9eg.17.2) ------------------------ */

/* Reuse this setup, optionally more than once per task.

   Duplication is REGISTRATION only: it copies the experiment record, gives it
   a new id and hands it back for a runner. Nothing paid starts here, and no
   target, rubric, hypothesis or review gate is required to ask for n attempts
   — the repeat count is the declared attempt count and nothing else. */
function renderRepeatSetup(container, row, nav) {
  var card = el("div", "card");
  sectionHeader(card, "Run this setup again");
  card.appendChild(el("div", "sub",
    "Copies this experiment's setup into a new registration you can hand to a "
    + "runner. Repeating a task is how a flaky result is told from a real one, "
    + "so the only thing it asks for is how many attempts per task. Creating "
    + "or duplicating an experiment never executes anything."));
  var current = row.runs_per_task === null || row.runs_per_task === undefined
    ? 1 : row.runs_per_task;
  card.appendChild(el("div", "sub",
    "This experiment declares " + current + " attempt(s) per task."));
  var field = el("label", "formField");
  field.appendChild(el("div", "sub", "Attempts per task for the copy"));
  var count = el("input", "expNotes");
  count.type = "number";
  count.min = "1";
  count.max = "100";
  count.step = "1";
  count.value = String(current);
  count.setAttribute("aria-label", "Attempts per task for the copy");
  field.appendChild(count);
  card.appendChild(field);
  var description = benchmarkText(card, "Description for the copy (optional)",
                                  row.description || "", true);
  var note = el("div", "sub", "");
  var go = el("button", "primary", "Duplicate this setup");
  go.type = "button";
  go.addEventListener("click", function () {
    note.textContent = "duplicating…";
    lastActionButton = go;
    apiPost("/api/benchmark-experiments/"
            + encodeURIComponent(row.experiment_id) + "/duplicate",
            { runs_per_task: count.value.trim(),
              description: description.value })
      .then(function (result) {
        if (benchNavStale(nav)) { return; }
        note.textContent = "";
        var changed = (result.changed_fields || []).join(", ");
        showNotice("Setup duplicated",
                   null, changed ? "changed: " + changed : undefined);
        refreshConvs().then(function () {
          if (benchNavStale(nav)) { return; }
          showBenchmarkExperiment(result.experiment.experiment_id);
        });
      }).catch(function (e) { note.textContent = e.message; });
  });
  card.appendChild(go);
  card.appendChild(note);
  container.appendChild(card);
}

/* -- the task's Compare view, and the one comparison component --------- */

/* Which pair is on screen. Scoped to one task: arriving at another task with
   "attempt 3" selected would name an attempt that task may not have. */
var taskCompare = { key: null, left: null, right: null, rightExperiment: null,
                    view: "answers", leftPass: null, rightPass: null };

function resetTaskCompare(experimentId, taskId) {
  var key = experimentId + "\u001f" + taskId;
  if (taskCompare.key === key) { return; }
  taskCompare = { key: key, left: null, right: null,
                  rightExperiment: experimentId, view: "answers",
                  leftPass: null, rightPass: null };
}

/* Called by the source switcher when the page changes which evidence database
   it is reading (see `resetSourceScopedState`).

   The identity above is experiment-and-task, and both recur across workflows:
   two live stores can hold "todo-list-v1" / "add an item" and mean different
   runs. Keeping the selected pair across a source change would show
   attempt numbers from the old database under the new one's label, which is the
   one way this view can lie. Dropping the key makes the next visit re-derive
   the pair from the new source's own runs list. */
function onSourceSwitch() {
  taskCompare = { key: null, left: null, right: null, rightExperiment: null,
                  view: "answers", leftPass: null, rightPass: null };
  trainingHistoryReset();
}

var COMPARE_VIEWS = [
  { value: "answers", label: "Answers and artifacts" },
  { value: "steps", label: "Every aligned step" },
  { value: "differences", label: "Differences only" }
];

/* How two steps were matched, in words. `recorded` is the only basis that came
   from the producer; the rest are this comparison's own key tiers, and saying
   which one matched is how a reader knows how much the row is claiming. */
var COMPARE_BASIS = {
  recorded: "matched by a recorded alignment",
  "command+context+parameters": "same command, context and parameters",
  "command+context": "same command and context",
  command: "same command only",
  unmatched: "no counterpart",
  unknown: "the evidence does not say what this step was"
};

function compareBasisLabel(basis) {
  return COMPARE_BASIS[basis] || basis || "unknown basis";
}

function compareKindLabel(kind) {
  if (kind === "matched") { return "both runs"; }
  if (kind === "left_only") { return "left only"; }
  if (kind === "right_only") { return "right only"; }
  return kind || "unpaired";
}

/* -- Consistency between the repeated runs of one task (fix-9eg.17.5) -------

   One compact band, on the Runs view and again under the Compare view, fed by
   GET .../tasks/<id>/consistency. Everything it prints is a value the route
   computed: the page performs no similarity, no averaging and no rounding of
   its own beyond display precision, so a coding agent reading the same JSON
   and a person reading this panel are looking at the same numbers with the
   same definitions and the same coverage.

   It never renders a verdict. There is no target line, no pass/fail colour and
   no "good"/"poor" wording, because agreement between runs is consistency and
   this product does not claim consistency is correctness. Clicking a pair
   opens the EXISTING comparison for those two attempts, which is where the
   existing feedback taxonomy already lets somebody say what they made of it. */
function renderConsistency(container, experimentId, taskId, label, nav, focus) {
  clear(container);
  container.appendChild(el("div", "sub", "measuring consistency…"));
  selectionRead(taskSelectionPath(experimentId, taskId, "/consistency"))
    .then(function (report) {
      if (expNavStale(nav)) { return; }
      clear(container);
      if (report.unavailable) {
        /* The route's own sentence. "this experiment joined no contest" is an
           answer; "Could not load" is not. */
        selectionNote(container,
          "Consistency cannot be measured for this task.", report.error);
        return;
      }
      paintConsistency(container, experimentId, taskId, label, report, focus);
    }).catch(function (e) {
      if (expNavStale(nav)) { return; }
      clear(container);
      container.appendChild(
        el("div", "sub", "Could not measure consistency: " + e.message));
    });
}

function fmtCosine(value) {
  return (value === null || value === undefined) ? "—" : value.toFixed(3);
}
function fmtStat(value, places) {
  return (value === null || value === undefined) ? "—" : value.toFixed(places);
}

function paintConsistency(container, experimentId, taskId, label, report, focus) {
  var box = el("div", "consistencyPanel");
  box.setAttribute("data-consistency", "1");
  box.appendChild(el("h3", null, "Consistency across repeated runs"));
  box.appendChild(el("div", "sub", report.interpretation));

  var identity = report.metric_identity || {};
  var embedding = identity.embedding || {};
  if (embedding.available === false) {
    /* Not an error and not a silent gap: the similarity metrics need a local
       model this machine does not have, the step metrics beside them do not,
       and the remedy is the route's own words rather than this page's guess. */
    selectionNote(box, "Text similarity is unavailable on this machine.",
      (embedding.reason || "") + (embedding.remedy ? " " + embedding.remedy : ""));
  }
  if (report.insufficient_repeats) {
    selectionNote(box, "Not enough repeats to compare.",
      report.insufficient_repeats_note);
  }

  var summary = report.summary || {};
  var stats = el("div", "consistencyStats");
  stats.appendChild(similarityStat("Planning similarity",
    summary.planning_similarity,
    "cosine between the recorded plan sequences, turn boundaries kept"));
  stats.appendChild(similarityStat("Final-answer similarity",
    summary.final_answer_similarity,
    "cosine between the last recorded turn's answers; similar wording is not "
    + "proof of equal facts or artifacts"));
  stats.appendChild(stepCountStat(summary.step_counts));
  box.appendChild(stats);
  box.appendChild(el("div", "sub", summary.independence_note || ""));

  var coverage = report.coverage || {};
  if (coverage.pairs_capped || coverage.runs_capped) {
    selectionNote(box, "This is a bounded calculation, not all of them.",
      "Reported " + fmtCount(coverage.pairs_reported) + " of "
      + fmtCount(coverage.pairs_possible) + " possible pairs"
      + ((coverage.runs_cap_detail && coverage.runs_cap_detail.attempts_omitted)
          ? ", over " + fmtCount(coverage.runs_cap_detail.attempts_reported)
            + " of " + fmtCount(coverage.runs_cap_detail.attempts_recorded)
            + " recorded attempts"
          : "")
      + ".");
  }

  var reference = report.reference || {};
  if (reference.usable && (report.reference_rows || []).length) {
    box.appendChild(el("div", "sub",
      "Best run (attempt " + reference.attempt + ") against each other run. "
      + "These rows follow the choice; the figures above do not."));
    report.reference_rows.forEach(function (pair) {
      box.appendChild(consistencyPairButton(
        experimentId, taskId, label, pair, focus));
    });
  } else if (reference.reason) {
    box.appendChild(el("div", "sub", reference.reason));
  }

  if ((report.pairs || []).length) {
    var all = el("details", "consistencyPairs");
    all.appendChild(el("summary", null,
      "Every pair (" + fmtCount(report.pairs.length) + ")"));
    report.pairs.forEach(function (pair) {
      all.appendChild(consistencyPairButton(
        experimentId, taskId, label, pair, focus));
    });
    box.appendChild(all);
  }

  if (report.comparison) { renderConsistencyComparison(box, report.comparison); }
  box.appendChild(el("div", "sub",
    "Measured as " + (identity.metrics_version || "?") + " / "
    + (identity.text_projection_version || "?")
    + (embedding.model_id ? " / " + embedding.model_id : "")
    + (embedding.revision ? "@" + String(embedding.revision).slice(0, 12) : "")
    + ". Steps counted by " + (identity.step_count_rule || "?") + "."
    + (embedding.long_text_rule ? " " + embedding.long_text_rule : "")));
  container.appendChild(box);
}

/* One similarity figure, with its denominator beside it.

   The denominator is not decoration: a mean of 0.95 over two eligible pairs
   out of fifteen is a different fact from the same mean over all fifteen, and
   printing only the mean is how the second gets read as the first. */
function similarityStat(title, stat, definition) {
  var node = el("div", "consistencyStat");
  node.appendChild(el("strong", null, title));
  node.appendChild(el("div", "sub", definition));
  if (!stat || !stat.pairs_computed) {
    node.appendChild(el("div", "figures", "no pair could be compared"));
    var reasons = (stat && stat.unknown_reasons) || {};
    Object.keys(reasons).forEach(function (reason) {
      node.appendChild(el("div", "sub", reasons[reason] + " × " + reason));
    });
    return node;
  }
  node.appendChild(el("div", "figures",
    "mean " + fmtCosine(stat.mean) + " · min " + fmtCosine(stat.min)
    + " · max " + fmtCosine(stat.max) + " · spread " + fmtCosine(stat.spread)));
  node.appendChild(el("div", "sub", stat.spread_formula));
  node.appendChild(el("div", "sub",
    fmtCount(stat.pairs_computed) + " of " + fmtCount(stat.pairs_considered)
    + " pairs compared"
    + (stat.pairs_unknown ? ", " + fmtCount(stat.pairs_unknown)
       + " had no text on one side" : "")
    + (stat.pairs_unavailable ? ", " + fmtCount(stat.pairs_unavailable)
       + " need the local model" : "")));
  if (stat.pairs_partial_coverage) {
    node.appendChild(el("div", "sub",
      fmtCount(stat.pairs_partial_coverage) + " computed over a shortened "
      + "projection, so they are not similarities of the full text"));
  }
  return node;
}

function stepCountStat(stat) {
  var node = el("div", "consistencyStat");
  node.appendChild(el("strong", null, "Executed steps"));
  node.appendChild(el("div", "sub",
    "distinct dispatches in the execution ledger, including failures, retries "
    + "and navigation; wrappers and repeated span updates count once"));
  if (!stat || !stat.runs_with_known_count) {
    node.appendChild(el("div", "figures", "no run has a known step count"));
    return node;
  }
  node.appendChild(el("div", "figures",
    "mean " + fmtStat(stat.mean, 2) + " · min " + stat.min + " · max "
    + stat.max + " · population SD " + fmtStat(stat.population_sd, 2)));
  node.appendChild(el("div", "sub", stat.sd_formula));
  /* Unknown is not zero, and the attempts are named so the reader can open
     the ones that have no count rather than wonder which they were. */
  if ((stat.attempts_unknown || []).length) {
    node.appendChild(el("div", "sub",
      "no count recorded for attempt(s) " + stat.attempts_unknown.join(", ")
      + " — unknown, not zero"));
  }
  if ((stat.attempts_partial || []).length) {
    node.appendChild(el("div", "sub",
      "attempt(s) " + stat.attempts_partial.join(", ")
      + " could only be read in part, so their counts are excluded here"));
  }
  return node;
}

/* One pair, and a way into the comparison it describes.

   The click sets exactly what the Compare view's own controls set, so the pair
   opens in the existing component with the existing pair composer — the same
   Observations / Analysis, Conclusions and Recommendations taxonomy, against
   the same review pair key this row was counted under. */
function consistencyPairButton(experimentId, taskId, label, pair, focus) {
  var button = el("button", "consistencyPair");
  button.type = "button";
  button.setAttribute("data-pair",
    String(pair.left_attempt) + ":" + String(pair.right_attempt));
  if (focus && String(focus.left) === String(pair.left_attempt)
      && String(focus.right) === String(pair.right_attempt)) {
    button.className = "consistencyPair pairFocused";
  }
  button.appendChild(el("div", null,
    "attempt " + pair.left_attempt + " vs attempt " + pair.right_attempt));
  button.appendChild(el("div", "sub",
    "plan " + pairMetricText(pair.planning)
    + " · answer " + pairMetricText(pair.final_answer)
    + " · " + pairStepText(pair.steps)));
  button.addEventListener("click", function () {
    taskCompare.left = String(pair.left_attempt);
    taskCompare.right = String(pair.right_attempt);
    taskCompare.rightExperiment = pair.right_experiment_id || experimentId;
    taskView = "compare";
    showExperimentTask(experimentId, taskId, label);
  });
  return button;
}

function pairMetricText(metric) {
  if (!metric) { return "—"; }
  if (metric.state === "computed") {
    return fmtCosine(metric.cosine)
      + (metric.coverage === "partial" ? " (partial)" : "");
  }
  if (metric.state === "unavailable") { return "needs the local model"; }
  /* Which side was missing, not just that something was. */
  return metric.reason || "unknown";
}

function pairStepText(steps) {
  if (!steps || steps.state !== "computed") {
    return "steps " + ((steps && steps.reason) || "unknown");
  }
  var delta = steps.delta > 0 ? "+" + steps.delta : String(steps.delta);
  return "steps " + steps.left + " vs " + steps.right + " (" + delta + ", "
    + steps.normalized_difference_pct.toFixed(1) + "%)";
}

/* The same task under two experiments. Refused deltas are printed as refusals
   with the mismatch that caused them, because a number nobody may subtract is
   more dangerous than a blank. */
function renderConsistencyComparison(box, comparison) {
  var node = el("div", "consistencyStat");
  node.appendChild(el("strong", null, "Against "
    + String((comparison.baseline || {}).experiment_id || "").slice(-8)));
  node.appendChild(el("div", "sub", comparison.note));
  node.appendChild(el("div", "sub",
    "repeats: " + fmtCount((comparison.baseline || {}).runs || 0) + " vs "
    + fmtCount((comparison.candidate || {}).runs || 0)));
  if (!comparison.metric_identity_matches) {
    selectionNote(node, "These two cannot be subtracted.",
      "They were not measured the same way: "
      + (comparison.mismatches || []).join("; "));
  }
  var deltas = comparison.deltas || {};
  Object.keys(deltas).forEach(function (name) {
    var delta = deltas[name];
    node.appendChild(el("div", delta.state === "computed" ? "figures" : "sub",
      name + ": " + (delta.state === "computed"
        ? (delta.delta >= 0 ? "+" : "") + delta.delta.toFixed(3)
        : (delta.reason || "not comparable"))));
  });
  box.appendChild(node);
}

function renderTaskCompare(container, experimentId, taskId, nav) {
  var card = el("div", "card");
  card.appendChild(el("h2", null, "Compare two recorded runs"));
  card.appendChild(el("div", "sub",
    "Two runs of this task side by side: what each answered, then the "
    + "artifacts, then how they got there."));
  var controls = el("div", "selectionBar");
  var body = el("div");
  /* The consistency band sits below the pair, in its own container, and is
     repainted with the pair so the row for the two runs on screen is the
     highlighted one. Kept out of `body` because `paint()` clears that. */
  var consistency = el("div");
  card.appendChild(controls);
  card.appendChild(body);
  card.appendChild(consistency);
  container.appendChild(card);
  body.appendChild(el("div", "empty", "loading runs…"));

  /* `storeId` is THIS page's database, read off the task's own runs. A link on
     either side compares against it to decide whether opening a trace changes
     the store scope at all — and an ad-hoc experiment with no authoring
     registration keeps working because nothing is scoped that need not be. */
  var context = { experimentId: experimentId, taskId: taskId, winnerId: null,
                  storeId: null, comparison: null,
                  reload: function () { paint(); } };

  Promise.all([
    selectionRead(taskSelectionPath(experimentId, taskId, "/runs")),
    selectionRead(selectionPath(experimentId, "/winner"))
  ]).then(function (results) {
    if (expNavStale(nav)) { return; }
    var runs = results[0], contest = results[1];
    if (runs.unavailable) {
      clear(body);
      selectionNote(body,
        "This task's runs cannot be listed, so there is nothing to compare.",
        runs.error);
      return;
    }
    context.storeId = runs.store_id || null;
    context.winnerId = (!contest.unavailable && contest.winner)
      ? contest.winner.experiment_id : null;
    var members = (!contest.unavailable && contest.members) ? contest.members : [];
    if (taskCompare.rightExperiment === null) {
      taskCompare.rightExperiment = experimentId;
    }
    renderCompareControls(controls, experimentId, taskId, runs, members, paint);
    paint();
  }).catch(function (e) {
    if (expNavStale(nav)) { return; }
    clear(body);
    body.appendChild(el("div", "empty", "Could not load runs: " + e.message));
  });

  function paint() {
    clear(body);
    body.appendChild(el("div", "empty", "comparing…"));
    var query = "?view=" + encodeURIComponent(taskCompare.view);
    if (taskCompare.left !== null) { query += "&left_attempt=" + taskCompare.left; }
    if (taskCompare.right !== null) { query += "&right_attempt=" + taskCompare.right; }
    if (taskCompare.rightExperiment
        && taskCompare.rightExperiment !== experimentId) {
      query += "&right_experiment="
        + encodeURIComponent(taskCompare.rightExperiment);
    }
    if (taskCompare.leftPass) {
      query += "&left_pass=" + encodeURIComponent(taskCompare.leftPass);
    }
    if (taskCompare.rightPass) {
      query += "&right_pass=" + encodeURIComponent(taskCompare.rightPass);
    }
    selectionRead(taskSelectionPath(experimentId, taskId, "/comparison" + query))
      .then(function (cmp) {
        if (expNavStale(nav)) { return; }
        clear(body);
        if (cmp.unavailable) {
          /* The route's own words. "attempt 4 recorded no turns" is the
             answer; an empty two-pane layout is not. */
          selectionNote(body, "These two cannot be compared.", cmp.error);
          clear(consistency);
          return;
        }
        renderRunComparison(body, cmp, context);
        /* The band annotates the pair, so it is fetched once the pair is on
           screen rather than beside it. Issued together, the two requests
           compete and the slower one (this: it embeds text) delays the
           comparison the reader actually asked for. */
        renderConsistency(consistency, experimentId, taskId, null, nav,
          { left: taskCompare.left, right: taskCompare.right });
      }).catch(function (e) {
        if (expNavStale(nav)) { return; }
        clear(body);
        clear(consistency);
        body.appendChild(el("div", "empty", "Could not compare: " + e.message));
      });
  }
}

/* The pair picker. Both sides list EVERY recorded attempt; the ones with no
   recorded turns are present and disabled, carrying the evidence's own reason,
   because an attempt silently missing from this list is indistinguishable from
   one that never ran. */
function renderCompareControls(controls, experimentId, taskId, runs, members,
                               paint) {
  clear(controls);
  function attemptOptions(rows) {
    var options = [{ value: "", label: "the best run, or the Reference" }];
    (rows || []).forEach(function (row) {
      options.push({
        value: row.attempt, label: attemptOptionLabel(row),
        disabled: row.comparable === false
      });
    });
    return options;
  }
  labelledSelect(controls, "Left run", attemptOptions(runs.attempts),
    taskCompare.left === null ? "" : taskCompare.left,
    function (value) {
      /* Kept as the exact decimal text the option carried. The route parses
         only that form, so there is no client-side int()/float() to truncate
         "2.9" into a reference to somebody else's run. */
      taskCompare.left = value === "" ? null : value;
      taskCompare.leftPass = null;
      paint();
    });
  if (members.length > 1) {
    var experimentOptions = members.map(function (row) {
      return {
        value: row.experiment_id,
        label: "Experiment · " + String(row.experiment_id).slice(-8)
          + (row.experiment_id === experimentId ? " (this one)" : "")
      };
    });
    labelledSelect(controls, "Right experiment", experimentOptions,
      taskCompare.rightExperiment || experimentId,
      function (value) {
        taskCompare.rightExperiment = value;
        taskCompare.right = null;
        taskCompare.rightPass = null;
        reloadRight();
      });
  }
  var rightRow = el("span");
  controls.appendChild(rightRow);
  labelledSelect(controls, "View", COMPARE_VIEWS, taskCompare.view,
    function (value) { taskCompare.view = value; paint(); });
  reloadRight();

  function reloadRight() {
    clear(rightRow);
    var target = taskCompare.rightExperiment || experimentId;
    if (target === experimentId) {
      rightAttemptSelect(runs.attempts);
      paint();
      return;
    }
    rightRow.appendChild(el("span", "sub", "loading the other experiment…"));
    selectionRead(taskSelectionPath(target, taskId, "/runs")).then(function (other) {
      clear(rightRow);
      if (other.unavailable) {
        rightRow.appendChild(el("span", "sub", other.error));
        return;
      }
      rightAttemptSelect(other.attempts);
      paint();
    }).catch(function (e) {
      clear(rightRow);
      rightRow.appendChild(el("span", "sub", e.message));
    });
  }

  function rightAttemptSelect(rows) {
    labelledSelect(rightRow, "Right run", attemptOptions(rows),
      taskCompare.right === null ? "" : taskCompare.right,
      function (value) {
        taskCompare.right = value === "" ? null : value;
        taskCompare.rightPass = null;
        paint();
      });
  }
}

/* ONE comparison component for every pair this product compares: the contest
   winner against a candidate, a task's best run against another attempt, and a
   recorded teacher pass against a student pass. They share it because the
   question is the same one in every case — what did each produce, and how did
   it get there — and because a second implementation would be a second place
   for the answer and the alignment to disagree.

   The order is the owner's: the ANSWER first, then the artifacts, then the plan
   and execution as aligned rows a reader can drill into. */
function renderRunComparison(container, cmp, ctx) {
  clear(container);
  /* Every side lookup below reads its store and experiment off this payload's
     references, so a pane never inherits the other side's source. */
  ctx.comparison = cmp;
  var head = el("div", "diffToolbar");
  head.appendChild(runSideBadges("Left", cmp.left_run, cmp.left, ctx));
  head.appendChild(el("span", null, "versus"));
  head.appendChild(runSideBadges("Right", cmp.right_run, cmp.right, ctx));
  container.appendChild(head);
  renderCaptureLabels(container, cmp);
  renderPassControls(container, cmp, ctx);

  container.appendChild(el("h2", null, "Answers"));
  var answers = el("div", "comparePanes");
  [["left", cmp.left, cmp.left_run], ["right", cmp.right, cmp.right_run]]
    .forEach(function (side) {
      answers.appendChild(renderAnswerPane(side[0], side[1], side[2], ctx));
    });
  container.appendChild(answers);

  container.appendChild(el("h2", null, "Artifacts"));
  var artifacts = el("div", "comparePanes");
  [["left", cmp.left, cmp.left_run], ["right", cmp.right, cmp.right_run]]
    .forEach(function (side) {
      artifacts.appendChild(renderArtifactPane(side[0], side[1], side[2], ctx));
    });
  container.appendChild(artifacts);

  renderPairReview(container, cmp, ctx);
  renderComparisonSummary(container, cmp);
  renderCommandSummary(container, cmp, ctx);
  renderComparisonCallCosts(container, cmp, ctx);
  renderAlignment(container, cmp, ctx);
}

/* One side's chrome. The recorded state travels with the label, so a pane
   pinned as "the best run" still says it failed if it failed, and a pane is
   never called "teacher" unless a recorded pass id says so. */
function runSideBadges(which, run, projection, ctx) {
  var box = el("span", "diffToolbar");
  var name = which + ": attempt " + (run.attempt === undefined ? "—" : run.attempt);
  box.appendChild(el("strong", null, name));
  if (run.experiment_id && run.experiment_id !== ctx.experimentId) {
    box.appendChild(el("span", "pill",
      "Experiment · " + String(run.experiment_id).slice(-8)));
  }
  if (ctx.winnerId && run.experiment_id === ctx.winnerId) {
    box.appendChild(el("span", "pill ok", "Winner"));
  }
  if (run.is_best) { box.appendChild(el("span", "pill ok", "Best run")); }
  if (run.is_reference) { box.appendChild(el("span", "pill", "Reference")); }
  if (run.execution_status) {
    box.appendChild(el("span", executionPill(run.execution_status),
                       run.execution_status));
  }
  if (run.outcome) { box.appendChild(el("span", "sub", run.outcome)); }
  var pass = projection && projection.ref && projection.ref.pass_id;
  if (pass) { box.appendChild(el("span", "pill", "pass " + pass)); }
  return box;
}

/* What was captured, and what was not. A half-readable side is still worth
   looking at, so it renders — with the count of turns that could not be read
   beside it, because the alternative is a shorter list that reads as a
   shorter run. */
function renderCaptureLabels(container, cmp) {
  [["Left", cmp.left], ["Right", cmp.right]].forEach(function (side) {
    var projection = side[1] || {};
    var missing = (projection.unavailable || []).length;
    if (missing) {
      selectionNote(container,
        side[0] + " side: partial capture.",
        missing + " turn(s) this run's reference names could not be read, so "
        + "this side is what survived rather than the whole run.");
    }
    if (projection.readable === false) {
      selectionNote(container, side[0] + " side: nothing readable.",
        "None of the turns this reference names is in its evidence store.");
    }
  });
  var summary = cmp.summary || {};
  if (summary.degraded) {
    selectionNote(container, "The step alignment was degraded.",
      summary.degraded_reason
      || "This run was long enough that the exact alignment was abandoned for "
         + "a cheaper one; rows may pair differently than a full pass would.");
  }
}

/* A pass selector only where the EVIDENCE stamped a pass.

   A distilled turn stamps two (`fix-txxy`, teacher and student); an ordinary
   turn stamps none, so this asks once per side and renders nothing at all in
   that case. It never offers a pass a client would then have to assert
   membership for. */
function renderPassControls(container, cmp, ctx) {
  var sides = [
    ["Left", cmp.left_run, "leftPass", ctx.experimentId],
    ["Right", cmp.right_run, "rightPass",
     (cmp.right_run && cmp.right_run.experiment_id) || ctx.experimentId]
  ];
  var row = el("div", "selectionBar");
  container.appendChild(row);
  sides.forEach(function (side) {
    var run = side[1] || {};
    if (run.attempt === undefined || run.attempt === null) { return; }
    selectionRead(taskSelectionPath(side[3], ctx.taskId,
      "/runs/" + encodeURIComponent(run.attempt) + "/passes"))
      .then(function (data) {
        if (data.unavailable || !(data.passes || []).length) { return; }
        var options = [{ value: "", label: "the whole turn" }];
        data.passes.forEach(function (pass) {
          options.push({
            value: pass.pass_id,
            label: pass.pass_id + " (" + fmtCount(pass.turn_count) + " turn(s))"
          });
        });
        labelledSelect(row, side[0] + " recorded pass", options,
          taskCompare[side[2]] || "",
          function (value) {
            taskCompare[side[2]] = value || null;
            ctx.reload();
          });
      }).catch(function () { /* a pass selector is an extra, never a blocker */ });
  });
}

/* What a pass recorded about itself, beside the answer it produced.

   All three of these are absent unless a producer recorded pass content, and
   all three stay absent rather than being filled in from the turn: a pass that
   recorded nothing reads as shared, which is what it is. */

function appendRecordedEnvelope(block, turn) {
  /* BOTH recorded fields: either can be capped by the tracing attribute limit.
     Showing a prefix with no badge presents part of an answer as the whole of
     one. */
  var content = turn.pass_content || {};
  [["answer", "answer"], ["plan", "plan"]].forEach(function (field) {
    var recorded = content[field[0]];
    if (recorded && typeof recorded === "object" && recorded.truncated) {
      /* BYTES: `tracing.cap_attr_value` encodes to UTF-8 and measures the
         encoding, so calling these characters overstates the value by however
         much of it was not ASCII. */
      block.appendChild(el("div", "chipMarker unknown",
        "this pass's " + field[1] + " was truncated on recording — "
        + (recorded.original_length === undefined || recorded.original_length === null
            ? "original size not recorded"
            : fmtCount(recorded.original_length) + " bytes were produced")));
    }
  });
}

function appendPassPlan(block, turn) {
  if (!turn.plan) { return; }
  var box = el("details", "artifactPreview");
  box.appendChild(el("summary", null, "Plan this pass generated"));
  var steps = null;
  try { steps = JSON.parse(turn.plan); } catch (err) { steps = null; }
  if (!Array.isArray(steps)) {
    /* Recorded text that is not the shape this reader expects is shown
       verbatim rather than guessed at or dropped. */
    box.appendChild(document.createTextNode(turn.plan));
  } else if (!steps.length) {
    box.appendChild(el("div", "sub", "This pass recorded no planning step."));
  } else {
    steps.forEach(function (step) {
      var lines = (step && step.generated_plan) || [];
      box.appendChild(el("div", "sub",
        "planning step " + ((step && step.step_number) === undefined
          ? "(unnumbered)" : step.step_number)));
      box.appendChild(document.createTextNode(
        Array.isArray(lines) ? lines.join("\n") : String(lines)));
    });
  }
  block.appendChild(box);
}

function appendSharedTurnAnswer(block, turn, answer) {
  if (!turn.turn_answer || turn.turn_answer === answer.answer) { return; }
  /* Both passes share one turn row, and it says something different from what
     this pass said. Kept beside the pass's own answer, never in place of it. */
  var box = el("details", "artifactPreview");
  box.appendChild(el("summary", null,
    "The turn recorded a different answer"
    + (turn.turn_status ? " (" + turn.turn_status + ")" : "")));
  box.appendChild(document.createTextNode(turn.turn_answer));
  block.appendChild(box);
}

function renderAnswerPane(which, projection, run, ctx) {
  var side = pairSide(ctx.comparison, which);
  var pane = el("div", "comparePane");
  pane.appendChild(el("h3", null,
    which === "left" ? "Left" : "Right"));
  var rows = (projection && projection.answers) || [];
  /* The turn rows carry what the answer rows cannot: the plan this pass
     generated, the shared turn answer its own answer is being distinguished
     from, and the recorded envelope that says whether what is shown is the
     whole thing. Joined by turn index, the one identity both halves of the
     payload state. */
  var turnsByIndex = {};
  ((projection && projection.turns) || []).forEach(function (turn) {
    turnsByIndex[turn.turn_index] = turn;
  });
  if (!rows.length) {
    pane.appendChild(el("div", "sub", "No recorded turn on this side."));
    return pane;
  }
  rows.forEach(function (answer) {
    var block = el("div");
    var meta = el("div", "sub",
      "turn " + answer.turn_index + " · " + (answer.status || "no status"));
    if (answer.failure_reason) {
      meta.appendChild(el("span", "chipMarker bad", answer.failure_reason));
    }
    block.appendChild(meta);
    var turn = turnsByIndex[answer.turn_index] || {};
    /* Verbatim, through textContent. The recorded answer is the author's and is
       never re-wrapped, summarised or parsed on its way to the screen. */
    block.appendChild(document.createTextNode(
      answer.answer === null || answer.answer === undefined
        ? "(no answer recorded)" : answer.answer));
    if (answer.attribution === "pass") {
      /* Recorded AS this pass's own (`fix-txxy`). Every non-turn attribution
         used to be labelled "shared", which denied exactly the divergence a
         teacher/student comparison is opened to look at. */
      block.appendChild(el("div", "chipMarker",
        "recorded for this pass"
        + (answer.pass_id ? " — " + answer.pass_id : "")));
      appendRecordedEnvelope(block, turn);
      appendPassPlan(block, turn);
      appendSharedTurnAnswer(block, turn, answer);
    } else if (answer.attribution && answer.attribution !== "turn") {
      /* Still the honest label for every trace recorded before a producer
         stamped its passes, and for a pass that recorded no content of its
         own: the text above is the turn's and both passes share it. */
      block.appendChild(el("div", "chipMarker unknown",
        "shared across passes — this text is the turn's, not this pass's"));
    }
    var open = el("button", "evidenceLink", "Open this turn");
    open.type = "button";
    var linkNote = el("div", "sub", "");
    open.addEventListener("click", function () {
      openPairTurn(ctx, side, answer.turn_key, linkNote);
    });
    block.appendChild(open);
    block.appendChild(linkNote);
    pane.appendChild(block);
  });
  return pane;
}

/* Artifacts, inspectable where they are being compared.

   The answer is often a pointer at the artifact, so sending a reader away to a
   trace view to see it loses the other side. Each row expands in place, through
   the SAME safe renderer the turn view uses — `artifactNode`, which puts
   anything HTML-ish in a sandboxed srcdoc iframe with its own
   `default-src 'none'` and prints everything else as text. No markup from a
   record reaches this page any other way, and the sandbox is not relaxed.

   Loading is explicitly scoped to the side's own database and explicitly
   bounded: a large value is named and left unloaded until somebody asks for it,
   and a value that is gone says whether it was pruned or simply not readable
   rather than rendering as empty. */
var ARTIFACT_PREVIEW_MAX_BYTES = 262144;

function renderArtifactPane(which, projection, run, ctx) {
  var side = pairSide(ctx.comparison, which);
  var pane = el("div", "comparePane");
  pane.appendChild(el("h3", null, which === "left" ? "Left" : "Right"));
  var rows = (projection && projection.artifacts) || [];
  var orphans = (projection && projection.unattributed_artifacts) || [];
  if (!rows.length && !orphans.length) {
    pane.appendChild(el("div", "sub", "No artifact recorded on this side."));
    return pane;
  }
  function artifactRow(artifact, note) {
    var box = el("div", "artifact");
    var head = el("div", "aHead");
    head.appendChild(el("span", "aKey", artifact.key));
    head.appendChild(el("span", "aMeta",
      (artifact.inline ? "inline" : "offloaded")
      + (artifact.content_type ? " · " + artifact.content_type : "")
      + (artifact.size_bytes === null || artifact.size_bytes === undefined
          ? "" : " · " + artifact.size_bytes + " bytes")));
    box.appendChild(head);
    box.appendChild(el("div", "aMeta",
      (artifact.command_name || "no command recorded")
      + " · " + (artifact.attribution || "turn")));
    if (artifact.error) {
      /* The record's own words about why the value is not here. */
      box.appendChild(el("div", "err", artifact.error));
    }
    if (note) { box.appendChild(el("div", "aMeta", note)); }
    var preview = el("details", "artifactPreview");
    preview.appendChild(el("summary", null, "Show this artifact here"));
    var host = el("div");
    preview.appendChild(host);
    preview.addEventListener("toggle", function () {
      if (!preview.open || preview.dataset.loaded) { return; }
      preview.dataset.loaded = "1";
      loadArtifactPreview(host, artifact, side, ctx, false);
    });
    box.appendChild(preview);
    var open = el("button", "evidenceLink", "Open the turn that recorded it");
    open.type = "button";
    var linkNote = el("div", "sub", "");
    open.addEventListener("click", function () {
      openPairTurn(ctx, side, artifact.turn_key, linkNote);
    });
    box.appendChild(open);
    box.appendChild(linkNote);
    pane.appendChild(box);
  }
  rows.forEach(function (artifact) { artifactRow(artifact, null); });
  orphans.forEach(function (artifact) {
    artifactRow(artifact,
      "recorded no command call, so it belongs to neither pass — it is not "
      + "claimed as this one's");
  });
  return pane;
}

/* One artifact's value, from the side's own store.

   Read through `pairReadScope`, the same rule the deep links use: a side in
   the current database needs no scope named. An inline value is lifted out of
   the turn record; offloaded bytes are served by the artifact endpoint. */
function loadArtifactPreview(host, artifact, side, ctx, forced) {
  clear(host);
  var size = artifact.size_bytes;
  if (!forced && typeof size === "number" && size > ARTIFACT_PREVIEW_MAX_BYTES) {
    host.appendChild(el("div", "aMeta",
      "This artifact is " + fmtCount(size) + " bytes, larger than the "
      + fmtCount(ARTIFACT_PREVIEW_MAX_BYTES) + " this pane loads without being "
      + "asked. It is recorded; it is simply not fetched yet."));
    var anyway = el("button", "evidenceLink", "Load it anyway");
    anyway.type = "button";
    anyway.addEventListener("click", function () {
      loadArtifactPreview(host, artifact, side, ctx, true);
    });
    host.appendChild(anyway);
    return;
  }
  var scope = pairReadScope(ctx, side);
  if (scope.kind === "unaddressable") {
    host.appendChild(el("div", "aMeta", scope.why));
    return;
  }
  host.appendChild(el("div", "aMeta", "loading…"));
  /* Offloaded bytes: only the live artifact endpoint serves them. */
  if (artifact.artifact_id) {
    apiRaw("/api/artifact/" + encodeURIComponent(artifact.artifact_id))
      .then(function (r) {
        if (r.status === 404) {
          clear(host);
          host.appendChild(el("div", "aMeta",
            "The record names this artifact, but its value is no longer in the "
            + "store — pruned by retention or never offloaded."));
          return null;
        }
        if (!r.ok) { throw new Error("HTTP " + r.status); }
        var ctype = (r.headers.get("Content-Type") || "").toLowerCase();
        return r.text().then(function (text) {
          clear(host);
          /* The same renderer the turn view uses: it decides HTML-ish from the
             text itself and sandboxes it, so a content type claiming
             text/plain cannot smuggle markup into this page. */
          host.appendChild(artifactNode(artifact.key, text,
            "offloaded · " + (ctype || "no content type")));
          return null;
        });
      }).catch(function (e) {
        clear(host);
        host.appendChild(el("div", "err", "Could not read it: " + e.message));
      });
    return;
  }
  selectionRead("/api/turn/" + encodeURIComponent(artifact.turn_key)).then(function (data) {
      clear(host);
      if (data.unavailable) {
        host.appendChild(el("div", "aMeta",
          "The turn holding this value could not be read from here: "
          + data.error));
        return;
      }
      var value = inlineArtifactValue(data.turn, artifact);
      if (value === undefined) {
        host.appendChild(el("div", "aMeta",
          "The turn was read, but it no longer records a value under this "
          + "key — it was not retained."));
        return;
      }
      host.appendChild(artifactNode(artifact.key, value, "inline"));
    }).catch(function (e) {
      clear(host);
      host.appendChild(el("div", "err", "Could not read it: " + e.message));
    });
}

/* The recorded value of one key, found the way the record files it: under the
   command output whose call id the artifact reference names. The call id is
   matched first so two commands that wrote the same key are not confused. */
function inlineArtifactValue(turn, artifact) {
  var record = (turn && turn.record) || {};
  var outputs = ((record.turn_output || {}).command_outputs) || [];
  var fallback;
  for (var i = 0; i < outputs.length; i += 1) {
    var artifacts = ((outputs[i].command_response || {}).artifacts) || {};
    if (!Object.prototype.hasOwnProperty.call(artifacts, artifact.key)) { continue; }
    if (artifact.command_call_id
        && outputs[i].command_call_id === artifact.command_call_id) {
      return artifacts[artifact.key];
    }
    if (fallback === undefined) { fallback = artifacts[artifact.key]; }
  }
  return fallback;
}

/* One side of the pair, named by the evidence rather than by the screen.

   `store_id` comes from the projection's own reference, so "which database is
   this side in" is a recorded fact and not a guess from the experiment id or
   from whichever source the page is currently pointed at. */
function pairSide(cmp, which) {
  var projection = which === "left" ? cmp.left : cmp.right;
  var run = which === "left" ? cmp.left_run : cmp.right_run;
  var ref = (projection && projection.ref) || {};
  return {
    which: which, run: run || {}, projection: projection || {},
    storeId: ref.store_id || null,
    experimentId: ref.experiment_id || (run && run.experiment_id) || null
  };
}

/* WHERE one side of a pair is read from — decided once, for every link and
   every inline preview, from the recorded reference rather than from the
   experiment id or from whichever source the page happens to be pointed at.

   Two answers, and the difference between them is the whole point:

   - `current`: the side is in the database this page is already reading, so no
     change of scope.
   - `unaddressable`: the reference names no way in, including a side recorded
     in another live database, which this build does not read. Said out loud,
     because a silent no-op reads as a broken link. */
function pairReadScope(ctx, side) {
  if (!side.storeId || !ctx.storeId || side.storeId === ctx.storeId) {
    return { kind: "current" };
  }
  return { kind: "unaddressable",
           why: "This side's evidence is in another database, which this "
                + "workflow does not read." };
}

/* Open the turn a row points at, in the database that RECORDED it. */
function openPairTurn(ctx, side, turnKey, note, spanId) {
  var scope = pairReadScope(ctx, side);
  if (scope.kind === "unaddressable") {
    if (note) { note.textContent = scope.why; }
    return;
  }
  selectTurn(turnKey, spanId, note);
}

/* ONE exact recorded call, on the side that recorded it, by the canonical span
   id the projection published for it.

   The shared entry point for every pane that lists contributing calls or steps:
   the source is resolved from the side's own reference (`openPairTurn`), and the
   span is focused inside the stale-guarded render of the trace that holds it
   (`focusLoadedSpan`). Nothing here polls, so no link can focus a same-named
   span in another store, and a span the evidence does not hold is reported as
   absent rather than quietly becoming the top of a turn. */
function openPairSpan(ctx, side, turnKey, spanId, note) {
  openPairTurn(ctx, side, turnKey, note, spanId);
}

/* Review progress and comment count, kept apart on purpose.

   "Nobody has marked this reviewed" and "nobody has commented on it" are
   different facts: a reviewer can look and have nothing to say, and a comment
   can be written by somebody who never marked anything. Showing one as the
   other is how a pair gets counted as unseen. */
function renderPairReview(container, cmp, ctx) {
  var card = el("div", "card");
  card.appendChild(el("h2", null, "This pair"));
  card.appendChild(el("div", "sub", "pair " + cmp.review_pair_key));
  var state = el("div", "sub", "reading review progress…");
  var counted = el("div", "sub", "counting recorded comments…");
  card.appendChild(state);
  card.appendChild(counted);
  container.appendChild(card);
  var query = "?reviewer=" + encodeURIComponent(SELECTION_ACTOR.actor);
  if (cmp.left_run && cmp.left_run.attempt !== undefined
      && cmp.left_run.attempt !== null) {
    query += "&left_attempt=" + cmp.left_run.attempt;
  }
  if (cmp.right_run && cmp.right_run.experiment_id
      && cmp.right_run.experiment_id !== ctx.experimentId) {
    query += "&right_experiment="
      + encodeURIComponent(cmp.right_run.experiment_id);
  }
  selectionRead(taskSelectionPath(ctx.experimentId, ctx.taskId,
                                  "/review-pairs" + query))
    .then(function (data) {
      if (data.unavailable) { state.textContent = data.error; return; }
      var mine = (data.pairs || []).filter(function (row) {
        return row.pair_key === cmp.review_pair_key;
      })[0];
      var reviewed = mine && mine.state && mine.state.state === "reviewed";
      clear(state);
      state.appendChild(el("span", reviewed ? "pill ok" : "pill",
                           reviewed ? "you marked this reviewed"
                                    : "you have not marked this reviewed"));
      state.appendChild(el("span", null,
        "  " + (data.progress ? data.progress.reviewed : 0) + " of "
        + (data.progress ? data.progress.pairs : 0)
        + " pairs of this task marked reviewed by you."));
      var mark = el("button", null,
        reviewed ? "Un-mark this pair" : "Mark this pair reviewed");
      mark.type = "button";
      mark.addEventListener("click", function () {
        mark.disabled = true;
        selectionWrite(
          taskSelectionPath(ctx.experimentId, ctx.taskId, "/review-pairs"),
          "POST",
          {
            reviewer: SELECTION_ACTOR.actor, reviewer_kind: "human",
            state: reviewed ? "not_reviewed" : "reviewed",
            left_attempt: cmp.left_run ? cmp.left_run.attempt : undefined,
            right_attempt: cmp.right_run ? cmp.right_run.attempt : undefined,
            right_experiment:
              (cmp.right_run && cmp.right_run.experiment_id
               !== ctx.experimentId)
                ? cmp.right_run.experiment_id : undefined,
            left_pass: taskCompare.leftPass || undefined,
            right_pass: taskCompare.rightPass || undefined
          }
        ).then(function (result) {
          mark.disabled = false;
          if (result.ok) { showNotice("Review progress recorded"); ctx.reload(); }
          else { renderSelectionRefusal(card, result, ctx.reload); }
        });
      });
      state.appendChild(mark);
    }).catch(function (e) { state.textContent = e.message; });
  countPairComments(counted, cmp, ctx);
  renderPairComposer(card, cmp, ctx, null, "the whole pair");
}

/* How many recorded comments name THIS pair. Counted over the same
   consolidated task read the Feedback view uses, so the two cannot disagree,
   and reported as "at least" when the page is full rather than as a total the
   count does not support. */
function countPairComments(node, cmp, ctx) {
  var path = "/api/task-feedback?experiment="
    + encodeURIComponent(ctx.experimentId)
    + "&task=" + encodeURIComponent(ctx.taskId) + "&limit=200";
  api(path).then(function (data) {
    var mine = (data.feedback || []).filter(function (row) {
      return row.pair_key === cmp.review_pair_key;
    });
    node.textContent = (data.has_more ? "At least " : "")
      + mine.length + " recorded comment(s) name this exact pair. "
      + "A comment is not a review mark and a review mark is not a comment.";
  }).catch(function (e) { node.textContent = e.message; });
}

