/* -- training history [fix-9eg.2] --------------------------------------
   A read of the `train_runs` rows the trainer already wrote at publication
   time. Nothing here starts a training run, opens ___command_info or fetches
   anything from a network.

   Two things this section must not do, because both would be the page making
   a claim the evidence does not:

   * rename what was measured. `in_distribution_f1`, `routing` and
     `escalation` are the intent classifier's held-out numbers on the
     trainer's synthetic evaluation set. They keep those names here and are
     captioned with that dataset; a "score" column would be read as a
     task-success rate, which none of them is.
   * guess which recorded run used which trained model set. The only link
     drawn is an exact match between the training run's published version id
     and the version id a server STAMPED on an attempt. A workflow's source
     fingerprint does not move when only its trained artifacts do, and "the
     newest training run" is a guess, so neither is ever used: a run with no
     recorded version id reads as unavailable, not as unmatched. */
var trainingHistory = {runs: [], runId: null, seq: 0};

/* Every read this section makes is scoped to ONE source, and it can change
   while a read is in flight. Two independent guards, because they fail
   differently:

   * the SEQUENCE moves whenever a list or a detail starts, so a slower
     earlier read finds itself disowned. Without it, a list issued against the
     store the user just left could land last and overwrite `runs` -- after
     which a click resolved a run id belonging to the old store against the
     new one.
   * the SOURCE is captured at the moment the read is issued and re-checked
     when it lands. It covers what a counter cannot: a switch away and back
     while one read is outstanding leaves the sequence looking current again.

   This is the same rule `expNavToken` / `turnFind.seq`
   already state for the panes they own; `resetSourceScopedState` reaches this
   one through `onSourceSwitch`. */
function trainingSource() {
  return sourceIdentity(session);
}

function trainingToken() { return ++trainingHistory.seq; }

function trainingStale(token, source) {
  return token !== trainingHistory.seq || source !== trainingSource();
}

function trainingHistoryReset() {
  /* Called from the page's source boundary. In-flight reads are invalidated
     rather than awaited, and the section is closed: what it is showing belongs to evidence that is
     no longer selected. */
  trainingHistory.seq++;
  trainingHistory.runs = [];
  trainingHistory.runId = null;
  var dialog = document.getElementById("trainingDialog");
  if (!dialog) { return; }
  if (dialog.open) {
    if (dialog.close) { dialog.close(); } else { dialog.removeAttribute("open"); }
  }
  clear(document.getElementById("trainingRunList"));
  clear(document.getElementById("trainingRunDetail"));
}

var TRAINING_METRICS_NOTE = {
  unreadable: "This run's recorded metrics could not be read back.",
  absent: "This run recorded no metrics."
};

var TRAINING_HELDOUT_CAPTION =
  "Held-out intent-classification metrics, under the names the trainer recorded "
  + "them with, measured on its own generated evaluation set. They are not a "
  + "task-success rate and say nothing about whether a benchmark task passed.";

function trainingRunTitle(run) {
  /* The published version id IS the trained set's identity, so it is the
     name. A run that recorded none is named by its run id and says so in the
     sub-line rather than borrowing a version it never published. */
  return run.version_id || ("Run " + String(run.run_id || ""));
}

function trainingField(parent, label, value) {
  var row = el("div", "sub");
  row.appendChild(el("strong", null, label + ": "));
  row.appendChild(document.createTextNode(
    value === null || value === undefined || value === "" ? "not recorded" : String(value)));
  parent.appendChild(row);
  return row;
}

function trainingRuntimeLink(parent, link) {
  var card = el("div", "card");
  card.appendChild(el("h3", null, "Runs on this trained model set"));
  if (!link || link.status === "unavailable") {
    card.appendChild(el("p", "trainingUnavailable",
      (link && link.reason) || "No recorded model identity, so no association is available."));
    parent.appendChild(card);
    return card;
  }
  if (link.status === "no_match") {
    card.appendChild(el("p", "sub",
      "No recorded attempt stamped version " + link.version_id + "."
      + (link.scan_bounded
        ? " Only the " + link.experiments_scanned + " most recent experiments were searched."
        : "")));
    parent.appendChild(card);
    return card;
  }
  card.appendChild(el("p", "sub",
    link.attempts.length + (link.attempts.length === 1 ? " attempt" : " attempts")
    + " recorded the published version " + link.version_id
    + " as the model set they ran on."));
  var list = el("div", "convTurns");
  link.attempts.forEach(function (row) {
    var item = el("div", "listItem");
    item.appendChild(el("div", "title", row.task_id + " · attempt " + row.attempt));
    item.appendChild(el("div", "sub",
      "experiment " + row.experiment_id
      + (row.outcome ? " · " + row.outcome : " · no outcome recorded")
      + " · " + fmtTs(row.started_at)));
    list.appendChild(item);
  });
  card.appendChild(list);
  parent.appendChild(card);
  return card;
}

function trainingContextCard(parent, entry) {
  var card = el("details", "card");
  card.appendChild(el("summary", null, entry.context_folder));
  var thresholds = entry.thresholds || {};
  var names = Object.keys(thresholds);
  if (names.length) {
    names.sort().forEach(function (name) {
      trainingField(card, name, thresholds[name]);
    });
  } else {
    card.appendChild(el("p", "trainingUnavailable", "No thresholds recorded for this context."));
  }
  if (entry.heldout) {
    var heldout = entry.heldout;
    if (heldout.context !== undefined) {
      trainingField(card, "evaluated as context", heldout.context);
    }
    if (heldout.in_distribution_f1 !== undefined) {
      trainingField(card, "in_distribution_f1", heldout.in_distribution_f1);
    }
    ["routing", "escalation"].forEach(function (key) {
      if (heldout[key] === undefined) { return; }
      var box = el("div", "sub");
      box.appendChild(el("strong", null, key + ": "));
      box.appendChild(el("pre", null, pretty(heldout[key])));
      card.appendChild(box);
    });
  } else {
    /* A carried-forward context of a selective run publishes thresholds and
       no held-out report. Printing zeros here would invent a measurement. */
    card.appendChild(el("p", "trainingUnavailable",
      "No held-out evaluation recorded for this context in this run."));
  }
  parent.appendChild(card);
  return card;
}

function renderTrainingRun(run) {
  var pane = document.getElementById("trainingRunDetail");
  clear(pane);
  if (!run) {
    emptyState(pane, "Select a training run",
      "Each row is one published training run, newest first.");
    return;
  }
  pane.appendChild(el("h3", null, trainingRunTitle(run)));
  if (!run.model_identity_recorded) {
    pane.appendChild(el("p", "trainingUnavailable",
      "This run published no version id. A workflow trained on the "
      + "pre-versioning artifact layout records none while being fully "
      + "trained, so its model identity is unavailable rather than absent."));
  }
  var note = TRAINING_METRICS_NOTE[run.metrics_status];
  if (note) { pane.appendChild(el("p", "trainingUnavailable", note)); }

  var identity = el("div", "card");
  identity.appendChild(el("h3", null, "What was recorded"));
  trainingField(identity, "run id", run.run_id);
  trainingField(identity, "version id", run.version_id);
  trainingField(identity, "previous version", run.previous_version);
  trainingField(identity, "workflow fingerprint", run.workflow_fingerprint);
  trainingField(identity, "started", fmtTs(run.started_at));
  trainingField(identity, "completed", fmtTs(run.completed_at));
  trainingField(identity, "seed", run.seed);
  trainingField(identity, "training duration (s)", run.train_duration_seconds);
  trainingField(identity, "contexts retrained",
    (run.contexts_retrained || []).join(", "));
  trainingField(identity, "contexts carried forward",
    (run.contexts_carried_forward || []).join(", "));
  Object.keys(run.base_models || {}).sort().forEach(function (role) {
    /* The base checkpoint that was fine-tuned, NOT the trained set's
       identity: two different trained sets share these strings. */
    trainingField(identity, "base model (" + role + ")", run.base_models[role]);
  });
  pane.appendChild(identity);

  trainingRuntimeLink(pane, run.runtime_link);

  var metrics = el("div", "card");
  metrics.appendChild(el("h3", null, "Thresholds and held-out metrics"));
  metrics.appendChild(el("p", "trainingCaption", TRAINING_HELDOUT_CAPTION));
  if ((run.contexts || []).length) {
    run.contexts.forEach(function (entry) { trainingContextCard(metrics, entry); });
  } else {
    metrics.appendChild(el("p", "trainingUnavailable",
      "No per-context metrics are available for this run."));
  }
  if (Object.keys(run.totals || {}).length) {
    var totals = el("details", "card");
    totals.appendChild(el("summary", null, "totals (as recorded)"));
    totals.appendChild(el("pre", null, pretty(run.totals)));
    metrics.appendChild(totals);
  }
  if (Object.keys(run.commands || {}).length) {
    var commands = el("details", "card");
    commands.appendChild(el("summary", null,
      "utterance counts per command (" + Object.keys(run.commands).length + ")"));
    commands.appendChild(el("pre", null, pretty(run.commands)));
    metrics.appendChild(commands);
  }
  pane.appendChild(metrics);
}

function selectTrainingRun(runId) {
  trainingHistory.runId = runId;
  renderTrainingRunList();
  var token = trainingToken();
  var source = trainingSource();
  /* The source is captured with the request, so a response that lands after
     a switch is recognised by `trainingStale` and dropped. */
  var path = "/api/training-run/" + encodeURIComponent(runId);
  var pane = document.getElementById("trainingRunDetail");
  clear(pane);
  pane.appendChild(el("div", "empty", "Loading training run…"));
  return api(path).then(function (data) {
    if (trainingStale(token, source)) { return; }
    renderTrainingRun(data.training_run);
  }).catch(function (error) {
    if (trainingStale(token, source)) { return; }
    clear(pane);
    pane.appendChild(el("div", "empty", "Could not load this training run: " + error.message));
  });
}

function renderTrainingRunList() {
  var list = document.getElementById("trainingRunList");
  clear(list);
  if (!trainingHistory.runs.length) {
    list.appendChild(el("div", "empty", "No training runs recorded in this store."));
    return;
  }
  trainingHistory.runs.forEach(function (run) {
    var item = el("div", "listItem");
    if (run.run_id === trainingHistory.runId) { item.classList.add("selected"); }
    item.appendChild(el("div", "title", trainingRunTitle(run)));
    item.appendChild(el("div", "sub",
      fmtTs(run.completed_at || run.started_at)
      + " · " + run.context_count
      + (run.context_count === 1 ? " context" : " contexts")
      + (run.metrics_status === "recorded" ? "" : " · " + run.metrics_status)));
    makeRowActivatable(item, function () { selectTrainingRun(run.run_id); });
    list.appendChild(item);
  });
}

function loadTrainingRuns() {
  var list = document.getElementById("trainingRunList");
  clear(list);
  list.appendChild(el("div", "empty", "Loading training runs…"));
  /* The token moves when the LIST starts, not only when a run is selected, so
     a detail still in flight from the previous store is disowned here rather
     than surviving until the next list happens to finish. */
  var token = trainingToken();
  var source = trainingSource();
  var path = "/api/training-runs";
  trainingHistory.runs = [];
  trainingHistory.runId = null;
  renderTrainingRun(null);
  return api(path).then(function (data) {
    if (trainingStale(token, source)) { return; }
    trainingHistory.runs = data.training_runs || [];
    renderTrainingRunList();
    if (trainingHistory.runs.length) {
      return selectTrainingRun(trainingHistory.runs[0].run_id);
    }
  }).catch(function (error) {
    if (trainingStale(token, source)) { return; }
    clear(list);
    list.appendChild(el("div", "empty", "Could not load training runs: " + error.message));
  });
}

function openTrainingHistory() {
  var dialog = document.getElementById("trainingDialog");
  loadTrainingRuns();
  if (dialog.showModal) { dialog.showModal(); } else { dialog.setAttribute("open", ""); }
}

document.getElementById("trainingHistoryBtn").addEventListener("click", openTrainingHistory);
document.getElementById("trainingClose").addEventListener("click", function () {
  var dialog = document.getElementById("trainingDialog");
  if (dialog.close) { dialog.close(); } else { dialog.removeAttribute("open"); }
});

