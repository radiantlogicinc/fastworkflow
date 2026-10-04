/* -- the experiment browser (fix-bn1.6) --------------------------------
   experiment -> task -> attempt is the same three-level grouping the rail
   already builds over channel -> conversation -> turn, so this drills down
   inside #detail (the showCorpus precedent) rather than adding a second tree
   or a third top-level panel. Every attempt row links into the existing trace
   views through openTurnInDebug, so no trace rendering is rewritten. */
/* A monotonic navigation token. The three experiment views and selectTurn all
   own #detail and all render from a promise, so a slow response would otherwise
   repaint a view the user has already navigated away from. Every async handler
   below bails when its captured token is stale. */
var expNav = 0;
function expNavToken() { return ++expNav; }
function expNavStale(token) { return token !== expNav; }

function expCrumbs(container, trail) {
  var bar = el("div", "crumbs");
  trail.forEach(function (step, i) {
    if (i) { bar.appendChild(el("span", "sep", "/")); }
    if (step.onClick) {
      var b = el("button", null, step.label);
      b.addEventListener("click", step.onClick);
      bar.appendChild(b);
    } else {
      bar.appendChild(el("span", "here", step.label));
    }
  });
  container.appendChild(bar);
}

/* The same experiment id can be recorded in more than one store, so a node is
   this page's only when its source is the one the page reads from. */
function experimentNodeMatches(experimentId) {
  return function (n) {
    return n.kind === "experiment" && n.experiment_id === experimentId &&
      (n.source && n.source.benchmark_experiment || null) === benchmarkExperimentSource;
  };
}

function hierarchyCrumb(path) {
  var node = path[path.length - 1];
  return { label: hierarchyLabel(node), onClick: function () {
    activateHierarchy(findHierarchyByKey(node.key) || path, true);
  } };
}

/* An experiment's crumbs start where the rail does: Benchmarks, then the
   branch the experiment sits under, each landing where its rail row would.
   The rail can hold no node for it (a workspace, an evidence store it could
   not open, a run recorded since the last refresh), so the benchmark the
   experiment record names stands in, remembered per source for the task page,
   which renders before any experiment record is read. */
var experimentBenchmarks = {};
function experimentAncestorCrumbs(experimentId, exp) {
  var scoped = String(benchmarkExperimentSource) + "\u001f" + experimentId;
  if (exp) { experimentBenchmarks[scoped] = exp.benchmark_id || null; }
  var path = findHierarchy(experimentNodeMatches(experimentId));
  if (path) { return path.slice(0, -1).map(function (_, i) { return hierarchyCrumb(path.slice(0, i + 1)); }); }
  var crumbs = [hierarchyRoot ? hierarchyCrumb([hierarchyRoot]) : { label: "Benchmarks", onClick: showBenchmarks }];
  if (!(scoped in experimentBenchmarks)) { return crumbs; }
  var benchmarkId = experimentBenchmarks[scoped];
  var branch = findHierarchy(function (n) { return n.kind === "benchmark" && (n.benchmark_id || null) === benchmarkId; });
  if (branch) { crumbs.push(hierarchyCrumb(branch)); }
  else if (benchmarkId) { crumbs.push({ label: benchmarkId, onClick: function () { showBenchmark(benchmarkId); } }); }
  else { crumbs.push({ label: "Experiments without a benchmark" }); }
  return crumbs;
}

function expStatusPill(status) {
  var cls = status === "complete" ? "pill ok"
    : (status === "invalid" ? "pill err" : "pill wait");
  var pill = el("span", cls);
  pill.appendChild(el("span", "dot"));
  pill.appendChild(el("span", null, status));
  return pill;
}

function showExperiments() {
  benchmarkExperimentSource = null;
  var nav = expNavToken();
  state.experimentId = null;
  state.experimentTask = null;
  var d = document.getElementById("detail");
  clear(d);
  d.appendChild(el("div", "empty", "loading experiments…"));
  /* limit=201 so 201 rows means "there are more than 200", which is said out
     loud rather than silently truncating the list. */
  api("/api/experiments?limit=201").then(function (data) {
    if (expNavStale(nav)) { return; }
    clear(d);
    var card = el("div", "card");
    expCrumbs(card, [{ label: "Experiments" }]);
    card.appendChild(el("h2", null, "Experiments"));
    card.appendChild(el("div", "sub",
      "A labelled set of tasks, each run one or more times, scored as one " +
      "object. An experiment is not a channel: attempts run on their own " +
      "channels so they stay independent."));
    var allRows = data.experiments || [];
    var hiddenArchived = allRows.filter(function (row) {
      return row.archived && !archivedExperimentsShown[row.benchmark_id];
    }).length;
    var rows = allRows.filter(function (row) {
      return !row.archived || archivedExperimentsShown[row.benchmark_id];
    });
    var truncated = rows.length > 200;
    if (truncated) { rows = rows.slice(0, 200); }
    if (!rows.length) {
      card.appendChild(el("div", "empty", hiddenArchived
        ? "archived experiments are hidden; show them from their benchmark in the left rail"
        : "no experiments recorded"));
      d.appendChild(card);
      return;
    }
    if (truncated) {
      card.appendChild(el("div", "sub",
        "Showing the 200 most recent experiments; there are more. "
        + "Add ?offset=200 to /api/experiments to see the next page."));
    }
    rows.forEach(function (row) {
      var item = el("div", "listItem" + (row.archived ? " archived" : ""));
      var title = el("div", "title");
      title.appendChild(el("span", null, "Experiment · " + row.experiment_id.slice(-8) + "  "));
      title.appendChild(expStatusPill(row.status));
      if (row.archived) { title.appendChild(el("span", "pill", "Archived")); }
      if (row.arm) { title.appendChild(el("span", null, "  [" + row.arm + "]")); }
      item.appendChild(title);
      if (row.description) { item.appendChild(el("div", "sub", row.description)); }
      var declared = row.declared_tasks + "×" + row.declared_attempts;
      var sub = declared + " declared, " + row.attempts_finished + " finished"
        + (row.invalid_reason ? "  — " + row.invalid_reason : "")
        + (row.benchmark_id
            ? "  · " + row.benchmark_id + "@" + (row.benchmark_version || "?")
            : "")
        + "  · " + (row.created_at || "");
      item.appendChild(el("div", "sub", sub));
      makeRowActivatable(item, function () { showExperiment(row.experiment_id); });
      card.appendChild(item);
    });
    d.appendChild(card);
  }).catch(function (e) {
    if (expNavStale(nav)) { return; }
    clear(d);
    d.appendChild(el("div", "empty", "No experiments available: " + e.message));
  });
}

function showExperiment(experimentId) {
  focusHierarchy(experimentNodeMatches(experimentId));
  var nav = expNavToken();
  state.experimentId = experimentId;
  state.experimentTask = null;
  var d = document.getElementById("detail");
  clear(d);
  d.appendChild(el("div", "empty", "loading experiment…"));
  Promise.all([
    api("/api/experiment/" + encodeURIComponent(experimentId)),
    api("/api/experiment/" + encodeURIComponent(experimentId) + "/score"),
    api("/api/experiment/" + encodeURIComponent(experimentId) + "/tasks")
  ]).then(function (results) {
    if (expNavStale(nav)) { return; }
    var exp = results[0].experiment;
    var score = results[1].score;
    var tasks = results[2].tasks || [];
    clear(d);
    var card = el("div", "card");
    /* Named by its id, described by its label: the label is free text the
       author may or may not have written, and a page whose heading is a
       paragraph (or blank) is no longer a heading. */
    var expName = "Experiment · " + experimentId.slice(-8);
    expCrumbs(d, experimentAncestorCrumbs(experimentId, exp).concat([{ label: expName }]));
    var actions = pageHeader(d, exp.archived ? "ARCHIVED EXPERIMENT" : "RECORDED EXPERIMENT", expName,
      exp.description
        ? ("Description: " + exp.description)
        : "Review outcomes, follow the evidence, and capture what you learn.");
    var intro = actions.parentNode.querySelector(".intro");
    intro.insertBefore(el("div", "recordId", "ID: " + experimentId), intro.querySelector("p"));
    if (exp.benchmark_id && !(session && session.workspace_mode)) {
      var archive = el("button", "ghost", exp.archived ? "Unarchive experiment" : "Archive experiment");
      archive.title = exp.archived
        ? "Return this experiment to benchmark lists"
        : "Hide this experiment from benchmark lists";
      archive.addEventListener("click", function () {
        apiPatch(
          "/api/experiment/" + encodeURIComponent(experimentId),
          {archived: !exp.archived},
          exp.archived ? "Experiment unarchived" : "Experiment archived"
        ).then(function () {
          refreshConvs().then(function () {
            showExperiment(experimentId);
          });
        });
      });
      actions.appendChild(archive);
    }
    /* "Where did this run go wrong" is a question about recorded turns, so
       it opens the finder already scoped rather than a second search UI. */
    turnFindEntryButton(actions, { experiment: experimentId },
      "Find problems in this experiment",
      "Search the turns this source recorded for this experiment.");
    if (!turnFindEntrySupported()) {
      card.appendChild(el("div", "sub",
        "The turn finder searches one store at a time. This experiment is "
        + "read through a workspace whose evidence spans several stores, so "
        + "no scoped search is offered here."));
    }

    /* An invalid experiment first, and loudly. A run whose turns were dropped
       or erased has an unreconstructable denominator; showing its label beside
       a valid one with only a small badge to tell them apart is how a dead run
       gets quoted. */
    if (exp.status === "invalid") {
      var bad = el("div", "expInvalid");
      bad.appendChild(el("strong", null, "This experiment is INVALID and carries no score."));
      bad.appendChild(el("div", null,
        "Reason: " + (exp.invalid_reason || "unknown") + ". "
        + "Its attempts can no longer be counted against the denominator it "
        + "declared, so no pass@1 or pass^k is computed for it."));
      if (exp.invalid_detail) { bad.appendChild(el("div", null, exp.invalid_detail)); }
      card.appendChild(bad);
    }
    /* The evidence-run verdict beside the status, first: whether the turns
       this experiment rests on were all written is prior to any score. */
    var verdictLine = el("div", "diffToolbar");
    verdictLine.appendChild(expStatusPill(exp.status));
    verdictLine.appendChild(evidenceBadge(exp.evidence));
    if (exp.archived) { verdictLine.appendChild(el("span", "pill", "Archived")); }
    card.appendChild(verdictLine);

    if (exp.status !== "running") {
      var annotations = el("details", "card");
      annotations.appendChild(el("summary", null, "Postmortem"));
      var notes = el("textarea", "expNotes");
      notes.value = exp.notes || "";
      annotations.appendChild(notes);
      var saveRow = el("div", "diffToolbar");
      var save = el("button", null, "Save postmortem");
      var saveMsg = el("span", "sub", "");
      save.addEventListener("click", function () {
        saveMsg.textContent = "saving…";
        apiPatch("/api/experiment/" + encodeURIComponent(experimentId),
                 { notes: notes.value },
                 "Postmortem saved").then(function () {
          saveMsg.textContent = "saved";
        }).catch(function (e) { saveMsg.textContent = e.message; });
      });
      saveRow.appendChild(save);
      saveRow.appendChild(saveMsg);
      annotations.appendChild(saveRow);
      card.appendChild(annotations);
    }

    card.appendChild(el("h2", null, "Result"));
    var kv = el("dl", "kv");
    function pair(k, v) {
      kv.appendChild(el("dt", null, k));
      kv.appendChild(el("dd", null, v === null || v === undefined ? "—" : String(v)));
    }
    pair("status", exp.status);
    pair("declared", exp.declared_tasks + " tasks × " + exp.declared_attempts + " attempts");
    pair("scored attempts", score.scored_attempts + " of " + score.expected_attempts);
    pair("pass@1", score.reportable ? score.pass_at_1.toFixed(3) : "not reportable");
    pair("pass^" + exp.declared_attempts,
         score.reportable ? score.pass_at_k.toFixed(3) : "not reportable");
    pair("verdict sources", (score.outcome_sources || []).join(", "));
    card.appendChild(kv);
    /* Collapsed: what this experiment ran under, keys verbatim, with
       "not recorded" where the store holds nothing (fix-aou (c)). */
    /* The API retains provenance for agents and diagnostics; the ordinary
       experiment page omits the specialist field dump. */
    if (!score.reportable && score.reason_not_reportable) {
      card.appendChild(el("div", "sub", score.reason_not_reportable));
    }
    /* Named for what it is. A verdict source of "derived" is the turn-status
       fallback, which measures "no command reported a failure code" and not
       whether the task was accomplished. */
    if ((score.outcome_sources || []).indexOf("derived") >= 0) {
      var derived = el("div", "nonComparable");
      derived.appendChild(el("strong", null, "Some verdicts are derived, not graded."));
      derived.appendChild(el("div", null,
        "A 'derived' outcome reports only that no command returned a failure " +
        "code. It is not a judgement that the task was accomplished."));
      card.appendChild(derived);
    }

    /* The baseline comparison, when this experiment declares one. Rendered
       here rather than in a separate view because "did the change help" is a
       question about THIS experiment, and a separate route is one the UI would
       never call. */
    if (exp.baseline_experiment_id) {
      card.appendChild(el("h2", null, "Versus baseline"));
      var cmpBox = el("div");
      card.appendChild(cmpBox);
      cmpBox.appendChild(el("div", "sub", "comparing…"));
      apiAllowing409("/api/experiment/" + encodeURIComponent(experimentId) + "/compare")
        .then(function (cmp) { renderComparison(cmpBox, cmp); })
        .catch(function (e) {
          clear(cmpBox);
          var box = el("div", "nonComparable");
          box.appendChild(el("strong", null, "Not comparable."));
          box.appendChild(el("div", null, e.message));
          cmpBox.appendChild(box);
        });
    }

    /* The verdict, its stored reasons verbatim, and each segment's
       writer-health delta folded under it. */
    /* A valid verdict is already visible beside the experiment status. Keep
       the detailed section for states that need investigation. */
    if (!exp.evidence || exp.evidence.state !== "valid") {
      card.appendChild(el("h2", null, "Evidence runs"));
      renderEvidenceVerdict(card, exp.evidence);
    }

    /* The contest, before the tasks: "is this the workflow's current answer"
       is a fact about the whole experiment, and each task's best run below is
       a different decision that never moves it. */
    renderWinnerPanel(card, experimentId, function () {
      showExperiment(experimentId);
    });

    card.appendChild(el("h2", null, "Tasks"));
    if (!tasks.length) {
      card.appendChild(el("div", "empty", "no attempts recorded yet"));
    }
    tasks.forEach(function (task) {
      var item = el("div", "listItem");
      var title = el("div", "title", task.task_id);
      item.appendChild(title);
      var outcomes = (task.outcomes || []).map(function (o) { return o || "unfinished"; });
      item.appendChild(el("div", "sub",
        outcomes.join(", ") + (task.passed_all ? "  — all passed" : "")));
      makeRowActivatable(item, function () {
        showExperimentTask(experimentId, task.task_id, expName);
      });
      card.appendChild(item);
    });
    d.appendChild(card);
  }).catch(function (e) {
    if (expNavStale(nav)) { return; }
    clear(d);
    d.appendChild(el("div", "empty", "Could not load experiment: " + e.message));
  });
}

function renderComparison(container, cmp) {
  clear(container);
  if (!cmp.comparable) {
    var box = el("div", "nonComparable");
    box.appendChild(el("strong", null, "Not comparable."));
    (cmp.problems || []).forEach(function (p) { box.appendChild(el("div", null, p)); });
    container.appendChild(box);
    renderProvenanceDifferences(container, cmp.provenance_differences);
    return;
  }
  var kv = el("dl", "kv");
  function pair(k, v) {
    kv.appendChild(el("dt", null, k));
    kv.appendChild(el("dd", null, v === null || v === undefined ? "—" : String(v)));
  }
  pair("tasks compared", cmp.tasks_compared);
  pair("improved", (cmp.improved || []).join(", ") || "none");
  pair("regressed", (cmp.regressed || []).join(", ") || "none");
  pair("observed flips", cmp.observed_flips);
  pair("expected flips if nothing changed", cmp.expected_flips_if_nothing_changed);
  container.appendChild(kv);
  /* The caveat has to sit next to the number, not in a doc. "3 tasks flipped"
     reads as "3 tasks improved" unless the expectation under no change is
     right there beside it. */
  var note = el("div", "sub");
  note.textContent =
    "Two arms that differ in nothing still flip tasks, because pass^k is a "
    + "threshold on a noisy quantity. The expectation above is what chance "
    + "alone would produce at these pass rates. It is not a significance test.";
  container.appendChild(note);
  /* The comparability check: every provenance field the two arms differ on,
     quoted. It qualifies the comparison above; it does not withhold it. */
  renderProvenanceDifferences(container, cmp.provenance_differences);
}

/* The task page's views, in the order the owner named them.
   showExperimentTask renders the strip from this list and calls the view's
   render(container, experimentId, taskId, nav). The three share one task, one
   crumb trail and one feedback vocabulary: a comment written from a Compare
   row is an ordinary comment and shows up under Feedback. */
var TASK_VIEWS = [
  { key: "runs", label: "Runs" },
  { key: "compare", label: "Compare", render: renderTaskCompare },
  { key: "feedback", label: "Feedback", render: renderTaskFeedback }
];
var taskView = "runs";

function taskViewStrip(container, experimentId, taskId, label) {
  var strip = el("div", "feedbackTabs");
  strip.setAttribute("role", "tablist");
  strip.setAttribute("aria-label", "Task views");
  TASK_VIEWS.forEach(function (view) {
    var button = el("button", view.key === taskView ? "active" : null, view.label);
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("data-task-view", view.key);
    button.setAttribute("aria-selected", view.key === taskView ? "true" : "false");
    button.tabIndex = view.key === taskView ? 0 : -1;
    button.addEventListener("click", function () {
      taskView = view.key;
      showExperimentTask(experimentId, taskId, label);
    });
    strip.appendChild(button);
  });
  container.appendChild(strip);
}

function showExperimentTask(experimentId, taskId, label) {
  var conversationPath = findHierarchy(function (n) { return n.kind === "conversation" &&
    n.info.task_id === taskId && n.source && n.source.benchmark_experiment === experimentId; });
  if (conversationPath) {
    hierarchyPath = conversationPath;
    conversationPath.forEach(function (n) { hierarchyExpanded[n.key] = true; });
    renderHierarchy();
  }
  var nav = expNavToken();
  state.experimentId = experimentId;
  state.experimentTask = taskId;
  var d = document.getElementById("detail");
  clear(d);
  var crumbs = experimentAncestorCrumbs(experimentId).concat([
    /* the experiment's NAME, matching the parent view; a deep link that has
       no name in hand builds the same one out of the id */
    { label: label || ("Experiment · " + experimentId.slice(-8)),
      onClick: function () { showExperiment(experimentId); } },
    { label: taskId }
  ]);
  /* Compare state belongs to one task. Arriving at a different task resets it
     rather than carrying "attempt 3" into a task that has two. */
  resetTaskCompare(experimentId, taskId);
  resetSelectedRuns(experimentId, taskId);
  if (taskView !== "runs") {
    var view = TASK_VIEWS.filter(function (v) { return v.key === taskView; })[0];
    var viewCard = el("div", "card");
    expCrumbs(viewCard, crumbs);
    taskViewStrip(viewCard, experimentId, taskId, label);
    d.appendChild(viewCard);
    view.render(d, experimentId, taskId, nav);
    return;
  }
  renderTaskRuns(d, experimentId, taskId, label, crumbs, nav);
}

/* -- the task Feedback view (fix-9eg.19.1) ----------------------------
   Every authorized comment on one task, across attempts, turns and
   components, comparison comments included. The default has NO filter: a
   hidden one would quietly answer a narrower question than the reader asked,
   and "everything recorded about this task" is the whole point of the view.
   Paging is server-side and bounded, over a deterministic order. */
var taskFeedbackFilters = {};
var TASK_FEEDBACK_PAGE = 50;

function taskFeedbackFilterSelect(row, name, labelText, options) {
  var wrap = el("label", null);
  wrap.appendChild(el("div", null, labelText));
  var select = el("select");
  select.setAttribute("aria-label", labelText);
  var all = el("option", null, "All");
  all.value = "";
  select.appendChild(all);
  options.forEach(function (option) {
    var node = el("option", null, option.label);
    node.value = option.value;
    select.appendChild(node);
  });
  select.value = taskFeedbackFilters[name] || "";
  select.addEventListener("change", function () {
    taskFeedbackFilters[name] = select.value;
    /* A narrowed view starts at its own first page: keeping the old offset
       would land the reader past the end of a shorter result. */
    taskFeedbackFilters.offset = 0;
    if (taskFeedbackFilters.reload) { taskFeedbackFilters.reload(); }
  });
  wrap.appendChild(select);
  row.appendChild(wrap);
  return select;
}

function taskFeedbackSubcategoryOptions() {
  var options = [];
  FEEDBACK_TAXONOMY.forEach(function (category) {
    category.subcategories.forEach(function (sub) {
      options.push({ value: sub.value, label: category.label + " · " + sub.label });
    });
  });
  return options;
}

function renderTaskFeedbackRow(row) {
  var item = el("div", "listItem");
  var title = el("div", "title");
  var classification = feedbackClassification(row);
  title.appendChild(el("span",
    row.classified ? null : "feedbackUnclassified", classification));
  item.appendChild(title);
  var where = "attempt " + (row.attempt === null || row.attempt === undefined ? "—" : row.attempt)
    + " · " + row.target_kind + " · " + row.target_label;
  item.appendChild(el("div", "sub", where));
  item.appendChild(el("div", "sub",
    row.created_at + " · " + feedbackProvenanceLabel(row.provenance)
    + " · store " + (row.store_id || "—")));
  item.appendChild(el("pre", "json", row.comment || ""));
  var links = el("div", "sub");
  var open = el("button", "evidenceLink", "Open the trace");
  open.type = "button";
  open.addEventListener("click", function () { selectTurn(row.turn_key); });
  links.appendChild(open);
  item.appendChild(links);
  renderFeedbackPair(item, row);
  return item;
}

function renderTaskFeedback(container, experimentId, taskId) {
  var card = el("div", "card");
  card.appendChild(el("h2", null, "Feedback"));
  card.appendChild(el("div", "sub",
    "Every recorded comment on this task, across attempts, turns and "
    + "components, including comparison comments anchored on the other side "
    + "of a pair and task-level summaries. Filters are optional."));
  var filters = el("div", "feedbackFilters");
  card.appendChild(filters);
  var list = el("div");
  var note = el("p", "sub", "Loading feedback…");
  var pager = el("div", "sub");
  card.appendChild(list); card.appendChild(pager); card.appendChild(note);
  container.appendChild(card);

  function load() {
    /* A workspace reads through its manifest: the store-aware route scopes
       the task to the segments the manifest declares for the experiment, so
       a comparison comment recorded beside one archive shows in the other
       archive's task view too. The live route takes no store_id at all. */
    var base = (session && session.workspace_mode)
      ? "/api/workspace/task-feedback?experiment="
      : "/api/task-feedback?experiment=";
    var path = base + encodeURIComponent(experimentId)
      + "&task=" + encodeURIComponent(taskId)
      + "&limit=" + TASK_FEEDBACK_PAGE
      + "&offset=" + (taskFeedbackFilters.offset || 0);
    ["category", "subcategory", "provenance", "target_kind", "component"].forEach(function (name) {
      if (taskFeedbackFilters[name]) {
        path += "&" + name + "=" + encodeURIComponent(taskFeedbackFilters[name]);
      }
    });
    if (benchmarkExperimentSource) {
      path += "&benchmark_experiment=" + encodeURIComponent(benchmarkExperimentSource);
    }
    api(path).then(function (data) {
      clear(list); clear(pager);
      var rows = data.feedback || [];
      if (!rows.length) {
        list.appendChild(el("div", "empty",
          data.total ? "No comment on this page." : "No feedback recorded for this task yet."));
      }
      rows.forEach(function (row) { list.appendChild(renderTaskFeedbackRow(row)); });
      note.textContent = "Showing " + rows.length + " of " + data.total
        + " comment(s) across " + (data.stores || []).length + " store(s).";
      if ((data.offset || 0) > 0) {
        var back = el("button", null, "Previous");
        back.type = "button";
        back.addEventListener("click", function () {
          taskFeedbackFilters.offset = Math.max(0, (data.offset || 0) - TASK_FEEDBACK_PAGE);
          load();
        });
        pager.appendChild(back);
      }
      if (data.has_more) {
        var next = el("button", null, "Next");
        next.type = "button";
        next.addEventListener("click", function () {
          taskFeedbackFilters.offset = (data.offset || 0) + TASK_FEEDBACK_PAGE;
          load();
        });
        pager.appendChild(next);
      }
    }).catch(function (e) {
      clear(list); note.textContent = e.message;
    });
  }
  taskFeedbackFilters.reload = load;
  taskFeedbackFilterSelect(filters, "category", "Category",
    FEEDBACK_TAXONOMY.map(function (c) { return { value: c.value, label: c.label }; }));
  taskFeedbackFilterSelect(filters, "subcategory", "Subcategory",
    taskFeedbackSubcategoryOptions());
  taskFeedbackFilterSelect(filters, "provenance", "Author", [
    { value: "human", label: "Human" },
    { value: "coding_agent", label: "Coding Agent" },
    { value: "distillation_agent", label: "Distillation Agent" }
  ]);
  taskFeedbackFilterSelect(filters, "target_kind", "Component", [
    { value: "task", label: "Task summary" },
    { value: "turn", label: "Turn" },
    { value: "phase", label: "Phase" },
    { value: "step", label: "Step" },
    { value: "span", label: "Span" }
  ]);
  load();
}

function openExperimentAttempt(experimentId, taskId, attempt) {
  var nav = expNavToken();
  /* Resolve the attempt to its turns through the SHIPPED /api/turns route,
     which now filters by experiment/task/attempt, then hand the first turn to
     the existing trace view. Nothing about trace rendering is duplicated. */
  api("/api/turns?experiment=" + encodeURIComponent(experimentId)
      + "&task=" + encodeURIComponent(taskId)
      + "&attempt=" + encodeURIComponent(attempt)
      + "&limit=500").then(function (data) {
    if (expNavStale(nav)) { return; }
    var turns = data.turns || [];
    if (!turns.length) {
      var d = document.getElementById("detail");
      clear(d);
      d.appendChild(el("div", "empty",
        "This attempt has no turn records. If it was restarted, its earlier "
        + "turns were deleted so the attempt could be re-run under the same "
        + "labels."));
      return;
    }
    /* /api/turns returns newest-first, so the last element is the attempt's
       FIRST turn — which is where a reader wants to start. The route caps at
       500, so past that the last element is merely the oldest turn ON THIS PAGE
       and opening it silently would be opening the wrong turn.
       The warning is a STOP, not a note rendered a moment before
       openTurnInDebug repaints #detail over it. */
    if (turns.length >= 500) {
      var d = document.getElementById("detail");
      clear(d);
      var card = el("div", "card");
      card.appendChild(el("h2", null, "Attempt too long to open at its start"));
      card.appendChild(el("div", null,
        "This attempt has 500 or more turns, which is the page limit, so its "
        + "first turn cannot be identified from one page. Opening the oldest "
        + "turn on this page would start you mid-trajectory."));
      var go = el("button", null, "Open the oldest turn on this page anyway");
      go.addEventListener("click", function () {
        selectTurn(turns[turns.length - 1].turn_key);
      });
      card.appendChild(go);
      d.appendChild(card);
      return;
    }
    selectTurn(turns[turns.length - 1].turn_key);
  }).catch(function (e) {
    if (expNavStale(nav)) { return; }
    var d = document.getElementById("detail");
    clear(d);
    d.appendChild(el("div", "empty", "Could not resolve the attempt: " + e.message));
  });
}


/* -- run selection, the winner, and one shared comparison ---------------
   (fix-9eg.17.2 / .17.3 / .17.4 and the browser half of fix-9eg.4)

   Every route here is run_chatbot/selection_api.py's, and nothing below
   derives a turn key, a store path or a step alignment of its own: a client
   names an experiment, a task and an attempt NUMBER, and the references, the
   evidence anchors and the pair identity come back from the recorded
   evidence. That is why a comment written from a Compare row and the pair's
   review progress agree about which two executions they are about.

   Three product rules this section exists to keep visible:

   - The WINNER of a contest and a task's BEST RUN are different decisions.
     Choosing a best run never moves a winner, and the winner badge carries the
     winning experiment's own recorded status instead of implying one.
   - "Reference" is a viewing default (the first completed attempt). Only a
     recorded decision is "Best run", and a failed attempt chosen as best stays
     visibly failed.
   - A recorded pair is frozen. Re-picking a best run produces a different pair
     and leaves every earlier comment naming the runs it was written about. */

/* Who the browser says it is. selection_api refuses actor_kind "system" --
   that kind marks the winner nobody chose -- so a person's decision is filed
   as a person's. */
var SELECTION_ACTOR = { actor: "observability UI", actor_kind: "human" };

/* One decision body: who is deciding, plus whatever this decision says. Keys
   whose value is undefined are dropped, so "no reason given" is an absent
   field rather than an empty string the store would have to interpret. */
function decisionBody(fields) {
  var body = {
    actor: SELECTION_ACTOR.actor, actor_kind: SELECTION_ACTOR.actor_kind
  };
  Object.keys(fields || {}).forEach(function (key) {
    if (fields[key] !== undefined) { body[key] = fields[key]; }
  });
  return body;
}

function selectionPath(experimentId, suffix) {
  return "/api/experiments/" + encodeURIComponent(experimentId) + (suffix || "");
}
function taskSelectionPath(experimentId, taskId, suffix) {
  return selectionPath(experimentId,
    "/tasks/" + encodeURIComponent(taskId) + (suffix || ""));
}

/* A GET whose refusal IS the answer.

   403 sealed, 404 not recorded and 409 (no control yet, this experiment joined
   no contest, the evidence is unreadable) are all states the page has to
   render as themselves. Throwing would replace an explanation with "Could not
   load", and a workflow whose experiments predate the selection control would
   lose its attempt list along with its absent decisions. */
function selectionRead(path) {
  return fetch(path, { headers: chatbotAuthHeaders() })
    .then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (r.ok) { return data; }
        if (r.status === 403 || r.status === 404 || r.status === 409) {
          return {
            unavailable: true, status: r.status, sealed: !!data.sealed,
            error: data.error || ("API " + path + " -> " + r.status)
          };
        }
        throw new Error(data.error || ("API " + path + " -> " + r.status));
      });
    });
}

/* A selection write. Its refusals carry the contract: a 409 answers with the
   pointer that replaced the one this page read, which is what makes "read it
   again and decide again" a mechanical retry. The shared mutation helper
   flattens a body into an Error message, so this one returns the body. */
function selectionWrite(path, method, body) {
  return fetch(path, {
    method: method,
    headers: chatbotAuthHeaders({"Content-Type": "application/json"}),
    body: JSON.stringify(body || {})
  }).then(function (r) {
    return r.json().catch(function () { return {}; }).then(function (data) {
      return { ok: r.ok, status: r.status, data: data };
    });
  });
}

/* A read scoped to ONE named source, never to the page's current global one.

   The two sides of a comparison routinely live in different evidence
   databases, so a link or a preview on the right-hand side must not be served
   by whatever store the page happens to be pointed at — that is how a turn key
   that exists in both databases opens the wrong side's trace. These two take
   the source explicitly and add nothing implicitly.

   `benchmark_experiment` is the only handle the server resolves: it refuses to
   find a store by searching the disk, so a database is addressed by the
   experiment registered against it. When that registration does not exist the
   refusal is the answer, and the caller says so rather than guessing. */
function scopedRead(path, experimentId) {
  var scoped = path;
  if (experimentId) {
    scoped += (path.indexOf("?") < 0 ? "?" : "&")
      + "benchmark_experiment=" + encodeURIComponent(experimentId);
  }
  return selectionRead(scoped);
}
function scopedRaw(path, experimentId) {
  var scoped = path;
  if (experimentId) {
    scoped += (path.indexOf("?") < 0 ? "?" : "&")
      + "benchmark_experiment=" + encodeURIComponent(experimentId);
  }
  return fetch(scoped, { headers: chatbotAuthHeaders() });
}

function selectionNote(container, title, detail) {
  var box = el("div", "selectionNote");
  box.appendChild(el("strong", null, title));
  if (detail) { box.appendChild(el("div", null, detail)); }
  container.appendChild(box);
  return box;
}

function labelledSelect(row, labelText, options, value, onChange) {
  var wrap = el("label", null);
  wrap.appendChild(el("div", null, labelText));
  var select = el("select");
  select.setAttribute("aria-label", labelText);
  options.forEach(function (option) {
    var node = el("option", null, option.label);
    node.value = String(option.value);
    if (option.disabled) { node.disabled = true; }
    select.appendChild(node);
  });
  select.value = value === null || value === undefined ? "" : String(value);
  select.addEventListener("change", function () { onChange(select.value); });
  wrap.appendChild(select);
  row.appendChild(wrap);
  return select;
}

/* An optional reason, offered for every decision and required by none. The
   owner's rule: no rubric, no target, no mandatory justification. */
function reasonField(container, labelText) {
  var wrap = el("label", "formField");
  wrap.appendChild(el("div", "sub", labelText));
  var input = el("input", "expNotes");
  input.type = "text";
  input.maxLength = 4000;
  input.setAttribute("aria-label", labelText);
  input.placeholder = "Optional";
  wrap.appendChild(input);
  container.appendChild(wrap);
  return input;
}

function executionPill(status) {
  return status === "completed" ? "pill ok"
    : (status === "failed" ? "pill err" : "pill wait");
}

/* An attempt's own recorded state, unlaundered. A failed attempt may be the
   best run of its task and must keep saying "failed". */
function attemptTitle(row) {
  var title = el("div", "title");
  title.appendChild(el("span", null, "attempt " + row.attempt + "  "));
  title.appendChild(el("span", null, row.outcome || "unfinished"));
  if (row.is_best) { title.appendChild(el("span", "pill ok", "Best run")); }
  if (row.is_reference) { title.appendChild(el("span", "pill", "Reference")); }
  if (row.execution_status) {
    title.appendChild(el("span", executionPill(row.execution_status),
                         row.execution_status));
  }
  return title;
}

function attemptOptionLabel(row) {
  var label = "attempt " + row.attempt
    + " · " + (row.outcome || row.execution_status || "unfinished");
  if (row.is_best) { label += " · Best run"; }
  else if (row.is_reference) { label += " · Reference"; }
  if (!row.comparable) {
    label += " · " + (row.evidence_label || "nothing recorded to compare");
  }
  return label;
}

/* -- the task's Runs view --------------------------------------------- */

function renderTaskRuns(container, experimentId, taskId, label, crumbs, nav) {
  container.appendChild(el("div", "empty", "loading attempts…"));
  Promise.all([
    api("/api/experiment/" + encodeURIComponent(experimentId)
        + "/attempts?task=" + encodeURIComponent(taskId)),
    selectionRead(taskSelectionPath(experimentId, taskId, "/runs"))
  ]).then(function (results) {
    if (expNavStale(nav)) { return; }
    clear(container);
    var evidence = results[0].attempts || [];
    var runs = results[1];
    var card = el("div", "card");
    expCrumbs(card, crumbs);
    taskViewStrip(card, experimentId, taskId, label);
    card.appendChild(el("h2", null, "Attempts"));
    card.appendChild(el("div", "sub",
      "Per-attempt outcomes are retained rather than collapsed to a per-task " +
      "rate: an agent that passes a task often can still fail pass^k, and a " +
      "rate stored at write time cannot say which."));
    var findRow = el("div", "runDecision");
    turnFindEntryButton(findRow, { experiment: experimentId, task: taskId },
      "Find problems in this task",
      "Search the turns this source recorded for this task, across every "
      + "attempt of it.");
    if (findRow.childNodes.length) { card.appendChild(findRow); }
    container.appendChild(card);
    renderTaskRunList(card, experimentId, taskId, label, evidence, runs, nav);
    /* Consistency reads the same recorded attempts this list just printed, so
       it belongs under them rather than on a page of its own. Its own
       container: it loads separately and must not blank the list if it fails. */
    var consistency = el("div");
    card.appendChild(consistency);
    renderConsistency(consistency, experimentId, taskId, label, nav, null);
  }).catch(function (e) {
    if (expNavStale(nav)) { return; }
    clear(container);
    container.appendChild(el("div", "empty", "Could not load attempts: " + e.message));
  });
}

/* Every attempt this task recorded, failed and unfinished included, with the
   selection state beside each one.

   The two reads are kept apart on purpose. `/api/experiment/<id>/attempts` is
   the EVIDENCE view and carries the writer-health verdict, the server
   configuration and the token/cost chips; `.../tasks/<id>/runs` is the
   DECISION view and carries Best run, Reference, comparability and the CAS
   token. A workflow whose experiments predate the selection control answers
   the second with a refusal, and then this list is exactly what it was before
   plus a line saying decisions are not available here. */
function renderTaskRunList(card, experimentId, taskId, label, evidence, runs, nav) {
  var decisions = !runs.unavailable && runs.decisions_available !== false;
  if (runs.unavailable) {
    selectionNote(card,
      runs.sealed
        ? "This is sealed evidence, so it records no decisions."
        : "No best run can be recorded for this task yet.",
      runs.error);
  } else if (!decisions) {
    selectionNote(card,
      "This archive carries the evidence, not the decisions.",
      "Best run and Reference are the live workflow's judgements. An archive "
      + "does not hold them, so none is shown rather than an empty badge that "
      + "would read as 'nobody has chosen yet'.");
  }
  var byAttempt = {};
  evidence.forEach(function (row) { byAttempt[row.attempt] = row; });
  var selected = {};
  (runs.attempts || []).forEach(function (row) { selected[row.attempt] = row; });
  /* The decision view is authoritative about WHICH attempts exist when it can
     answer, because it projects the recorded attempt rows; the evidence view
     is the fallback and the source of the extra chrome either way. */
  var rows = (runs.attempts && runs.attempts.length)
    ? runs.attempts
    : evidence.map(function (row) { return row; });
  if (!rows.length) {
    card.appendChild(el("div", "empty", "no attempts recorded for this task"));
    return;
  }
  /* Existence of the DECISION, not readability of the row it names: a best run
     whose attempt row cannot be read right now is still a recorded choice, and
     reporting it as "nobody has chosen" would erase somebody's decision. */
  var pinned = runs.best_run || null;
  if (!runs.unavailable && decisions) {
    var head = el("div", "sub");
    head.textContent = pinned
      ? ("Best run: attempt " + pinned.attempt + ". " + bestRunProvenance(pinned))
      : (runs.reference
          ? ("No best run has been chosen. Attempt " + runs.reference.attempt
             + " is the Reference, which is a viewing default and not a decision.")
          : "No best run has been chosen and no completed attempt is available "
            + "as a Reference.");
    card.appendChild(head);
    if (pinned && pinned.attempt_resolved === false) {
      card.appendChild(el("div", "sub",
        "That attempt's own record cannot be read from here, so what it did is "
        + "unknown — the decision to prefer it stands regardless."));
    }
  }
  /* Assigned by the panel below, which renders after these rows. A tick
     before it exists cannot happen -- the panel is appended synchronously in
     the same paint -- and the no-op default says so rather than crashing. */
  var selectionChanged = function () {};
  rows.forEach(function (row) {
    var extra = byAttempt[row.attempt] || {};
    var item = el("div", "listItem");
    item.setAttribute("data-attempt", String(row.attempt));
    appendRunSelector(item, row);
    var sub = "source: " + (row.outcome_source || extra.outcome_source || "—")
      + (row.reward === null || row.reward === undefined
          ? "" : "  reward " + row.reward)
      + (row.restarts ? "  restarts " + row.restarts : "")
      + (extra.channel_id ? "  · " + extra.channel_id : "")
      + (row.turn_count === undefined
          ? "" : "  · " + fmtCount(row.turn_count) + " recorded turn(s)");
    var subLine = el("div", "sub", sub);
    appendTokenLimitChip(subLine, extra.llm_calls_cut_at_limit);
    appendCostChip(subLine, extra.llm_cost);
    item.appendChild(subLine);
    /* The attempt row carries the evidence verdict with its stored reasons
       and the collapsed server configuration; the per-segment deltas stay
       on the experiment page above, once. */
    var verdictBox = el("div", "sub");
    renderEvidenceVerdict(verdictBox, extra.evidence, { segments: false });
    item.appendChild(verdictBox);
    renderServerConfiguration(item, extra);
    var open = el("button", "evidenceLink", "Open the recorded turns");
    open.type = "button";
    open.addEventListener("click", function () {
      openExperimentAttempt(experimentId, taskId, row.attempt);
    });
    var actions = el("div", "runDecision");
    actions.appendChild(open);
    renderAttemptActions(actions, experimentId, taskId, label, row, runs, decisions);
    item.appendChild(actions);
    card.appendChild(item);
  });
  selectionChanged = renderSelectedRunsPanel(
    card, experimentId, taskId, label, runs, nav
  );
  selectionChanged();
  if (!runs.unavailable && decisions) {
    renderBestRunHistory(card, experimentId, taskId, nav);
  }

  /* The tick itself. Default unticked: a summary is something a reader asks
     for, never the state the page opens in. Only a FINISHED run can be
     ticked, because a run still going has not produced the behaviour a
     summary would be about -- and it is shown disabled with the reason
     rather than omitted, so the reader can see it was not quietly counted.
     An attempt whose finished state is not known here (an evidence-only
     fallback list, with no decision view to say) gets no tick at all rather
     than one whose meaning nobody can state. */
  function appendRunSelector(item, row) {
    var finished = row.selectable === true || row.finished === true;
    var unfinished = row.selectable === false || row.finished === false;
    if (!finished && !unfinished) { item.appendChild(attemptTitle(row)); return; }
    var wrap = el("label", "sub");
    var box = el("input");
    box.type = "checkbox";
    box.setAttribute("data-run-select", String(row.attempt));
    box.setAttribute("aria-label", "Include attempt " + row.attempt
                                   + " in the summary of selected runs");
    if (unfinished) {
      box.disabled = true;
      box.title = "This attempt has not finished, so it cannot be summarized "
        + "with the others yet.";
    } else {
      box.checked = !!selectedRuns.chosen[row.attempt];
      box.addEventListener("change", function () {
        selectedRuns.chosen[row.attempt] = box.checked;
        selectionChanged();
      });
    }
    wrap.appendChild(box);
    wrap.appendChild(el("span", null, " summarize"));
    item.appendChild(wrap);
    item.appendChild(attemptTitle(row));
    if (unfinished) {
      item.appendChild(el("span", "sub",
        "Not finished, so it cannot be part of a summary of runs."));
    }
  }
}

