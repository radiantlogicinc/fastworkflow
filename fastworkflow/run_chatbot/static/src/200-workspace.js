/* -- read-only multi-store workspace ----------------------------------- */
var workspaceStores = {};

function workspaceStoreText(storeId) {
  var store = workspaceStores[storeId] || {};
  var digest = store.observed_sha256 || store.sha256 || "";
  return (store.label || storeId) + " · " + (store.mode || "unknown")
    + " evidence · integrity " + (store.integrity || "unknown")
    + (digest ? " · sha256 " + digest.slice(0, 12) + "…" : "");
}

function scopedTraceHref(storeId, logicalTurnKey) {
  return "/trace?token=" + encodeURIComponent(TOKEN)
    + "&store_id=" + encodeURIComponent(storeId)
    + "&logical_turn_key=" + encodeURIComponent(logicalTurnKey);
}

function workspaceTurnLink(storeId, logicalTurnKey, label, cutAtLimit, stamps) {
  var row = el("div", "listItem");
  row.appendChild(el("div", "title", label || logicalTurnKey));
  var sub = el("div", "sub",
    "store " + storeId + " · logical turn key " + logicalTurnKey);
  appendTokenLimitChip(sub, cutAtLimit);
  if (stamps) {
    appendSignalChips(sub, stamps.decision_signals);
    appendCostChip(sub, stamps.llm_cost);
  }
  row.appendChild(sub);
  row.setAttribute("data-store-id", storeId);
  row.setAttribute("data-logical-turn-key", logicalTurnKey);
  makeRowActivatable(row, function () {
    selectWorkspaceTurn(storeId, logicalTurnKey);
  });
  var durable = el("a", null, "durable link");
  durable.setAttribute("href", scopedTraceHref(storeId, logicalTurnKey));
  durable.addEventListener("click", function (evt) { evt.stopPropagation(); });
  row.appendChild(durable);
  return row;
}

function renderWorkspaceAttempt(container, row, provenance) {
  var box = el("div", "card");
  box.appendChild(el("h2", null,
    provenance + " · task " + (row.task_id || "—")
    + " · attempt " + (row.attempt == null ? "—" : row.attempt)));
  var source = row.outcome_source
    || (row.resolved_attempt && row.resolved_attempt.outcome_source) || "pending";
  var outcome = row.outcome
    || (row.resolved_attempt && row.resolved_attempt.outcome) || "pending evaluation";
  var outcomeLine = el("div", "sub",
    "outcome " + outcome + " · source " + source);
  appendTokenLimitChip(outcomeLine, row.llm_calls_cut_at_limit);
  appendCostChip(outcomeLine, row.llm_cost);
  box.appendChild(outcomeLine);
  /* Native rows carry their own verdict and stamp; a projected row's come
     from the first source the archive resolved for it. */
  var resolvedSource = (row.resolved_sources || []).filter(function (sourceRow) {
    return sourceRow && (sourceRow.evidence || sourceRow.resolved_attempt);
  })[0] || null;
  var verdictBox = el("div", "sub");
  renderEvidenceVerdict(verdictBox,
    row.evidence || (resolvedSource && resolvedSource.evidence) || null,
    { segments: false });
  box.appendChild(verdictBox);
  renderServerConfiguration(box,
    ("runtime_snapshot" in row) ? row
      : (resolvedSource && resolvedSource.resolved_attempt) || null);

  var refs = [];
  (row.turn_refs || []).forEach(function (ref) {
    refs.push({
      store_id: ref.store_id,
      logical_turn_key: ref.logical_turn_key,
      cut: ref.llm_calls_cut_at_limit,
      stamps: ref,
      label: provenance === "projected history"
        ? "projected logical turn" : "labelled turn"
    });
  });
  (row.resolved_sources || []).forEach(function (sourceRow) {
    var ref = sourceRow.turn_ref || {};
    if (ref.store_id && ref.logical_turn_key) {
      refs.push({
        store_id: ref.store_id,
        logical_turn_key: ref.logical_turn_key,
        cut: (sourceRow.resolved_turn || {}).llm_calls_cut_at_limit,
        stamps: sourceRow.resolved_turn || null,
        label: "projected logical turn"
      });
    }
  });
  (row.resolved_turns || []).forEach(function (turn) {
    if (turn && turn.store_id && turn.logical_turn_key) {
      refs.push({
        store_id: turn.store_id,
        logical_turn_key: turn.logical_turn_key,
        cut: turn.llm_calls_cut_at_limit,
        stamps: turn,
        label: "projected logical turn"
      });
    }
  });
  if (!refs.length) {
    box.appendChild(el("div", "empty", "No turn references recorded."));
  }
  refs.forEach(function (ref, index) {
    box.appendChild(workspaceTurnLink(
      ref.store_id,
      ref.logical_turn_key,
      ref.label + " " + (index + 1),
      ref.cut,
      ref.stamps
    ));
  });
  container.appendChild(box);
}

function showWorkspaceExperiment(experiment) {
  var nav = ++workspaceNav;
  writePageLink({experiment: experiment.experiment_id});
  state.experimentId = experiment.experiment_id;
  var d = document.getElementById("detail");
  clear(d);
  d.appendChild(el("div", "empty", "loading workspace experiment…"));
  var encoded = encodeURIComponent(experiment.experiment_id);
  Promise.all([
    api("/api/workspace/experiment/" + encoded + "/segments"),
    api("/api/workspace/experiment/" + encoded + "/tasks"),
    api("/api/workspace/experiment/" + encoded + "/attempts"),
    api("/api/workspace/projected_attempts?experiment=" + encoded)
  ]).then(function (results) {
    if (nav !== workspaceNav) { return; }
    clear(d);
    var head = el("div", "card");
    head.appendChild(el("h2", null,
      "Logical experiment · " + (experiment.label || experiment.experiment_id)));
    head.appendChild(el("div", "sub", "workspace " + session.workspace.label));
    (results[0].segments || []).forEach(function (segment) {
      var segBox = el("div");
      segBox.appendChild(el("div", "sub",
        "segment " + segment.segment_id + " · " + workspaceStoreText(segment.store_id)
        + " · native capture"));
      /* The archive's own evidence verdict for this segment, reasons
         verbatim, writer-health delta folded under it. */
      renderEvidenceVerdict(segBox, segment.evidence);
      renderProvenance(segBox, segment.provenance,
        "provenance · segment " + segment.segment_id);
      /* Pinned corpus, checked against the catalogue when this workspace
         names the workflow folder it was sealed from. */
      renderBenchmarkPin(segBox, segment.benchmark_pin);
      head.appendChild(segBox);
    });
    d.appendChild(head);
    (results[2].attempts || []).forEach(function (attempt) {
      renderWorkspaceAttempt(d, attempt, "native capture");
    });
    (results[3].projected_attempts || []).forEach(function (attempt) {
      renderWorkspaceAttempt(d, attempt, "projected history");
    });
    if (!(results[2].attempts || []).length
        && !(results[3].projected_attempts || []).length) {
      d.appendChild(el("div", "empty",
        "No attempts recorded; evaluation is pending."));
    }
  }).catch(function (e) {
    if (nav !== workspaceNav) { return; }
    clear(d);
    d.appendChild(el("div", "empty",
      "Could not load workspace experiment: " + e.message));
  });
}

function loadWorkspaceChrome() {
  Promise.all([
    api("/api/workspace/stores"),
    api("/api/workspace/experiments")
  ]).then(function (results) {
    workspaceStores = {};
    (results[0].stores || []).forEach(function (store) {
      workspaceStores[store.store_id] = store;
    });
    var chrome = document.getElementById("workspaceChrome");
    chrome.className = "visible";
    chrome.textContent = session.workspace.label + " · "
      + session.workspace.store_count + " stores · read-only";
    refreshConvs();
  }).catch(function (e) {
    var d = document.getElementById("detail");
    clear(d);
    d.appendChild(el("div", "empty", "Could not load workspace: " + e.message));
  });
}

/* `spanId` and `note` are optional and used by `openPairSpan` only: a caller
   that wants one exact recorded call focused once this archive's trace is on
   screen. Every existing caller opens the turn and passes neither. `level` is
   the position a page link names inside the turn. */
var turnLoadAbort = null;
function selectWorkspaceTurn(storeId, logicalTurnKey, spanId, note, level) {
  var detailNav = expNavToken();
  var nav = ++workspaceNav;
  /* Clear first: a slow read from another archive must never leave the old
     trace visible under the newly selected store label. */
  state.storeId = storeId;
  state.turnKey = logicalTurnKey;
  if (!review.progress) { alignHierarchyTurn(logicalTurnKey); }
  state.turn = null;
  state.path = [];
  var d = document.getElementById("detail");
  clear(d);
  d.appendChild(el("div", "empty",
    "loading " + logicalTurnKey + " from store " + storeId + "…"));
  if (review.progress) {
    location.hash = reviewHash(review.progress.assignment.rows[review.rowIndex]);
  } else {
    writePageLink({store: storeId, turn: logicalTurnKey});
  }
  if (turnLoadAbort) { turnLoadAbort.abort(); }
  turnLoadAbort = (typeof AbortController !== "undefined") ? new AbortController() : null;
  var signal = turnLoadAbort && turnLoadAbort.signal;
  var requests;
  if (review.progress) {
    var row = review.progress.assignment.rows[review.rowIndex];
    var reviewBase = "/api/review/assignments/"
      + encodeURIComponent(review.assignmentId) + "/rows/"
      + encodeURIComponent(row.id);
    requests = [
      reviewApi(reviewBase + "/turn", "GET"),
      reviewApi(reviewBase + "/trace", "GET")
    ];
  } else {
    requests = [
      api("/api/workspace/turn/" + encodeURIComponent(storeId)
        + "/" + encodeURIComponent(logicalTurnKey), { signal: signal }),
      api("/api/workspace/trace/" + encodeURIComponent(storeId)
        + "/" + encodeURIComponent(logicalTurnKey), { signal: signal })
    ];
  }
  Promise.all(requests).then(function (results) {
    if (nav !== workspaceNav || state.storeId !== storeId || expNavStale(detailNav)) { return; }
    var turn = results[0].turn;
    turn.turn_key = turn.logical_turn_key || turn.turn_key;
    renderDetail(turn, results[1].spans || []);
    focusLoadedSpan(spanId, note);
    focusLoadedLevel(level);
  }).catch(function (e) {
    if (requestWasAborted(e) || nav !== workspaceNav || state.storeId !== storeId || expNavStale(detailNav)) { return; }
    clear(d);
    d.appendChild(el("div", "empty", "Failed to load scoped turn: " + e.message));
  });
}

