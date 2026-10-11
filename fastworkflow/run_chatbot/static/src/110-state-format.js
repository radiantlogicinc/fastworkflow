/* -- state ------------------------------------------------------------- */
var state = {
  channel: "",              // "" = all channels
  turnKey: null,
  turn: null,               // the loaded turn record for the open drill-down
  path: [],                 // node path from the turn down to the open level
  experimentId: null,
  experimentTask: null,
  benchmarkId: null,
  benchmarkVersion: null
};
var session = null;          // /api/session payload: workflow + server + identity

/* -- formatting -------------------------------------------------------- */
function fmtNs(ns) {
  if (ns === null || ns === undefined) { return "…"; }
  var ms = ns / 1e6;
  if (ms < 1) { return ms.toFixed(2) + " ms"; }
  if (ms < 1000) { return Math.round(ms) + " ms"; }
  return (ms / 1000).toFixed(2) + " s";
}
function fmtMs(ms) {
  if (ms === null || ms === undefined) { return ""; }
  if (ms < 1000) { return ms + " ms"; }
  return (ms / 1000).toFixed(1) + " s";
}
function fmtCount(value) {
  /* Grouped: a turn total runs to five figures and "10234" misreads. */
  return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
function fmtTokens(tokens) {
  /* Split in/out, because they are not interchangeable: input scales with the
     context you resend every step, output with what the model actually wrote,
     and only the split tells you which one a slow turn is paying for. Falls
     back to the total when a provider reports no breakdown.

     A level that made LLM calls always says something (fix-9eg.5). It used to
     fall silent whenever the total was falsy, which put a call that recorded
     `total_tokens: 0` and a call that recorded no usage at all in the same
     blank space; the counts on the token object now tell those apart, and the
     one that nothing counted says so. */
  if (!tokens || typeof tokens !== "object") { return ""; }
  if (!tokens.calls) { return tokens.total ? fmtCount(tokens.total) + " tok" : ""; }
  if (!tokens.counted && !tokens.partial) { return "tokens not recorded"; }
  var text = (!tokens.in && !tokens.out)
    ? fmtCount(tokens.total) + " tok"
    : "tok - " + fmtCount(tokens.in) + " in, " + fmtCount(tokens.out) + " out";
  var incomplete = tokens.partial + tokens.unrecorded;
  if (incomplete) {
    text += " (" + incomplete + " of " + tokens.calls
      + (tokens.calls === 1 ? " call" : " calls") + " incompletely counted)";
  }
  /* Named separately from `incomplete`: a shared call's tokens are not missing,
     they are counted once under the call that is credited with the response. */
  if (countOf(tokens.shared)) {
    text += " (" + tokens.shared + " counted under another call quoting the "
      + "same provider response)";
  }
  return text;
}
function fmtCost(duration, tokens, cost) {
  /* Latency, tokens and money are the costs of a level; a level that spent no
     tokens says so by omission rather than by a "0 tok" that reads like a
     measurement failure, and a level whose calls recorded no cost says
     "cost not recorded" rather than nothing (fix-aou (d)). */
  var parts = duration ? [duration] : [];
  var text = fmtTokens(tokens);
  if (text) { parts.push(text); }
  var money = fmtCostAmount(cost);
  if (money) { parts.push(money); }
  return parts.join(" | ");
}
function fmtTs(iso) {
  if (!iso) { return "—"; }
  return iso.replace("T", " ").replace(/\.\d+.*$/, "");
}
function pretty(objOrText) {
  var value = objOrText;
  if (typeof value === "string") {
    try { value = JSON.parse(value); }
    catch (e) { return value; }
  }
  return JSON.stringify(value, null, 2);
}

/* -- token-limit detection [fix-49m.6 (a)] ------------------------------ */
/* An fw.llm.call whose completion_tokens equals the max_tokens it was called
   with stopped because it hit the cap, not because it finished. call_kwargs
   is flat (call_kwargs.max_tokens). Anything missing or non-integral answers
   false: the chip is never a guess, so absence of evidence is "no". */
function exactInt(value) {
  return typeof value === "number" && isFinite(value) && Math.floor(value) === value
    ? value : null;
}
function llmCallCutAtLimit(span) {
  if (!span || span.name !== "fw.llm.call") { return false; }
  var a = span.attributes || {};
  var usage = parsedAttr(a.usage);
  var kwargs = parsedAttr(a.call_kwargs);
  if (!usage || typeof usage !== "object" || !kwargs || typeof kwargs !== "object") {
    return false;
  }
  var produced = exactInt(usage.completion_tokens);
  var cap = exactInt(kwargs.max_tokens);
  return produced !== null && cap !== null && cap > 0 && produced === cap;
}
function sumCut(nodes) {
  return nodes.reduce(function (total, node) { return total + (node.cut || 0); }, 0);
}
function tokenLimitChip(count) {
  /* No count = the chip on one call. A number = a per-turn or per-attempt
     tally, and a tally of zero is no chip rather than a "0" that reads like a
     measurement. */
  if (count === undefined) { return el("span", "chipLimit", "cut at limit"); }
  if (!count) { return null; }
  return el("span", "chipLimit",
    fmtCount(count) + (count === 1 ? " call" : " calls") + " cut at limit");
}
function appendTokenLimitChip(parent, count) {
  var chip = tokenLimitChip(count);
  if (chip) { parent.appendChild(chip); }
}

/* -- evidence-run verdict [fix-49m.6 (b)] ------------------------------- */
/* `verdict` is the server's evidence_verdict(): state valid | invalid |
   unrecorded, the segments with their stored problems and writer-health
   deltas, the invalid segments' problems flattened, and the valid segments'
   problems as warnings. Reasons are quoted, never paraphrased. */
function evidenceBadge(verdict) {
  if (!verdict || verdict.state === "unrecorded") {
    return el("span", "badge unknown", "no evidence run recorded");
  }
  var n = (verdict.segments || []).length;
  var segs = n + (n === 1 ? " segment" : " segments");
  if (verdict.state === "valid") {
    return el("span", "badge ok", "evidence valid · " + segs);
  }
  return el("span", "badge fail", "evidence INVALID · " + segs);
}
function renderEvidenceVerdict(container, verdict, opts) {
  opts = opts || {};
  container.appendChild(evidenceBadge(verdict));
  if (!verdict || verdict.state === "unrecorded") { return; }
  var problems = verdict.problems || [];
  var warnings = verdict.warnings || [];
  if (verdict.state === "invalid") {
    var box = el("div", "expInvalid evidenceProblems");
    if (!problems.length) {
      box.appendChild(el("div", null,
        "The invalid segment's stored record retains no reason."));
    }
    problems.forEach(function (problem) { box.appendChild(el("div", null, problem)); });
    container.appendChild(box);
  }
  if (warnings.length) {
    var warn = el("div", "nonComparable evidenceProblems");
    warnings.forEach(function (problem) { warn.appendChild(el("div", null, problem)); });
    container.appendChild(warn);
  }
  if (opts.segments !== false) {
    renderEvidenceSegments(container, verdict.segments || []);
  }
}
function renderEvidenceSegments(container, segments) {
  segments.forEach(function (seg) {
    var det = el("details", "evidenceSegment");
    det.appendChild(el("summary", null,
      "#" + seg.seq + "  " + seg.evidence_run_id + "  "
      + (seg.valid ? "valid" : "INVALID") + " · writer-health delta"));
    (seg.problems || []).forEach(function (problem) {
      det.appendChild(el("div", seg.valid ? "sub" : "expInvalid", problem));
    });
    var delta = seg.writer_health_delta;
    if (!delta || typeof delta !== "object") {
      det.appendChild(el("div", "sub",
        "writer-health delta not recorded for this segment"));
    } else {
      var kv = el("dl", "kv");
      Object.keys(delta).forEach(function (key) {
        kv.appendChild(el("dt", null, key));
        var value = delta[key];
        kv.appendChild(el("dd", null, Array.isArray(value)
          ? (value.length ? value.join(", ") : "(none)")
          : String(value)));
      });
      det.appendChild(kv);
    }
    det.addEventListener("click", function (evt) { evt.stopPropagation(); });
    container.appendChild(det);
  });
}

/* -- server configuration [fix-49m.6 (d)] ------------------------------- */
/* The runtime readiness snapshot the binding server stamped on the attempt
   (runtime_readiness.runtime_readiness_snapshot, decoded by the store into
   `runtime_snapshot`). Keys print verbatim; nothing is labelled that the
   snapshot did not record, and a null stamp says so. */
function renderServerConfiguration(container, attempt) {
  var det = el("details", "serverConfig");
  det.appendChild(el("summary", null, "server configuration"));
  var snapshot = attempt ? attempt.runtime_snapshot : null;
  if (!snapshot || typeof snapshot !== "object") {
    det.appendChild(el("div", "sub",
      attempt && attempt.runtime_snapshot_json !== undefined
        ? "stamped for this attempt but not readable as an object; raw stamp: "
          + attempt.runtime_snapshot_json
        : "not recorded for this attempt"));
  } else {
    var kv = el("dl", "kv");
    Object.keys(snapshot).forEach(function (key) {
      kv.appendChild(el("dt", null, key));
      var value = snapshot[key];
      var dd = el("dd");
      if (value !== null && typeof value === "object") {
        dd.appendChild(el("pre", "json", pretty(value)));
      } else {
        dd.textContent = String(value);
      }
      kv.appendChild(dd);
    });
    det.appendChild(kv);
  }
  /* The section folds inside a clickable row; toggling it must not open the
     attempt. */
  det.addEventListener("click", function (evt) { evt.stopPropagation(); });
  container.appendChild(det);
}

/* -- decision-signal chips [fix-aou (b)] --------------------------------- */
/* `signals` is the server's turn_decision_signals(): the least confident
   intent decision's classifier top-k margin (null when no decision recorded
   one -- an exact-prefix match records no number, which is neither confident
   nor low) and how many times the user was asked. No signal = no chip, and
   never a filter hit. The threshold is the user's: no calibrated one is on record, because
   decision_signals is capture-only, so the default below is a viewing aid
   the rail says as much about. */
var LOW_CONFIDENCE_DEFAULT_MARGIN = 0.2;
function lowConfidenceThreshold() { return LOW_CONFIDENCE_DEFAULT_MARGIN; }
function finiteNumber(value) {
  return typeof value === "number" && isFinite(value) ? value : null;
}
function signalChips(signals) {
  var chips = [];
  if (!signals || typeof signals !== "object") { return chips; }
  var margin = finiteNumber(signals.intent_margin_min);
  if (margin !== null) {
    var low = margin < lowConfidenceThreshold();
    var n = signals.intent_margin_decisions || 0;
    chips.push(el("span", "chipSignal" + (low ? " low" : ""),
      (low ? "low confidence · " : "") + "margin " + margin.toFixed(3)
      + (n > 1 ? " (min of " + n + ")" : "")));
  }
  if (signals.asked_user) {
    chips.push(el("span", "chipSignal asked",
      "asked the user" + (signals.asked_user > 1 ? " ×" + signals.asked_user : "")));
  }
  return chips;
}
function appendSignalChips(parent, signals) {
  signalChips(signals).forEach(function (chip) { parent.appendChild(chip); });
}

/* -- diagnostic markers [fix-9eg.18.2/.18.3] ------------------------------
   The server's `diagnosis` projection, rendered verbatim. Each marker names
   something the trace or the turn record RECORDED; none of them is inferred
   from a count. Two pairs are deliberately kept apart here because conflating
   them is the failure mode the beads name:

   - "repeated a command" is an observation, "suspected loop" is a judgement
     the server made under a stated policy, and they are different chips.
   - an unknown is never rendered as a negative: a context whose handles are
     type-only says "context unknown", not "context unchanged".

   Order follows the server's MARKER_ORDER, which arrives as the key order of
   `facets`, so the vocabulary is not duplicated as a client-side list. */
var MARKER_LABEL = {
  context_navigation: {text: "navigated context", glyph: "\u2192", tone: "warn",
    help: "A dispatch's recorded context handles differ, or a producer flagged the move."},
  step_unsuccessful: {text: "command returned failure", glyph: "\u2715", tone: "bad",
    help: "A dispatch recorded success=false."},
  step_error: {text: "dispatch errored", glyph: "\u2715", tone: "bad",
    help: "A dispatch span's own status is error or failed."},
  intent_ambiguous: {text: "ambiguous intent", glyph: "?", tone: "warn",
    help: "Intent detection recorded ambiguous=true."},
  intent_error: {text: "intent detection errored", glyph: "\u2715", tone: "bad",
    help: "An fw.nlu.intent span's status is error."},
  parameter_extraction_invalid: {text: "parameters invalid", glyph: "\u2715", tone: "bad",
    help: "Extraction recorded invalid or missing fields."},
  parameter_extraction_error: {text: "extraction errored", glyph: "\u2715", tone: "bad",
    help: "An fw.nlu.param_extraction span's status is error."},
  parameter_extraction_retry: {text: "extraction retried", glyph: "\u21ba", tone: "warn",
    help: "Extraction recorded a retry round."},
  awaiting_user: {text: "asked the user", glyph: "\u270b", tone: "warn",
    help: "The turn suspended on a question. A suspension is not a failure."},
  repeated_command: {text: "repeated a command", glyph: "\u21bb", tone: "warn",
    help: "The same command ran more than once. On its own this is an observation, not a fault."},
  suspected_loop: {text: "suspected loop", glyph: "\u21bb", tone: "bad",
    help: "Repeats that themselves recorded trouble, under the policy shown below."},
  low_confidence: {text: "low confidence", glyph: "\u2248", tone: "warn",
    help: "The least confident recorded intent margin is under the threshold you set."},
  partial_evidence: {text: "partial evidence", glyph: "\u25CC", tone: "unknown",
    help: "Something this turn did was not fully recorded; treat the summary as incomplete."}
};
function markerMeta(name) {
  return MARKER_LABEL[name]
    || {text: name, glyph: "\u25CC", tone: "unknown",
        help: "This build has no description for this marker."};
}
function markerChip(name, count) {
  var meta = markerMeta(name);
  var chip = el("span", "chipMarker " + meta.tone);
  chip.appendChild(el("span", "glyph", meta.glyph));
  chip.appendChild(document.createTextNode(
    meta.text + (count > 1 ? " \u00d7" + count : "")));
  chip.title = meta.help;
  return chip;
}
function appendMarkerChips(parent, markers, counts) {
  (markers || []).forEach(function (name) {
    parent.appendChild(markerChip(name, (counts || {})[name] || 0));
  });
}

/* -- cost roll-ups [fix-aou (d)] ------------------------------------------ */
/* An fw.llm.call's `cost` attribute is what dspy_logger copied from the DSPy
   history entry (the provider cost as litellm reported it, in USD). A call
   that recorded none is counted as unrecorded, never as free. */
function noCost() { return { calls: 0, recorded: 0, unrecorded: 0, total: null }; }
function spanCost(span) {
  if (!span || span.name !== "fw.llm.call") { return noCost(); }
  var value = finiteNumber((span.attributes || {}).cost);
  if (value === null || value < 0) {
    return { calls: 1, recorded: 0, unrecorded: 1, total: null };
  }
  return { calls: 1, recorded: 1, unrecorded: 0, total: value };
}
function addCost(a, b) {
  var recorded = a.recorded + b.recorded;
  return {
    calls: a.calls + b.calls,
    recorded: recorded,
    unrecorded: a.unrecorded + b.unrecorded,
    total: recorded ? (a.total || 0) + (b.total || 0) : null
  };
}
function sumCost(nodes) {
  return nodes.reduce(function (total, node) {
    return addCost(total, node.cost || noCost());
  }, noCost());
}
function fmtCostAmount(cost) {
  /* "" for a level that made no LLM call; "cost not recorded" when calls were
     made and none recorded a cost; the sum otherwise, with the count of calls
     it does not cover. Never a zero standing in for unknown. */
  if (!cost || typeof cost !== "object" || !cost.calls) { return ""; }
  if (!cost.recorded || finiteNumber(cost.total) === null) { return "cost not recorded"; }
  var text = "cost " + cost.total.toFixed(4);
  if (cost.unrecorded) {
    text += " (" + cost.unrecorded
      + (cost.unrecorded === 1 ? " call" : " calls") + " not recorded)";
  }
  return text;
}
function appendCostChip(parent, cost) {
  var text = fmtCostAmount(cost);
  if (!text) { return; }
  parent.appendChild(el("span", "chipCost" + (cost.recorded ? "" : " unrecorded"), text));
}

/* -- LLM cache state [fix-9eg.6] ------------------------------------------ */
/* `cache_hit` is a boolean dspy_logger writes when the DSPy history entry
   carried a response to read it from, and does not write at all otherwise. So
   there are three answers here, not two, and the absent one is `unknown` — a
   call served from the cache is an observation about HOW the answer arrived,
   never a claim that it is stale, replayed or wrong. Nothing here changes any
   cache behaviour; it only reports what was recorded. */
function noCache() { return { hit: 0, miss: 0, unknown: 0 }; }
function cacheStateOf(span) {
  if (!span || span.name !== "fw.llm.call") { return ""; }
  var value = (span.attributes || {}).cache_hit;
  if (value === true) { return "hit"; }
  if (value === false) { return "miss"; }
  return "unknown";
}
function spanCache(span) {
  var state = cacheStateOf(span);
  if (!state) { return noCache(); }
  var counts = noCache();
  counts[state] = 1;
  return counts;
}
function addCache(a, b) {
  return { hit: a.hit + b.hit, miss: a.miss + b.miss, unknown: a.unknown + b.unknown };
}
function sumCache(nodes) {
  return nodes.reduce(function (total, node) {
    return addCache(total, node.cache || noCache());
  }, noCache());
}
function fmtCacheCounts(cache) {
  /* "" for a level with no LLM call. Otherwise the states that were recorded,
     with unknown named rather than folded into "miss". */
  if (!cache || typeof cache !== "object") { return ""; }
  var calls = cache.hit + cache.miss + cache.unknown;
  if (!calls) { return ""; }
  if (cache.unknown === calls) { return "cache state not recorded"; }
  var parts = [];
  if (cache.hit) { parts.push(cache.hit + " from cache"); }
  if (cache.miss) { parts.push(cache.miss + " from the provider"); }
  if (cache.unknown) { parts.push(cache.unknown + " not recorded"); }
  return "cache - " + parts.join(", ");
}
function appendCacheChip(parent, cache) {
  var text = fmtCacheCounts(cache);
  if (!text) { return; }
  parent.appendChild(el("span",
    "chipCost" + (cache.hit ? "" : " unrecorded"), text));
}

/* -- recorded usage, as the server projected it [fix-9eg.5] --------------- */
/* The comparison panes read their tokens, money and cache state off the
   projection's `usage` (comparison.usage_rollup) instead of re-deriving them
   from spans the pane never loads. Same numbers, same canonical calls, one
   accounting — which is the point: a chip a person reads and a field an agent
   reads must not be two independent tallies of the same evidence.

   Every absence keeps its own name. `coverage: "none"` means nothing counted
   tokens, which is not the same as a recorded zero, and a shared provider
   response is reported rather than added twice. */
function fmtUsageTokens(usage) {
  if (!usage || typeof usage !== "object" || !usage.calls) { return ""; }
  var tokens = usage.tokens || {};
  if (tokens.coverage === "none") { return "tokens not recorded"; }
  /* Each half is printed only if something recorded it, so a provider that
     reported a total and no split does not appear to have sent 0 prompt
     tokens. A recorded 0 does print, because it is a measurement. */
  var split = [];
  if (exactInt(tokens.prompt) !== null) {
    split.push(fmtCount(tokens.prompt) + " in");
  }
  if (exactInt(tokens.completion) !== null) {
    split.push(fmtCount(tokens.completion) + " out");
  }
  var text = split.length
    ? "tok - " + split.join(", ")
    : fmtCount(tokens.total) + " tok";
  var uncounted = (tokens.unrecorded || 0) + (tokens.partial || 0)
    + (tokens.shared || 0);
  if (uncounted) {
    text += " (" + uncounted + " of " + usage.calls
      + (usage.calls === 1 ? " call" : " calls") + " incompletely counted)";
  }
  return text;
}
function fmtUsageCalls(usage) {
  if (!usage || typeof usage !== "object") { return ""; }
  if (!usage.calls) { return "no LLM call recorded"; }
  var text = fmtCount(usage.calls) + (usage.calls === 1 ? " LLM call" : " LLM calls");
  if (usage.open) { text += ", " + usage.open + " still open"; }
  /* Why this number can be lower than the span count on screen. Both folds
     are named because they are different facts about the recording. */
  var folded = [];
  if (usage.records_folded) {
    folded.push(usage.records_folded + " repeated record(s)");
  }
  if (usage.wrappers_folded) {
    folded.push(usage.wrappers_folded + " nested wrapper(s)");
  }
  if (folded.length) { text += " — counted once despite " + folded.join(" and "); }
  if (usage.shared_responses) {
    text += " — " + usage.shared_responses
      + " quote a provider response another call already accounts for, so "
      + "their tokens are counted once";
  }
  return text;
}
function appendUsageChips(parent, usage) {
  /* The compact renderer: calls, then tokens, then cache, each omitted when
     there is nothing recorded to say. */
  if (!usage || typeof usage !== "object") { return; }
  var calls = fmtUsageCalls(usage);
  if (calls) { parent.appendChild(el("span", "chipCost", calls)); }
  var tokens = fmtUsageTokens(usage);
  if (tokens) {
    parent.appendChild(el("span",
      "chipCost" + (tokens === "tokens not recorded" ? " unrecorded" : ""),
      tokens));
  }
  appendCostChip(parent, usage.cost);
  appendCacheChip(parent, usage.cache);
}

/* -- execution ledger [fix-aou (a)] --------------------------------------- */
/* `ledger` is the server's execution_ledger(): the turn record's
   execution_records joined on command_call_id with the trace's
   fw.command.execute spans and the child_calls those spans file. A row's
   status is the span's own status column -- ok, error, or cancelled for a
   control signal -- and a row with no span has none and says so. A resumed
   turn is one ledger: dispatches from before the suspension keep their spans
   and come first, flagged as no longer listed by the record itself. */
function renderExecutionLedger(container, turn, openSpan) {
  var ledger = turn.execution_ledger;
  if (!ledger || typeof ledger !== "object") {
    container.appendChild(el("div", "empty", "No execution ledger in this view."));
    return;
  }
  var rows = ledger.rows || [];
  var summary = rows.length + (rows.length === 1 ? " dispatch" : " dispatches")
    + " · " + ledger.record_rows + " in the turn record · "
    + ledger.span_rows + " execute spans";
  if (ledger.rows_not_in_record) {
    summary += " · " + ledger.rows_not_in_record + " known only from spans"
      + (turn.suspended_ms || turn.continuation_of
        ? " (the record lists only dispatches since the last resume)" : "");
  }
  if (ledger.asked_user_outside_dispatch) {
    summary += " · asked the user " + ledger.asked_user_outside_dispatch
      + "× outside any dispatch";
  }
  container.appendChild(el("div", "sub", summary));
  if (!rows.length) {
    container.appendChild(el("div", "empty", "This turn dispatched no command."));
    return;
  }
  /* The diagnosis indexes its steps by the same command_call_id the ledger
     rows carry, so a marked step is the marked ROW rather than a second list
     beside it. Absent (a reader that did not ask for a diagnosis) the ledger
     renders exactly as it did before. */
  var diagnosis = turn.diagnosis;
  var stepByCall = {};
  ((diagnosis && diagnosis.steps) || []).forEach(function (step) {
    if (step.command_call_id) { stepByCall[step.command_call_id] = step; }
  });
  var wrap = el("div", "ledgerWrap");
  var table = el("table", "ledger");
  var head = el("tr");
  var columns = ["#", "command_call_id", "parent", "command", "context", "status",
    "duration", "asked user", "recorded in"];
  if (diagnosis) { columns.push("what was recorded"); }
  columns.forEach(function (h) { head.appendChild(el("th", null, h)); });
  table.appendChild(head);
  rows.forEach(function (row) {
    var tr = el("tr", row.parent_call_id ? "child" : null);
    tr.appendChild(el("td", null, String(row.position)));
    tr.appendChild(el("td", "mono", row.command_call_id));
    tr.appendChild(el("td", "mono",
      row.parent_call_id ? "↳ " + row.parent_call_id.slice(0, 8) : "—"));
    tr.appendChild(el("td", null,
      row.command_name ? row.command_name : "(not recorded)"));
    tr.appendChild(el("td", null, row.context ? row.context : "—"));
    var statusText = row.status ? row.status : "no span recorded";
    if (row.status && row.success === false) { statusText += " · success false"; }
    tr.appendChild(el("td", null, statusText));
    tr.appendChild(el("td", null,
      row.duration_ns === null || row.duration_ns === undefined
        ? (row.span_recorded ? "open…" : "—") : fmtNs(row.duration_ns)));
    tr.appendChild(el("td", null,
      row.asked_user ? "yes" + (row.asked_user > 1 ? " ×" + row.asked_user : "") : "no"));
    var where = [];
    if (row.in_record) { where.push("turn record"); }
    if (row.span_recorded) { where.push("span"); }
    if (row.child_call) { where.push("parent's child_calls"); }
    tr.appendChild(el("td", null, where.join(" + ") || "—"));
    if (diagnosis) {
      var step = stepByCall[row.command_call_id];
      var cell = el("td");
      if (step && step.markers && step.markers.length) {
        tr.classList.add("markedStep");
        appendMarkerChips(cell, step.markers, null);
      }
      if (step) { cell.appendChild(navigationNote(step.navigation)); }
      /* A dispatch the record knows and the trace does not stays listed and
         says so, rather than being dropped for having nothing to point at. */
      if (step && !row.span_id) {
        cell.appendChild(el("div", "diagNote",
          "recorded only in the turn record \u2014 no span to open"));
      }
      if (!cell.childNodes.length) { cell.textContent = "\u2014"; }
      tr.appendChild(cell);
    }
    if (row.span_id && openSpan) {
      tr.classList.add("openable");
      tr.title = "open this dispatch's span";
      tr.tabIndex = 0;
      tr.addEventListener("keydown", function (event) {
        if (event.target === tr && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); openSpan(row.span_id); }
      });
      tr.addEventListener("click", function () { openSpan(row.span_id); });
    }
    table.appendChild(tr);
  });
  wrap.appendChild(table);
  container.appendChild(wrap);
}
/* Navigation is a two-state answer and is rendered as one. `unknown` is the
   common case when a command stays put -- the recorded value is a context
   TYPE, so two equal types prove nothing about whether the context moved --
   and printing "unchanged" there would be a claim the evidence does not
   support. The basis is shown so the reason is legible. */
var NAVIGATION_BASIS_NOTE = {
  context_type_change: "the recorded context types differ",
  recorded_flag: "the producer flagged the move",
  same_context_type: "same context type before and after, so a move between instances cannot be ruled out",
  no_context_recorded: "this dispatch recorded no context type"
};
function navigationNote(navigation) {
  if (!navigation || !navigation.state) { return el("span"); }
  var note = el("div", "diagNote");
  var where = navigation.from || navigation.to
    ? " (" + (navigation.from || "?") + " \u2192 " + (navigation.to || "?") + ")" : "";
  note.textContent = (navigation.state === "changed" ? "context changed" : "context unknown") + where;
  var why = NAVIGATION_BASIS_NOTE[navigation.basis];
  if (why) { note.textContent += " \u2014 " + why; }
  return note;
}

/* The turn-level summary: what the whole turn recorded, what it could not,
   and a way into the exact spans behind each claim. Rendered from the
   server's projection; nothing here re-derives a marker. */
function renderTurnDiagnosis(container, turn, openSpan) {
  var diagnosis = turn.diagnosis;
  if (!diagnosis) {
    container.appendChild(el("div", "empty", "No diagnosis in this view."));
    return;
  }
  var markers = diagnosis.markers || [];
  if (!markers.length) {
    container.appendChild(el("div", "sub",
      "Nothing this build diagnoses was recorded in this turn."));
  } else {
    var chips = el("div");
    appendMarkerChips(chips, markers, diagnosis.counts);
    container.appendChild(chips);
  }
  var spanMarkers = (diagnosis.span_markers || {});
  (diagnosis.markers_only_in_record || []).forEach(function (name) {
    container.appendChild(el("div", "diagNote",
      markerMeta(name).text + " is known only from the turn record; "
      + "no span recorded it."));
  });
  /* Every marker's supporting spans, as a jump. The anchors come from the
     server so the span opened here is the span the marker was computed from,
     not a second guess at which one it meant. */
  var evidence = spanMarkers.evidence || {};
  markers.forEach(function (name) {
    var support = evidence[name];
    var ids = (support && support.span_ids) || [];
    if (!ids.length || !openSpan) { return; }
    var line = el("div", "diagNote");
    line.appendChild(document.createTextNode(markerMeta(name).text + ": "));
    ids.forEach(function (spanId, index) {
      var jump = el("button", "traceLink diagJump", "span " + (index + 1));
      jump.type = "button";
      jump.title = "open " + spanId;
      jump.addEventListener("click", function () { openSpan(spanId); });
      line.appendChild(jump);
    });
    if (support.truncated) {
      line.appendChild(document.createTextNode(
        " and " + (support.count - ids.length) + " more"));
    }
    container.appendChild(line);
  });
  (diagnosis.repeats || []).forEach(function (group) {
    var line = el("div", "diagNote");
    line.textContent = (group.command_name || "(unnamed command)")
      + " ran " + group.count + "\u00d7"
      + (group.suspected_loop
        ? " and those runs recorded " + (group.trouble || []).join(", ")
          + " \u2014 suspected loop"
        : " \u2014 repetition only; nothing in those runs recorded trouble");
    container.appendChild(line);
  });
  var policy = diagnosis.loop_policy;
  if (policy) {
    container.appendChild(el("div", "diagNote",
      "Loop policy " + policy.name + ": at least " + policy.min_repeats
      + " runs within " + policy.window_steps + " steps"
      + (policy.require_recorded_trouble
        ? ", and the runs themselves must have recorded trouble." : ".")));
  }
  var coverage = diagnosis.coverage || {};
  var gaps = Object.keys(coverage).filter(function (key) { return coverage[key]; });
  if (gaps.length) {
    var note = el("div", "diagNote");
    note.textContent = "Not fully recorded: " + gaps.map(function (key) {
      return key.replace(/_/g, " ") + " (" + coverage[key] + ")";
    }).join(", ") + ".";
    container.appendChild(note);
  }
}

function findSpanPath(node, spanId) {
  if (node.span && node.span.span_id === spanId) { return [node]; }
  for (var i = 0; i < node.children.length; i++) {
    var found = findSpanPath(node.children[i], spanId);
    if (found) { return [node].concat(found); }
  }
  return null;
}
function openSpanInTree(spanId) {
  var root = state.path[0];
  var path = root ? findSpanPath(root, spanId) : null;
  if (!path) { return; }
  state.path = path;
  renderLevel();
}

/* Focus one recorded span in the trace that has JUST been rendered.

   Called from inside `selectTurn`'s own stale-guarded
   completion and never from a timer: a span id identifies a call only within the
   trace that recorded it, so a focus request left running after a later
   navigation could land on a same-named span of ANOTHER run. Waiting for the
   render that was asked for is the only way to know which trace is on screen.

   A trace with no span of that id says so. A reader who asked for one exact
   call and silently got the top of a turn cannot tell that the evidence is
   missing, which is the whole thing this message exists to prevent. */
function focusLoadedSpan(spanId, note) {
  if (!spanId) { return; }
  if (state.path.length && findSpanPath(state.path[0], spanId)) {
    openSpanInTree(spanId);
    if (note) { note.textContent = ""; }
    return;
  }
  var missing = "The recorded trace of this turn holds no span " + spanId
    + ", so that exact call is not in the evidence this store holds. The turn "
    + "is open at its top level.";
  if (note) { note.textContent = missing; }
  /* Said on the screen the reader is now looking at. The note belongs to the
     pane the link was clicked on, and `renderDetail` has just replaced that
     pane -- so a message left only there is one nobody can see, which is
     indistinguishable from a link that silently opened the wrong thing. */
  var host = document.getElementById("detail");
  if (host) {
    var notice = el("div", "card");
    notice.appendChild(el("div", "empty", missing));
    host.insertBefore(notice, host.firstChild);
  }
}

/* -- provenance and comparability [fix-aou (c)] --------------------------- */
/* `differences` is the server's provenance_differences(): what the two
   compared experiments do not agree on, quoted, and never a reason to
   withhold the comparison. */
function provenanceValueText(value) {
  if (value === null || value === undefined) { return "null"; }
  return typeof value === "string" ? value : JSON.stringify(value);
}
function renderProvenanceDifferences(container, differences) {
  var box = el("div", "provDiff");
  if (!differences) {
    box.appendChild(el("div", "sub",
      "provenance comparability not checked in this view"));
    container.appendChild(box);
    return;
  }
  if (!differences.length) {
    box.appendChild(el("div", "sub",
      "provenance: no field differs between the two experiments"));
    container.appendChild(box);
    return;
  }
  var warn = el("div", "nonComparable");
  warn.appendChild(el("strong", null, "Provenance differs in " + differences.length
    + (differences.length === 1 ? " field" : " fields")
    + " (the comparison is shown regardless)."));
  var kv = el("dl", "kv");
  differences.forEach(function (d) {
    kv.appendChild(el("dt", null, d.key + " (" + d.source + ")"));
    var dd = el("dd");
    dd.appendChild(el("div", null, "treatment: " + (d.treatment === null || d.treatment === undefined
      ? "not recorded" : provenanceValueText(d.treatment))));
    dd.appendChild(el("div", null, "baseline: " + (d.baseline === null || d.baseline === undefined
      ? "not recorded" : provenanceValueText(d.baseline))));
    kv.appendChild(dd);
  });
  warn.appendChild(kv);
  box.appendChild(warn);
  container.appendChild(box);
}

