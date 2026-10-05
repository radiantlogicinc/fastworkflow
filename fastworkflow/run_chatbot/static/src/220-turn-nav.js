/* -- navigation --------------------------------------------------------- */
function renderCrumbs() {
  if (review.progress) {
    var reviewBar = el("nav", "crumbs");
    state.path.forEach(function (node, index) {
      if (index) { reviewBar.appendChild(el("span", "sep", "›")); }
      var button = el("button", null, node.crumb);
      button.addEventListener("click", function () { state.path = state.path.slice(0, index + 1); renderLevel(); });
      reviewBar.appendChild(button);
    });
    return reviewBar;
  }
  syncTraceHierarchy();
  var bar = el("nav", "crumbs");
  fillHierarchyCrumbs(bar);
  return bar;
}

function descend(node) {
  state.path = state.path.concat([node]);
  renderLevel();
}

function feedbackAnchor(node) {
  if (node.kind === "turn") { return []; }
  var ids = {};
  function collect(n) {
    if (n.span && n.span.span_id) { ids[n.span.span_id] = true; }
    else { n.children.forEach(collect); }
  }
  collect(node);
  return Object.keys(ids).sort();
}

function feedbackProvenanceLabel(provenance) {
  return {
    human: "Human",
    coding_agent: "Coding Agent",
    distillation_agent: "Distillation Agent"
  }[provenance] || provenance;
}

/* The owner-confirmed taxonomy, mirrored from observability/feedback.py. The
   selectors and watermarks below help a person choose; a coding agent posts
   the same two enum values to the same route. A test pins this constant
   against GET /api/feedback-taxonomy so the two cannot drift.

   This replaced a three-heading composer that wrote "What went wrong:" into
   the comment text and parsed it back out on read. That parser is gone: the
   category is metadata now, and re-deriving it from a heading would record a
   guess as the author's choice. */
var FEEDBACK_TAXONOMY = [
  { value: "observations_analysis", label: "Observations / Analysis", subcategories: [
    { value: "observation", label: "Observation",
      watermark: "What did you see? Quote or point at the evidence \u2014 the request, the answer, the step \u2014 without judging it yet." },
    { value: "analysis", label: "Analysis",
      watermark: "What do you think the observation means? You do not need to diagnose the implementation to say what it points at." }
  ] },
  { value: "conclusions", label: "Conclusions", subcategories: [
    { value: "what_went_right", label: "What went right",
      watermark: "What is worth preserving? Justified clarifying questions and honest reporting of incomplete work count." },
    { value: "what_went_wrong", label: "What went wrong",
      watermark: "What was missing or incorrect? Anchor it to the first point where behavior went wrong, not only to the answer." }
  ] },
  { value: "recommendations", label: "Recommendations", subcategories: [
    { value: "what_to_do", label: "What to do",
      watermark: "What should happen instead, next time? State the expected behavior rather than the code change." },
    { value: "what_not_to_do", label: "What not to do",
      watermark: "What should be avoided? Name the tempting fix that would make things worse, and why." }
  ] }
];
var FEEDBACK_HINTS = [
  "Read the request and final answer first. Describe what was useful, missing, or incorrect.",
  "Attach feedback to the relevant turn or component, especially the first point where behavior went wrong.",
  "Observations quote the evidence; analysis says what it means; conclusions judge it; recommendations say what to do next time.",
  "One comment, one category. Save again to record a second one \u2014 nothing is overwritten.",
  "For example: \"The request names three people, but this plan covers only two. It should retain all three and track completion separately.\"",
  "For clarification stops, distinguish whether the question was appropriate from whether the harness answered it."
];

function feedbackCategory(value) {
  for (var i = 0; i < FEEDBACK_TAXONOMY.length; i++) {
    if (FEEDBACK_TAXONOMY[i].value === value) { return FEEDBACK_TAXONOMY[i]; }
  }
  return null;
}

function feedbackSubcategory(category, value) {
  var found = feedbackCategory(category);
  if (!found) { return null; }
  for (var i = 0; i < found.subcategories.length; i++) {
    if (found.subcategories[i].value === value) { return found.subcategories[i]; }
  }
  return null;
}

/* What a recorded row is filed under, in words. A row written before the
   taxonomy has neither, and says so rather than borrowing a category. */
function feedbackClassification(row) {
  var category = feedbackCategory(row.category);
  var sub = feedbackSubcategory(row.category, row.subcategory);
  if (!category || !sub) { return "Unclassified (recorded before categories)"; }
  return category.label + " \u00b7 " + sub.label;
}

function feedbackRefLabel(ref) {
  if (!ref) { return ""; }
  var parts = [];
  if (ref.task_id) { parts.push(ref.task_id); }
  if (ref.attempt !== null && ref.attempt !== undefined) { parts.push("attempt " + ref.attempt); }
  if (ref.pass_id) { parts.push("pass " + ref.pass_id); }
  (ref.turn_keys || []).forEach(function (key) { parts.push(headSnippet(key, 28)); });
  return parts.join(" \u00b7 ");
}

/* Both sides of a comparison comment, each a link to its own evidence. The
   pair is frozen: re-picking a winner does not move these. */
function renderFeedbackPair(item, row) {
  if (!row.paired) { return; }
  var pair = el("div", "sub");
  pair.appendChild(el("span", null, "Compared with: "));
  var primaryRef = (row.anchors && row.anchors.primary && row.anchors.primary.ref) || null;
  [[primaryRef, row.target_label, "this side"],
   [row.paired.ref, row.paired.target_label, "other side"]].forEach(function (side, index) {
    var ref = side[0];
    if (!ref) { return; }
    if (index) { pair.appendChild(el("span", null, "  \u2194  ")); }
    var key = (ref.turn_keys || [])[0];
    var link = el("button", "evidenceLink", (side[1] || side[2]) + " (" + feedbackRefLabel(ref) + ")");
    link.type = "button";
    link.setAttribute("aria-label", "Open the evidence for " + (side[1] || side[2]));
    link.addEventListener("click", function () { if (key) { selectTurn(key); } });
    pair.appendChild(link);
  });
  item.appendChild(pair);
}

function renderFeedbackHistoryRow(row) {
  var item = el("div", "feedbackHistoryItem");
  item.appendChild(el("div", "sub",
    row.created_at + " \u00b7 " + feedbackProvenanceLabel(row.provenance)
    + " \u00b7 " + feedbackClassification(row)));
  /* Verbatim. The stored comment is the author's text and is never re-split,
     re-headed or summarised on the way to the screen. */
  item.appendChild(el("pre", "json", row.comment || ""));
  renderFeedbackPair(item, row);
  return item;
}

function renderFeedback(parent, node) {
  // Formal blinded assignments retain their own capability-gated review UI.
  if (review.progress) { return; }
  var ids = feedbackAnchor(node);
  var card = el("div", "card feedbackCard");
  /* Just the field name: the level this feedback is anchored to is named in
     the heading at the top of the same page, and for a turn the qualifier was
     the user message printed a few blocks above. */
  card.appendChild(el("h2", null, "Feedback"));
  parent.appendChild(card);
  if (node.kind !== "turn" && !ids.length) {
    card.appendChild(el("p", "sub", "No recorded span identifies this component. Record feedback on the turn."));
    return;
  }
  /* Distinct read and write surfaces (fix-9eg.16): listing notes is a GET on
     its own path, and /post_feedback only ever appends one. */
  var scope = "turn_key=" + encodeURIComponent(state.turn.turn_key);
  if (session && session.workspace_mode) { scope += "&store_id=" + encodeURIComponent(state.storeId); }
  var readPath = "/api/feedback-notes?" + scope;
  var writePath = "/post_feedback?" + scope;
  var history = el("div"), note = el("p", "sub", "Loading feedback\u2026");
  var selected = { category: FEEDBACK_TAXONOMY[0].value,
                   subcategory: FEEDBACK_TAXONOMY[0].subcategories[0].value };
  var tabs = el("div", "feedbackTabs");
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Feedback category");
  var subTabs = el("div", "feedbackTabs");
  subTabs.setAttribute("role", "tablist");
  subTabs.setAttribute("aria-label", "Feedback subcategory");
  var composer = el("div", "feedbackComposer");
  var area = el("textarea", "expNotes");
  area.id = "feedback-comment";
  area.maxLength = 100000;
  composer.appendChild(subTabs);
  composer.appendChild(area);
  var save = el("button", null, "Save feedback"); save.disabled = true;
  var categoryButtons = {};
  /* One category and one of its subcategories at a time, with a roving
     tabindex -- the same switch setNavigationTabButtons makes on the rail.
     The watermark follows the subcategory, because that is the level the six
     confirmed meanings live at. */
  function selectSubcategory(value) {
    selected.subcategory = value;
    var sub = feedbackSubcategory(selected.category, value);
    area.placeholder = sub ? sub.watermark : "";
    area.setAttribute("aria-label",
      (sub ? sub.label : "Comment") + " for " + node.title);
    Array.prototype.forEach.call(subTabs.childNodes, function (button) {
      var on = button.getAttribute("data-value") === value;
      button.className = on ? "active" : "";
      button.setAttribute("aria-selected", on ? "true" : "false");
      button.tabIndex = on ? 0 : -1;
    });
  }
  function selectCategory(value) {
    selected.category = value;
    var category = feedbackCategory(value);
    Object.keys(categoryButtons).forEach(function (key) {
      var on = key === value;
      categoryButtons[key].className = on ? "active" : "";
      categoryButtons[key].setAttribute("aria-selected", on ? "true" : "false");
      categoryButtons[key].tabIndex = on ? 0 : -1;
    });
    clear(subTabs);
    category.subcategories.forEach(function (sub) {
      var button = el("button", null, sub.label);
      button.type = "button";
      button.setAttribute("data-value", sub.value);
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", area.id);
      button.addEventListener("click", function () {
        selectSubcategory(sub.value);
        area.focus();
      });
      subTabs.appendChild(button);
    });
    selectSubcategory(category.subcategories[0].value);
  }
  FEEDBACK_TAXONOMY.forEach(function (category) {
    var button = el("button", null, category.label);
    button.type = "button";
    button.id = "feedback-tab-" + category.value;
    button.setAttribute("data-value", category.value);
    button.setAttribute("role", "tab");
    categoryButtons[category.value] = button;
    button.addEventListener("click", function () { selectCategory(category.value); });
    tabs.appendChild(button);
  });
  selectCategory(FEEDBACK_TAXONOMY[0].value);
  var head = el("div", "feedbackComposerHead");
  var hints = el("button", "feedbackHints", "Hints");
  hints.type = "button";
  hints.setAttribute("aria-label", "Hints for drafting feedback");
  var tip = el("div", "feedbackHintTip");
  tip.setAttribute("role", "tooltip");
  FEEDBACK_HINTS.forEach(function (line) { tip.appendChild(el("p", null, line)); });
  hints.appendChild(tip);
  head.appendChild(tabs);
  head.appendChild(hints);
  /* Hints shares the tab row (bottom-aligned, right) so the strip sits on
     the box it switches without a pointer row between them. */
  card.appendChild(history); card.appendChild(head);
  card.appendChild(composer); card.appendChild(save); card.appendChild(note);
  function show(data) {
    clear(history);
    (data.feedback || []).filter(function (row) {
      return row.target_kind === node.kind && JSON.stringify(row.span_ids) === JSON.stringify(ids);
    }).forEach(function (row) {
      history.appendChild(renderFeedbackHistoryRow(row));
    });
    /* Read-only evidence is not read-only feedback. When the server answers
       `annotated`, the comment is recorded in the workflow's live database,
       keyed by the archive's digest (fix-9eg.19.1), and the evidence file is
       not touched, so the composer stays open on a sealed store. */
    var recordable = !data.read_only || !!data.annotated;
    composer.hidden = save.hidden = head.hidden = !recordable;
    save.disabled = !recordable;
    note.textContent = !recordable
      ? "Read-only snapshot. Add feedback in the working experiment database."
      : (data.read_only
        ? "This evidence is read-only. Your comment is recorded beside it and never changes it."
        : "Each save adds a timestamped comment.");
  }
  api(readPath).then(show).catch(function (e) { note.textContent = e.message; });
  save.addEventListener("click", function () {
    if (!area.value.trim()) { note.textContent = "Enter feedback before saving."; return; }
    var body = {
      target_kind: node.kind, span_ids: ids, target_label: node.title,
      provenance: "human", category: selected.category,
      subcategory: selected.subcategory, comment: area.value
    };
    save.disabled = true; note.textContent = "Saving\u2026";
    apiPost(writePath, body)
      .then(function (data) {
        area.value = "";
        show(data); note.textContent = "Saved";
      })
      .catch(function (e) { save.disabled = false; note.textContent = e.message; });
  });
}

function renderLevel() {
  var d = document.getElementById("detail");
  clear(d);
  var node = state.path[state.path.length - 1];
  if (!node) {
    d.appendChild(el("div", "empty", "Select a turn to inspect it."));
    return;
  }
  d.appendChild(renderCrumbs());

  var info = el("div", "card");
  var head = el("div", "levelHead");
  /* The heading carries the qualifier (tool name, step count) that the parent
     chart shows as muted secondary text, so arriving at a level reads the
     same as the row you clicked to get here. */
  var headingText = node.title + (node.detail ? " · " + node.detail : "");
  if (node.kind === "turn") {
    /* A turn's title is the whole user message, which is printed in full a few
       lines below; the heading only has to identify the turn, so it stays on
       one line and keeps the full text as its tooltip. */
    var turnHead = el("h2", "oneLine", headSnippet(headingText));
    turnHead.title = headingText;
    head.appendChild(turnHead);
  } else {
    head.appendChild(el("h2", null, headingText));
  }
  if (node.cut) {
    head.appendChild(tokenLimitChip(
      node.span && node.span.name === "fw.llm.call" ? undefined : node.cut));
  }
  if (!node.extent.open && node.extent.end > node.extent.start) {
    head.appendChild(el("span", "levelDur", fmtCost(
      fmtNs(node.extent.end - node.extent.start), node.tokens, node.cost)));
  } else if (node.tokens.calls || node.cost.calls) {
    head.appendChild(el("span", "levelDur", fmtCost("", node.tokens, node.cost)));
  }
  /* How the LLM calls under this level were answered, beside what they cost
     (fix-9eg.6). Separate from fmtCost so the cache state can be absent
     without shifting the latency and money it sits next to. */
  appendCacheChip(head, node.cache);
  info.appendChild(head);
  node.render(info);
  d.appendChild(info);
  if (node.kind === "step") { renderStepArtifacts(d, node); }

  var recordHome = node === turnRecordHome();
  /* Record home only (see turnRecordHome): the whole recorded turn, expanded,
     per the design — other levels are about their own slice and should not
     repeat it. */
  if (recordHome && state.turn.diagnosis) {
    var diagCard = el("div", "card");
    diagCard.appendChild(el("h2", null, "What was recorded"));
    renderTurnDiagnosis(diagCard, state.turn, openSpanInTree);
    d.appendChild(diagCard);
  }
  renderFeedback(d, node);

  if (node.children.length) {
    var wf = el("div", "card");
    wf.appendChild(el("h2", null, "Inside this " + levelNoun(node)));
    renderWaterfall(wf, node);
    d.appendChild(wf);
  }

  if (recordHome) {
    var ledgerCard = ledgerDisclosure();
    renderExecutionLedger(ledgerCard, state.turn, openSpanInTree);
    d.appendChild(ledgerCard);
  }

  if (node.kind === "turn") {
    var arts = el("div", "card");
    arts.appendChild(el("h2", null, "Artifacts"));
    renderArtifacts(arts, state.turn);
    d.appendChild(arts);
  }
}

function headSnippet(text, max) {
  /* Collapse the message to one flowing line and cut it on a word boundary so
     the heading never wraps even before the CSS ellipsis kicks in. */
  var limit = max || 80;
  var flat = String(text).replace(/\s+/g, " ").trim();
  if (flat.length <= limit) { return flat; }
  var cut = flat.slice(0, limit).replace(/\s+\S*$/, "");
  return (cut || flat.slice(0, limit)) + "…";
}

function levelNoun(node) {
  if (node.kind === "turn") { return "turn"; }
  if (node.kind === "phase") { return "stage"; }
  if (node.kind === "step") { return "step"; }
  return "span";
}

/* The page that carries the turn-wide record (what was recorded, the
   execution ledger): the first Execution stage, since that is where the
   dispatches it lists ran. A turn without one -- planning only, a failed
   turn, a direct command -- keeps them on the turn page. */
function turnRecordHome() {
  var root = state.path[0];
  return root.children.filter(function (child) {
    return child.kind === "phase" && child.phase === PHASE_EXECUTION;
  })[0] || root;
}

function ledgerDisclosure() {
  var box = el("details", "card ledgerDisclosure");
  box.appendChild(el("summary", null, "Execution ledger"));
  return box;
}

function renderDetail(turn, spans) {
  state.turn = turn;
  state.path = [buildTurnTree(turn, spans)];
  attachTraceHierarchy();
  if (!spans.length) {
    var d = document.getElementById("detail");
    clear(d);
    d.appendChild(renderCrumbs());
    var info = el("div", "card");
    info.appendChild(el("h2", null, policedText(turn.user_message) || "(no message)"));
    renderTurnLevel(info, turn);
    d.appendChild(info);
    renderFeedback(d, state.path[0]);
    var note = el("div", "card");
    note.appendChild(el("div", "empty", "No spans recorded for this turn."));
    d.appendChild(note);
    /* The record's own ledger still stands without spans: rows without a
       span say so. */
    var ledgerCard = ledgerDisclosure();
    renderExecutionLedger(ledgerCard, turn, null);
    d.appendChild(ledgerCard);
    return;
  }
  renderLevel();
}

function spanCategory(span) {
  if (span.kind === "human_wait" || span.name === "fw.ask_user") { return "cat-human"; }
  if (span.name === "fw.turn") { return "cat-turn"; }
  if (span.name.indexOf("fw.planner.") === 0) { return "cat-planner"; }
  if (span.name === "fw.agent.tool_call") { return "cat-tool"; }
  if (span.name === "fw.agent.execute" || span.name === "fw.agent.step") { return "cat-tool"; }
  if (span.name === "fw.command.execute") { return "cat-command"; }
  if (span.name === "fw.llm.call") { return "cat-llm"; }
  return "cat-other";
}

function parsedAttr(value) {
  if (typeof value !== "string") { return value; }
  try { return JSON.parse(value); }
  catch (e) { return value; }
}

/* -- what each fw.llm.call was asked to do ----------------------------- */
/* The DSPy module chain alone cannot name the call: the planner and the
   conversation summarizer are both "ChainOfThought > Predict", and ReAct's
   two modules build their signatures dynamically, so DSPy names them both
   StringSignature. Identification therefore goes structural first (the
   enclosing span already says "this happened during planning"), then the
   module chain, then the signature's output fields — which are what the call
   was actually asked to produce and are distinct across every runtime role. */
var LLM_ROLE_BY_PARENT = {
  "fw.planner.plan": "planning",
  "fw.planner.replan": "replanning",
  "fw.nlu.param_extraction": "parameter extraction",
  "fw.nlu.intent": "intent clarification"
};
var LLM_ROLE_BY_OUTPUT_FIELD = [
  ["conversation_summary", "conversation summary"],
  ["clarified_command", "intent clarification"],
  ["clarification_question", "intent clarification"],
  ["next_steps", "planning"],
  ["next_thought", "agent step"],
  ["final_answer", "final answer"]
];
function llmCallLabel(span, byId) {
  var parent = byId ? byId[span.parent_span_id] : null;
  if (parent && LLM_ROLE_BY_PARENT[parent.name]) {
    return LLM_ROLE_BY_PARENT[parent.name];
  }
  var attrs = span.attributes || {};
  var chain = String(attrs.module_chain || attrs.module || "");
  if (chain.indexOf("fastWorkflowReAct") === 0) {
    /* ReAct runs two different modules over the same trajectory: a Predict
       that picks the next step, and a ChainOfThought that turns the finished
       trajectory into the answer once the loop has stopped. */
    return chain.indexOf("ChainOfThought") === -1 ? "agent step" : "final answer";
  }
  if (chain.indexOf("ParamExtractor") !== -1) { return "parameter extraction"; }
  var fields = parsedAttr(attrs.module_output);
  if (!fields || typeof fields !== "object") { return ""; }
  if ("topic" in fields && "summary" in fields) { return "conversation topic"; }
  for (var i = 0; i < LLM_ROLE_BY_OUTPUT_FIELD.length; i++) {
    if (LLM_ROLE_BY_OUTPUT_FIELD[i][0] in fields) { return LLM_ROLE_BY_OUTPUT_FIELD[i][1]; }
  }
  return "";
}

var ATTR_LAZY_THRESHOLD = 20 * 1024;
function estimateSerializedSize(value) {
  if (value === null || value === undefined) { return 0; }
  if (typeof value === "string") { return value.length; }
  try { return JSON.stringify(value).length; }
  catch (error) { return ATTR_LAZY_THRESHOLD + 1; }
}
function formatByteSize(n) {
  if (n >= 1048576) { return (n / 1048576).toFixed(1) + " MB"; }
  if (n >= 1024) { return Math.round(n / 1024) + " KB"; }
  return n + " B";
}
function appendAttrSection(container, label, value) {
  if (value === null || value === undefined || value === "") { return; }
  var parsed = parsedAttr(value);
  var size = estimateSerializedSize(parsed);
  if (size > ATTR_LAZY_THRESHOLD) {
    var det = el("details");
    det.appendChild(el("summary", null,
      label + " (" + formatByteSize(size) + " — expand to load)"));
    det.addEventListener("toggle", function () {
      if (!det.open || det.dataset.loaded) { return; }
      det.dataset.loaded = "1";
      det.appendChild(el("pre", "json", pretty(parsed)));
    });
    container.appendChild(det);
    return;
  }
  var block = el("div", "msgBlock");
  block.appendChild(el("span", "lbl", label));
  block.appendChild(document.createTextNode(pretty(parsed)));
  container.appendChild(block);
}

function appendCollapsedSection(container, label, value) {
  /* Detail that is worth keeping but almost never what you came for — folded
     away next to the raw attributes rather than pushing the answer offscreen. */
  if (value === null || value === undefined || value === "") { return; }
  var parsed = parsedAttr(value);
  var size = estimateSerializedSize(parsed);
  var det = el("details");
  if (size > ATTR_LAZY_THRESHOLD) {
    det.appendChild(el("summary", null,
      label + " (" + formatByteSize(size) + " — expand to load)"));
    det.addEventListener("toggle", function () {
      if (!det.open || det.dataset.loaded) { return; }
      det.dataset.loaded = "1";
      det.appendChild(el("pre", "json", pretty(parsed)));
    });
  } else {
    det.appendChild(el("summary", null, label));
    det.appendChild(el("pre", "json", pretty(parsed)));
  }
  container.appendChild(det);
}

function renderWaterfall(container, node) {
  /* One row per DIRECT child, laid out against this level's own time window,
     so each level reads as "what happened inside this thing" rather than as
     the whole turn re-drawn. A row is the drill-down control. */
  var t0 = node.extent.start, t1 = node.extent.end, hasOpen = node.extent.open;
  node.children.forEach(function (child) {
    t0 = Math.min(t0, child.extent.start);
    t1 = Math.max(t1, child.extent.end);
    if (child.extent.open) { hasOpen = true; }
  });
  if (t1 <= t0) { t1 = t0 + 1; }
  var total = t1 - t0;

  node.children.forEach(function (child) {
    var rowEl = el("div", "wfRow");
    rowEl.setAttribute("aria-label", "Inspect " + child.title + (child.detail ? " · " + child.detail : ""));
    makeRowActivatable(rowEl, function () { descend(child); });
    var name = el("div", "wfName", child.title + " ");
    if (child.detail) { name.appendChild(el("span", "cmd", "· " + child.detail)); }
    if (child.status === "error") { name.appendChild(el("span", "cmd", " · ERROR")); }
    if (child.cut) {
      name.appendChild(tokenLimitChip(
        child.span && child.span.name === "fw.llm.call" ? undefined : child.cut));
    }
    if (child.children.length) {
      name.appendChild(el("span", "wfGo", "›"));
    }
    rowEl.appendChild(name);

    var track = el("div", "wfTrack");
    var bar = el("div", "wfBar " + child.category);
    var open = child.extent.open;
    var end = open ? t1 : child.extent.end;
    var left = ((child.extent.start - t0) / total) * 100;
    var width = Math.max(((end - child.extent.start) / total) * 100, 0.4);
    bar.style.left = left + "%";
    bar.style.width = Math.min(width, 100 - left) + "%";
    if (open) { bar.classList.add("open"); bar.title = "still open (in progress)"; }
    track.appendChild(bar);
    rowEl.appendChild(track);
    rowEl.appendChild(el("div", "wfDur", open
      ? fmtCost("open…", child.tokens, child.cost)
      : fmtCost(fmtNs(child.extent.end - child.extent.start), child.tokens, child.cost)));

    container.appendChild(rowEl);
  });

  var legend = el("div", "legend");
  [["cat-turn", "turn"], ["cat-planner", "planner"], ["cat-tool", "tool call"],
   ["cat-command", "command execute"], ["cat-llm", "LLM call"],
   ["cat-human", "human wait (hatched)"],
   ["cat-other", "other"]].forEach(function (pair) {
    var item = el("span");
    item.appendChild(el("span", "chip " + pair[0]));
    item.appendChild(document.createTextNode(pair[1]));
    legend.appendChild(item);
  });
  if (hasOpen) {
    legend.appendChild(el("span", null, "dashed edge = span still open (in progress)"));
  }
  container.appendChild(legend);
}

/* -- artifacts [R22] --------------------------------------------------- */
function looksHtml(text) {
  return /^\s*(<!doctype|<html|<svg|<body|<head)/i.test(text);
}
function sandboxedFrame(htmlText) {
  var frame = document.createElement("iframe");
  frame.setAttribute("sandbox", "");   /* max restrictions: no scripts, no forms, no origin */
  frame.srcdoc =
    "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'\">" +
    htmlText;
  return frame;
}
function artifactNode(key, value, meta) {
  var box = el("div", "artifact");
  var head = el("div", "aHead");
  head.appendChild(el("span", "aKey", key));
  if (meta) { head.appendChild(el("span", "aMeta", meta)); }
  box.appendChild(head);
  if (typeof value === "string" && looksHtml(value)) {
    /* HTML-ish: sandboxed iframe only, plus safe source view */
    box.appendChild(sandboxedFrame(value));
    var src = el("details");
    src.appendChild(el("summary", null, "View source (as text)"));
    src.appendChild(el("pre", "json", value));
    box.appendChild(src);
  } else if (typeof value === "string") {
    box.appendChild(el("pre", "json", value));       /* textContent — safe */
  } else {
    box.appendChild(el("pre", "json", pretty(value)));
  }
  return box;
}

function turnArtifacts(turn, callIds) {
  var record = turn.record || {};
  return tmArtifactsIn((record.turn_output || {}).command_outputs || [], callIds);
}

function renderArtifacts(container, turn, callIds) {
  /* The chat's link and side panel, owned by this card and kept inside the
     detail pane. */
  var viewer = tmArtifactViewer(turnArtifacts(turn, callIds), null, {
    owner: container,
    label: callIds ? "Artifacts of this step" : "Artifacts of this turn",
    placement: "below",
    anchor: function () { return viewer.link; },
    frame: function () { return document.getElementById("detail"); }
  });
  if (!viewer) {
    container.appendChild(el("div", "empty", "No artifacts on this turn."));
    return;
  }
  container.appendChild(viewer.linkRow);
  container.appendChild(viewer.panel);
}

var STEP_DISPATCH_SPANS = { "fw.agent.tool_call": true, "fw.command.execute": true };
function stepCallIds(node, turn) {
  /* The dispatches this step made, by the command_call_id its tool-call and
     execute spans carry, plus any execution_records ref of the turn record
     whose span_id is one of this step's spans. Both are the id the command's
     CommandOutput carries; nothing is matched by name, order or time. */
  var ids = {}, spanIds = {};
  (function walk(n) {
    if (n.span) {
      spanIds[n.span.span_id] = true;
      var callId = (n.span.attributes || {}).command_call_id;
      if (STEP_DISPATCH_SPANS[n.span.name] === true && typeof callId === "string" && callId) {
        ids[callId] = true;
      }
    }
    n.children.forEach(walk);
  })(node);
  ((turn.record || {}).execution_records || []).forEach(function (ref) {
    if (ref && typeof ref.command_call_id === "string" && ref.command_call_id
        && ref.span_id && spanIds[ref.span_id] === true) {
      ids[ref.command_call_id] = true;
    }
  });
  return ids;
}

function renderStepArtifacts(parent, node) {
  /* Only a step whose own dispatches returned artifacts gets the card. */
  var callIds = stepCallIds(node, state.turn);
  if (!turnArtifacts(state.turn, callIds).length) { return; }
  var card = el("div", "card");
  card.appendChild(el("h2", null, "Artifacts"));
  renderArtifacts(card, state.turn, callIds);
  parent.appendChild(card);
}

