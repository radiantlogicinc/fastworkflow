/* -- turn detail ------------------------------------------------------- */
/* `spanId` and `note` are optional and used by `openPairSpan` only: the span is
   focused inside the stale guard below, after the trace it belongs to has been
   rendered, so a focus request can never be applied to another run's trace.
   Every existing caller opens the turn and passes neither. */
function selectTurn(turnKey, spanId, note) {
  var nav = expNavToken();   // this view now owns #detail
  state.turnKey = turnKey;
  state.turn = null;
  state.path = [];
  alignHierarchyTurn(turnKey);
  if (turnLoadAbort) { turnLoadAbort.abort(); }
  turnLoadAbort = (typeof AbortController !== "undefined") ? new AbortController() : null;
  var signal = turnLoadAbort && turnLoadAbort.signal;
  Promise.all([
    api("/api/turn/" + encodeURIComponent(turnKey), { signal: signal }),
    api("/api/spans/" + encodeURIComponent(turnKey), { signal: signal })
  ]).then(function (results) {
    if (expNavStale(nav)) { return; }
    renderDetail(results[0].turn, results[1].spans || []);
    focusLoadedSpan(spanId, note);
  }).catch(function (e) {
    if (requestWasAborted(e) || expNavStale(nav)) { return; }
    var d = document.getElementById("detail");
    clear(d);
    d.appendChild(el("div", "empty", "Failed to load turn: " + e.message));
  });
}

function statusBadge(turn) {
  if (review.progress && review.progress.assignment.blinded) {
    return el("span", "badge", "blinded review trace");
  }
  if (turn.status === "awaiting_user") { return el("span", "badge progress", "awaiting_user — in progress"); }
  if (turn.status === "completed" && turn.success) { return el("span", "badge ok", "completed · success"); }
  return el("span", "badge " + (turn.success ? "ok" : "fail"),
    turn.status + (turn.success ? " · success" : " · failure"));
}

/* ======================================================================
   Progressive-disclosure turn viewer
   ======================================================================
   The levels are fastWorkflow's own abstractions, not the raw span table: a
   turn plans and then executes; execution is an agent taking steps; a step
   hands a command to the assistant, which detects intent and then extracts
   parameters. Each level shows what is known AT that level, then a waterfall
   of its direct children as the drill-down control, and breadcrumbs to climb.

   Most of the tree is real — the store persists parent_span_id and the old
   viewer simply ignored it. Two levels are not spans yet: the agent loop as
   a phase, and the individual step. Those are derived in
   deriveExecutionChildren(), which is the single seam: when fw.agent.step is
   emitted for real, that function reads spans instead of inferring them and
   nothing else here changes. */

var PHASE_PLANNING = "planning";
var PHASE_EXECUTION = "execution";
var PHASE_OTHER = "other";

function byStartNs(a, b) { return a.start_ns - b.start_ns; }

function spanExtent(spans) {
  /* Union of a node's own spans; an unclosed span contributes only its start
     so an in-progress turn still lays out instead of collapsing to zero. */
  var t0 = Infinity, t1 = -Infinity, open = false;
  spans.forEach(function (s) {
    t0 = Math.min(t0, s.start_ns);
    if (s.end_ns === null || s.end_ns === undefined) {
      open = true;
      t1 = Math.max(t1, s.start_ns);
    } else {
      t1 = Math.max(t1, s.end_ns);
    }
  });
  if (!isFinite(t0)) { return { start: 0, end: 1, open: false }; }
  return { start: t0, end: Math.max(t1, t0), open: open };
}

function mergeExtents(list) {
  var t0 = Infinity, t1 = -Infinity, open = false;
  list.forEach(function (x) {
    t0 = Math.min(t0, x.start);
    t1 = Math.max(t1, x.end);
    if (x.open) { open = true; }
  });
  if (!isFinite(t0)) { return { start: 0, end: 1, open: false }; }
  return { start: t0, end: t1, open: open };
}

function noTokens() {
  return {
    in: 0, out: 0, total: 0, calls: 0, counted: 0, partial: 0, unrecorded: 0,
    /* Calls that quote a provider response an earlier call is already charged
       for. They are real calls, so they stay in `calls`, but their tokens and
       money are not added a second time. */
    shared: 0
  };
}

function countOf(value) {
  return typeof value === "number" && isFinite(value) ? value : 0;
}

function spanTokens(span) {
  /* Only fw.llm.call carries usage, and only when the server kept a DSPy
     history entry to read it from (see server_memory inspect mode). Turns
     recorded without one contribute nothing rather than a wrong zero.

     The three counts beside the sums are what let a reader tell a level whose
     calls all reported their usage from one where some did not: a call with no
     usage attribute is `unrecorded`, one that reported part of the split is
     `partial`, and the sums only ever add what was actually recorded. */
  if (!span || span.name !== "fw.llm.call") { return noTokens(); }
  var usage = parsedAttr((span.attributes || {}).usage);
  var counts = noTokens();
  counts.calls = 1;
  if (!usage || typeof usage !== "object") {
    counts.unrecorded = 1;
    return counts;
  }
  var recorded = ["prompt_tokens", "completion_tokens", "total_tokens"]
    .filter(function (key) { return exactInt(usage[key]) !== null; });
  if (!recorded.length) {
    counts.unrecorded = 1;
    return counts;
  }
  counts.in = countOf(usage.prompt_tokens);
  counts.out = countOf(usage.completion_tokens);
  counts.total = countOf(usage.total_tokens);
  if (recorded.length === 3) { counts.counted = 1; } else { counts.partial = 1; }
  return counts;
}

function addTokens(a, b) {
  return {
    in: a.in + b.in,
    out: a.out + b.out,
    total: a.total + b.total,
    calls: a.calls + b.calls,
    counted: a.counted + b.counted,
    partial: a.partial + b.partial,
    unrecorded: a.unrecorded + b.unrecorded,
    shared: countOf(a.shared) + countOf(b.shared)
  };
}

function sumTokens(nodes) {
  return nodes.reduce(function (total, node) {
    return addTokens(total, node.tokens);
  }, noTokens());
}

function spanResponseId(span) {
  /* DSPy's id for the provider response a call's usage and cost were copied
     from. Empty when the span is not an LLM call or recorded no history entry,
     in which case nothing can be recognised as a duplicate of it. */
  if (!span || span.name !== "fw.llm.call") { return ""; }
  var value = (span.attributes || {}).history_uuid;
  return typeof value === "string" ? value : "";
}

function sumResponses(nodes) {
  return nodes.reduce(function (all, node) {
    return all.concat(node.responses || []);
  }, []);
}

function sharedTokens() {
  /* A call that quoted a response already charged: still a call, charged
     nothing. Counting its tokens again would report twice what was spent. */
  var counts = noTokens();
  counts.calls = 1;
  counts.shared = 1;
  return counts;
}

function sharedCost() {
  return { calls: 1, recorded: 0, unrecorded: 1, total: null };
}

function spanOrderKey(span) {
  /* The same tie-break the server's `_order_key` uses, so the call credited
     with a shared response is the same one on both sides. */
  var start = exactInt((span || {}).start_ns);
  return [start === null ? 1 : 0, start === null ? 0 : start,
          String((span || {}).span_id || "")];
}

function compareOrderKeys(a, b) {
  for (var i = 0; i < a.length; i += 1) {
    if (a[i] < b[i]) { return -1; }
    if (a[i] > b[i]) { return 1; }
  }
  return 0;
}

var _chargeFoldCache = null;
var _chargeFoldSource = null;
function chargeFolds(byId) {
  /* Which spans must NOT charge their own tokens, money or cache, and why.

     Decided from the whole span set rather than from a node's position in the
     tree, because the duplicate is not always an ancestor. Two shapes occur and
     the server distinguishes them, so this does too:

       "wrapper" — an fw.llm.call with a DESCENDANT quoting the same
       history_uuid. One provider response recorded at two levels is one call,
       and the inner record is the one closest to the provider, so the outer
       stops being a call at all.

       "shared" — two calls quoting one response WITHOUT being nested. Which of
       them really spent the tokens is not something this layer can decide, so
       both stay visible as calls and the earlier one is credited. Folding these
       only when they happened to be nested was the gap: sibling records charged
       twice on screen while the server charged once (shared-response-cost).

     Memoised on the span map identity because a trace is walked many times per
     render and the answer cannot change between walks of the same map. */
  if (_chargeFoldSource === byId && _chargeFoldCache) { return _chargeFoldCache; }
  var spans = Object.keys(byId || {}).map(function (id) { return byId[id]; });
  var folds = {};
  var groups = {};
  spans.forEach(function (span) {
    var response = spanResponseId(span);
    if (response === "") { return; }
    (groups[response] = groups[response] || []).push(span);
    /* Mark every ancestor of this call that quotes the same response. */
    var seen = {}, parent = byId[span.parent_span_id];
    while (parent && !seen[parent.span_id]) {
      seen[parent.span_id] = true;
      if (spanResponseId(parent) === response) { folds[parent.span_id] = "wrapper"; }
      parent = byId[parent.parent_span_id];
    }
  });
  Object.keys(groups).forEach(function (response) {
    var quoting = groups[response].filter(function (span) {
      return folds[span.span_id] !== "wrapper";
    });
    if (quoting.length < 2) { return; }
    quoting.sort(function (a, b) {
      return compareOrderKeys(spanOrderKey(a), spanOrderKey(b));
    });
    quoting.slice(1).forEach(function (span) { folds[span.span_id] = "shared"; });
  });
  _chargeFoldSource = byId;
  _chargeFoldCache = folds;
  return folds;
}

var _nodeSeq = 0;
function makeNode(kind, title, opts) {
  opts = opts || {};
  _nodeSeq += 1;
  return {
    id: kind + "-" + _nodeSeq,
    kind: kind,
    title: title,
    crumb: opts.crumb || title,
    detail: opts.detail || "",
    category: opts.category || "cat-other",
    status: opts.status || "",
    span: opts.span || null,
    children: opts.children || [],
    extent: opts.extent || { start: 0, end: 1, open: false },
    tokens: opts.tokens || noTokens(),
    cost: opts.cost || noCost(),
    cache: opts.cache || sumCache(opts.children || []),
    /* The provider responses (fw.llm.call history_uuid) already charged inside
       this level. A wrapper LM that invokes another LM records a second
       fw.llm.call for the SAME response, and charging both would report twice
       the calls, tokens and money that were spent — the same fold the server's
       usage_rollup applies, kept in step with it here. */
    responses: opts.responses || sumResponses(opts.children || []),
    /* calls cut at the token limit inside this level, own span included */
    cut: opts.cut !== undefined ? opts.cut : sumCut(opts.children || []),
    render: opts.render || function () {}
  };
}

/* -- span-level titles -------------------------------------------------- */
function spanTitle(span, role) {
  switch (span.name) {
    case "fw.planner.plan": return "Planning";
    case "fw.planner.replan": return "Replanning";
    case "fw.agent.execute": return "Execution";
    case "fw.agent.step": return "Step";
    case "fw.agent.tool_call":
      return "Agent tool call" + (span.command_name ? " · " + policedText(span.command_name) : "");
    case "fw.command.execute":
      return "Assistant" + (span.command_name ? " · " + policedText(span.command_name) : "");
    case "fw.nlu.intent": return "Intent detection";
    case "fw.nlu.param_extraction": return "Parameter extraction";
    case "fw.ask_user": return "Ask user";
    case "fw.llm.call": return "LLM call" + (role ? " · " + role : "");
    default: return policedText(span.name);
  }
}

function orderedAttrKeys(attrs) {
  /* Input immediately followed by output, wherever both are present. */
  var keys = Object.keys(attrs);
  if (keys.indexOf("module_input") === -1 || keys.indexOf("module_output") === -1) {
    return keys;
  }
  return ["module_input", "module_output"].concat(keys.filter(function (key) {
    return key !== "module_input" && key !== "module_output";
  }));
}

/* -- per-span level content -------------------------------------------- */
function renderSpanLevel(container, span, role, fold) {
  var kv = el("dl", "kv");
  function row(k, v) {
    if (v === null || v === undefined || v === "") { return; }
    kv.appendChild(el("dt", null, k));
    kv.appendChild(el("dd", null, v));
  }
  row("span", policedText(span.name) + (role ? " · " + role : ""));
  row("span_id", span.span_id);
  row("kind", span.kind);
  row("status", span.status);
  if (span.command_name) { row("command", policedText(span.command_name)); }
  if (span.context) { row("context", policedText(span.context)); }
  if (span.end_ns) { row("duration", fmtNs(span.end_ns - span.start_ns)); }
  if (llmCallCutAtLimit(span)) {
    var cutAt = parsedAttr((span.attributes || {}).call_kwargs).max_tokens;
    row("token limit", "cut at limit: completion_tokens = max_tokens = "
      + fmtCount(cutAt) + " — the output stopped at the cap, not at an end");
  }
  /* The recorded cache state of THIS call, said plainly (fix-9eg.6). A hit
     explains where the answer came from; it is not a fault, and it does not
     make the answer stale. An unrecorded flag stays unrecorded: the DSPy
     history entry the flag is read from is absent in some processes, and
     calling that a miss would invent a provider round trip. */
  var cacheState = cacheStateOf(span);
  if (cacheState === "hit") {
    row("LLM cache", "cache hit — this answer was served from the LLM cache "
      + "rather than by calling the provider. That is how it arrived, not a "
      + "judgement about it.");
  } else if (cacheState === "miss") {
    row("LLM cache", "cache miss — the provider was called for this answer.");
  } else if (cacheState === "unknown") {
    row("LLM cache", "not recorded — this call recorded no cache flag, so "
      + "whether it was served from the cache is unknown.");
  }
  /* Why a call on screen may contribute nothing to the totals above it. Said
     here rather than only in the roll-up, so the reader looking at the call
     itself is not left to wonder where its tokens went. */
  if (fold === "wrapper") {
    row("counted once", "this call wraps an inner call that recorded the same "
      + "provider response, so the inner record is the one counted. This level "
      + "adds no tokens, money or cache state of its own.");
  } else if (fold === "shared") {
    row("counted once", "another call already accounts for this provider "
      + "response, so its tokens and cost are counted there rather than twice. "
      + "This is still a real call and is still counted as one.");
  }
  container.appendChild(kv);

  var a = span.attributes || {};
  if (span.name === "fw.llm.call") {
    appendAttrSection(container, "module",
      (a.module_chain || a.module || "DSPy") + " · " +
      (a.model || a.response_model || "unknown model") +
      " · captured via " +
      (a.capture_source === "module_decorator" ? "module logging" : "the DSPy API"));
    /* Input then output, adjacent: the pair is the question a reader has, and
       anything that separates them (the wire messages especially) makes them
       scroll apart. Everything below is corroborating detail. */
    appendAttrSection(container, "module input", a.module_input);
    appendAttrSection(container, "module output (parsed)", a.module_output);
    appendAttrSection(container, "reasoning", a.reasoning);
    appendAttrSection(container, "LLM input (messages)", a.messages);
    appendAttrSection(container, "LLM input (prompt)", a.prompt);
    appendAttrSection(container, "LLM output (raw)", a.output);
    appendAttrSection(container, "provider response", a.provider_response);
    appendAttrSection(container, "usage", a.usage);
    appendAttrSection(container, "usage", a.usage_capture);
    appendAttrSection(container, "module exception", a.module_exception);
    appendAttrSection(container, "exception", a.exception);
    appendCollapsedSection(container, "call kwargs", a.call_kwargs);
  } else if (span.name === "fw.agent.tool_call" || span.name === "fw.command.execute") {
    appendAttrSection(container, "command as the agent wrote it", a.raw_command);
    appendAttrSection(container, "extracted parameters", a.parameters);
    appendAttrSection(container, "response", a.response_text);
    appendAttrSection(container, "error type", a.error_type);
  } else if (span.name === "fw.ask_user") {
    appendAttrSection(container, "question to the user", a.agent_query);
    appendAttrSection(container, "the user's answer", a.user_response);
  } else {
    /* NLU and anything else: every attribute, one per row, since these spans
       are small and their whole point is the detail. */
    orderedAttrKeys(a).forEach(function (key) {
      appendAttrSection(container, key.replace(/_/g, " "), a[key]);
    });
  }

  var rawDetails = el("details");
  var rawSize = estimateSerializedSize(a);
  if (rawSize > ATTR_LAZY_THRESHOLD) {
    rawDetails.appendChild(el("summary", null,
      "Raw span attributes (" + formatByteSize(rawSize) + " — expand to load)"));
    rawDetails.addEventListener("toggle", function () {
      if (!rawDetails.open || rawDetails.dataset.loaded) { return; }
      rawDetails.dataset.loaded = "1";
      rawDetails.appendChild(el("pre", "json", pretty(a)));
    });
  } else {
    rawDetails.appendChild(el("summary", null, "Raw span attributes"));
    rawDetails.appendChild(el("pre", "json", pretty(a)));
  }
  container.appendChild(rawDetails);
}

function spanNode(span, kids, byId) {
  var role = span.name === "fw.llm.call" ? llmCallLabel(span, byId) : "";
  var childSpans = (kids[span.span_id] || []).slice().sort(byStartNs);
  var children = childSpans.map(function (c) { return spanNode(c, kids, byId); });
  /* Whether this span charges its own usage, money and cache, and if not, why.
     `chargeFolds` reads the whole span set, so a duplicate is caught whether it
     nests (wrapper) or sits beside its twin (shared) — the sibling case used to
     charge twice here while the server charged once.

     A WRAPPER stops being a call: the inner record is the one closest to the
     provider. A SHARED call stays a call and is charged nothing, because which
     of two calls quoting one response actually spent the tokens is not decidable
     from the trace. Either way the level still renders, so the duplication stays
     visible to a reader who opens it. */
  var fold = chargeFolds(byId)[span.span_id] || "";
  var charged = sumResponses(children);
  var response = spanResponseId(span);
  var own = { tokens: spanTokens(span), cost: spanCost(span), cache: spanCache(span) };
  if (fold === "wrapper") {
    own = { tokens: noTokens(), cost: noCost(), cache: noCache() };
  } else if (fold === "shared") {
    own = { tokens: sharedTokens(), cost: sharedCost(), cache: spanCache(span) };
  }
  return makeNode("span", spanTitle(span, role), {
    crumb: spanTitle(span, role),
    detail: span.status === "error" ? "error" : "",
    category: spanCategory(span),
    status: span.status,
    span: span,
    children: children,
    extent: spanExtent([span]),
    tokens: addTokens(own.tokens, sumTokens(children)),
    cost: addCost(own.cost, sumCost(children)),
    cache: addCache(own.cache, sumCache(children)),
    responses: fold === "wrapper" || response === ""
      ? charged
      : charged.concat([response]),
    cut: (llmCallCutAtLimit(span) ? 1 : 0) + sumCut(children),
    render: function (container) { renderSpanLevel(container, span, role, fold); }
  });
}

/* -- phase classification ---------------------------------------------- */
function reactChain(span) {
  if (span.name !== "fw.llm.call") { return ""; }
  var chain = String((span.attributes || {}).module_chain || "");
  return chain.indexOf("fastWorkflowReAct") === 0 ? chain : "";
}

function phaseOf(span) {
  if (span.name === "fw.planner.plan" || span.name === "fw.planner.replan") {
    return PHASE_PLANNING;
  }
  if (span.name === "fw.agent.execute" || span.name === "fw.agent.step" ||
      span.name === "fw.agent.tool_call" || span.name === "fw.command.execute" ||
      span.name === "fw.ask_user" || reactChain(span)) {
    return PHASE_EXECUTION;
  }
  return PHASE_OTHER;
}

/* -- recorded agent loop: fw.agent.execute / fw.agent.step -------------- */
function buildRecordedStepNode(span, ordinal, incomingObservation, kids, byId) {
  var attrs = span.attributes || {};
  var children = (kids[span.span_id] || []).slice().sort(byStartNs)
    .map(function (child) { return spanNode(child, kids, byId); });
  return makeNode("step", "Step " + ordinal, {
    crumb: "Step " + ordinal,
    detail: attrs.tool_name || "",
    category: "cat-tool",
    status: span.status,
    span: span,
    children: children,
    extent: spanExtent([span]),
    tokens: sumTokens(children),
    cost: sumCost(children),
    render: function (container) {
      container.appendChild(el("div", "levelLead",
        "The agent reads the observation from its trajectory, states its " +
        "reasoning, and names the next tool to call."));
      appendAttrSection(container, "observation in",
        incomingObservation || "(none — this is the agent's first step)");
      appendAttrSection(container, "reasoning", attrs.thought);
      appendAttrSection(container, "next tool", attrs.tool_name);
      appendAttrSection(container, "tool arguments", attrs.tool_args);
      appendAttrSection(container, "observation out", attrs.observation);
      appendAttrSection(container, "question asked of the user", attrs.clarification);
      appendAttrSection(container, "tool error", attrs.tool_error);
    }
  });
}

function buildRecordedExecutionNode(span, kids, byId, ordinalLabel) {
  var attrs = span.attributes || {};
  var childSpans = (kids[span.span_id] || []).slice().sort(byStartNs);
  var children = [], stepCount = 0, lastObservation = "";
  childSpans.forEach(function (child) {
    if (child.name === "fw.agent.step") {
      stepCount += 1;
      children.push(buildRecordedStepNode(
        child, stepCount, lastObservation, kids, byId));
      lastObservation = (child.attributes || {}).observation || "";
      return;
    }
    var node = spanNode(child, kids, byId);
    if (child.name === "fw.llm.call" && llmCallLabel(child, byId) === "final answer") {
      /* ReAct's extract call: it runs inside forward(), after the loop and
         before the agent returns, so it belongs to execution rather than to
         the turn — even though its output becomes the turn's answer. */
      node.title = "Answer synthesis";
      node.crumb = "Answer synthesis";
    }
    children.push(node);
  });
  var label = "Execution" + (ordinalLabel || "") + (attrs.resumed ? " (resumed)" : "");
  return makeNode("phase", label, {
    crumb: label,
    detail: stepCount + (stepCount === 1 ? " step" : " steps"),
    category: "cat-tool",
    status: span.status,
    span: span,
    children: children,
    extent: spanExtent([span]),
    tokens: sumTokens(children),
    cost: sumCost(children),
    render: function (container) {
      container.appendChild(el("div", "levelLead",
        "The agent loop: each step turns an observation into reasoning and a " +
        "next action. Tool calls go to the assistant, which resolves intent " +
        "and parameters before running the command."));
      var kv = el("dl", "kv");
      kv.appendChild(el("dt", null, "steps"));
      kv.appendChild(el("dd", null, String(stepCount)));
      if (attrs.attempts && attrs.attempts > 1) {
        kv.appendChild(el("dt", null, "agent retries"));
        kv.appendChild(el("dd", null, String(attrs.attempts - 1)));
      }
      container.appendChild(kv);
      appendAttrSection(container,
        attrs.resumed ? "the user's answer, resuming the agent"
                      : "input received from the planner",
        attrs.agent_input);
      appendAttrSection(container, "answer returned to the turn", attrs.final_answer);
      appendAttrSection(container, "still awaiting the user", attrs.clarification);
      if (attrs.exhausted) {
        appendAttrSection(container, "note",
          "the loop hit max_iters without the agent choosing finish");
      }
    }
  });
}

/* -- legacy: agent steps derived for turns recorded before fw.agent.step -- */
function deriveExecutionChildren(spans) {
  /* Kept only for turns already in the database. Live turns carry real
     fw.agent.execute / fw.agent.step spans and take the recorded path above;
     this reconstruction runs when those are absent, so old recordings stay
     readable instead of collapsing into a flat list.

     ReAct's own two modules tell the steps apart: the Predict picks the next
     tool, the ChainOfThought turns the finished trajectory into the answer.
     A step therefore opens on a Predict call and owns every tool call until
     the next one. */
  var out = [], step = null, stepNo = 0;
  spans.forEach(function (s) {
    var chain = reactChain(s);
    var isExtract = chain && chain.indexOf("ChainOfThought") !== -1;
    if (chain && !isExtract) {
      stepNo += 1;
      step = { no: stepNo, reasoning: s, actions: [] };
      out.push({ type: "step", step: step });
    } else if (isExtract) {
      step = null;
      out.push({ type: "final", span: s });
    } else if (step) {
      step.actions.push(s);
    } else {
      out.push({ type: "span", span: s });
    }
  });
  return out;
}

function attrObject(value) {
  var parsed = parsedAttr(value);
  return (parsed && typeof parsed === "object") ? parsed : null;
}

function moduleKwargs(span) {
  var input = attrObject((span.attributes || {}).module_input);
  return (input && attrObject(input.kwargs)) || input || null;
}

function buildStepNode(step, incomingObservation, kids, byId) {
  var out = attrObject((step.reasoning.attributes || {}).module_output) || {};
  var toolName = out.next_tool_name || "";
  var children = [spanNode(step.reasoning, kids, byId)].concat(
    step.actions.map(function (s) { return spanNode(s, kids, byId); })
  );
  var covered = [step.reasoning].concat(step.actions);
  return makeNode("step", "Step " + step.no, {
    crumb: "Step " + step.no,
    detail: toolName,
    category: "cat-tool",
    children: children,
    extent: spanExtent(covered),
    tokens: sumTokens(children),
    cost: sumCost(children),
    render: function (container) {
      var lead = el("div", "levelLead",
        "The agent reads the observation from its trajectory, states its " +
        "reasoning, and names the next tool to call.");
      container.appendChild(lead);
      appendAttrSection(container, "observation in",
        incomingObservation || "(none — this is the agent's first step)");
      appendAttrSection(container, "reasoning", out.next_thought);
      appendAttrSection(container, "next tool", toolName);
      appendAttrSection(container, "tool arguments", out.next_tool_args);
      var last = step.actions[step.actions.length - 1];
      if (last) {
        appendAttrSection(container, "observation out",
          (last.attributes || {}).response_text);
      }
    }
  });
}

function buildExecutionNode(spans, kids, byId, ordinalLabel) {
  var recorded = spans.filter(function (s) { return s.name === "fw.agent.execute"; });
  if (recorded.length) {
    /* The executor recorded its own shape — read it instead of inferring one. */
    return recorded.map(function (s) {
      return buildRecordedExecutionNode(s, kids, byId, ordinalLabel);
    });
  }
  var derived = deriveExecutionChildren(spans);
  var children = [], stepCount = 0, finalAnswer = null, firstReact = null;
  var lastObservation = "";
  derived.forEach(function (item) {
    if (item.type === "step") {
      stepCount += 1;
      if (!firstReact) { firstReact = item.step.reasoning; }
      children.push(buildStepNode(item.step, lastObservation, kids, byId));
      var last = item.step.actions[item.step.actions.length - 1];
      if (last) { lastObservation = (last.attributes || {}).response_text || ""; }
    } else if (item.type === "final") {
      /* Not the turn's answer field, but the call that produces it: ReAct
         runs this ChainOfThought inside forward(), after the loop stops and
         BEFORE the agent returns, so it is the tail of execution rather than
         something the turn does afterwards. Its output then becomes the
         turn's answer verbatim. */
      finalAnswer = item.span;
      var node = spanNode(item.span, kids, byId);
      node.title = "Answer synthesis";
      node.crumb = "Answer synthesis";
      children.push(node);
    } else {
      children.push(spanNode(item.span, kids, byId));
    }
  });
  var answerText = "";
  if (finalAnswer) {
    var out = attrObject((finalAnswer.attributes || {}).module_output) || {};
    answerText = out.final_answer || "";
  }
  var handoff = firstReact ? (moduleKwargs(firstReact) || {}).user_query : "";
  if (!stepCount && !finalAnswer) {
    /* Deterministic ("/" prefixed) and assistant-mode turns run no agent
       loop at all — the turn goes straight to a command. Hoisting the
       children keeps the levels to the ones that actually happened instead
       of interposing an "Execution · 0 steps" hop that explains nothing. */
    return children;
  }
  return makeNode("phase", "Execution" + (ordinalLabel || ""), {
    crumb: "Execution" + (ordinalLabel || ""),
    detail: stepCount + (stepCount === 1 ? " step" : " steps"),
    category: "cat-tool",
    children: children,
    extent: spanExtent(spans),
    tokens: sumTokens(children),
    cost: sumCost(children),
    render: function (container) {
      container.appendChild(el("div", "levelLead",
        "The agent loop: each step turns an observation into reasoning and a " +
        "next action. Tool calls go to the assistant, which resolves intent " +
        "and parameters before running the command."));
      var kv = el("dl", "kv");
      kv.appendChild(el("dt", null, "steps"));
      kv.appendChild(el("dd", null, String(stepCount)));
      container.appendChild(kv);
      appendAttrSection(container, "input received from the planner", handoff);
      appendAttrSection(container, "answer returned to the turn", answerText);
    }
  });
}

function buildPlanningNode(spans, kids, byId, ordinalLabel) {
  /* One planner span per run in practice; if a run ever holds more, the
     extra spans still appear as children rather than being dropped. */
  var planner = spans.filter(function (s) {
    return s.name === "fw.planner.plan" || s.name === "fw.planner.replan";
  })[0] || spans[0];
  var attrs = planner ? (planner.attributes || {}) : {};
  var isReplan = planner && planner.name === "fw.planner.replan";
  var children = [];
  spans.forEach(function (s) {
    (kids[s.span_id] || []).slice().sort(byStartNs).forEach(function (c) {
      children.push(spanNode(c, kids, byId));
    });
  });
  var llm = spans.reduce(function (found, s) {
    return found || (kids[s.span_id] || []).filter(function (c) {
      return c.name === "fw.llm.call";
    })[0];
  }, null);
  var planIn = llm ? moduleKwargs(llm) : null;
  var label = (isReplan ? "Replanning" : "Planning") + (ordinalLabel || "");
  return makeNode("phase", label, {
    crumb: label,
    detail: attrs.replan_trigger && attrs.replan_trigger !== "None"
      ? "triggered by " + attrs.replan_trigger : "",
    category: "cat-planner",
    children: children,
    extent: spanExtent(spans),
    tokens: sumTokens(children),
    cost: sumCost(children),
    render: function (container) {
      container.appendChild(el("div", "levelLead",
        "The planner turns the user's message into the numbered next steps " +
        "it hands to the executor."));
      var kv = el("dl", "kv");
      if (attrs.model) {
        kv.appendChild(el("dt", null, "model"));
        kv.appendChild(el("dd", null, attrs.model));
      }
      if (attrs.replan_trigger && attrs.replan_trigger !== "None") {
        kv.appendChild(el("dt", null, "replan trigger"));
        kv.appendChild(el("dd", null, attrs.replan_trigger));
      }
      container.appendChild(kv);
      appendAttrSection(container, "planner input", planIn);
      appendAttrSection(container, "plan handed to the executor", attrs.plan);
    }
  });
}

function buildOtherNode(spans, kids, byId) {
  /* Work that belongs to neither phase — today the post-turn conversation
     summarizer. Named for what it is rather than hidden. */
  var children = spans.map(function (s) { return spanNode(s, kids, byId); });
  var only = children.length === 1 ? children[0] : null;
  if (only) { return only; }
  return makeNode("phase", "Post-turn work", {
    crumb: "Post-turn",
    category: "cat-other",
    children: children,
    extent: spanExtent(spans),
    tokens: sumTokens(children),
    cost: sumCost(children),
    render: function (container) {
      container.appendChild(el("div", "levelLead",
        "Work outside planning and execution — bookkeeping the turn does " +
        "once the answer exists."));
    }
  });
}

function buildTurnTree(turn, spans) {
  var byId = {}, kids = {};
  spans.forEach(function (s) { byId[s.span_id] = s; kids[s.span_id] = []; });
  var rootSpan = spans.filter(function (s) { return s.name === "fw.turn"; })[0] || null;
  var top = [];
  spans.forEach(function (s) {
    if (rootSpan && s.span_id === rootSpan.span_id) { return; }
    var parent = s.parent_span_id;
    if (parent && kids[parent] && (!rootSpan || parent !== rootSpan.span_id)) {
      kids[parent].push(s);
    } else {
      /* Children of the root, and any span whose parent was never recorded,
         both belong at the top rather than disappearing. */
      top.push(s);
    }
  });
  top.sort(byStartNs);

  var runs = [];
  top.forEach(function (s) {
    var phase = phaseOf(s);
    var last = runs[runs.length - 1];
    if (last && last.phase === phase) { last.spans.push(s); }
    else { runs.push({ phase: phase, spans: [s] }); }
  });

  var counts = {};
  runs.forEach(function (run) { counts[run.phase] = (counts[run.phase] || 0) + 1; });
  var seen = {};
  var children = [];
  runs.forEach(function (run) {
    seen[run.phase] = (seen[run.phase] || 0) + 1;
    /* Only number a phase when the turn really has more than one of it —
       a replan makes "Planning 2 / Execution 2" meaningful, and a plain turn
       should not carry a "1" that implies a missing sibling. */
    var ordinal = counts[run.phase] > 1 ? " " + seen[run.phase] : "";
    var built;
    if (run.phase === PHASE_PLANNING) {
      built = buildPlanningNode(run.spans, kids, byId, ordinal);
    } else if (run.phase === PHASE_EXECUTION) {
      built = buildExecutionNode(run.spans, kids, byId, ordinal);
    } else {
      built = buildOtherNode(run.spans, kids, byId);
    }
    /* A phase builder may decline to wrap and hand back its children instead. */
    children = children.concat(built);
  });

  var extent = rootSpan
    ? spanExtent([rootSpan])
    : mergeExtents(children.map(function (c) { return c.extent; }));

  return makeNode("turn", policedText(turn.user_message) || "(no message)", {
    crumb: (turn.ordinal ? "Turn " + turn.ordinal : "Turn"),
    category: "cat-turn",
    status: turn.status,
    children: children,
    extent: extent,
    tokens: sumTokens(children),
    cost: sumCost(children),
    render: function (container) { renderTurnLevel(container, turn); }
  });
}

function renderTurnLevel(container, turn) {
  container.appendChild(statusBadge(turn));
  appendSignalChips(container, turn.decision_signals);
  var kv = el("dl", "kv");
  function row(k, v) {
    if (v === null || v === undefined || v === "") { return; }
    kv.appendChild(el("dt", null, k));
    kv.appendChild(el("dd", null, v));
  }
  row("turn_key", turn.turn_key);
  row("channel", turn.channel_id);
  row("conversation", turn.conversation_id === null ? "(none)" : "#" + turn.conversation_id +
      (turn.ordinal ? " · turn " + turn.ordinal : ""));
  row("context", turn.entry_context);
  row("workflow", turn.entry_workflow_name);
  row("started", fmtTs(turn.started_at));
  row("completed", turn.completed_at ? fmtTs(turn.completed_at) : "(still open)");
  if (turn.started_at && turn.completed_at) {
    row("wall time", fmtMs(Date.parse(turn.completed_at) - Date.parse(turn.started_at)));
  }
  if (turn.suspended_ms) { row("suspended (human wait)", fmtMs(turn.suspended_ms)); }
  row("failure_reason", policedText(turn.failure_reason));
  row("LLM cost", fmtCostAmount(turn.llm_cost));

  var um = el("div", "msgBlock");
  um.appendChild(el("span", "lbl", "user message"));
  appendPoliced(um, turn.user_message || "");
  appendReuseAction(um, turn);
  container.appendChild(um);
  if (turn.answer) {
    var ans = el("div", "msgBlock");
    ans.appendChild(el("span", "lbl", "answer"));
    appendPoliced(ans, turn.answer);
    container.appendChild(ans);
  }
  var metadata = el("details", "turnMetadata"); metadata.appendChild(el("summary", null, "Turn details & timing"));
  metadata.appendChild(kv); container.appendChild(metadata);
}

function appendReuseAction(block, turn) {
  /* Hands the recorded message to the live composer [fix-9eg.7.4]. It reads
     the record and writes a text box: no request goes out, the recorded turn
     is untouched, and nothing runs until the person presses Send. */
  var actions = el("div", "reuseAction");
  if (captureEnvelope(turn.user_message)) {
    actions.appendChild(el("span", "muted",
      "The recorded message is not available in full, so there is nothing to reuse."));
    block.appendChild(actions);
    return;
  }
  if (!turn.user_message) { return; }
  var button = el("button", null, "Reuse this message in chat");
  button.title = "Copies the text into the live chat composer. Nothing is sent "
    + "until you press Send, and the recorded turn is unchanged.";
  button.addEventListener("click", function () {
    tmReuseRecordedMessage(String(turn.user_message), { turnKey: turn.turn_key });
  });
  actions.appendChild(button);
  actions.appendChild(el("span", "muted",
    "Copies it into the live chat for editing — a new turn under current settings, not a replay."));
  block.appendChild(actions);
}

