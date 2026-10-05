/* -- header status pill ------------------------------------------------ */
function setPill(cls, text) {
  var pill = document.getElementById("statusPill");
  pill.className = "pill" + (cls ? " " + cls : "");
  document.getElementById("statusPillText").textContent = text;
}

/* -- health banner [R13] ----------------------------------------------- */
var healthDismissed = false;
var dbMissingBannerShown = false;
function refreshHealth() {
  api("/api/health").then(function (data) {
    if (data.db_available !== false && dbMissingBannerShown) {
      dbMissingBannerShown = false;
      document.getElementById("healthBanner").classList.remove("visible");
    }
    if (data.db_available === false && !healthDismissed) {
      dbMissingBannerShown = true;
      /* Absent (cold start before the first turn) or unreadable (e.g. a WAL
         snapshot needing recovery this read-only viewer will never write). */
      document.getElementById("healthText").textContent =
        "no recorded turns yet — the observability DB appears after the " +
        "workflow's first turn. Debug views stay empty until then.";
      document.getElementById("healthBanner").classList.add("visible");
      return;
    }
    var h = data.writer_health;
    if (!h || healthDismissed) { return; }
    var drops = (h.spans_dropped || 0) + (h.records_dropped || 0);
    var errors = (h.write_errors || 0);
    if (drops > 0 || errors > 0) {
      var parts = [];
      if (h.records_dropped) { parts.push(h.records_dropped + " turn record(s) dropped"); }
      if (h.spans_dropped) { parts.push(h.spans_dropped + " span(s) dropped"); }
      if (errors) { parts.push(errors + " write error(s)"); }
      if (h.last_error) { parts.push("last error: " + h.last_error); }
      document.getElementById("healthText").textContent =
        "the observability writer reported " + parts.join(", ") +
        ". Some data in this view may be incomplete.";
      document.getElementById("healthBanner").classList.add("visible");
    }
  }).catch(function () {
    /* Banner is advisory; a dead chatbot is surfaced by checkSession. */
  });
}
document.getElementById("healthDismiss").addEventListener("click", function () {
  healthDismissed = true;
  document.getElementById("healthBanner").classList.remove("visible");
});

/* -- meta -------------------------------------------------------------- */
function refreshMeta() {
  if (session && session.workspace_mode) { return; }
  api("/api/meta").then(function (m) {
    var size = m.db_size_bytes;
    var sizeTxt = size > 1048576 ? (size / 1048576).toFixed(1) + " MB" : Math.round(size / 1024) + " KB";
    document.getElementById("metaLine").textContent =
      m.workflow_name ? (m.workflow_name + " · " + sizeTxt + " recorded") : "";
  }).catch(function () {
    /* Meta is decorative header text; leave the previous line in place. */
  });
}

/* -- debug rail: channel > conversation > turns ------------------------- */
var conversationRefresh = 0;
var hierarchyRoot = null;
var hierarchyScope = null;
var hierarchyPath = [];
var hierarchyExpanded = {};
var hierarchyIndex = null;
var navigationETag = null;
var navigationAbort = null;
var navigationTab = "conversations";
var navigationSelection = {conversations: null, benchmarks: null};
var archivedExperimentsShown = {};

function rebuildHierarchyIndex() {
  hierarchyIndex = { byKey: {}, recordedByExperimentId: {}, byExperimentId: {} };
  function walk(node, trail) {
    var path = trail.concat([node]);
    hierarchyIndex.byKey[node.key] = path;
    if (node.kind === "experiment" && node.experiment_id) {
      if (!hierarchyIndex.byExperimentId[node.experiment_id]) {
        hierarchyIndex.byExperimentId[node.experiment_id] = path;
      }
      if (node.recorded) {
        hierarchyIndex.recordedByExperimentId[node.experiment_id] = path;
      }
    }
    (node.children || []).forEach(function (child) { walk(child, path); });
  }
  if (hierarchyRoot) { walk(hierarchyRoot, []); }
}

function findHierarchyByKey(key) {
  if (!hierarchyIndex) { rebuildHierarchyIndex(); }
  return (hierarchyIndex.byKey[key] || null);
}

function findRecordedExperimentPath(experimentId) {
  if (!hierarchyIndex) { rebuildHierarchyIndex(); }
  return hierarchyIndex.recordedByExperimentId[experimentId] || null;
}

function findExperimentPath(experimentId) {
  if (!hierarchyIndex) { rebuildHierarchyIndex(); }
  return hierarchyIndex.byExperimentId[experimentId]
    || hierarchyIndex.recordedByExperimentId[experimentId]
    || null;
}

function hierarchyTab(path) {
  for (var i = 0; i < path.length; i++) {
    if (path[i].kind === "adhoc" || path[i].kind === "date") { return "conversations"; }
  }
  return "benchmarks";
}

function setNavigationTabButtons(tab) {
  [
    ["conversations", "navConversations"],
    ["benchmarks", "navBenchmarks"]
  ].forEach(function (item) {
    var active = item[0] === tab;
    var button = document.getElementById(item[1]);
    button.className = active ? "active" : "";
    button.setAttribute("aria-selected", active ? "true" : "false");
    button.tabIndex = active ? 0 : -1;
  });
}

function onNavigationTabKey(event) {
  var tabs = [
    document.getElementById("navConversations"),
    document.getElementById("navBenchmarks")
  ];
  var names = ["conversations", "benchmarks"];
  var index = tabs.indexOf(event.currentTarget);
  if (index < 0) { return; }
  var next = index;
  if (event.key === "ArrowRight") { next = (index + 1) % tabs.length; }
  else if (event.key === "ArrowLeft") { next = (index - 1 + tabs.length) % tabs.length; }
  else if (event.key === "Home") { next = 0; }
  else if (event.key === "End") { next = tabs.length - 1; }
  else { return; }
  event.preventDefault();
  if (next !== index) { setNavigationTab(names[next]); }
  tabs[next].focus();
}

function tabNodes(tab) {
  if (!hierarchyRoot) { return []; }
  if (tab === "conversations") {
    var adhoc = hierarchyRoot.children.find(function (node) { return node.kind === "adhoc"; });
    return adhoc ? adhoc.children : [];
  }
  return hierarchyRoot.children.filter(function (node) { return node.kind !== "adhoc"; });
}

function showNavigationTabEmpty() {
  /* The rail's periodic refresh lands here whenever no record is selected,
     which is the normal state during a search -- the operator is looking at
     results, not at a conversation. Repainting the placeholder over them would
     erase a walk that takes several round trips, so the finder keeps the pane
     until a navigation gesture releases it (turnFindRelease). */
  if (turnFind.active) { return; }
  expNavToken();
  state.turnKey = null;
  state.turn = null;
  state.path = [];
  var d = document.getElementById("detail");
  clear(d);
  var nodes = tabNodes(navigationTab);
  if (navigationTab === "conversations") {
    emptyState(d, nodes.length ? "Explore a conversation" : "No conversations yet",
      nodes.length
        ? "Select a conversation on the left to revisit its turns and inspect what happened."
        : "Start chatting with your workflow and its ad-hoc conversations will appear here.");
  } else {
    emptyState(d, nodes.length ? "Explore a benchmark" : "No benchmarks yet",
      nodes.length
        ? "Select a benchmark on the left to review its tasks, experiments, and results."
        : "Create a benchmark to start measuring repeatable workflow tasks.");
  }
}

function setNavigationTab(tab, restoreDetail) {
  navigationTab = tab;
  setNavigationTabButtons(tab);
  var selected = navigationSelection[tab];
  hierarchyPath = selected
    ? (findHierarchyByKey(selected) || [])
    : [];
  renderHierarchy();
  if (restoreDetail === false) { return; }
  if (hierarchyPath.length) {
    var node = hierarchyPath[hierarchyPath.length - 1];
    activateHierarchy(hierarchyPath, !!hierarchyExpanded[node.key]);
  } else {
    showNavigationTabEmpty();
  }
}

function findHierarchy(predicate) {
  function walk(node, trail) {
    var path = trail.concat([node]);
    if (predicate(node)) { return path; }
    for (var i = 0; i < node.children.length; i++) {
      var found = walk(node.children[i], path);
      if (found) { return found; }
    }
    return null;
  }
  return hierarchyRoot ? walk(hierarchyRoot, []) : null;
}

function focusHierarchy(predicate) {
  var path = findHierarchy(predicate);
  if (!path) { return false; }
  navigationTab = hierarchyTab(path);
  navigationSelection[navigationTab] = path[path.length - 1].key;
  setNavigationTabButtons(navigationTab);
  hierarchyPath = path;
  path.slice(0, -1).forEach(function (node) { hierarchyExpanded[node.key] = true; });
  renderHierarchy();
  return true;
}

function hierarchyLabel(node) {
  /* An experiment node carries the author's free-text description, which belongs
     on its page and not in a rail row: here it is named by its id, the one
     thing that identifies it at a glance and always fits. */
  if (node.kind === "experiment" && node.experiment_id) {
    return "Experiment · " + node.experiment_id.slice(-8);
  }
  return policedText(node.label);
}

function visibleHierarchyChildren(node) {
  if (node.kind !== "benchmark" || archivedExperimentsShown[node.benchmark_id]) {
    return node.children;
  }
  return node.children.filter(function (child) {
    return child.kind !== "experiment" || !(child.info && child.info.archived);
  });
}

function benchmarkArchivedCount(node) {
  return node.children.filter(function (child) {
    return child.kind === "experiment" && child.info && child.info.archived;
  }).length;
}

function navSnippet(text, words) {
  /* The rail and the crumbs are for orientation, not reading: three words is
     enough to recognise a record, and the full text stays as the tooltip. */
  var limit = words || 3;
  var flat = String(text === null || text === undefined ? "" : text)
    .replace(/\s+/g, " ").trim();
  /* Slug-cased labels ("roster-walk-davis-and-sons-plc") carry their words on
     hyphens and underscores instead of spaces, so those break a word too —
     splitting on spaces alone left those rows at full length. */
  var parts = flat.match(/[^\s\-_]+[\s\-_]?/g) || [];
  var snippet = parts.length <= limit ? flat
    : parts.slice(0, limit).join("").replace(/[\s\-_]+$/, "") + " …";
  /* A label can also be one long unbroken token, which no word count trims. */
  return snippet.length > 34 ? snippet.slice(0, 33) + "…" : snippet;
}

function fillHierarchyCrumbs(container) {
  clear(container);
  (hierarchyPath.length ? hierarchyPath : (hierarchyRoot ? [hierarchyRoot] : [])).forEach(function (node, index) {
    if (index) { container.appendChild(el("span", "sep", "›")); }
    var button = el("button", null, navSnippet(hierarchyLabel(node)));
    button.title = policedText(node.label);
    button.addEventListener("click", function () {
      var path = findHierarchyByKey(node.key);
      if (path) { activateHierarchy(path, true); }
    });
    container.appendChild(button);
  });
}

/* Turns recorded outside any conversation (CLI runs, embedders) stay
   reachable rather than being filtered out with the empty ones. */
function renderHierarchy() {
  var list = document.getElementById("convList");
  clear(list);
  if (!hierarchyRoot) { return; }
  var visiblePath = hierarchyPath.filter(function (n) { return n.kind !== "component"; });
  var selected = visiblePath.length ? visiblePath[visiblePath.length - 1].key : null;
  function render(node, trail) {
    var path = trail.concat([node]);
    /* The rail stops at the turn: a turn's children are trace components,
       which the right pane already walks with its own breadcrumb. */
    /* Channel and conversation are the enclosing groups; repeating them on
       every row would just crowd out the message. */
    var leaf = node.kind === "turn";
    var archived = node.kind === "experiment" && node.info && node.info.archived;
    var row = el("details", "navNode" + (node.kind === "conversation" ? " conversation" : "") + (selected === node.key ? " selected" : "") + (archived ? " archived" : ""));
    row.open = !leaf && !!hierarchyExpanded[node.key];
    row.dataset.kind = node.kind;
    var summary = el("summary");
    var childNodes = visibleHierarchyChildren(node);
    var navLabel = navSnippet(hierarchyLabel(node));
    summary.title = policedText(node.label) || hierarchyLabel(node);
    var labelNode = el("span", "navLabel", navLabel);
    if (node.kind === "experiment" && node.experiment_id) {
      row.dataset.experimentId = node.experiment_id;
    }
    summary.appendChild(labelNode);
    if (archived) {
      summary.appendChild(el("span", "pill", "Archived"));
    }
    var archivedCount = node.kind === "benchmark" ? benchmarkArchivedCount(node) : 0;
    if (archivedCount) {
      var showingArchived = !!archivedExperimentsShown[node.benchmark_id];
      var archiveToggle = el("button", "navArchiveToggle", showingArchived ? "⊘" : "◉");
      var archiveLabel = showingArchived ? "Hide archived experiments" : "Show all experiments";
      archiveToggle.title = archiveLabel;
      archiveToggle.setAttribute("aria-label", archiveLabel);
      archiveToggle.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        archivedExperimentsShown[node.benchmark_id] = !showingArchived;
        renderHierarchy();
        var current = hierarchyPath[hierarchyPath.length - 1];
        if (current && current.kind === "benchmark" && current.key === node.key) {
          showBenchmark(node.benchmark_id);
        }
      });
      summary.appendChild(archiveToggle);
    }
    if (childNodes.length && !leaf) {
      var count = el("span", "navCount", childNodes.length); count.setAttribute("aria-hidden", "true"); summary.appendChild(count);
    }
    if (selected === node.key) { summary.setAttribute("aria-current", "page"); }
    if (node.kind === "turn") { summary.title = policedText(node.label); }
    summary.addEventListener("click", function (event) {
      event.preventDefault();
      activateHierarchy(path, !hierarchyExpanded[node.key]);
    });
    row.appendChild(summary);
    if (!leaf && childNodes.length) {
      var childrenContainer = el("div", "navChildren");
      childNodes.forEach(function (child) { childrenContainer.appendChild(render(child, path)); });
      row.appendChild(childrenContainer);
    } else if (!leaf && row.open) {
      row.appendChild(el("div", "sub", node.kind === "benchmark" && archivedCount ? "Archived experiments are hidden." :
        node.kind === "experiment" ? "No conversations recorded yet." :
        node.kind === "conversation" ? "No turns recorded yet." : "No child records."));
    }
    return row;
  }
  if (navigationTab === "conversations") {
    var adhoc = hierarchyRoot.children.find(function (node) { return node.kind === "adhoc"; });
    if (adhoc) {
      adhoc.children.forEach(function (node) { list.appendChild(render(node, [hierarchyRoot, adhoc])); });
    }
  } else if (navigationTab === "benchmarks") {
    hierarchyRoot.children.filter(function (node) { return node.kind !== "adhoc"; })
      .forEach(function (node) { list.appendChild(render(node, [hierarchyRoot])); });
  }
  (hierarchyRoot.info.warnings || []).forEach(function (warning) {
    list.appendChild(el("p", "err", warning));
  });
  /* Every path change repaints the rail, so this is the one seam where the
     arrows can be told what is reachable from where the reader now is. */
  renderRecordNavigator();
}

function showHierarchyInfo(node) {
  expNavToken();
  var d = document.getElementById("detail"); clear(d);
  /* .crumbs cancels #detail's padding with negative margins so it can sit flush
     at the top of the pane and stick there; nested inside the card it painted
     over the heading above it instead. Same placement as renderLevel. */
  var crumbs = el("nav", "crumbs"); fillHierarchyCrumbs(crumbs); d.appendChild(crumbs);
  var card = el("div", "card");
  card.appendChild(el("h2", null, policedText(node.label)));
  if (node.kind === "adhoc" || node.kind === "date") {
    card.appendChild(el("p", "sub", "Conversations recorded outside experiments, grouped by UTC date."));
  }
  card.appendChild(el("p", "sub", node.children.length + " " +
    (node.kind === "conversation" ? "turns" : node.kind === "adhoc" ? "dates" : "child records")));
  Object.keys(node.info || {}).forEach(function (key) {
    var value = node.info[key];
    if (value === null || value === undefined || key === "record_json") { return; }
    var pair = el("div"); pair.appendChild(el("strong", null, key.replace(/_/g, " ") + ": "));
    pair.appendChild(el("span", null, typeof value === "object" ? JSON.stringify(value) : String(value)));
    card.appendChild(pair);
  });
  if (node.kind === "conversation") {
    card.appendChild(el("h3", null, "Turns"));
    var conversationPath = hierarchyPath.slice();
    node.children.forEach(function (turn) {
      var button = el("button", "listItem", turn.label);
      button.appendChild(el("div", "sub", (turn.info.status || "") + " · " + fmtTs(turn.info.started_at)));
      button.addEventListener("click", function () { activateHierarchy(conversationPath.concat([turn]), true); });
      card.appendChild(button);
    });
    if (!node.children.length) { card.appendChild(el("p", "empty", "No turns recorded yet.")); }
  }
  d.appendChild(card);
}

function activateHierarchy(path, expanded) {
  navigationTab = hierarchyTab(path);
  var visibleSelection = path.filter(function (node) { return node.kind !== "component"; });
  navigationSelection[navigationTab] =
    visibleSelection.length ? visibleSelection[visibleSelection.length - 1].key : null;
  setNavigationTabButtons(navigationTab);
  hierarchyPath = path;
  var node = path[path.length - 1];
  path.slice(0, -1).forEach(function (n) { hierarchyExpanded[n.key] = true; });
  hierarchyExpanded[node.key] = expanded;
  if (node.source && node.source.store_id) { state.storeId = node.source.store_id; }
  renderHierarchy();
  if (node.kind === "component") {
    expNavToken();
    state.path = node.tracePath;
    renderLevel();
  } else if (node.kind === "turn") {
    if (node.source && node.source.store_id) { selectWorkspaceTurn(node.source.store_id, node.turn_key); }
    else { selectTurn(node.turn_key); }
  } else {
    state.turnKey = null; state.turn = null; state.path = [];
    if (node.kind === "root") { showBenchmarks(); }
    else if (node.kind === "benchmark" && node.benchmark_id) { showBenchmark(node.benchmark_id); }
    else if (node.kind === "experiment" && !(node.source && node.source.store_id)) {
      if (node.recorded) { showExperiment(node.experiment_id); }
      else if (node.registered) { showBenchmarkExperiment(node.experiment_id); }
      else { showHierarchyInfo(node); }
    } else { showHierarchyInfo(node); }
  }
}

function alignHierarchyTurn(turnKey) {
  var path = findHierarchy(function (n) {
    if (n.kind !== "turn" || n.turn_key !== turnKey) { return false; }
    if (session && session.workspace_mode) { return n.source && n.source.store_id === state.storeId; }
    return true;
  });
  if (path) {
    navigationTab = hierarchyTab(path);
    navigationSelection[navigationTab] = path[path.length - 1].key;
    setNavigationTabButtons(navigationTab);
    hierarchyPath = path;
    path.slice(0, -1).forEach(function (n) { hierarchyExpanded[n.key] = true; });
    renderHierarchy();
  }
}

function attachTraceHierarchy() {
  if (!state.turn || !state.path.length || review.progress) { return; }
  alignHierarchyTurn(state.turn.turn_key);
  var turnNode = hierarchyPath[hierarchyPath.length - 1];
  if (!turnNode || turnNode.kind !== "turn") { return; }
  function wrap(trace, tracePath) {
    var path = tracePath.concat([trace]);
    return {key: turnNode.key + "/" + trace.id, kind: "component", label: trace.title,
      source: turnNode.source, tracePath: path,
      children: trace.children.map(function (child) { return wrap(child, path); })};
  }
  turnNode.children = state.path[0].children.map(function (child) { return wrap(child, [state.path[0]]); });
}

function syncTraceHierarchy() {
  if (review.progress || !state.turn || !state.path.length) { return; }
  var trace = state.path[state.path.length - 1];
  if (state.path.length === 1) { alignHierarchyTurn(state.turn.turn_key); }
  else {
    var path = findHierarchy(function (n) { return n.kind === "component" && n.tracePath[n.tracePath.length - 1] === trace; });
    if (path) {
      hierarchyPath = path;
      path.slice(0, -1).forEach(function (n) { hierarchyExpanded[n.key] = true; });
    }
  }
  renderHierarchy();
}

function detailScroller() {
  /* Below 700px #detail is overflow:visible inside a column that scrolls as a
     whole, so the offsets live on the workbench element instead. */
  var detail = document.getElementById("detail");
  return detail.scrollHeight > detail.clientHeight
    ? detail : document.getElementById("debugMain");
}

/* A record that overflows brings a scrollbar with it and a short one takes it
   away, so the tab is re-seated beside whichever the new record has. The find
   tab shares that inset, including while the arrows themselves are hidden. */
new MutationObserver(function () {
  if (document.getElementById("debugMain").classList.contains("visible")) {
    placeRecordNavigator();
  }
}).observe(document.getElementById("detail"), {childList: true});

/* -- the record navigator ------------------------------------------------
   Every right-pane view is a node of the rail hierarchy the crumbs already
   spell out, so reading through the content is four moves on hierarchyPath:
   up to the parent, down into the first child, and back and forth between
   siblings. A move with nowhere to go disables its arrow instead of hiding
   it, so the first and last step of an execution are visible at a glance. */
var RECORD_NAV_MOVES = [
  {id: "up", element: "recordNavUp", verb: "Up to"},
  {id: "down", element: "recordNavDown", verb: "Down into"},
  {id: "prev", element: "recordNavPrev", verb: "Previous:"},
  {id: "next", element: "recordNavNext", verb: "Next:"}
];

function recordNavTarget(move) {
  if (!hierarchyRoot || review.progress) { return null; }
  if (!hierarchyPath.length) {
    /* Nothing is selected, so the only move that means anything is into the
       first row of the tab the rail is showing. */
    var first = tabNodes(navigationTab)[0];
    return move === "down" && first
      ? findHierarchy(function (n) { return n.key === first.key; })
      : null;
  }
  var node = hierarchyPath[hierarchyPath.length - 1];
  if (move === "up") { return hierarchyPath.length > 1 ? hierarchyPath.slice(0, -1) : null; }
  if (move === "down") {
    return node.children.length ? hierarchyPath.concat([node.children[0]]) : null;
  }
  if (hierarchyPath.length < 2) { return null; }
  var siblings = hierarchyPath[hierarchyPath.length - 2].children;
  var at = siblings.indexOf(node), index = at + (move === "next" ? 1 : -1);
  if (at < 0 || index < 0 || index >= siblings.length) { return null; }
  return hierarchyPath.slice(0, -1).concat([siblings[index]]);
}

function placeTurnFindDialog() {
  /* The popup's top edge is the same line as the magnifying-glass tab (the
     header band's bottom, in CSS). Its right edge clears that tab. */
  var tab = document.getElementById("turnFindNav");
  var dialog = document.getElementById("turnFindDialog");
  if (!tab || !dialog) { return; }
  var rect = tab.getBoundingClientRect();
  if (!rect.width) { return; }
  dialog.style.right = (window.innerWidth - rect.left + 8) + "px";
}

function placeRecordNavigator() {
  /* Pinned to the window edge the tab would cover the pane's scrollbar for the
     height of the tab, and that scrollbar belongs to the very pane the arrows
     move through. Sitting just inside it costs nothing where the platform
     draws no scrollbar at all. The find tab uses the same inset so the two
     stay on one vertical line. */
  var scroller = detailScroller();
  var bar = Math.max(0, scroller.offsetWidth - scroller.clientWidth);
  var nav = document.getElementById("recordNav");
  nav.style.right = bar + "px";
  document.getElementById("turnFindNav").style.right = bar + "px";
  if (document.getElementById("turnFindDialog").open) { placeTurnFindDialog(); }
  if (!nav.className) { return; }
  /* The reading edge clears the measured tab and scrollbar by 3px. Computing
     the reserve here keeps that gap true when either responsive padding or
     the platform's scrollbar width changes. The scrollbar is already outside
     the content box and is also the tab's inset, so those two widths cancel. */
  var detailPad = parseFloat(getComputedStyle(document.documentElement)
    .getPropertyValue("--detail-pad")) || 0;
  document.getElementById("debugMain").style.setProperty(
    "--record-nav-gutter",
    Math.max(0, nav.getBoundingClientRect().width + 2.5 - detailPad) + "px"
  );
}
window.addEventListener("resize", function () {
  if (document.getElementById("debugMain").classList.contains("visible")) {
    placeRecordNavigator();
  }
});

function renderRecordNavigator() {
  var main = document.getElementById("debugMain");
  /* A formal review walks its own assigned rows through the review pane, and
     that order is the point of the assignment; the hierarchy arrows would
     wander out of it. */
  var usable = !!hierarchyRoot && !review.progress
    && main.classList.contains("visible") && !main.classList.contains("reviewActive");
  document.getElementById("recordNav").className = usable ? "visible" : "";
  if (usable) { placeRecordNavigator(); }
  RECORD_NAV_MOVES.forEach(function (move) {
    var button = document.getElementById(move.element);
    var target = usable ? recordNavTarget(move.id) : null;
    button.disabled = !target;
    button.title = target
      ? move.verb + " " + navSnippet(hierarchyLabel(target[target.length - 1]), 5) : "";
  });
}

function navigateRecord(move) {
  var target = recordNavTarget(move);
  if (!target) { return false; }
  activateHierarchy(target, true);
  return true;
}

RECORD_NAV_MOVES.forEach(function (move) {
  document.getElementById(move.element).addEventListener("click", function () {
    navigateRecord(move.id);
  });
});

function refreshConvs(reportFailure) {
  var refresh = ++conversationRefresh;
  var force = reportFailure === true;
  var workflow = session && session.workflow_path;
  var scope = JSON.stringify([workflow, session && session.workspace]);
  if (scope !== hierarchyScope) {
    hierarchyScope = scope; hierarchyRoot = null; hierarchyPath = []; hierarchyExpanded = {};
    hierarchyIndex = null; navigationETag = null;
    navigationSelection = {conversations: null, benchmarks: null};
    archivedExperimentsShown = {};
  }
  if (!hierarchyRoot) {
    var list = document.getElementById("convList"); clear(list);
    list.appendChild(el("div", "empty", "Loading navigation…"));
  }
  if (navigationAbort) { navigationAbort.abort(); }
  navigationAbort = (typeof AbortController !== "undefined") ? new AbortController() : null;
  var headers = {};
  if (!force && navigationETag) { headers["If-None-Match"] = navigationETag; }
  return api("/api/navigation", {
    signal: navigationAbort && navigationAbort.signal,
    headers: headers,
    allow304: !force,
    captureETag: true
  }).then(function (result) {
    if (refresh !== conversationRefresh || scope !== JSON.stringify([session && session.workflow_path, session && session.workspace])) { return; }
    if (result && result.notModified) {
      /* Unchanged body: keep the rail as painted; do not rebuild. */
      return;
    }
    var data = result && result.body !== undefined ? result.body : result;
    if (result && result.etag) { navigationETag = result.etag; }
    hierarchyRoot = data.root;
    rebuildHierarchyIndex();
    var selected = navigationSelection[navigationTab];
    hierarchyPath = selected
      ? (findHierarchyByKey(selected) || [])
      : [];
    attachTraceHierarchy();
    syncTraceHierarchy();
    renderHierarchy();
    /* A record that is SELECTED keeps the pane, even if the rail has no path
       to it. `attachTraceHierarchy` above can only align a turn whose trace
       has finished loading (`state.turn`), but a turn is selected the moment
       somebody clicks it or opens a deep link — so a navigation read landing
       inside that window used to find `hierarchyPath` empty and repaint "No
       conversations yet" over a trace that was still arriving. In workspace
       mode the rail may have no path to a scoped turn at all, which made
       every periodic refresh do it, not just the first.

       This is the same rule the finder guard above states: a background
       refresh does not get to take the pane. A navigation GESTURE still
       does, because `setNavigationTab` calls the placeholder directly, and
       clearing conversations nulls `state.turnKey` before refreshing so the
       "all cleared" state still appears. */
    if (!hierarchyPath.length && !state.turnKey) { showNavigationTabEmpty(); }
  }).catch(function (error) {
    if (requestWasAborted(error) || refresh !== conversationRefresh) { return; }
    var list = document.getElementById("convList"); clear(list);
    list.appendChild(el("div", "empty", "Could not load navigation: " + error.message));
    if (reportFailure === true) { throw error; }
  });
}

document.getElementById("navConversations").addEventListener("click", function () {
  setNavigationTab("conversations");
});
document.getElementById("navConversations").addEventListener("keydown", onNavigationTabKey);
document.getElementById("navBenchmarks").addEventListener("click", function () {
  setNavigationTab("benchmarks");
});
document.getElementById("navBenchmarks").addEventListener("keydown", onNavigationTabKey);

