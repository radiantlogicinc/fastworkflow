/* -- page links ----------------------------------------------------------
   The fragment names the page on screen, so a reader can copy it and point
   someone at that exact record. It is written with replaceState: moving
   between pages adds no history entries (the record navigator is how a
   reader moves back), and no hashchange fires back into navigation.

   The fragment and not the whole URL is the link. The port and the token in
   front of it change with every launch, so only the part after # still means
   something after a restart, and the token is not something to paste into
   an issue. Pasting a fragment after any live chatbot URL opens its page.

   A formal review owns its fragment. Its row link is already written by the
   review pane, and a blinded review must not gain the identity of the
   experiment or the run it hides, so nothing here writes while one is open. */
var initialPageLink = location.hash.replace(/^#/, "");
var debugPageLink = "";

function reviewOwnsFragment() {
  return !!(review.progress || (session && session.workspace_mode
    && review.assignmentId && review.capability));
}

function replaceFragment(text) {
  if (reviewOwnsFragment()) { return; }
  var url = location.pathname + location.search + (text ? "#" + text : "");
  if (url !== location.pathname + location.search + location.hash) {
    history.replaceState(history.state, "", url);
  }
}

/* Ids in this app are full of colons (experiment channels, workspace turn
   keys), and a fragment may carry `:` `/` `@` `,` as they are; escaping them
   as URLSearchParams does only makes the link unreadable. `&`, `=`, `+`, `#`
   and `%` stay escaped, so URLSearchParams still reads the text back. */
function pageLinkText(params) {
  return Object.keys(params).filter(function (key) {
    var value = params[key];
    return value !== null && value !== undefined && value !== "";
  }).map(function (key) {
    return key + "=" + encodeURIComponent(String(params[key]))
      .replace(/%3A/gi, ":").replace(/%2F/gi, "/").replace(/%40/gi, "@").replace(/%2C/gi, ",");
  }).join("&");
}

/* `params` names the page; keys with no value are left out. A debug view
   that names nothing in particular (a search, an emptied source) is still
   debug mode, which is what its link opens. */
function writePageLink(params) {
  if (reviewOwnsFragment()) { return; }
  debugPageLink = pageLinkText(params) || "debug";
  /* A view can repaint while the Chat tab is showing (the rail's background
     refresh, for one); the address bar names the tab on screen, and debug's
     page comes back with the tab. */
  if (document.getElementById("debugMain").classList.contains("visible")) {
    replaceFragment(debugPageLink);
  }
}

/* -- rail records -------------------------------------------------------
   A rail key is JSON, and a conversation's nests its parent's key inside it,
   so as a fragment it escapes into a wall of %22%5C. The records people link
   to are named the way the recording names them instead: a conversation by
   its store, channel and conversation id, a date by the date. Any other
   record, or a name that would not single out this one node, keeps the key,
   which always does. */
function railNodeLink(node) {
  var params = null;
  if (node.kind === "date") { params = {date: node.info.date}; }
  else if (node.kind === "adhoc") { params = {page: "adhoc"}; }
  else if (node.kind === "conversation") {
    params = {store: node.source && node.source.store_id,
              channel: node.info.channel_id, conversation: node.info.conversation_id};
  }
  if (params) {
    var named = railNodesNamed(new URLSearchParams(pageLinkText(params)));
    if (named && named.length === 1 && named[0][named[0].length - 1] === node) { return params; }
  }
  return {node: node.key};
}

/* The paths to every rail record `params` names, or null when it names no
   kind of rail record at all. A missing value matches a null one: a
   conversation recorded without a channel has no `channel=`. */
function railNodesNamed(params) {
  function same(value, key) {
    return String(value === null || value === undefined ? "" : value) === (params.get(key) || "");
  }
  var match;
  if (params.has("conversation") || params.has("channel")) {
    match = function (n) {
      return n.kind === "conversation" && same(n.source && n.source.store_id, "store")
        && same(n.info.channel_id, "channel") && same(n.info.conversation_id, "conversation");
    };
  } else if (params.has("date")) {
    match = function (n) { return n.kind === "date" && n.info.date === params.get("date"); };
  } else if (params.get("page") === "adhoc") {
    match = function (n) { return n.kind === "adhoc"; };
  } else {
    return null;
  }
  var paths = [];
  function walk(node, trail) {
    var path = trail.concat([node]);
    if (match(node)) { paths.push(path); }
    node.children.forEach(function (child) { walk(child, path); });
  }
  if (hierarchyRoot) { walk(hierarchyRoot, []); }
  return paths;
}

/* -- the level open inside a turn ---------------------------------------
   The ids on trace nodes are a page-wide counter and change on every load,
   so a level is named by what the recording itself names: the span it shows
   when it shows one, or else its position under the turn (an execution
   phase or a step inferred from an older recording, which has no span). */
function traceLevelLink() {
  var params = session && session.workspace_mode
    ? {store: state.storeId, turn: state.turnKey} : {turn: state.turnKey};
  var node = state.path[state.path.length - 1];
  if (state.path.length < 2 || !node) { return params; }
  var spanPath = node.span ? findSpanPath(state.path[0], node.span.span_id) : null;
  if (spanPath && spanPath[spanPath.length - 1] === node) {
    params.span = node.span.span_id;
    return params;
  }
  var positions = [];
  for (var i = 1; i < state.path.length; i++) {
    var at = state.path[i - 1].children.indexOf(state.path[i]);
    if (at < 0) { return params; }
    positions.push(at);
  }
  params.level = positions.join(".");
  return params;
}

/* Called from inside `selectTurn`/`selectWorkspaceTurn`'s stale-guarded
   completion, for the same reason `focusLoadedSpan` is. A position the trace
   does not have says so on screen rather than quietly opening the turn. */
function focusLoadedLevel(level) {
  if (!level || !state.path.length) { return; }
  var path = [state.path[0]];
  var parts = String(level).split(".");
  for (var i = 0; i < parts.length; i++) {
    var at = /^\d+$/.test(parts[i]) ? Number(parts[i]) : -1;
    var child = path[path.length - 1].children[at];
    if (!child) { path = null; break; }
    path.push(child);
  }
  if (path) {
    state.path = path;
    renderLevel();
    return;
  }
  var host = document.getElementById("detail");
  var notice = el("div", "card");
  notice.appendChild(el("div", "empty", "This link names level " + level
    + " of the turn, which its recorded trace does not have. The turn is "
    + "open at its top level."));
  host.insertBefore(notice, host.firstChild);
}

/* -- opening a link ----------------------------------------------------- */
function pageLinkMissing(text) {
  var d = document.getElementById("detail");
  clear(d);
  d.appendChild(el("div", "empty", text));
}

function whenNavigationLoaded(attemptsLeft) {
  if (hierarchyRoot) { return Promise.resolve(); }
  /* A refresh superseded by a newer one settles without painting, so it is
     asked again rather than taken to mean the source holds nothing. */
  return refreshConvs(true).then(function () {
    if (!hierarchyRoot && attemptsLeft > 0) { return whenNavigationLoaded(attemptsLeft - 1); }
  });
}

function openRailPageLink(params) {
  var nodeKey = params.get("node");
  var experiment = params.get("experiment");
  var task = params.get("task");
  var benchmark = params.get("benchmark");
  var tab = params.get("tab");
  var named = railNodesNamed(params);
  if (nodeKey) {
    var path = findHierarchyByKey(nodeKey);
    if (path) { activateHierarchy(path, true); }
    else { pageLinkMissing("This link names a record this source's navigation does not hold."); }
  } else if (named) {
    if (named.length === 1) { activateHierarchy(named[0], true); }
    else { pageLinkMissing("This link names a record this source's navigation does not hold."); }
  } else if (experiment && session && session.workspace_mode) {
    showWorkspaceExperiment({experiment_id: experiment});
  } else if (experiment && task) {
    var view = params.get("view");
    taskView = TASK_VIEWS.some(function (v) { return v.key === view; }) ? view : "runs";
    showExperimentTask(experiment, task);
  } else if (experiment) {
    openBenchmarkExecution(experiment);
  } else if (benchmark) {
    showBenchmark(benchmark, params.get("version") || undefined);
  } else if (params.get("page") === "benchmarks") {
    showBenchmarks();
  } else if (tab === "conversations" || tab === "benchmarks") {
    setNavigationTab(tab);
  }
}

/* Every link the writers above produce opens here, from a fresh load and
   from a fragment pasted into a live page alike. Returns whether the text
   named a page, so the caller can keep its default landing view otherwise. */
function openPageLink(text) {
  if (!text) { return false; }
  if (text === "test") { setTopMode("test"); return true; }
  if (text === "debug") { setTopMode("debug"); return true; }
  var params = new URLSearchParams(text);
  var turn = params.get("turn");
  if (session && session.workspace_mode && turn && params.get("store")) {
    setTopMode("debug");
    selectWorkspaceTurn(params.get("store"), turn, params.get("span"), null, params.get("level"));
    return true;
  }
  if (session && session.workspace_mode && turn) {
    pageLinkMissing("Unscoped turn links are refused in workspace mode; include store and turn.");
    return true;
  }
  if (turn) {
    setTopMode("debug");
    selectTurnWithRetry(turn, 3, params.get("span"), params.get("level"));
    return true;
  }
  if (["node", "conversation", "channel", "date", "experiment", "benchmark", "page", "tab"].some(function (key) { return params.has(key); })) {
    setTopMode("debug");
    whenNavigationLoaded(2).then(function () { openRailPageLink(params); }).catch(function (error) {
      pageLinkMissing("Could not load navigation to open this link: " + error.message);
    });
    return true;
  }
  return false;
}

window.addEventListener("hashchange", function () {
  if (!session || reviewOwnsFragment()) { return; }
  openPageLink(location.hash.replace(/^#/, ""));
});

document.getElementById("copyLinkBtn").addEventListener("click", function () {
  var link = "#" + (debugPageLink || "debug");
  function copied() { showNotice("Page link copied", "ok", link); }
  function shown() {
    showNotice("Copy this page link", "ok", link);
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(link).then(copied, shown);
  } else {
    shown();
  }
});
