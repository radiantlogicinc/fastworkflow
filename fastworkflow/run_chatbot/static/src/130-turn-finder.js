/* -- finding turns across the whole store [fix-9eg.18.1] -----------------
   The rail's filters used to be applied to a page the server had already cut,
   so a turn matching further down the store was reported as no match. The
   server now filters the whole authorized dataset; this is the control for it.

   Cost is bounded the way the API intends: each request scans at most
   TURN_FIND_SCAN turns and returns at most TURN_FIND_PAGE matches, and the
   walk continues by handing `next_scan_cursor` back as `resume_after`. Because
   the page limit is smaller than the scan limit, this is exactly the case that
   used to lose matches, and every segment's rows are accumulated rather than
   replacing the last.

   Three things this must never do: rescan the store on every keystroke or
   chip toggle (hence searching only on Find, and the stale-response guard),
   say "no matches" while the walk
   is unfinished (hence the auto-continue below, which only stops when
   something was found, the walk completed, or the operator stopped it), and
   DROP a row it has already fetched. There used to be a 200-row ceiling here
   that silently discarded later matches while still advancing the cursor past
   them, which put a match beyond the 200th permanently out of reach -- the
   same class of false negative on the page that fix-9eg.18.1 removed from the
   API. Nothing is discarded now. Growth is bounded by the operator instead:
   the walk auto-continues only while the list is still empty, so every row
   after the first segment arrives on a deliberate "keep searching" click. */
var TURN_FIND_PAGE = 25;
var TURN_FIND_SCAN = 200;
var turnFind = {
  seq: 0, timer: null, rows: [], cursor: null, scanned: 0, matched: 0,
  complete: true, facets: null, markers: {}, text: "", running: false,
  stopped: false, active: false, error: null, basis: "", nav: 0,
  scope: null, source: null, sourceMoved: false, abort: null
};

/* -- scoping the search to one experiment, task or attempt [fix-9eg.3.1.1]
   `/api/turns` has filtered by experiment/task/attempt since the attempt
   opener was written against it (see openExperimentAttempt); the finder
   simply never sent them, so "what went wrong in this run" meant searching
   the whole store and reading past every other run to find out.

   A scope belongs to the SOURCE it was chosen in. One experiment id can
   exist in two stores and mean two different sets of runs, so a scope
   carried across a source change would ask the new database for the old
   one's labels and present whatever came back under the old one's heading.
   The scope is stamped with the source it was chosen in and is dropped, with
   the walk and its cursor, when that source moves.

   What a scoped result is: the turns RECORDED under that experiment, task
   or attempt. A marker on one of them says that turn recorded it; it does
   not attribute the trouble to the scope, and the counts stay the scan's
   own the way they are for an unscoped search. */
function turnFindSourceKey() {
  /* What `api()` will actually read: a workspace read is pinned to its
     store by turnFindPath below, and everything else is this workflow's own
     store. */
  if (session && session.workspace_mode) { return "workspace:" + (state.storeId || ""); }
  return "current";
}

function turnFindSourceMoved() {
  return turnFind.source !== null && turnFind.source !== turnFindSourceKey();
}

function turnFindScopeText(scope) {
  var parts = [];
  if (scope.experiment) { parts.push("experiment " + scope.experiment); }
  if (scope.task) { parts.push("task " + scope.task); }
  /* attempt 0 is a legal attempt number, so emptiness is tested and not
     truthiness -- the one that would silently widen the scope. */
  if (scope.attempt !== null && scope.attempt !== undefined) {
    parts.push("attempt " + scope.attempt);
  }
  return parts.join(" \u00b7 ");
}

function turnFindScopePhrase(prefix) {
  /* "the whole store" stops being true the moment a scope is on: the walk
     covered the scope, and the rest of the store was never looked at. */
  return prefix + (turnFind.scope ? "the whole of this scope" : "the whole store");
}

function turnFindSelected() {
  return Object.keys(turnFind.markers).filter(function (k) { return turnFind.markers[k]; });
}

function turnFindPath(cursor) {
  var path = "/api/turns";
  var params = ["limit=" + TURN_FIND_PAGE, "scan_limit=" + TURN_FIND_SCAN];
  if (session && session.workspace_mode) {
    /* A workspace read is store-scoped or it is refused; the unscoped route
       says so rather than searching across archives. */
    path = "/api/workspace/turns";
    params.push("store_id=" + encodeURIComponent(state.storeId || ""));
  }
  if (turnFind.text) { params.push("text=" + encodeURIComponent(turnFind.text)); }
  var markers = turnFindSelected();
  /* Selected chips narrow together: a turn must carry every one of them. */
  if (markers.length) { params.push("markers_all=" + encodeURIComponent(markers.join(","))); }
  if (markers.indexOf("low_confidence") >= 0) {
    params.push("low_confidence_below=" + lowConfidenceThreshold());
  }
  /* The store's own filters, passed through by the same names the route has
     always taken, so a scoped search here is the scoped search an agent
     makes and an attempt link opens. */
  if (turnFind.scope) {
    if (turnFind.scope.experiment) {
      params.push("experiment=" + encodeURIComponent(turnFind.scope.experiment));
    }
    if (turnFind.scope.task) {
      params.push("task=" + encodeURIComponent(turnFind.scope.task));
    }
    if (turnFind.scope.attempt !== null && turnFind.scope.attempt !== undefined) {
      params.push("attempt=" + encodeURIComponent(turnFind.scope.attempt));
    }
  }
  if (cursor) { params.push("resume_after=" + encodeURIComponent(cursor)); }
  return path + "?" + params.join("&");
}

function turnFindReset() {
  turnFind.seq += 1;
  if (turnFind.abort) { turnFind.abort.abort(); turnFind.abort = null; }
  turnFind.rows = []; turnFind.cursor = null; turnFind.scanned = 0;
  turnFind.matched = 0; turnFind.complete = true; turnFind.facets = null;
  turnFind.stopped = false; turnFind.error = null; turnFind.basis = "";
  turnFind.sourceMoved = false;
  /* The scope survives: a marker chip, a new search word and every
     continuation are all asked WITHIN the scope the operator chose, and
     silently widening back to the store is how a scoped count becomes a
     store-wide one under a scoped heading. It is dropped by the Clear
     control, by a source change, and by the source boundary. */
  return turnFind.seq;
}

function turnFindDropScopedWalk() {
  /* The source moved under an open walk. Its cursor, counts and scope all
     describe the database being left, so the next segment would be asked of
     a different one and would read as more of the same result. Nothing is
     carried over and the page says why, rather than leaving an answer on
     screen whose source is gone.

     The finder also stands down rather than repainting itself empty: an empty
     result list with a finished walk is rendered as "no turn in this store
     matches", which would be this page answering a question it never asked
     the source now open. */
  var owned = turnFind.active;
  turnFind.seq += 1;
  if (turnFind.abort) { turnFind.abort.abort(); turnFind.abort = null; }
  turnFind.rows = []; turnFind.cursor = null; turnFind.scanned = 0;
  turnFind.matched = 0; turnFind.complete = true; turnFind.facets = null;
  turnFind.stopped = false; turnFind.error = null; turnFind.basis = "";
  turnFind.scope = null;
  turnFind.running = false;
  turnFind.active = false;
  turnFind.sourceMoved = true;
  turnFind.source = turnFindSourceKey();
  turnFindRenderScope();
  turnFindRenderMarkers();
  turnFindRender();
  if (owned) {
    /* Only the pane the search itself was holding: a reader who had navigated
       to a turn keeps what they were reading. */
    var pane = document.getElementById("detail");
    clear(pane);
    pane.appendChild(el("div", "empty", turnFindStatusText()));
  }
}

function turnFindRun(seq, cursor) {
  /* Before the request, because a continuation issued against the source
     now open is the wrong question asked of the wrong database. */
  if (turnFindSourceMoved()) { turnFindDropScopedWalk(); return; }
  turnFind.running = true;
  turnFindRender();
  if (turnFind.abort) { turnFind.abort.abort(); }
  turnFind.abort = (typeof AbortController !== "undefined") ? new AbortController() : null;
  api(turnFindPath(cursor), { signal: turnFind.abort && turnFind.abort.signal }).then(function (page) {
    if (seq !== turnFind.seq) { return; }   /* a later search owns the view */
    /* And after it: the switch can land while this one is in flight. */
    if (turnFindSourceMoved()) { turnFindDropScopedWalk(); return; }
    turnFind.running = false;
    /* Every fetched row is kept. The cursor moves past these rows whether or
       not they are rendered, so a row dropped here could never be asked for
       again. */
    (page.turns || []).forEach(function (row) { turnFind.rows.push(row); });
    /* Segment counts sum; they are not a dataset total until the walk ends. */
    turnFind.matched += page.total_matched || 0;
    turnFind.scanned += page.total_scanned || 0;
    turnFind.complete = !page.scan_truncated;
    turnFind.cursor = page.next_scan_cursor || null;
    turnFind.basis = page.basis || "";
    /* Facets accumulate with the walk for the same reason the totals do: each
       segment counts its own stretch, and a segment's count presented as the
       store's answer would say "0 turns repeated a command" about a store
       whose repeats are simply further down. A null facet stays null -- the
       query could not answer it, and summing would invent a zero. */
    if (page.facets) {
      if (!turnFind.facets) {
        turnFind.facets = page.facets;
      } else {
        Object.keys(page.facets).forEach(function (name) {
          var seen = turnFind.facets[name], now = page.facets[name];
          turnFind.facets[name] =
            (typeof seen === "number" && typeof now === "number") ? seen + now : now;
        });
      }
    }
    turnFindRenderMarkers();
    /* Keep walking only while the answer would otherwise be a premature
       "no matches"; once something is on screen the operator decides. */
    if (!turnFind.complete && !turnFind.stopped && !turnFind.rows.length
        && turnFind.cursor) {
      turnFindRun(seq, turnFind.cursor);
      return;
    }
    turnFindRender();
  }).catch(function (error) {
    if (requestWasAborted(error) || seq !== turnFind.seq) { return; }
    turnFind.running = false;
    turnFind.error = error.message || String(error);
    turnFindRender();
  });
}

/* Who owns #detail. A search walks the store over several round trips, and
   the rail's periodic refresh repaints the pane between them, so claiming the
   navigation token once at the start is not enough -- the finder would be
   declared stale by a refresh it has nothing to do with. Instead it holds the
   pane until the operator navigates away, which is exactly the two gestures
   below: opening a record from the rail, or opening one of its own results. */
function turnFindRelease() { turnFind.active = false; }
document.getElementById("convList")
  .addEventListener("click", turnFindRelease, true);
document.getElementById("recordNav")
  .addEventListener("click", turnFindRelease, true);

function turnFindStart() {
  /* A scope chosen in another source names runs this one may not have, or
     may have under the same labels and different evidence; either way it is
     not the scope the operator picked, so it goes rather than travels. */
  if (turnFind.scope && turnFindSourceMoved()) { turnFind.scope = null; }
  /* The box is read here rather than on every keystroke: typing only edits
     the question, and whichever control asks it searches what the box says. */
  var box = document.getElementById("turnFindText");
  if (box) { turnFind.text = box.value.trim(); }
  turnFind.source = turnFindSourceKey();
  turnFind.active = true;
  turnFindRenderScope();
  turnFindRun(turnFindReset(), null);
}

/* Entry from the run being read, on the source that is SELECTED.

   Carrying a scope ACROSS a source change is a different thing and is not
   offered: the guards above discard an open walk, its cursor and its scope the
   moment the selected source moves, rather than asking the new database for
   the old one's labels.

   What is deferred is a WORKSPACE: its logical experiments span several sealed
   stores and /api/workspace/turns takes one store_id per request, so scoping
   one needs multi-store routing this slice does not build (fix-luut). */
function turnFindEntrySupported() {
  return !(session && session.workspace_mode);
}

function turnFindEntryButton(container, scope, label, help) {
  if (!turnFindEntrySupported()) { return null; }
  var button = el("button", "ghost", label);
  button.type = "button";
  button.title = help;
  button.setAttribute("data-find-scope",
    (scope.attempt === null || scope.attempt === undefined)
      ? (scope.task ? "task" : "experiment") : "attempt");
  button.addEventListener("click", function () { turnFindScopeTo(scope); });
  container.appendChild(button);
  return button;
}

function turnFindScopeTo(scope) {
  turnFind.scope = {
    experiment: scope.experiment || null,
    task: scope.task || null,
    attempt: (scope.attempt === undefined ? null : scope.attempt)
  };
  turnFind.source = turnFindSourceKey();
  /* Whatever is in the box is what the box says the search is; entering a
     scope narrows that search rather than quietly replacing it. */
  var box = document.getElementById("turnFindText");
  if (box) { turnFind.text = box.value.trim(); }
  turnFindStart();
}

function turnFindRenderScope() {
  var box = document.getElementById("turnFindScope");
  if (!box) { return; }
  clear(box);
  box.className = turnFind.scope ? "scoped" : "";
  if (!turnFind.scope) { return; }
  box.appendChild(el("span", null, "Scope: " + turnFindScopeText(turnFind.scope)));
  var drop = el("button", "traceLink", "Clear scope");
  drop.type = "button";
  drop.id = "turnFindScopeClear";
  drop.title = "Search this source's whole store again";
  drop.addEventListener("click", function () {
    turnFind.scope = null;
    turnFindRenderScope();
    turnFindStart();
  });
  box.appendChild(drop);
}

function turnFindStatusText() {
  if (turnFind.error) { return "Search failed: " + turnFind.error; }
  if (turnFind.sourceMoved) {
    return "The evidence source changed, so this search was discarded along "
      + "with its scope. Search again to look in the source now open.";
  }
  if (!turnFind.active) { return ""; }
  var scanned = turnFind.scanned + " turn" + (turnFind.scanned === 1 ? "" : "s") + " scanned";
  if (turnFind.running) {
    return "Searching \u2014 " + scanned + " so far"
      + (turnFind.rows.length ? ", " + turnFind.matched + " matching" : "") + "\u2026";
  }
  if (!turnFind.complete) {
    /* The count is a floor and is written as one: the rest of the store has
       not been looked at, so neither "12 matches" nor "no matches" is true. */
    return (turnFind.matched ? "At least " + turnFind.matched + " matching" : "No match yet")
      + " \u2014 " + scanned + ", " + turnFindScopePhrase("not ")
      + (turnFind.stopped ? " (stopped)" : "");
  }
  return (turnFind.matched
    ? turnFind.matched + " matching turn" + (turnFind.matched === 1 ? "" : "s")
    : "No matching turns") + " \u2014 " + scanned + ", " + turnFindScopePhrase("");
}

function turnFindRenderMarkers() {
  var box = document.getElementById("turnFindMarkers");
  if (!box) { return; }
  clear(box);
  /* The order and the vocabulary are the server's: `facets` arrives keyed in
     MARKER_ORDER once a search has run. Before that, the local labels stand in
     and a test pins them to the server's list so they cannot drift apart. */
  var names = turnFind.facets ? Object.keys(turnFind.facets) : Object.keys(MARKER_LABEL);
  /* Shown alphabetically by the label the operator reads, so a chip is found
     by scanning for its word; the server's order is kept for everything else. */
  names = names.slice().sort(function (a, b) {
    return markerMeta(a).text.localeCompare(markerMeta(b).text);
  });
  names.forEach(function (name) {
    var meta = markerMeta(name);
    var count = turnFind.facets ? turnFind.facets[name] : undefined;
    var chip = el("button", "chipMarker " + meta.tone);
    chip.type = "button";
    chip.setAttribute("aria-pressed", turnFind.markers[name] ? "true" : "false");
    chip.appendChild(el("span", "glyph", meta.glyph));
    var suffix = "";
    if (count === null) {
      /* Null is "this query could not answer that", which a zero would
         misreport as "nobody matched". */
      suffix = " (not counted)";
      chip.classList.add("unknown");
    } else if (typeof count === "number") {
      /* While the walk is unfinished the count is a floor, and is written as
         one. A bare "0" from a partial scan is the same false negative as a
         bare "no matches" from one. */
      suffix = " " + count + (turnFind.complete ? "" : "+");
    }
    chip.appendChild(document.createTextNode(meta.text + suffix));
    chip.title = meta.help
      + (count === null ? " Not counted for this search."
        : (turnFind.complete ? "" : " Counted so far; the store is not fully searched."));
    chip.addEventListener("click", function () {
      turnFind.markers[name] = !turnFind.markers[name];
      /* A chip only edits the question; Find asks it. The server does not
         abandon a superseded scan, so searching per toggle paid for a full
         scan on every chip picked on the way to the one the operator meant. */
      turnFindRenderMarkers();
    });
    box.appendChild(chip);
  });
}

function turnFindRender() {
  var status = document.getElementById("turnFindStatus");
  if (status) { status.textContent = turnFindStatusText(); }
  if (!turnFind.active) { return; }
  turnFind.nav = expNavToken();
  var d = document.getElementById("detail");
  clear(d);
  var card = el("div", "card");
  var heading = turnFind.text || turnFindSelected().length
    ? "Turns matching this search" : "All recorded turns";
  card.appendChild(el("h2", null, heading));
  card.appendChild(el("div", "sub", turnFindStatusText()));
  if (turnFind.scope) {
    /* Beside the count, not in a tooltip: "12 matching turns" under a
       scoped search is a statement about the scope, and a reader who does
       not know the scope is on reads it as one about the store. */
    card.appendChild(el("div", "diagNote",
      "Scoped to " + turnFindScopeText(turnFind.scope)
      + " in the source this page has open. Turns recorded outside it were "
      + "not searched, and these are turns that recorded the marker rather "
      + "than commands blamed for it."));
  }
  if (turnFind.basis === "spans") {
    card.appendChild(el("div", "diagNote",
      "Searched recorded traces. A dispatch whose span was dropped is found "
      + "only when a failure filter is on, which reads the turn record too."));
  }
  turnFind.rows.forEach(function (row) {
    var item = el("div", "listItem");
    item.appendChild(el("div", "title",
      (row.ordinal ? "#" + row.ordinal + " " : "")
      + (policedText(row.user_message) || "(no message)")));
    var sub = el("div", "sub", row.status + " \u00b7 " + fmtTs(row.started_at));
    item.appendChild(sub);
    var chips = el("div");
    appendMarkerChips(chips, row.markers, (row.diagnosis || {}).counts);
    item.appendChild(chips);
    item.title = "open this turn";
    var open = function () { turnFindRelease(); selectTurn(row.turn_key); };
    makeRowActivatable(item, open);
    card.appendChild(item);
  });
  if (!turnFind.rows.length && turnFind.complete && !turnFind.running) {
    card.appendChild(el("div", "empty", turnFind.scope
      ? "No turn in this scope matches. The whole scope was searched."
      : "No turn in this store matches. The whole store was searched."));
  }
  if (turnFind.rows.length && turnFind.complete && !turnFind.running) {
    /* Says the list is all of them, which is only true once the walk ended.
       Before that the continuation control below is what the operator needs. */
    card.appendChild(el("div", "diagNote",
      "Showing all " + turnFind.rows.length + " matching turn"
      + (turnFind.rows.length === 1 ? "" : "s") + "."));
  }
  if (turnFind.running) {
    var stop = el("button", "traceLink", "Stop searching");
    stop.type = "button";
    stop.id = "turnFindStop";
    stop.addEventListener("click", function () {
      turnFind.stopped = true;
      turnFind.seq += 1;          /* the in-flight answer is no longer ours */
      turnFind.running = false;
      turnFindRender();
    });
    card.appendChild(stop);
  } else if (!turnFind.complete && turnFind.cursor) {
    /* One click is one bounded request, not "the rest of the store" -- saying
       the latter would promise a finished walk the click does not deliver. */
    var more = el("button", "traceLink",
      "Keep searching \u2014 next " + TURN_FIND_SCAN + " turns");
    more.type = "button";
    more.id = "turnFindMore";
    more.addEventListener("click", function () {
      turnFind.stopped = false;
      turnFindRun(turnFind.seq, turnFind.cursor);
    });
    card.appendChild(more);
  }
  d.appendChild(card);
}

document.getElementById("turnFind").addEventListener("submit", function (event) {
  event.preventDefault();
  if (turnFind.timer) { clearTimeout(turnFind.timer); turnFind.timer = null; }
  turnFindStart();
  /* Results land in the detail pane, which the popup would otherwise cover;
     focus goes back to the tab so it is not stranded in a closed dialog. */
  setTurnFindOpen(false);
  document.getElementById("turnFindOpen").focus();
});
turnFindRenderMarkers();
turnFindRenderScope();

function setTurnFindOpen(open) {
  var dialog = document.getElementById("turnFindDialog");
  var button = document.getElementById("turnFindOpen");
  if (open) {
    placeTurnFindDialog();
    if (typeof dialog.show === "function") {
      if (!dialog.open) { dialog.show(); }
    } else {
      dialog.setAttribute("open", "");
    }
    button.setAttribute("aria-expanded", "true");
    document.getElementById("turnFindText").focus();
  } else if (dialog.open || dialog.hasAttribute("open")) {
    if (typeof dialog.close === "function") { dialog.close(); }
    else { dialog.removeAttribute("open"); }
    button.setAttribute("aria-expanded", "false");
  }
}
document.getElementById("turnFindOpen").addEventListener("click", function () {
  setTurnFindOpen(!document.getElementById("turnFindDialog").open);
});
document.getElementById("turnFindDialog").addEventListener("close", function () {
  document.getElementById("turnFindOpen").setAttribute("aria-expanded", "false");
});
document.addEventListener("keydown", function (event) {
  if (event.key !== "Escape") { return; }
  var dialog = document.getElementById("turnFindDialog");
  if (!dialog.open) { return; }
  event.preventDefault();
  setTurnFindOpen(false);
}, true);
document.addEventListener("click", function (event) {
  var dialog = document.getElementById("turnFindDialog");
  if (!dialog.open) { return; }
  /* The path, not contains(): a marker chip repaints itself on click and is
     no longer in the dialog by the time this listener runs, which would
     close the popup on the very click that set the filter. */
  var path = event.composedPath ? event.composedPath() : [];
  if (path.indexOf(dialog) >= 0) { return; }
  if (path.indexOf(document.getElementById("turnFindNav")) >= 0) { return; }
  setTurnFindOpen(false);
});

