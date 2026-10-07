/* -- chat artifacts [fix-9eg.20.3] -------------------------------------- */
/* Artifacts belong beside the answer, not behind a Debug tab. Rendering
   reuses the debug renderer's safety rules verbatim [R22]: values print via
   textContent, HTML-ish content only ever reaches a sandboxed iframe, and no
   URL outside this chatbot's own artifact route is ever fetched. */

/* Auto-fetching stops here; above it a card offers the content on request so
   a multi-megabyte artifact never blocks the answer it belongs to. */
var TM_ARTIFACT_LAZY_BYTES = 262144;
/* How many restored turns fetch their stored record on sight. Older turns keep
   a button, so reopening a long conversation is not a hundred round trips. */
var TM_RESTORED_ARTIFACT_AUTOLOAD = 10;

function tmArtifactPath(ref) {
  /* The chat's artifacts live in the LIVE workflow's store, which is this
     route's default source. */
  return "/api/artifact/" + encodeURIComponent(ref);
}
function tmArtifactFetch(ref) {
  return fetch(tmArtifactPath(ref), {
    headers: chatbotAuthHeaders()
  });
}
function tmChatApi(path) {
  /* A chat turn's record and artifacts belong to the store that produced
     them, so the chat reads are deliberately unscoped. */
  return fetch(path, { headers: chatbotAuthHeaders() })
    .then(function (r) {
      if (!r.ok) { throw new Error("API " + path + " -> " + r.status); }
      return r.json();
    });
}
function tmArtifactHref(ref) {
  /* <img> and download links cannot carry an Authorization header, so they use
     the same ?token= the page itself was opened with. Same origin only. */
  return tmArtifactPath(ref) + "?token=" + encodeURIComponent(TOKEN);
}

function tmArtifactRef(value) {
  return (value && typeof value === "object" && value.__fw_artifact_ref__)
    ? value : null;
}

function tmArtifactSourceLabel(commandName, index) {
  /* Two commands in one turn may both return "report". The source keeps them
     apart; the ordinal keeps two calls of the SAME command apart. */
  return "from " + (commandName || "(command)") + " #" + (index + 1);
}

function tmArtifactCard(key, meta) {
  var box = el("div", "artifact");
  var head = el("div", "aHead");
  head.appendChild(el("span", "aKey", key));
  if (meta) { head.appendChild(el("span", "aMeta", meta)); }
  box.appendChild(head);
  return box;
}

function tmArtifactActions(box, ref) {
  var actions = el("div", "aActions");
  var open = el("a", null, "open / download");
  open.href = tmArtifactHref(ref);
  open.target = "_blank";
  open.rel = "noopener noreferrer";
  actions.appendChild(open);
  box.appendChild(actions);
  return actions;
}

function tmRenderArtifactBody(box, contentType, text) {
  var ctype = (contentType || "").toLowerCase();
  var htmlish = ctype.indexOf("text/html") === 0 ||
                ctype.indexOf("image/svg") === 0 ||
                ctype.indexOf("application/xhtml") === 0 ||
                looksHtml(text);
  if (htmlish) {
    box.appendChild(sandboxedFrame(text));
    var src = el("details");
    src.appendChild(el("summary", null, "View source (as text)"));
    src.appendChild(el("pre", "json", text));
    box.appendChild(src);
    return;
  }
  box.appendChild(el("pre", "json", pretty(text)));
}

function tmLoadOffloadedArtifact(box, placeholder, value) {
  var ref = value.__fw_artifact_ref__;
  placeholder.textContent = "loading…";
  tmArtifactFetch(ref)
    .then(function (r) {
      if (r.status === 404) {
        throw { missing: true };
      }
      if (!r.ok) { throw new Error("HTTP " + r.status); }
      var ctype = (r.headers.get("Content-Type") || "").toLowerCase();
      return r.text().then(function (text) { return { ctype: ctype, text: text }; });
    })
    .then(function (got) {
      placeholder.parentNode.removeChild(placeholder);
      tmRenderArtifactBody(box, got.ctype, got.text);
      tmArtifactActions(box, ref);
      tmScrollIfFollowing();
    })
    .catch(function (e) {
      if (e && e.missing) {
        placeholder.textContent =
          "this artifact is no longer in the store (pruned or never recorded).";
        return;
      }
      placeholder.textContent = "could not load this artifact: " +
        (e && e.message ? e.message : "unknown error");
      var retry = el("button", "traceLink", "retry");
      retry.addEventListener("click", function () {
        if (retry.parentNode) { retry.parentNode.removeChild(retry); }
        tmLoadOffloadedArtifact(box, placeholder, value);
      });
      placeholder.parentNode.appendChild(retry);
    });
}

function tmOffloadedArtifactNode(key, value, source) {
  var ref = value.__fw_artifact_ref__;
  var ctype = (value.content_type || "").toLowerCase();
  var size = typeof value.size === "number" ? value.size : null;
  var box = tmArtifactCard(key, [
    "offloaded",
    value.content_type || "unknown type",
    size === null ? "size unknown" : fmtCount(size) + " bytes",
    source
  ].join(" · "));

  if (ctype.indexOf("image/") === 0 && ctype.indexOf("image/svg") !== 0) {
    /* A raster image is shown by the browser, not read into the page. */
    var img = document.createElement("img");
    img.src = tmArtifactHref(ref);
    img.alt = key;
    box.appendChild(img);
    tmArtifactActions(box, ref);
    return box;
  }
  if (size !== null && size > TM_ARTIFACT_LAZY_BYTES) {
    var note = el("div", "aMeta",
      "large artifact — not loaded automatically so the answer stays responsive.");
    box.appendChild(note);
    var load = el("button", "traceLink", "load content");
    load.addEventListener("click", function () {
      box.removeChild(load);
      tmLoadOffloadedArtifact(box, note, value);
    });
    box.appendChild(load);
    tmArtifactActions(box, ref);
    return box;
  }
  var placeholder = el("div", "aMeta", "loading…");
  box.appendChild(placeholder);
  tmLoadOffloadedArtifact(box, placeholder, value);
  return box;
}

function tmArtifactsIn(commandOutputs, callIds) {
  /* [{key, value, source}] for every artifact this turn actually returned.
     With `callIds`, only the outputs whose command_call_id is in it; the
     source ordinal stays the output's place in the whole turn. */
  var found = [];
  (commandOutputs || []).forEach(function (co, index) {
    if (callIds && !(co && co.command_call_id && callIds[co.command_call_id] === true)) { return; }
    var resp = (co && co.command_response) || {};
    var artifacts = resp.artifacts || {};
    Object.keys(artifacts).forEach(function (key) {
      found.push({
        key: key,
        value: artifacts[key],
        source: tmArtifactSourceLabel(co && co.command_name, index)
      });
    });
  });
  return found;
}

function tmRenderArtifacts(bubbleMsg, commandOutputs, note) {
  var viewer = tmArtifactViewer(tmArtifactsIn(commandOutputs), note, {
    owner: bubbleMsg,
    label: "Artifacts of this answer",
    anchor: function () { return bubbleMsg.querySelector(".bubble") || bubbleMsg; },
    frame: function () { return document.getElementById("chatLog"); }
  });
  if (!viewer) { return null; }
  tmAppendBeforeMeta(bubbleMsg, viewer.linkRow);
  tmAppendBeforeMeta(bubbleMsg, viewer.panel);
  return viewer.panel;
}

function tmArtifactViewer(items, note, place) {
  /* The answer stays the thing on screen. Its artifacts sit behind one link
     under it, and open in a panel beside it that shows one at a time. Every
     card is still built (and fetched, within the lazy-load rules) up front, so
     stepping through them never waits on the network.

     `place` says where that is: `owner` is the element the link and panel are
     put in (one open panel per owner), `anchor()` the element the panel sits
     beside, and `frame()` the scroll box it must stay inside. With
     `placement: "below"` the panel hangs under the anchor instead, starting
     at its left edge and stopping short of the record navigator. */
  if (!items.length) { return null; }
  var owner = place.owner;
  tmArtifactPanelSeq += 1;
  var panelId = "artifactPanel" + tmArtifactPanelSeq;

  var link = el("button", "artifactsLink",
    (items.length === 1 ? "1 artifact" : items.length + " artifacts"));
  link.type = "button";
  link.setAttribute("aria-expanded", "false");
  link.setAttribute("aria-controls", panelId);
  var linkRow = el("div", "artifactsLinkRow");
  linkRow.appendChild(link);

  var wrap = el("div", "artifacts");
  wrap.id = panelId;
  wrap.hidden = true;
  wrap.setAttribute("role", "dialog");
  wrap.setAttribute("aria-label", place.label);
  var head = el("div", "aPanelHead");
  head.appendChild(el("div", "aTitle", "Artifacts"));
  var position = el("span", "aPosition");
  position.setAttribute("aria-live", "polite");
  head.appendChild(position);
  var close = el("button", "aClose", "×");
  close.type = "button";
  close.setAttribute("aria-label", "Close artifacts");
  close.title = "Close";
  head.appendChild(close);
  wrap.appendChild(head);
  if (note) { wrap.appendChild(el("div", "aMeta aNote", note)); }

  var body = el("div", "aPanelBody");
  var stage = el("div", "aStage");
  body.appendChild(stage);
  var nav = el("nav", "aNav");
  nav.setAttribute("aria-label", "Move through artifacts");
  var moves = {};
  TM_ARTIFACT_MOVES.forEach(function (move) {
    var button = el("button");
    button.type = "button";
    button.setAttribute("aria-label", move.label);
    button.title = move.label;
    button.appendChild(document.getElementById("artifactNavIcons").content
      .querySelector('[data-move="' + move.id + '"]').cloneNode(true));
    button.addEventListener("click", function () { show(move.to(current, cards.length)); });
    nav.appendChild(button);
    moves[move.id] = button;
  });
  body.appendChild(nav);
  wrap.appendChild(body);

  var cards = items.map(function (item) {
    var ref = tmArtifactRef(item.value);
    if (ref) {
      /* offloaded: fetch from the artifact endpoint */
      return tmOffloadedArtifactNode(item.key, item.value, item.source);
    }
    return artifactNode(item.key, item.value, "inline · " + item.source);
  });
  cards.forEach(function (card) { stage.appendChild(card); });

  var current = 0;
  function show(index) {
    current = Math.max(0, Math.min(cards.length - 1, index));
    cards.forEach(function (card, i) { card.hidden = i !== current; });
    position.textContent = (current + 1) + " of " + cards.length;
    moves.first.disabled = moves.prev.disabled = current === 0;
    moves.next.disabled = moves.last.disabled = current === cards.length - 1;
    stage.scrollTop = 0;
  }
  show(0);

  link.addEventListener("click", function () {
    if (wrap.hidden) { tmOpenArtifactPanel(owner, wrap, link, place); }
    else { tmCloseArtifactPanel(owner); }
  });
  close.addEventListener("click", function () {
    tmCloseArtifactPanel(owner);
    link.focus();
  });
  wrap.addEventListener("keydown", function (event) {
    var to = null;
    if (event.key === "Home") { to = 0; }
    else if (event.key === "End") { to = cards.length - 1; }
    else if (event.key === "ArrowUp" || event.key === "PageUp") { to = current - 1; }
    else if (event.key === "ArrowDown" || event.key === "PageDown") { to = current + 1; }
    else if (event.key === "Escape") {
      tmCloseArtifactPanel(owner);
      link.focus();
      event.preventDefault();
      return;
    }
    if (to === null || event.target.closest(".aStage")) { return; }
    show(to);
    event.preventDefault();
  });

  return {link: link, linkRow: linkRow, panel: wrap};
}

var tmArtifactPanelSeq = 0;
/* Top, up, down, bottom: the same arrow language as the record navigator. */
var TM_ARTIFACT_MOVES = [
  {id: "first", label: "First artifact", to: function () { return 0; }},
  {id: "prev", label: "Previous artifact", to: function (at) { return at - 1; }},
  {id: "next", label: "Next artifact", to: function (at) { return at + 1; }},
  {id: "last", label: "Last artifact", to: function (at, count) { return count - 1; }}
];
/* One panel is open at a time, whichever answer it belongs to. */
var tmOpenArtifacts = null;

function tmPlaceArtifactPanel() {
  /* Fixed to the viewport so the chat log's scroll box cannot clip it. It
     takes the room right of the answer, widening over the answer's right edge
     when that room is too narrow to read an artifact in, and its top follows
     the answer while the answer is on screen. */
  if (!tmOpenArtifacts) { return; }
  if (tmOpenArtifacts.place.placement === "below") {
    tmPlaceArtifactPanelBelow(tmOpenArtifacts.panel, tmOpenArtifacts.place);
    return;
  }
  var panel = tmOpenArtifacts.panel;
  var log = tmOpenArtifacts.place.frame().getBoundingClientRect();
  var answer = tmOpenArtifacts.place.anchor().getBoundingClientRect();
  var gap = 12;
  var room = log.right - answer.right - gap * 2;
  var width = Math.min(640, Math.max(room, Math.min(480, log.width - gap * 2)));
  var left = room >= width ? answer.right + gap : log.right - gap - width;
  var top = Math.max(log.top + gap, Math.min(answer.top, log.bottom - 260));
  panel.style.left = Math.max(log.left + gap, left) + "px";
  panel.style.top = top + "px";
  panel.style.width = width + "px";
  panel.style.maxHeight = Math.max(200, log.bottom - gap - top) + "px";
}

var TM_RECORD_NAV_TABS = ["recordNav", "turnFindNav"];

function tmRecordNavLeft() {
  /* The left edge of whichever window-edge navigator tabs are showing, so a
     panel's own arrows never sit against the record navigator's arrows. */
  var left = Infinity;
  TM_RECORD_NAV_TABS.forEach(function (id) {
    var tab = document.getElementById(id);
    if (!tab) { return; }
    var box = tab.getBoundingClientRect();
    if (box.width > 0 && box.height > 0) { left = Math.min(left, box.left); }
  });
  return left;
}

function tmPlaceArtifactPanelBelow(panel, place) {
  /* Under the link and flush with its left edge, kept inside the frame and a
     clear gap short of the record navigator. When the room under the link is
     short and there is more above it, the panel opens upward instead. */
  var frame = place.frame().getBoundingClientRect();
  var anchor = place.anchor().getBoundingClientRect();
  var gap = 12, railGap = 24, offset = 6;
  var edge = Math.min(frame.right - gap, tmRecordNavLeft() - railGap);
  var minLeft = frame.left + gap;
  var left = Math.max(minLeft, anchor.left);
  var width = Math.min(640, edge - left);
  if (width < 360) {
    width = Math.max(240, Math.min(640, edge - minLeft));
    left = Math.max(minLeft, edge - width);
  }
  var floor = frame.bottom - gap, ceiling = frame.top + gap;
  var below = floor - (anchor.bottom + offset);
  var above = (anchor.top - offset) - ceiling;
  panel.style.left = left + "px";
  panel.style.width = width + "px";
  if (below >= 260 || below >= above) {
    var top = Math.max(ceiling, Math.min(anchor.bottom + offset, floor - 200));
    panel.style.top = top + "px";
    panel.style.bottom = "";
    panel.style.maxHeight = Math.max(200, floor - top) + "px";
  } else {
    var bottomEdge = Math.min(floor, Math.max(ceiling + 200, anchor.top - offset));
    panel.style.top = "auto";
    panel.style.bottom = (window.innerHeight - bottomEdge) + "px";
    panel.style.maxHeight = Math.max(200, bottomEdge - ceiling) + "px";
  }
}

function tmOpenArtifactPanel(bubbleMsg, panel, link, place) {
  if (tmOpenArtifacts) { tmCloseArtifactPanel(tmOpenArtifacts.msg); }
  tmOpenArtifacts = {msg: bubbleMsg, panel: panel, link: link, place: place};
  panel.hidden = false;
  link.setAttribute("aria-expanded", "true");
  tmPlaceArtifactPanel();
  var first = panel.querySelector(".aNav button:not(:disabled)")
    || panel.querySelector(".aClose");
  if (first) { first.focus(); }
}

function tmCloseArtifactPanel(bubbleMsg) {
  if (!tmOpenArtifacts || tmOpenArtifacts.msg !== bubbleMsg) { return; }
  tmOpenArtifacts.panel.hidden = true;
  tmOpenArtifacts.link.setAttribute("aria-expanded", "false");
  tmOpenArtifacts = null;
}

function tmCloseArtifactPanelIn(container) {
  /* A panel whose owner is about to be cleared away must not stay the open
     one: the next placement would measure detached nodes. */
  if (tmOpenArtifacts && container.contains(tmOpenArtifacts.msg)) {
    tmCloseArtifactPanel(tmOpenArtifacts.msg);
  }
}

document.getElementById("chatLog").addEventListener("scroll", tmPlaceArtifactPanel);
document.getElementById("detail").addEventListener("scroll", tmPlaceArtifactPanel);
window.addEventListener("resize", tmPlaceArtifactPanel);

function tmAppendBeforeMeta(bubbleMsg, node) {
  /* The status/"view trace" line stays last, so anything added after a turn
     was painted lands above it rather than below the footer. */
  var meta = bubbleMsg.querySelector(".meta");
  if (meta) { bubbleMsg.insertBefore(node, meta); }
  else { bubbleMsg.appendChild(node); }
}

