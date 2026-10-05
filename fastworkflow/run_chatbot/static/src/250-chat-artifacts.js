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

function tmArtifactsIn(commandOutputs) {
  /* [{key, value, source}] for every artifact this turn actually returned. */
  var found = [];
  (commandOutputs || []).forEach(function (co, index) {
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
  /* The answer stays the thing on screen. Its artifacts sit behind one link
     under it, and open in a panel beside it that shows one at a time. Every
     card is still built (and fetched, within the lazy-load rules) up front, so
     stepping through them never waits on the network. */
  var items = tmArtifactsIn(commandOutputs);
  if (!items.length) { return null; }
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
  wrap.setAttribute("aria-label", "Artifacts of this answer");
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
      return tmOffloadedArtifactNode(item.key, item.value, item.source);
    }
    var env = captureEnvelope(item.value);
    if (env) {
      /* Withheld by the capture policy: say so where the content would be,
         so a digest never reads as the artifact it stands in for. */
      var box = tmArtifactCard(item.key, "withheld · " + item.source);
      appendPoliced(box, item.value);
      return box;
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
    if (wrap.hidden) { tmOpenArtifactPanel(bubbleMsg, wrap, link); }
    else { tmCloseArtifactPanel(bubbleMsg); }
  });
  close.addEventListener("click", function () {
    tmCloseArtifactPanel(bubbleMsg);
    link.focus();
  });
  wrap.addEventListener("keydown", function (event) {
    var to = null;
    if (event.key === "Home") { to = 0; }
    else if (event.key === "End") { to = cards.length - 1; }
    else if (event.key === "ArrowUp" || event.key === "PageUp") { to = current - 1; }
    else if (event.key === "ArrowDown" || event.key === "PageDown") { to = current + 1; }
    else if (event.key === "Escape") {
      tmCloseArtifactPanel(bubbleMsg);
      link.focus();
      event.preventDefault();
      return;
    }
    if (to === null || event.target.closest(".aStage")) { return; }
    show(to);
    event.preventDefault();
  });

  tmAppendBeforeMeta(bubbleMsg, linkRow);
  tmAppendBeforeMeta(bubbleMsg, wrap);
  return wrap;
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
  var panel = tmOpenArtifacts.panel;
  var log = document.getElementById("chatLog").getBoundingClientRect();
  var answer = (tmOpenArtifacts.msg.querySelector(".bubble")
    || tmOpenArtifacts.msg).getBoundingClientRect();
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

function tmOpenArtifactPanel(bubbleMsg, panel, link) {
  if (tmOpenArtifacts) { tmCloseArtifactPanel(tmOpenArtifacts.msg); }
  tmOpenArtifacts = {msg: bubbleMsg, panel: panel, link: link};
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

document.getElementById("chatLog").addEventListener("scroll", tmPlaceArtifactPanel);
window.addEventListener("resize", tmPlaceArtifactPanel);

function tmAppendBeforeMeta(bubbleMsg, node) {
  /* The status/"view trace" line stays last, so anything added after a turn
     was painted lands above it rather than below the footer. */
  var meta = bubbleMsg.querySelector(".meta");
  if (meta) { bubbleMsg.insertBefore(node, meta); }
  else { bubbleMsg.appendChild(node); }
}

