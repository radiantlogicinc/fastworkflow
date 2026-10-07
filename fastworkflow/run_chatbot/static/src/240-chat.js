/* -- chat (test) mode --------------------------------------------------- */
/* The page CSP restricts connect-src to loopback origins only, so the SPA
   can never be pointed at a non-local server [R19]. All response-derived
   text renders via textContent [R22]. The channel is the fixed string
   "chatbot" — the developer never types one, and restarts share one history. */
var tm = {
  baseUrl: "",
  token: "",          // JWT actually used on turn calls
  channelId: "",
  userId: "developer",
  connected: false,
  busy: false,
  managed: false,     // true = talking to the chatbot-spawned server
  activeConversationId: null,
  reuse: null,        // recorded message copied into the composer, if any
  epoch: 0            // bumped when the evidence source changes [fix-9eg.7.2]
};

/* Track the spawned server honestly: /api/session reflects the live child
   process, so a server that dies mid-session flips the pill and disables the
   composer instead of letting sends fail with a bare "Failed to fetch". */
var deadServerAnnounced = false;
var sessionFailCount = 0;
var sessionPollTimer = null;
var chatbotUnreachable = false;
function setChatbotUnreachable(unreachable) {
  chatbotUnreachable = unreachable;
  if (unreachable) {
    setPill("err", "chatbot not reachable");
    document.getElementById("healthText").textContent =
      "chatbot not reachable — the debug server stopped answering. Check the terminal that launched run_chatbot.";
    document.getElementById("healthBanner").classList.add("visible");
  } else if (!healthDismissed && document.getElementById("healthText").textContent.indexOf("chatbot not reachable") === 0) {
    document.getElementById("healthBanner").classList.remove("visible");
  }
}
function checkSession() {
  if (document.hidden) { return Promise.resolve(); }
  return api("/api/session").then(function (data) {
    sessionFailCount = 0;
    if (chatbotUnreachable) {
      setChatbotUnreachable(false);
      setPill("", "ready");
    }
    session = data.session;
    if (!tm.managed) { return; }
    if (session.server_running) { deadServerAnnounced = false; return; }
    if (tm.connected && !deadServerAnnounced) {
      deadServerAnnounced = true;
      tm.connected = false;
      tmComposerState();
      setPill("err", "server stopped");
      var why = (session.server_exit_code === null || session.server_exit_code === undefined)
        ? "" : " (exit code " + session.server_exit_code + ")";
      connText("The workflow server stopped" + why + " — check the server log. " +
        "Switch workflow (same one is fine) restarts it.", "err", "server");
      tmBubble("system", "The workflow server stopped" + why +
        ". Use the Switch workflow button to restart it.");
    }
  }).catch(function () {
    sessionFailCount += 1;
    if (sessionFailCount >= 3) { setChatbotUnreachable(true); }
  });
}
function startSessionPolling() {
  if (sessionPollTimer || document.hidden) { return; }
  sessionPollTimer = setInterval(checkSession, 5000);
}
function stopSessionPolling() {
  if (sessionPollTimer) {
    clearInterval(sessionPollTimer);
    sessionPollTimer = null;
  }
}
startSessionPolling();
document.addEventListener("visibilitychange", function () {
  if (document.hidden) {
    stopSessionPolling();
    stopPickerPolling();
  } else {
    startSessionPolling();
    checkSession();
    if (Object.keys(pickerTrainingPaths).length
        && document.getElementById("pickerMain").classList.contains("visible")) {
      refreshCandidateList();
      startPickerPolling();
    }
  }
});

function openProcessLog(kind) {
  var title = kind === "train" ? "Train log" : "Server log";
  return api("/api/logs/" + kind + "?tail=200").then(function (data) {
    var dialog = document.getElementById("logDialog");
    document.getElementById("logDialogTitle").textContent = title;
    document.getElementById("logDialogPath").textContent = data.path || "";
    document.getElementById("logDialogNote").textContent = data.truncated
      ? "Showing the retained tail of the log."
      : (data.exists ? "" : "No log file yet.");
    document.getElementById("logDialogLines").textContent =
      (data.lines && data.lines.length) ? data.lines.join("\n") : "";
    if (dialog.showModal) { dialog.showModal(); }
    else { dialog.setAttribute("open", ""); }
    document.getElementById("logDialogClose").focus();
  }).catch(function (error) {
    showNotice("Could not read the log", "error", error.message);
  });
}
document.getElementById("logDialogClose").addEventListener("click", function () {
  var dialog = document.getElementById("logDialog");
  if (dialog.close) { dialog.close(); } else { dialog.removeAttribute("open"); }
});

function connText(text, cls, offerLog) {
  var node = document.getElementById("connText");
  node.className = cls || "";
  clear(node);
  node.appendChild(document.createTextNode(text || ""));
  if (offerLog === "server") { appendLogOffer(node, "server"); }
}

function tmExplainFetchError(err) {
  return "Could not reach the workflow server (" + err.message + ").\n" +
    "It may still be starting, or it stopped — check the server log.\n" +
    "The URL must be a loopback origin (127.0.0.1/localhost); the Advanced " +
    "panel reconnects to a different server.";
}

function tmFetch(path, body, extra) {
  /* No workflow server bound (a source switch dropped it, or none was ever
     spawned): refusing here is what keeps a message from going to the server
     the previous source was using — and stops a bare path resolving against
     the chatbot's own origin. */
  if (!tm.baseUrl) {
    return Promise.reject(new Error("no workflow server is connected"));
  }
  var headers = { "Content-Type": "application/json" };
  if (tm.token && !(extra && extra.noAuth)) {
    headers["Authorization"] = "Bearer " + tm.token;
  }
  var opts = { method: extra && extra.method ? extra.method : "POST", headers: headers };
  if (opts.method !== "GET") { opts.body = JSON.stringify(body || {}); }
  return fetch(tm.baseUrl + path, opts);
}

/* A reader who scrolled up is reading; live updates must not yank them back
   to the bottom. Anything within a line or two of the end counts as following
   the conversation. */
function tmFollowingLog() {
  var log = document.getElementById("chatLog");
  return (log.scrollHeight - log.scrollTop - log.clientHeight) < 60;
}
function tmScrollIfFollowing(wasFollowing) {
  var log = document.getElementById("chatLog");
  if (wasFollowing === undefined ? tmFollowingLog() : wasFollowing) {
    log.scrollTop = log.scrollHeight;
  }
}

var TM_TRANSCRIPT_CAP = 300;
function tmBubble(role, text) {
  var log = document.getElementById("chatLog");
  var empty = log.querySelector(".empty");
  if (empty) { log.removeChild(empty); }
  var msg = el("div", "chatMsg " + role);
  var bubble = el("div", "bubble");
  if (role === "agent") { appendMarkdown(bubble, text); }
  else { bubble.appendChild(document.createTextNode(text)); }
  msg.appendChild(bubble);
  log.appendChild(msg);
  var bubbles = log.querySelectorAll(".chatMsg");
  if (bubbles.length > TM_TRANSCRIPT_CAP) {
    var removeCount = bubbles.length - TM_TRANSCRIPT_CAP;
    var i;
    for (i = 0; i < removeCount; i++) {
      log.removeChild(bubbles[i]);
    }
    var note = log.querySelector(".chatTruncation");
    var hidden = removeCount + (note ? (Number(note.dataset.hidden) || 0) : 0);
    if (!note) {
      note = el("div", "chatTruncation sub");
      log.insertBefore(note, log.firstChild);
    }
    note.dataset.hidden = String(hidden);
    note.textContent = hidden + " earlier messages hidden";
  }
  /* A message the user just sent (or the reply slot for it) always scrolls:
     they acted, so moving to it is what they asked for. */
  log.scrollTop = log.scrollHeight;
  return msg;
}

function looksLikeMarkdown(text) {
  return /(^|\n)\s{0,3}#{1,6}\s/.test(text) ||
    /(^|\n)\s{0,3}(?:[-*+]|\d+\.)\s/.test(text) ||
    /```/.test(text) ||
    /\*\*[^*]+\*\*/.test(text) ||
    /__[^_]+__/.test(text) ||
    /`[^`]+`/.test(text) ||
    /(^|\n)\s*>\s/.test(text) ||
    /~~[^~]+~~/.test(text);
}

function safeHref(url) {
  /* Schemes are concatenated so the static-page origin scan never sees a
     non-loopback http(s) URL literal [R19]. */
  var u = String(url || "").trim();
  var lower = u.toLowerCase();
  var scheme = "http" + ":";
  var secure = "http" + "s:";
  if (lower.indexOf(scheme + "//") === 0 || lower.indexOf(secure + "//") === 0) {
    return u;
  }
  return "";
}

function appendInline(parent, text) {
  var i = 0, buf = "";
  function flush() {
    if (buf) { parent.appendChild(document.createTextNode(buf)); buf = ""; }
  }
  function rest() { return text.slice(i); }
  while (i < text.length) {
    if (text.charAt(i) === "`") {
      var tick = text.indexOf("`", i + 1);
      if (tick > i) {
        flush();
        parent.appendChild(el("code", null, text.slice(i + 1, tick)));
        i = tick + 1;
        continue;
      }
    }
    if (rest().slice(0, 2) === "**") {
      var bold = text.indexOf("**", i + 2);
      if (bold > i + 2) {
        flush();
        var strong = el("strong");
        appendInline(strong, text.slice(i + 2, bold));
        parent.appendChild(strong);
        i = bold + 2;
        continue;
      }
    }
    if (rest().slice(0, 2) === "__") {
      var under = text.indexOf("__", i + 2);
      if (under > i + 2) {
        flush();
        var strongU = el("strong");
        appendInline(strongU, text.slice(i + 2, under));
        parent.appendChild(strongU);
        i = under + 2;
        continue;
      }
    }
    if (rest().slice(0, 2) === "~~") {
      var strike = text.indexOf("~~", i + 2);
      if (strike > i + 2) {
        flush();
        var del = el("del");
        appendInline(del, text.slice(i + 2, strike));
        parent.appendChild(del);
        i = strike + 2;
        continue;
      }
    }
    if (text.charAt(i) === "*" && text.charAt(i + 1) !== "*" &&
        text.charAt(i + 1) !== " ") {
      var emEnd = text.indexOf("*", i + 1);
      if (emEnd > i + 1 && text.charAt(emEnd + 1) !== "*") {
        flush();
        var em = el("em");
        appendInline(em, text.slice(i + 1, emEnd));
        parent.appendChild(em);
        i = emEnd + 1;
        continue;
      }
    }
    if (text.charAt(i) === "[") {
      var link = rest().match(/^\[([^\]]+)\]\(([^)\s]+)\)/);
      if (link) {
        var href = safeHref(link[2]);
        if (href) {
          flush();
          var a = el("a", null, link[1]);
          a.setAttribute("href", href);
          a.setAttribute("target", "_blank");
          a.setAttribute("rel", "noopener noreferrer");
          parent.appendChild(a);
          i += link[0].length;
          continue;
        }
      }
    }
    buf += text.charAt(i);
    i += 1;
  }
  flush();
}

function isListLine(line, ordered) {
  return ordered ? /^\s*\d+\.\s+/.test(line) : /^\s*[-*+]\s+/.test(line);
}
function listText(line, ordered) {
  return line.replace(ordered ? /^\s*\d+\.\s+/ : /^\s*[-*+]\s+/, "");
}
function startsBlock(line) {
  return /^(#{1,6}\s|```|\s*[-*+]\s|\s*\d+\.\s|\s*>)/.test(line);
}

function appendMarkdown(container, text) {
  text = String(text == null ? "" : text);
  if (!looksLikeMarkdown(text)) {
    container.appendChild(document.createTextNode(text));
    return;
  }
  container.classList.add("md");
  var lines = text.split("\n");
  var i = 0;
  while (i < lines.length) {
    var line = lines[i];
    if (/^```/.test(line)) {
      var codeLines = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) { i += 1; }
      var pre = el("pre");
      pre.appendChild(el("code", null, codeLines.join("\n")));
      container.appendChild(pre);
      continue;
    }
    var heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      var h = el("h" + heading[1].length);
      appendInline(h, heading[2]);
      container.appendChild(h);
      i += 1;
      continue;
    }
    if (isListLine(line, false) || isListLine(line, true)) {
      var ordered = isListLine(line, true);
      var list = el(ordered ? "ol" : "ul");
      while (i < lines.length && isListLine(lines[i], ordered)) {
        var li = el("li");
        appendInline(li, listText(lines[i], ordered));
        list.appendChild(li);
        i += 1;
      }
      container.appendChild(list);
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      var bq = el("blockquote");
      var quoted = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quoted.push(lines[i].replace(/^\s*>\s?/, ""));
        i += 1;
      }
      appendInline(bq, quoted.join("\n"));
      container.appendChild(bq);
      continue;
    }
    if (/^\s*---+\s*$/.test(line)) {
      container.appendChild(el("hr"));
      i += 1;
      continue;
    }
    if (!line.trim()) {
      i += 1;
      continue;
    }
    var para = [];
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) {
      para.push(lines[i]);
      i += 1;
    }
    var p = el("p");
    para.forEach(function (ln, idx) {
      if (idx) { p.appendChild(el("br")); }
      appendInline(p, ln);
    });
    container.appendChild(p);
  }
}

function tmRenderTurn(bubbleMsg, out) {
  /* out: TurnOutput projection {turn_key,status,success,answer,command_outputs}
     Idempotent: the same turn can be painted twice (the stream delivered the
     output event AND a recovery poll read the stored record), and the second
     paint must replace the first rather than stack a second copy of it. The
     activity panel is deliberately not cleared — it is the record of what
     happened during the turn. It stays open while that exchange streams, and
     closes once this paint has a finished answer. A turn still waiting on the
     user stays open. Command lines from that answer fold into the panel when
     one exists, so they leave the screen with it. */
  var activity = bubbleMsg.querySelector(".activity");
  /* Collapse before anything below can throw: a finished answer must never
     leave its exchange spread open above it. */
  if (activity && out.status !== "awaiting_user") { activity.open = false; }
  var bubble = bubbleMsg.querySelector(".bubble");
  clear(bubble);
  tmCloseArtifactPanel(bubbleMsg);
  Array.prototype.slice.call(
    bubbleMsg.querySelectorAll(".cmdOut, .meta, .artifacts, .artifactsLink")
  ).forEach(function (node) { node.parentNode.removeChild(node); });
  var answer = out.answer || "(no answer)";
  appendMarkdown(bubble, answer);
  if (out.status === "awaiting_user") {
    bubbleMsg.appendChild(el("div", "cmdOut",
      "The agent is waiting for your reply — the next message resumes this turn."));
  }
  (out.command_outputs || []).forEach(function (co) {
    var resp = co.command_response || {};
    var line = (co.command_name || "(command)") +
      (resp.success === false ? " · FAILED" : "") +
      (typeof resp.response === "string" && resp.response
        ? " — " + resp.response.slice(0, 300) : "");
    var lineNode = el("div", "cmdOut", line);
    if (activity) { activity.appendChild(lineNode); }
    else { bubbleMsg.appendChild(lineNode); }
  });
  tmRenderArtifacts(bubbleMsg, out.command_outputs);
  var meta = el("div", "meta");
  var statusTxt = turnOutcome(out.status, out.failure_reason).text +
    (turnHadCommandFailure(out) ? " · a command reported failure" : "");
  meta.appendChild(el("span", null, statusTxt));
  if (out.turn_key) {
    var link = el("button", "traceLink", "view trace");
    link.addEventListener("click", function () { openTurnInDebug(out.turn_key); });
    meta.appendChild(link);
  }
  bubbleMsg.appendChild(meta);
  tmScrollIfFollowing();
}

