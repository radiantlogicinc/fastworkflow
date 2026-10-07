/* -- live interactions [fix-9eg.20.2] ----------------------------------- */
/* The activity panel shows the public exchange between the agent and the
   workflow — the command it asked for and the response it got back — as the
   turn runs. It shows nothing the turn endpoints do not already return: the
   agent's private reasoning is not part of the trace contract and is never
   requested, fetched or rendered here. */

function tmActivityPanel(bubbleMsg) {
  var box = el("details", "activity");
  box.open = true;
  var summary = el("summary");
  var label = el("span", null, "Activity");
  var count = el("span", "actCount", "no steps yet");
  var state = el("span", "actState", "· running");
  /* Text, not colour, carries the state, and it is announced politely so a
     screen-reader user is told without losing their place. */
  state.setAttribute("role", "status");
  state.setAttribute("aria-live", "polite");
  summary.appendChild(label);
  summary.appendChild(count);
  summary.appendChild(state);
  box.appendChild(summary);
  var list = el("ol", "actList");
  box.appendChild(list);
  /* Above the answer: the exchange is what led to it, so it reads first. */
  var bubble = bubbleMsg.querySelector(".bubble");
  if (bubble) { bubbleMsg.insertBefore(box, bubble); }
  else { bubbleMsg.appendChild(box); }
  var steps = 0;

  function note(text) {
    var following = tmFollowingLog();
    box.appendChild(el("div", "actNote", text));
    tmScrollIfFollowing(following);
  }

  return {
    node: box,
    step: function (trace) {
      var following = tmFollowingLog();
      steps += 1;
      var row = el("li", "actRow");
      var head = el("div", "actHead");
      var toWorkflow = trace.direction === "agent_to_workflow";
      head.appendChild(el("span", "actWho",
        toWorkflow ? "agent → workflow" : "workflow → agent"));
      if (trace.command_name) {
        head.appendChild(el("span", "actCmd", trace.command_name));
      }
      if (trace.success === true) { head.appendChild(el("span", null, "· ok")); }
      if (trace.success === false) { head.appendChild(el("span", null, "· FAILED")); }
      row.appendChild(head);
      var body = toWorkflow ? trace.raw_command : trace.response_text;
      if (body === null || body === undefined || body === "") {
        if (trace.parameters) { body = pretty(trace.parameters); }
      }
      if (body !== null && body !== undefined && body !== "") {
        var text = el("div", "actText");
        text.appendChild(document.createTextNode(String(body)));
        row.appendChild(text);
      }
      list.appendChild(row);
      count.textContent = steps === 1 ? "1 step" : steps + " steps";
      tmScrollIfFollowing(following);
    },
    setState: function (text) { state.textContent = "· " + text; },
    note: note
  };
}

/* Frames arrive over a single body in whichever framing the session was
   initialized with, which the response states in X-FW-Stream-Format. Both
   framings are read here: a session restored as SSE would otherwise lose its
   live interactions, and silently re-negotiating a persisted session's format
   to suit this page is not ours to do.

   Either way a chunk can split a frame in half, so the tail is held back until
   its delimiter arrives, and a frame whose seq was already rendered is dropped
   rather than rendered twice — the deduplication rule the endpoint documents.

   NDJSON carries the whole envelope per line. SSE carries `id:` (the seq),
   `event:` (the type) and `data:` (the payload alone) per block, so the
   envelope is rebuilt from the block; its turn identity comes from the
   X-FW-Turn-Key header the caller already read. */
function tmFrameReader(streamFormat, onFrame, onBadFrame) {
  var buffer = "";
  var seen = {};
  var sse = streamFormat === "sse";
  var delimiter = sse ? "\n\n" : "\n";

  function deliver(frame) {
    if (frame.seq !== null && frame.seq !== undefined) {
      if (seen[frame.seq]) { return; }
      seen[frame.seq] = true;
    }
    onFrame(frame);
  }

  function takeNdjson(line) {
    if (!line.trim()) { return; }
    var frame;
    try { frame = JSON.parse(line); }
    catch (e) { if (onBadFrame) { onBadFrame(line); } return; }
    deliver(frame);
  }

  function takeSse(block) {
    if (!block.trim()) { return; }
    var type = null, id = null, data = [];
    block.split("\n").forEach(function (line) {
      if (line.charAt(0) === ":") { return; }        /* SSE comment/heartbeat */
      var colon = line.indexOf(":");
      if (colon < 0) { return; }
      var field = line.slice(0, colon);
      var value = line.slice(colon + 1);
      if (value.charAt(0) === " ") { value = value.slice(1); }
      if (field === "event") { type = value; }
      else if (field === "id") { id = value; }
      else if (field === "data") { data.push(value); }
    });
    if (type === null && !data.length) { return; }
    var payload;
    try { payload = JSON.parse(data.join("\n")); }
    catch (e) { if (onBadFrame) { onBadFrame(block); } return; }
    deliver({
      type: type,
      seq: id === null || id === "" ? null : Number(id),
      turn_key: null,
      logical_turn_key: null,
      data: payload
    });
  }

  var take = sse ? takeSse : takeNdjson;
  return {
    push: function (chunk) {
      buffer += chunk;
      var parts = buffer.split(delimiter);
      buffer = parts.pop();
      parts.forEach(take);
    },
    end: function () { take(buffer); buffer = ""; }
  };
}

/* One decoded frame, applied to the live turn's UI. Separate from
   tmStreamTurn so the mapping from the server's event types to what a reader
   sees can be driven by real recorded frames in a browser test, rather than
   only through a live socket. */
function tmApplyStreamFrame(frame, activity, sink) {
  if (frame.type === "trace") { activity.step(frame.data || {}); return; }
  if (frame.type === "output") { sink.output(frame.data || {}); return; }
  if (frame.type === "timeout") {
    /* Non-terminal: the delivery deadline passed, the turn did not. Say so and
       keep reading — its output is still coming, and resubmitting would start
       a second turn. */
    activity.setState("still working (past the " +
      ((frame.data && frame.data.timeout_seconds) || "requested") +
      "s deadline)");
    activity.note((frame.data && frame.data.detail) ||
      "The turn passed its delivery deadline and is still running.");
    return;
  }
  if (frame.type === "error") {
    var detail = (frame.data && frame.data.detail) || "the turn reported an error";
    activity.note("Server reported: " + detail);
    sink.error(detail);
  }
}

function tmStreamTurn(text, pending) {
  /* One live turn over /invoke_agent_stream. The execution key arrives on the
     response head, so a body that dies mid-turn is recovered by polling that
     key — never by submitting the query again, which would be a second turn
     and a second bill. */
  var activity = tmActivityPanel(pending);
  var turnKey = null;
  /* The recovery handle that outlives the execution: a chat execution leaves
     the registry as soon as it retires, so its key stops resolving, while the
     logical key reads the stored record. NDJSON frames carry it from the
     first one; an SSE body does not, and recovery there starts from the
     header key and adopts the logical key from the first poll answer. */
  var logicalKey = null;
  var finished = false;
  var lastError = null;
  /* The source this turn belongs to; see tmPollTurn. */
  var epoch = tm.epoch;

  function done() {
    if (tm.epoch !== epoch) { return; }
    tm.busy = false; tmComposerState();
  }

  function finalize(out) {
    if (tm.epoch !== epoch) { return; }
    if (out.logical_turn_key) { out.turn_key = out.logical_turn_key; }
    finished = true;
    activity.setState(
      out.status === "awaiting_user" ? "waiting for your reply"
        : (out.success ? "completed" : "failed"));
    tmRenderTurn(pending, out);
    done();
  }

  function recover(reason) {
    /* The body is gone but the turn is not: it owns its own lifecycle server
       side. Reconcile against the stored record rather than inventing the
       events this client never saw. */
    if (finished || tm.epoch !== epoch) { return; }
    var key = logicalKey || turnKey;
    if (!key) {
      /* No key at all: the request failed before the server named a turn. It
         may nonetheless have started one. Do NOT promise that resending is
         free — a resend only rejoins while that turn is still running, and
         runs the command again once it has finished. */
      tmRenderError(pending, reason + " The server may still have started a " +
        "turn for this message. Reload the conversation to see whether it " +
        "landed before sending it again.");
      activity.setState("disconnected");
      done();
      return;
    }
    activity.setState("disconnected — recovering");
    activity.note(reason + " Recovering the result from the server's record " +
      "(the activity above may be incomplete; the stored result is " +
      "authoritative).");
    tmPollTurn(key, pending, null, activity);
  }

  activity.setState("running");
  tmFetch("/invoke_agent_stream", { user_query: text, timeout_seconds: 300 })
    .then(function (r) {
      turnKey = r.headers.get("X-FW-Turn-Key") || null;
      if (r.status === 202) {
        /* This exact query is already in flight (a retry or a second tab).
           No second run was started; attach by polling its key. */
        return r.json().then(function (data) {
          turnKey = data.logical_turn_key || data.turn_key || turnKey;
          activity.setState("already running — attaching");
          activity.note("This message was already being processed; showing " +
            "that turn's result rather than running it twice.");
          tmPollTurn(turnKey, pending, null, activity);
        });
      }
      if (r.status === 409) {
        return r.json().catch(function () { return {}; }).then(function (data) {
          activity.setState("refused");
          tmRenderError(pending, "Another turn is already in progress on this " +
            "channel (HTTP 409" +
            (data.turn_key ? ", turn " + data.turn_key : "") +
            "). Wait for it to finish.");
          done();
        });
      }
      if (r.status === 401) {
        activity.setState("refused");
        tmRenderError(pending, "The server rejected the bearer token (HTTP 401). " +
          "Reconnect (Advanced panel) to mint new tokens.");
        done();
        return null;
      }
      if (!r.ok || !r.body || !r.body.getReader) {
        /* No readable body (an error response, or an environment without
           streaming): fall back to the same recovery path. */
        return r.text().then(function (body) {
          activity.setState("no live stream");
          recover("The server did not stream this turn (HTTP " + r.status +
            (body ? ": " + body.slice(0, 200) : "") + ").");
        });
      }

      /* Read the framing the server says it sent, not the one we hoped for:
         the session's format was fixed at /initialize and is not this page's
         to renegotiate. */
      var reader = r.body.getReader();
      var decoder = new TextDecoder();
      var frames = tmFrameReader(
        r.headers.get("X-FW-Stream-Format") || "ndjson",
        function (frame) {
          if (frame.turn_key) { turnKey = frame.turn_key; }
          if (frame.logical_turn_key) { logicalKey = frame.logical_turn_key; }
          tmApplyStreamFrame(frame, activity, {
            output: finalize,
            error: function (detail) { lastError = detail; }
          });
        },
        function () {
          activity.note("Skipped an unreadable frame from the server.");
        }
      );

      function pump() {
        return reader.read().then(function (chunk) {
          if (tm.epoch !== epoch) {
            /* The source changed under this turn: stop reading its body
               rather than painting it into somebody else's chat. */
            if (reader.cancel) {
              /* Cancel can reject after the stream is already closed. */
              reader.cancel().catch(function () {});
            }
            return;
          }
          if (chunk.done) {
            frames.end();
            if (!finished) {
              if (turnKey) {
                activity.setState("stream ended early — recovering");
                tmPollTurn(turnKey, pending, null, activity);
              } else {
                activity.setState("failed");
                tmRenderError(pending, lastError ||
                  "The stream ended before the turn produced an answer.");
                done();
              }
            }
            return;
          }
          frames.push(decoder.decode(chunk.value, { stream: true }));
          return pump();
        });
      }
      return pump();
    })
    .catch(function (err) {
      recover("Lost the connection to the workflow server (" +
        (err && err.message ? err.message : "unknown error") + ").");
    });
}

function tmPollTurn(turnKey, bubbleMsg, deadlineMs, activity) {
  /* A 202 handed back an in-flight execution, or a stream body died: poll
     GET /turns/{key}. This never re-submits the command — it reads the
     execution the server already owns, so a disconnect costs a read, not a
     second run. */
  var deadline = Date.now() + (deadlineMs || 300000);
  /* A turn belongs to the source it was started under: if that source is gone
     the poll stops, rather than reading (or reporting) into a page that has
     moved on. */
  var epoch = tm.epoch;
  /* A completed turn moves from the registry to the store, and for a moment
     neither answers. Tolerate that gap, but not forever: a 404 that persists
     past this window is a turn the server no longer has. */
  var missingUntil = null;
  /* An EXECUTION key only resolves while the registry holds the execution: a
     chat turn is not a retained kind, so that key 404s the instant the turn
     retires. Every answer carries `logical_turn_key`, which is what the store
     is keyed by, so switch to it as soon as the server names it — otherwise a
     poll that started one second before the turn ended would report a
     perfectly good answer as lost. */
  function adoptDurableKey(data) {
    if (data.logical_turn_key && data.logical_turn_key !== turnKey) {
      turnKey = data.logical_turn_key;
    }
  }
  function settle() {
    if (tm.epoch !== epoch) { return; }
    tm.busy = false; tmComposerState();
  }
  function fail(text, state) {
    if (tm.epoch !== epoch) { return; }
    if (activity) { activity.setState(state || "failed"); }
    tmRenderError(bubbleMsg, text);
    settle();
  }
  function poll() {
    if (tm.epoch !== epoch) { return; }
    if (Date.now() > deadline) {
      fail("Timed out waiting for the turn to finish (key " + turnKey + ").",
        "timed out");
      return;
    }
    tmFetch("/turns/" + encodeURIComponent(turnKey), null, { method: "GET" })
      .then(function (r) {
        if (r.status === 404) { return null; }         /* registry→store gap */
        if (!r.ok) { throw new Error("HTTP " + r.status); }
        return r.json();
      })
      .then(function (data) {
        if (tm.epoch !== epoch) { return; }
        if (data === null) {
          if (missingUntil === null) { missingUntil = Date.now() + 20000; }
          if (Date.now() > missingUntil) {
            fail("The server no longer has this turn (key " + turnKey +
              "). It may have been lost in a restart; nothing was re-run.",
              "lost");
            return;
          }
          setTimeout(poll, 1000);
          return;
        }
        missingUntil = null;
        adoptDurableKey(data);
        if (data.exec_state === "done") {
          /* the observability DB is keyed by the LOGICAL turn key */
          if (data.logical_turn_key) { data.turn_key = data.logical_turn_key; }
          if (data.error && !data.status) {
            fail("The turn failed on the server: " + data.error);
            return;
          }
          if (activity) {
            activity.setState(
              data.status === "awaiting_user" ? "waiting for your reply"
                : (data.success ? "completed (from the stored record)"
                                : "failed (from the stored record)"));
          }
          tmRenderTurn(bubbleMsg, data);
          settle();
        } else if (data.exec_state === "lost") {
          fail("The server restarted and lost this turn (key " + turnKey + ").",
            "lost");
        } else {
          if (activity) { activity.setState("running (reading the server's record)"); }
          setTimeout(poll, 1000);
        }
      })
      .catch(function (err) {
        fail(tmExplainFetchError(err));
      });
  }
  setTimeout(poll, 1000);
}

function tmRenderError(bubbleMsg, text) {
  bubbleMsg.className = "chatMsg system";
  var bubble = bubbleMsg.querySelector(".bubble");
  clear(bubble);
  bubble.appendChild(document.createTextNode(text));
}

/* -- reusing a recorded message [fix-9eg.7.4] --------------------------- */
/* Copying a recorded message into the composer is a convenience, not a
   replay: the text lands in the live chat session as ordinary input, under
   whatever workflow and settings are current, and nothing runs until the
   person presses Send. tm.reuse only remembers where the text came from so
   the turn it eventually produces can say so. */
function tmSizeComposer() {
  var input = document.getElementById("chatInput");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

function tmShowReuseNotice(text) {
  var bar = document.getElementById("reuseNotice");
  document.getElementById("reuseNoticeText").textContent = text;
  bar.className = "visible";
}

function tmClearReuse() {
  tm.reuse = null;
  document.getElementById("reuseNotice").className = "";
  document.getElementById("reuseNoticeText").textContent = "";
}

function tmReuseRecordedMessage(text, source) {
  setTopMode("test");
  var input = document.getElementById("chatInput");
  input.value = text;                       /* verbatim, line breaks and all */
  tm.reuse = { text: text, turnKey: source && source.turnKey };
  tmSizeComposer();
  if (!input.disabled) {
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
  var from = source && source.turnKey ? "turn " + source.turnKey : "a recorded turn";
  tmShowReuseNotice("Copied from " + from + ". Edit it if you like — sending "
    + "starts a new turn in this chat session, under the current workflow and "
    + "settings. It does not re-run the recorded one.");
  if (!tm.connected) {
    showNotice("Message copied to the composer",
      "info", "Connect to the workflow server to send it.");
  }
}

function tmReuseLabel(text) {
  /* Called as the message goes out: says what the new turn is, without
     claiming it reproduces the old one. */
  if (!tm.reuse) { return null; }
  var edited = tm.reuse.text !== text;
  var from = tm.reuse.turnKey ? "turn " + tm.reuse.turnKey : "a recorded turn";
  return "New turn, under the current settings · "
    + (edited ? "edited copy of " : "message copied from ") + from;
}

function tmComposerState() {
  var enabled = tm.connected && !tm.busy;
  document.getElementById("chatInput").disabled = !enabled;
  document.getElementById("chatSend").disabled = !enabled;
  document.getElementById("newConvBtn").style.display =
    (tm.connected && document.getElementById("testMain").className === "visible") ? "" : "none";
  if (enabled) { document.getElementById("chatInput").focus(); }
}

function tmClearLog(placeholder) {
  var log = document.getElementById("chatLog");
  tmOpenArtifacts = null;
  clear(log);
  if (placeholder) {
    log.appendChild(el("div", "empty", placeholder));
  }
}

function tmLatestConversation(conversations, idsWithTurns) {
  /* Prefer the highest conversation_id that actually has turns. An unused
     reserved id (the empty slot /new_conversation leaves behind) is not a
     conversation the developer can continue. last_turn_at ranking is ignored
     so a later write to an older id cannot steal the pane. */
  var latest = null;
  var latestWithTurns = null;
  (conversations || []).forEach(function (conv) {
    if (!conv || conv.conversation_id == null) { return; }
    if (!latest || conv.conversation_id > latest.conversation_id) {
      latest = conv;
    }
    if (idsWithTurns && idsWithTurns[conv.conversation_id]) {
      if (!latestWithTurns || conv.conversation_id > latestWithTurns.conversation_id) {
        latestWithTurns = conv;
      }
    }
  });
  return latestWithTurns || latest;
}

function tmSortTurnsChronologically(turns) {
  return (turns || []).slice().sort(function (a, b) {
    var ao = a.ordinal == null ? 0 : a.ordinal;
    var bo = b.ordinal == null ? 0 : b.ordinal;
    if (ao !== bo) { return ao - bo; }
    return String(a.turn_key || "").localeCompare(String(b.turn_key || ""));
  });
}

function tmRenderStoredTurn(turn, autoloadArtifacts) {
  if (turn.user_message) {
    tmBubble("user", turn.user_message);
  }
  var agent = tmBubble("agent", "…");
  tmRenderTurn(agent, {
    turn_key: turn.turn_key,
    status: turn.status,
    success: !!turn.success,
    answer: turn.answer,
    command_outputs: []
  });
  tmAttachStoredArtifacts(agent, turn.turn_key, autoloadArtifacts);
  return agent;
}

function tmAttachStoredArtifacts(bubbleMsg, turnKey, autoload) {
  /* The turn list carries no artifacts — only the stored record does — so a
     reopened conversation fetches it per turn. Reading a history of hundreds
     of turns must not be hundreds of round trips, so only the most recent
     turns load on sight and the rest keep a button. Nothing here can lose the
     answer: every failure paints beside it. */
  if (!turnKey) { return; }
  var slot = el("div", "cmdOut");
  tmAppendBeforeMeta(bubbleMsg, slot);
  var loaded = false;

  function load() {
    if (loaded) { return; }
    loaded = true;
    clear(slot);
    slot.appendChild(document.createTextNode("loading this turn's artifacts…"));
    tmChatApi("/api/turn/" + encodeURIComponent(turnKey))
      .then(function (data) {
        var record = (data.turn && data.turn.record) || {};
        var outputs = (record.turn_output || {}).command_outputs || [];
        bubbleMsg.removeChild(slot);
        /* Say where this came from: it is the recorded turn, not a live one. */
        tmRenderArtifacts(bubbleMsg, outputs, "from the stored record of this turn");
      })
      .catch(function (err) {
        clear(slot);
        slot.appendChild(document.createTextNode(
          "Could not load this turn's artifacts (" + err.message + ")."));
        loaded = false;
        var retry = el("button", "traceLink", "retry");
        retry.addEventListener("click", load);
        slot.appendChild(retry);
      });
  }

  if (autoload) { load(); return; }
  var button = el("button", "traceLink", "show artifacts");
  button.addEventListener("click", function () {
    slot.removeChild(button);
    load();
  });
  slot.appendChild(button);
}

function tmActivateConversation(conversationId) {
  /* Bind the live session to the thread we painted, so the next message
     continues it rather than the empty reserved id /initialize restored. */
  if (!conversationId || !tm.baseUrl || !tm.token) {
    return Promise.resolve();
  }
  return tmFetch("/activate_conversation", { conversation_id: conversationId })
    .then(function (r) {
      if (r.ok || r.status === 404) { return; }
      showNotice("Could not resume conversation", "error",
        "activate_conversation returned " + r.status);
    })
    .catch(function (error) {
      showNotice("Could not resume conversation", "error", error.message);
    });
}

function tmLoadLatestConversation() {
  tmClearLog();
  if (!tm.channelId) {
    return Promise.resolve([]);
  }
  var channel = encodeURIComponent(tm.channelId);
  return Promise.all([
    api("/api/conversations?channel=" + channel + "&limit=500"),
    api("/api/turns?channel=" + channel + "&limit=500")
  ]).then(function (results) {
    var convs = results[0].conversations || [];
    var allTurns = results[1].turns || [];
    var idsWithTurns = {};
    allTurns.forEach(function (t) {
      if (t.conversation_id != null) { idsWithTurns[t.conversation_id] = true; }
    });
    var latest = tmLatestConversation(convs, idsWithTurns);
    if (!latest) { return []; }
    tm.activeConversationId = latest.conversation_id;
    var turns = tmSortTurnsChronologically(allTurns.filter(function (t) {
      return t.conversation_id === latest.conversation_id;
    }));
    var maxId = null;
    convs.forEach(function (c) {
      if (c.conversation_id != null && (maxId == null || c.conversation_id > maxId)) {
        maxId = c.conversation_id;
      }
    });
    /* /initialize restores the reserved last id; if that slot is empty we
       painted an older thread and must bind the session to it. */
    var bind = (latest.conversation_id !== maxId && idsWithTurns[latest.conversation_id] && tm.token)
      ? tmActivateConversation(latest.conversation_id)
      : Promise.resolve();
    return bind.then(function () {
      tmClearLog();
      if (turns.length) {
        var autoloadFrom = turns.length - TM_RESTORED_ARTIFACT_AUTOLOAD;
        turns.forEach(function (turn, index) {
          tmRenderStoredTurn(turn, index >= autoloadFrom);
        });
      }
      return turns;
    });
  });
}

function tmConnect() {
  /* Chat is interactive: the session is driven by what you type, so no
     startup_command/startup_action/context is sent. Those stay server-launch
     decisions (run_fastapi_mcp flags) or programmatic /initialize fields. */
  var pasted = document.getElementById("tToken").value.trim();
  var body = { channel_id: tm.channelId, user_id: tm.userId };
  var epoch = tm.epoch;   /* the source this connection is being made for */
  connText("Connecting to " + tm.baseUrl + " …");
  tm.token = pasted;  /* a pasted token also authorizes the turn calls */
  tmFetch("/initialize", body, { noAuth: !pasted })
    .then(function (r) {
      return r.json().then(function (data) { return { status: r.status, data: data }; });
    })
    .then(function (got) {
      if (tm.epoch !== epoch) { return; }   /* switched away mid-connect */
      if (got.status !== 200 && got.status !== 202) {
        var detail = got.data && got.data.detail ? String(got.data.detail) : ("HTTP " + got.status);
        throw { handled: true, message: "/initialize failed: " + detail };
      }
      if (!pasted && got.data.access_token) { tm.token = got.data.access_token; }
      tm.connected = true;
      deadServerAnnounced = false;
      var note = (tm.managed && session && session.server_note)
        ? " (" + session.server_note + ")" : "";
      connText("Connected to " + tm.baseUrl + " — loading conversation…" + note, "ok");
      setPill("ok", "server running");
      tm.busy = true;
      tmComposerState();
      /* /initialize already rejoins the channel's last conversation; paint
         those turns so the developer can continue instead of starting blank. */
      tmLoadLatestConversation()
        .catch(function () {
          tmClearLog();
          tmBubble("system",
            "Could not load previous turns. You can still continue this conversation.");
          return [];
        })
        .then(function (turns) {
          var keys = {};
          (turns || []).forEach(function (t) {
            if (t.turn_key) { keys[t.turn_key] = true; }
          });
          connText("Connected to " + tm.baseUrl +
            ((turns && turns.length)
              ? " — continuing the latest conversation."
              : " — say something below.") + note, "ok");
          /* The chat never sends a startup command, but the server may have
             been launched with one; render it rather than leaving an
             unexplained first turn, unless history already includes it. */
          if (got.data.startup_output && !keys[got.data.startup_output.turn_key]) {
            var msg = tmBubble("agent", "…");
            tmRenderTurn(msg, got.data.startup_output);
            tmBubble("system", "The server's own startup command ran as this session's first turn.");
          } else if (got.status === 202 && got.data.startup_turn_key) {
            var pending = tmBubble("agent", "startup turn still running…");
            tmPollTurn(got.data.startup_logical_turn_key || got.data.startup_turn_key, pending);
            return;
          }
          if (got.data.startup_error) {
            tmBubble("system", "Startup turn failed: " + got.data.startup_error);
          }
          tm.busy = false;
          tmComposerState();
        });
    })
    .catch(function (err) {
      if (tm.epoch !== epoch) { return; }
      tm.connected = false;
      tmComposerState();
      setPill("err", "not connected");
      connText(err && err.handled ? err.message : tmExplainFetchError(err), "err",
        err && err.handled ? "" : "server");
    });
}

/* Auto-connect: wait for the spawned server's readiness probe, then
   /initialize with the chatbot-managed identity. Model loading can take a
   while on first start, so be patient and say what is happening. */
function tmAutoConnect(deadlineMs) {
  var deadline = Date.now() + (deadlineMs || 180000);
  /* A probe loop outlives the source that started it unless it says so: a
     switch away must not end in a connection to (or a pill about) a server
     this page is no longer looking at. */
  var epoch = tm.epoch;
  setPill("wait", "server starting…");
  connText("Starting the workflow server (first start loads models — this can take a minute)…");
  function probe() {
    if (tm.epoch !== epoch || !tm.baseUrl) { return; }
    /* `checkSession` keeps `session` current, and a server that died while
       starting will never answer readyz: say so instead of probing it until
       the deadline. */
    if (session && session.server_running === false
        && session.server_exit_code !== null && session.server_exit_code !== undefined) {
      setPill("err", "server failed to start");
      connText("The workflow server exited during startup (exit code " +
        session.server_exit_code + ") — check the server log. " +
        "Switch workflow (same one is fine) restarts it.", "err", "server");
      return;
    }
    if (Date.now() > deadline) {
      setPill("err", "server not ready");
      connText("The workflow server did not become ready. Check the server log, " +
        "then use the Advanced panel to reconnect.", "err", "server");
      return;
    }
    fetch(tm.baseUrl + "/probes/readyz")
      .then(function (r) {
        if (r.ok) { tmConnect(); }
        else { setTimeout(probe, 1500); }
      })
      .catch(function () { setTimeout(probe, 1500); });
  }
  probe();
}

document.getElementById("advToggle").addEventListener("click", function () {
  var panel = document.getElementById("advPanel");
  var open = panel.hasAttribute("hidden");
  if (open) { panel.removeAttribute("hidden"); }
  else { panel.setAttribute("hidden", ""); }
  this.setAttribute("aria-expanded", open ? "true" : "false");
});

document.getElementById("tConnect").addEventListener("click", function () {
  var typed = document.getElementById("tBaseUrl").value.trim().replace(/\/+$/, "");
  if (typed) { tm.baseUrl = typed; }
  if (!tm.baseUrl) { connText("Enter the FastAPI base URL first.", "err"); return; }
  /* Only the chatbot-spawned server is liveness-tracked via /api/session. */
  tm.managed = !!(session && session.server_url && tm.baseUrl === session.server_url);
  tmConnect();
});

function tmSend() {
  var input = document.getElementById("chatInput");
  var text = input.value.trim();
  if (!text || !tm.connected || tm.busy) { return; }
  var reuseLabel = tmReuseLabel(text);
  input.value = "";
  tmSizeComposer();
  tmClearReuse();
  tm.busy = true;
  tmComposerState();
  var sent = tmBubble("user", text);
  if (reuseLabel) { sent.appendChild(el("div", "reuseNote", reuseLabel)); }
  /* "/"-prefixed → deterministic execution via /invoke_assistant */
  var deterministic = text.charAt(0) === "/";
  var path = deterministic ? "/invoke_assistant" : "/invoke_agent";
  var pending = tmBubble("agent", deterministic ? "running command…" : "thinking…");
  if (!deterministic) {
    /* Agent turns stream: the interactions arrive as they happen instead of
       appearing all at once when the turn is already over. */
    tmStreamTurn(text, pending);
    return;
  }
  tmFetch(path, { user_query: text, timeout_seconds: 60 })
    .then(function (r) {
      return r.json().then(function (data) { return { status: r.status, data: data }; });
    })
    .then(function (got) {
      if (got.status === 200) {
        tmRenderTurn(pending, got.data);
        tm.busy = false; tmComposerState();
      } else if (got.status === 202 && got.data.turn_key) {
        tmPollTurn(got.data.logical_turn_key || got.data.turn_key, pending);
      } else if (got.status === 409) {
        tmRenderError(pending, "Another turn is already in progress on this channel (HTTP 409). Wait for it to finish.");
        tm.busy = false; tmComposerState();
      } else if (got.status === 401) {
        tmRenderError(pending, "The server rejected the bearer token (HTTP 401). Reconnect (Advanced panel) to mint new tokens.");
        tm.busy = false; tmComposerState();
      } else {
        var detail = got.data && got.data.detail ? String(got.data.detail) : "";
        tmRenderError(pending, path + " failed: HTTP " + got.status + (detail ? " — " + detail : ""));
        tm.busy = false; tmComposerState();
      }
    })
    .catch(function (err) {
      tmRenderError(pending, tmExplainFetchError(err));
      /* Keep the typed message so a transient failure costs nothing. */
      if (!input.value) { input.value = text; }
      tm.busy = false; tmComposerState();
      checkSession();
    });
}
document.getElementById("chatSend").addEventListener("click", tmSend);
document.getElementById("chatInput").addEventListener("keydown", function (e) {
  /* Shift+Enter writes a line instead of sending: a multi-line message (a
     reused one, or one typed here) has to be possible to compose. */
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); tmSend(); }
});
document.getElementById("chatInput").addEventListener("input", tmSizeComposer);
document.getElementById("reuseClear").addEventListener("click", function () {
  var input = document.getElementById("chatInput");
  input.value = "";
  tmSizeComposer();
  tmClearReuse();
  if (!input.disabled) { input.focus(); }
});

document.getElementById("newConvBtn").addEventListener("click", function () {
  if (!tm.connected || tm.busy) { return; }
  var button = this; button.disabled = true; button.setAttribute("aria-busy", "true");
  tmFetch("/new_conversation", {})
    .then(function (r) {
      if (!r.ok) { throw new Error("HTTP " + r.status); }
      tm.activeConversationId = null;
      tmClearLog();
      showNotice("New conversation started"); document.getElementById("chatInput").focus();
    })
    .catch(function (err) {
      tmBubble("system", "Could not start a new conversation: " + err.message);
      showNotice("Could not start a conversation", "error", err.message);
    }).finally(function () { button.disabled = false; button.removeAttribute("aria-busy"); });
});

/* -- deep link: chat → debug mode filtered to one turn ------------------ */
function openTurnInDebug(turnKey) {
  setTopMode("debug");
  selectTurnWithRetry(turnKey, 8);
}
function selectTurnWithRetry(turnKey, attemptsLeft, spanId, level) {
  /* The observability writer is asynchronous: the turn may land in the DB a
     moment after the HTTP response. Retry briefly before giving up. */
  api("/api/turn/" + encodeURIComponent(turnKey)).then(function () {
    selectTurn(turnKey, spanId, null, level);
  }).catch(function () {
    if (attemptsLeft > 0) {
      setTimeout(function () { selectTurnWithRetry(turnKey, attemptsLeft - 1, spanId, level); }, 700);
    } else {
      var d = document.getElementById("detail");
      clear(d);
      d.appendChild(el("div", "empty",
        "Turn " + turnKey + " is not in the observability DB yet. Hit Refresh in a moment."));
    }
  });
}

