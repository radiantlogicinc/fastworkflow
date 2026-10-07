/* -- boot --------------------------------------------------------------- */
function refreshAll() {
  refreshMeta();
  refreshHealth();
  refreshConvs();
}
document.getElementById("refreshBtn").addEventListener("click", function () {
  var button = this; button.disabled = true; button.setAttribute("aria-busy", "true");
  refreshMeta(); refreshHealth();
  refreshConvs(true).then(function () { showNotice("Navigation refreshed"); })
    .catch(function (error) { showNotice("Could not refresh", "error", error.message); })
    .finally(function () { button.disabled = false; button.removeAttribute("aria-busy"); });
});
function clearedSummary(data) {
  /* The server returns clear_conversations()'s per-table delete counts under
     `deleted`, or an empty object when there was no store to clear. The
     process-cache count is not a deletion, so it does not make a clear count
     as having removed something. */
  var deleted = (data && data.deleted) || {};
  var total = 0;
  Object.keys(deleted).forEach(function (key) {
    if (key !== "offload_scopes_released") { total += Number(deleted[key]) || 0; }
  });
  if (!total) { return "There was nothing to clear."; }
  var convs = Number(deleted.conversations) || 0;
  var turns = Number(deleted.turns) || 0;
  if (!convs && !turns) {
    return "No conversations or turns were recorded; the remaining trace data was cleared.";
  }
  return "Cleared " + fmtCount(convs) + (convs === 1 ? " conversation" : " conversations") +
    " and " + fmtCount(turns) + (turns === 1 ? " turn." : " turns.");
}
document.getElementById("clearConvsBtn").addEventListener("click", function () {
  var button = this;
  confirmDialog({
    opener: button,
    title: "Clear every conversation?",
    description: "Clear every recorded conversation and turn for this workflow? This cannot be undone.",
    confirmLabel: "Clear conversations",
    cancelLabel: "Keep conversations"
  }).then(function (confirmed) {
    if (!confirmed) { return; }
    var rotate = (tm.connected && !tm.busy)
      ? tmFetch("/new_conversation", {}).then(function (r) {
          if (!r.ok) {
            throw new Error("Could not rotate the live chat conversation (" + r.status + ")");
          }
        })
      : Promise.resolve();
    return rotate.then(function () {
      return mutationRequest("/api/clear_conversations", "POST", {
        confirm: "clear all conversations"
      }, clearedSummary);
    }).then(function (data) {
      state.turnKey = null;
      clear(document.getElementById("detail"));
      document.getElementById("detail").appendChild(
        el("div", "empty", clearedSummary(data)));
      clear(document.getElementById("chatLog"));
      document.getElementById("chatLog").appendChild(
        el("div", "empty", "Conversation history cleared. Start a new message."));
      refreshAll();
    }).catch(function (e) {
      showNotice("Could not clear conversations", "error", e.message);
    });
  });
});

/* -- the source boundary [fix-9eg.7.2] ----------------------------------
   Switching evidence sources without relaunching the page is navigation, not
   federation: nothing from the source being left may appear under, or be sent
   to, the one being opened. Everything scoped to a source is dropped here,
   once, and only when the source actually changes — a routine re-apply of the
   SAME session (a restarted server, a configured env, a poll) must not throw
   away a live chat or the turn someone is reading. */
var activeSourceIdentity = null;

function sourceIdentity(sess) {
  if (!sess) { return "none"; }
  return "workflow:" + (sess.db_path || sess.workflow_path || "");
}

function resetSourceScopedState() {
  /* In-flight reads of the old source are invalidated rather than awaited:
     both view-load tokens move, so a response that arrives after the switch
     finds itself stale and paints nothing. */
  expNavToken();

  /* The turn finder owns #detail while it is active and walks the store over
     several round trips, so it has to be stood down explicitly: its sequence
     moves (pages already in flight are disowned), its debounce never fires,
     and its query and rows go with the source they were found in. */
  if (turnFind.timer) { clearTimeout(turnFind.timer); turnFind.timer = null; }
  turnFindReset();
  turnFind.active = false;
  turnFind.running = false;
  turnFind.text = "";
  turnFind.markers = {};
  /* The scope named an experiment, task or attempt IN THE SOURCE BEING
     LEFT. The one being opened can hold those same labels over other runs,
     so it goes with the rest of the search rather than being re-applied. */
  turnFind.scope = null;
  var findText = document.getElementById("turnFindText");
  if (findText) { findText.value = ""; }
  turnFindRenderMarkers();
  turnFindRenderScope();
  turnFindRender();

  state.channel = "";
  state.turnKey = null;
  state.turn = null;
  state.path = [];
  state.experimentId = null;
  state.experimentTask = null;
  state.benchmarkId = null;
  state.benchmarkVersion = null;
  hierarchyRoot = null;
  hierarchyScope = null;
  hierarchyPath = [];
  hierarchyExpanded = {};
  navigationSelection = {conversations: null, benchmarks: null};
  archivedExperimentsShown = {};
  var detail = document.getElementById("detail");
  clear(detail);
  detail.appendChild(el("div", "empty", "Select a turn in this source."));
  writePageLink({});

  /* The live chat belonged to the previous workflow's server. Dropping the
     binding is what stops a message going to it: tmFetch has nowhere to send
     until applySession binds the new server (or leaves this a viewer). The
     epoch stops a turn that is still in flight from repainting or polling. */
  tm.epoch++;
  tm.connected = false;
  tm.managed = false;
  tm.busy = false;
  tm.baseUrl = "";
  tm.token = "";
  tm.channelId = "";
  tm.activeConversationId = null;
  tmClearReuse();
  tmClearLog("Switched evidence source. Connect to its workflow server to chat.");
  tmComposerState();

  /* Views owned elsewhere (comparison, task pages) clean up here if they say
     they need to; the boundary does not reach into them. */
  if (typeof onSourceSwitch === "function") { onSourceSwitch(); }
}

function applySession() {
  /* session: {workflow_path, workflow_name, server_url, server_running,
               channel_id, user_id, jwt_mode, spawn_error} */
  var identity = sourceIdentity(session);
  if (activeSourceIdentity !== null && identity !== activeSourceIdentity) {
    activeSourceIdentity = identity;
    resetSourceScopedState();
  } else {
    activeSourceIdentity = identity;
  }
  if (!session || !session.workflow_path) {
    setPill("", "no workflow");
    setTopMode("picker");
    loadPicker();
    return;
  }
  document.getElementById("switchWfBtn").style.display = "";
  refreshMeta();
  refreshHealth();
  if (session.env_setup_required) {
    setPill("wait", "environment needed");
    setTopMode("picker");
    showEnvSetup();
    return;
  }
  document.getElementById("envSetup").className = "";
  tm.channelId = session.channel_id;
  tm.userId = session.user_id || "developer";
  if (session.server_url) {
    document.getElementById("tBaseUrl").value = session.server_url;
  }
  if (session.server_running && session.server_url) {
    tm.baseUrl = session.server_url;
    tm.managed = true;
    document.getElementById("tBaseUrl").value = session.server_url;
    setTopMode("test");
    if (session.jwt_mode === "signed") {
      setPill("wait", "signed JWT server");
      connText("The server requires signed JWTs: paste a token in the Advanced " +
        "panel, then Reconnect.", "err");
    } else if (!tm.connected) {
      tmAutoConnect();
    }
  } else {
    /* trace viewer only — say why the chat is not live */
    setPill(session.spawn_error ? "err" : "", "viewer only");
    setTopMode("debug");
    if (session.spawn_error) {
      connText("The workflow server was not started: " + session.spawn_error, "err", "server");
    } else {
      connText("No FastAPI server was spawned. Use the Advanced panel to " +
        "connect to one already running.", "err");
    }
  }
}

api("/api/session").then(function (data) {
  session = data.session;
  applySession();
  /* deep links override the default landing view. The fragment is the one
     the page was opened with: applySession has already written the landing
     mode over the address bar. */
  openPageLink(initialPageLink);
}).catch(function () {
  /* No control plane (very old server?) — behave like the plain viewer. */
  setPill("", "viewer");
  setTopMode("debug");
});
