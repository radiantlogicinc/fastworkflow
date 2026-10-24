/* -- archived conversations ------------------------------------------------
   Reached from Tools. Live conversations that do not belong to an experiment
   are archived here, several at a time. The file is the record; the live rail
   no longer lists one after it is archived. Delete removes the file. */
var archivedConversationsNav = 0;
function archivedConversationsOwnDetail() {
  return archivedConversationsNav !== 0 && !expNavStale(archivedConversationsNav);
}

function conversationArchivePath(row) {
  return "/api/archived-conversations/"
    + encodeURIComponent(row.channel_id) + "/" + row.conversation_id;
}

function conversationArchiveTitle(row) {
  return row.topic || ("Conversation #" + row.conversation_id);
}

function conversationSelectValue(row) {
  return JSON.stringify({
    channel_id: row.channel_id,
    conversation_id: row.conversation_id
  });
}

function nConversations(n) {
  return n + " conversation" + (n === 1 ? "" : "s");
}

function conversationRow(container, row, details) {
  var item = el("div", "listItem sealedRow");
  var body = el("div", "sealedBody");
  body.appendChild(el("div", "title", conversationArchiveTitle(row)));
  body.appendChild(el("div", "sub", details.filter(Boolean).join(" · ")));
  body.appendChild(el("div", "recordId",
    (row.channel_id || "") + " #" + row.conversation_id));
  var actions = el("div", "runDecision");
  body.appendChild(actions);
  item.appendChild(body);
  container.appendChild(item);
  actions.item = item;
  actions.body = body;
  return actions;
}

function showArchivedConversations() {
  if (!document.getElementById("debugMain").classList.contains("visible")) { setTopMode("debug"); }
  var nav = expNavToken();
  archivedConversationsNav = nav;
  var d = document.getElementById("detail");
  clear(d);
  d.appendChild(el("div", "empty", "Loading archived conversations…"));
  function reload() { if (!expNavStale(nav)) { showArchivedConversations(); } }
  api("/api/archived-conversations").then(function (data) {
    if (expNavStale(nav)) { return; }
    refreshMeta();
    clear(d);
    var archived = data.archived || [];
    var ready = data.ready || [];
    pageHeader(d, "RECORDED CONVERSATIONS", "Archived conversations",
      "Select conversations and archive them to move their turns out of the live database "
      + "into read-only archives. Open an archived conversation here, and delete it when "
      + "you no longer need it. A conversation that belongs to an experiment is archived "
      + "with that experiment.");

    var readyHead = sectionHeader(d, "Ready to archive", ready.length);
    if (!ready.length) {
      emptyState(d, "No conversation is waiting",
        "A recorded conversation that does not belong to an experiment can be archived.");
    } else {
      var addReady = bulkBar(readyHead, {
        allLabel: "Select all ready to archive",
        actionLabel: "Archive selected",
        className: "primary",
        title: function (n) { return "Archive " + nConversations(n) + "?"; },
        one: "Its turns are copied into one read-only archive and removed from the live database. You can still open it.",
        many: "Their turns are copied into read-only archives and removed from the live database. You can still open them.",
        confirmOne: "Archive conversation",
        confirmMany: "Archive conversations",
        verb: "Archived",
        noun: "conversation",
        reload: reload,
        oneRequest: function (id) {
          var key = JSON.parse(id);
          return requestJson("/api/archived-conversations", "POST", {
            channel_id: key.channel_id,
            conversation_id: key.conversation_id
          });
        }
      });
      ready.forEach(function (row) {
        var actions = conversationRow(d, row, [
          row.turn_count + (row.turn_count === 1 ? " turn" : " turns"),
          row.last_turn_at ? "last turn " + fmtTs(row.last_turn_at) : ""
        ]);
        addReady(rowCheck(actions.item, conversationSelectValue(row),
          "Select conversation " + conversationArchiveTitle(row)));
        var archive = el("button", "primary", "Archive");
        archive.addEventListener("click", function () {
          confirmDialog({
            opener: archive,
            title: "Archive this conversation?",
            description: "Its turns are copied into one read-only archive and removed from the live database. You can still open it.",
            confirmLabel: "Archive conversation",
            cancelLabel: "Keep it live"
          }).then(function (confirmed) {
            if (!confirmed) { return; }
            lastActionButton = archive;
            mutationRequest("/api/archived-conversations", "POST", {
              channel_id: row.channel_id,
              conversation_id: row.conversation_id
            }, "Conversation archived").then(function () {
              refreshMeta();
              refreshConvs();
              reload();
            }).catch(function () {});
          });
        });
        actions.appendChild(archive);
      });
    }

    var archivedHead = sectionHeader(d, "Archived", archived.length);
    if (!archived.length) {
      emptyState(d, "Nothing archived yet", "Archive a conversation above.");
    } else {
      var addArchived = bulkBar(archivedHead, {
        allLabel: "Select all archived",
        actionLabel: "Delete selected",
        className: "danger",
        title: function (n) { return "Delete " + nConversations(n) + "?"; },
        one: "Its archive file and the record of it are deleted. This cannot be undone.",
        many: "Their archive files and the records of them are deleted. This cannot be undone.",
        confirmOne: "Delete archived conversation",
        confirmMany: "Delete archived conversations",
        verb: "Deleted",
        noun: "conversation",
        reload: reload,
        oneRequest: function (id) {
          return requestJson(conversationArchivePath(JSON.parse(id)), "DELETE");
        }
      });
      archived.forEach(function (row) {
        var actions = conversationRow(d, row, [
          row.turn_count + (row.turn_count === 1 ? " turn" : " turns"),
          row.archived_at ? "archived " + fmtTs(row.archived_at) : ""
        ]);
        addArchived(rowCheck(actions.item, conversationSelectValue(row),
          "Select archived conversation " + conversationArchiveTitle(row)));
        var open = el("button", null, "Open");
        open.addEventListener("click", function () {
          api(conversationArchivePath(row)).then(function (detail) {
            var turns = el("div");
            (detail.turns || []).forEach(function (turn) {
              var item = el("div", "listItem");
              item.appendChild(el("strong", null, turn.user_message || "(no message)"));
              item.appendChild(el("div", "sub",
                (turn.status || "") + " · " + fmtTs(turn.started_at)));
              if (turn.answer) { item.appendChild(el("p", null, turn.answer)); }
              turns.appendChild(item);
            });
            if (!(detail.turns || []).length) {
              turns.appendChild(el("p", "empty", "No turns were recorded."));
            }
            actions.body.appendChild(turns);
            open.disabled = true;
          }).catch(function (error) {
            showNotice("Could not open this conversation", "error", error.message);
          });
        });
        var remove = el("button", "danger", "Delete");
        remove.addEventListener("click", function () {
          confirmDialog({
            opener: remove,
            title: "Delete this archived conversation?",
            description: "Its archive file and the record of it are deleted. This cannot be undone.",
            confirmLabel: "Delete archived conversation",
            cancelLabel: "Keep it"
          }).then(function (confirmed) {
            if (!confirmed) { return; }
            lastActionButton = remove;
            mutationRequest(conversationArchivePath(row), "DELETE", null,
              "Archived conversation deleted").then(function () {
              refreshMeta();
              reload();
            }).catch(function () {});
          });
        });
        actions.appendChild(open);
        actions.appendChild(remove);
      });
    }
  }).catch(function (error) {
    if (expNavStale(nav)) { return; }
    clear(d);
    d.appendChild(el("div", "empty", "Could not load archived conversations: " + error.message));
  });
}

document.getElementById("archivedConversationsBtn").addEventListener("click", function () {
  document.getElementById("toolsMenu").open = false;
  showArchivedConversations();
});
