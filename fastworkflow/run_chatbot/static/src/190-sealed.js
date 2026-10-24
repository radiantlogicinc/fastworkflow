/* -- archived experiments [fix-ajir.24] ---------------------------------
   A development page over /api/archived-experiments. Archiving a complete
   experiment writes one read-only archive of its evidence and frees its turns
   from the live database; the experiment then opens exactly as before, its
   turns read from the archive. Nothing here happens on its own: every archive,
   release and delete is confirmed in the page first. A released experiment is
   removed with delete; this page does not call that action prune. */
function sealExperiment(experimentId, opener) {
  return confirmDialog({
    opener: opener,
    title: "Archive this experiment?",
    description: "Its evidence is copied into one read-only archive and removed from the live "
      + "database. You can still open it, compare it and choose it; its turns are read from the archive.",
    confirmLabel: "Archive experiment",
    cancelLabel: "Keep it live"
  }).then(function (confirmed) {
    if (!confirmed) { return null; }
    lastActionButton = opener;
    return mutationRequest("/api/archived-experiments", "POST", {experiment_id: experimentId},
      function (data) { return data.warning ? "Experiment archived, with a warning" : "Experiment archived"; }
    ).then(function (data) {
      refreshMeta();
      return data;
    });
  });
}

function sealedTitle(row) {
  return row.description || ("Experiment · " + String(row.experiment_id).slice(-8));
}

function sealedRow(container, row, details) {
  var item = el("div", "listItem sealedRow");
  var body = el("div", "sealedBody");
  body.appendChild(el("div", "title", sealedTitle(row)));
  body.appendChild(el("div", "sub", details.filter(Boolean).join(" · ")));
  body.appendChild(el("div", "recordId", row.experiment_id));
  var actions = el("div", "runDecision");
  body.appendChild(actions);
  item.appendChild(body);
  container.appendChild(item);
  actions.item = item;
  actions.body = body;
  return actions;
}

function benchmarkLabel(row) {
  return row.benchmark_id ? row.benchmark_id + " " + (row.benchmark_version || "") : "";
}

function selectedBlock(row) {
  if (row.selected_as === "experiment") {
    return "This experiment is its contest's winner; withdraw that first";
  }
  if (row.selected_as) {
    return "A run of this experiment is a task's best run; withdraw that first";
  }
  return "";
}

function nExperiments(n) {
  return n + " experiment" + (n === 1 ? "" : "s");
}

/* One JSON write, with no notice of its own: a selection reports once. */
function requestJson(path, method, body) {
  var options = {
    method: method,
    headers: chatbotAuthHeaders({"Content-Type": "application/json"})
  };
  if (method !== "DELETE") { options.body = JSON.stringify(body || {}); }
  return fetch(path, options).then(function (response) {
    return response.json().catch(function () { return {}; }).then(function (data) {
      if (!response.ok) {
        throw new Error(data.error || ("Request failed (" + response.status + ")"));
      }
      return data;
    });
  });
}

function runSelected(button, ids, one) {
  var failures = [];
  var warnings = 0;
  var index = 0;
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  function step() {
    if (index >= ids.length) {
      return Promise.resolve({failures: failures, warnings: warnings});
    }
    var id = ids[index++];
    return one(id).then(function (data) {
      if (data && data.warning) { warnings += 1; }
      refreshMeta();
      return step();
    }, function (error) {
      failures.push(String(id).slice(-8) + ": " + error.message);
      return step();
    });
  }
  return step().finally(function () {
    if (button.isConnected) {
      button.disabled = false;
      button.removeAttribute("aria-busy");
    }
  });
}

function finishBulk(verb, count, outcome, reload, nounWord) {
  var failures = outcome.failures;
  var done = count - failures.length;
  var noun = " " + (nounWord || "experiment") + (count === 1 ? "" : "s");
  var message = verb + " " + done + (failures.length ? " of " + count : "") + noun;
  if (outcome.warnings) { message += ", with a warning"; }
  if (!failures.length) {
    showNotice(message);
  } else {
    showNotice(message, "error", failures.join("; "));
  }
  if (done) { refreshConvs(); reload(); }
}

/* Select all and one action for the checkboxes registered into this bar.
   A checkbox that is disabled (a winner, a best run) is never included. */
function bulkBar(head, spec) {
  var boxes = [];
  var tools = el("div", "sectionTools");
  var allLabel = el("label");
  var all = el("input");
  all.type = "checkbox";
  all.setAttribute("aria-label", spec.allLabel);
  allLabel.appendChild(all);
  allLabel.appendChild(document.createTextNode(" Select all"));
  var action = el("button", spec.className || null, spec.actionLabel);
  action.disabled = true;
  function usable() { return boxes.filter(function (box) { return !box.disabled; }); }
  function picked() { return usable().filter(function (box) { return box.checked; }); }
  function sync() {
    var open = usable();
    var chosen = picked();
    action.disabled = chosen.length === 0;
    all.disabled = open.length === 0;
    all.checked = open.length > 0 && chosen.length === open.length;
    all.indeterminate = chosen.length > 0 && chosen.length < open.length;
  }
  all.addEventListener("change", function () {
    usable().forEach(function (box) { box.checked = all.checked; });
    sync();
  });
  action.addEventListener("click", function () {
    var ids = picked().map(function (box) { return box.value; });
    if (!ids.length) { return; }
    var many = ids.length !== 1;
    confirmDialog({
      opener: action,
      title: spec.title(ids.length),
      description: many ? spec.many : spec.one,
      confirmLabel: many ? spec.confirmMany : spec.confirmOne,
      cancelLabel: many ? "Keep them" : "Keep it"
    }).then(function (confirmed) {
      if (!confirmed) { return; }
      runSelected(action, ids, spec.oneRequest).then(function (outcome) {
        finishBulk(spec.verb, ids.length, outcome, spec.reload, spec.noun);
      });
    });
  });
  tools.appendChild(allLabel);
  tools.appendChild(action);
  head.appendChild(tools);
  return function (box) {
    boxes.push(box);
    box.addEventListener("change", sync);
    sync();
  };
}

function rowCheck(item, experimentId, label, blocked) {
  var box = el("input", "sealedCheck");
  box.type = "checkbox";
  box.value = experimentId;
  box.setAttribute("aria-label", label);
  if (blocked) {
    box.disabled = true;
    box.title = blocked;
  }
  item.insertBefore(box, item.firstChild);
  return box;
}

/* The page is reached from the Tools menu, not the rail, so no rail
   selection holds it: the rail's background refresh is kept from painting
   its placeholder over the page while the page still owns #detail. */
var sealedPageNav = 0;
function sealedPageOwnsDetail() { return sealedPageNav !== 0 && !expNavStale(sealedPageNav); }

function showSealedExperiments() {
  if (!document.getElementById("debugMain").classList.contains("visible")) { setTopMode("debug"); }
  var nav = expNavToken();
  sealedPageNav = nav;
  var d = document.getElementById("detail");
  clear(d);
  d.appendChild(el("div", "empty", "Loading archived experiments…"));
  function reload() { if (!expNavStale(nav)) { showSealedExperiments(); } }
  api("/api/archived-experiments").then(function (data) {
    if (expNavStale(nav)) { return; }
    refreshMeta();
    clear(d);
    var released = data.released || (data.unarchivable || []).filter(function (row) { return row.released; });
    var unsealable = (data.unarchivable || []).filter(function (row) { return !row.released; });
    pageHeader(d, "BENCHMARK EVIDENCE", "Archived experiments",
      "Archive complete experiments to move their evidence out of the live database into "
      + "read-only archives. Open an archived experiment like any other, and delete it when "
      + "you no longer need it. Release an experiment that cannot be archived, then delete it.");

    var sealedHead = sectionHeader(d, "Archived", data.archived.length);
    if (!data.archived.length) {
      emptyState(d, "Nothing archived yet", "Archive a complete experiment below.");
    } else {
      var addSealed = bulkBar(sealedHead, {
        allLabel: "Select all archived",
        actionLabel: "Delete selected",
        className: "danger",
        title: function (n) { return "Delete " + n + " archived experiment" + (n === 1 ? "" : "s") + "?"; },
        one: "Its archive file and every record of it in the live database are deleted. This cannot be undone.",
        many: "Their archive files and every record of them in the live database are deleted. This cannot be undone.",
        confirmOne: "Delete archived experiment",
        confirmMany: "Delete archived experiments",
        verb: "Deleted",
        reload: reload,
        oneRequest: function (id) {
          return requestJson("/api/archived-experiments/" + encodeURIComponent(id), "DELETE");
        }
      });
      data.archived.forEach(function (row) {
        var blocked = selectedBlock(row);
        var actions = sealedRow(d, row, [
          benchmarkLabel(row), "archived " + fmtTs(row.archived_at), formatByteSize(row.size_bytes),
          row.archive_sha256 ? "sha256 " + String(row.archive_sha256).slice(0, 12) : "",
          row.readable ? "" : "archive file missing"]);
        actions.body.appendChild(el("div", "sub", row.path));
        addSealed(rowCheck(actions.item, row.experiment_id,
          "Select archived experiment " + sealedTitle(row), blocked));
        var open = el("button", null, "Open");
        open.addEventListener("click", function () { showExperiment(row.experiment_id); });
        actions.appendChild(open);
        var remove = el("button", "danger", "Delete archived experiment");
        if (blocked) { remove.disabled = true; remove.title = blocked; }
        remove.addEventListener("click", function () {
          confirmDialog({
            opener: remove,
            title: "Delete this archived experiment?",
            description: "Its archive file and every record of it in the live database are deleted. "
              + "This cannot be undone.",
            confirmLabel: "Delete archived experiment",
            cancelLabel: "Keep it"
          }).then(function (confirmed) {
            if (!confirmed) { return; }
            lastActionButton = remove;
            mutationRequest("/api/archived-experiments/" + encodeURIComponent(row.experiment_id),
              "DELETE", null, "Archived experiment deleted").then(function () {
              refreshConvs(); reload();
            }).catch(function () {});
          });
        });
        actions.appendChild(remove);
      });
    }

    var readyHead = sectionHeader(d, "Ready to archive", data.ready.length);
    if (!data.ready.length) {
      emptyState(d, "No complete experiment is waiting", "An experiment can be archived once it is complete.");
    } else {
      var addReady = bulkBar(readyHead, {
        allLabel: "Select all ready to archive",
        actionLabel: "Archive selected",
        className: "primary",
        title: function (n) { return "Archive " + nExperiments(n) + "?"; },
        one: "Its evidence is copied into one read-only archive and removed from the live database. You can still open it.",
        many: "Their evidence is copied into read-only archives and removed from the live database. You can still open them.",
        confirmOne: "Archive experiment",
        confirmMany: "Archive experiments",
        verb: "Archived",
        reload: reload,
        oneRequest: function (id) {
          return requestJson("/api/archived-experiments", "POST", {experiment_id: id});
        }
      });
      data.ready.forEach(function (row) {
        var actions = sealedRow(d, row, [benchmarkLabel(row), row.status,
          row.completed_at ? "completed " + fmtTs(row.completed_at) : ""]);
        addReady(rowCheck(actions.item, row.experiment_id,
          "Select experiment " + sealedTitle(row)));
        var seal = el("button", "primary", "Archive");
        seal.addEventListener("click", function () {
          sealExperiment(row.experiment_id, seal).then(function (done) { if (done) { reload(); } })
            .catch(function () {});
        });
        actions.appendChild(seal);
      });
    }

    var invalid = unsealable.filter(function (row) { return row.status === "invalid"; });
    var heldHead = sectionHeader(d, "Cannot be archived", unsealable.length);
    if (unsealable.length) {
      d.appendChild(el("p", "sub", "Only a complete experiment can be archived. Release one that "
        + "will never be archived; one still running is closed as invalid first. "
        + "A released experiment can then be deleted."));
    }
    var addInvalid = invalid.length ? bulkBar(heldHead, {
      allLabel: "Select all invalid",
      actionLabel: "Release selected",
      title: function (n) { return "Release " + nExperiments(n) + "?"; },
      one: "You can then delete it. This cannot be undone.",
      many: "You can then delete them. This cannot be undone.",
      confirmOne: "Release evidence",
      confirmMany: "Release evidence",
      verb: "Released",
      reload: reload,
      oneRequest: function (id) {
        return requestJson("/api/evidence-releases", "POST", {experiment_id: id});
      }
    }) : null;
    unsealable.forEach(function (row) {
      var actions = sealedRow(d, row, [benchmarkLabel(row),
        row.status + (row.invalid_reason ? " (" + row.invalid_reason + ")" : ""),
        "created " + fmtTs(row.created_at)]);
      if (row.status === "invalid" && addInvalid) {
        addInvalid(rowCheck(actions.item, row.experiment_id,
          "Select invalid experiment " + sealedTitle(row)));
      }
      var release = el("button", null,
        row.status === "invalid" ? "Release evidence" : "Close as invalid and release");
      release.addEventListener("click", function () {
        confirmDialog({
          opener: release,
          title: row.status === "invalid" ? "Release this experiment's evidence?"
            : "Close this experiment as invalid and release it?",
          description: (row.status === "invalid" ? "" : "It is marked invalid and can no longer be "
            + "resumed or scored. ") + "You can then delete it. This cannot be undone.",
          confirmLabel: "Release evidence",
          cancelLabel: "Keep it"
        }).then(function (confirmed) {
          if (!confirmed) { return; }
          lastActionButton = release;
          mutationRequest("/api/evidence-releases", "POST", {experiment_id: row.experiment_id},
            "Evidence released").then(reload).catch(function () {});
        });
      });
      actions.appendChild(release);
    });

    var releasedHead = sectionHeader(d, "Released", released.length);
    if (!released.length) {
      emptyState(d, "Nothing released",
        "Release an experiment that cannot be archived, then delete it here.");
    } else {
      var addReleased = bulkBar(releasedHead, {
        allLabel: "Select all released",
        actionLabel: "Delete selected",
        className: "danger",
        title: function (n) { return "Delete " + n + " released experiment" + (n === 1 ? "" : "s") + "?"; },
        one: "Every record of it in the live database is deleted. This cannot be undone.",
        many: "Every record of them in the live database is deleted. This cannot be undone.",
        confirmOne: "Delete released experiment",
        confirmMany: "Delete released experiments",
        verb: "Deleted",
        reload: reload,
        oneRequest: function (id) {
          return requestJson("/api/released-experiments/" + encodeURIComponent(id), "DELETE");
        }
      });
      released.forEach(function (row) {
        var blocked = selectedBlock(row);
        var actions = sealedRow(d, row, [benchmarkLabel(row),
          row.status + (row.invalid_reason ? " (" + row.invalid_reason + ")" : ""),
          "created " + fmtTs(row.created_at)]);
        addReleased(rowCheck(actions.item, row.experiment_id,
          "Select released experiment " + sealedTitle(row), blocked));
        var remove = el("button", "danger", "Delete released experiment");
        if (blocked) { remove.disabled = true; remove.title = blocked; }
        remove.addEventListener("click", function () {
          confirmDialog({
            opener: remove,
            title: "Delete this released experiment?",
            description: "Every record of it in the live database is deleted. This cannot be undone.",
            confirmLabel: "Delete released experiment",
            cancelLabel: "Keep it"
          }).then(function (confirmed) {
            if (!confirmed) { return; }
            lastActionButton = remove;
            mutationRequest("/api/released-experiments/" + encodeURIComponent(row.experiment_id),
              "DELETE", null, "Released experiment deleted").then(function () {
              refreshConvs(); reload();
            }).catch(function () {});
          });
        });
        actions.appendChild(remove);
      });
    }
  }).catch(function (e) {
    if (expNavStale(nav)) { return; }
    clear(d);
    d.appendChild(el("div", "empty", "Could not load archived experiments: " + e.message));
  });
}

document.getElementById("sealedExperimentsBtn").addEventListener("click", function () {
  document.getElementById("toolsMenu").open = false;
  showSealedExperiments();
});
