/* -- sealed experiments [fix-ajir.24] -----------------------------------
   A development page over /api/sealed-experiments. Sealing a complete
   experiment writes one read-only archive of its evidence and frees its turns
   from the live database; the experiment then opens exactly as before, its
   turns read from the archive. Nothing here happens on its own: every seal,
   release and delete is one click, and the two that cannot be undone are
   confirmed in the page first. */
function sealExperiment(experimentId, opener) {
  return confirmDialog({
    opener: opener,
    title: "Seal this experiment?",
    description: "Its evidence is copied into one read-only archive and removed from the live "
      + "database. You can still open it, compare it and choose it; its turns are read from the archive.",
    confirmLabel: "Seal experiment",
    cancelLabel: "Keep it live"
  }).then(function (confirmed) {
    if (!confirmed) { return null; }
    lastActionButton = opener;
    return mutationRequest("/api/sealed-experiments", "POST", {experiment_id: experimentId},
      function (data) { return data.warning ? "Experiment sealed, with a warning" : "Experiment sealed"; });
  });
}

function sealedTitle(row) {
  return row.description || ("Experiment · " + String(row.experiment_id).slice(-8));
}

function sealedRow(container, row, details) {
  var item = el("div", "listItem sealedRow");
  item.appendChild(el("div", "title", sealedTitle(row)));
  item.appendChild(el("div", "sub", details.filter(Boolean).join(" · ")));
  item.appendChild(el("div", "recordId", row.experiment_id));
  var actions = el("div", "runDecision");
  item.appendChild(actions);
  container.appendChild(item);
  return actions;
}

function benchmarkLabel(row) {
  return row.benchmark_id ? row.benchmark_id + " " + (row.benchmark_version || "") : "";
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
  d.appendChild(el("div", "empty", "Loading sealed experiments…"));
  function reload() { if (!expNavStale(nav)) { showSealedExperiments(); } }
  api("/api/sealed-experiments").then(function (data) {
    if (expNavStale(nav)) { return; }
    clear(d);
    pageHeader(d, "BENCHMARK EVIDENCE", "Sealed experiments",
      "Seal a complete experiment to move its evidence out of the live database into one "
      + "read-only archive. Open a sealed experiment like any other; delete it when you no longer need it.");

    sectionHeader(d, "Sealed", data.sealed.length);
    if (!data.sealed.length) { emptyState(d, "Nothing sealed yet", "Seal a complete experiment below."); }
    data.sealed.forEach(function (row) {
      var actions = sealedRow(d, row, [
        benchmarkLabel(row), "sealed " + fmtTs(row.sealed_at), formatByteSize(row.size_bytes),
        "sha256 " + String(row.archive_sha256).slice(0, 12), row.readable ? "" : "archive file missing"]);
      actions.parentNode.appendChild(el("div", "sub", row.path));
      var open = el("button", null, "Open");
      open.addEventListener("click", function () { showExperiment(row.experiment_id); });
      actions.appendChild(open);
      var remove = el("button", "danger", "Delete sealed experiment");
      if (row.selected_as) {
        remove.disabled = true;
        remove.title = row.selected_as === "experiment"
          ? "This experiment is its contest's winner; withdraw that first"
          : "A run of this experiment is a task's best run; withdraw that first";
      }
      remove.addEventListener("click", function () {
        confirmDialog({
          opener: remove,
          title: "Delete this sealed experiment?",
          description: "Its archive file and every record of it in the live database are deleted. "
            + "This cannot be undone.",
          confirmLabel: "Delete sealed experiment",
          cancelLabel: "Keep it"
        }).then(function (confirmed) {
          if (!confirmed) { return; }
          lastActionButton = remove;
          mutationRequest("/api/sealed-experiments/" + encodeURIComponent(row.experiment_id),
            "DELETE", null, "Sealed experiment deleted").then(function () {
            refreshConvs(); reload();
          }).catch(function () {});
        });
      });
      actions.appendChild(remove);
    });

    sectionHeader(d, "Ready to seal", data.ready.length);
    if (!data.ready.length) { emptyState(d, "No complete experiment is waiting", "An experiment can be sealed once it is complete."); }
    data.ready.forEach(function (row) {
      var actions = sealedRow(d, row, [benchmarkLabel(row), row.status,
        row.completed_at ? "completed " + fmtTs(row.completed_at) : ""]);
      var seal = el("button", "primary", "Seal");
      seal.addEventListener("click", function () {
        sealExperiment(row.experiment_id, seal).then(function (done) { if (done) { reload(); } })
          .catch(function () {});
      });
      actions.appendChild(seal);
    });

    sectionHeader(d, "Cannot be sealed", data.unsealable.length);
    if (data.unsealable.length) {
      d.appendChild(el("p", "sub", "Only a complete experiment can be sealed. Prune keeps the "
        + "evidence of these until it is released; one still running is first closed as invalid."));
    }
    data.unsealable.forEach(function (row) {
      var actions = sealedRow(d, row, [benchmarkLabel(row),
        row.status + (row.invalid_reason ? " (" + row.invalid_reason + ")" : ""),
        "created " + fmtTs(row.created_at)]);
      if (row.released) { actions.appendChild(el("span", "pill", "Released")); return; }
      var release = el("button", null,
        row.status === "invalid" ? "Release evidence" : "Close as invalid and release");
      release.addEventListener("click", function () {
        confirmDialog({
          opener: release,
          title: row.status === "invalid" ? "Release this experiment's evidence?"
            : "Close this experiment as invalid and release it?",
          description: (row.status === "invalid" ? "" : "It is marked invalid and can no longer be "
            + "resumed or scored. ") + "Prune may then delete its evidence like any other. This cannot be undone.",
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
  }).catch(function (e) {
    if (expNavStale(nav)) { return; }
    clear(d);
    d.appendChild(el("div", "empty", "Could not load sealed experiments: " + e.message));
  });
}

document.getElementById("sealedExperimentsBtn").addEventListener("click", function () {
  document.getElementById("toolsMenu").open = false;
  showSealedExperiments();
});
