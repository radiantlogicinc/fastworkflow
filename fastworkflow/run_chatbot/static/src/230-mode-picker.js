/* -- top-level views: picker | chat (test) | debug ---------------------- */
function setTopMode(mode) {
  if (mode === "test" || mode === "picker") { benchmarkExperimentSource = null; }
  if (session && session.workspace_mode && mode !== "debug") { mode = "debug"; }
  document.getElementById("pickerMain").className = mode === "picker" ? "visible" : "";
  document.getElementById("debugMain").className = mode === "debug" ? "visible" : "";
  document.getElementById("testMain").className = mode === "test" ? "visible" : "";
  document.getElementById("modeDebug").className = mode === "debug" ? "active" : "";
  document.getElementById("modeTest").className = mode === "test" ? "active" : "";
  document.getElementById("refreshBtn").style.display = mode === "debug" ? "" : "none";
  document.getElementById("clearConvsBtn").style.display =
    (mode === "debug" && session && session.workflow_path
      && !session.workspace_mode) ? "" : "none";
  document.getElementById("newConvBtn").style.display =
    (mode === "test" && tm.connected) ? "" : "none";
  if (mode !== "debug") { setTurnFindOpen(false); }
  if (mode !== "picker") { stopPickerPolling(); }
  renderRecordNavigator();
  if (mode === "debug") { placeRecordNavigator(); refreshAll(); }
}
document.getElementById("modeDebug").addEventListener("click", function () { benchmarkExperimentSource = null; setTopMode("debug"); });
document.getElementById("modeTest").addEventListener("click", function () { setTopMode("test"); });
document.getElementById("switchWfBtn").addEventListener("click", function () {
  setTopMode("picker");
  loadPicker();
});

/* -- workflow picker ---------------------------------------------------- */
function wfTag(w) {
  if (w.training) { return el("span", "tag training", "Training…"); }
  return w.trained ? el("span", "tag trained", "trained")
                   : el("span", "tag untrained", "not trained");
}
var pickerTrainLogPath = "";
function appendLogOffer(node, kind) {
  var path = kind === "train"
    ? (pickerTrainLogPath || (session && session.train_log_path) || "")
    : ((session && session.server_log_path) || "");
  if (path) {
    node.appendChild(document.createTextNode(" "));
    node.appendChild(el("span", "logPath", path));
  }
  node.appendChild(document.createTextNode(" "));
  var btn = el("button", "viewLogBtn", "View log");
  btn.type = "button";
  btn.id = kind === "train" ? "viewTrainLogBtn" : "viewServerLogBtn";
  btn.addEventListener("click", function (event) {
    event.stopPropagation();
    openProcessLog(kind);
  });
  node.appendChild(btn);
}
function pickerStatus(text, cls, offerLog) {
  var node = document.getElementById("pickerStatus");
  node.className = cls || "";
  clear(node);
  node.appendChild(document.createTextNode(text || ""));
  if (offerLog === "train") { appendLogOffer(node, "train"); }
}
function chooseWorkflow(path) {
  pickerStatus("Starting " + path + " …");
  apiPost("/api/select_workflow", { path: path }).then(function (data) {
    session = data.session;
    pickerStatus("");
    applySession();
  }).catch(function (e) {
    pickerStatus("Could not activate the workflow: " + e.message, "err");
  });
}

function chooseWorkspace(path) {
  pickerStatus("Opening read-only workspace " + path + " …");
  apiPost("/api/select_workspace", { path: path }).then(function (data) {
    session = data.session;
    pickerStatus("");
    applySession();
  }).catch(function (e) {
    pickerStatus("Could not open workspace: " + e.message, "err");
  });
}

var pickerPollTimer = null;
var pickerTrainingPaths = {};
function stopPickerPolling() {
  if (pickerPollTimer) {
    clearInterval(pickerPollTimer);
    pickerPollTimer = null;
  }
}
function startPickerPolling() {
  if (document.hidden) { return; }
  if (!pickerPollTimer) {
    pickerPollTimer = setInterval(refreshCandidateList, 2500);
  }
}
function startTrain(path) {
  pickerStatus("Starting training…");
  apiPost("/api/train", { path: path }).then(function (data) {
    pickerTrainLogPath = (data && data.log_path) || pickerTrainLogPath;
    pickerTrainingPaths[path] = true;
    pickerStatus("Training… you can leave the chatbot; it will keep going.");
    refreshCandidateList();
  }).catch(function (e) {
    pickerStatus(e.message, "err");
  });
}
function renderWfItem(w) {
  var item = el("div", "wfItem");
  var grow = el("div", "grow");
  grow.appendChild(el("div", "name", w.name + (w.source === "examples" ? "  (bundled example)" : "")));
  grow.appendChild(el("div", "path", w.path));
  item.appendChild(grow);
  item.appendChild(wfTag(w));
  if (w.trainable) {
    var train = el("button", "primary trainBtn", "Train");
    train.addEventListener("click", function (evt) {
      evt.stopPropagation();
      startTrain(w.path);
    });
    item.appendChild(train);
  }
  makeRowActivatable(item, function () { chooseWorkflow(w.path); });
  return item;
}
function countTreeWorkflows(node) {
  var n = node.items.length;
  Object.keys(node.folders).forEach(function (name) {
    n += countTreeWorkflows(node.folders[name]);
  });
  return n;
}
function workflowTree(wfs) {
  /* Group by rel path so top-level workflows stay flat and folders expand
     with (possibly nested) workflows under them. */
  var root = { folders: {}, items: [] };
  function ensure(node, name) {
    if (!node.folders[name]) {
      node.folders[name] = { name: name, folders: {}, items: [] };
      node.items = node.items.filter(function (w) {
        if (w.name === name) {
          node.folders[name].items.push(w);
          return false;
        }
        return true;
      });
    }
    return node.folders[name];
  }
  wfs.forEach(function (w) {
    var rel = String(w.rel || w.name || "").replace(/\\/g, "/");
    var parts = rel.split("/").filter(function (p) { return p && p !== "."; });
    if (!parts.length) { parts = [w.name]; }
    var node = root;
    for (var i = 0; i < parts.length - 1; i++) {
      node = ensure(node, parts[i]);
    }
    var last = parts[parts.length - 1];
    if (node.folders[last]) {
      node.folders[last].items.push(w);
    } else {
      node.items.push(w);
    }
  });
  return root;
}
function byName(a, b) {
  return String(a).toLowerCase().localeCompare(String(b).toLowerCase());
}
function renderTreeNode(container, node) {
  node.items.slice().sort(function (a, b) { return byName(a.name, b.name); })
    .forEach(function (w) { container.appendChild(renderWfItem(w)); });
  Object.keys(node.folders).sort(byName).forEach(function (name) {
    container.appendChild(renderWfFolder(node.folders[name]));
  });
}
function renderWfFolder(folder) {
  var wrap = el("details", "wfFolder");
  wrap.open = true;
  var summary = el("summary");
  summary.appendChild(el("span", "folderName", folder.name));
  var n = countTreeWorkflows(folder);
  summary.appendChild(el("span", "folderCount",
    n + (n === 1 ? " workflow" : " workflows")));
  wrap.appendChild(summary);
  var body = el("div", "wfFolderBody");
  renderTreeNode(body, folder);
  wrap.appendChild(body);
  return wrap;
}
function renderCandidateList(wfs) {
  var list = document.getElementById("wfCandidates");
  clear(list);
  if (!wfs.length) {
    list.appendChild(el("div", "empty",
      "No workflows found near the launch directory. Browse to one on the right."));
    return;
  }
  renderTreeNode(list, workflowTree(wfs));
}
function refreshCandidateList() {
  api("/api/workflows").then(function (data) {
    var wfs = data.workflows || [];
    renderCandidateList(wfs);
    var still = {};
    wfs.forEach(function (w) {
      if (w.training) { still[w.path] = true; }
      if (pickerTrainingPaths[w.path] && !w.training && w.trained) {
        pickerStatus("Training complete.", "ok");
      } else if (pickerTrainingPaths[w.path] && !w.training && !w.trained) {
        pickerStatus(
          "Training failed — see the train log in the workflow state directory.",
          "err",
          "train"
        );
      }
    });
    pickerTrainingPaths = still;
    if (Object.keys(still).length) {
      startPickerPolling();
    } else {
      stopPickerPolling();
    }
  }).catch(function (e) {
    var list = document.getElementById("wfCandidates");
    clear(list);
    list.appendChild(el("div", "empty", "Could not scan for workflows: " + e.message));
  });
}
function showEnvSetup() {
  var setup = document.getElementById("envSetup");
  setup.className = "visible";
  var missing = [];
  if (!session.env_file_path) { missing.push("fastworkflow.env"); }
  if (!session.passwords_file_path) { missing.push("fastworkflow.passwords.env"); }
  document.getElementById("envSetupText").textContent =
    "Missing " + missing.join(" and ") + " for " + session.workflow_path +
    ". Choose existing files (their contents are copied into the workflow) or " +
    "create both from the bundled templates. Edit placeholder API keys before chatting.";
}

function configureEnv(body) {
  pickerStatus("Preparing environment files and starting the workflow server …");
  apiPost("/api/configure_env", body).then(function (data) {
    session = data.session;
    document.getElementById("envSetup").className = "";
    pickerStatus("");
    applySession();
  }).catch(function (e) {
    pickerStatus("Could not configure environment files: " + e.message, "err");
  });
}
document.getElementById("createEnvBtn").addEventListener("click", function () {
  configureEnv({ create_from_templates: true });
});
document.getElementById("uploadEnvBtn").addEventListener("click", function () {
  var envFile = document.getElementById("envFilePick").files[0];
  var passwordsFile = document.getElementById("passwordsFilePick").files[0];
  if (!envFile && !passwordsFile) {
    pickerStatus("Choose at least one env file first.", "err");
    return;
  }
  Promise.all([
    envFile ? envFile.text() : Promise.resolve(null),
    passwordsFile ? passwordsFile.text() : Promise.resolve(null)
  ]).then(function (contents) {
    configureEnv({ env_content: contents[0], passwords_content: contents[1] });
  }).catch(function (e) {
    pickerStatus("Could not read the selected files: " + e.message, "err");
  });
});
function loadPicker() {
  var list = document.getElementById("wfCandidates");
  clear(list);
  list.appendChild(el("div", "empty", "scanning…"));
  refreshCandidateList();
  browseTo("");
}
function browseTo(dir) {
  api("/api/browse" + (dir ? "?dir=" + encodeURIComponent(dir) : "")).then(function (data) {
    if (data.error) { pickerStatus(data.error, "err"); return; }
    document.getElementById("browsePath").textContent = data.dir;
    var list = document.getElementById("browseList");
    clear(list);
    if (data.parent) {
      var up = el("div", "dirRow");
      up.appendChild(el("span", "dnm", "⬑ up to " + data.parent));
      makeRowActivatable(up, function () { browseTo(data.parent); });
      list.appendChild(up);
    }
    (data.entries || []).forEach(function (entry) {
      var rowEl = el("div", "dirRow");
      var nm = el("span", "dnm", entry.name + (entry.is_workflow ? "  ·  workflow" : ""));
      rowEl.appendChild(nm);
      if (entry.is_workflow) {
        rowEl.appendChild(wfTag(entry));
        var use = el("button", "primary", "Use");
        use.addEventListener("click", function (evt) {
          evt.stopPropagation();
          chooseWorkflow(entry.path);
        });
        rowEl.appendChild(use);
      }
      makeRowActivatable(rowEl, function () { browseTo(entry.path); });
      list.appendChild(rowEl);
    });
    if (!(data.entries || []).length) {
      list.appendChild(el("div", "empty", "No subfolders here."));
    }
    var manifests = document.getElementById("workspaceManifestList");
    clear(manifests);
    if ((data.workspace_manifests || []).length) {
      manifests.appendChild(el("h3", null, "Workspace manifests"));
      (data.workspace_manifests || []).forEach(function (manifest) {
        var rowEl = el("div", "wfItem");
        var grow = el("div", "grow");
        /* The label is the collection folder for a manifest found one level
           down ("exp029-trial-3.3-lifecycle-2026-09-06"), and the file name
           for one sitting in this directory. The full path stays on the row
           below it, so the label never hides which file will be opened. */
        grow.appendChild(el("div", "name", manifest.label || manifest.name));
        grow.appendChild(el("div", "path", manifest.path));
        rowEl.appendChild(grow);
        var use = el("button", "primary", "Open read-only");
        use.addEventListener("click", function () { chooseWorkspace(manifest.path); });
        rowEl.appendChild(use);
        manifests.appendChild(rowEl);
      });
    }
  }).catch(function (e) { pickerStatus("Browse failed: " + e.message, "err"); });
}

