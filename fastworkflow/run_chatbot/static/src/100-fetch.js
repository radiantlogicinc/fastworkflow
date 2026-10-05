"use strict";
/* All record-derived text is rendered via textContent — never via markup
   injection [R22]. HTML-ish artifacts render only inside sandboxed iframes
   (srcdoc, meta CSP default-src 'none'); the parent page CSP also governs
   srcdoc documents. */

var TOKEN = new URLSearchParams(location.search).get("token") || "";

var API_NOT_MODIFIED = { notModified: true };

function chatbotAuthHeaders(extra) {
  var headers = { "Authorization": "Bearer " + TOKEN };
  if (extra) {
    Object.keys(extra).forEach(function (key) {
      headers[key] = extra[key];
    });
  }
  return headers;
}
function api(path, options) {
  options = options || {};
  var headers = chatbotAuthHeaders(options.headers);
  var init = { headers: headers };
  if (options.signal) { init.signal = options.signal; }
  return fetch(path, init)
    .then(function (r) {
      if (r.status === 304 && options.allow304) {
        return API_NOT_MODIFIED;
      }
      /* Read the body even on a failure. Routes that refuse deliberately —
         /api/experiment/<id>/compare answers 409 with the per-problem reason
         the pair is incomparable — put the whole diagnostic in the body, and
         rejecting on status alone threw it away and left the user with a bare
         number. apiPost/apiPatch already did this; api() did not. */
      var etag = r.headers.get("ETag");
      return r.json().then(function (data) {
        if (!r.ok) {
          throw new Error(
            data && data.error ? data.error : ("API " + path + " -> " + r.status));
        }
        if (options.captureETag) {
          return { body: data, etag: etag };
        }
        return data;
      }, function () {
        if (!r.ok) { throw new Error("API " + path + " -> " + r.status); }
        if (options.captureETag) {
          return { body: {}, etag: etag };
        }
        return {};
      });
    });
}
function requestWasAborted(error) {
  return !!(error && (error.name === "AbortError" || error.aborted));
}
/* A refusal whose BODY is the answer. 409 from the compare route is not an
   error to report, it is the comparison result saying "these two are not
   comparable, here is why" — so it resolves rather than rejects. */
function apiAllowing409(path) {
  return fetch(path, { headers: chatbotAuthHeaders() })
    .then(function (r) {
      return r.json().then(function (data) {
        if (!r.ok && r.status !== 409) {
          throw new Error(
            data && data.error ? data.error : ("API " + path + " -> " + r.status));
        }
        return data;
      });
    });
}
function apiRaw(path) {
  return fetch(path, { headers: chatbotAuthHeaders() });
}
function apiPost(path, body) { return mutationRequest(path, "POST", body); }

function reviewApi(path, method, body) {
  return fetch(path, {
    method: method || "GET",
    headers: chatbotAuthHeaders({
      "X-Review-Capability": review.capability,
      "Content-Type": "application/json"
    }),
    body: method === "POST" ? JSON.stringify(body || {}) : undefined
  }).then(function (r) {
    return r.json().then(function (data) {
      if (!r.ok) {
        throw new Error(
          data && data.error ? data.error : ("API " + path + " -> " + r.status));
      }
      return data;
    });
  });
}

function apiPatch(path, body, successMessage) {
  return mutationRequest(path, "PATCH", body, successMessage);
}
function apiPut(path, body) { return mutationRequest(path, "PUT", body); }

function analysisText(value) {
  if (value === null || value === undefined) { return ""; }
  if (typeof value === "string") { return value; }
  if (typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length) { return ""; }
  return JSON.stringify(value, null, 2);
}

/* -- tiny DOM helpers (text always via textContent) -------------------- */
function el(tag, cls, text) {
  var node = document.createElement(tag);
  if (cls) { node.className = cls; }
  if (text !== undefined && text !== null) { node.textContent = String(text); }
  return node;
}
function clear(node) {
  if (node.id === "detail") { detailAttentionPending = true; node.scrollTop = 0; node.setAttribute("aria-busy", "true"); }
  while (node.firstChild) { node.removeChild(node.firstChild); }
}
function makeRowActivatable(row, onActivate) {
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  row.addEventListener("click", onActivate);
  row.addEventListener("keydown", function (event) {
    /* A control inside the row (Train, Use, a durable link) keeps its own
       key. Cancelling Space here is what stops the page from scrolling. */
    if (event.target === row && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      onActivate();
    }
  });
}

/* Theme preference is local to this browser; no remote fonts or assets. */
var themeSelect = document.getElementById("themeSelect");
function setTheme(value) {
  if (value === "auto") { document.documentElement.removeAttribute("data-theme"); }
  else { document.documentElement.dataset.theme = value; }
}
try { themeSelect.value = localStorage.getItem("fw-theme") || "auto"; } catch (error) {}
setTheme(themeSelect.value);
themeSelect.addEventListener("change", function () {
  setTheme(themeSelect.value); try { localStorage.setItem("fw-theme", themeSelect.value); } catch (error) {}
});
document.addEventListener("click", function (event) {
  var menu = document.getElementById("workspaceTools");
  if (!menu.contains(event.target)) { menu.open = false; }
});
document.addEventListener("keydown", function (event) {
  var menu = document.getElementById("workspaceTools");
  if (event.key === "Escape" && menu.open) { menu.open = false; menu.querySelector("summary").focus(); }
});

/* Shared action feedback. Notifications never move focus; the destination view does. */
var lastActionButton = null;
document.addEventListener("click", function (event) {
  lastActionButton = event.target.closest ? event.target.closest("button") : null;
}, true);
/* The band's bottom edge moves: the header wraps at narrow widths and
   #healthBanner sits above it, appearing and disappearing. Measuring it keeps
   the notices tucked under the band instead of over it or adrift below it. */
var noticeBand = document.querySelector("header");
var noticeBanner = document.getElementById("healthBanner");
function placeNotices() {
  /* offsetHeight, not getBoundingClientRect().bottom: the rect is measured
     against the viewport, so a scrolled document would place the notices off
     the top of the screen. A hidden banner contributes 0. */
  document.documentElement.style.setProperty(
    "--notice-top", (noticeBanner.offsetHeight + noticeBand.offsetHeight) + "px");
}
if (window.ResizeObserver) {
  var bandObserver = new ResizeObserver(placeNotices);
  bandObserver.observe(noticeBand);
  bandObserver.observe(noticeBanner);
}
window.addEventListener("resize", placeNotices);
placeNotices();
function showNotice(message, kind, detail) {
  var stack = document.getElementById("noticeStack");
  var notice = el("div", "notice" + (kind === "error" ? " error" : ""));
  notice.setAttribute("role", kind === "error" ? "alert" : "status");
  var icon = el("span", "noticeIcon", kind === "error" ? "!" : "✓"); icon.setAttribute("aria-hidden", "true");
  notice.appendChild(icon);
  var text = el("div", "noticeText"); text.appendChild(el("strong", null, message));
  if (detail) { text.appendChild(el("span", null, detail)); } notice.appendChild(text);
  var dismiss = el("button", null, "×"); dismiss.setAttribute("aria-label", "Dismiss notification");
  notice.appendChild(dismiss); stack.appendChild(notice);
  /* Two phases: the notice sits at full opacity for `dwell`, then fades for
     `fade` and is removed. The removal is a timer, not a transitionend, because
     prefers-reduced-motion disables the transition and the event never fires. */
  var dwell = kind === "error" ? 6000 : 3000, fade = kind === "error" ? 6000 : 1500;
  var timer, remaining = dwell, started, running = false, fading = false;
  function remove() { clearTimeout(timer); notice.remove(); }
  function beginFade() { running = false; fading = true; notice.classList.add("leaving"); timer = setTimeout(remove, fade); }
  /* A hover that arrives mid-fade restores the notice and re-arms the full
     dwell: the reader who reached for it gets the whole message back, not the
     tail of a fade that was already half gone. */
  function pause() {
    if (fading) { clearTimeout(timer); fading = false; notice.classList.remove("leaving"); remaining = dwell; return; }
    if (!running) { return; }
    running = false; clearTimeout(timer); remaining = Math.max(0, remaining - (Date.now() - started));
  }
  function resume() { if (running || fading) { return; } running = true; started = Date.now(); timer = setTimeout(beginFade, remaining); }
  notice.addEventListener("mouseenter", pause);
  notice.addEventListener("mouseleave", function () { if (!notice.contains(document.activeElement)) { resume(); } });
  notice.addEventListener("focusin", pause);
  notice.addEventListener("focusout", function (event) { if (!notice.contains(event.relatedTarget)) { resume(); } });
  dismiss.addEventListener("click", remove); notice.dismissNotice = remove; resume();
  while (stack.children.length > 4) { stack.firstElementChild.dismissNotice(); }
}
function actionSuccess(path, method) {
  if (method === "DELETE") { return "Empty experiment deleted"; }
  if (path.indexOf("/post_feedback") >= 0) { return "Feedback saved"; }
  if (path.indexOf("/analysis") >= 0) { return "Analysis saved"; }
  if (path === "/api/benchmark-setup") { return "Benchmark version saved"; }
  if (/\/benchmarks\/.*\/experiments$/.test(path)) { return "Experiment created"; }
  if (path === "/api/select_workflow") { return "Workflow selected"; }
  if (path === "/api/select_workspace") { return "Workspace opened"; }
  if (path === "/api/train") { return "Training requested"; }
  if (path === "/api/configure_env") { return "Environment files saved"; }
  if (path === "/api/clear_conversations") { return "Conversations cleared"; }
  if (method === "PATCH") { return "Notes saved"; }
  return "Changes saved";
}
function mutationRequest(path, method, body, successMessage) {
  var button = lastActionButton;
  if (button && button.isConnected) { button.disabled = true; button.setAttribute("aria-busy", "true"); }
  var options = {method: method, headers: chatbotAuthHeaders({"Content-Type": "application/json"})};
  if (method !== "DELETE") { options.body = JSON.stringify(body || {}); }
  return fetch(path, options).then(function (response) {
    return response.json().catch(function () { return {}; }).then(function (data) {
      if (!response.ok) { throw new Error(data.error || "Request failed (" + response.status + "). Please try again."); }
      /* A caller may pass a function when the right wording depends on what
         the server reports it actually did. */
      var message = typeof successMessage === "function" ? successMessage(data) : successMessage;
      showNotice(message || actionSuccess(path, method));
      return data;
    });
  }).catch(function (error) {
    showNotice("Could not complete the action", "error", error.message);
    throw error;
  }).finally(function () {
    if (button) { button.disabled = false; button.removeAttribute("aria-busy"); }
  });
}
function apiDelete(path) { return mutationRequest(path, "DELETE"); }

function confirmDialog(options) {
  options = options || {};
  var opener = options.opener || document.activeElement;
  var dialog = document.getElementById("confirmDialog");
  var title = document.getElementById("confirmTitle");
  var description = document.getElementById("confirmDescription");
  var confirmBtn = document.getElementById("confirmDelete");
  var cancelBtn = document.getElementById("cancelDelete");
  var previous = {
    title: title.textContent,
    description: description.textContent,
    confirm: confirmBtn.textContent,
    cancel: cancelBtn.textContent
  };
  if (options.title) { title.textContent = options.title; }
  if (options.description) { description.textContent = options.description; }
  if (options.confirmLabel) { confirmBtn.textContent = options.confirmLabel; }
  if (options.cancelLabel) { cancelBtn.textContent = options.cancelLabel; }
  return new Promise(function (resolve) {
    function finish(value) {
      dialog.removeEventListener("close", onClose);
      dialog.removeEventListener("cancel", onCancel);
      confirmBtn.removeEventListener("click", onDelete);
      cancelBtn.removeEventListener("click", onCancel);
      if (dialog.close) { dialog.close(); } else { dialog.removeAttribute("open"); }
      title.textContent = previous.title;
      description.textContent = previous.description;
      confirmBtn.textContent = previous.confirm;
      cancelBtn.textContent = previous.cancel;
      if (opener && opener.isConnected) { opener.focus({preventScroll: true}); }
      resolve(value);
    }
    function onDelete() { finish(true); }
    function onCancel(event) { if (event) { event.preventDefault(); } finish(false); }
    function onClose() { finish(false); }
    dialog.addEventListener("close", onClose); dialog.addEventListener("cancel", onCancel);
    confirmBtn.addEventListener("click", onDelete);
    cancelBtn.addEventListener("click", onCancel);
    if (dialog.showModal) { dialog.showModal(); } else { dialog.setAttribute("open", ""); }
    cancelBtn.focus();
  });
}
function confirmEmptyDeletion(opener) {
  return confirmDialog({
    opener: opener,
    title: "Delete this empty experiment?",
    description: "This removes the unused experiment. Its benchmark and tasks will stay in place. Create a new experiment if you need it again.",
    confirmLabel: "Delete experiment",
    cancelLabel: "Keep experiment"
  });
}

/* Focus once per replacement, not when comments, polling or analysis arrive. */
var detailAttentionPending = false;
var detailObserver = new MutationObserver(function () {
  if (!detailAttentionPending) { return; }
  var detail = document.getElementById("detail");
  var target = detail.querySelector("[data-autofocus], h1, h2, .err");
  if (!target) { return; }
  detailAttentionPending = false; detail.removeAttribute("aria-busy");
  var active = document.activeElement;
  if (active && active.isConnected && /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName) && active !== target) { return; }
  if (!target.matches("input, textarea, select")) { target.tabIndex = -1; target.setAttribute("data-view-heading", ""); }
  if (!document.getElementById("debugMain").classList.contains("visible")) { return; }
  target.focus({preventScroll: true});
  if (window.matchMedia && window.matchMedia("(max-width: 700px)").matches && target.scrollIntoView) {
    detail.scrollIntoView({block: "start", behavior: "instant"});
  }
  target.classList.remove("viewArrival"); void target.offsetWidth; target.classList.add("viewArrival");
});
detailObserver.observe(document.getElementById("detail"), {childList: true, subtree: true});
function pageHeader(container, eyebrow, title, description) {
  var header = el("div", "pageHead"), intro = el("div", "intro");
  intro.appendChild(el("div", "eyebrow", eyebrow)); intro.appendChild(el("h1", null, title));
  if (description) { intro.appendChild(el("p", null, description)); }
  header.appendChild(intro); var actions = el("div", "actions"); header.appendChild(actions);
  container.appendChild(header); return actions;
}
function sectionHeader(container, title, count) {
  var head = el("div", "sectionHead"), label = el("h2", null, title);
  if (count !== undefined) { label.appendChild(el("span", "count", count)); }
  head.appendChild(label); container.appendChild(head); return head;
}
function emptyState(container, title, message) {
  var box = el("div", "emptyState"); box.appendChild(el("strong", null, title));
  box.appendChild(el("p", null, message)); container.appendChild(box); return box;
}
function taskPreview(container, task, index) {
  var card = el("div", "card taskCard"), body = el("div", "taskBody");
  card.appendChild(el("span", "taskNumber", String(index + 1).padStart(2, "0")));
  body.appendChild(el("div", "recordId", task.task_id));
  body.appendChild(el("p", null, benchmarkPrompt(task) || "No prompt yet. Your runner supplies the task input."));
  card.appendChild(body); container.appendChild(card);
}

