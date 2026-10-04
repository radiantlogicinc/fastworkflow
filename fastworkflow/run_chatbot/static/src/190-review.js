/* -- capability-gated formal review ----------------------------------- */
function reviewAnswerMap(progress) {
  var answers = {};
  (progress.current_answers || []).forEach(function (item) {
    if (!answers[item.row_id]) { answers[item.row_id] = {}; }
    answers[item.row_id][item.question_id] = item.answer;
  });
  return answers;
}

function reviewAnsweredCount() {
  return Object.keys(review.captured).reduce(function (total, rowId) {
    return total + Object.keys(review.captured[rowId] || {}).length;
  }, 0);
}

function reviewStatusText(node, text, cls) {
  node.className = "reviewSave" + (cls ? " " + cls : "");
  node.textContent = text;
}

function queueReviewAnswer(rowId, questionId, answer, statusNode) {
  if (!review.answers[rowId]) { review.answers[rowId] = {}; }
  review.answers[rowId][questionId] = answer;
  delete review.dirty[questionId];
  reviewStatusText(statusNode, "saving…");
  var path = "/api/review/assignments/" + encodeURIComponent(review.assignmentId)
    + "/answers";
  var previous = review.pending[questionId] || Promise.resolve();
  /* Prior answer save may have failed; still attempt the latest value. */
  var operation = previous.catch(function () {}).then(function () {
    return reviewApi(path, "POST", {
      row_id: rowId,
      question_id: questionId,
      answer: answer
    });
  }).then(function (data) {
    if (!review.captured[rowId]) { review.captured[rowId] = {}; }
    review.captured[rowId][questionId] = data.answer.answer;
    reviewStatusText(statusNode,
      "saved · revision " + data.answer.revision, "ok");
    renderReviewProgress();
    return data;
  }).catch(function (error) {
    review.dirty[questionId] = true;
    reviewStatusText(statusNode, "not saved: " + error.message, "err");
    throw error;
  });
  review.pending[questionId] = operation;
  /* Rejection is already shown on the status node; keep the chain settled. */
  operation.catch(function () {});
  return operation;
}

function saveReviewDirty() {
  if (!review.progress) { return []; }
  var row = review.progress.assignment.rows[review.rowIndex];
  var jobs = [];
  Object.keys(review.dirty).forEach(function (questionId) {
    if (!review.dirty[questionId]) { return; }
    var status = document.getElementById("review-save-" + questionId);
    jobs.push(queueReviewAnswer(
      row.id, questionId, (review.answers[row.id] || {})[questionId], status));
  });
  return jobs;
}

function flushReviewCaptures() {
  var jobs = saveReviewDirty();
  Object.keys(review.pending).forEach(function (questionId) {
    if (jobs.indexOf(review.pending[questionId]) === -1) {
      jobs.push(review.pending[questionId]);
    }
  });
  return Promise.all(jobs);
}

function renderReviewProgress() {
  if (!review.progress) { return; }
  var node = document.getElementById("reviewProgress");
  if (!node) { return; }
  node.textContent = reviewAnsweredCount() + " of "
    + review.progress.total_answers + " answers captured";
}

function reviewHash(row) {
  var ref = row.turn_ref || {};
  var values = [
    "review=" + encodeURIComponent(review.assignmentId),
    "review_capability=" + encodeURIComponent(review.capability),
    "row=" + encodeURIComponent(row.id)
  ];
  if (ref.store_id && ref.logical_turn_key) {
    values.push("store=" + encodeURIComponent(ref.store_id));
    values.push("turn=" + encodeURIComponent(ref.logical_turn_key));
  }
  return values.join("&");
}

function openReviewRow(index) {
  var assignment = review.progress.assignment;
  if (index < 0 || index >= assignment.rows.length) { return; }
  review.rowIndex = index;
  review.dirty = {};
  var row = assignment.rows[index];
  location.hash = reviewHash(row);
  renderReviewPane();
  var ref = row.turn_ref || {};
  if (ref.store_id && ref.logical_turn_key) {
    selectWorkspaceTurn(ref.store_id, ref.logical_turn_key);
  } else {
    var d = document.getElementById("detail");
    clear(d);
    d.appendChild(el("div", "empty",
      "This workspace review row has no scoped store_id/logical_turn_key trace."));
  }
}

function navigateReviewRow(offset) {
  flushReviewCaptures().then(function () {
    openReviewRow(review.rowIndex + offset);
  }).catch(function () {
    /* Stay on the row whose answer failed so the rater can retry it. */
  });
}

function renderReviewQuestion(container, row, question) {
  var box = el("section", "reviewQuestion");
  box.appendChild(el("strong", null, question.prompt));
  var current = (review.answers[row.id] || {})[question.id];
  var status = el("div", "reviewSave", current === undefined ? "" : "saved");
  status.id = "review-save-" + question.id;
  if (question.type === "single-select") {
    var select = el("select");
    select.appendChild(el("option", null, "Choose…"));
    select.firstChild.value = "";
    (question.vocabulary || []).forEach(function (value) {
      var option = el("option", null, value);
      option.value = value;
      select.appendChild(option);
    });
    select.value = current === undefined ? "" : current;
    select.addEventListener("change", function () {
      if (select.value) {
        queueReviewAnswer(row.id, question.id, select.value, status);
      }
    });
    box.appendChild(select);
  } else if (question.type === "multi-select") {
    var fieldset = el("fieldset");
    (question.vocabulary || []).forEach(function (value) {
      var label = el("label");
      var checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = value;
      checkbox.checked = Array.isArray(current) && current.indexOf(value) >= 0;
      checkbox.addEventListener("change", function () {
        var selected = Array.prototype.slice.call(
          fieldset.querySelectorAll("input:checked")).map(function (input) {
          return input.value;
        });
        queueReviewAnswer(row.id, question.id, selected, status);
      });
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(" " + value));
      fieldset.appendChild(label);
    });
    box.appendChild(fieldset);
  } else {
    var textarea = el("textarea");
    textarea.maxLength = question.max_length;
    textarea.value = current === undefined ? "" : current;
    textarea.addEventListener("input", function () {
      if (!review.answers[row.id]) { review.answers[row.id] = {}; }
      review.answers[row.id][question.id] = textarea.value;
      review.dirty[question.id] = true;
      reviewStatusText(status, "not yet saved");
    });
    textarea.addEventListener("blur", function () {
      if (review.dirty[question.id]) {
        queueReviewAnswer(row.id, question.id, textarea.value, status);
      }
    });
    box.appendChild(textarea);
    box.appendChild(el("div", "sub",
      "Maximum " + question.max_length + " characters."));
  }
  box.appendChild(status);
  container.appendChild(box);
}

function renderReviewPane() {
  var pane = document.getElementById("reviewPane");
  clear(pane);
  var assignment = review.progress.assignment;
  var row = assignment.rows[review.rowIndex];
  var head = el("div", "reviewHead");
  head.appendChild(el("h2", null, "Formal review · " + assignment.id));
  head.appendChild(el("div", "sub",
    "Independent human rating · separate from developer/agent feedback · rubric row "
    + row.id));
  var progress = el("div", "sub");
  progress.id = "reviewProgress";
  head.appendChild(progress);
  var nav = el("div", "reviewNav");
  var previous = el("button", null, "← Previous");
  previous.disabled = review.rowIndex === 0;
  previous.addEventListener("click", function () { navigateReviewRow(-1); });
  var position = el("span", "sub",
    (review.rowIndex + 1) + " of " + assignment.rows.length);
  var next = el("button", null, "Next →");
  next.disabled = review.rowIndex === assignment.rows.length - 1;
  next.addEventListener("click", function () { navigateReviewRow(1); });
  nav.appendChild(previous);
  nav.appendChild(position);
  nav.appendChild(next);
  head.appendChild(nav);
  pane.appendChild(head);
  (assignment.questions || []).forEach(function (question) {
    renderReviewQuestion(pane, row, question);
  });
  renderReviewProgress();
}

function loadReviewAssignment() {
  if (!review.assignmentId || !review.capability) { return; }
  var pane = document.getElementById("reviewPane");
  pane.className = "visible";
  pane.dataset.blinded = "";
  document.getElementById("debugMain").classList.add("reviewActive");
  renderRecordNavigator();
  pane.appendChild(el("div", "empty", "loading review assignment…"));
  reviewApi("/api/review/assignments/"
    + encodeURIComponent(review.assignmentId) + "/progress", "GET")
    .then(function (data) {
      review.progress = data.progress;
      review.answers = reviewAnswerMap(review.progress);
      review.captured = reviewAnswerMap(review.progress);
      /* Hook only: fix-9eg.14 must enforce blinded suppression server-side. */
      pane.dataset.blinded = String(!!review.progress.assignment.blinded);
      var requestedRow = initialHashParams.get("row");
      var index = review.progress.assignment.rows.findIndex(function (row) {
        return row.id === requestedRow;
      });
      openReviewRow(index >= 0 ? index : 0);
    }).catch(function (error) {
      clear(pane);
      pane.appendChild(el("div", "empty",
        "Could not load review assignment: " + error.message));
    });
}

