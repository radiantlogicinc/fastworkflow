/* -- recorded command outcomes, per side [fix-9eg.3.1.3] ------------------ */
/* The server's `command_summary` reducer, rendered. Deliberately not
   re-derived here: the default comparison view does not load the steps at all,
   and a second tally on the page is exactly how a chip and an agent's
   structured read come to disagree about one recorded execution.

   Every number is per DISPATCH. A turn that contained a failure does not make
   every command in it a failure, so opening a group lists the exact dispatches
   that were counted -- it does not filter the turn by command name, which
   would hand back every failure in every turn the group appears in. */

function commandOutcomeWord(value) {
  if (value === true) { return "recorded success"; }
  if (value === false) { return "recorded failure"; }
  /* Not "no failure". Nothing recorded an outcome for this dispatch, which is
     a fact about the recording and not about the command. */
  return "outcome not recorded";
}

function commandGroupName(group) {
  return group.unknown_command
    ? "(no command name recorded)" : policedText(group.command_name);
}

/* Recorded dispatch timing [fix-9eg.3.1.4].

   Three order statistics over the durations that WERE recorded, and never a
   total: a dispatch's recorded duration already contains the inner dispatches
   recorded beneath it, so a sum of these would read as elapsed time and is
   not one. A group nothing timed says so instead of showing zeroes. */
function fmtCommandDuration(duration) {
  if (!duration || !duration.timed) { return "not timed"; }
  if (duration.timed === 1) { return fmtNs(duration.median_ns) + " (one dispatch)"; }
  return fmtNs(duration.min_ns) + " / " + fmtNs(duration.median_ns)
    + " / " + fmtNs(duration.max_ns);
}

function fmtCommandTimedCoverage(duration) {
  if (!duration) { return "—"; }
  if (!duration.untimed) { return fmtCount(duration.timed) + " timed"; }
  /* Untimed is untimed. Folding it into the timed count would put dispatches
     nobody measured inside a median. */
  return fmtCount(duration.timed) + " timed, "
    + fmtCount(duration.untimed) + " not"
    + (duration.invalid
        ? " (" + fmtCount(duration.invalid) + " recorded an unusable duration)"
        : "");
}

function commandHeadline(summary) {
  var totals = summary.totals || {};
  var outcomes = totals.response_success || {};
  return fmtCount(totals.dispatches || 0)
    + ((totals.dispatches === 1) ? " dispatch" : " dispatches")
    + " · " + fmtCount(totals.groups || 0)
    + ((totals.groups === 1) ? " command" : " commands")
    + " · " + fmtCount(outcomes.true || 0) + " succeeded, "
    + fmtCount(outcomes.false || 0) + " failed, "
    + fmtCount(outcomes.unknown || 0) + " not recorded";
}

/* What this side's numbers are about, and what they are NOT about. An
   unreadable turn is missing coverage; presenting the counts without it would
   let "0 failures" stand for a run half of which could not be read. */
/* `labels` lets a caller that is NOT one side of one pair say so. The
   defaults are this pane's own words; an aggregate over several runs passes
   its own, because "One run, on its own" and "on this side" would each be a
   false statement about a pooled population. */
function appendCommandCoverage(box, summary, labels) {
  var observed = (summary.scope && summary.scope.observed) || {};
  var coverage = summary.coverage || {};
  var totals = summary.totals || {};
  var named = (labels && labels.subject) || "this run";
  var where = (labels && labels.where) || "on this side";
  box.appendChild(el("div", "sub", (labels && labels.headline) || (
    "One run, on its own: " + fmtCount(observed.turns || 0)
    + " recorded turn(s) examined, counted in " + (summary.unit || "dispatches")
    + " · " + summary.metric_version)));
  /* Two different questions, and the second is the one that misleads. Every
     turn being readable does not make every dispatch inside them recorded, so
     a run can enumerate completely and still be missing the spans, the record
     entries or the outcomes the counts above are made of. */
  if (coverage.enumeration_complete === false) {
    box.appendChild(el("div", "chipMarker unknown",
      fmtCount(coverage.turns_unreadable || 0) + " turn(s) " + named
      + " names could not be read, so these counts describe what was examined "
      + "— not a claim that the unread turns recorded no failure"));
  }
  if (coverage.capture === "none") {
    box.appendChild(el("div", "chipMarker unknown",
      "No dispatch was observed " + where + " at all, so there is nothing "
      + "here whose recording could be called complete — this is not a finding "
      + "that " + named + " dispatched nothing"));
  } else if (coverage.capture === "partial") {
    var gaps = el("div", "chipMarker unknown",
      "Partially recorded, so a zero above is a zero among what was recorded:");
    box.appendChild(gaps);
    (coverage.capture_gaps || []).forEach(function (gap) {
      box.appendChild(el("div", "sub", "— " + gap));
    });
  }
  if (totals.unknown_command_dispatches) {
    box.appendChild(el("div", "chipMarker unknown",
      fmtCount(totals.unknown_command_dispatches) + " dispatch(es) recorded no "
      + "command name and are grouped on their own rather than guessed at"));
  }
  if (totals.conflicts) {
    box.appendChild(el("div", "chipMarker warn",
      fmtCount(totals.conflicts) + " dispatch(es) where the turn record and the "
      + "span disagree about the outcome; both are kept and neither is "
      + "overwritten"));
  }
  var timing = totals.duration;
  if (timing) {
    box.appendChild(el("div", "sub",
      "Timing: " + fmtCount(timing.timed || 0) + " of "
      + fmtCount((timing.timed || 0) + (timing.untimed || 0))
      + " dispatch(es) recorded a usable duration · " + timing.metric_version
      + " · " + timing.unit
      + ". Each duration is INCLUSIVE — it covers the inner dispatches recorded "
      + "beneath it — so these are not added up, and they are neither elapsed "
      + "time, critical-path time nor time with the wait for a person removed."));
  }
}

/* The dispatches a group counted, each openable where it was recorded.

   The link is the same `openPairTurn` the answer and alignment rows use, so a
   drill-down lands in the current trace/feedback navigation with this side's
   own source rather than a new one.

   `side` may be a FUNCTION of the contributor. A pair has one source per
   side, but a group pooled over several runs has one per contributor, and
   resolving it per row is what keeps "open this dispatch" opening the run
   that recorded it instead of whichever run the pane was labelled with.

   `ctx.beforeOpen(contributor, note, open)`, when a caller supplies it, runs
   between the click and the navigation. The aggregate uses it to re-check
   that the run still records what was counted, so a drill-down cannot
   silently show evidence that no longer supports the totals above it. */
function appendCommandContributors(cell, group, sides, ctx) {
  cell.appendChild(el("div", "sub",
    "The exact dispatches counted in this row:"));
  (group.contributors || []).forEach(function (contributor) {
    var row = el("div", "sub");
    row.setAttribute("data-command-contributor", contributor.command_call_id);
    row.appendChild(el("span", "mono", contributor.command_call_id));
    row.appendChild(el("span", null,
      " · turn " + contributor.turn_index + " · step " + contributor.position
      + " · " + commandOutcomeWord(contributor.response_success)
      /* The observation the group's statistics were computed from, kept on the
         row: a reader can check the median against the dispatches, and an
         untimed one says so rather than showing 0. */
      + " · " + (contributor.duration_recorded
          ? fmtNs(contributor.duration_ns) + " inclusive"
          : (contributor.duration_invalid
              ? "recorded an unusable duration" : "not timed"))));
    if (contributor.conflict) {
      row.appendChild(el("span", "chipMarker warn",
        "the span recorded " + contributor.span_outcome + " for this same "
        + "dispatch"));
    }
    if (!contributor.span_recorded) {
      row.appendChild(el("span", "chipMarker unknown", "no span recorded"));
    }
    if (!contributor.in_record) {
      row.appendChild(el("span", "chipMarker unknown", "not in the turn record"));
    }
    if (contributor.child_call) {
      row.appendChild(el("span", "pill", "inner call"));
    }
    if (contributor.pass_id) {
      row.appendChild(el("span", "pill", "pass " + contributor.pass_id));
    }
    /* The EXACT dispatch, not its turn, wherever the trace recorded a span
       for it: `openPairSpan` focuses the span through the same source-scoped
       navigation `openPairTurn` uses. A dispatch the trace never held has no
       span to focus, so that row opens the turn and SAYS it is doing that
       rather than presenting a turn-scoped landing as the dispatch. */
    var focusable = !!contributor.span_id;
    var open = el("button", "evidenceLink", focusable
      ? "Open this dispatch" : "Open the recorded turn");
    open.type = "button";
    var note = el("div", "sub", "");
    if (!focusable) {
      row.appendChild(el("span", "chipMarker unknown",
        "no span was recorded for this dispatch, so this opens the whole turn"));
    }
    open.addEventListener("click", function () {
      /* One side per pane is a pair's shape, not a pooled group's: resolved
         per row when the caller passes a function, so a dispatch opens in the
         run that recorded it. */
      var side = (typeof sides === "function") ? sides(contributor) : sides;
      var go = function () {
        if (focusable) {
          openPairSpan(ctx, side, contributor.turn_key, contributor.span_id, note);
          return;
        }
        openPairTurn(ctx, side, contributor.turn_key, note);
      };
      if (ctx && typeof ctx.beforeOpen === "function") {
        ctx.beforeOpen(contributor, note, go);
        return;
      }
      go();
    });
    row.appendChild(open);
    row.appendChild(note);
    cell.appendChild(row);
  });
}

function commandGroupRows(group, side, ctx) {
  var outcomes = group.response_success || {};
  var tr = el("tr", "openable");
  tr.setAttribute("data-command-group", group.unknown_command
    ? "" : String(group.command_name));
  tr.appendChild(el("td", null, commandGroupName(group)));
  tr.appendChild(el("td", null, fmtCount(group.dispatches)));
  tr.appendChild(el("td", null, fmtCount(outcomes.true || 0)));
  tr.appendChild(el("td", null, fmtCount(outcomes.false || 0)));
  tr.appendChild(el("td", null, fmtCount(outcomes.unknown || 0)));
  tr.appendChild(el("td", null, group.conflict_count
    ? fmtCount(group.conflict_count) : "—"));
  tr.appendChild(el("td", null, fmtCommandDuration(group.duration)));
  tr.appendChild(el("td", null, fmtCommandTimedCoverage(group.duration)));
  var detail = el("tr", "child");
  detail.hidden = true;
  var cell = el("td");
  cell.colSpan = 8;
  detail.appendChild(cell);
  var loaded = false;
  function toggle() {
    if (!loaded) {
      appendCommandContributors(cell, group, side, ctx);
      loaded = true;
    }
    detail.hidden = !detail.hidden;
  }
  tr.title = "show the dispatches this row counted";
  tr.tabIndex = 0;
  tr.addEventListener("click", toggle);
  tr.addEventListener("keydown", function (event) {
    if (event.target === tr && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      toggle();
    }
  });
  return [tr, detail];
}

function renderCommandSide(label, side, summary, ctx) {
  var box = el("details", "artifactPreview");
  box.setAttribute("data-command-summary", label.toLowerCase());
  box.appendChild(el("summary", null, label + " — " + commandHeadline(summary)));
  appendCommandCoverage(box, summary);
  var groups = summary.groups || [];
  if (!groups.length) {
    box.appendChild(el("div", "empty",
      "No command dispatch is recorded on this side."));
    return box;
  }
  var wrap = el("div", "ledgerWrap");
  var table = el("table", "ledger");
  var head = el("tr");
  ["command", "dispatches", "succeeded", "failed", "not recorded",
   "record vs span", "inclusive min / median / max", "timed"]
    .forEach(function (column) {
      head.appendChild(el("th", null, column));
    });
  table.appendChild(head);
  groups.forEach(function (group) {
    commandGroupRows(group, side, ctx).forEach(function (row) {
      table.appendChild(row);
    });
  });
  wrap.appendChild(table);
  box.appendChild(wrap);
  return box;
}

function renderCommandSummary(container, cmp, ctx) {
  var sides = [["Left", "left"], ["Right", "right"]].filter(function (side) {
    var projection = side[1] === "left" ? cmp.left : cmp.right;
    return projection && projection.command_summary;
  });
  if (!sides.length) { return; }
  container.appendChild(el("h2", null, "Commands"));
  container.appendChild(el("div", "sub",
    "Grouped by the command name the evidence recorded, per dispatch, and "
    + "independently for each side: these are two runs' tallies, never one "
    + "pooled figure, and the better answer is still the one above."));
  sides.forEach(function (side) {
    var projection = side[1] === "left" ? cmp.left : cmp.right;
    container.appendChild(renderCommandSide(
      side[0], pairSide(cmp, side[1]), projection.command_summary, ctx));
  });
}

function renderComparisonSummary(container, cmp) {
  var summary = cmp.summary || {};
  var kv = el("dl", "kv");
  function pair(k, v) {
    kv.appendChild(el("dt", null, k));
    kv.appendChild(el("dd", null,
      v === null || v === undefined ? "—" : String(v)));
  }
  pair("aligned rows", summary.pairs);
  pair("matched", summary.matched);
  pair("ambiguous", summary.ambiguous);
  pair("left only", summary.left_only);
  pair("right only", summary.right_only);
  pair("steps the evidence does not identify", summary.unknown);
  pair("matched by a recorded alignment", summary.recorded_matches);
  pair("differences", cmp.difference_count);
  container.appendChild(el("h2", null, "Plan and execution"));
  container.appendChild(kv);
  container.appendChild(el("div", "sub",
    "Rows are matched from recorded command identity, context and parameters — "
    + "never from list position, so an inserted or repeated call shows up as a "
    + "gap instead of being paired up to make the two lists the same length."));
  renderComparisonUsage(container, cmp);
}

/* What each side spent, from the projection's own roll-up (fix-9eg.5/.6).

   Read off `usage`, not re-derived: the comparison pane never loads the spans,
   and a second tally here is exactly how a chip and an agent's structured read
   come to disagree about one recorded execution. A side with no recorded LLM
   call says so; a difference in cache hits is reported and nothing is
   concluded from it, because being answered from the cache is not a reason to
   prefer or reject a run. */
function renderComparisonUsage(container, cmp) {
  var sides = [["Left", cmp.left], ["Right", cmp.right]].filter(function (side) {
    return side[1] && side[1].usage;
  });
  if (!sides.length) { return; }
  container.appendChild(el("h2", null, "Tokens, cost and cache"));
  sides.forEach(function (side) {
    var box = el("div", "sub");
    box.appendChild(el("strong", null, side[0] + ": "));
    appendUsageChips(box, side[1].usage);
    container.appendChild(box);
  });
  var hits = sides.map(function (side) {
    return (side[1].usage.cache || {}).hit || 0;
  });
  if (sides.length === 2 && hits[0] !== hits[1]) {
    container.appendChild(el("div", "sub",
      "The two sides were served from the LLM cache a different number of "
      + "times (" + hits[0] + " versus " + hits[1] + "). That is an "
      + "observation about how each answer arrived; it is not a difference in "
      + "what either run did, and it does not decide between them."));
  }
}

/* The expensive recorded LLM calls of each side are rendered by
   `renderComparisonCallCosts`, which lives below the pair composer rather than
   here beside its neighbours: it orders one side's calls by the cost the server
   published for each of them, and this region is pinned to derive no ordering of
   its own, because the step alignment and every reference on the compare view
   must come from the recorded evidence and nothing else. */

function renderAlignment(container, cmp, ctx) {
  if (!cmp.alignment) {
    var ask = el("button", null, "Show the aligned steps");
    ask.type = "button";
    ask.addEventListener("click", function () {
      taskCompare.view = "steps";
      ctx.reload();
    });
    container.appendChild(el("div", "sub",
      "The step alignment of a long run is large, so it is not loaded until "
      + "somebody opens it."));
    container.appendChild(ask);
    return;
  }
  var rows = cmp.alignment.rows || [];
  if (!rows.length) {
    container.appendChild(el("div", "empty", taskCompare.view === "differences"
      ? "These two runs differ in no recorded step."
      : "No step is recorded on either side."));
    return;
  }
  var list = el("div");
  container.appendChild(list);
  rows.forEach(function (row, index) {
    list.appendChild(renderAlignmentRow(row, index, cmp, ctx));
  });
}

/* One aligned row, saying exactly how much it claims.

   A one-sided row says which side it is on rather than pairing it with
   something. An ambiguous row is not a soft match: the recorded evidence
   admits more than one correspondence, so both sides are shown WITHOUT
   asserting they are the same call. A step the trace has no span for can only
   be commented on at turn scope, and the row says so instead of offering a
   finer anchor the store would refuse. */
function renderAlignmentRow(row, index, cmp, ctx) {
  var item = el("div", "compareRow" + (row.ambiguous ? " rowAmbiguous" : ""));
  var meta = el("div", "rowMeta");
  meta.appendChild(el("span", "pill", compareKindLabel(row.kind)));
  meta.appendChild(el("span", "sub", compareBasisLabel(row.basis)));
  if (row.ambiguous) {
    meta.appendChild(el("span", "chipMarker warn",
      "ambiguous — the evidence admits more than one match"));
    if (row.ambiguity_reason) {
      meta.appendChild(el("span", "sub", row.ambiguity_reason));
    }
  }
  var anchors = row.anchors || {};
  ["left", "right"].forEach(function (side) {
    var anchor = anchors[side];
    if (anchor && anchor.anchorable === false) {
      meta.appendChild(el("span", "chipMarker unknown",
        side + ": no recorded span for this hop, so a comment here is "
        + "anchored to the whole turn"));
    }
  });
  item.appendChild(meta);
  [["left", row.left, cmp.left_run], ["right", row.right, cmp.right_run]]
    .forEach(function (side) {
      item.appendChild(renderAlignmentSide(side[0], side[1], side[2], ctx));
    });
  var actions = el("div", "rowMeta");
  var comment = el("button", "evidenceLink", "Comment on this step");
  comment.type = "button";
  comment.setAttribute("data-compare-comment", String(index));
  comment.addEventListener("click", function () {
    if (comment.dataset.open) { return; }
    comment.dataset.open = "1";
    var named = (row.left && row.left.command_name)
      || (row.right && row.right.command_name);
    renderPairComposer(item, cmp, ctx, row,
      named ? "step " + named : "an unnamed step of this pair");
  });
  actions.appendChild(comment);
  item.appendChild(actions);
  return item;
}

function renderAlignmentSide(which, step, run, ctx) {
  var side = pairSide(ctx.comparison, which);
  var box = el("div", "side" + (step ? "" : " absent"));
  if (!step) {
    box.textContent = which === "left"
      ? "not recorded on the left run" : "not recorded on the right run";
    return box;
  }
  box.appendChild(el("div", "title", step.command_name || "(no command name)"));
  var sub = "turn " + step.turn_index + " · step " + step.position
    + (step.context ? " · " + step.context : "")
    + (step.status ? " · " + step.status : "")
    + (step.child_call ? " · inner call" : "");
  box.appendChild(el("div", "sub", sub));
  if (step.pass_id) { box.appendChild(el("span", "pill", "pass " + step.pass_id)); }
  /* The span drilldown: the recorded identity of this dispatch, plus its
     parameters exactly as one of the two recording sources holds them. */
  var detail = el("details");
  detail.appendChild(el("summary", null, "Recorded detail"));
  var kv = el("dl", "kv");
  function pair(k, v) {
    kv.appendChild(el("dt", null, k));
    kv.appendChild(el("dd", null,
      v === null || v === undefined ? "not recorded" : String(v)));
  }
  pair("command call", step.command_call_id);
  pair("parent call", step.parent_call_id);
  pair("span", step.span_id);
  pair("recorded in the turn record", step.in_record);
  pair("recorded as a span", step.span_recorded);
  pair("asked the user", step.asked_user);
  pair("duration", step.duration_ns === null || step.duration_ns === undefined
    ? null : fmtNs(step.duration_ns));
  pair("parameters source", step.parameters_source);
  detail.appendChild(kv);
  if (step.parameters !== null && step.parameters !== undefined) {
    detail.appendChild(el("pre", "json", pretty(step.parameters)));
  }
  var open = el("button", "evidenceLink", "Open the recorded turn");
  open.type = "button";
  var linkNote = el("div", "sub", "");
  open.addEventListener("click", function () {
    openPairTurn(ctx, side, step.turn_key, linkNote);
  });
  detail.appendChild(open);
  detail.appendChild(linkNote);
  box.appendChild(detail);
  return box;
}

/* -- a comment written from a comparison ------------------------------- */

/* The taxonomy picker, exactly the vocabulary observability/feedback.py
   defines. One category and one of its subcategories at a time, with a roving
   tabindex and the subcategory's own watermark in the box — because the six
   confirmed meanings live at the subcategory level. */
function taxonomyPicker(container, area, selected, labelFor) {
  var tabs = el("div", "feedbackTabs");
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Feedback category");
  var subTabs = el("div", "feedbackTabs");
  subTabs.setAttribute("role", "tablist");
  subTabs.setAttribute("aria-label", "Feedback subcategory");
  var categoryButtons = {};
  function selectSubcategory(value) {
    selected.subcategory = value;
    var sub = feedbackSubcategory(selected.category, value);
    area.placeholder = sub ? sub.watermark : "";
    area.setAttribute("aria-label",
      (sub ? sub.label : "Comment") + " for " + labelFor);
    Array.prototype.forEach.call(subTabs.childNodes, function (button) {
      var on = button.getAttribute("data-value") === value;
      button.className = on ? "active" : "";
      button.setAttribute("aria-selected", on ? "true" : "false");
      button.tabIndex = on ? 0 : -1;
    });
  }
  function selectCategory(value) {
    selected.category = value;
    var category = feedbackCategory(value);
    Object.keys(categoryButtons).forEach(function (key) {
      var on = key === value;
      categoryButtons[key].className = on ? "active" : "";
      categoryButtons[key].setAttribute("aria-selected", on ? "true" : "false");
      categoryButtons[key].tabIndex = on ? 0 : -1;
    });
    clear(subTabs);
    category.subcategories.forEach(function (sub) {
      var button = el("button", null, sub.label);
      button.type = "button";
      button.setAttribute("data-value", sub.value);
      button.setAttribute("role", "tab");
      button.addEventListener("click", function () {
        selectSubcategory(sub.value);
        area.focus();
      });
      subTabs.appendChild(button);
    });
    selectSubcategory(category.subcategories[0].value);
  }
  FEEDBACK_TAXONOMY.forEach(function (category) {
    var button = el("button", null, category.label);
    button.type = "button";
    button.setAttribute("data-value", category.value);
    button.setAttribute("role", "tab");
    categoryButtons[category.value] = button;
    button.addEventListener("click", function () {
      selectCategory(category.value);
    });
    tabs.appendChild(button);
  });
  var composer = el("div", "feedbackComposer");
  composer.appendChild(subTabs);
  composer.appendChild(area);
  container.appendChild(tabs);
  container.appendChild(composer);
  selectCategory(FEEDBACK_TAXONOMY[0].value);
}

/* A comment about a comparison, in the ordinary vocabulary.

   There is no comparison schema and no verdict workflow: this posts the SAME
   row the turn composer posts, to the same route, with the same six
   subcategories and the same free-form text. The only thing it adds is the
   pair — and it adds it by handing back the anchors the API returned,
   including each side's whole reference, so the comment is filed against the
   exact pair that was on screen rather than a reference rebuilt here.

   `row` is an alignment row for a step comment, or null for a comment about
   the pair as a whole, which anchors at turn scope on both sides. */
function renderPairComposer(container, cmp, ctx, row, labelFor) {
  var anchors = row ? (row.anchors || {}) : pairTurnAnchors(cmp);
  var card = el("div", "card feedbackCard");
  card.appendChild(el("h2", null, "Comment on " + labelFor));
  if (!anchors.left && !anchors.right) {
    card.appendChild(el("div", "sub",
      "Neither side of this row names a recorded turn, so there is nothing to "
      + "anchor a comment to."));
    container.appendChild(card);
    return;
  }
  var sides = describePairSides(cmp, anchors);
  card.appendChild(el("div", "sub", sides.text));
  if (sides.fallback) {
    card.appendChild(el("div", "chipMarker unknown",
      "no recorded span for one side of this hop, so that side is anchored to "
      + "its whole turn"));
  }
  var area = el("textarea", "expNotes");
  area.maxLength = 100000;
  area.setAttribute("data-compare-comment-box", "1");
  var selected = { category: FEEDBACK_TAXONOMY[0].value,
                   subcategory: FEEDBACK_TAXONOMY[0].subcategories[0].value };
  taxonomyPicker(card, area, selected, labelFor);
  var save = el("button", "primary", "Save this comment");
  save.type = "button";
  var note = el("div", "sub", "");
  card.appendChild(save);
  card.appendChild(note);
  var history = el("div");
  card.appendChild(history);
  container.appendChild(card);
  /* Which side the comment is anchored to decides where it is filed, so the
     scope is worked out once, here, from the same authorized source the reads
     use — not from whichever database the page is pointed at. */
  var primary = anchors.left || anchors.right;
  var paired = (anchors.left && anchors.right) ? anchors.right : null;
  var primarySide = pairSide(cmp, anchors.left ? "left" : "right");
  var scope = pairWriteScope(ctx, primarySide, primary);
  if (scope.refusal) {
    save.disabled = true;
    note.textContent = scope.refusal;
    return;
  }
  if (scope.annotated) {
    /* Sealed evidence is never appended to, and that is not a reason to refuse
       the comment: it is filed beside the archive and read back with it. */
    note.textContent = "The evidence is read-only, so this comment is recorded "
      + "alongside it without changing it.";
  }
  save.addEventListener("click", function () {
    if (!area.value.trim()) {
      note.textContent = "Enter a comment before saving.";
      return;
    }
    save.disabled = true;
    note.textContent = "saving…";
    var path = "/post_feedback?turn_key="
      + encodeURIComponent(primary.turn_key) + scope.query;
    var body = {
      /* The anchor the API returned, verbatim: the whole reference plus the
         turn the comment is anchored to. Narrowing it here would give the
         comment a different pair identity from the comparison it was written
         in, and the earlier comments on that pair would stop matching. */
      ref: primary.ref,
      target_kind: primary.anchorable === false ? "turn" : primary.target_kind,
      span_ids: primary.anchorable === false ? [] : (primary.span_ids || []),
      target_label: labelFor,
      provenance: "human",
      category: selected.category,
      subcategory: selected.subcategory,
      comment: area.value
    };
    if (paired) {
      body.paired = {
        ref: paired.ref,
        turn_key: paired.turn_key,
        target_kind: paired.anchorable === false ? "turn" : paired.target_kind,
        span_ids: paired.anchorable === false ? [] : (paired.span_ids || []),
        target_label: labelFor + " (other side)"
      };
    }
    apiPost(path, body).then(function (data) {
      area.value = "";
      save.disabled = false;
      note.textContent = data.annotated
        ? "Saved alongside the read-only evidence"
        : "Saved";
      clear(history);
      (data.feedback || []).forEach(function (recorded) {
        if (recorded.pair_key && recorded.pair_key !== cmp.review_pair_key) {
          return;
        }
        history.appendChild(renderFeedbackHistoryRow(recorded));
      });
    }).catch(function (e) {
      save.disabled = false;
      note.textContent = e.message;
    });
  });
}

/* Turn-scope anchors for the pair as a whole, built from the two references
   the API returned. The anchored turn is each side's FIRST recorded turn and
   the reference stays whole, so this comment carries the same pair identity a
   comment on any row of the same comparison does. */
/* -- the expensive recorded LLM calls of one side [fix-9eg.3.1.2] ---------- */
/* WHICH recorded calls a side's totals above are made of, dearest first, for
   each side separately and never pooled across the two.

   Read off `turns[].usage.calls_detail` -- the per-call anchors the server
   publishes beside the roll-up -- and never re-derived from spans: this pane
   never loads them, and a second tally here is exactly how a list and a total
   come to disagree about one recorded execution. Nothing is charged again
   either: a wrapper recorded at two levels was folded away upstream, and a call
   quoting a provider response another call already accounts for carries no cost
   of its own, so the rows add up to the money the chips already reported.

   Three absences, kept apart, because collapsing any two of them is the failure
   this section exists to avoid: a call that recorded 0 spent nothing, a call
   that recorded no cost at all is unknown, and a shared response is accounted
   under its twin. The order is by recorded cost WITHIN one side; a side that
   spent less is not thereby the better answer and nothing here ranks the two
   runs against each other. */

/* How many rows a side shows before it says how many it is not showing.
   Nothing is aggregated away: the rest stay in their own turns' traces. */
var COMPARISON_CALLS_SHOWN = 20;

var CALL_COST_RECORDED = 0;     /* a cost is on record -- a real 0 included */
var CALL_COST_UNRECORDED = 1;   /* nothing recorded one: unknown, not free */
var CALL_COST_SHARED = 2;       /* accounted under another call's response */

function callCostGroup(call) {
  if (!call || typeof call !== "object") { return CALL_COST_UNRECORDED; }
  if (call.usage_state === "shared_response") { return CALL_COST_SHARED; }
  return finiteNumber(call.cost) === null
    ? CALL_COST_UNRECORDED : CALL_COST_RECORDED;
}

/* Every canonical call one side recorded, dearest known cost first.

   `turnsWithoutDetail` is its own answer rather than a shorter list: a payload
   that carries a turn's roll-up without its per-call anchors is one this pane
   cannot list, and saying so is the difference between "no expensive call here"
   and "not published here". */
function comparisonSideCalls(projection) {
  var calls = [];
  var turnsWithoutDetail = 0;
  var callsWithoutDetail = 0;
  ((projection && projection.turns) || []).forEach(function (turn) {
    var usage = turn.usage || {};
    if (!usage.calls_detail) {
      if (usage.calls) {
        turnsWithoutDetail += 1;
        callsWithoutDetail += usage.calls;
      }
      return;
    }
    usage.calls_detail.forEach(function (call) {
      calls.push({
        call: call,
        turnKey: call.turn_key || turn.turn_key,
        turnIndex: turn.turn_index,
        passId: turn.pass_id || null
      });
    });
  });
  /* Dearest recorded cost first, then the unknown ones, then the ones charged
     under another call. Ties fall back to the recorded turn and span id so two
     readers of one payload see one order. */
  calls.sort(function (a, b) {
    var groupA = callCostGroup(a.call), groupB = callCostGroup(b.call);
    if (groupA !== groupB) { return groupA - groupB; }
    if (groupA === CALL_COST_RECORDED && a.call.cost !== b.call.cost) {
      return b.call.cost - a.call.cost;
    }
    if (a.turnIndex !== b.turnIndex) {
      return (a.turnIndex || 0) - (b.turnIndex || 0);
    }
    return String(a.call.span_id).localeCompare(String(b.call.span_id));
  });
  return {
    calls: calls,
    turnsWithoutDetail: turnsWithoutDetail,
    callsWithoutDetail: callsWithoutDetail
  };
}

function fmtRecordedCallCost(call) {
  if (callCostGroup(call) === CALL_COST_SHARED) {
    return "cost counted under another call quoting the same provider response";
  }
  var cost = finiteNumber(call.cost);
  if (cost === null) { return "cost not recorded for this call"; }
  if (cost === 0) { return "cost 0.0000 — recorded, not missing"; }
  return "cost " + cost.toFixed(4);
}

function fmtRecordedCallTokens(call) {
  if (callCostGroup(call) === CALL_COST_SHARED) {
    return "tokens counted under that call too";
  }
  var total = exactInt(call.total_tokens);
  if (total === null) { return "tokens not recorded"; }
  return fmtCount(total) + " tok";
}

function fmtRecordedCallCache(call) {
  if (call.cache_state === "hit") { return "from cache"; }
  if (call.cache_state === "miss") { return "from the provider"; }
  return "cache state not recorded";
}

/* One recorded call: what it spent, how its answer arrived, and where it is. */
function renderRecordedCallRow(entry, cmp, which, label, ctx) {
  var call = entry.call;
  var row = el("div", "compareRow");
  var meta = el("div", "rowMeta");
  var priced = callCostGroup(call) === CALL_COST_RECORDED;
  meta.appendChild(el("span", "chipCost" + (priced ? "" : " unrecorded"),
    fmtRecordedCallCost(call)));
  meta.appendChild(el("span",
    "chipCost" + (exactInt(call.total_tokens) === null ? " unrecorded" : ""),
    fmtRecordedCallTokens(call)));
  meta.appendChild(el("span",
    "chipCost" + (call.cache_state === "hit" ? "" : " unrecorded"),
    fmtRecordedCallCache(call)));
  if (call.usage_state === "partial") {
    meta.appendChild(el("span", "chipMarker unknown",
      "usage partly recorded, so this call's tokens are not a whole count"));
  }
  if (call.completed === false) {
    meta.appendChild(el("span", "chipMarker warn",
      "still open — recorded before this call finished"));
  }
  row.appendChild(meta);
  row.appendChild(el("div", "sub",
    (call.model || "model not recorded")
    + " · " + label.toLowerCase() + " side"
    + " · turn " + (exactInt(entry.turnIndex) === null ? "—" : entry.turnIndex)
    + " " + (entry.turnKey || "not recorded")
    + " · span " + (call.span_id || "not recorded")
    + (entry.passId ? " · pass " + entry.passId : "")));
  var note = el("div", "sub", "");
  var open = el("button", "evidenceLink", "Open this recorded call");
  open.type = "button";
  open.setAttribute("data-recorded-call", call.span_id || "");
  open.setAttribute("data-recorded-call-side", which);
  open.addEventListener("click", function () {
    openPairCall(ctx, pairSide(cmp, which), entry, note);
  });
  row.appendChild(open);
  row.appendChild(note);
  return row;
}

/* One side's list, named by the evidence it was read from: the store that
   recorded it, the experiment it belongs to, and the recorded pass this
   projection was scoped to -- or that there is none, which is not the same as
   a pass nobody stamped a name on. */
function renderSideCallCosts(cmp, which, label, ctx) {
  var projection = which === "left" ? cmp.left : cmp.right;
  var run = (which === "left" ? cmp.left_run : cmp.right_run) || {};
  var side = pairSide(cmp, which);
  var usage = (projection && projection.usage) || {};
  var found = comparisonSideCalls(projection);
  function grouped(group) {
    return found.calls.filter(function (entry) {
      return callCostGroup(entry.call) === group;
    });
  }
  var priced = grouped(CALL_COST_RECORDED);
  var unknown = grouped(CALL_COST_UNRECORDED);
  var shared = grouped(CALL_COST_SHARED);
  var box = el("details", "side");
  var attempt = (run.attempt === undefined || run.attempt === null)
    ? "—" : run.attempt;
  /* Counted BEFORE the summary is written. The fold is closed when a reader
     first sees this side, so the summary is the only line showing -- and
     heading a side whose evidence could not be read with "0 recorded LLM
     calls" would state the one fact this list exists to avoid stating. */
  var unreadable = ((projection && projection.unavailable) || []).length;
  var withheld = unreadable || found.turnsWithoutDetail;
  box.appendChild(el("summary", null,
    label + ": attempt " + attempt + " — "
    + ((!found.calls.length && withheld)
        ? "no recorded LLM call can be listed, and that is missing evidence"
        : found.calls.length
          + (found.calls.length === 1
              ? " recorded LLM call" : " recorded LLM calls")
          + (priced.length
              ? ", dearest " + priced[0].call.cost.toFixed(4)
              : ", none of them priced"))));
  box.appendChild(el("div", "sub",
    "source " + (side.storeId || "not recorded")
    + (side.experimentId ? " · experiment " + side.experimentId : "")
    + ((projection && projection.ref && projection.ref.pass_id)
        ? " · pass " + projection.ref.pass_id
        : " · no recorded pass scope")));
  box.appendChild(el("div", "sub",
    priced.length + " priced, " + unknown.length + " with no recorded cost, "
    + shared.length + " quoting a response another call accounts for. These are "
    + "the same calls the totals above were counted over, so nothing is "
    + "charged twice here."));
  var coverage = (usage.tokens || {}).coverage;
  var cache = fmtCacheCounts(usage.cache);
  box.appendChild(el("div", "sub",
    "tokens " + (coverage ? "coverage " + coverage : "coverage not reported")
    + (cache ? " · " + cache : "")));
  /* Turns this side could not be read at all. Their calls are missing from the
     list because the evidence is missing, which is not the same as a run that
     made fewer calls -- so the count is shown whether or not anything else is
     listed. */
  if (unreadable) {
    box.appendChild(el("div", "sub",
      unreadable + " turn(s) of this side could not be read, so any call they "
      + "recorded is missing from this list rather than absent from the run."));
  }
  if (found.turnsWithoutDetail) {
    box.appendChild(el("div", "sub",
      found.turnsWithoutDetail + " turn(s) of this side published a roll-up "
      + "without per-call anchors, so " + found.callsWithoutDetail
      + " recorded call(s) of it cannot be listed here."));
  }
  var counted = exactInt(usage.calls) || 0;
  if (!found.calls.length) {
    /* "Nothing to list" and "nothing happened" are different findings. A side
       whose roll-up counted calls, or whose turns could not be read, has
       missing evidence -- and reporting that as a run which made no LLM call
       would be inventing the one fact this list is read for. */
    box.appendChild(el("div", "empty",
      (counted || unreadable || found.turnsWithoutDetail)
        ? "No call of this side can be listed here, and that is missing "
          + "evidence rather than a side that made no LLM call."
        : "No LLM call is recorded on this side."));
    return box;
  }
  if (counted && found.calls.length < counted) {
    box.appendChild(el("div", "sub",
      "Listing " + found.calls.length + " of the " + counted
      + " calls counted above; the rest recorded no per-call anchor here."));
  }
  var rows = el("div");
  box.appendChild(rows);
  paintRecordedCallRows(rows, found.calls, COMPARISON_CALLS_SHOWN,
                        cmp, which, label, ctx);
  return box;
}

/* Bounded first, complete on request.

   Every call of this side is already in the payload on screen, so showing the
   rest reveals what is in hand and issues no read, no scan and no second
   accounting. The hidden ones are counted BY KIND because the order puts the
   unknown-cost and shared-response calls last: without that line, a side with
   many priced calls would leave a reader unable to tell whether any unpriced
   call exists at all. */
function paintRecordedCallRows(rows, entries, limit, cmp, which, label, ctx) {
  clear(rows);
  var shown = Math.min(limit, entries.length);
  entries.slice(0, shown).forEach(function (entry) {
    rows.appendChild(renderRecordedCallRow(entry, cmp, which, label, ctx));
  });
  if (shown >= entries.length) { return; }
  var hidden = entries.slice(shown);
  function hiddenOf(group) {
    return hidden.filter(function (entry) {
      return callCostGroup(entry.call) === group;
    }).length;
  }
  rows.appendChild(el("div", "sub",
    "Showing the " + shown + " dearest of " + entries.length
    + " recorded calls. Not shown: " + hiddenOf(CALL_COST_RECORDED)
    + " priced, " + hiddenOf(CALL_COST_UNRECORDED) + " with no recorded cost, "
    + hiddenOf(CALL_COST_SHARED)
    + " quoting a response another call accounts for. None of them is "
    + "aggregated away."));
  var more = el("button", "evidenceLink",
    "Show all " + entries.length + " recorded calls of this side");
  more.type = "button";
  more.setAttribute("data-show-all-calls", which);
  more.addEventListener("click", function () {
    paintRecordedCallRows(rows, entries, entries.length, cmp, which, label, ctx);
  });
  rows.appendChild(more);
}

function renderComparisonCallCosts(container, cmp, ctx) {
  var sides = [["left", "Left"], ["right", "Right"]].filter(function (side) {
    var projection = side[0] === "left" ? cmp.left : cmp.right;
    return projection && projection.usage;
  });
  if (!sides.length) { return; }
  container.appendChild(el("h2", null, "Expensive recorded LLM calls"));
  container.appendChild(el("div", "sub",
    "Each side's own recorded calls, dearest recorded cost first, with the turn "
    + "and span that recorded each one. Costing less is not being right: these "
    + "two lists are read separately and neither picks a winner."));
  var panes = el("div", "comparePanes");
  sides.forEach(function (side) {
    panes.appendChild(renderSideCallCosts(cmp, side[0], side[1], ctx));
  });
  container.appendChild(panes);
}

/* One row's link, through the shared `openPairSpan`: the side's own reference
   decides the source, and the anchor is the CANONICAL span id this call was
   published under -- not its parent span and not its position in this list, so
   a folded wrapper or a re-ordered list can never redirect a reader to a
   different recorded call. */
function openPairCall(ctx, side, entry, note) {
  var spanId = entry.call.span_id;
  if (!spanId) {
    /* An anonymous record is still a call that was made, so the roll-up counts
       it -- but there is no span to anchor to, and only its turn can be
       opened. */
    openPairTurn(ctx, side, entry.turnKey, note);
    if (!note.textContent) {
      note.textContent = "This call recorded no span id, so only the turn that "
        + "recorded it could be opened.";
    }
    return;
  }
  openPairSpan(ctx, side, entry.turnKey, spanId, note);
}

function pairTurnAnchors(cmp) {
  function anchor(projection) {
    var ref = projection && projection.ref;
    if (!ref || !(ref.turn_keys || []).length) { return null; }
    return { ref: ref, turn_key: ref.turn_keys[0], target_kind: "turn",
             span_ids: [], anchorable: true, store_id: ref.store_id,
             manifest_store_id: projection.manifest_store_id || null };
  }
  return { left: anchor(cmp.left), right: anchor(cmp.right) };
}

/* WHERE a pair comment is recorded — the authorized source of the side the
   comment is anchored to, decided by the same rule the reads use.

   The two scopes the write route accepts are not two spellings of one thing. A
   sealed archive is named by the MANIFEST's store id, which is how every
   workspace route addresses it. A live experiment is named by its registration,
   which is the only handle the server resolves a working database by. And in
   neither case is it the evidence identity the references inside the body
   carry: the server resolves the paired side through that separately, and
   handing it the identity as a scope is refused.

   A side whose source is not named by either is refused here rather than
   posted somewhere adjacent and plausible. */
function pairWriteScope(ctx, side, anchor) {
  if (session && session.workspace_mode) {
    var storeId = side.manifestStoreId
      || (anchor && anchor.manifest_store_id) || null;
    if (!storeId) {
      return { refusal: "This archive's manifest does not name the store this "
        + "side was recorded in, so a comment has nowhere to be filed." };
    }
    return { query: "&store_id=" + encodeURIComponent(storeId),
             annotated: true };
  }
  var experimentId = (anchor && anchor.ref && anchor.ref.experiment_id)
    || side.experimentId || ctx.experimentId;
  if (!experimentId) {
    return { refusal: "This side names no experiment, so there is no recorded "
      + "database to file a comment against." };
  }
  return { query: "&benchmark_experiment=" + encodeURIComponent(experimentId) };
}

function describePairSides(cmp, anchors) {
  var parts = [];
  var fallback = false;
  [["Left", anchors.left, cmp.left_run], ["Right", anchors.right, cmp.right_run]]
    .forEach(function (side) {
      if (!side[1]) {
        parts.push(side[0] + ": nothing recorded");
        return;
      }
      if (side[1].anchorable === false) { fallback = true; }
      var run = side[2] || {};
      parts.push(side[0] + ": attempt " + run.attempt
        + (run.experiment_id ? " of " + String(run.experiment_id).slice(-8) : "")
        + " · " + headSnippet(side[1].turn_key, 24));
    });
  return {
    text: "This comment names " + parts.join("  ↔  ")
      + ". The pair is frozen at the moment you save: choosing a different "
      + "best run later leaves this comment naming the runs it was written "
      + "about.",
    fallback: fallback
  };
}

/* Benchmark setup: title, optional description, optional task prompts. */
var benchNav = 0;
function benchNavToken() { return expNavToken(); }
function benchNavStale(token) { return expNavStale(token); }
function benchmarkText(card, label, value, multiline) {
  var wrapper = el("label", "formField");
  wrapper.appendChild(el("div", "title", label));
  var input = el(multiline ? "textarea" : "input", "expNotes");
  input.setAttribute("aria-label", label); input.value = value || "";
  if (!multiline) { input.type = "text"; } else { input.rows = 3; }
  wrapper.appendChild(input); card.appendChild(wrapper); return input;
}
function benchmarkPrompt(task) {
  return task.prompt !== undefined ? task.prompt : (task.description || "");
}
function showBenchmarks() {
  focusHierarchy(function (n) { return n.kind === "root"; });
  benchmarkExperimentSource = null;
  var nav = benchNavToken(), d = document.getElementById("detail"); clear(d);
  var actions = pageHeader(d, "BENCHMARK LIBRARY", "Build confidence in every change", "Define repeatable tasks, compare experiments, and turn observations into better workflows.");
  if (!(session && session.workspace_mode)) {
    var create = el("button", "primary", "New benchmark");
    create.addEventListener("click", function () { editBenchmark(null); }); actions.appendChild(create);
  }
  var list = el("div", "recordGrid"); d.appendChild(list);
  api("/api/benchmarks").then(function (data) {
    if (benchNavStale(nav)) { return; }
    if (!data.benchmarks.length) { emptyState(d, "Start with one real task", "Create a benchmark for something your users need to do. Add a prompt, then create an experiment when you are ready to test."); }
    data.benchmarks.forEach(function (row) {
      var b = el("button", "recordCard");
      var top = el("span", "eyebrow", "BENCHMARK"); top.appendChild(el("span", "arrow", "↗")); b.appendChild(top);
      b.appendChild(el("strong", null, row.title || row.benchmark_id));
      b.appendChild(el("span", "sub", (row.versions || []).length + " versions · " + ((row.versions || []).slice(-1)[0] || "No version")));
      b.addEventListener("click", function () { showBenchmark(row.benchmark_id); }); list.appendChild(b);
    });
  }).catch(function (e) { if (!benchNavStale(nav)) { d.appendChild(el("p", "err fieldError", e.message)); } });
}
function editBenchmark(current) {
  var nav = benchNavToken(), d = document.getElementById("detail"); clear(d);
  expCrumbs(d, [{label: "Benchmarks", onClick: showBenchmarks}, {label: current ? "Edit benchmark" : "New benchmark"}]);
  pageHeader(d, "BENCHMARK SETUP", current ? "Refine your benchmark" : "What should your workflow do?", "Describe the work you want to test. Each task keeps its own identity across experiments.");
  var form = el("div", "card"); d.appendChild(form);
  var title = benchmarkText(form, "Title", current && (current.title || current.benchmark_id), false); title.setAttribute("data-autofocus", ""); title.placeholder = "e.g. Customer support essentials";
  var description = benchmarkText(form, "Description (optional)", current && current.description, true); description.placeholder = "What does this benchmark help you evaluate?";
  form.appendChild(el("div", "sub", current ? "Saving creates the next version after " + current.version + ". Existing experiments keep their original tasks." : "Version v1 and task IDs are assigned when you save."));
  sectionHeader(d, "Tasks");
  var tasks = el("div"), readers = []; d.appendChild(tasks);
  var add = el("button", null, "+ Add task");
  function addTask(task, focus) {
    var card = el("div", "card taskCard"), removed = false, body = el("div", "taskBody"); tasks.appendChild(card);
    card.appendChild(el("span", "taskNumber", String(readers.length + 1).padStart(2, "0")));
    card.appendChild(body);
    body.appendChild(el("div", "recordId", task.task_id || "New task · ID assigned on save"));
    var prompt = benchmarkText(body, "Task prompt (optional)", benchmarkPrompt(task), true); prompt.placeholder = "Write what a user would ask your workflow…";
    var remove = el("button", "ghost danger", "Remove task");
    remove.addEventListener("click", function () {
      removed = true; card.remove();
      var next = tasks.querySelector("textarea"); (next || add).focus();
      showNotice("Task removed from this draft", null, "Save the benchmark to publish your changes.");
    }); body.appendChild(remove);
    readers.push(function () { return removed ? null : {task_id: task.task_id, prompt: prompt.value}; });
    if (focus) { prompt.focus(); if (card.scrollIntoView) { card.scrollIntoView({block: "nearest"}); } }
  }
  (current ? current.tasks : [{}]).forEach(function (task) { addTask(task, false); });
  add.addEventListener("click", function () { addTask({}, true); });
  var toolbar = el("div", "formActions"), save = el("button", "primary", "Save benchmark"), msg = el("p", "err fieldError");
  msg.setAttribute("role", "alert"); d.appendChild(msg);
  toolbar.appendChild(add);
  var cancel = el("button", "ghost", "Cancel"); cancel.addEventListener("click", function () { if (current) { showBenchmark(current.benchmark_id); } else { showBenchmarks(); } }); toolbar.appendChild(cancel);
  save.addEventListener("click", function () {
    var draftTasks = readers.map(function (read) { return read(); }).filter(Boolean);
    if (!title.value.trim() || !draftTasks.length) {
      msg.textContent = !title.value.trim() ? "Give your benchmark a title." : "Add at least one task to save this benchmark.";
      showNotice("Check your benchmark", "error", msg.textContent); (!title.value.trim() ? title : add).focus(); return;
    }
    msg.textContent = "";
    apiPost("/api/benchmark-setup", {benchmark_id: current && current.benchmark_id,
      expected_version: current && current.version, title: title.value, description: description.value, tasks: draftTasks
    }).then(function (data) {
      if (!benchNavStale(nav)) {
        refreshConvs().then(function () { if (!benchNavStale(nav)) { showBenchmark(data.version.benchmark_id, data.version.version); } });
      }
    }).catch(function (e) { msg.textContent = e.message; });
  }); toolbar.appendChild(save); d.appendChild(toolbar);
}
/* The name a person can recognise: the rail's short id, then the description
   the author wrote. The card index ("Experiment 3") is a position in the
   newest-first list and moves, so the winner callout does not use it. */
function benchmarkExperimentTitle(row) {
  var recordedPath = findRecordedExperimentPath(row.experiment_id);
  var recordedNode = recordedPath && recordedPath[recordedPath.length - 1];
  var described = recordedNode ? recordedNode.label : row.description;
  var shortId = "Experiment · " + String(row.experiment_id || "").slice(-8);
  return { shortId: shortId, described: described || "" };
}

/* Newest-first puts the winner, often the oldest registration, at the bottom
   of a long list, and an archived winner is hidden until the rail asks for
   it. The callout names the pointer where the list starts. It is the same
   fact the experiment page reports, including when nobody chose and the first
   experiment still holds the contest. */
function renderBenchmarkWinner(host, result, benchmarkId) {
  clear(host);
  var winnerId = result.winner_experiment_id;
  if (!winnerId) { return; }
  var row = null;
  (result.experiments || []).some(function (candidate) {
    if (candidate.experiment_id === winnerId) { row = candidate; return true; }
  });
  var hidden = row && row.archived && !archivedExperimentsShown[benchmarkId];
  var card = el("div", "statusCard winnerCallout");
  var content = el("div");
  var badges = el("div", "diffToolbar");
  badges.appendChild(el("span", "pill ok", "Winner"));
  if (result.winner_automatic) {
    badges.appendChild(el("span", "pill", "automatic — first experiment"));
  }
  content.appendChild(badges);
  var title = row ? benchmarkExperimentTitle(row) : { shortId: winnerId, described: "" };
  content.appendChild(el("strong", null, title.shortId));
  if (title.described) { content.appendChild(el("p", null, title.described)); }
  var note = "This workflow's current answer for the benchmark.";
  if (hidden) {
    note += " It is archived, so the list hides it until archived experiments are shown.";
  } else if (!row) {
    note = "Recorded as the winner, but it is not in this benchmark's experiment list.";
  }
  content.appendChild(el("p", null, note));
  if (row) {
    var open = el("button", "winnerOpen", "Open winning experiment");
    open.type = "button";
    open.addEventListener("click", function () { openBenchmarkRecord(row); });
    content.appendChild(open);
  }
  card.appendChild(content);
  host.appendChild(card);
}

function showBenchmark(benchmarkId, selectedVersion) {
  focusHierarchy(function (n) { return n.kind === "benchmark" && n.benchmark_id === benchmarkId; });
  benchmarkExperimentSource = null;
  var nav = benchNavToken(), d = document.getElementById("detail"); clear(d);
  d.appendChild(el("div", "empty", "Loading benchmark…"));
  api("/api/benchmarks/" + encodeURIComponent(benchmarkId)).then(function (data) {
    if (benchNavStale(nav)) { return; }
    var versions = data.versions || [], version = selectedVersion || versions[versions.length - 1];
    if (!version) { clear(d); emptyState(d, "No versions yet", "Save a benchmark version to begin."); return; }
    return api("/api/benchmarks/" + encodeURIComponent(benchmarkId) + "/versions/" + encodeURIComponent(version)).then(function (loaded) {
      if (benchNavStale(nav)) { return; }
      clear(d); var manifest = loaded.version;
      expCrumbs(d, [{label: "Benchmarks", onClick: showBenchmarks}, {label: manifest.title || benchmarkId}]);
      var actions = pageHeader(d, "BENCHMARK · " + version, manifest.title || benchmarkId, manifest.description || "A repeatable set of tasks for your workflow.");
      if (versions.length > 1) {
        var select = el("select"); select.setAttribute("aria-label", "View version");
        versions.forEach(function (v) { var opt = el("option", null, v); opt.value = v; select.appendChild(opt); }); select.value = version;
        select.addEventListener("change", function () { showBenchmark(benchmarkId, select.value); }); actions.appendChild(select);
      }
      if (!(session && session.workspace_mode)) {
        var edit = el("button", null, "Edit benchmark"); edit.disabled = version !== versions[versions.length - 1];
        edit.title = edit.disabled ? "Select the latest version to edit" : "Save changes as a new version";
        edit.addEventListener("click", function () { editBenchmark(manifest); }); actions.appendChild(edit);
      }
      var strip = el("div", "summaryStrip");
      [[manifest.tasks.length, "Tasks in this version"], [version, "Selected version"], [versions.length, "Published versions"]].forEach(function (item) {
        var stat = el("div"); stat.appendChild(el("strong", null, item[0])); stat.appendChild(el("span", null, item[1])); strip.appendChild(stat);
      }); d.appendChild(strip);
      var experimentsHead = sectionHeader(d, "Experiments"), experimentList = el("div", "recordGrid"), msg = el("p", "err fieldError"); msg.setAttribute("role", "alert");
      if (!(session && session.workspace_mode)) {
        var create = el("button", "primary", "New experiment");
        create.addEventListener("click", function () {
          apiPost("/api/benchmarks/" + encodeURIComponent(benchmarkId) + "/experiments", {version: version}).then(function (result) {
            if (!benchNavStale(nav)) {
              refreshConvs().then(function () { if (!benchNavStale(nav)) { showBenchmarkExperiment(result.experiment.experiment_id); } });
            }
          }).catch(function (e) { msg.textContent = e.message; });
        }); experimentsHead.appendChild(create);
      }
      d.appendChild(msg);
      var winnerHost = el("div");
      d.appendChild(winnerHost);
      d.appendChild(experimentList);
      experimentList.appendChild(el("div", "empty", "Loading experiments…"));
      api("/api/benchmarks/" + encodeURIComponent(benchmarkId) + "/experiments").then(function (result) {
        if (benchNavStale(nav)) { return; }
        clear(experimentList);
        renderBenchmarkWinner(winnerHost, result, benchmarkId);
        var archivedCount = result.experiments.filter(function (row) { return row.archived; }).length;
        var experiments = archivedExperimentsShown[benchmarkId]
          ? result.experiments
          : result.experiments.filter(function (row) { return !row.archived; });
        /* An unreadable evidence store is a workspace-wide condition, so the
           sidebar's warning band (/api/navigation) is the one place that says
           so; repeating it over the cards told the reader nothing new. */
        if (!experiments.length) {
          if (archivedCount) {
            emptyState(experimentList, "Archived experiments are hidden", "Use Show all experiments beside this benchmark in the left rail to include them.");
          } else {
            emptyState(experimentList, "Ready for your first experiment", "Create an experiment to reserve the task IDs for a run. Your runner records the results here.");
          }
        }
        experiments.forEach(function (row, index) {
          var classes = "recordCard" + (row.archived ? " archived" : "") + (row.is_winner ? " winning" : "");
          var b = el("button", classes), top = el("span", "eyebrow");
          top.appendChild(el("span", null, row.benchmark_version || version));
          var marks = el("span", "marks");
          if (row.is_winner) { marks.appendChild(el("span", "pill ok", "Winner")); }
          if (row.archived) { marks.appendChild(el("span", "pill", "Archived")); }
          top.appendChild(marks);
          top.appendChild(el("span", "arrow", "↗")); b.appendChild(top);
          var recordedPath = findRecordedExperimentPath(row.experiment_id);
          var recordedNode = recordedPath && recordedPath[recordedPath.length - 1];
          var described = recordedNode ? recordedNode.label : row.description;
          b.appendChild(el("strong", null, "Experiment " + (index + 1)));
          if (described) { b.appendChild(el("span", "sub", described)); }
          b.appendChild(el("span", "sub", recordedPath || !row.registered ? "Execution recorded" : row.store ? "Execution registered" : "Not yet run"));
          b.appendChild(el("span", "recordId", row.experiment_id));
          b.addEventListener("click", function () { openBenchmarkRecord(row); }); experimentList.appendChild(b);
        });
      }).catch(function (e) { if (!benchNavStale(nav)) { experimentList.appendChild(el("p", "err", e.message)); } });
      sectionHeader(d, "Tasks", manifest.tasks.length);
      manifest.tasks.forEach(function (task, index) { taskPreview(d, task, index); });
      var analysis = el("details", "card"); analysis.appendChild(el("summary", null, "Analysis (optional)")); d.appendChild(analysis);
      analysis.addEventListener("toggle", function () {
        if (!analysis.open || analysis.dataset.loaded) { return; } analysis.dataset.loaded = "1";
        api("/api/benchmarks/" + encodeURIComponent(benchmarkId) + "/analysis").then(function (value) {
          if (benchNavStale(nav)) { return; }
          var area = benchmarkText(analysis, "Analysis", analysisText(value.analysis), true);
          area.readOnly = !!(session && session.workspace_mode);
          if (!area.readOnly) {
            var save = el("button", null, "Save analysis"), note = el("p", "err fieldError");
            save.addEventListener("click", function () {
              apiPut("/api/benchmarks/" + encodeURIComponent(benchmarkId) + "/analysis", {analysis: area.value})
                .then(function () { note.textContent = ""; }).catch(function (e) { note.textContent = e.message; });
            }); analysis.appendChild(save); analysis.appendChild(note);
          }
        }).catch(function (e) { analysis.dataset.loaded = ""; analysis.appendChild(el("p", "err", e.message)); });
      });
    });
  }).catch(function (e) { if (!benchNavStale(nav)) { clear(d); d.appendChild(el("p", "err fieldError", e.message)); } });
}
function showBenchmarkVersion(benchmarkId, version) { showBenchmark(benchmarkId, version); }
function openBenchmarkExecution(id) {
  var path = findExperimentPath(id);
  if (path) { activateHierarchy(path, true); }
  else { benchmarkExperimentSource = id; showExperiment(id); refreshConvs(); }
}

/* One experiment, one page, whichever route the reader took. The rail opens an
   experiment through activateHierarchy, which reads `recorded` off the node and
   opens a finished run's results; the benchmark's cards used to branch on
   registration alone, so the same recorded run showed its results from the rail
   and its pre-run handoff page from the card. Both now dispatch on the node.
   Registration state is the fallback for an experiment the navigation holds no
   node for -- an evidence store it could not open, or a run recorded since the
   last refresh. */
function openBenchmarkRecord(row) {
  if (row.workspace) { showWorkspaceExperiment(row); return; }
  var path = findExperimentPath(row.experiment_id);
  if (path) { activateHierarchy(path, true); }
  else if (row.registered) { showBenchmarkExperiment(row.experiment_id); }
  else { benchmarkExperimentSource = null; showExperiment(row.experiment_id); }
}

function showBenchmarkExperiment(id) {
  focusHierarchy(function (n) { return n.kind === "experiment" && n.experiment_id === id; });
  benchmarkExperimentSource = null;
  var nav = benchNavToken(), d = document.getElementById("detail"); clear(d);
  d.appendChild(el("div", "empty", "Loading experiment…"));
  api("/api/benchmark-experiments/" + encodeURIComponent(id)).then(function (data) {
    if (benchNavStale(nav)) { return; }
    clear(d); var row = data.experiment, manifest = data.benchmark;
    expCrumbs(d, [{label: "Benchmarks", onClick: showBenchmarks},
      {label: manifest.title || row.benchmark_id, onClick: function () { showBenchmark(row.benchmark_id, row.benchmark_version); }}, {label: "Experiment · " + id.slice(-8)}]);
    var actions = pageHeader(d, "EXPERIMENT · " + row.benchmark_version, manifest.title || row.benchmark_id, "A dedicated run of this benchmark, with " + manifest.tasks.length + " tasks ready to track.");
    if (data.can_delete) {
      var remove = el("button", "danger", "Delete empty experiment");
      remove.addEventListener("click", function () {
        confirmEmptyDeletion(remove).then(function (confirmed) {
          if (!confirmed || benchNavStale(nav)) { return; }
          lastActionButton = remove;
          apiDelete("/api/benchmark-experiments/" + encodeURIComponent(id)).then(function () {
            if (benchNavStale(nav)) { return; }
            refreshConvs().then(function () { if (!benchNavStale(nav)) { showBenchmark(row.benchmark_id, row.benchmark_version); } });
          }).catch(function (e) { error.textContent = e.message; });
        });
      }); actions.appendChild(remove);
    }
    var status = el("div", "statusCard"), content = el("div");
    status.appendChild(el("span", "statusSymbol", data.recorded ? "✓" : "↗"));
    content.appendChild(el("strong", null, data.recorded ? "Your execution records are ready" : row.store ? "Execution registered" : "Ready for a run"));
    content.appendChild(el("p", null, data.recorded ? "Explore conversations, inspect decisions, and leave feedback on what should improve." : "Pass this experiment ID and the task IDs below to your runner. Creating an experiment does not execute tasks."));
    status.appendChild(content); d.appendChild(status);
    var error = el("p", "err fieldError", data.warning || ""); error.setAttribute("role", "alert"); d.appendChild(error);
    if (data.recorded) {
      var open = el("button", "primary", "Open execution records");
      open.addEventListener("click", function () { openBenchmarkExecution(id); }); actions.appendChild(open);
    }
    var note = el("div", "card");
    sectionHeader(note, "Description");
    var description = benchmarkText(note, "Description (optional)", row.description || "", true);
    description.placeholder = "What are you trying to find out with this run?";
    /* Editable only until a runner claims the registration: after that the
       description the run declared belongs to its evidence store, which this
       page cannot write. */
    if (row.store || (session && session.workspace_mode)) {
      description.readOnly = true;
      note.appendChild(el("p", "sub", "This experiment has been handed to a runner; its description is part of the recorded run."));
    } else {
      var saveNote = el("button", null, "Save description"), noteMsg = el("p", "sub", "Saved with the registration, and recorded with the run.");
      saveNote.addEventListener("click", function () {
        noteMsg.textContent = "saving…";
        apiPatch("/api/benchmark-experiments/" + encodeURIComponent(id), {description: description.value})
          .then(function (result) { row.description = result.experiment.description; noteMsg.textContent = "saved"; refreshConvs(); })
          .catch(function (e) { noteMsg.textContent = e.message; });
      });
      note.appendChild(saveNote); note.appendChild(noteMsg);
    }
    d.appendChild(note);
    /* The contest this registration is in, on the page that exists before the
       run does: the first experiment of a contest holds it from creation, so
       this page has a winner to show even though nothing has executed. */
    renderWinnerPanel(d, id, function () { showBenchmarkExperiment(id); });
    renderRepeatSetup(d, row, nav);
    var identity = el("div", "card"), idActions = sectionHeader(identity, "Runner handoff");
    var identifier = benchmarkText(identity, "Experiment ID", id, false); identifier.readOnly = true;
    var copy = el("button", null, "Copy experiment ID");
    copy.addEventListener("click", function () {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(id).then(function () { showNotice("Experiment ID copied"); })
          .catch(function () { identifier.focus(); identifier.select(); showNotice("Select and copy the experiment ID", "error", "Clipboard access is unavailable. Use your keyboard copy shortcut."); });
      } else { identifier.focus(); identifier.select(); showNotice("Experiment ID selected", null, "Use your keyboard copy shortcut."); }
    }); idActions.appendChild(copy); d.appendChild(identity);
    sectionHeader(d, "Tasks", manifest.tasks.length);
    manifest.tasks.forEach(function (task, index) { taskPreview(d, task, index); });
  }).catch(function (e) { if (!benchNavStale(nav)) { clear(d); d.appendChild(el("p", "err fieldError", e.message)); } });
}
function openBenchmarkSetup() {
  benchmarkExperimentSource = null;
  if (!session || (!session.workflow_path && !session.workspace_mode)) {
    setTopMode("picker"); loadPicker();
    document.getElementById("pickerStatus").textContent = "Choose a workflow, then open Benchmark setup."; return;
  }
  setTopMode("debug"); showBenchmarks();
}
document.getElementById("benchmarkSetupBtn").addEventListener("click", openBenchmarkSetup);


