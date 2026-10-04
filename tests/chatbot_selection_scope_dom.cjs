/* WHERE the compare view reads each side of a pair from, driven in a real DOM.
 *
 * The reviewer-required half of fix-9eg.17.4: every deep link and every inline
 * artifact preview must be scoped to the database that RECORDED that side, and
 * the two ways of getting it wrong are both invisible in a one-store fixture.
 *
 * Mode `workspace`: two sealed archives whose attempts share the logical turn
 * key `shared-turn` and both record an artifact under `roster.txt` with
 * DIFFERENT values. A preview that read "the turn" without naming the store, or
 * named the wrong store, would show one side's value in both panes and nobody
 * would notice. Sealed routes also refuse an unscoped read outright, so a link
 * built the live way fails closed instead of quietly.
 *
 * Mode `adhoc`: a live experiment recorded in the workflow's default store with
 * no authoring registration -- typed at a prompt rather than declared. Its two
 * attempts are perfectly readable, but `?benchmark_experiment=` resolves only
 * REGISTERED experiments and answers 409 for this one. A view that scoped every
 * live read by experiment id would break a run that is sitting right there. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], plan = JSON.parse(process.argv[4]);
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;

  async function until(fn, what) {
    for (let i = 0; i < 300; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out waiting for ' + what + '. detail=' +
      d.getElementById('detail').textContent.slice(0, 1200));
  }
  const detail = () => d.getElementById('detail').textContent;
  const button = text =>
    [...d.querySelectorAll('#detail button')].find(node => node.textContent === text);
  const select = label => [...d.querySelectorAll('#detail select')]
    .find(node => node.getAttribute('aria-label') === label);
  /* Each control's handler repaints asynchronously and the old pair stays on
     screen until the new one arrives, so the node found a moment ago may be
     detached by the time it is changed -- a change on a detached node does
     nothing at all. The change is therefore re-issued against a freshly found
     control until the page shows what was asked for. */
  async function changeUntil(label, value, ready, what) {
    for (let attempt = 0; attempt < 10; attempt++) {
      const node = [...d.querySelectorAll('#detail select')]
        .find(item => item.getAttribute('aria-label') === label);
      if (node) {
        node.value = value;
        node.dispatchEvent(new w.Event('change', {bubbles: true}));
      }
      for (let i = 0; i < 20; i++) {
        if (ready()) return;
        await new Promise(r => setTimeout(r, 50));
      }
    }
    throw Error('Timed out setting ' + label + ' to ' + value + ' for ' + what
      + '. detail=' + detail().slice(0, 1200));
  }
  const panes = () => [...d.querySelectorAll('#detail .comparePane')];

  /* The task container is opened directly, the way a benchmark experiment page
     opens it, and re-opened if the page's own startup paint lands on #detail
     after the call. */
  async function openCompare() {
    for (let attempt = 0; attempt < 8; attempt++) {
      w.taskView = 'compare';
      w.showExperimentTask(plan.experiment, plan.task, 'Scope check');
      for (let i = 0; i < 20; i++) {
        if (select('View')) return;
        await new Promise(r => setTimeout(r, 50));
      }
    }
    throw Error('the compare view never opened. detail=' + detail().slice(0, 1200));
  }

  /* Both sides named explicitly. Left unnamed, BOTH default to the same pinned
     run, which would compare an attempt with itself and agree for the wrong
     reason -- exactly the false pass this test exists to avoid.
     Waited on the rendered header rather than the select's value, because each
     change repaints asynchronously and the previous pair stays on screen until
     the new one arrives. */
  async function pickBothSides() {
    /* Waited on the picker being USABLE, not merely present. The selects are
       rendered before the attempt list they are filled from arrives, so a wait
       that stops at "the element exists" can set a value the element has no
       option for -- the assignment is dropped, nothing repaints, and the
       failure surfaces further down as a timeout on the header. */
    await until(() => {
      const left = select('Left run');
      const right = select('Right run');
      if (!left || !right) { return null; }
      const offered = side => [...side.options].map(option => option.value);
      return offered(left).includes('1') && offered(right).includes('2')
        ? left : null;
    }, 'the pair picker, with both attempts offered');
    await changeUntil('Left run', '1',
      () => detail().includes('Left: attempt 1'), 'the left side');
    await changeUntil('Right run', '2',
      () => detail().includes('Right: attempt 2'), 'the right side');
  }

  d.getElementById('modeDebug').click();
  await until(() => w.session, 'the session');
  if (plan.mode === 'adhoc') {
    /* Deliberately NOT set: this run has no registration to scope by, and the
       page must read it out of the source it is already pointed at. */
    assert.equal(w.benchmarkExperimentSource, null,
      'the ad-hoc run is in the default store, so nothing scopes it');
  }

  /* ================================================================
   * The pair, with both sides' answers and artifacts
   * ================================================================ */
  await openCompare();
  await pickBothSides();
  await until(() => panes().length >= 2, 'both panes of the pair');
  await until(() => detail().includes('Show this artifact here'),
    'an artifact offered for inspection in place');

  const artifactPanes = panes().filter(pane =>
    pane.textContent.includes('Show this artifact here'));
  assert.equal(artifactPanes.length, 2,
    'both sides recorded an artifact: ' + panes().map(p => p.textContent.slice(0, 120)));

  /* Expanded IN PLACE, both at once: the point of the inline preview is that
     inspecting one side does not lose the other. */
  const previews = artifactPanes.map(pane => {
    const node = pane.querySelector('details.artifactPreview');
    node.open = true;
    node.dispatchEvent(new w.Event('toggle'));
    return node;
  });
  await until(() => previews.every(node => !node.textContent.includes('loading…')),
    'both previews to resolve');
  assert.equal(artifactPanes.length, 2, 'and both panes are still on screen');

  const shown = previews.map(node => {
    const frame = node.querySelector('iframe');
    return (frame ? frame.getAttribute('srcdoc') : '') + ' ' + node.textContent;
  });
  for (const expected of plan.values) {
    assert.ok(shown.some(text => text.includes(expected)),
      'the recorded value ' + expected + ' is shown where it was recorded: '
      + shown.join(' ||| ').slice(0, 900));
  }
  assert.ok(!shown[0].includes(plan.values[1]) && !shown[1].includes(plan.values[0]),
    'and neither pane shows the other side\'s value: '
    + shown.join(' ||| ').slice(0, 900));

  /* ================================================================
   * The deep link out of the pane, into the right database
   * ================================================================ */
  const open = [...artifactPanes[1].querySelectorAll('button')]
    .find(node => node.textContent === 'Open the turn that recorded it');
  assert.ok(open, 'the artifact still links to the turn that recorded it');
  open.click();
  if (plan.mode === 'workspace') {
    /* Addressed by store id, which is the only thing that separates two
       archives holding the same logical turn key. */
    await until(() => w.state.storeId === plan.right_store,
      'the right side\'s own archive to be selected, not the left\'s');
    await until(() => w.state.turn, 'the scoped turn to load');
    assert.equal(w.state.turnKey, plan.turn_key);
    assert.ok(w.state.turn.answer.includes(plan.right_answer),
      'and it is the RIGHT archive\'s turn under that shared key: '
      + w.state.turn.answer);
  } else {
    await until(() => w.state.turn, 'the turn to load with no scope invented');
    assert.equal(w.benchmarkExperimentSource, null,
      'an unregistered run is not scoped by an experiment id that resolves to '
      + 'nothing: benchmarkExperimentSource=' + w.benchmarkExperimentSource);
    assert.ok(w.state.turn.answer.includes(plan.right_answer),
      'and the right side\'s turn is what opened: ' + w.state.turn.answer);
  }

  /* ================================================================
   * A step drilldown follows the same rule
   * ================================================================ */
  await openCompare();
  await pickBothSides();
  await changeUntil('View', 'steps',
    () => d.querySelectorAll('#detail .compareRow').length > 0,
    'the aligned steps');
  const trace = [...d.querySelectorAll('#detail .compareRow button')]
    .find(node => node.textContent === 'Open the recorded turn');
  assert.ok(trace, 'a step offers its trace: '
    + [...d.querySelectorAll('#detail .compareRow button')]
        .map(n => n.textContent).join(' | '));
  trace.click();
  await until(() => w.state.turn, 'the step\'s turn');
  if (plan.mode === 'workspace') {
    assert.ok(w.state.storeId, 'a sealed step opened through its store id');
  } else {
    assert.equal(w.benchmarkExperimentSource, null,
      'and a live step needed no scope named either');
  }

  /* ================================================================
   * A comment on a pair of SEALED archives, saved and read back
   * ================================================================
   * Read-only evidence is not a reason to refuse the note: the comment is
   * filed beside the archive and read back with it. The two things that can go
   * wrong here are invisible in a one-store fixture and both are refused by the
   * server, so a green result means the page named the right source: a workspace
   * write is scoped by the MANIFEST's store id, while the references inside it
   * name their archives by evidence identity. Swap them and the write is 409. */
  if (plan.comment) {
    await openCompare();
    await pickBothSides();
    const composerOf = label => [...d.querySelectorAll('#detail .feedbackCard')]
      .find(card => card.textContent.includes('Comment on ' + label)
        && card.querySelector('[data-compare-comment-box]'));
    const tabOf = (card, value) =>
      card.querySelector('button[data-value="' + value + '"]');

    async function writeComment(label, text) {
      const card = await until(() => composerOf(label), 'the composer for ' + label);
      const save = [...card.querySelectorAll('button')]
        .find(node => node.textContent === 'Save this comment');
      assert.ok(save, 'the composer for ' + label + ' offers a save');
      assert.equal(save.disabled, false,
        'sealed evidence does not disable the composer for ' + label + ': '
        + card.textContent.slice(0, 500));
      /* Ordinary vocabulary, chosen the same way a turn comment chooses it. */
      tabOf(card, plan.comment.category).click();
      await until(() => tabOf(card, plan.comment.subcategory),
        'the subcategories of ' + plan.comment.category);
      tabOf(card, plan.comment.subcategory).click();
      const box = card.querySelector('[data-compare-comment-box]');
      box.value = text;
      save.click();
      const note = await until(
        () => [...card.querySelectorAll('.sub')]
          .find(node => /^Saved|does not|nowhere|refus|unknown|no workspace/.test(
            node.textContent)),
        'the save of ' + label + ' to report itself');
      assert.match(note.textContent, /^Saved/,
        'the comment on ' + label + ' was recorded: ' + note.textContent);
      assert.ok(note.textContent.includes('alongside the read-only evidence'),
        'and it says where, because the archive itself did not change: '
        + note.textContent);
    }

    await writeComment('the whole pair', plan.comment.text);

    /* The selected side's step, through the same authorized source. */
    await changeUntil('View', 'steps',
      () => d.querySelectorAll('#detail .compareRow').length > 0,
      'the aligned steps');
    const rowToggle = await until(() => [...d.querySelectorAll('#detail .compareRow')]
      .map(row => [...row.querySelectorAll('button')]
        .find(node => /^Comment/.test(node.textContent)))
      .find(Boolean), 'a step offering a comment');
    rowToggle.click();
    const stepLabel = await until(() => {
      const card = [...d.querySelectorAll('#detail .feedbackCard')]
        .find(node => node.querySelector('[data-compare-comment-box]')
          && !node.textContent.includes('Comment on the whole pair'));
      const heading = card && card.querySelector('h2');
      return heading && heading.textContent.replace(/^Comment on /, '');
    }, 'the step composer');
    await writeComment(stepLabel, plan.comment.step_text);

    /* And they are ordinary comments: the task's Feedback view is where they
       show up, with no second read layer and no pair-only view. */
    d.querySelector('#detail button[data-task-view="feedback"]').click();
    await until(() => detail().includes(plan.comment.text),
      'the pair comment under the task\'s Feedback view');
    const rows = [...d.querySelectorAll('#detail .listItem')]
      .filter(node => node.textContent.includes(plan.comment.text));
    assert.equal(rows.length, 1,
      'the comment is listed exactly once, not once per archive: '
      + rows.map(node => node.textContent.slice(0, 160)).join(' ||| '));
    assert.ok(rows[0].textContent.includes(plan.comment.category_label),
      'carrying the category it was written under: '
      + rows[0].textContent.slice(0, 300));
    assert.ok(rows[0].textContent.includes('Compared with'),
      'and naming the other side of the pair it was written about: '
      + rows[0].textContent.slice(0, 300));
    assert.ok(detail().includes(plan.comment.step_text),
      'the step comment is listed under the same view');
  }

  /* ================================================================
   * The selected pair does not survive a change of source
   * ================================================================ */
  if (plan.comment) {
    await openCompare();
    await pickBothSides();
  }
  assert.ok(w.taskCompare.key, 'a pair is selected for this task');
  /* The hook the source switcher calls. The same experiment and task ids recur
     across archives, so keeping attempt 2 selected would show a number from the
     old database under the new one's label. */
  w.onSourceSwitch();
  assert.equal(w.taskCompare.key, null, 'the pair identity is dropped');
  assert.equal(w.taskCompare.right, null, 'along with the attempt it named');

  assert.deepEqual(errors, [], 'no page errors');
  process.stdout.write('selection scope DOM checks passed\n');
  /* Exited rather than slept-then-closed. The sleep was a guess at how long
     the last navigation's reads take, and when one landed later than that the
     page called `document.createElement` on a closed window and failed inside
     its own callback -- an intermittent failure of the harness, reported as a
     failure of the product. Every assertion above has already run at this
     point, so there is nothing left for a late response to tell us. */
  process.exit(0);
})().catch(error => {
  process.stderr.write(String((error && error.stack) || error) + '\n');
  process.exit(1);
});
