/* WHERE the compare view reads each side of a pair from, driven in a real DOM.
 *
 * The reviewer-required half of fix-9eg.17.4: every deep link and every inline
 * artifact preview must be scoped to the database that RECORDED that side.
 *
 * Mode `adhoc`: a live experiment recorded in the workflow's default store with
 * no authoring registration -- typed at a prompt rather than declared. Its two
 * attempts are perfectly readable from the store the page is pointed at. */
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
  await until(() => w.state.turn, 'the turn to load with no scope invented');
  assert.ok(w.state.turn.answer.includes(plan.right_answer),
    'and the right side\'s turn is what opened: ' + w.state.turn.answer);

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

  /* ================================================================
   * The selected pair does not survive a change of source
   * ================================================================ */
  assert.ok(w.taskCompare.key, 'a pair is selected for this task');
  /* The hook the source switcher calls. The same experiment and task ids recur
     across workflows' databases, so keeping attempt 2 selected would show a
     number from the old database under the new one's label. */
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
