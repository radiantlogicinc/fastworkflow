/* Recorded pass content on the comparison screen, in a real DOM (fix-txxy).
 *
 * The evidence behind this run is two attempts of one task in one real store:
 *
 *   attempt 1  a REAL distilled turn, recorded by `distillation.py` through a
 *              real `WorkflowExecutionContext` -- one `fw.distillation.pass`
 *              span per pass, each carrying that pass's own answer and plan.
 *   attempt 2  a turn whose spans are pass-STAMPED but which recorded no pass
 *              content, which is how a producer that marks passes without
 *              describing them reads, and how every pre-`fix-txxy` trace reads
 *              once anything stamps one.
 *
 * Those two must not render the same way. Before this the page labelled every
 * attribution other than `turn` "shared across passes", so a pass that really
 * did record its own answer was reported as quoting the turn's -- denying the
 * divergence a teacher/student comparison is opened to look at. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const experiment = process.argv[4], task = process.argv[5];
const teacherAnswer = process.argv[6], studentAnswer = process.argv[7];
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
      d.getElementById('detail').textContent.slice(0, 1500));
  }
  const detail = () => d.getElementById('detail').textContent;
  const select = label => [...d.querySelectorAll('#detail select')]
    .find(node => node.getAttribute('aria-label') === label);
  /* Each control repaints asynchronously and the old pair stays on screen
     until the new one arrives, so the node found a moment ago may already be
     detached -- a change on a detached node does nothing. Re-issued against a
     freshly found control until the page shows what was asked for. */
  async function changeUntil(label, value, ready, what) {
    for (let attempt = 0; attempt < 12; attempt++) {
      const node = select(label);
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
      + '. detail=' + detail().slice(0, 1500));
  }
  /* The answer panes are the first pair; the artifact panes follow. */
  const answerPanes = () => [...d.querySelectorAll('#detail .comparePane')].slice(0, 2);
  /* '' while a repaint is in flight and the panes are momentarily gone: a
     readiness check must answer "not yet", not throw. */
  const paneText = index => {
    const pane = answerPanes()[index];
    return pane ? pane.textContent : '';
  };
  const leftPane = () => paneText(0);
  const rightPane = () => paneText(1);

  d.getElementById('modeDebug').click();
  await until(() => w.session, 'the session');

  async function openCompare() {
    for (let attempt = 0; attempt < 8; attempt++) {
      w.taskView = 'compare';
      w.showExperimentTask(experiment, task, 'Pass capture');
      for (let i = 0; i < 20; i++) {
        if (select('View')) return;
        await new Promise(r => setTimeout(r, 50));
      }
    }
    throw Error('the compare view never opened. detail=' + detail().slice(0, 1500));
  }

  await openCompare();
  await until(() => select('Left run') && select('Right run'), 'the pair picker');
  await changeUntil('Left run', '1',
    () => detail().includes('Left: attempt 1'), 'the left side');
  await changeUntil('Right run', '2',
    () => detail().includes('Right: attempt 2'), 'the right side');
  await until(() => answerPanes().length >= 2, 'both answer panes');

  /* ================================================================
   * The selector offers the passes the producer actually recorded
   * ================================================================ */
  const passSelect = await until(() => select('Left recorded pass'),
    'the left pass selector');
  const offered = [...passSelect.options].map(option => option.value);
  assert.deepEqual(offered.slice().sort(), ['', 'student', 'teacher'],
    'the two recorded passes, plus the whole turn: ' + offered.join(','));

  /* Unscoped first: the turn's own answer, with no pass label either way. */
  assert.ok(!leftPane().includes('recorded for this pass'),
    'the whole-turn read claims no pass: ' + leftPane().slice(0, 300));
  assert.ok(!leftPane().includes('shared across passes'),
    'nor does it call the turn\'s own answer shared: ' + leftPane().slice(0, 300));

  /* ================================================================
   * A pass that recorded its content shows ITS answer and ITS plan
   * ================================================================ */
  await changeUntil('Left recorded pass', 'teacher',
    () => leftPane().includes('recorded for this pass'), 'the teacher pass');
  assert.ok(leftPane().includes(teacherAnswer),
    'the teacher pass shows the answer it produced: ' + leftPane().slice(0, 400));
  assert.ok(leftPane().includes('recorded for this pass — teacher'),
    'and says whose it is: ' + leftPane().slice(0, 400));
  assert.ok(leftPane().includes('Plan this pass generated'),
    'the plan only a pass records is offered: ' + leftPane().slice(0, 400));
  assert.ok(!leftPane().includes('shared across passes'),
    'recorded content is not labelled shared: ' + leftPane().slice(0, 400));

  /* The other pass of the SAME turn shows something different. One turn row
     holds one answer, so this is the whole point. */
  /* Waited on the ATTRIBUTION, not on the answer text: the student's answer is
     a prefix of the teacher's here, so "the student's text is on screen" is
     already true of the teacher's pane and would pass before any repaint. */
  await changeUntil('Left recorded pass', 'student',
    () => leftPane().includes('recorded for this pass — student'),
    'the student pass');
  assert.ok(leftPane().includes(studentAnswer),
    'the student pass shows the answer it produced: ' + leftPane().slice(0, 400));
  assert.ok(!leftPane().includes(teacherAnswer),
    'the student pass does not quote the teacher: ' + leftPane().slice(0, 400));

  /* ================================================================
   * A stamped pass that recorded nothing still reads as shared
   * ================================================================ */
  await changeUntil('Right recorded pass', 'teacher',
    () => rightPane().includes('shared across passes'), 'the contentless pass');
  assert.ok(rightPane().includes(
    'shared across passes — this text is the turn\'s, not this pass\'s'),
    'the legacy label is kept verbatim: ' + rightPane().slice(0, 400));
  assert.ok(!rightPane().includes('recorded for this pass'),
    'and nothing is claimed for a pass that recorded nothing: '
    + rightPane().slice(0, 400));
  assert.ok(!rightPane().includes('Plan this pass generated'),
    'no plan is invented for it: ' + rightPane().slice(0, 400));

  /* Back to the teacher pass, so the artifact checks below read the pane they
     were written for. */
  await changeUntil('Left recorded pass', 'teacher',
    () => leftPane().includes('recorded for this pass — teacher'),
    'the teacher pass again');

  /* ================================================================
   * Artifacts are scoped to the pass that produced them
   * ================================================================ */
  const artifactPanes = [...d.querySelectorAll('#detail .comparePane')].slice(2, 4);
  assert.ok(artifactPanes.length === 2, 'both artifact panes render');
  assert.ok(artifactPanes[0].textContent.includes('pass'),
    'the left artifacts say which pass they belong to: '
    + artifactPanes[0].textContent.slice(0, 400));

  assert.deepEqual(errors, [], 'page errors: ' + errors.join(' | '));
  process.stdout.write('pass comparison DOM checks passed\n');
  /* Exited rather than closed: an in-flight page fetch resolving after
     teardown fails inside the page's own callback, which is a harness
     artefact and not a finding. */
  process.exit(0);
})().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exit(1); });
