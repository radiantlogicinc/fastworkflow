/* The consistency band, clicked in a real DOM (fix-9eg.17.5).
 *
 * What a Python test cannot claim: that a person opening a repeated task SEES
 * the distribution under the attempts, that the run with nothing recorded is
 * reported as unknown rather than as zero steps, that a metric with no data
 * says so instead of printing a number, that the panel never renders a verdict,
 * and that clicking a pair lands in the EXISTING comparison for exactly those
 * two attempts with that row highlighted.
 *
 * The world is the API half's fixture: one task under two experiments. The
 * BASELINE ran it twice with 2 and 4 dispatches, one of those runs recording no
 * plan at all, plus a third attempt that finished having recorded nothing. The
 * CANDIDATE ran it three times, twice identically. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], ids = JSON.parse(process.argv[4]);
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
    for (let i = 0; i < 400; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out waiting for ' + what + '. detail=' +
      d.getElementById('detail').textContent.slice(0, 1200));
  }
  const detail = () => d.getElementById('detail').textContent;
  const panel = () => d.querySelector('#detail [data-consistency]');
  const panelText = () => (panel() || {textContent: ''}).textContent;
  const pairButtons = () => [...d.querySelectorAll('#detail button.consistencyPair')];
  const pairButton = key =>
    pairButtons().find(node => node.getAttribute('data-pair') === key);

  d.getElementById('modeDebug').click();
  await until(() => w.session, 'the session');
  w.benchmarkExperimentSource = ids.baseline;

  /* ================================================================
   * The Runs view of the baseline: the distribution, and the gaps
   * ================================================================ */
  w.taskView = 'runs';
  w.showExperimentTask(ids.baseline, ids.task, 'Roster baseline');
  await until(() => panelText().includes('Consistency across repeated runs'),
    'the consistency panel on the Runs view');

  // The step distribution the route computed, printed as itself.
  assert.ok(panelText().includes('Executed steps'),
    'the step metric is on screen: ' + panelText().slice(0, 600));
  assert.ok(/mean 3\.00 · min 2 · max 4 · population SD 1\.00/.test(panelText()),
    'the distribution is printed: ' + panelText().slice(0, 900));
  assert.ok(panelText().includes('population SD = sqrt'),
    'the SD formula is stated rather than assumed: ' + panelText().slice(0, 900));

  // The attempt that recorded nothing: named, and explicitly not zero.
  assert.ok(panelText().includes('unknown, not zero'),
    'a missing count is not reported as no steps: ' + panelText().slice(0, 900));
  assert.ok(panelText().includes('attempt(s) 3'),
    'and the attempt is named so it can be opened: ' + panelText().slice(0, 900));

  // Planning similarity has no data here, because one run recorded no plan.
  // The tempting wrong answers are 1.000 and 0.000; neither is on screen.
  assert.ok(panelText().includes('no pair could be compared'),
    'a metric with no data says so: ' + panelText().slice(0, 900));

  // Descriptive, never a verdict: no target, no pass/fail, no grade.
  const forbidden = ['target', 'passes the', 'failed the', 'score:', 'grade'];
  forbidden.forEach(word => assert.ok(!panelText().toLowerCase().includes(word),
    'the panel renders a verdict word: ' + word));
  assert.ok(panelText().includes('not correctness'),
    'the panel says what consistency is not: ' + panelText().slice(0, 400));

  /* ================================================================
   * The candidate: three repeats, two of them identical
   * ================================================================ */
  w.benchmarkExperimentSource = ids.candidate;
  w.taskView = 'runs';
  w.showExperimentTask(ids.candidate, ids.task, 'Roster candidate');
  await until(() => panelText().includes('Consistency across repeated runs')
    && pairButtons().length >= 3, 'the candidate panel with its pairs');

  assert.ok(panelText().includes('Planning similarity'),
    'planning similarity is on screen: ' + panelText().slice(0, 600));
  assert.ok(panelText().includes('Final-answer similarity'),
    'final-answer similarity is a separate figure, not folded into planning');
  assert.ok(/3 of 3 pairs compared/.test(panelText()),
    'the denominator travels with the mean: ' + panelText().slice(0, 900));

  // Attempts 1 and 2 recorded the same plan and the same final answer.
  const identical = pairButton('1:2');
  assert.ok(identical, 'the 1-vs-2 pair is listed: '
    + pairButtons().map(b => b.getAttribute('data-pair')).join(','));
  assert.ok(/plan 1\.000/.test(identical.textContent),
    'two identical plans score about one: ' + identical.textContent);
  assert.ok(/answer 1\.000/.test(identical.textContent),
    'and so do two identical answers: ' + identical.textContent);
  assert.ok(/steps 4 vs 4 \(\+?0, 0\.0%\)/.test(identical.textContent),
    'equal counts are 0 and 0%: ' + identical.textContent);

  // The failed run is present, not filtered out of the population.
  const withFailure = pairButton('1:3');
  assert.ok(withFailure, 'the pair against the failed run is listed');

  /* ================================================================
   * A pair opens the existing comparison, focused
   * ================================================================ */
  withFailure.click();
  await until(() => detail().includes('Compare two recorded runs'),
    'the Compare view');
  await until(() => panelText().includes('Consistency across repeated runs'),
    'the consistency panel under the comparison');
  assert.equal(w.taskCompare.left, '1', 'the left side is the pair it was');
  assert.equal(w.taskCompare.right, '3', 'and so is the right');

  const focused = await until(
    () => d.querySelector('#detail button.consistencyPair.pairFocused'),
    'the open pair highlighted in the band');
  assert.equal(focused.getAttribute('data-pair'), '1:3',
    'the highlighted row is the pair on screen');

  // A pair opens the EXISTING comparison, not a private one: the step rows and
  // their comment affordance are the ones already on this screen, so a note
  // written from here goes through the same feedback taxonomy as every other
  // comment on the product.
  await until(() => detail().includes('name this exact pair'),
    'the existing pair feedback surface in the comparison');

  assert.deepEqual(errors, [], 'page errors');
  console.log && 0;
  process.stdout.write('consistency DOM ok\n');
  process.exit(0);
})().catch(e => { process.stderr.write(String(e && e.stack || e) + '\n'); process.exit(1); });
