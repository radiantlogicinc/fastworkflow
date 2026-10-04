/* The training-history section on the shipped page, in a real DOM against a
 * real server (fix-9eg.2).
 *
 * Driven end to end through the page's own controls: the Tools-menu button
 * opens the section, the rendered rows are clicked, and every assertion is
 * made on text the page actually put on screen. The evidence behind it is
 * real `train_runs` rows and real attempt stamps written by the test's
 * fixture through the store's own write methods.
 *
 * What must NOT happen: a held-out classifier metric presented as a
 * task-success rate, a training run linked to a run that merely shares its
 * workflow fingerprint or merely happens to be the newest, a missing
 * version id rendered as "nothing ran on it", or a context with no held-out
 * report rendered as a row of zeros. */
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
    for (let i = 0; i < 300; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out waiting for ' + what + '. list=' +
      d.getElementById('trainingRunList').textContent.slice(0, 600) +
      ' detail=' + d.getElementById('trainingRunDetail').textContent.slice(0, 900));
  }
  const rows = () => [...d.querySelectorAll('#trainingRunList .listItem')];
  const detail = () => d.getElementById('trainingRunDetail').textContent;
  const rowFor = title => rows().find(node =>
    node.querySelector('.title').textContent === title);

  await until(() => w.session, 'the session');

  /* The nav hook: one button, in the menu that already holds the workbench
     tools. Nothing about the rail or the conversation list is touched. */
  const button = d.getElementById('trainingHistoryBtn');
  assert.ok(button, 'the Tools menu carries a training-history entry');
  button.click();

  /* ================================================================
   * The list: every recorded run, newest first, named by its identity
   * ================================================================ */
  await until(() => rows().length === 3, 'three recorded training runs');
  assert.deepEqual(rows().map(node => node.querySelector('.title').textContent),
    ['Run ' + ids.legacy, ids.version_b, ids.version_a],
    'newest first, and a run with no published version is named by its run id');

  /* ================================================================
   * A linked run
   * ================================================================ */
  rowFor(ids.version_a).click();
  await until(() => detail().includes(ids.version_a) &&
    detail().includes('Runs on this trained model set'), 'the linked run');
  let text = detail();
  assert.ok(text.includes('2 attempts recorded the published version'),
    'both stamped attempts are reported: ' + text.slice(0, 600));
  assert.ok(text.includes('attempt 1') && text.includes('attempt 2'),
    'and each is named: ' + text.slice(0, 600));
  assert.ok(!text.includes('exp-unstamped'),
    'an attempt whose server stamped no snapshot is never swept in: ' + text);

  // The numbers keep their own names and say what they were measured on.
  assert.ok(text.includes('in_distribution_f1'),
    'the recorded metric name survives: ' + text.slice(0, 900));
  assert.ok(text.includes('not a task-success rate'),
    'and is captioned as not being one: ' + text.slice(0, 900));
  assert.ok(!/\bpass rate\b/i.test(text) && !/task success rate/i.test(text.replace('not a task-success rate', '')),
    'nothing renames it into a task verdict: ' + text.slice(0, 900));

  // The base checkpoint is labelled as a base model, not as the identity.
  assert.ok(text.includes('base model (tiny)'),
    'the fine-tuned-from checkpoint is named as such: ' + text.slice(0, 900));

  // A carried-forward context published thresholds and no evaluation.
  const carried = [...d.querySelectorAll('#trainingRunDetail details')]
    .find(node => node.querySelector('summary').textContent === 'TodoListManager');
  assert.ok(carried, 'the carried-forward context is listed: ' + text.slice(0, 900));
  assert.ok(carried.textContent.includes('No held-out evaluation recorded'),
    'and says it was not evaluated instead of showing zeros: ' + carried.textContent);
  assert.ok(!carried.textContent.includes('in_distribution_f1'),
    'no metric is invented for it: ' + carried.textContent);

  /* ================================================================
   * The two near-misses
   * ================================================================ */
  rowFor(ids.version_b).click();
  await until(() => detail().includes(ids.version_b) &&
    detail().includes('Runs on this trained model set'), 'the unmatched run');
  text = detail();
  assert.ok(text.includes('No recorded attempt stamped version ' + ids.version_b),
    'the newest training run is not linked to the runs that share its '
    + 'workflow fingerprint: ' + text.slice(0, 600));
  assert.ok(text.includes(ids.fingerprint),
    'the fingerprint it would have matched on is still shown: ' + text.slice(0, 900));
  assert.ok(!text.includes('attempt 1'),
    'and no attempt is listed under it: ' + text.slice(0, 600));

  rowFor('Run ' + ids.legacy).click();
  await until(() => detail().includes(ids.legacy), 'the unversioned run');
  text = detail();
  assert.ok(text.includes('published no version id'),
    'a pre-versioning run says its identity is unavailable: ' + text.slice(0, 600));
  assert.ok(text.includes('unavailable'), text.slice(0, 600));
  assert.ok(!text.includes('No recorded attempt stamped'),
    'which is NOT the same statement as having looked and found none: '
    + text.slice(0, 600));

  /* The rail was never involved: opening the section does not disturb the
     record the debug pane is showing. */
  assert.equal(d.getElementById('trainingRunList').closest('dialog').id,
    'trainingDialog', 'the section is its own overlay');

  assert.deepEqual(errors, []);
  process.exit(0);
})().catch(e => {
  process.stderr.write(String((e && e.stack) || e) + '\n');
  process.exit(1);
});
