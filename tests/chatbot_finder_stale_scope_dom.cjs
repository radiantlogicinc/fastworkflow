/* The turn finder under an experiment scope the store cannot resolve (fix-neo2).

   The server refuses a scoped read it cannot honour with 409 and says why in
   the body. The finder must show that refusal and stop; a search that stays
   "in progress" forever reads as a slow store rather than a dead scope. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], missingScope = process.argv[4];
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | ' + JSON.stringify({
      running: w.turnFind.running, error: w.turnFind.error,
      rows: w.turnFind.rows.length, status: d.getElementById('turnFindStatus').textContent
    }) + ' | errors: ' + JSON.stringify(errors));
  }
  const search = text => {
    d.getElementById('turnFindText').value = text;
    d.getElementById('turnFind').dispatchEvent(new w.Event('submit'));
  };

  await until(() => w.session && w.session.workflow_path, 'the page never loaded a session');

  /* A scope the store does not have: the server answers 409. */
  w.benchmarkExperimentSource = missingScope;
  search('recorded');
  await until(() => w.turnFind.running === false,
    'the search never stopped running under a dead scope');
  assert.equal(w.turnFind.rows.length, 0);
  assert.ok(w.turnFind.error, 'the refusal was never recorded');
  const status = d.getElementById('turnFindStatus').textContent;
  assert.match(status, /Search failed/);
  assert.ok(status.includes(missingScope),
    'the status did not say which scope was refused: ' + status);

  /* And the finder recovers: clearing the scope searches the store again. */
  w.benchmarkExperimentSource = null;
  search('recorded');
  await until(() => w.turnFind.rows.length > 0,
    'the finder did not recover once the dead scope was cleared');
  assert.equal(w.turnFind.error, null);
  assert.equal(w.turnFind.running, false);
  assert.match(d.getElementById('turnFindStatus').textContent, /scanned/);

  /* Two searches started back to back (the debounce a keystroke schedules,
     then a second start before its answer): the later one owns the view and
     the finder still settles rather than being left "in progress". */
  d.getElementById('turnFindText').value = 'recorded';
  d.getElementById('turnFindText').dispatchEvent(new w.Event('input'));
  w.turnFindStart();
  await until(() => w.turnFind.running === false && w.turnFind.rows.length > 0,
    'overlapping searches left the finder running');

  /* The same overlap under the dead scope: still a reported refusal. */
  w.benchmarkExperimentSource = missingScope;
  d.getElementById('turnFindText').value = 'recorded';
  d.getElementById('turnFindText').dispatchEvent(new w.Event('input'));
  w.turnFindStart();
  await until(() => w.turnFind.running === false && w.turnFind.error,
    'overlapping searches under a dead scope left the finder running');
  assert.match(d.getElementById('turnFindStatus').textContent, /Search failed/);

  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
