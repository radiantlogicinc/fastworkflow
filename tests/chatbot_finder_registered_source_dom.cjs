/* The scoped finder on a SELECTED registered source (fix-9eg.3.1.1).

   The other scope script drives this workflow's own store. This one drives an
   experiment registered against the workflow whose evidence lives in its own
   database, opened the way a reader opens it, and asks the question that
   matters for that shape: does the scoped search go to the database the page
   is reading, or to the workflow's default store under the same labels?

   The default store holds decoy turns carrying the same experiment, task and
   attempt, so an answer from the wrong database is recognisable rather than
   merely plausible. The last section covers the other half: a source that
   moves while a scoped walk is open takes the walk, its cursor and its scope
   with it, instead of asking the new database for the old one's labels. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], world = JSON.parse(process.argv[4]);
const errors = [];
const requests = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => {
        requests.push(String(path));
        return fetch(new URL(path, url), options);
      };
    }});
  const w = dom.window, d = w.document;

  const state = () => JSON.stringify({
    running: w.turnFind.running, error: w.turnFind.error, scope: w.turnFind.scope,
    source: w.turnFind.source, moved: w.turnFind.sourceMoved,
    selected: w.benchmarkExperimentSource, rows: w.turnFind.rows.length,
    status: d.getElementById('turnFindStatus').textContent
  });
  async function until(fn, what) {
    for (let i = 0; i < 400; i++) {
      if (fn()) return;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out: ' + what + ' | ' + state() + ' | errors: ' + JSON.stringify(errors));
  }
  async function openUntil(navigate, selector, what) {
    for (let round = 0; round < 40; round++) {
      navigate();
      for (let i = 0; i < 20; i++) {
        const node = d.querySelector(selector);
        if (node) return node;
        await new Promise(r => setTimeout(r, 50));
      }
    }
    throw Error('Timed out: ' + what + ' | ' + state() + ' | errors: ' + JSON.stringify(errors));
  }

  const settled = () => w.turnFind.running === false;
  const scopeBox = () => d.getElementById('turnFindScope');
  const status = () => d.getElementById('turnFindStatus').textContent;
  /* Array.from first: the page's arrays come from the jsdom realm. */
  const keys = () => Array.from(w.turnFind.rows).map(row => row.turn_key);
  const searches = from => requests.slice(from).filter(p => p.indexOf('/api/turns') === 0);
  const sorted = list => list.slice().sort();
  const routed = (path, params) => params.every(p => path.includes(p));
  const external = world.attemptOneTurns.concat([world.failedTurn]);
  /* The reader's own gesture for a registered execution: the benchmark cards
     and the navigation rail both land here. */
  const experimentPage = () => w.openBenchmarkExecution(world.experiment);
  const taskPage = () => w.showExperimentTask(world.experiment, world.task);

  await until(() => w.session && w.session.workflow_path, 'the page never loaded a session');

  /* -- the registered experiment, scoped from its own page ------------- */
  let entry = await openUntil(experimentPage, '[data-find-scope="experiment"]',
    'the selected registered source offered no scoped search');
  await until(() => w.benchmarkExperimentSource === world.experiment,
    'opening the registered execution did not select its source');

  let mark = requests.length;
  entry.click();
  await until(() => settled() && w.turnFind.rows.length > 0,
    'the scoped search on the registered source returned nothing');

  const scopedRequests = searches(mark);
  assert.ok(scopedRequests.length > 0, 'the click issued no search');
  scopedRequests.forEach(path => assert.ok(
    routed(path, ['experiment=' + encodeURIComponent(world.experiment),
                  'benchmark_experiment=' + encodeURIComponent(world.experiment)]),
    'a scoped request did not carry the selected source: ' + path));
  /* The decision: these rows came out of the registered store, not out of the
     default store's turns wearing the same experiment and task. */
  assert.deepEqual(sorted(keys()), sorted(external),
    'the scoped search answered from the wrong database: ' + JSON.stringify(keys()));
  world.decoys.forEach(key => assert.ok(keys().indexOf(key) < 0,
    'a default-store turn answered under the registered source: ' + key));
  assert.ok(scopeBox().textContent.includes(world.experiment), scopeBox().textContent);

  /* -- its task, and one of its attempts -------------------------------- */
  entry = await openUntil(taskPage, '[data-find-scope="task"]',
    'the registered task page offered no scoped search');
  mark = requests.length;
  entry.click();
  await until(() => settled() && w.turnFind.rows.length === external.length,
    'the task-scoped search on the registered source returned the wrong count');

  searches(mark).forEach(path => assert.ok(
    routed(path, ['task=' + encodeURIComponent(world.task),
                  'benchmark_experiment=' + encodeURIComponent(world.experiment)]),
    'a task-scoped request did not carry the selected source: ' + path));
  assert.deepEqual(sorted(keys()), sorted(external));

  entry = await openUntil(taskPage, '[data-attempt="2"] [data-find-scope="attempt"]',
    'the registered attempt row offered no scoped search');
  mark = requests.length;
  entry.click();
  await until(() => settled() && w.turnFind.rows.length === 1,
    'the attempt-scoped search on the registered source did not return the attempt');

  assert.deepEqual(keys(), [world.failedTurn]);
  assert.equal(w.turnFind.scope.attempt, 2);
  searches(mark).forEach(path => assert.ok(routed(path, ['attempt=2', 'benchmark_experiment=']),
    'an attempt-scoped request did not carry the selected source: ' + path));
  assert.ok(status().includes('the whole of this scope'), 'status: ' + status());

  /* -- the source moves out from under an open walk -------------------- */
  /* The request already in flight was addressed to the registered store. Its
     answer must not land in a page that now reads this workflow's own store,
     under a scope chosen somewhere else. */
  entry = await openUntil(taskPage, '[data-attempt="1"] [data-find-scope="attempt"]',
    'the registered attempt row offered no scoped search again');
  entry.click();
  assert.equal(w.turnFind.running, true, 'the scoped search never started');
  w.benchmarkExperimentSource = null;
  await until(() => settled() && w.turnFind.sourceMoved === true,
    'the walk survived the source moving under it');

  assert.equal(w.turnFind.rows.length, 0,
    'the registered store answered into a page reading another source: ' + JSON.stringify(keys()));
  assert.equal(w.turnFind.scope, null, 'the scope outlived the source it was chosen in');
  assert.ok(status().includes('source'), 'the page did not say why: ' + status());
  external.forEach(key => assert.ok(d.getElementById('detail').textContent.indexOf(key) < 0,
    'a row from the abandoned source is still on the page: ' + key));

  /* -- and the next search reads the source now selected ---------------- */
  mark = requests.length;
  d.getElementById('turnFindText').value = '';
  w.turnFindStart();
  await until(() => settled() && w.turnFind.rows.length > 0,
    'no search ran after the source changed');

  assert.ok(world.decoys.every(key => keys().indexOf(key) >= 0),
    'the default store did not answer its own search: ' + JSON.stringify(keys()));
  searches(mark).forEach(path => {
    assert.ok(!path.includes('benchmark_experiment='),
      'a search still addressed the abandoned source: ' + path);
    assert.ok(!path.includes('experiment='), 'a dropped scope was still sent: ' + path);
  });
  assert.equal(scopeBox().className, '');

  /* -- what is offered, and the one shape still deferred ---------------- */
  assert.equal(w.turnFindEntrySupported(), true, 'the default store lost its entries');
  w.benchmarkExperimentSource = world.experiment;
  assert.equal(w.turnFindEntrySupported(), true,
    'a selected registered source was refused a scoped search it can answer');
  /* A workspace logical experiment spans sealed stores and the turn route
     takes one store per request, so it is not offered yet (fix-luut). */
  w.session.workspace_mode = true;
  const deferred = d.createElement('div');
  w.turnFindEntryButton(deferred, {experiment: world.experiment}, 'Find problems', 'help');
  assert.equal(deferred.childNodes.length, 0,
    'a scope was offered across stores the finder cannot route to');
  w.session.workspace_mode = false;
  w.benchmarkExperimentSource = null;

  assert.deepEqual(errors, []);
  dom.window.close();
  console.log('ok');
})().catch(err => { console.error(err && err.stack || String(err)); process.exit(1); });
