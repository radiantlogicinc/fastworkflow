/* The turn finder scoped to an experiment, task or attempt (fix-9eg.3.1.1).

   Real clicks on the shipped controls against a real server and a real store.
   What is being proven is not that a scope can be set, but that every request
   the walk makes carries it, that the page says which scope it is answering
   about, and that continuation stays inside it.

   Every navigation is retried rather than assumed: the rail's periodic
   refresh repaints #detail whenever no record is selected and the finder is
   not holding the pane, so a control that was there a moment ago can be gone
   before it is clicked. */
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
    running: w.turnFind.running, error: w.turnFind.error,
    scope: w.turnFind.scope, complete: w.turnFind.complete,
    rows: w.turnFind.rows.length, status: d.getElementById('turnFindStatus').textContent
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
    throw Error('Timed out: ' + what + ' | errors: ' + JSON.stringify(errors));
  }

  const settled = () => w.turnFind.running === false;
  const scopeBox = () => d.getElementById('turnFindScope');
  const status = () => d.getElementById('turnFindStatus').textContent;
  /* Array.from first: the page's arrays come from the jsdom realm, and a
     strict deepEqual compares prototypes as well as contents. */
  const keys = () => Array.from(w.turnFind.rows).map(row => row.turn_key);
  const searches = from => requests.slice(from).filter(p => p.indexOf('/api/turns') === 0);
  const failureChip = () => Array.from(d.querySelectorAll('#turnFindMarkers button'))
    .filter(b => b.textContent.includes('command returned failure'))[0];
  const experimentPage = () => w.showExperiment(world.experiment);
  const taskPage = () => w.showExperimentTask(world.experiment, world.task);

  await until(() => w.session && w.session.workflow_path, 'the page never loaded a session');

  /* -- the experiment's own entry ------------------------------------- */
  let entry = await openUntil(experimentPage, '[data-find-scope="experiment"]',
    'the experiment page never offered a scoped search');
  let mark = requests.length;
  entry.click();
  await until(() => settled() && w.turnFind.rows.length > 0,
    'the experiment-scoped search returned nothing');

  assert.equal(w.turnFind.scope.experiment, world.experiment);
  assert.equal(w.turnFind.scope.task, null);
  assert.equal(w.turnFind.scope.attempt, null);
  const scopedRequests = searches(mark);
  assert.ok(scopedRequests.length > 0, 'the click issued no search');
  scopedRequests.forEach(path => assert.ok(
    path.includes('experiment=' + encodeURIComponent(world.experiment)),
    'a request left the scope behind: ' + path));
  assert.ok(w.turnFind.rows.every(row => row.experiment_id === world.experiment),
    'another experiment answered under this one: ' + JSON.stringify(keys()));
  assert.match(scopeBox().className, /scoped/);
  assert.ok(scopeBox().textContent.includes(world.experiment),
    'the scope is applied but not shown: ' + scopeBox().textContent);
  assert.ok(d.getElementById('turnFindScopeClear'), 'no control clears the scope');

  /* -- the task's entry, across its attempts --------------------------- */
  entry = await openUntil(taskPage, '[data-find-scope="task"]',
    'the task page never offered a scoped search');
  mark = requests.length;
  entry.click();
  await until(() => settled() && w.turnFind.rows.length > 0,
    'the task-scoped search returned nothing');

  assert.equal(w.turnFind.scope.task, world.task);
  searches(mark).forEach(path => {
    assert.ok(path.includes('task=' + encodeURIComponent(world.task)), path);
    assert.ok(path.includes('experiment=' + encodeURIComponent(world.experiment)), path);
  });
  assert.ok(w.turnFind.rows.every(row =>
    row.task_id === world.task && row.experiment_id === world.experiment),
    'a turn from outside the task: ' + JSON.stringify(keys()));

  /* -- one attempt ----------------------------------------------------- */
  entry = await openUntil(taskPage, '[data-attempt="2"] [data-find-scope="attempt"]',
    'the attempt row never offered a scoped search');
  entry.click();
  await until(() => settled() && w.turnFind.rows.length === world.attemptTwoTurns.length,
    'the attempt-scoped search did not return the attempt');

  assert.deepEqual(keys().slice().sort(), world.attemptTwoTurns.slice().sort());
  assert.equal(w.turnFind.scope.attempt, 2);
  assert.ok(scopeBox().textContent.includes('attempt 2'), scopeBox().textContent);
  /* The walk covered the scope. Saying it covered the store would be a claim
     about turns it never read. */
  assert.ok(status().includes('the whole of this scope'), 'status: ' + status());
  assert.ok(!status().includes('the whole store'), 'status: ' + status());
  assert.ok(d.getElementById('detail').textContent.includes('Scoped to'),
    'the results do not say what they are scoped to');

  /* -- a marker chip narrows WITHIN the scope -------------------------- */
  const chip = failureChip();
  assert.ok(chip, 'the failure chip is not on the page');
  mark = requests.length;
  chip.click();
  // A chip only edits the question; nothing is asked until Find.
  assert.equal(requests.length, mark, 'toggling a chip searched on its own');
  d.getElementById('turnFindGo').click();
  await until(() => settled() && w.turnFind.markers.step_unsuccessful === true
    && w.turnFind.rows.length === 1, 'the scoped marker search never settled');

  assert.deepEqual(keys(), [world.failedTurn]);
  assert.equal(w.turnFind.scope.attempt, 2, 'the marker chip dropped the scope');
  searches(mark).forEach(path => {
    assert.ok(path.includes('markers_all=step_unsuccessful'), path);
    assert.ok(path.includes('attempt=2'), path);
  });

  /* -- clearing the scope widens the same search ----------------------- */
  mark = requests.length;
  d.getElementById('turnFindScopeClear').click();
  await until(() => settled() && w.turnFind.scope === null && w.turnFind.rows.length === 2,
    'clearing the scope did not widen the search');

  assert.ok(keys().includes(world.otherFailedTurn),
    'the other experiment\'s failure stayed hidden after the scope was cleared');
  assert.equal(scopeBox().className, '');
  assert.ok(status().includes('the whole store'), 'status: ' + status());
  searches(mark).forEach(path => assert.ok(!path.includes('experiment='),
    'a cleared scope was still sent: ' + path));

  failureChip().click();
  d.getElementById('turnFindGo').click();
  await until(() => settled() && w.turnFind.markers.step_unsuccessful === false,
    'the marker filter never came off');

  /* -- a scope dropped mid-flight disowns the page it asked for -------- */
  /* The outstanding request was made WITH the scope; letting its answer land
     would add that scope's rows to a list the page now calls unscoped. */
  entry = await openUntil(taskPage, '[data-attempt="2"] [data-find-scope="attempt"]',
    'the attempt row never offered a scoped search again');
  entry.click();
  assert.equal(w.turnFind.running, true, 'the scoped search never started');
  d.getElementById('turnFindScopeClear').click();
  await until(() => settled() && w.turnFind.scope === null && w.turnFind.rows.length > 0,
    'the search never settled after the scope was dropped mid-flight');

  assert.equal(w.turnFind.rows.length, world.page,
    'the disowned scoped page was merged into the unscoped result: ' + JSON.stringify(keys()));
  assert.equal(new Set(keys()).size, keys().length, 'a row arrived twice');
  assert.ok(keys().some(key => world.attemptTwoTurns.indexOf(key) < 0),
    'the unscoped result holds nothing but the dropped scope\'s rows');

  /* -- continuation stays inside the scope ----------------------------- */
  entry = await openUntil(taskPage, '[data-attempt="1"] [data-find-scope="attempt"]',
    'the first attempt never offered a scoped search');
  entry.click();
  await until(() => settled() && w.turnFind.rows.length === world.page,
    'the first segment of the scoped walk never arrived');

  assert.equal(w.turnFind.complete, false,
    'an attempt of ' + world.bulkTurns + ' turns cannot be walked in one scan');
  assert.ok(status().includes('not the whole of this scope'), 'status: ' + status());
  const more = d.getElementById('turnFindMore');
  assert.ok(more, 'an unfinished scoped walk offered no way to continue');
  mark = requests.length;
  more.click();
  await until(() => settled() && w.turnFind.rows.length === world.page * 2,
    'the continuation never arrived');

  searches(mark).forEach(path => {
    assert.ok(path.includes('attempt=1'), 'the continuation left the scope: ' + path);
    assert.ok(path.includes('task=' + encodeURIComponent(world.task)), path);
    assert.ok(path.includes('resume_after='), path);
  });
  assert.equal(new Set(keys()).size, keys().length, 'the continuation repeated a row');
  assert.ok(w.turnFind.rows.every(row => row.attempt === 1 && row.task_id === world.task),
    'the continuation reached outside the attempt');

  /* -- an overlapping search: the later one owns the view -------------- */
  entry = await openUntil(experimentPage, '[data-find-scope="experiment"]',
    'the experiment page never offered a scoped search again');
  entry.click();
  w.turnFindStart();            /* a second search before the first answered */
  await until(() => settled() && w.turnFind.rows.length > 0,
    'overlapping scoped searches left the finder running');

  assert.equal(new Set(keys()).size, keys().length,
    'a disowned response was merged into the view');
  assert.equal(w.turnFind.rows.length, world.page,
    'a disowned response added rows: ' + w.turnFind.rows.length);
  assert.equal(w.turnFind.scope.experiment, world.experiment,
    'the overlap dropped the scope');

  /* -- the source boundary takes the scope with it --------------------- */
  w.resetSourceScopedState();
  assert.equal(w.turnFind.scope, null, 'the source boundary left a scope behind');
  assert.equal(scopeBox().className, '');
  assert.equal(d.getElementById('turnFindText').value, '');

  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
