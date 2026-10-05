/* A selected turn survives the navigation refresh that lands on top of it.
 *
 * The product race, driven deterministically: `/api/navigation` is delayed in
 * transit so its response is GUARANTEED to arrive after a scoped turn has
 * been selected and while that turn's own trace is still loading. That window
 * is the bug — `attachTraceHierarchy` can only align a turn whose trace has
 * already arrived, so a refresh landing inside it used to find no path in the
 * rail and repaint "No conversations yet" over the trace.
 *
 * Both refreshes are exercised: the one startup fires, and a later one, which
 * in workspace mode is the dangerous case because the rail may have no path
 * to a scoped turn at all. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], storeId = process.argv[4], turnKey = process.argv[5];
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
    throw Error('Timed out waiting for ' + what + ': '
      + d.getElementById('detail').textContent.slice(0, 400));
  }
  const detailText = () => d.getElementById('detail').textContent;

  d.getElementById('modeDebug').click();
  await until(() => w.session && w.session.workspace_mode, 'the session');

  /* Deliberately NOT waiting for startup to finish. The navigation read is
     still in flight and its response is held back by the proxy, so selecting
     a turn now puts the selection squarely inside the window the refresh
     used to overwrite. */
  w.selectWorkspaceTurn(storeId, turnKey);
  assert.equal(w.state.turnKey, turnKey, 'the turn is selected synchronously');

  /* The trace arrives and stays. Without the guard the delayed navigation
     response lands in between and the assertion below sees the placeholder. */
  await until(() => detailText().includes('Feedback'), 'the turn detail');
  await until(() => d.getElementById('feedback-comment'), 'the composer');
  assert.ok(!detailText().includes('No conversations yet'),
    'the navigation refresh repainted over the selected turn: '
    + detailText().slice(0, 300));
  assert.equal(w.state.turnKey, turnKey,
    'the refresh cleared the selection out from under the open trace');

  /* Now the periodic case: a refresh fired long after the turn is fully
     loaded and rendered. In workspace mode the rail may hold no path to a
     scoped turn, so this is not a one-time startup problem. */
  await w.refreshConvs();
  await new Promise(r => setTimeout(r, 100));
  assert.equal(w.state.turnKey, turnKey,
    'a later refresh cleared the selection');
  assert.ok(!detailText().includes('No conversations yet'),
    'a later refresh repainted over the open trace: '
    + detailText().slice(0, 300));
  assert.ok(d.getElementById('feedback-comment'),
    'the composer is gone after a later refresh');

  /* And the empty state is still reachable the way it is meant to be: by a
     navigation gesture, not by a background read. The Conversations tab has
     nothing selected in a sealed workspace, so switching to it is exactly
     the case the placeholder exists for. */
  w.setNavigationTab('conversations');
  await until(() => detailText().includes('No conversations yet'),
    'the tab empty state');
  assert.equal(w.state.turnKey, null,
    'switching tabs with nothing selected still releases the pane');

  await new Promise(r => setTimeout(r, 200));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
