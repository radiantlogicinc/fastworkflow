/* The composer on READ-ONLY, annotated evidence, driven in a real DOM.
 *
 * The other DOM harness drives a writable store, where the composer has
 * always been open. This one drives the branch fix-9eg.19.1 added: a sealed
 * workspace archive, which the page must NOT treat as read-only feedback.
 * The whole branch is three lines of `renderFeedback` deciding whether to
 * hide the composer, and HTTP coverage cannot see it — a page that hid the
 * box would pass every server-side assertion while leaving a person with
 * nothing to type into. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], storeId = process.argv[4], turnKey = process.argv[5];
const experimentId = process.argv[6], taskId = process.argv[7];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn) {
    for (let i = 0; i < 200; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out: ' + d.getElementById('detail').textContent.slice(0, 600));
  }
  const buttons = () => [...d.querySelectorAll('#detail button')];
  const button = text => buttons().find(node => node.textContent === text);
  const detailText = () => d.getElementById('detail').textContent;

  /* Wait for the page to actually BE ready, not just for a session object:
     workspace startup fetches the manifest's stores and experiments and only
     then reads the navigation. Acting while that is in flight is a real thing
     a person can do, and `refreshConvs` is guarded against it, but starting
     the run mid-startup would test the guard instead of the composer. The
     signal is the product's own: the workspace chrome goes visible when its
     reads have landed. */
  d.getElementById('modeDebug').click();
  await until(() => w.session && w.session.workspace_mode);
  await until(() => d.getElementById('workspaceChrome').className === 'visible');

  /* Scoped by store id, the way a workspace deep link opens a trace. */
  w.selectWorkspaceTurn(storeId, turnKey);
  await until(() => button('Save feedback'));

  /* The branch under test: sealed evidence, composer OPEN, and the page
     saying where the comment will go rather than refusing to take one. */
  const area = d.getElementById('feedback-comment');
  const save = button('Save feedback');
  assert.equal(area.closest('div').hidden, false,
    'the composer must stay open on annotated read-only evidence');
  await until(() => !save.disabled);
  assert.ok(detailText().includes(
    'This evidence is read-only. Your comment is recorded beside it and never changes it.'),
    'the page says the archive is not being modified: ' + detailText().slice(0, 300));

  /* A categorized comment, picked through the same tabs a person clicks. */
  button('Conclusions').click();
  await until(() => button('What went wrong'));
  button('What went wrong').click();
  assert.equal(button('What went wrong').getAttribute('aria-selected'), 'true');
  area.value = 'The sealed run stopped one item short of the request.';
  save.click();
  await until(() => detailText().includes('stopped one item short'));
  assert.ok(detailText().includes('Conclusions \u00b7 What went wrong'),
    'the stored classification is the one that was clicked');
  assert.equal(area.value, '');

  /* And it is in the task Feedback view, which reads the manifest's stores
     rather than the one database a live workflow would have. */
  w.taskView = 'feedback';
  w.showExperimentTask(experimentId, taskId);
  await until(() => detailText().includes('Every recorded comment on this task'));
  await until(() => [...d.querySelectorAll('#detail .listItem')].length >= 1);
  const rows = [...d.querySelectorAll('#detail .listItem')]
    .map(row => row.textContent).join('\n');
  assert.ok(rows.includes('stopped one item short'),
    'the comment recorded beside the archive is in the task view: ' + rows.slice(0, 400));
  assert.ok(rows.includes('Conclusions \u00b7 What went wrong'));

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
