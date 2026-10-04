const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], eid = process.argv[4];
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
    for (let i = 0; i < 150; i++) {
      if (fn()) return;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out: ' + d.getElementById('detail').textContent);
  }
  const experimentName = 'Experiment \u00b7 ' + eid.slice(-8);
  const summaries = () => [...d.querySelectorAll('#convList summary')];
  const summary = text => summaries().find(node => node.textContent.includes(text));
  const detailButton = text => [...d.querySelectorAll('#detail button')]
    .find(node => node.textContent === text);

  // Startup is two reads, not one: /api/session lands first and only then does
  // the page read /api/navigation under that session's scope. Navigating before
  // the session arrives reads the rail under the empty scope, and the scope
  // change that follows drops the selection made in between -- which left this
  // walk clicking the experiment and then watching the benchmarks placeholder
  // repaint over it. So wait for the session AND for the navigation that scope
  // owns before touching anything.
  await until(() => w.session && w.hierarchyRoot);
  d.getElementById('modeDebug').click();
  await until(() => summaries().length);
  d.getElementById('navBenchmarks').click();
  await until(() => summary('Tuning benchmark'));
  await until(() => summary(experimentName));
  assert.equal(d.querySelector('#convList .navArchiveToggle'), null);

  summary(experimentName).click();
  await until(() => detailButton('Archive experiment'));
  detailButton('Archive experiment').click();
  await until(() => detailButton('Unarchive experiment'));
  assert.equal(summary(experimentName), undefined);
  assert.ok(d.querySelector('#convList [aria-label="Show all experiments"]'));

  summary('Tuning benchmark').click();
  await until(() => d.querySelectorAll('#detail .recordCard').length === 1);
  d.querySelector('#convList [aria-label="Show all experiments"]').click();
  await until(() => summary(experimentName));
  assert.ok(summary(experimentName).parentElement.classList.contains('archived'));
  await until(() => d.querySelector('#detail .recordCard.archived'));
  assert.equal(d.querySelector('#detail .recordCard.archived .pill').textContent, 'Archived');
  assert.ok(d.querySelector('#convList [aria-label="Hide archived experiments"]'));

  d.querySelector('#convList [aria-label="Hide archived experiments"]').click();
  await until(() => !summary(experimentName));
  d.querySelector('#convList [aria-label="Show all experiments"]').click();
  await until(() => summary(experimentName));
  summary(experimentName).click();
  await until(() => detailButton('Unarchive experiment'));
  detailButton('Unarchive experiment').click();
  await until(() => detailButton('Archive experiment'));
  assert.ok(summary(experimentName));
  assert.equal(d.querySelector('#convList .navArchiveToggle'), null);

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
