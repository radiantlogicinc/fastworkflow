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

  await until(() => w.session && w.hierarchyRoot);
  d.getElementById('modeDebug').click();
  d.getElementById('navBenchmarks').click();
  await until(() => summary('Tuning benchmark'));
  await until(() => summary('v1'));
  assert.equal(summary(experimentName), undefined);

  summary('v1').click();
  await until(() => d.getElementById('convList').textContent.includes(
    'Archived experiments are listed under Tools.'));

  summary('Tuning benchmark').click();
  await until(() => d.getElementById('detail').textContent.includes(
    'Archived experiments are listed separately'));
  const detail = d.getElementById('detail').textContent;
  assert.ok(detail.includes('It is archived, so it is listed under Archived experiments.'), detail);
  assert.equal(
    [...d.querySelectorAll('#detail .recordCard')].some(card => card.textContent.includes(eid)),
    false);

  [...d.querySelectorAll('#detail button')]
    .find(button => button.textContent === 'Open archived experiments').click();
  await until(() => d.getElementById('detail').textContent.includes(eid)
    && d.getElementById('detail').textContent.includes('BENCHMARK EVIDENCE'));

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
