const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
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
  const summaries = () => [...d.querySelectorAll('#convList summary')];
  const summary = text => summaries().find(node => node.textContent.includes(text));
  const button = text => [...d.querySelectorAll('button')].find(node => node.textContent === text);

  await until(() => w.session && w.hierarchyRoot);
  d.getElementById('modeDebug').click();
  d.getElementById('navBenchmarks').click();
  await until(() => summary('Tuning benchmark'));

  d.getElementById('benchmarkSetupBtn').click();
  await until(() => d.querySelector('#detail input[aria-label="Select Tuning benchmark"]'));
  d.querySelector('#detail input[aria-label="Select Tuning benchmark"]').click();
  button('Archive selected').click();
  await until(() => d.getElementById('confirmDelete').textContent === 'Archive benchmark');
  d.getElementById('confirmDelete').click();
  await until(() => !summary('Tuning benchmark'));
  await until(() => d.getElementById('detail').textContent.includes('Archived benchmarks are hidden'));

  button('Show archived benchmarks').click();
  await until(() => summary('Tuning benchmark'));
  await until(() => [...d.querySelectorAll('#detail .pill')].some(pill => pill.textContent === 'Archived'));
  d.querySelector('#detail input[aria-label="Select Tuning benchmark"]').click();
  button('Unarchive selected').click();
  await until(() => d.getElementById('confirmDelete').textContent === 'Unarchive benchmark');
  d.getElementById('confirmDelete').click();
  await until(() => summary('Tuning benchmark') && button('Archive selected'));

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
