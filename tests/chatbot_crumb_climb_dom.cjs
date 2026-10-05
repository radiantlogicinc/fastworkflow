/* Climbing out of a trace with the breadcrumb rather than the Up arrow: every
   crumb above the current level has to land on its own level, the trace
   components below the turn included. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], turnKey = process.argv[4];
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  const detail = d.getElementById('detail');
  async function until(fn) {
    for (let i = 0; i < 150; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + detail.textContent + ' | errors: ' + JSON.stringify(errors));
  }

  const heading = () => (detail.querySelector('.levelHead h2') || {}).textContent || '';
  const crumbButtons = () => Array.from(detail.querySelectorAll('nav.crumbs button'));
  const crumb = label => {
    const found = crumbButtons().find(b => b.textContent === label);
    assert.ok(found, 'no "' + label + '" crumb in: ' + crumbButtons().map(b => b.textContent).join(' › '));
    return found;
  };
  const row = title => {
    const found = Array.from(detail.querySelectorAll('.wfRow'))
      .find(r => r.getAttribute('aria-label').startsWith('Inspect ' + title));
    assert.ok(found, 'no "' + title + '" row');
    return found;
  };

  d.getElementById('modeDebug').click();
  await until(() => d.querySelectorAll('#convList summary').length);
  d.getElementById('recordNavDown').click();
  await until(() => crumbButtons().some(b => b.textContent === '2026-09-08'));
  d.getElementById('recordNavDown').click();
  await until(() => detail.textContent.includes('1 turns'));
  d.getElementById('recordNavDown').click();
  await until(() => heading().includes('climb the crumbs') && detail.querySelector('.wfRow'));

  // turn -> Execution -> Step 1, walked through the waterfall rows.
  row('Execution').click();
  await until(() => heading().startsWith('Execution'));
  row('Step 1').click();
  await until(() => heading().startsWith('Step 1'));
  assert.ok(crumbButtons().some(b => b.textContent === 'Execution'));

  crumb('Execution').click();
  await until(() => heading().startsWith('Execution'));
  assert.ok(detail.querySelector('.wfRow'), 'Execution level lists its steps');
  assert.equal(crumbButtons()[crumbButtons().length - 1].textContent, 'Execution');

  // The turn crumb climbs from a step as well, and the stage below it is
  // reachable again afterwards.
  row('Step 2').click();
  await until(() => heading().startsWith('Step 2'));
  crumb('climb the crumbs').click();
  await until(() => heading().includes('climb the crumbs'));
  row('Planning').click();
  await until(() => heading().startsWith('Planning'));
  crumb('climb the crumbs').click();
  await until(() => heading().includes('climb the crumbs'));
  row('Execution').click();
  await until(() => heading().startsWith('Execution'));
  row('Step 2').click();
  await until(() => heading().startsWith('Step 2'));
  crumb('Execution').click();
  await until(() => heading().startsWith('Execution'));

  await new Promise(r => setTimeout(r, 300));
  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
