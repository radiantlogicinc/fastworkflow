/* Rail tabs, a picker row, and an experiment row, driven from the keyboard. */
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
  async function until(fn, what) {
    for (let i = 0; i < 400; i++) {
      if (fn()) return;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out waiting for ' + what + ': ' + d.getElementById('detail').textContent.slice(0, 240));
  }
  function key(el, name) {
    el.focus();
    const event = new w.KeyboardEvent('keydown', {key: name, bubbles: true, cancelable: true});
    el.dispatchEvent(event);
    return event;
  }

  await until(() => d.getElementById('convList').textContent.includes('2026-09-07'), 'conversations');
  const conv = d.getElementById('navConversations');
  const bench = d.getElementById('navBenchmarks');
  assert.equal(conv.tabIndex, 0);
  assert.equal(bench.tabIndex, -1);

  key(conv, 'ArrowRight');
  assert.equal(bench.getAttribute('aria-selected'), 'true');
  assert.equal(conv.getAttribute('aria-selected'), 'false');
  assert.equal(bench.tabIndex, 0);
  assert.equal(conv.tabIndex, -1);
  assert.equal(d.activeElement, bench);
  await until(() => d.getElementById('convList').textContent.includes('Tuning benchmark'), 'benchmarks tab');
  assert.equal(d.getElementById('convList').textContent.includes('2026-09-07'), false);

  key(bench, 'ArrowLeft');
  assert.equal(conv.getAttribute('aria-selected'), 'true');
  assert.equal(bench.getAttribute('aria-selected'), 'false');
  assert.equal(conv.tabIndex, 0);
  assert.equal(bench.tabIndex, -1);
  assert.equal(d.activeElement, conv);
  assert.ok(d.getElementById('convList').textContent.includes('2026-09-07'));

  key(conv, 'End');
  assert.equal(bench.getAttribute('aria-selected'), 'true');
  assert.equal(d.activeElement, bench);
  assert.equal(bench.tabIndex, 0);
  assert.equal(conv.tabIndex, -1);

  key(bench, 'Home');
  assert.equal(conv.getAttribute('aria-selected'), 'true');
  assert.equal(d.activeElement, conv);

  key(conv, 'ArrowLeft');
  assert.equal(bench.getAttribute('aria-selected'), 'true', 'ArrowLeft wraps to the last tab');
  key(bench, 'ArrowRight');
  assert.equal(conv.getAttribute('aria-selected'), 'true', 'ArrowRight wraps to the first tab');

  w.showExperiments();
  await until(() => d.querySelector('#detail .listItem'), 'experiment rows');
  const expRow = d.querySelector('#detail .listItem');
  assert.equal(expRow.getAttribute('role'), 'button');
  assert.equal(expRow.tabIndex, 0);
  const space = key(expRow, ' ');
  assert.equal(space.defaultPrevented, true);
  await until(() => d.getElementById('detail').textContent.includes('RECORDED EXPERIMENT'), 'experiment page');

  w.setTopMode('picker');
  w.loadPicker();
  await until(() => d.querySelector('#browseList .dirRow'), 'browse rows');
  const before = d.getElementById('browsePath').textContent;
  const dir = d.querySelector('#browseList .dirRow');
  assert.equal(dir.getAttribute('role'), 'button');
  assert.equal(dir.tabIndex, 0);
  const dirSpace = key(dir, ' ');
  assert.equal(dirSpace.defaultPrevented, true);
  await until(() => d.getElementById('browsePath').textContent !== before, 'browse moved');

  await until(() => d.querySelector('#wfCandidates .wfItem'), 'picker rows');
  const picker = d.querySelector('#wfCandidates .wfItem');
  assert.equal(picker.getAttribute('role'), 'button');
  assert.equal(picker.tabIndex, 0);
  const enter = key(picker, 'Enter');
  assert.equal(enter.defaultPrevented, true);
  assert.ok(
    d.getElementById('pickerStatus').textContent.startsWith('Starting '),
    d.getElementById('pickerStatus').textContent
  );
  /* Let the selection request finish before the window is closed. Its
     continuation writes the status line, and a closed document is not a
     keyboard failure. */
  await until(() => !d.getElementById('pickerStatus').textContent.startsWith('Starting '),
    'workflow selection settled');

  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
