/* Unchanged /api/navigation answers 304 and must not rebuild the rail.
 *
 * After the first load, a background refreshConvs with the stored ETag must
 * leave the rail DOM intact. An explicit Refresh still forces a body and
 * shows the "Navigation refreshed" notice.
 */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  let navigationCalls = 0;
  let lastStatus = null;
  const realFetch = fetch;
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = async (path, options) => {
        const target = new URL(path, url);
        const response = await realFetch(target, options);
        if (target.pathname === '/api/navigation') {
          navigationCalls += 1;
          lastStatus = response.status;
        }
        return response;
      };
    }});
  const w = dom.window, d = w.document;
  const notices = [], notify = w.showNotice;
  w.showNotice = (message, kind, detail) => {
    notices.push(message); return notify.call(w, message, kind, detail);
  };
  async function until(fn, what) {
    for (let i = 0; i < 300; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out waiting for ' + what);
  }

  d.getElementById('modeDebug').click();
  await until(() => w.hierarchyRoot, 'hierarchy root');
  await until(() => w.navigationETag, 'navigation etag');
  const rail = d.getElementById('convList');
  const before = rail.innerHTML;
  const callsBefore = navigationCalls;

  await w.refreshConvs();
  await new Promise(r => setTimeout(r, 50));
  assert.equal(lastStatus, 304, 'background refresh should 304');
  assert.ok(navigationCalls > callsBefore);
  assert.equal(rail.innerHTML, before, '304 must not rebuild the rail');

  notices.length = 0;
  const button = d.getElementById('refreshBtn');
  button.disabled = false;
  button.click();
  await until(() => notices.includes('Navigation refreshed'), 'force refresh notice');
  assert.equal(lastStatus, 200, 'explicit refresh must fetch a body');

  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
