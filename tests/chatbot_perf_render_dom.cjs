/* Abort supersedes without error notices; large attrs stay lazy; transcript caps.
 */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const realFetch = fetch;
  let hangFirstPair = true;
  let hung = 0;
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = async (path, options) => {
        const target = new URL(path, url);
        if (target.pathname.startsWith('/api/turn/') || target.pathname.startsWith('/api/spans/')) {
          if (hangFirstPair && hung < 2) {
            hung += 1;
            await new Promise((resolve, reject) => {
              const signal = options && options.signal;
              if (signal) {
                if (signal.aborted) {
                  reject(Object.assign(new Error('aborted'), {name: 'AbortError'}));
                  return;
                }
                signal.addEventListener('abort', () => {
                  reject(Object.assign(new Error('aborted'), {name: 'AbortError'}));
                });
              }
            });
          }
          return new window.Response(JSON.stringify({error: 'missing'}), {
            status: 404, headers: {'Content-Type': 'application/json'}
          });
        }
        return realFetch(target, options);
      };
    }});
  const w = dom.window, d = w.document;
  const notices = [], notify = w.showNotice;
  w.showNotice = (message, kind, detail) => {
    notices.push({message, kind, detail});
    return notify.call(w, message, kind, detail);
  };
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) {
      if (fn()) return;
      await new Promise(r => setTimeout(r, 25));
    }
    throw Error('Timed out waiting for ' + what);
  }

  await until(() => w.session, 'session');

  /* --- abort on supersede --- */
  notices.length = 0;
  w.selectTurn('missing-first');
  await new Promise(r => setTimeout(r, 30));
  hangFirstPair = false;
  w.selectTurn('missing-second');
  await until(() => d.getElementById('detail').textContent.includes('Failed to load turn'),
    'second turn failure');
  assert.ok(!notices.some(n => n.kind === 'error'),
    'aborted request must not emit an error notice: ' + JSON.stringify(notices));
  assert.ok(!d.getElementById('detail').textContent.includes('aborted'));

  /* --- lazy attribute sections --- */
  const container = d.createElement('div');
  const big = 'x'.repeat(25 * 1024);
  w.appendAttrSection(container, 'huge payload', big);
  const details = container.querySelector('details');
  assert.ok(details, 'large attribute must render inside details');
  assert.equal(details.querySelector('pre'), null, 'content absent until opened');
  details.open = true;
  details.dispatchEvent(new w.Event('toggle'));
  assert.ok(details.querySelector('pre'), 'opening loads the pretty content');
  assert.ok(details.querySelector('pre').textContent.includes('xxx'));

  /* --- transcript cap --- */
  const log = d.getElementById('chatLog');
  log.innerHTML = '';
  w.TM_TRANSCRIPT_CAP = 5;
  for (let i = 0; i < 8; i++) { w.tmBubble('user', 'msg ' + i); }
  assert.equal(log.querySelectorAll('.chatMsg').length, 5);
  assert.ok(log.textContent.includes('earlier messages hidden'));

  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
