/* checkSession unreachable banner after 3 failures, then recovery.
 */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const realFetch = fetch;
  let failSession = false;
  let hidden = false;
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      Object.defineProperty(window.document, 'hidden', {
        configurable: true,
        get: () => hidden
      });
      window.fetch = async (path, options) => {
        const target = new URL(path, url);
        if (target.pathname === '/api/session' && failSession) {
          throw Object.assign(new Error('Failed to fetch'), {name: 'TypeError'});
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
    throw Error('Timed out waiting for ' + what +
      ' unreachable=' + w.chatbotUnreachable + ' fails=' + w.sessionFailCount);
  }

  await until(() => w.session, 'session');
  failSession = true;
  w.sessionFailCount = 0;
  w.chatbotUnreachable = false;
  await w.checkSession();
  await w.checkSession();
  await w.checkSession();
  await until(() => w.chatbotUnreachable, 'unreachable state');
  assert.ok(d.getElementById('healthBanner').classList.contains('visible'));
  assert.ok(d.getElementById('healthText').textContent.includes('chatbot not reachable'));
  assert.ok(d.getElementById('statusPillText').textContent.includes('chatbot not reachable'));

  failSession = false;
  await w.checkSession();
  await until(() => !w.chatbotUnreachable, 'recovery');
  assert.equal(d.getElementById('clearConvsBtn'), null);
  assert.ok(d.getElementById('archivedConversationsBtn'));

  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
