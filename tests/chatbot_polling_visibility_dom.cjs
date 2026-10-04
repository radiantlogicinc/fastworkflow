/* Picker polling stops when leaving picker mode; interval polls pause while hidden.
 */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const paths = [];
  let hidden = false;
  const realFetch = fetch;
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      Object.defineProperty(window.document, 'hidden', {
        configurable: true,
        get: () => hidden
      });
      window.fetch = async (path, options) => {
        const target = new URL(path, url);
        paths.push(target.pathname);
        return realFetch(target, options);
      };
    }});
  const w = dom.window, d = w.document;
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) {
      if (fn()) return;
      await new Promise(r => setTimeout(r, 25));
    }
    throw Error('Timed out waiting for ' + what);
  }

  await until(() => w.session, 'session');
  hidden = false;

  /* Drive the poll helpers directly with a training set so the interval is
     guaranteed to start without depending on the candidate-list payload. */
  w.setTopMode('picker');
  w.pickerTrainingPaths = {'/tmp/demo_wf': true};
  w.startPickerPolling();
  assert.ok(w.pickerPollTimer, 'startPickerPolling arms the interval');
  const beforeLeave = paths.filter(p => p === '/api/workflows').length;
  /* Force one interval tick by calling the poller, then leave picker mode. */
  await w.refreshCandidateList();
  w.setTopMode('debug');
  assert.equal(w.pickerPollTimer, null, 'leaving picker clears the poll timer');
  const afterModeSwitch = paths.filter(p => p === '/api/workflows').length;
  await new Promise(r => setTimeout(r, 2800));
  const afterLeave = paths.filter(p => p === '/api/workflows').length;
  assert.equal(afterLeave, afterModeSwitch, 'no /api/workflows polls after leaving picker');
  assert.ok(afterModeSwitch >= beforeLeave);

  w.setTopMode('picker');
  w.pickerTrainingPaths = {'/tmp/demo_wf': true};
  w.startPickerPolling();
  assert.ok(w.pickerPollTimer);
  w.startSessionPolling();
  assert.ok(w.sessionPollTimer);
  const sessionBefore = paths.filter(p => p === '/api/session').length;
  hidden = true;
  d.dispatchEvent(new w.Event('visibilitychange'));
  assert.equal(w.pickerPollTimer, null);
  assert.equal(w.sessionPollTimer, null);
  await new Promise(r => setTimeout(r, 5200));
  const sessionWhileHidden = paths.filter(p => p === '/api/session').length;
  assert.equal(sessionWhileHidden, sessionBefore, 'hidden tab must not poll /api/session');

  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
