/* View log on the shipped page, against a real chatbot server (fix-hzux.3).
 *
 * The stopped-server banner and the picker's training-failure status both
 * offer a button. Clicking it reads /api/logs/* through the page's api()
 * helper and puts the redacted tail in the log dialog via textContent. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const ids = JSON.parse(process.argv[4]);
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (err) => {
  if (err.type !== 'css-parsing') errors.push(String(err.message || err));
});

(async () => {
  const dom = await JSDOM.fromURL(url, {
    runScripts: 'dangerously',
    virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    },
  });
  const w = dom.window;
  const d = w.document;

  async function until(fn, what) {
    for (let i = 0; i < 100; i++) {
      if (fn()) return;
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    throw new Error(
      'Timed out waiting for ' + what + '. body=' +
      d.body.textContent.slice(0, 500)
    );
  }

  await until(() => w.session && w.session.workflow_path, 'the session');
  /* jsdom loads the document as hidden, and the page skips session polls
     while hidden. The stopped-server banner is that poll. */
  Object.defineProperty(d, 'hidden', {configurable: true, get() { return false; }});
  Object.defineProperty(d, 'visibilityState', {configurable: true, get() { return 'visible'; }});
  w.tm.managed = true;
  w.tm.connected = true;
  w.deadServerAnnounced = false;
  await w.checkSession();

  const button = d.getElementById('viewServerLogBtn');
  assert.equal(button && button.tagName, 'BUTTON', 'server-stopped banner offers View log');
  assert.equal(button.type, 'button');
  assert.ok(
    d.getElementById('connText').textContent.includes(ids.server_log),
    'the banner shows the absolute server log path'
  );
  button.click();
  await until(
    () => d.getElementById('logDialogLines').textContent.includes('server-ready-line'),
    'server log lines'
  );
  let lines = d.getElementById('logDialogLines').textContent;
  assert.ok(!lines.includes('fw-dom-secret'), lines);
  assert.equal(d.getElementById('logDialogPath').textContent, ids.server_log);
  d.getElementById('logDialogClose').click();

  w.pickerTrainLogPath = ids.train_log;
  w.pickerStatus(
    'Training failed — see the train log in the workflow state directory.',
    'err',
    'train'
  );
  const trainBtn = d.getElementById('viewTrainLogBtn');
  assert.equal(trainBtn && trainBtn.tagName, 'BUTTON', 'training status offers View log');
  assert.equal(trainBtn.type, 'button');
  assert.ok(
    d.getElementById('pickerStatus').textContent.includes(ids.train_log),
    'training status shows the absolute train log path'
  );
  trainBtn.click();
  await until(
    () => d.getElementById('logDialogLines').textContent.includes('train-ready-line'),
    'train log lines'
  );
  lines = d.getElementById('logDialogLines').textContent;
  assert.ok(!lines.includes('fw-dom-train-secret'), lines);
  assert.equal(d.getElementById('logDialogPath').textContent, ids.train_log);
  if (errors.length) throw new Error(errors.join('\n'));
})().catch((err) => {
  process.stderr.write(String(err && err.stack || err) + '\n');
  process.exit(1);
});
