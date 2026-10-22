/* Seal, open and delete a benchmark experiment through the page (fix-ajir.24).

   Driven against a real chatbot server. Every destructive step goes through
   the page's own dialog: window.confirm throws if anything reaches for it. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], eid = process.argv[4];
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
      window.confirm = () => { throw Error('window.confirm must not be used'); };
    }});
  const w = dom.window, d = w.document;
  const detail = () => d.getElementById('detail').textContent;
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | ' + detail() + ' | errors: ' + JSON.stringify(errors));
  }
  const button = text => [...d.querySelectorAll('#detail button')].find(node => node.textContent === text);
  const dialogOpen = () => d.getElementById('confirmDialog').hasAttribute('open');

  await until(() => w.session && w.hierarchyRoot, 'the page never loaded');
  d.getElementById('sealedExperimentsBtn').click();
  await until(() => button('Seal'), 'the experiment was never offered for sealing');
  assert.ok(detail().includes('Nothing sealed yet'));

  // Cancelling the dialog seals nothing.
  button('Seal').click();
  await until(dialogOpen, 'sealing asked for no confirmation');
  assert.equal(d.getElementById('confirmTitle').textContent, 'Seal this experiment?');
  d.getElementById('cancelDelete').click();
  await until(() => !dialogOpen(), 'the dialog did not close');
  assert.ok(button('Seal'));

  button('Seal').click();
  await until(dialogOpen, 'sealing asked for no confirmation');
  d.getElementById('confirmDelete').click();
  await until(() => button('Delete sealed experiment'), 'the sealed archive was never listed');
  assert.ok(detail().includes(eid));
  assert.ok(detail().includes('sha256 '));

  // Opened like a live experiment, read from its archive.
  button('Open').click();
  await until(() => [...d.querySelectorAll('#detail .pill')].some(p => p.textContent === 'Sealed'),
    'the sealed experiment did not open');
  assert.equal(button('Seal experiment'), undefined);

  d.getElementById('sealedExperimentsBtn').click();
  await until(() => button('Delete sealed experiment'), 'the sealed list did not come back');
  button('Delete sealed experiment').click();
  await until(dialogOpen, 'deleting asked for no confirmation');
  assert.equal(d.getElementById('confirmTitle').textContent, 'Delete this sealed experiment?');
  d.getElementById('confirmDelete').click();
  await until(() => detail().includes('Nothing sealed yet') && !detail().includes(eid),
    'the sealed experiment was not deleted');

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
