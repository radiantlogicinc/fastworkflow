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
  await until(() => button('Archive'), 'the experiment was never offered for archiving');
  assert.ok(detail().includes('Nothing archived yet'));

  // Cancelling the dialog archives nothing.
  button('Archive').click();
  await until(dialogOpen, 'archiving asked for no confirmation');
  assert.equal(d.getElementById('confirmTitle').textContent, 'Archive this experiment?');
  d.getElementById('cancelDelete').click();
  await until(() => !dialogOpen(), 'the dialog did not close');
  assert.ok(button('Archive'));

  button('Archive').click();
  await until(dialogOpen, 'archiving asked for no confirmation');
  d.getElementById('confirmDelete').click();
  await until(() => button('Delete archived experiment'), 'the archive was never listed');
  assert.ok(detail().includes(eid));
  assert.ok(detail().includes('archived '));
  assert.equal(detail().includes('sha256 '), false);

  // Opened like a live experiment, read from its archive.
  button('Open').click();
  await until(() => [...d.querySelectorAll('#detail .pill')].some(p => p.textContent === 'Archived'),
    'the archived experiment did not open');
  assert.equal(button('Archive experiment'), undefined);

  d.getElementById('sealedExperimentsBtn').click();
  await until(() => button('Delete archived experiment'), 'the archived list did not come back');
  button('Delete archived experiment').click();
  await until(dialogOpen, 'deleting asked for no confirmation');
  assert.equal(d.getElementById('confirmTitle').textContent, 'Delete this archived experiment?');
  d.getElementById('confirmDelete').click();
  await until(() => detail().includes('Nothing archived yet') && !detail().includes(eid),
    'the archived experiment was not deleted');

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
