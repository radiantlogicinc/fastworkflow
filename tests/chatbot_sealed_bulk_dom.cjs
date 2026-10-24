/* Select several experiments and seal, release, or delete them together.

   Driven against a real chatbot server. window.confirm throws if anything
   reaches for it; the page's own dialog is the confirmation. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const first = process.argv[4], second = process.argv[5], invalid = process.argv[6];
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
    for (let i = 0; i < 400; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | ' + detail() + ' | errors: ' + JSON.stringify(errors));
  }
  const button = text => [...d.querySelectorAll('#detail button')].find(node => node.textContent === text);
  const check = label => d.querySelector(`#detail input[aria-label="${label}"]`);
  const dialogOpen = () => d.getElementById('confirmDialog').hasAttribute('open');

  await until(() => w.session && w.hierarchyRoot, 'the page never loaded');
  d.getElementById('sealedExperimentsBtn').click();
  await until(() => button('Archive selected') && button('Archive selected').disabled
    && button('Release selected') && button('Release selected').disabled,
    'no selection was offered');
  assert.ok(detail().includes(invalid));
  assert.ok(detail().includes('Nothing archived yet'));
  assert.ok(detail().includes('Nothing released'));
  assert.equal(detail().includes('Prune'), false);
  assert.equal(detail().includes('prune'), false);

  check('Select all ready to archive').click();
  await until(() => button('Archive selected') && !button('Archive selected').disabled, 'select all did not enable archiving');
  button('Archive selected').click();
  await until(dialogOpen, 'archiving the selection asked for no confirmation');
  assert.equal(d.getElementById('confirmTitle').textContent, 'Archive 2 experiments?');
  d.getElementById('cancelDelete').click();
  await until(() => !dialogOpen(), 'the dialog did not close');
  assert.ok(button('Archive'));

  button('Archive selected').click();
  await until(dialogOpen, 'archiving the selection asked for no confirmation');
  d.getElementById('confirmDelete').click();
  await until(() => detail().includes('Nothing is waiting') === false
    && detail().includes('No complete experiment is waiting')
    && detail().includes(first) && detail().includes(second)
    && [...d.querySelectorAll('#detail button')].filter(node => node.textContent === 'Delete archived experiment').length === 2,
    'the two experiments were not archived');

  check('Select all archived').click();
  await until(() => button('Delete selected') && !button('Delete selected').disabled, 'select all did not enable deletion');
  button('Delete selected').click();
  await until(dialogOpen, 'deleting the selection asked for no confirmation');
  assert.equal(d.getElementById('confirmTitle').textContent, 'Delete 2 archived experiments?');
  d.getElementById('confirmDelete').click();
  await until(() => detail().includes('Nothing archived yet')
    && !detail().includes(first) && !detail().includes(second),
    'the sealed experiments were not deleted');

  check('Select all invalid').click();
  await until(() => button('Release selected') && !button('Release selected').disabled,
    'select all did not enable release');
  button('Release selected').click();
  await until(dialogOpen, 'releasing the selection asked for no confirmation');
  assert.equal(d.getElementById('confirmTitle').textContent, 'Release 1 experiment?');
  d.getElementById('confirmDelete').click();
  await until(() => button('Delete released experiment') && detail().includes(invalid),
    'the invalid experiment was not released');

  check('Select all released').click();
  await until(() => button('Delete selected') && !button('Delete selected').disabled,
    'select all did not enable deleting released experiments');
  button('Delete selected').click();
  await until(dialogOpen, 'deleting released experiments asked for no confirmation');
  assert.equal(d.getElementById('confirmTitle').textContent, 'Delete 1 released experiment?');
  d.getElementById('confirmDelete').click();
  await until(() => detail().includes('Nothing released') && !detail().includes(invalid),
    'the released experiment was not deleted');

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
