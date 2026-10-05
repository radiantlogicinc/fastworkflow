/* Where the turn-wide record sits in the drill-down, through the real page
 * against a real server.
 *
 * Feedback comes before the waterfall on every level. "What was recorded" and
 * the execution ledger live on the Execution stage, the ledger folded shut
 * behind a link; a turn with no Execution stage keeps both on the turn page.
 * The rows and jumps still open their spans from the new home. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], agentTurn = process.argv[4], directTurn = process.argv[5];
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | detail=' +
      d.getElementById('detail').textContent.slice(0, 600) + ' | errors: ' + JSON.stringify(errors));
  }
  /* One label per top-level block of #detail, in page order. */
  const blocks = () => [...d.getElementById('detail').children].map(node => {
    if (node.matches('nav.crumbs')) return 'crumbs';
    if (node.matches('.feedbackCard')) return 'Feedback';
    if (node.matches('details.ledgerDisclosure')) return 'ledger';
    const h2 = node.querySelector(':scope > h2');
    return h2 ? h2.textContent : 'header';
  });
  const named = prefix => blocks().filter(label => label.startsWith(prefix)).length;
  const ledger = () => d.querySelector('#detail details.ledgerDisclosure');
  const row = label => [...d.querySelectorAll('#detail .wfRow')]
    .find(r => r.getAttribute('aria-label').startsWith('Inspect ' + label));

  /* ---- an agent turn: the record moves to its Execution stage ---------- */
  w.selectTurn(agentTurn);
  await until(() => named('Inside this turn'), 'the agent turn to open');
  assert.deepEqual(blocks(), ['crumbs', 'header', 'Feedback', 'Inside this turn',
    'Artifacts'], 'turn page order');
  assert.equal(named('Raw turn record'), 0, 'the turn page carries no raw record card');
  assert.equal(d.querySelector('#detail > .card > pre.json'), null, 'nor the record_json dump');
  assert.equal(ledger(), null, 'the ledger is not on the turn page of an agent turn');

  row('Execution').click();
  await until(() => named('Inside this stage'), 'the Execution stage');
  assert.deepEqual(blocks(), ['crumbs', 'header', 'What was recorded', 'Feedback',
    'Inside this stage', 'ledger'], 'Execution stage order');
  assert.equal(ledger().open, false, 'the ledger starts folded');
  assert.equal(ledger().querySelector(':scope > summary').textContent, 'Execution ledger');
  ledger().querySelector(':scope > summary').click();
  assert.equal(ledger().open, true, 'the summary unfolds it');

  /* A ledger row still opens its span from here, and that lower level puts
     Feedback before its waterfall without repeating the turn-wide cards. */
  const openable = ledger().querySelector('tr.openable');
  assert.ok(openable, 'a dispatch with a span is openable');
  openable.click();
  await until(() => !named('Inside this stage') && named('Inside this span'),
    'the ledger row to open its span');
  assert.ok(d.querySelector('#detail .levelHead h2').textContent.startsWith('Assistant'),
    'the opened span is the dispatch: ' + d.querySelector('#detail .levelHead h2').textContent);
  assert.deepEqual(blocks(), ['crumbs', 'header', 'Feedback', 'Inside this span'],
    'span page order');

  /* ---- a direct-command turn: no Execution stage, so the turn keeps both - */
  w.selectTurn(directTurn);
  await until(() => named('Inside this turn') && ledger(), 'the direct turn to open');
  assert.deepEqual(blocks(), ['crumbs', 'header', 'What was recorded', 'Feedback',
    'Inside this turn', 'ledger', 'Artifacts'],
    'fallback turn page order');
  assert.equal(named('Raw turn record'), 0, 'the fallback turn page has no raw record card');
  assert.equal(row('Execution'), undefined,
    'a direct-command turn has no Execution stage to move them to');
  assert.equal(ledger().open, false, 'the fallback ledger starts folded too');

  assert.deepEqual(errors, []);
  process.exit(0);
})().catch(e => {
  process.stderr.write(String((e && e.stack) || e) + '\n');
  process.exit(1);
});
