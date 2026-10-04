/* The training section's source boundary, in a real DOM against a real
 * workspace server holding two archives (fix-9eg.2).
 *
 * The two archives deliberately hold the SAME run_id under different
 * published versions, which is the situation the guards exist for: a
 * response that belongs to the store the reader just left is not merely
 * late, it is a different training run wearing the same name. Nothing here
 * is stubbed -- every response is the server's own; the harness only delays
 * when the request for one store is issued, so it lands after the reader has
 * already moved on.
 *
 * What must NOT happen: a list from the abandoned store repainting the rows,
 * a run id being resolved against a store it was not listed from, or a
 * detail from the abandoned store surviving in the pane. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], ids = JSON.parse(process.argv[4]);
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

/* Requests whose URL contains one of these patterns are held back before
   being sent. The response itself is the server's, unaltered. */
const delays = [];
const slow = (pattern, ms) => { delays.push({pattern, ms}); };

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = async (path, options) => {
        const target = new URL(path, url);
        const rule = delays.find(r => target.href.includes(r.pattern));
        if (rule) await new Promise(r => setTimeout(r, rule.ms));
        return fetch(target, options);
      };
    }});
  const w = dom.window, d = w.document;
  const HOLD = 700;

  async function until(fn, what) {
    for (let i = 0; i < 300; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 25));
    }
    throw Error('Timed out waiting for ' + what + '. store=' +
      w.trainingHistory.storeId + ' list=' +
      d.getElementById('trainingRunList').textContent.slice(0, 400) +
      ' detail=' + d.getElementById('trainingRunDetail').textContent.slice(0, 400));
  }
  const rows = () => [...d.querySelectorAll('#trainingRunList .listItem')];
  const listText = () => d.getElementById('trainingRunList').textContent;
  const detail = () => d.getElementById('trainingRunDetail').textContent;
  /* The run id is the same string in both archives -- that is the point of
     the fixture -- and it happens to contain one archive's version id, so it
     is masked out before looking for a version. What is left names a store. */
  const section = () =>
    (listText() + ' ' + detail()).split(ids.run_id).join('<run>');
  const picker = () => d.getElementById('trainingStore');

  /* Watch the section for as long as the held-back request could still land,
     and report anything from the abandoned store that appeared even for one
     frame. Reading only the final state would pass a page that painted the
     wrong run and then corrected itself. */
  async function neverShows(needle, ms, what) {
    const seen = [];
    for (let i = 0; i * 25 < ms; i++) {
      if (section().includes(needle)) { seen.push(section().slice(0, 300)); }
      await new Promise(r => setTimeout(r, 25));
    }
    assert.deepEqual(seen, [], what);
  }

  const chooseStore = value => {
    picker().value = value;
    picker().dispatchEvent(new w.Event('change'));
  };

  await until(() => w.session && w.session.workspace_mode, 'a workspace session');

  /* ================================================================
   * A list from the store the reader left does not repaint the rows
   * ================================================================ */
  slow('store_id=left', HOLD);
  slow('/training-run/left/', HOLD);

  d.getElementById('trainingHistoryBtn').click();
  await until(() => picker().options.length === 2, 'both archives offered');
  assert.equal(w.trainingHistory.storeId, 'left',
    'the section opens on the first store in the manifest');

  // While that store's list is still held back, the reader picks the other.
  chooseStore('right');
  await until(() => detail().includes(ids.version_b), "the chosen store's run");
  assert.deepEqual(rows().map(node => node.querySelector('.title').textContent),
    [ids.version_b], 'the rows are the chosen store’s');

  await neverShows(ids.version_a, HOLD + 500,
    'the abandoned store’s list never repaints the section');
  assert.equal(w.trainingHistory.storeId, 'right');
  assert.equal(rows().length, 1,
    'and its rows were not appended to the ones on screen: ' + listText().slice(0, 300));

  /* ================================================================
   * A run id is resolved against the store it was listed from
   * ================================================================
   * Both archives hold `RUN_A`. If the detail URL were rebuilt when the
   * response landed, or the id carried across the switch, this would show
   * the other archive's training run under the same id and nothing would
   * look wrong. */
  assert.ok(detail().includes(ids.run_id),
    'the run id is shown: ' + detail().slice(0, 300));

  chooseStore('left');              // list and detail both held back
  await until(() => w.trainingHistory.storeId === 'left', 'the switch back');
  chooseStore('right');             // reader moves on before either lands
  await until(() => detail().includes(ids.version_b), 'the current store again');
  await neverShows(ids.version_a, HOLD + 500,
    'a detail issued for the other archive never lands in the pane');

  /* ================================================================
   * The page's own source boundary stands the section down
   * ================================================================ */
  slow('store_id=right', HOLD);
  slow('/training-run/right/', HOLD);
  chooseStore('right');
  await until(() => !rows().length, 'the reload to be in flight');

  w.onSourceSwitch();               // what resetSourceScopedState calls
  assert.equal(d.getElementById('trainingDialog').open, false,
    'the section closes: what it holds belongs to evidence no longer selected');
  assert.equal(w.trainingHistory.storeId, null,
    'and the remembered store goes with the source that named it');
  assert.equal(picker().options.length, 0);

  await neverShows(ids.version_b, HOLD + 500,
    'the read that was in flight across the boundary paints nothing');

  assert.deepEqual(errors, []);
  process.exit(0);
})().catch(e => {
  process.stderr.write(String((e && e.stack) || e) + '\n');
  process.exit(1);
});
