/* Workspace links carry the store as well as the turn. A formal review owns
   its fragment: walking the trace of a blinded row must not write the names
   of what it shows into the address bar. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const base = process.argv[3], capability = process.argv[4];

async function open(hash) {
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });
  const dom = await JSDOM.fromURL(base + hash, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, base), options);
    }});
  const w = dom.window, d = w.document, detail = d.getElementById('detail');
  return {
    w, d, detail, errors,
    hash: () => w.location.hash,
    heading: () => (detail.querySelector('.levelHead h2') || {}).textContent || '',
    async until(fn, what) {
      for (let i = 0; i < 150; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
      throw Error('Timed out waiting for ' + what + ' at ' + w.location.hash + ': '
        + detail.textContent.slice(0, 400) + ' | errors: ' + JSON.stringify(errors));
    },
    row(title) {
      const found = Array.from(detail.querySelectorAll('.wfRow'))
        .find(r => r.getAttribute('aria-label').startsWith('Inspect ' + title));
      assert.ok(found, 'no "' + title + '" row');
      return found;
    },
    async close() {
      await new Promise(r => setTimeout(r, 200));
      assert.deepEqual(errors, []);
      w.close();
    }
  };
}

(async () => {
  let p = await open('#store=sealed&turn=review-turn&span=rv-step-1');
  await p.until(() => p.heading().startsWith('Step 1'), 'Step 1 in the sealed store');
  assert.equal(p.hash(), '#store=sealed&turn=review-turn&span=rv-step-1');
  await p.close();
  p = await open('#store=sealed&turn=review-turn');
  await p.until(() => p.detail.querySelector('.wfRow'), 'the sealed turn');
  p.row('Planning').click();
  await p.until(() => p.heading().startsWith('Planning'), 'Planning');
  assert.equal(p.hash(), '#store=sealed&turn=review-turn&level=0');
  await p.close();

  const reviewLink = '#review=page-link-review&review_capability='
    + encodeURIComponent(capability) + '&row=row-1';
  p = await open(reviewLink);
  await p.until(() => p.detail.querySelector('.wfRow'), 'the reviewed trace');
  const owned = p.hash();
  assert.ok(owned.startsWith(reviewLink), 'the review pane writes its own row link: ' + owned);
  assert.equal(p.d.getElementById('copyLinkBtn').style.display, 'none',
    'no copy button during a review');
  p.row('Execution').click();
  await p.until(() => p.heading().startsWith('Execution'), 'Execution in review');
  p.row('Step 1').click();
  await p.until(() => p.heading().startsWith('Step 1'), 'Step 1 in review');
  assert.equal(p.hash(), owned, 'moving through a reviewed trace leaves the fragment alone');
  await p.close();

  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
