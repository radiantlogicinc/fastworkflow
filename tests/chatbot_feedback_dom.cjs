/* The feedback UI, driven in a real DOM.
 *
 * Two surfaces, both of which are the point of fix-9eg.16/.19.1 and neither
 * of which a Python test can see: the composer a person actually types into,
 * and the task Feedback view that has to show EVERY comment. Asserted by
 * clicking, because "the function exists" is not the claim. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], eid = process.argv[4], task = process.argv[5];
const turnKey = process.argv[6];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn) {
    for (let i = 0; i < 200; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out: ' + d.getElementById('detail').textContent.slice(0, 600));
  }
  const buttons = () => [...d.querySelectorAll('#detail button')];
  const button = text => buttons().find(node => node.textContent === text);
  const detailText = () => d.getElementById('detail').textContent;

  /* Debug mode is where the experiment views live. The task has recorded
     turns but no ad-hoc conversation, so the rail stays empty and the view
     is opened the way a deep link opens it. */
  d.getElementById('modeDebug').click();
  await until(() => w.session);

  /* ---- the task Feedback view ---------------------------------------- */
  w.showExperimentTask(eid, task);
  await until(() => button('Feedback') && button('Runs'));
  // Runs is the default view and the strip switches without leaving the task.
  assert.ok(detailText().includes('Attempts'));
  button('Feedback').click();
  await until(() => detailText().includes('Every recorded comment on this task'));

  /* No hidden default filter: every attempt, every kind, and the legacy
     unclassified row are all on screen before anybody picks anything. */
  const rows = () => [...d.querySelectorAll('#detail .listItem')];
  await until(() => rows().length >= 5);
  const text = rows().map(row => row.textContent).join('\n');
  assert.ok(text.includes('attempt 1') && text.includes('attempt 2'));
  assert.ok(text.includes('Observations / Analysis \u00b7 Observation'));
  assert.ok(text.includes('Conclusions \u00b7 What went wrong'));
  assert.ok(text.includes('Recommendations \u00b7 What to do'));
  assert.ok(text.includes('Unclassified (recorded before categories)'),
    'a pre-taxonomy comment must say so rather than borrow a category');
  assert.ok(text.includes('legacy note kept verbatim'),
    'legacy text is shown exactly as it was stored');
  // A comparison comment links BOTH recorded executions.
  const pair = rows().find(row => row.textContent.includes('Compared with:'));
  assert.ok(pair, 'the comparison comment shows both sides');
  const pairLine = [...pair.querySelectorAll('.sub')]
    .find(node => node.textContent.startsWith('Compared with:'));
  assert.equal(pairLine.querySelectorAll('.evidenceLink').length, 2,
    'one link per recorded execution the comment names');
  assert.ok(pairLine.textContent.includes('task-2'),
    'the other side names the task it lives in');
  // Plus the row's own deep link to the trace it is anchored to.
  assert.ok(pair.textContent.includes('Open the trace'));

  /* Filters narrow and can be cleared; none of them is on by default. */
  const select = label => [...d.querySelectorAll('#detail select')]
    .find(node => node.getAttribute('aria-label') === label);
  const rowText = () => rows().map(row => row.textContent).join('\n');
  const categorySelect = select('Category');
  assert.equal(categorySelect.value, '');
  const before = rows().length;
  categorySelect.value = 'recommendations';
  categorySelect.dispatchEvent(new w.Event('change'));
  await until(() => rows().length && rows().length < before);
  assert.ok(!rowText().includes('Conclusions \u00b7 What went wrong'));
  assert.ok(rowText().includes('Recommendations \u00b7 What to do'));
  categorySelect.value = '';
  categorySelect.dispatchEvent(new w.Event('change'));
  await until(() => rows().length === before);

  /* ---- the composer -------------------------------------------------- */
  w.selectTurn(turnKey);
  await until(() => button('Save feedback'));
  const tab = text => buttons().find(node => node.textContent === text);
  assert.ok(tab('Observations / Analysis') && tab('Conclusions') && tab('Recommendations'));
  const area = d.getElementById('feedback-comment');
  assert.ok(area.placeholder.startsWith('What did you see?'),
    'the watermark follows the selected subcategory');
  tab('Recommendations').click();
  await until(() => tab('What not to do'));
  tab('What not to do').click();
  assert.ok(area.placeholder.startsWith('What should be avoided?'));
  assert.equal(tab('What not to do').getAttribute('aria-selected'), 'true');
  assert.equal(tab('What to do').getAttribute('aria-selected'), 'false');

  area.value = 'Do not paper over the missing third item with a retry.';
  const save = button('Save feedback');
  // The existing notes have to arrive before the composer unlocks: a save
  // posted into a card that has not read yet would render onto a stale list.
  await until(() => !save.disabled);
  save.click();
  await until(() => detailText().includes('Do not paper over'));
  assert.ok(detailText().includes('Recommendations \u00b7 What not to do'));
  assert.equal(area.value, '', 'the composer clears so the next note is not a duplicate');

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
