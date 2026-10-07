/* The shipped page, in a real DOM, against a real server (fix-9eg.18.1/.2/.3).
 *
 * What this has to prove is behavioural, not cosmetic: that a match sitting
 * past the first scan segment is reached and shown, that an unfinished walk is
 * never rendered as "no matches", that a repeated command and a suspected loop
 * stay different things on screen, that a same-type context is not reported as
 * unchanged, and that every marker can be reached from the keyboard. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], keys = JSON.parse(process.argv[4]);
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn, what) {
    for (let i = 0; i < 400; i++) {
      if (fn()) return;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out waiting for ' + what + '. status=' +
      d.getElementById('turnFindStatus').textContent +
      ' detail=' + d.getElementById('detail').textContent.slice(0, 600));
  }
  const status = () => d.getElementById('turnFindStatus').textContent;
  const detail = () => d.getElementById('detail').textContent;
  const rows = () => [...d.querySelectorAll('#detail .listItem')];
  const chip = name => [...d.querySelectorAll('#turnFindMarkers button')]
    .find(b => b.textContent.includes(name));
  // Chips and the text box only edit the question; Find is what asks it.
  const find = () => d.getElementById('turnFindGo').click();

  // The filter vocabulary is present before any search has run, so the
  // operator can pick a filter without first searching for everything.
  await until(() => d.querySelectorAll('#turnFindMarkers button').length >= 13,
    'the marker filter chips');
  assert.ok(chip('navigated context'), 'context navigation chip');
  assert.ok(chip('repeated a command'), 'repeated-command chip');
  assert.ok(chip('suspected loop'), 'suspected-loop chip');
  assert.notEqual(chip('repeated a command'), chip('suspected loop'),
    'repetition and a suspected loop are separate filters, not one');

  // ---------------------------------------------------------------
  // A match past the first scan segment.
  //
  // The store holds 210 plain turns ahead of the navigating one, and the page
  // scans 200 per request, so the first segment CANNOT reach it. The old
  // behaviour -- filter whatever came back -- would show "no matches" here.
  // ---------------------------------------------------------------
  chip('navigated context').click();
  assert.equal(chip('navigated context').getAttribute('aria-pressed'), 'true');
  assert.equal(w.turnFind.active, false, 'toggling a chip searched without Find');
  find();
  // While the walk is unfinished the page must not answer; it must say it is
  // still looking. Asserted on the status line rather than by timing.
  await until(() => rows().length === 1, 'the navigating turn, found past the first segment');
  assert.ok(!detail().includes('No turn in this store matches'),
    'a found match must not be accompanied by an empty state');
  assert.ok(status().includes('the whole store'), 'status: ' + status());
  assert.ok(rows()[0].textContent.includes('navigated context'),
    'the row carries the marker that matched: ' + rows()[0].textContent);

  // The rail refreshes on a timer and, with no record selected -- which is
  // exactly the state a search is in -- repaints the "Explore a conversation"
  // placeholder over #detail. That would erase results the walk spent several
  // round trips reaching, and it does not reproduce reliably by waiting, so
  // the refresh is driven directly here.
  const found = rows()[0].textContent;
  await w.refreshConvs();
  await new Promise(r => setTimeout(r, 50));
  assert.equal(rows().length, 1,
    'a rail refresh must not erase the results: ' + detail().slice(0, 300));
  assert.equal(rows()[0].textContent, found);

  // A filter with no match anywhere says so only once the walk is complete,
  // and says how much it looked at.
  chip('navigated context').click();
  await until(() => chip('extraction errored'), 'facet chips after a search');
  chip('extraction errored').click();
  find();
  await until(() => detail().includes('No turn in this store matches'),
    'the empty state, after a complete scan');
  assert.ok(status().includes('the whole store'), 'status: ' + status());

  // Two chips are ANDed: the navigating turn carries "navigated context" but
  // not "extraction errored", so asking for both finds nothing, where OR
  // would have found it.
  chip('navigated context').click();
  find();
  await until(() => detail().includes('No turn in this store matches'),
    'both chips together, after a complete scan');
  chip('navigated context').click();

  chip('extraction errored').click();
  find();
  await until(() => rows().length > 0, 'results to come back');

  // ---------------------------------------------------------------
  // Facets from an unfinished walk are floors and say so.
  //
  // An unfiltered search fills its page after 25 turns, so the walk stops
  // there with 191 turns unexamined. Printing "repeated a command 0" would
  // be the false-negative bug in facet form; the count carries a "+".
  // ---------------------------------------------------------------
  assert.ok(status().includes('not the whole store'),
    'the unfiltered search stopped at a full page: ' + status());
  assert.ok(/repeated a command 0\+/.test(chip('repeated a command').textContent),
    'a partial count is written as a floor: ' + chip('repeated a command').textContent);
  const lowChip = chip('low confidence');
  assert.ok(lowChip.textContent.includes('not counted'),
    'no threshold means not counted, never zero: ' + lowChip.textContent);

  // ---------------------------------------------------------------
  // More matches than the page used to be willing to keep.
  //
  // Every one of the 216 turns matches an unfiltered search. The page fetches
  // 25 at a time and used to discard everything past the 200th WHILE STILL
  // advancing the cursor past them, so the last 16 -- among them the only turn
  // in the store that navigated context -- could not be reached from the
  // browser however long the operator kept clicking. Continuing to the end
  // here is the check: every fetched row is kept, each appears once, and the
  // 216th is on screen.
  // ---------------------------------------------------------------
  const more = () => d.getElementById('turnFindMore');
  const busy = () => d.getElementById('turnFindStop');
  assert.ok(more(), 'an unfinished walk offers an explicit continuation');
  let batches = 0;
  for (; batches < 40; batches++) {
    const button = more();
    if (!button) break;
    const before = rows().length;
    button.click();
    // The continuation control is replaced by "Stop searching" while the
    // request is in flight, so settle on that rather than on the button
    // vanishing, which happens immediately and means nothing.
    await until(() => !busy() && (rows().length > before || !more()),
      'the next batch of results (batch ' + batches + ')');
  }
  assert.equal(more(), null, 'the walk finished');
  assert.ok(batches > 1, 'it took several deliberate continuations: ' + batches);
  assert.ok(status().includes('the whole store'), 'status: ' + status());

  const texts = rows().map(r => r.textContent);
  assert.ok(texts.length > 200,
    'the store has more matches than the old 200-row cap: ' + texts.length);
  assert.ok(status().includes(texts.length + ' matching turn'),
    'the count on screen is the number of rows on screen: ' + status());
  assert.ok(detail().includes('Showing all ' + texts.length + ' matching turns'),
    'the page says the list is complete, now that it is');

  // The 216th match is the navigating turn, and it is here exactly once.
  const navRows = texts.filter(t => t.includes('navigated context'));
  assert.equal(navRows.length, 1,
    'the late match appears once, not zero times and not twice');
  assert.ok(texts[texts.length - 1].includes('navigated context'),
    'and it is the last row, i.e. genuinely past the old cap');

  // ---------------------------------------------------------------
  // Text search reaches the whole store too, and the row opens from the
  // keyboard.
  // ---------------------------------------------------------------
  const box = d.getElementById('turnFindText');
  box.value = keys.repeated;
  box.dispatchEvent(new w.Event('input', {bubbles: true}));
  find();
  await until(() => rows().length === 1 && status().includes('the whole store'),
    'a completed search for the repeated turn');
  // Now the walk finished, so the counts are the answer and carry no "+".
  assert.ok(/repeated a command 1$/.test(chip('repeated a command').textContent.trim()),
    'the repetition is counted: ' + chip('repeated a command').textContent);
  assert.ok(/suspected loop 0$/.test(chip('suspected loop').textContent.trim()),
    'repetition alone is not a loop: ' + chip('suspected loop').textContent);

  box.value = keys.trouble;
  box.dispatchEvent(new w.Event('input', {bubbles: true}));
  find();
  await until(() => rows().length === 1 && rows()[0].textContent.includes('ambiguous intent'),
    'the intent-trouble turn by text');
  const row = rows()[0];
  assert.equal(row.getAttribute('role'), 'button');
  assert.equal(row.tabIndex, 0);
  row.focus();
  row.dispatchEvent(new w.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));

  // ---------------------------------------------------------------
  // The opened turn: the summary, the marked ledger row, and a jump into the
  // exact span the marker came from.
  // ---------------------------------------------------------------
  await until(() => detail().includes('What was recorded'), 'the diagnosis card');
  assert.ok(detail().includes('ambiguous intent'), detail().slice(0, 400));
  assert.ok(detail().includes('parameters invalid'), detail().slice(0, 400));
  const marked = [...d.querySelectorAll('#detail tr.markedStep')];
  assert.equal(marked.length, 1, 'the dispatch that carried the trouble is the marked row');
  assert.ok(marked[0].textContent.includes('ambiguous intent'),
    'the marker rides on the ledger row: ' + marked[0].textContent);
  // A direct-command turn has no Execution stage, so the turn page keeps the
  // record: summary above Feedback, the ledger folded below the waterfall.
  const order = [...d.getElementById('detail').children].map(node =>
    node.matches('.feedbackCard') ? 'Feedback'
      : node.matches('details.ledgerDisclosure') ? 'ledger'
      : (node.querySelector(':scope > h2') || {}).textContent);
  const at = label => order.findIndex(text => text && text.startsWith(label));
  assert.ok(at('What was recorded') >= 0, order.join(' | '));
  assert.ok(at('What was recorded') < at('Feedback'), order.join(' | '));
  assert.ok(at('Feedback') < at('Inside this turn'), order.join(' | '));
  assert.ok(at('Inside this turn') < at('ledger'), order.join(' | '));
  assert.ok(at('ledger') < at('Artifacts'), order.join(' | '));
  assert.equal(d.querySelector('#detail details.ledgerDisclosure').open, false,
    'the ledger starts folded');

  // Every marker links the spans it was computed from; clicking one moves the
  // trace viewer to that span rather than merely scrolling.
  const jumps = [...d.querySelectorAll('#detail .diagJump')];
  assert.ok(jumps.length >= 2, 'a jump per supporting span, got ' + jumps.length);
  const crumbsBefore = d.querySelector('#detail nav.crumbs').textContent;
  jumps[0].click();
  await until(() => d.querySelector('#detail nav.crumbs').textContent !== crumbsBefore,
    'the trace viewer to move to the linked span');

  // ---------------------------------------------------------------
  // Same context type before and after: unknown, never "unchanged".
  // ---------------------------------------------------------------
  box.value = keys.same_type;
  box.dispatchEvent(new w.Event('input', {bubbles: true}));
  find();
  await until(() => rows().length === 1, 'the same-type turn');
  rows()[0].click();
  await until(() => detail().includes('Execution ledger'), 'that turn to open');
  assert.ok(detail().includes('context unknown'), detail().slice(0, 800));
  assert.ok(detail().includes('same context type before and after'),
    'the reason is shown, not just the verdict');
  assert.ok(!detail().includes('context unchanged'),
    'two equal context types must never be reported as unchanged');

  // A recorded type change reads as a move, with both ends and the reason.
  box.value = keys.nav;
  box.dispatchEvent(new w.Event('input', {bubbles: true}));
  find();
  await until(() => rows().length === 1, 'the navigating turn');
  rows()[0].click();
  await until(() => detail().includes('Execution ledger'), 'that turn to open');
  assert.ok(detail().includes('context changed (Workspace \u2192 Project)'),
    detail().slice(0, 800));
  assert.ok(detail().includes('the recorded context types differ'),
    'the basis is shown, not just the verdict');

  // ---------------------------------------------------------------
  // Repetition is reported as repetition, with the policy that would have
  // made it a loop stated beside it.
  // ---------------------------------------------------------------
  box.value = keys.repeated;
  box.dispatchEvent(new w.Event('input', {bubbles: true}));
  find();
  await until(() => rows().length === 1, 'the repeated-command turn');
  rows()[0].click();
  await until(() => detail().includes('What was recorded'), 'the repeated turn to open');
  assert.ok(detail().includes('repetition only; nothing in those runs recorded trouble'),
    detail().slice(0, 900));
  assert.ok(!detail().includes('suspected loop'), 'three runs are not a loop by themselves');
  assert.ok(detail().includes('Loop policy repeated-dispatch-window/1'),
    'the bound that was applied is visible: ' + detail().slice(0, 900));

  // ---------------------------------------------------------------
  // A dispatch the record knows and the trace does not stays on screen.
  // ---------------------------------------------------------------
  box.value = keys.record_only;
  box.dispatchEvent(new w.Event('input', {bubbles: true}));
  find();
  await until(() => rows().length === 1, 'the record-only failure');
  rows()[0].click();
  await until(() => detail().includes('Execution ledger'), 'the record-only turn to open');
  assert.ok(detail().includes('command returned failure'), detail().slice(0, 900));
  assert.ok(detail().includes('recorded only in the turn record'),
    'a span-less dispatch is listed and explained: ' + detail().slice(0, 900));

  assert.deepEqual(errors, []);
  process.exit(0);
})().catch(e => { console_log(e); });

function console_log(e) {
  process.stderr.write(String((e && e.stack) || e) + '\n');
  process.exit(1);
}
