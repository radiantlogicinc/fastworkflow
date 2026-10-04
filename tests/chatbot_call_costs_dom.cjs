/* The expensive recorded LLM calls of each compared side, on the shipped page,
 * in a real DOM against a real server (fix-9eg.3.1.2).
 *
 * Three parts, all over real recorded evidence read through the real API.
 *
 * The pair: the page is navigated to the task's Compare view and the real Left
 * run / Right run controls put attempt 1 beside attempt 2. Attempt 1 recorded a
 * dear call, a cheap one, one that recorded a cost of exactly 0 and one that
 * recorded nothing at all. Attempt 2's trace holds one call recorded at two
 * levels (a wrapper quoting its own descendant's response) and two sibling calls
 * quoting ONE response. The list must order the first side by recorded cost, keep
 * "spent nothing" apart from "nobody counted", show the folded call once, and
 * charge the shared response once.
 *
 * The bounded list: attempt 3 recorded twenty-one priced calls plus an unpriced
 * one and a shared one, which is the shape where truncation would hide the
 * existence of the last two. They are counted by kind and reachable through the
 * local "show all", which reads the payload already in hand.
 *
 * The links: clicking a row opens the turn that recorded that call in the side
 * that recorded it, and focuses the span the projection published for it -- by
 * its canonical span id, never a parent span or a row position.
 *
 * What must NOT happen anywhere: an absence rendered as a zero, a shared or
 * wrapped response charged twice, a side that made no LLM call reading the same
 * as a side whose calls could not be listed, or a link that opens the other
 * side's evidence. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], experiment = process.argv[4], task = process.argv[5],
      storeId = process.argv[6];
/* Two comparison payloads produced by the real `project_execution` over the
 * same temp evidence store, for references that name a turn the store does not
 * hold. Written by the pytest that drives this harness, because no live route
 * builds such a reference (fix-9eg.3.1.2.1). */
const unreadablePath = process.argv[7];
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
    for (let i = 0; i < 300; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out waiting for ' + what + '. detail=' +
      d.getElementById('detail').textContent.slice(0, 900));
  }
  const detail = () => d.getElementById('detail').textContent;
  const select = label => [...d.querySelectorAll('#detail select')]
    .find(node => node.getAttribute('aria-label') === label);
  const change = (node, value) => {
    node.value = value;
    node.dispatchEvent(new w.Event('change', {bubbles: true}));
  };

  /* This section only, found through its own heading: other panes of the same
     comparison own their own expandable blocks. */
  const sectionIn = root => {
    const head = [...root.querySelectorAll('h2')]
      .find(node => node.textContent === 'Expensive recorded LLM calls');
    if (!head) { return null; }
    let node = head.nextElementSibling;
    while (node && !(node.className || '').includes('comparePanes')) {
      node = node.nextElementSibling;
    }
    return node;
  };
  const blocksIn = root => {
    const section = sectionIn(root);
    return section ? [...section.children] : [];
  };
  const rowsIn = block => [...block.querySelectorAll('.compareRow')];
  const costsIn = block => rowsIn(block)
    .map(row => row.querySelector('.chipCost').textContent);

  d.getElementById('modeDebug').click();
  await until(() => w.session, 'the session');

  /* ================================================================
   * Attempt 1 beside attempt 2, through the real controls
   * ================================================================ */
  w.benchmarkExperimentSource = experiment;
  w.taskView = 'compare';
  /* Re-issued until it sticks: the server is opened on a real evidence DB, so
     the rail's own first load resolves asynchronously and repaints `#detail`
     after this navigation. */
  const compareReady = () => detail().includes('Answers') && select('Left run');
  for (let attempt = 0; attempt < 20 && !compareReady(); attempt++) {
    w.showExperimentTask(experiment, task, 'Call cost experiment');
    for (let i = 0; i < 20 && !compareReady(); i++) {
      await new Promise(r => setTimeout(r, 50));
    }
  }
  await until(compareReady, 'the compare view');

  const paired = () => detail().includes('Right: attempt 2')
    && detail().includes('Expensive recorded LLM calls');
  for (let attempt = 0; attempt < 15 && !paired(); attempt++) {
    const right = select('Right run');
    if (right) { change(right, '2'); }
    for (let i = 0; i < 20 && !paired(); i++) {
      await new Promise(r => setTimeout(r, 50));
    }
  }
  assert.ok(paired(), 'the pair never landed: ' + detail().slice(0, 1200));
  assert.ok(detail().includes('Left: attempt 1'),
    'the left side is still the Reference: ' + detail().slice(0, 400));

  const pane = await until(() => {
    const found = blocksIn(d.getElementById('detail'));
    return found.length === 2 ? found : null;
  }, 'one expandable block per side');
  const [left, right] = pane;

  // The answer and the artifacts still come first: this section is below them.
  const order = detail();
  assert.ok(order.indexOf('Answers') < order.indexOf('Artifacts'),
    'answers still precede artifacts');
  assert.ok(order.indexOf('Artifacts')
    < order.indexOf('Expensive recorded LLM calls'),
    'and both still precede the call costs: ' + order.slice(0, 200));

  /* -- ordering, and the three absences kept apart -------------------- */
  assert.deepEqual(costsIn(left), [
    'cost 0.7500',
    'cost 0.0020',
    'cost 0.0000 \u2014 recorded, not missing',
    'cost not recorded for this call'
  ], 'the left side is ordered by recorded cost, with a recorded zero above an '
   + 'unrecorded one and neither read as the other: ' + left.textContent);

  assert.ok(left.textContent.includes('Left: attempt 1'),
    'the block names its side and attempt: ' + left.textContent.slice(0, 200));
  assert.ok(left.textContent.includes('4 recorded LLM calls, dearest 0.7500'),
    'and summarises what it holds: ' + left.textContent.slice(0, 200));
  assert.ok(left.textContent.includes('1 with no recorded cost'),
    'the unpriced call is counted, not dropped: ' + left.textContent);

  /* -- the retained cache and coverage labels ------------------------- */
  const leftRows = rowsIn(left);
  assert.ok(leftRows[0].textContent.includes('from cache'),
    'the dearest call kept its recorded cache hit: ' + leftRows[0].textContent);
  assert.ok(leftRows[1].textContent.includes('from the provider'),
    'a recorded miss is a miss: ' + leftRows[1].textContent);
  assert.ok(leftRows[3].textContent.includes('cache state not recorded'),
    'and an absent flag is never rendered as a miss: ' + leftRows[3].textContent);
  assert.ok(leftRows[0].textContent.includes('150 tok'),
    'recorded tokens are shown: ' + leftRows[0].textContent);
  assert.ok(leftRows[2].textContent.includes('0 tok'),
    'a recorded zero prints as zero: ' + leftRows[2].textContent);
  assert.ok(leftRows[3].textContent.includes('tokens not recorded'),
    'and an unrecorded count says so: ' + leftRows[3].textContent);
  assert.ok(left.textContent.includes('tokens coverage partial'),
    'the side keeps its coverage: ' + left.textContent);
  assert.ok(left.textContent.includes('cache - 1 from cache, 2 from the provider, '
    + '1 not recorded'), 'and its cache counts: ' + left.textContent);

  /* -- source identity, per side, off the reference ------------------- */
  assert.ok(left.textContent.includes('source ' + storeId),
    'the side names the store that recorded it: ' + left.textContent);
  assert.ok(left.textContent.includes('experiment ' + experiment),
    'and the experiment it belongs to: ' + left.textContent);
  assert.ok(left.textContent.includes('no recorded pass scope'),
    'a whole-attempt side says it is not scoped to a pass: ' + left.textContent);
  assert.ok(leftRows[0].textContent.includes('left side'),
    'a row says which side it is on: ' + leftRows[0].textContent);
  assert.ok(leftRows[0].textContent.includes('span a1-dear'),
    'and which recorded call it is: ' + leftRows[0].textContent);
  assert.ok(leftRows[0].textContent.includes('cost-a1-t1'),
    'and which turn recorded it: ' + leftRows[0].textContent);

  /* -- one call recorded twice, and one response quoted twice --------- */
  assert.deepEqual(costsIn(right), [
    'cost 0.5000',
    'cost 0.2500',
    'cost counted under another call quoting the same provider response'
  ], 'the wrapper is one row charged once and the shared twin says where its '
   + 'money went: ' + right.textContent);
  assert.ok(!right.textContent.includes('a2-outer'),
    'the folded wrapper is not offered as a call of its own: ' + right.textContent);
  assert.ok(right.textContent.includes('span a2-inner'),
    'the record closest to the provider is the one anchored: ' + right.textContent);
  assert.ok(right.textContent.includes('3 recorded LLM calls, dearest 0.5000'),
    'the right side summarises what it holds: ' + right.textContent.slice(0, 200));
  assert.ok(right.textContent.includes(
    '1 quoting a response another call accounts for'),
    'and counts the shared call as a call: ' + right.textContent);
  const shared = rowsIn(right)[2];
  assert.ok(shared.textContent.includes('tokens counted under that call too'),
    'the shared twin does not report the tokens again: ' + shared.textContent);
  assert.ok(shared.textContent.includes('from cache'),
    'while keeping its own recorded cache observation: ' + shared.textContent);

  /* The right side really spent 0.7500 of the 1.5000 its four `fw.llm.call`
     records carry. The list must agree with the totals above it, so neither the
     wrapper nor the shared response may appear as money. */
  assert.ok(detail().includes('cost 0.7500'),
    'the existing total is still 0.7500: ' + detail().slice(0, 400));
  assert.ok(!detail().includes('1.5000') && !detail().includes('1.2500'),
    'and no re-charged sum appears anywhere on the pane');
  assert.ok(right.textContent.includes('nothing is charged twice here'),
    'the block says the rows are the calls the totals were counted over');

  /* Nothing here ranks the two runs against each other. */
  assert.ok(sectionIn(d.getElementById('detail')).previousElementSibling
    .textContent.includes('Costing less is not being right'),
    'the section refuses to read cheapness as quality');
  assert.ok(!detail().toLowerCase().includes('winner of this pair'),
    'and picks no winner');

  /* ================================================================
   * A pass-scoped side, and a side that made no LLM call
   * ================================================================ */
  const read = query => w.selectionRead(
    w.taskSelectionPath(experiment, task, '/comparison' + query));
  const renderInto = payload => {
    const host = d.createElement('div');
    w.renderComparisonCallCosts(host, payload, {
      experimentId: experiment, taskId: task, storeId: storeId,
      winnerId: null, comparison: payload, reload: function () {}
    });
    return host;
  };

  const passed = await read('?left_attempt=1&right_attempt=2&left_pass=teacher');
  assert.ok(!passed.unavailable, 'the pass-scoped comparison answered: '
    + JSON.stringify(passed).slice(0, 300));
  const passBlocks = blocksIn(renderInto(passed));
  assert.ok(passBlocks[0].textContent.includes('pass teacher'),
    'a pass-scoped side names its pass: ' + passBlocks[0].textContent);
  assert.deepEqual(costsIn(passBlocks[0]), ['cost 0.7500'],
    'and lists that pass\u2019s calls only: ' + passBlocks[0].textContent);
  assert.ok(passBlocks[1].textContent.includes('no recorded pass scope'),
    'while the other side keeps its own identity: ' + passBlocks[1].textContent);
  assert.ok(passBlocks[0].textContent.includes('1 recorded LLM call,'),
    'one call is singular: ' + passBlocks[0].textContent);

  const wide = await read('?left_attempt=3&right_attempt=4');
  assert.ok(!wide.unavailable, 'attempt 3 versus attempt 4 answered: '
    + JSON.stringify(wide).slice(0, 300));
  const wideBlocks = blocksIn(renderInto(wide));
  const many = wideBlocks[0], silent = wideBlocks[1];

  // A side that made no LLM call says exactly that, and nothing about missing
  // evidence.
  assert.ok(silent.textContent.includes('No LLM call is recorded on this side'),
    'a run that called no LLM says so: ' + silent.textContent);
  assert.ok(!silent.textContent.includes('missing evidence'),
    'without claiming its evidence is missing: ' + silent.textContent);
  assert.equal(rowsIn(silent).length, 0, 'and lists no call');

  /* -- bounded first, complete on request ----------------------------- */
  assert.ok(many.textContent.includes('23 recorded LLM calls, dearest 0.2100'),
    'the crowded side counts every call it holds: ' + many.textContent.slice(0, 200));
  assert.equal(rowsIn(many).length, 20, 'the first render is bounded');
  assert.ok(many.textContent.includes('Showing the 20 dearest of 23'),
    'and says so: ' + many.textContent);
  assert.ok(many.textContent.includes('Not shown: 1 priced, 1 with no recorded '
    + 'cost, 1 quoting a response another call accounts for'),
    'naming by kind what is off the end, so an unpriced call cannot hide behind '
    + 'twenty-one priced ones: ' + many.textContent);
  assert.ok(many.textContent.includes('None of them is aggregated away'),
    'and promising nothing was summarised away: ' + many.textContent);
  assert.ok(!costsIn(many).includes('cost not recorded for this call'),
    'the unpriced call is genuinely not rendered yet');

  const showAll = many.querySelector('[data-show-all-calls]');
  assert.ok(showAll, 'the side offers to show the rest: ' + many.textContent);
  showAll.dispatchEvent(new w.Event('click', {bubbles: true}));
  assert.equal(rowsIn(many).length, 23, 'every recorded call is now listed');
  const wideCosts = costsIn(many);
  assert.equal(wideCosts[0], 'cost 0.2100', 'still dearest first');
  assert.equal(wideCosts[20], 'cost 0.0100', 'the cheapest priced call follows');
  assert.equal(wideCosts[21], 'cost not recorded for this call',
    'then the unknown one');
  assert.equal(wideCosts[22],
    'cost counted under another call quoting the same provider response',
    'then the one charged elsewhere');
  assert.ok(!many.textContent.includes('Showing the 20 dearest'),
    'and the bounded notice is gone: ' + many.textContent.slice(-400));

  /* -- every anchor is a canonical published call --------------------- */
  /* Read out of the payload rather than written down here: the claim is that
     each row links by the span id the projection published for that call, so a
     row's position in the list can never decide what it opens. */
  const published = {};
  (wide.left.turns || []).forEach(turn => {
    (turn.usage.calls_detail || []).forEach(call => { published[call.span_id] = call; });
  });
  const anchors = [...many.querySelectorAll('[data-recorded-call]')]
    .map(node => node.getAttribute('data-recorded-call'));
  assert.equal(anchors.length, 23, 'every listed call is linkable');
  assert.deepEqual([...anchors].sort(), Object.keys(published).sort(),
    'the anchors are exactly the published canonical calls, no more and no less');
  const pricedAnchors = anchors.filter(id => typeof published[id].cost === 'number');
  for (let i = 1; i < pricedAnchors.length; i++) {
    assert.ok(published[pricedAnchors[i - 1]].cost >= published[pricedAnchors[i]].cost,
      'priced rows descend by the cost the server published: '
      + pricedAnchors[i - 1] + ' then ' + pricedAnchors[i]);
  }
  rowsIn(many).forEach(row => {
    const id = row.querySelector('[data-recorded-call]').getAttribute('data-recorded-call');
    const cost = published[id].cost;
    if (typeof cost === 'number') {
      assert.ok(row.textContent.includes(cost.toFixed(4)),
        'the chip quotes that call\u2019s own recorded cost: ' + row.textContent);
    }
  });

  /* The shared navigation contract other panes of this comparison call. */
  assert.equal(typeof w.openPairSpan, 'function');
  assert.equal(w.openPairSpan.length, 5,
    'openPairSpan(ctx, side, turnKey, spanId, note)');

  /* ================================================================
   * The links: the exact call, on the side that recorded it
   * ================================================================ */
  /* Both buttons are taken before either is used: opening a turn replaces
     `#detail`, and a handler holds its own side and span. */
  const leftLink = left.querySelector('[data-recorded-call="a1-dear"]');
  const rightLink = right.querySelector('[data-recorded-call="a2-second"]');
  assert.ok(leftLink && rightLink, 'each row links to its own recorded call');
  assert.equal(leftLink.getAttribute('data-recorded-call-side'), 'left');
  assert.equal(rightLink.getAttribute('data-recorded-call-side'), 'right');

  const focused = () => {
    const path = w.state.path;
    const last = path.length ? path[path.length - 1] : null;
    return last && last.span ? last.span.span_id : null;
  };

  leftLink.dispatchEvent(new w.Event('click', {bubbles: true}));
  await until(() => w.state.turnKey === 'cost-a1-t1' && focused() === 'a1-dear',
    'the left side\u2019s dearest call to open where it was recorded');
  assert.equal(w.state.turn.turn_key, 'cost-a1-t1',
    'the turn on screen is the left side\u2019s');

  rightLink.dispatchEvent(new w.Event('click', {bubbles: true}));
  await until(() => w.state.turnKey === 'cost-a2-t1' && focused() === 'a2-second',
    'the right side\u2019s shared-response call to open on the right side');
  /* The shared twin is a real call with a real span, so the link lands on IT
     rather than on the call that carries the charge. */
  assert.notEqual(focused(), 'a2-first',
    'the anchor is the row\u2019s own canonical span, not the charged one');

  /* -- a span the evidence does not hold ------------------------------ */
  /* The anchor names a call this store has no span for -- a pruned span, or a
     projection read against evidence that has since lost it. The reader must be
     able to SEE that, on the screen they are now looking at, rather than being
     left on a turn that looks like the answer. */
  const absentNote = d.createElement('div');
  w.openPairSpan(
    {experimentId: experiment, taskId: task, storeId: storeId, winnerId: null,
     comparison: passed, reload: function () {}},
    w.pairSide(passed, 'left'), 'cost-a1-t1', 'a1-never-recorded', absentNote);
  await until(() => detail().includes('holds no span a1-never-recorded'),
    'the missing span to be reported on the screen the reader is looking at');
  assert.ok(absentNote.textContent.includes('holds no span a1-never-recorded'),
    'and in the note the link owns: ' + absentNote.textContent);
  assert.ok(detail().includes('is open at its top level'),
    'saying what was opened instead: ' + detail().slice(0, 300));

  /* -- two rapid cross-source clicks: the newest one owns the page ----- */
  /* `storeId` is deliberately not this ctx's store, so both links take the
     cross-source path: the side's evidence is resolved through the experiment it
     is registered against, exactly as a winner-versus-candidate pair is. The
     probes then race, and the older click must not land. */
  const crossCtx = {
    experimentId: experiment, taskId: task, storeId: 'some-other-store',
    winnerId: null, comparison: wide, reload: function () {}
  };
  const firstNote = d.createElement('div'), secondNote = d.createElement('div');
  const leftSide = w.pairSide(wide, 'left'), rightSide = w.pairSide(wide, 'right');
  assert.equal(w.pairReadScope(crossCtx, leftSide).kind, 'other',
    'the side is read through its own experiment, not this page\u2019s store');
  w.openPairSpan(crossCtx, leftSide, 'cost-a3-t1', 'a3-00', firstNote);
  w.openPairSpan(crossCtx, rightSide, 'cost-a4-t1', 'ex-cost-a4-t1', secondNote);

  await until(() => w.state.turnKey === 'cost-a4-t1',
    'the newest click to own the navigation');
  /* Awaited rather than read the instant the newest click lands: the overtaken
     probe writes its note when its OWN slower answer comes back, which is by
     definition after the winner's. Requiring the sentence immediately would be
     a race on probe ordering rather than a claim about behaviour -- the claim
     is that the loser does say it, and that it never moves the page. */
  await until(() => firstNote.textContent.includes('The page moved on'),
    'the overtaken click to say it opened nothing. note=' + firstNote.textContent);
  /* Held for a moment: the losing probe answering later must not move the page
     off the run the reader actually asked for. */
  await new Promise(r => setTimeout(r, 400));
  assert.equal(w.state.turnKey, 'cost-a4-t1',
    'and no late answer to the older click steals the navigation');

  /* ================================================================
   * A side whose evidence could not be read, beside one that really
   * made no LLM call
   * ================================================================ */
  /* Both payloads are the genuine output of `project_execution` over the same
     temp store, for a reference that names a turn the store does not hold --
     the sealed-archive and pruned-turn case. Rendered through the shipped
     helper, so what is asserted is what the pane says, not what a reducer
     returns. The two absences sit in ONE pane here on purpose: a reader must
     be able to tell "we cannot see what this run spent" from "this run spent
     nothing", and those two sentences are two rows apart. */
  const unreadable = JSON.parse(fs.readFileSync(unreadablePath, 'utf8'));
  const summaryOf = block => block.querySelector('summary').textContent;

  const [gone, none] = blocksIn(renderInto(unreadable.absent));
  assert.equal(rowsIn(gone).length, 0, 'the unreadable side lists no call');
  /* The fold is CLOSED when the reader first sees the pane, so the summary is
     the only line showing. A count of zero there would be the page asserting
     what it could not read. */
  assert.ok(summaryOf(gone).includes(
    'no recorded LLM call can be listed, and that is missing evidence'),
    'the visible summary reports absent evidence: ' + summaryOf(gone));
  assert.ok(!summaryOf(gone).includes('0 recorded LLM call'),
    'and never heads the side with a zero: ' + summaryOf(gone));
  assert.ok(gone.textContent.includes(
    '1 turn(s) of this side could not be read'),
    'the coverage line names how much is missing: ' + gone.textContent);
  assert.ok(gone.textContent.includes(
    'missing evidence rather than a side that made no LLM call'),
    'and the empty list says which absence this is: ' + gone.textContent);
  assert.ok(!gone.textContent.includes('No LLM call is recorded on this side'),
    'the sentence for a genuine zero is nowhere on this side: ' + gone.textContent);

  /* The other side of the SAME pane is attempt 4, which dispatched a command
     and really called no LLM. It still reads as the zero it is. */
  assert.equal(rowsIn(none).length, 0);
  assert.ok(summaryOf(none).includes('0 recorded LLM calls'),
    'a real zero is still summarised as zero: ' + summaryOf(none));
  assert.ok(none.textContent.includes('No LLM call is recorded on this side'),
    'and says so: ' + none.textContent);
  assert.ok(!none.textContent.includes('missing evidence'),
    'without borrowing the other side\u2019s finding: ' + none.textContent);

  /* Half an execution is still inspectable: the readable turn's calls are
     listed in full, and the unreadable one is counted beside them rather than
     quietly shortening the list. */
  const partial = blocksIn(renderInto(unreadable.partial))[0];
  assert.equal(rowsIn(partial).length, 4,
    'the turn that could be read is listed in full: ' + partial.textContent);
  assert.ok(summaryOf(partial).includes('4 recorded LLM calls, dearest 0.7500'),
    'the summary counts what was read: ' + summaryOf(partial));
  assert.ok(partial.textContent.includes(
    '1 turn(s) of this side could not be read'),
    'and the missing turn is still reported: ' + partial.textContent);

  assert.deepEqual(errors, []);
  process.exit(0);
})().catch(e => {
  process.stderr.write(String((e && e.stack) || e) + '\n');
  process.exit(1);
});
