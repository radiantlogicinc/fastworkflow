/* Recorded command outcomes on the comparison screen, in a real DOM
 * (`fix-9eg.3.1.3`).
 *
 * The evidence behind this run is two real attempts of one task in one store:
 *
 *   attempt 1  five dispatches over two turns -- `add_item` twice (the second
 *              recorded as failed), a `list_items` nothing recorded an outcome
 *              for, an inner hop the trace never gave a span of its own, and a
 *              `complete_item` in the second turn.
 *   attempt 2  two quiet dispatches.
 *
 * What a person must be able to read off the page, and what this checks:
 *
 *   - the failure belongs to the dispatch that failed, not to every command in
 *     the turn it happened in;
 *   - a dispatch nothing recorded an outcome for reads as not recorded, never
 *     as a success;
 *   - opening a group shows the exact dispatches counted, and opening one of
 *     them lands on the recorded evidence for THAT dispatch;
 *   - a run whose every turn was readable still says which dispatches were
 *     only partly recorded;
 *   - the two sides are two tallies, never one pooled figure. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const experiment = process.argv[4], task = process.argv[5];
const firstTurn = process.argv[6];
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
      d.getElementById('detail').textContent.slice(0, 1500));
  }
  const detail = () => d.getElementById('detail').textContent;
  const select = label => [...d.querySelectorAll('#detail select')]
    .find(node => node.getAttribute('aria-label') === label);
  /* Each control repaints asynchronously and the old pair stays on screen
     until the new one arrives, so a node found a moment ago may already be
     detached. Re-issued against a freshly found control. */
  async function changeUntil(label, value, ready, what) {
    for (let attempt = 0; attempt < 12; attempt++) {
      const node = select(label);
      if (node) {
        node.value = value;
        node.dispatchEvent(new w.Event('change', {bubbles: true}));
      }
      for (let i = 0; i < 20; i++) {
        if (ready()) return;
        await new Promise(r => setTimeout(r, 50));
      }
    }
    throw Error('Timed out setting ' + label + ' to ' + value + ' for ' + what
      + '. detail=' + detail().slice(0, 1500));
  }
  const pane = which => d.querySelector('#detail [data-command-summary="' + which + '"]');
  const paneText = which => { const node = pane(which); return node ? node.textContent : ''; };

  d.getElementById('modeDebug').click();
  await until(() => w.session, 'the session');

  async function openCompare() {
    for (let attempt = 0; attempt < 8; attempt++) {
      w.taskView = 'compare';
      w.showExperimentTask(experiment, task, 'Command outcomes');
      for (let i = 0; i < 20; i++) {
        if (select('View')) return;
        await new Promise(r => setTimeout(r, 50));
      }
    }
    throw Error('the compare view never opened. detail=' + detail().slice(0, 1500));
  }

  await openCompare();
  await until(() => select('Left run') && select('Right run'), 'the pair picker');
  await changeUntil('Left run', '1',
    () => detail().includes('Left: attempt 1'), 'the left side');
  await changeUntil('Right run', '2',
    () => detail().includes('Right: attempt 2'), 'the right side');
  await until(() => pane('left') && pane('right'), 'both command summaries');

  /* ================================================================
   * The headline counts, per side and never pooled
   * ================================================================ */
  assert.ok(paneText('left').includes('5 dispatches'),
    'the left side counts its five recorded dispatches: '
    + paneText('left').slice(0, 300));
  assert.ok(paneText('left').includes('2 succeeded, 1 failed, 2 not recorded'),
    'and splits them three ways, with unknown as its own answer: '
    + paneText('left').slice(0, 300));
  assert.ok(paneText('right').includes('2 dispatches'),
    'the right side counts its own, separately: ' + paneText('right').slice(0, 300));
  assert.ok(!detail().includes('7 dispatches'),
    'the two sides are never added together anywhere on the screen');

  /* ================================================================
   * A failure belongs to the dispatch that failed
   * ================================================================ */
  pane('left').open = true;
  const rowOf = (which, name) =>
    pane(which).querySelector('tr[data-command-group="' + name + '"]');
  const cells = row => [...row.querySelectorAll('td')].map(cell => cell.textContent);

  const addItem = await until(() => rowOf('left', 'add_item'), 'the add_item row');
  assert.deepEqual(cells(addItem).slice(0, 5), ['add_item', '2', '1', '1', '0'],
    'two dispatches of add_item, one of each outcome: ' + cells(addItem).join(' | '));

  /* ================================================================
   * Recorded timing, inclusive and never summed [fix-9eg.3.1.4]
   * ================================================================ */
  assert.deepEqual(cells(addItem).slice(6), ['1 ms / 4 ms / 7 ms', '2 timed'],
    'the two recorded durations, as order statistics of what was timed: '
    + cells(addItem).join(' | '));
  const innerTiming = cells(rowOf('left', 'inner_hop')).slice(6);
  assert.deepEqual(innerTiming, ['not timed', '0 timed, 1 not'],
    'a dispatch nothing timed is not timed at zero: ' + innerTiming.join(' | '));
  assert.ok(paneText('left').includes('INCLUSIVE'),
    'the page says what the durations mean: ' + paneText('left').slice(0, 700));
  assert.ok(paneText('left').includes('these are not added up'),
    'and refuses the elapsed-time reading: ' + paneText('left').slice(0, 700));
  assert.ok(!paneText('left').includes('total'),
    'no total is offered anywhere on the side: ' + paneText('left').slice(0, 700));
  const completeItem = rowOf('left', 'complete_item');
  assert.deepEqual(cells(completeItem).slice(0, 5),
    ['complete_item', '1', '1', '0', '0'],
    'the other command in that run did not inherit the failure: '
    + cells(completeItem).join(' | '));
  const listItems = rowOf('left', 'list_items');
  assert.deepEqual(cells(listItems).slice(0, 5), ['list_items', '1', '0', '0', '1'],
    'and a dispatch nothing recorded an outcome for is not a success: '
    + cells(listItems).join(' | '));

  /* ================================================================
   * Opening a group shows the exact dispatches it counted
   * ================================================================ */
  addItem.click();
  const contributors = await until(() => {
    const rows = [...pane('left').querySelectorAll('[data-command-contributor]')];
    return rows.length ? rows : null;
  }, 'the contributing dispatches');
  assert.equal(contributors.length, 2,
    'exactly the two dispatches the row counted, not the turn\'s other calls');
  const contributorText = contributors.map(node => node.textContent).join(' || ');
  assert.ok(contributorText.includes('recorded success')
    && contributorText.includes('recorded failure'),
    'each dispatch carries its own outcome: ' + contributorText.slice(0, 400));
  assert.ok(contributorText.includes('1 ms inclusive')
    && contributorText.includes('7 ms inclusive'),
    'and its own recorded duration, so the row above is checkable: '
    + contributorText.slice(0, 400));

  /* The inner hop the trace never held opens the turn and says so, rather than
     presenting a turn-scoped landing as the dispatch itself. */
  const innerRow = rowOf('left', 'inner_hop');
  assert.ok(innerRow, 'the span-less inner hop is still listed');
  innerRow.click();
  const inner = await until(() => {
    const rows = [...pane('left').querySelectorAll('[data-command-contributor]')]
      .filter(node => node.textContent.includes('inner-1'));
    return rows.length ? rows[0] : null;
  }, 'the inner hop\'s contributor');
  assert.ok(inner.textContent.includes('no span was recorded for this dispatch'),
    'its coverage is stated: ' + inner.textContent.slice(0, 300));
  const innerButton = inner.querySelector('button');
  assert.equal(innerButton.textContent, 'Open the recorded turn',
    'and the link says what it can actually open');

  /* ================================================================
   * Coverage: read whole, and still only partly recorded
   * ================================================================ */
  assert.ok(!paneText('left').includes('could not be read'),
    'every turn of this run was readable, and the page does not say otherwise');
  assert.ok(paneText('left').includes('Partially recorded'),
    'reading every turn is not the same claim as recording every dispatch: '
    + paneText('left').slice(0, 500));
  assert.ok(paneText('left').includes('recorded no span'),
    'the gap is named rather than left to be inferred: '
    + paneText('left').slice(0, 500));
  assert.ok(paneText('left').includes('recorded no outcome'),
    'including the dispatches whose outcome is unknown: '
    + paneText('left').slice(0, 500));

  /* ================================================================
   * A dispatch with a span opens that dispatch's evidence
   * ================================================================ */
  const dispatchRow = contributors
    .find(node => node.querySelector('button').textContent === 'Open this dispatch');
  assert.ok(dispatchRow,
    'a dispatch the trace recorded offers its own evidence, not just its turn');
  const dispatchNote = dispatchRow.querySelector('div.sub');
  dispatchRow.querySelector('button').click();
  await until(() => w.state.turn && w.state.turn.turn_key === firstTurn,
    'the dispatch\'s recorded turn');
  /* And it is the DISPATCH that opened, not merely the turn it happened in.
     `openPairSpan` says so when the store does not hold the span, so the
     absence of that sentence is the evidence the exact call was focused. */
  assert.ok(!detail().includes('holds no span'),
    'the exact recorded span was focused: ' + detail().slice(0, 400));
  assert.equal(dispatchNote.textContent, '',
    'and the link reported no failure to find it');

  assert.deepEqual(errors, [], 'page errors: ' + errors.join(' | '));
  process.stdout.write('command summary DOM checks passed\n');
  /* Exited rather than closed: an in-flight page fetch resolving after
     teardown fails inside the page's own callback, which is a harness
     artefact and not a finding. */
  process.exit(0);
})().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exit(1); });
