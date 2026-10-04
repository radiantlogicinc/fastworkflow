/* Summarizing the runs a reader ticked, in a real DOM (`fix-9eg.3.2.1`).
 *
 * The evidence behind this run is one task recorded four different ways:
 *
 *   attempt 1  three `cmd_a` dispatches, 10 / 20 / 30 microseconds
 *   attempt 2  one `cmd_a` at 100 microseconds and a failed `cmd_b`
 *   attempt 3  still running
 *   attempt 4  finished having recorded nothing at all
 *
 * Four phases, chosen by argv[6], because three of them are about WHEN an
 * answer arrives and each needs the page in a known state first:
 *
 *   basic    the ticks, the pooled median, the run that recorded nothing, and
 *            a drill-down landing on the run that recorded the dispatch
 *   discard  a summary that answers after the reader ticked something else
 *   race     two drill-downs in flight, the earlier one answering first
 *   stale    evidence edited under the page between summary and drill-down
 *
 * Every request is real and so is every answer; the harness only holds some
 * of them back, which is the one thing a local server makes too fast to
 * observe. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const experiment = process.argv[4], task = process.argv[5];
const phase = process.argv[6] || 'basic';
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

/* Delays applied to the next matching requests, in order. */
const summaryDelays = [];
const validationDelays = [];

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => {
        const target = new URL(path, url);
        const answer = fetch(target, options);
        const text = String(target);
        let delay = 0;
        if (text.includes('/selected-runs/validation')) {
          delay = validationDelays.shift() || 0;
        } else if (text.includes('/selected-runs?')) {
          delay = summaryDelays.shift() || 0;
        }
        if (!delay) return answer;
        return answer.then(r => new Promise(done => setTimeout(() => done(r), delay)));
      };
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
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const detail = () => d.getElementById('detail').textContent;
  const panel = () => d.querySelector('#detail [data-selected-runs]');
  const panelText = () => { const node = panel(); return node ? node.textContent : ''; };
  const body = () => d.querySelector('#detail [data-selected-runs-body]');
  const bodyText = () => { const node = body(); return node ? node.textContent : ''; };
  const submit = () => d.querySelector('#detail [data-selected-runs-submit]');
  const tick = attempt =>
    d.querySelector('#detail input[data-run-select="' + attempt + '"]');

  d.getElementById('modeDebug').click();
  /* Both, before navigating: the page bootstraps its session and its
     hierarchy independently, and a navigation issued between them is
     overwritten by whichever finishes second. */
  await until(() => w.session && w.hierarchyRoot, 'the session and the hierarchy');

  async function openRuns() {
    for (let attempt = 0; attempt < 8; attempt++) {
      w.taskView = 'runs';
      w.showExperimentTask(experiment, task, 'Selected runs');
      for (let i = 0; i < 20; i++) {
        if (tick(1) && panel()) return;
        await wait(50);
      }
    }
    throw Error('the runs view never offered a selection. detail='
      + detail().slice(0, 1500));
  }
  const check = async attempt => {
    const box = await until(() => tick(attempt), 'the tick for attempt ' + attempt);
    box.checked = true;
    box.dispatchEvent(new w.Event('change', {bubbles: true}));
  };
  const summarize = async what => {
    await until(() => submit() && !submit().disabled, 'the summarize button');
    submit().click();
    await until(() => bodyText().includes('run(s) asked for'), what);
  };
  const compareButton = attempt => {
    const member = body().querySelector('[data-selected-member="' + attempt + '"]');
    return member && [...member.querySelectorAll('button')]
      .find(node => node.textContent.indexOf('Compare') === 0);
  };

  await openRuns();

  if (phase === 'basic') {
    /* ==============================================================
     * Nothing is summarized until somebody asks
     * ============================================================== */
    assert.ok(panelText().includes('No run is selected yet.'),
      'the panel opens with nothing selected: ' + panelText().slice(0, 300));
    assert.ok(submit().disabled,
      'and nothing to summarize, so the button is not offered as live');
    assert.equal(bodyText(), '', 'no figures are on screen before anybody asks');
    [1, 2, 4].forEach(attempt => {
      assert.ok(tick(attempt) && !tick(attempt).checked,
        'attempt ' + attempt + ' is selectable and starts unticked');
    });

    /* ==============================================================
     * A run still going is visibly ineligible, not quietly absent
     * ============================================================== */
    const unfinished = tick(3);
    assert.ok(unfinished, 'the unfinished attempt is still listed');
    assert.ok(unfinished.disabled,
      'and cannot be ticked, because it has not produced what a summary is '
      + 'about');
    const unfinishedRow = d.querySelector('#detail [data-attempt="3"]');
    assert.ok(unfinishedRow.textContent.includes('Not finished'),
      'with the reason on the row: ' + unfinishedRow.textContent.slice(0, 300));

    /* ==============================================================
     * The pooled median is over the dispatches, not over the runs
     * ============================================================== */
    await check(1);
    await check(2);
    assert.ok(panelText().includes('2 run(s) selected'),
      'the count follows the ticks: ' + panelText().slice(0, 300));
    await summarize('the pooled summary');
    assert.ok(bodyText().includes('2 run(s) asked for · 2 in these counts'),
      'the population is stated as counts, never as a rate: '
      + bodyText().slice(0, 400));
    const rowOf = name =>
      body().querySelector('tr[data-command-group="' + name + '"]');
    const cells = row =>
      [...row.querySelectorAll('td')].map(cell => cell.textContent);
    const pooled = await until(() => rowOf('cmd_a'), 'the pooled cmd_a row');
    assert.deepEqual(cells(pooled).slice(0, 5), ['cmd_a', '4', '4', '0', '0'],
      'all four dispatches of cmd_a, from both runs: '
      + cells(pooled).join(' | '));
    assert.deepEqual(cells(pooled).slice(6),
      ['0.01 ms / 0.03 ms / 0.10 ms', '4 timed'],
      'the median is the median of 10/20/30/100, which is 25 microseconds — '
      + 'not the 60 an average of the two runs\' medians would give: '
      + cells(pooled).join(' | '));
    assert.ok(bodyText().includes(
        '5 dispatches · 2 commands · 4 succeeded, 1 failed, 0 not recorded'),
      'the headline counts the pooled dispatches of both runs: '
      + bodyText().slice(0, 300));
    assert.ok(bodyText().includes('1 run(s) contain a failed dispatch'),
      'a failed dispatch is kept apart from a failed run: '
      + bodyText().slice(0, 600));

    /* ==============================================================
     * The panel is not one run and not one side
     * ============================================================== */
    assert.ok(!panelText().includes('One run, on its own'),
      'the single-run wording is not reused over a pooled population');
    assert.ok(!panelText().includes('on this side'),
      'and neither is the pair wording: ' + panelText().slice(0, 400));
    assert.ok(panelText().includes('2 run(s) pooled'),
      'it says what it is instead: ' + panelText().slice(0, 400));

    /* ==============================================================
     * Opening a contributor lands on the run that recorded it
     * ============================================================== */
    pooled.click();
    const contributors = await until(() => {
      const rows = [...body().querySelectorAll('[data-command-contributor]')];
      return rows.length === 4 ? rows : null;
    }, 'the four contributing dispatches');
    const fromSecondRun = contributors
      .find(node => node.textContent.includes('sel-a2-t1'));
    assert.ok(fromSecondRun,
      'the second run\'s dispatch is listed by its own recorded call id');
    fromSecondRun.querySelector('button').click();
    await until(() => w.state.turn && w.state.turn.turn_key === 'sel-a2-t1',
      'the turn the second run recorded');

    /* ==============================================================
     * A run that recorded nothing is in the population and in no count
     * ============================================================== */
    await openRuns();
    await summarize('the summary again');
    await check(4);
    assert.equal(bodyText(), '',
      'ticking another run drops the answer rather than re-labelling it with '
      + 'a population it was not computed over');
    await summarize('the summary including the empty run');
    assert.ok(bodyText().includes('3 run(s) asked for · 3 in these counts'),
      'the empty run is a member: ' + bodyText().slice(0, 400));
    assert.ok(bodyText().includes('2 with readable evidence'),
      'and the readable ones are counted separately: '
      + bodyText().slice(0, 400));
    assert.ok(bodyText().includes('attempt 4 finished having recorded nothing'),
      'named, rather than dropped out of the denominator: '
      + bodyText().slice(0, 600));
    assert.ok(bodyText().includes('how much it would have contributed is unknown'),
      'and no missing contribution is invented for it');
  }

  if (phase === 'discard') {
    /* A summary asked for, then a different selection made while it is still
       in flight. The answer describes a population nobody is looking at any
       more, so it must be dropped rather than rendered under the new ticks. */
    await check(1);
    await check(2);
    summaryDelays.push(900);
    submit().click();
    await wait(150);
    await check(4);
    assert.equal(bodyText(), '',
      'the pending summary is not on screen after the selection changed');
    await wait(1400);
    assert.equal(bodyText(), '',
      'and the late answer for the OLD selection was discarded rather than '
      + 'shown beside the new one: ' + bodyText().slice(0, 300));
    assert.ok(panelText().includes('3 run(s) selected'),
      'the panel still describes what is actually ticked: '
      + panelText().slice(0, 300));
    /* And asking again answers the question that is now on screen. */
    await summarize('the summary of the new selection');
    assert.ok(bodyText().includes('3 run(s) asked for'),
      'the new population, computed for it: ' + bodyText().slice(0, 300));
  }

  if (phase === 'race') {
    await check(1);
    await check(2);
    await summarize('the summary');
    await until(() => compareButton(1) && compareButton(2), 'both members');
    /* A prior pass-scoped comparison of this same task, which a whole-run
       member must not inherit. */
    w.taskCompare.leftPass = 'teacher';
    w.taskCompare.rightPass = 'student';
    /* The first click's check answers FIRST, and still after the second
       click. Nothing else has navigated at that moment, so the navigation
       token alone would not stop it: only the click sequence does. */
    validationDelays.push(400, 1000);
    compareButton(1).click();
    await wait(100);
    compareButton(2).click();
    await wait(600);
    assert.notEqual(w.taskCompare.right, '1',
      'the earlier click answered first and did not navigate, because the '
      + 'reader had already clicked something else');
    await until(() => w.taskView === 'compare' && detail().includes('versus'),
      'the existing compare view');
    assert.equal(w.taskCompare.right, '2',
      'the run the reader clicked last is the one that opened');
    assert.equal(w.taskCompare.leftPass, null,
      'a whole run is compared whole, not under a pass scope left over from '
      + 'an earlier comparison');
    assert.equal(w.taskCompare.rightPass, null, 'on both sides');
    await wait(400);
    assert.equal(w.taskCompare.right, '2',
      'and nothing takes the page back afterwards: right='
      + w.taskCompare.right);
  }

  if (phase === 'stale') {
    await check(1);
    await check(2);
    await summarize('the summary');
    /* The driver edits one recorded outcome in place now -- same turn, same
       dispatch, same span, same every identifier -- and answers when it is
       done. */
    process.stdout.write('READY-FOR-MUTATION\n');
    await new Promise(resolve => process.stdin.once('data', resolve));
    const member = await until(() => compareButton(2), 'the second member');
    member.click();
    const note = await until(() => {
      const row = body().querySelector('[data-selected-member="2"]');
      const text = row ? row.textContent : '';
      return text.includes('no longer records') ? text : null;
    }, 'the refusal to open changed evidence');
    assert.ok(note.includes('Refresh the summary'),
      'and it says how to count it as it is now: ' + note.slice(0, 400));
    assert.notEqual(w.taskView, 'compare',
      'nothing was opened, so the totals on screen are not quietly standing '
      + 'beside different evidence');
    /* The run nobody touched still opens. */
    const untouched = await until(() => compareButton(1), 'the first member');
    untouched.click();
    await until(() => w.taskView === 'compare' && detail().includes('versus'),
      'the unchanged run opening normally');
  }

  assert.deepEqual(errors, [], 'page errors: ' + errors.join(' | '));
  process.stdout.write('selected runs DOM checks passed (' + phase + ')\n');
  /* Exited rather than closed: an in-flight page fetch resolving after
     teardown fails inside the page's own callback, which is a harness
     artefact and not a finding. */
  process.exit(0);
})().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exit(1); });
