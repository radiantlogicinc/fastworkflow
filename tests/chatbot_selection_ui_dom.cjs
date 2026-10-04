/* Run selection, the winner and the comparison, driven in a real DOM.
 *
 * fix-9eg.17.2 / .17.3 / .17.4 and the browser half of fix-9eg.4.
 *
 * The claims are behavioural and none of them is visible to a Python test:
 * that every attempt of a repeated task is on screen INCLUDING the failed and
 * unfinished ones, that Reference and Best run are different badges, that a
 * failed attempt chosen as best keeps saying it failed, that choosing it does
 * not move the contest's winner, that an attempt with nothing recorded has a
 * visibly disabled comparison, that a one-sided step is labelled as one rather
 * than paired up to make the two lists the same length, that a categorized
 * comment can be written on one step of the pair and is then found under the
 * task's Feedback tab, and that picking a different best run afterwards leaves
 * it exactly where it was.
 *
 * The world is the API worker's fixture: one task run four ways in one
 * database (completed / failed / unfinished / finished-with-no-turns) and a
 * second experiment recording the same task in a SECOND database. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], ids = JSON.parse(process.argv[4]);
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
  const buttons = () => [...d.querySelectorAll('#detail button')];
  const button = text => buttons().find(node => node.textContent === text);
  const buttonIn = (root, text) =>
    [...root.querySelectorAll('button')].find(node => node.textContent === text);
  const items = () => [...d.querySelectorAll('#detail .listItem')];
  const itemFor = attempt =>
    items().find(node => node.getAttribute('data-attempt') === String(attempt));
  const select = label => [...d.querySelectorAll('#detail select')]
    .find(node => node.getAttribute('aria-label') === label);
  const change = (node, value) => {
    node.value = value;
    node.dispatchEvent(new w.Event('change', {bubbles: true}));
  };

  d.getElementById('modeDebug').click();
  await until(() => w.session, 'the session');

  /* The two evidence databases are registered against the workflow rather than
     being the server's default store, so the page is scoped the way opening a
     benchmark experiment scopes it. */
  w.benchmarkExperimentSource = ids.experiment;

  /* ================================================================
   * The Runs view: the whole record of a repeated task
   * ================================================================ */
  w.taskView = 'runs';
  w.showExperimentTask(ids.experiment, ids.task, 'Roster experiment');
  await until(() => detail().includes('Attempts') && items().length >= 4,
    'every recorded attempt');
  assert.ok(button('Runs') && button('Compare') && button('Feedback'),
    'one task, three tabs');

  // Four attempts, and the two that a "successes only" list would drop.
  assert.ok(itemFor(1) && itemFor(2) && itemFor(3) && itemFor(4),
    'attempts 1-4 are all on screen: ' + items().length);
  assert.ok(itemFor(2).textContent.includes('failed'),
    'the failed attempt says so: ' + itemFor(2).textContent);
  assert.ok(itemFor(3).textContent.includes('Not finished, so it cannot be the best run yet'),
    'an unfinished attempt is present and explained: ' + itemFor(3).textContent);

  // Reference is a viewing default; Best run is a decision, and nobody has
  // made one yet.
  assert.ok(itemFor(1).textContent.includes('Reference'), 'attempt 1 is the Reference');
  assert.ok(!detail().includes('Best run'),
    'no best run has been decided, so no badge claims one');
  assert.ok(detail().includes('is a viewing default and not a decision'),
    'the difference is stated, not implied: ' + detail().slice(0, 600));

  // An attempt that finished with nothing recorded: comparison visibly off,
  // with the evidence's own reason, and still choosable as a best run.
  const refused = buttonIn(itemFor(4), 'Compare');
  assert.ok(refused && refused.disabled, 'attempt 4 cannot be compared');
  assert.ok(refused.title, 'and the button says why: ' + refused.title);
  assert.ok(itemFor(4).textContent.toLowerCase().includes('no recorded turns'),
    'the reason is on the row too: ' + itemFor(4).textContent);
  assert.ok(buttonIn(itemFor(4), 'Use as best run'),
    'it finished, so a reviewer may still prefer it');

  /* ================================================================
   * Choosing the FAILED attempt as the best run
   * ================================================================ */
  buttonIn(itemFor(2), 'Use as best run').click();
  await until(() => button('Record this best run'), 'the best-run prompt');
  const reason = [...d.querySelectorAll('#detail input')]
    .find(node => node.getAttribute('aria-label') === 'Reason');
  assert.ok(reason, 'a reason is offered');
  assert.equal(reason.placeholder, 'Optional', 'and required by nothing');
  reason.value = 'it failed in the interesting way';
  assert.ok(detail().includes('does not touch the experiment winner'),
    'the prompt says what it is not: ' + detail().slice(0, 900));
  button('Record this best run').click();

  await until(() => itemFor(2) && itemFor(2).textContent.includes('Best run'),
    'attempt 2 to become the best run');
  // The pointer did not launder the run: it still says it failed.
  assert.ok(itemFor(2).textContent.includes('failed'),
    'the chosen attempt keeps its recorded status: ' + itemFor(2).textContent);
  assert.ok(itemFor(1).textContent.includes('Reference'),
    'and the Reference did not move with it');
  assert.ok(detail().includes('Best run: attempt 2'), detail().slice(0, 400));

  // The append-only log says what the decision was taken against.
  const historyPanel = await until(
    () => [...d.querySelectorAll('#detail details')]
      .find(node => node.textContent.includes('Best-run decisions')),
    'the decisions panel');
  historyPanel.open = true;
  historyPanel.dispatchEvent(new w.Event('toggle'));
  await until(() => historyPanel.textContent.includes('decided against'),
    'the decision log');
  assert.ok(historyPanel.textContent.includes('attempt 2'),
    'the log names the attempt: ' + historyPanel.textContent.slice(0, 400));
  assert.ok(historyPanel.textContent.includes('observability UI'),
    'and who decided it');

  /* ================================================================
   * The winner is a different decision
   * ================================================================ */
  w.showExperiment(ids.experiment);
  await until(() => detail().includes('Winner of this contest'), 'the winner panel');
  await until(() => detail().includes('automatic — first experiment'),
    'the automatic-election badge');
  assert.ok(detail().includes('chosen separately and never move it'),
    'the two decisions are separated in words as well as in storage');
  assert.ok(detail().includes('Winner'), detail().slice(0, 700));

  // Promote / Keep / Undecided, each with an optional reason and no rubric.
  assert.ok(button('Keep the current winner'), 'keep is offered');
  assert.ok(button('Leave it undecided'), 'undecided is a recordable outcome');
  const promote = button('Promote this experiment');
  assert.ok(promote.disabled, 'this experiment already holds the contest');

  /* A decision that loses a race. The candidate is promoted out of band --
     exactly what a second reviewer in another tab would do -- and then the
     button this page rendered before that happened is clicked. */
  const winner = await (await w.fetch(
    '/api/experiments/' + encodeURIComponent(ids.experiment) + '/winner',
    {headers: {Authorization: 'Bearer ' + new w.URLSearchParams(w.location.search).get('token')}}
  )).json();
  const token = new w.URLSearchParams(w.location.search).get('token');
  const raced = await w.fetch(
    '/api/experiments/' + encodeURIComponent(ids.candidate) + '/winner/decisions',
    {method: 'POST',
     headers: {Authorization: 'Bearer ' + token, 'Content-Type': 'application/json'},
     body: JSON.stringify({actor: 'another reviewer', actor_kind: 'human',
                           decision: 'promote',
                           expected_selection_id: winner.expected_selection_id,
                           candidate_experiment_id: ids.candidate})});
  assert.equal(raced.status, 201, 'the other tab promoted the candidate');

  button('Keep the current winner').click();
  await until(() => detail().includes("Nobody's decision was overwritten"),
    'the stale explanation');
  assert.ok(detail().includes('It now names experiment ' + ids.candidate),
    'the refusal names what is current: ' + detail().slice(0, 900));
  assert.ok(button('Re-read and decide again'), 'and offers the mechanical retry');
  button('Re-read and decide again').click();
  await until(() => detail().includes('Not the winner'),
    'the re-read panel, now that this experiment lost the contest');

  // Losing the contest did NOT move the task's best run.
  w.taskView = 'runs';
  w.showExperimentTask(ids.experiment, ids.task, 'Roster experiment');
  await until(() => itemFor(2) && itemFor(2).textContent.includes('Best run'),
    'the best run after the winner moved');

  /* ================================================================
   * The comparison
   * ================================================================ */
  buttonIn(itemFor(1), 'Compare with the best run').click();
  /* Waited for on the RENDERED comparison, not on the text 'Answers': the
     pair picker's View option is called "Answers and artifacts" and is on
     screen the moment the controls paint, seconds before the comparison the
     assertions below are about. */
  const headingsIn = () =>
    [...d.querySelectorAll('#detail h2')].map(node => node.textContent);
  await until(() => detail().includes('Compare two recorded runs')
    && headingsIn().includes('Answers')
    && headingsIn().includes('Plan and execution'), 'the compare view');
  assert.ok(select('Left run') && select('Right run'), 'a pair picker for both sides');
  assert.ok(select('View'), 'and the view selector');

  // Answers and artifacts come FIRST, before any step alignment.
  const headings = headingsIn();
  assert.ok(headings.indexOf('Answers') < headings.indexOf('Plan and execution'),
    'the answer is above the plan: ' + headings.join(' | '));
  assert.ok(headings.includes('Artifacts'), 'artifacts too: ' + headings.join(' | '));

  // The pair identity, the review mark and the comment count are three
  // separate things on screen.
  await until(() => detail().includes('recorded comment(s) name this exact pair'),
    'the comment count');
  const pairCard = await until(
    () => [...d.querySelectorAll('#detail .card')].find(node =>
      node.textContent.includes('recorded comment(s) name this exact pair')
      && node.textContent.includes('marked this reviewed')),
    'the review-progress line beside the comment count');
  assert.ok(pairCard.textContent.includes('you have not marked this reviewed'),
    'review progress is the reviewer\'s own mark: '
    + pairCard.textContent.slice(0, 900));
  assert.ok(pairCard.textContent.includes('0 recorded comment(s)'),
    'and nobody has commented on this pair yet: '
    + pairCard.textContent.slice(0, 900));
  assert.ok(pairCard.textContent.includes('A comment is not a review mark'),
    'and the page says so rather than conflating them');

  // An unselected attempt cannot be silently absent from the picker: the one
  // with nothing recorded is present and disabled.
  const rightOptions = [...select('Right run').options];
  const dead = rightOptions.find(option => option.value === '4');
  assert.ok(dead, 'attempt 4 is offered rather than omitted');
  assert.ok(dead.disabled, 'and disabled, because there is nothing to compare');
  assert.ok(dead.textContent.toLowerCase().includes('no recorded turns'),
    'with the evidence\'s reason in the label: ' + dead.textContent);

  /* Aligned steps: attempt 1 ran add_item, list_items, complete_item and
     attempt 2 ran add_item, remove_item. Pairing by position would call
     list_items and remove_item the same step. */
  /* Re-issued against a freshly found control until the rows arrive: each
     repaint replaces the picker, and a change dispatched on the detached one
     does nothing at all. */
  const aligned = () => d.querySelectorAll('#detail .compareRow').length >= 3;
  for (let attempt = 0; attempt < 10 && !aligned(); attempt++) {
    const picker = select('View');
    if (picker) { change(picker, 'steps'); }
    for (let i = 0; i < 10 && !aligned(); i++) {
      await new Promise(r => setTimeout(r, 50));
    }
  }
  assert.ok(aligned(), 'the aligned rows never arrived: ' + detail().slice(0, 900));
  const rows = [...d.querySelectorAll('#detail .compareRow')];
  const rowText = rows.map(row => row.textContent);
  assert.ok(rowText.some(text => text.includes('both runs') && text.includes('add_item')),
    'the shared dispatch is one row: ' + rowText.join(' | ').slice(0, 500));
  /* Left is the BEST RUN (attempt 2) because the comparison was opened from
     the other attempt's "Compare with the best run", so `list_items` is the
     right-hand side's and `remove_item` the left's. The claim is that each is
     reported as one-sided, not which side it fell on. */
  const oneSided = command => rowText.find(text =>
    text.includes(command) && (text.includes('left only') || text.includes('right only')));
  assert.ok(oneSided('list_items'),
    'a step only one side ran is labelled as such: ' + rowText.join(' | ').slice(0, 600));
  assert.ok(oneSided('remove_item'), 'and so is the other side\'s');
  assert.ok(rowText.some(text => text.includes('left only'))
    && rowText.some(text => text.includes('right only')),
    'both one-sided labels are in use, so neither list was padded to match');
  assert.ok(rowText.some(text => text.includes('not recorded on the right run'))
    && rowText.some(text => text.includes('not recorded on the left run')),
    'the absent half says it is absent rather than rendering blank');
  assert.ok(detail().includes('never from list position'),
    'the page states the basis it does not use');

  // The span drilldown: the recorded identity of the dispatch.
  const matched = rows.find(row => row.textContent.includes('both runs'));
  const drill = [...matched.querySelectorAll('details')][0];
  drill.open = true;
  assert.ok(drill.textContent.includes('command call'), drill.textContent.slice(0, 300));
  assert.ok(drill.textContent.includes('span'), 'the span it came from');
  assert.ok(drill.textContent.includes('parameters source'),
    'and which of the two recording sources held the parameters');

  /* ================================================================
   * A categorized comment on one step of the pair
   * ================================================================ */
  buttonIn(matched, 'Comment on this step').click();
  await until(() => buttonIn(matched, 'Save this comment'), 'the pair composer');
  const composer = [...matched.querySelectorAll('.feedbackCard')][0];
  assert.ok(buttonIn(composer, 'Observations / Analysis'), 'the ordinary taxonomy');
  assert.ok(buttonIn(composer, 'Conclusions'));
  assert.ok(buttonIn(composer, 'Recommendations'));
  assert.ok(composer.textContent.includes('Left: attempt 2')
    || composer.textContent.includes('attempt 2'),
    'the composer names the pair it is about: ' + composer.textContent.slice(0, 400));

  buttonIn(composer, 'Conclusions').click();
  await until(() => buttonIn(composer, 'What went wrong'), 'the subcategory tabs');
  buttonIn(composer, 'What went wrong').click();
  const area = composer.querySelector('[data-compare-comment-box]');
  assert.ok(area.placeholder, 'the subcategory carries its own watermark');
  assert.equal(buttonIn(composer, 'What went wrong').getAttribute('aria-selected'), 'true');
  const remark = 'both runs dispatched add_item the same way and only one finished';
  area.value = remark;
  buttonIn(composer, 'Save this comment').click();
  await until(() => composer.textContent.includes(remark), 'the saved comment');
  assert.ok(composer.textContent.includes('Conclusions'),
    'recorded under the category that was picked: ' + composer.textContent.slice(-500));

  /* ================================================================
   * The task's Feedback tab sees it
   * ================================================================ */
  w.taskView = 'feedback';
  w.showExperimentTask(ids.experiment, ids.task, 'Roster experiment');
  await until(() => detail().includes(remark), 'the comment under the Feedback tab');
  const feedbackRow = items().find(node => node.textContent.includes(remark));
  assert.ok(feedbackRow.textContent.includes('Compared with:'),
    'the comparison comment names both recorded executions: '
    + feedbackRow.textContent.slice(0, 400));
  const anchored = feedbackRow.textContent;

  /* ================================================================
   * A different best run leaves the earlier comment alone
   * ================================================================ */
  w.taskView = 'runs';
  w.showExperimentTask(ids.experiment, ids.task, 'Roster experiment');
  await until(() => itemFor(1) && buttonIn(itemFor(1), 'Use as best run'),
    'the runs list again');
  buttonIn(itemFor(1), 'Use as best run').click();
  await until(() => button('Record this best run'), 'the second best-run prompt');
  button('Record this best run').click();
  await until(() => itemFor(1) && itemFor(1).textContent.includes('Best run'),
    'attempt 1 to become the best run');
  assert.ok(!itemFor(2).textContent.includes('Best run'),
    'and attempt 2 to stop being it');

  w.taskView = 'feedback';
  w.showExperimentTask(ids.experiment, ids.task, 'Roster experiment');
  await until(() => detail().includes(remark), 'the earlier comment, still there');
  const again = items().find(node => node.textContent.includes(remark));
  assert.equal(again.textContent, anchored,
    'the comment is unchanged: it names the runs it was written about, not '
    + 'whichever pair the current selection makes');

  /* Keyboard reachability of the task strip, which is how the three tabs are
     navigated without a mouse. */
  const strip = d.querySelector('#detail [role="tablist"]');
  const tabs = [...strip.querySelectorAll('button')];
  assert.equal(tabs.filter(node => node.tabIndex === 0).length, 1,
    'exactly one tab stop in the strip');
  assert.equal(tabs.find(node => node.tabIndex === 0).getAttribute('aria-selected'),
    'true', 'and it is the selected one');

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(String((error && error.stack) || error) + '\n');
  process.exit(1);
});
