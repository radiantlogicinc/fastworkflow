/* Tokens, cost and recorded cache state on the shipped page, in a real DOM
 * against a real server (fix-9eg.5 / fix-9eg.6).
 *
 * Two halves, because the two claims live in two places.
 *
 * The comparison pane is driven end to end: the page is navigated to the
 * task's Compare view, the real Left run / Right run controls are used to put
 * attempt 1 beside attempt 2, and the chips are read off the rendered DOM. The
 * evidence behind them is real recorded spans -- attempt 1 made two calls, one
 * with a complete usage report and a cache HIT and one that reported nothing;
 * attempt 2's trace holds a nested wrapper quoting its inner call's provider
 * response, which is ONE call recorded twice.
 *
 * The trace pane's per-call rendering is exercised through the page's own
 * functions with real span shapes, because the evidence for this fixture lives
 * in a benchmark-scoped database rather than the server's default store, so a
 * turn of it is not reachable by clicking the conversation rail. The functions
 * are the shipped ones and the assertions are made on real rendered elements.
 *
 * What must NOT happen anywhere: an absence rendered as a zero, an unrecorded
 * cache flag rendered as a miss, or a wrapper counted as a second call. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], experiment = process.argv[4], task = process.argv[5];
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

  d.getElementById('modeDebug').click();
  await until(() => w.session, 'the session');

  /* ================================================================
   * The comparison pane, over the real API
   * ================================================================ */
  w.benchmarkExperimentSource = experiment;
  w.taskView = 'compare';
  /* Re-issued until it sticks. The server is opened on a real evidence DB, so
     the rail's own first load resolves asynchronously and repaints `#detail`
     after this navigation -- a single call raced it and left the conversation
     empty-state on screen. */
  const compareReady = () => detail().includes('Answers') && select('Left run');
  for (let attempt = 0; attempt < 20 && !compareReady(); attempt++) {
    w.showExperimentTask(experiment, task, 'Usage experiment');
    for (let i = 0; i < 20 && !compareReady(); i++) {
      await new Promise(r => setTimeout(r, 50));
    }
  }
  await until(compareReady, 'the compare view');

  /* Attempt 1 is the Reference and so is already the left side; attempt 2 is
     put on the right through the real control. Each repaint replaces the
     picker, so the change is re-issued against a freshly found one until the
     pane's own badge says the pair has landed. */
  const paired = () => detail().includes('Right: attempt 2')
    && detail().includes('Tokens, cost and cache');
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

  const usageCard = await until(
    () => [...d.querySelectorAll('#detail .card')].find(node =>
      node.textContent.includes('Tokens, cost and cache')),
    'the card holding the usage section');
  const text = usageCard.textContent;

  // Both sides are named, and each reports how many calls it made.
  assert.ok(/Left:.*LLM call/.test(text), 'the left side reports its calls: ' + text);
  assert.ok(/Right:.*LLM call/.test(text), 'and so does the right: ' + text);

  // The recorded split is shown, not a bare total.
  assert.ok(text.includes('120 in'), 'prompt tokens are shown: ' + text);
  assert.ok(text.includes('30 out'), 'and completion tokens: ' + text);

  // Attempt 1 made two calls and only one of them reported usage. That has to
  // be visible: the total on screen is one call's, not the turn's.
  assert.ok(text.includes('incompletely counted'),
    'a partly-counted side says so rather than presenting its total as whole: '
    + text);

  // Attempt 2's trace holds a wrapper and its inner call. The page must say
  // ONE call, and say why it is not two.
  assert.ok(text.includes('counted once despite'),
    'the nested wrapper is named as a fold: ' + text);
  assert.ok(text.includes('nested wrapper'), 'and named as nesting: ' + text);

  // The cache state is reported on both sides, and the difference is stated
  // without being turned into a verdict.
  assert.ok(text.includes('from cache'), 'a recorded hit is shown: ' + text);
  assert.ok(text.includes('not recorded'),
    'and a call with no cache flag is not reported as a miss: ' + text);
  assert.ok(text.includes('it does not decide between them'),
    'a cache difference is an observation, not a reason to prefer a run: '
    + text);
  assert.ok(!/\b0 from cache\b/.test(text),
    'a side with no hits must omit the count, not print a zero: ' + text);

  /* ================================================================
   * The trace pane's per-call rendering, with real span shapes
   * ================================================================ */
  const llm = (id, attrs, parent) => ({
    span_id: id, parent_span_id: parent || null, name: 'fw.llm.call',
    kind: 'llm', status: 'ok', start_ns: 1000, end_ns: 2000, attributes: attrs
  });

  // Three answers, and the absent flag is its own.
  assert.equal(w.cacheStateOf(llm('a', {cache_hit: true})), 'hit');
  assert.equal(w.cacheStateOf(llm('b', {cache_hit: false})), 'miss');
  assert.equal(w.cacheStateOf(llm('c', {})), 'unknown');
  assert.equal(w.cacheStateOf({name: 'fw.command.execute', attributes: {}}), '',
    'a span that is not an LLM call has no cache state at all');

  // The per-call detail rows say what each state means, and a hit is not
  // described as a fault.
  const hitBox = d.createElement('div');
  w.renderSpanLevel(hitBox, llm('a', {cache_hit: true, model: 'm'}), '');
  assert.ok(hitBox.textContent.includes('cache hit'), hitBox.textContent);
  assert.ok(hitBox.textContent.includes('not a judgement about it'),
    'a hit is an observation: ' + hitBox.textContent);

  const unknownBox = d.createElement('div');
  w.renderSpanLevel(unknownBox, llm('c', {model: 'm'}), '');
  assert.ok(unknownBox.textContent.includes('whether it was served from the cache is unknown'),
    'an absent flag stays unknown: ' + unknownBox.textContent);
  assert.ok(!unknownBox.textContent.includes('cache miss'),
    'and is never rendered as a miss: ' + unknownBox.textContent);

  // A recorded zero and an unrecorded usage are different chips.
  const zero = w.spanTokens(llm('z', {
    usage: JSON.stringify({prompt_tokens: 0, completion_tokens: 0, total_tokens: 0})
  }));
  assert.equal(w.fmtTokens(zero), '0 tok',
    'a call that recorded zero tokens says zero');
  const silent = w.spanTokens(llm('s', {}));
  assert.equal(w.fmtTokens(silent), 'tokens not recorded',
    'a call that recorded nothing says so rather than falling silent');
  const partial = w.spanTokens(llm('p', {
    usage: JSON.stringify({prompt_tokens: 40})
  }));
  assert.ok(w.fmtTokens(partial).includes('incompletely counted'),
    'a partly reported call is not presented as complete: ' + w.fmtTokens(partial));

  // A level with no LLM call under it still says nothing about tokens.
  assert.equal(w.fmtTokens(w.noTokens()), '');
  assert.equal(w.fmtCacheCounts(w.noCache()), '');

  // The wrapper fold, in the tree the trace pane actually builds. The outer
  // and inner calls quote one provider response, so the level charges one.
  const shared = JSON.stringify({prompt_tokens: 120, completion_tokens: 30,
                                 total_tokens: 150});
  const outer = llm('outer', {usage: shared, cost: 0.004, history_uuid: 'r1'});
  const inner = llm('inner', {usage: shared, cost: 0.004, cache_hit: true,
                              history_uuid: 'r1'}, 'outer');
  const node = w.spanNode(outer, {outer: [inner]}, {outer: outer, inner: inner});
  assert.equal(node.cost.calls, 1, 'one call, not two');
  assert.equal(node.cost.total, 0.004, 'and its money once');
  assert.equal(node.tokens.total, 150, 'and its tokens once');
  /* Field by field rather than deepEqual: the object is built inside the page's
     realm, so its prototype is not this script's Object and a structural
     comparison would fail on that alone. */
  assert.equal(node.cache.hit, 1, 'the inner record kept its cache flag');
  assert.equal(node.cache.miss, 0);
  assert.equal(node.cache.unknown, 0);

  // Non-vacuous: two nested calls with their OWN responses are two calls.
  const ownOuter = llm('outer2', {usage: shared, cost: 0.004, history_uuid: 'r1'});
  const ownInner = llm('inner2', {usage: shared, cost: 0.004, history_uuid: 'r2'},
                       'outer2');
  const two = w.spanNode(ownOuter, {outer2: [ownInner]},
                         {outer2: ownOuter, inner2: ownInner});
  assert.equal(two.cost.calls, 2, 'nesting alone must not collapse two calls');
  assert.equal(two.tokens.total, 300);

  /* SIBLINGS quoting one response. Not nested, so the wrapper rule never saw
     them: the page charged both while the server charged one
     (shared-response-cost). Both stay visible as calls; the money and tokens
     are counted once, under the earlier of the two. */
  const step = {span_id: 'step', parent_span_id: null, name: 'fw.agent.step',
                kind: 'internal', status: 'ok', start_ns: 900, end_ns: 3000,
                attributes: {}};
  const first = llm('first', {usage: shared, cost: 0.25, cache_hit: true,
                              history_uuid: 'same-response'}, 'step');
  const second = llm('second', {usage: shared, cost: 0.25, cache_hit: true,
                                history_uuid: 'same-response'}, 'step');
  second.start_ns = 1500;
  const sibIds = {step: step, first: first, second: second};
  const sibs = w.spanNode(step, {step: [first, second]}, sibIds);
  assert.equal(sibs.cost.calls, 2, 'both are real calls');
  assert.equal(sibs.cost.recorded, 1, 'but one recorded charge between them');
  assert.equal(sibs.cost.unrecorded, 1,
    'the uncharged twin stays visible in the count rather than vanishing');
  assert.equal(sibs.cost.total, 0.25, 'the money once, not twice');
  assert.equal(sibs.tokens.total, 150, 'the tokens once, not twice');
  assert.equal(sibs.tokens.shared, 1, 'and the fold is named, not silent');
  assert.equal(sibs.cache.hit, 2,
    'a shared response is still two cache observations: both calls really were '
    + 'served from the cache');
  assert.ok(w.fmtTokens(sibs.tokens).includes('counted under another call'),
    'the chip explains where the second call\u2019s tokens went: '
    + w.fmtTokens(sibs.tokens));

  // Which twin is credited must not depend on the order they arrive in.
  const reversed = w.spanNode(step, {step: [second, first]}, sibIds);
  assert.equal(reversed.cost.total, 0.25);
  assert.equal(reversed.tokens.total, 150);

  // The credited call is the earlier one, by the same rule the server uses.
  const creditedBox = d.createElement('div');
  w.renderSpanLevel(creditedBox, first, '', w.chargeFolds(sibIds)['first'] || '');
  assert.ok(!creditedBox.textContent.includes('counted once'),
    'the credited call carries no fold notice: ' + creditedBox.textContent);
  const sharedBox = d.createElement('div');
  w.renderSpanLevel(sharedBox, second, '', w.chargeFolds(sibIds)['second'] || '');
  assert.ok(sharedBox.textContent.includes('already accounts for this provider'),
    'the shared twin says why it adds nothing: ' + sharedBox.textContent);
  assert.ok(sharedBox.textContent.includes('still a real call'),
    'without implying it never happened: ' + sharedBox.textContent);

  // Siblings with DIFFERENT responses are two charges, so the fold is not
  // collapsing calls merely for sharing a parent.
  const other = llm('other', {usage: shared, cost: 0.25, history_uuid: 'r9'},
                    'step');
  const distinct = w.spanNode(step, {step: [first, other]},
                              {step: step, first: first, other: other});
  assert.equal(distinct.cost.total, 0.5, 'two responses are two charges');
  assert.equal(distinct.tokens.total, 300);
  assert.equal(distinct.tokens.shared, 0);

  assert.deepEqual(errors, []);
  process.exit(0);
})().catch(e => {
  process.stderr.write(String((e && e.stack) || e) + '\n');
  process.exit(1);
});
