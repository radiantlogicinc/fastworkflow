/* Turn-page and step-page artifacts, driven through the real page against a
   real chatbot server.

   The turn page must offer the chat's own "N artifacts" link and side panel;
   a step page shows only the artifacts of the dispatches that step made,
   joined on command_call_id; navigating away must not leave a panel open. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], turnKey = process.argv[4], htmlArtifact = process.argv[5];
const errors = [];
const requested = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => {
        requested.push(String(path));
        return fetch(new URL(path, url), options);
      };
    }});
  const w = dom.window, d = w.document;
  const detail = () => d.getElementById('detail');
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out: ' + what + ' | detail=' + detail().textContent.slice(0, 600)
      + ' | errors: ' + JSON.stringify(errors));
  }
  const cardTitled = title => [...detail().querySelectorAll('.card')].find(
    c => { const h = c.querySelector(':scope > h2'); return h && h.textContent === title; });
  const row = label => [...detail().querySelectorAll('.wfRow')].find(
    r => (r.getAttribute('aria-label') || '').startsWith('Inspect ' + label));
  const keys = panel => [...panel.querySelectorAll('.artifact')]
    .map(c => c.querySelector('.aKey').textContent);
  const shown = panel => [...panel.querySelectorAll('.artifact')].filter(c => !c.hidden)
    .map(c => c.querySelector('.aKey').textContent);

  d.getElementById('modeDebug').click();
  await until(() => w.session, 'the session');
  w.selectTurn(turnKey);
  await until(() => cardTitled('Artifacts'), 'the turn Artifacts card');

  /* ---- turn page: the chat's link and panel, not a list of cards ------ */
  const turnCard = cardTitled('Artifacts');
  const link = turnCard.querySelector('.artifactsLink');
  const panel = turnCard.querySelector('.artifacts');
  assert.ok(link && panel, 'the turn card holds the link and its panel');
  assert.equal(link.textContent, '4 artifacts');
  assert.equal(panel.hidden, true, 'the panel starts closed behind the link');
  assert.equal(link.getAttribute('aria-controls'), panel.id);
  assert.equal(panel.getAttribute('role'), 'dialog');
  assert.equal(panel.getAttribute('aria-label'), 'Artifacts of this turn');
  assert.deepEqual(keys(panel), ['note', 'report', 'csv', 'orphan']);
  assert.ok(!turnCard.querySelector('.empty'));

  link.click();
  assert.equal(panel.hidden, false);
  assert.equal(link.getAttribute('aria-expanded'), 'true');
  assert.equal(w.tmOpenArtifacts.msg, turnCard, 'the card owns the open panel');
  assert.equal(w.tmOpenArtifacts.place.frame(), detail(),
    'the panel is kept inside the detail pane, not the chat log');
  assert.equal(w.tmOpenArtifacts.place.anchor(), link,
    'the panel is anchored to the link itself, not its full-width row');
  assert.ok(!cardTitled('Raw turn record (record_json)'), 'the turn page has no raw record card');

  /* ---- placement: under the link, left-aligned, clear of the record nav -
     jsdom has no layout, so the real elements report stubbed boxes and the
     page's own resize and scroll listeners re-place the panel. */
  const box = (left, top, width, height) => () => ({left, top, width, height,
    right: left + width, bottom: top + height, x: left, y: top});
  const recordNav = d.getElementById('recordNav');
  detail().getBoundingClientRect = box(360, 60, 920, 700);
  recordNav.getBoundingClientRect = box(1240, 300, 40, 140);
  const px = v => Number(String(v).replace(/px$/, ''));
  const RAIL_EDGE = 1240 - 24;

  link.getBoundingClientRect = box(412, 300, 80, 16);
  w.dispatchEvent(new w.Event('resize'));
  assert.equal(panel.style.left, '412px', 'left-aligned with the link');
  assert.equal(panel.style.top, '322px', 'just under the link');
  assert.equal(panel.style.bottom, '');
  assert.equal(panel.style.width, '640px');
  assert.equal(panel.style.maxHeight, (760 - 12 - 322) + 'px', 'kept inside the pane');
  assert.ok(px(panel.style.left) + px(panel.style.width) <= RAIL_EDGE,
    'its right edge stops short of the record navigator');

  /* A link near the right edge: the panel slides left rather than run under
     the record navigator. */
  link.getBoundingClientRect = box(900, 300, 80, 16);
  detail().dispatchEvent(new w.Event('scroll'));
  assert.equal(px(panel.style.left) + px(panel.style.width), RAIL_EDGE,
    'right edge pinned a clear gap left of the navigator');
  assert.equal(panel.style.top, '322px');

  /* Too little room below and more above: it opens upward from the link. */
  link.getBoundingClientRect = box(412, 680, 80, 16);
  detail().dispatchEvent(new w.Event('scroll'));
  assert.equal(panel.style.left, '412px');
  assert.equal(panel.style.top, 'auto');
  assert.equal(panel.style.bottom, (w.innerHeight - 674) + 'px', 'its bottom sits just above the link');
  assert.equal(panel.style.maxHeight, (674 - 72) + 'px');

  delete detail().getBoundingClientRect;
  delete recordNav.getBoundingClientRect;
  delete link.getBoundingClientRect;

  assert.equal(panel.querySelector('.aPosition').textContent, '1 of 4');
  const [first, prev, next, last] = [...panel.querySelectorAll('.aNav button')];
  assert.ok(first.disabled && prev.disabled && !next.disabled && !last.disabled);
  next.click();
  assert.deepEqual(shown(panel), ['report']);
  /* The offloaded artifact comes from the real store, into a sandboxed frame. */
  const report = [...panel.querySelectorAll('.artifact')][1];
  await until(() => report.querySelector('iframe'), 'the offloaded HTML artifact');
  assert.equal(report.querySelector('iframe').getAttribute('sandbox'), '');
  assert.ok(requested.some(p => p.startsWith('/api/artifact/' + htmlArtifact)),
    JSON.stringify(requested));
  const href = report.querySelector('.aActions a').getAttribute('href');
  assert.ok(href.startsWith('/api/artifact/' + htmlArtifact + '?token='), href);
  last.click();
  assert.deepEqual(shown(panel), ['orphan']);
  assert.match(panel.textContent, /from summarize #4/);

  /* Escape closes it; the link reopens it. */
  panel.dispatchEvent(new w.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  assert.equal(panel.hidden, true);
  assert.equal(link.getAttribute('aria-expanded'), 'false');
  assert.equal(w.tmOpenArtifacts, null);
  link.click();
  assert.equal(panel.hidden, false);

  /* ---- one panel at a time, across the chat and the turn page --------- */
  const bubble = w.tmBubble('agent', '…');
  w.tmRenderTurn(bubble, {turn_key: turnKey, status: 'completed', success: true,
    answer: 'chat answer', command_outputs: [{command_name: 'add_todo',
      command_response: {response: 'done', success: true, artifacts: {chat_note: 'x'}}}]});
  bubble.querySelector('.artifactsLink').click();
  assert.equal(bubble.querySelector('.artifacts').hidden, false);
  assert.equal(panel.hidden, true, 'opening a chat panel closes the turn page one');
  assert.equal(w.tmOpenArtifacts.place.frame(), d.getElementById('chatLog'));
  bubble.querySelector('.aClose').click();
  assert.equal(w.tmOpenArtifacts, null);

  /* ---- navigating between levels drops the open panel ----------------- */
  link.click();
  assert.equal(w.tmOpenArtifacts.msg, turnCard);
  row('Execution').click();
  await until(() => row('Step 1'), 'the execution level');
  assert.equal(w.tmOpenArtifacts, null, 'clearing the pane closes its panel');
  assert.equal(link.getAttribute('aria-expanded'), 'false');
  assert.ok(!cardTitled('Artifacts'), 'the stage page has no Artifacts card');

  /* ---- step pages: only the step's own dispatches --------------------- */
  async function openStep(n) {
    w.state.path = w.state.path.slice(0, 2);
    w.renderLevel();
    await until(() => row('Step ' + n), 'step row ' + n);
    row('Step ' + n).click();
    await until(() => detail().querySelector('.levelHead h2')
      && detail().querySelector('.levelHead h2').textContent.startsWith('Step ' + n),
      'step ' + n + ' page');
  }

  /* Step 1: joined through the spans' own command_call_id. */
  await openStep(1);
  const s1 = cardTitled('Artifacts');
  assert.ok(s1, 'step 1 returned artifacts');
  assert.equal(s1.querySelector('.artifactsLink').textContent, '2 artifacts');
  const s1Panel = s1.querySelector('.artifacts');
  assert.equal(s1Panel.getAttribute('aria-label'), 'Artifacts of this step');
  assert.deepEqual(keys(s1Panel), ['note', 'report']);
  assert.match(s1Panel.textContent, /from add_todo #1/);
  /* It sits directly under the step's header card. */
  const cards = [...detail().children].filter(n => n.classList.contains('card'));
  assert.equal(cards.indexOf(s1), 1, 'the step Artifacts card follows the header card');
  s1.querySelector('.artifactsLink').click();
  assert.equal(s1Panel.hidden, false);

  /* Step 2: its dispatch returned nothing, so there is no card at all. */
  await openStep(2);
  assert.equal(w.tmOpenArtifacts, null);
  assert.ok(!cardTitled('Artifacts'), 'a step without artifacts shows nothing');

  /* Step 3: no id on its spans; the record's execution_records ref joins it.
     The source ordinal is still the output's place in the whole turn. */
  await openStep(3);
  const s3 = cardTitled('Artifacts');
  assert.ok(s3, 'step 3 is attributed through execution_records');
  assert.equal(s3.querySelector('.artifactsLink').textContent, '1 artifact');
  assert.deepEqual(keys(s3.querySelector('.artifacts')), ['csv']);
  assert.match(s3.textContent, /from export_todos #3/);

  /* Step 4: nothing joins it, so the output with no call id is not guessed
     onto it; that artifact appears on the turn page only. */
  await openStep(4);
  assert.ok(!cardTitled('Artifacts'), 'an unattributable output is not claimed by a step');

  await new Promise(r => setTimeout(r, 250));
  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
