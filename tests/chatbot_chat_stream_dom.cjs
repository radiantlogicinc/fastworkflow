/* Chat live-activity and artifact rendering, driven through the real page
   against a real chatbot server (fix-9eg.20.2, fix-9eg.20.3).

   Everything here is the page's own code: the frame reader parses real split
   NDJSON chunks, and every artifact body is fetched over HTTP from the seeded
   store — nothing is stubbed. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], turnKey = process.argv[4];
const htmlArtifact = process.argv[5], textArtifact = process.argv[6];
const errors = [];
const requested = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      /* The real fetch, with the request recorded: the DOM test needs to see
         WHICH store the page asked, not just what came back. */
      window.fetch = (path, options) => {
        requested.push(String(path));
        return fetch(new URL(path, url), options);
      };
    }});
  const w = dom.window, d = w.document;
  async function until(fn, what) {
    for (let i = 0; i < 150; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | errors: ' + JSON.stringify(errors));
  }
  const cards = msg => Array.from(msg.querySelectorAll('.artifacts .artifact'));
  const card = (msg, key) => cards(msg).find(
    c => c.querySelector('.aKey').textContent === key);

  /* ---- partial frames, ordering and deduplication ---------------------- */
  const seen = [], bad = [];
  const reader = w.tmFrameReader('ndjson', f => seen.push(f), line => bad.push(line));
  /* A chunk boundary falls inside a frame: the tail waits for its newline. */
  reader.push('{"type":"trace","seq":0,"turn_key":"exec-1","data":{"direction":');
  reader.push('"agent_to_workflow","raw_command":"add 2 and 3"}}\n{"type":"tra');
  reader.push('ce","seq":1,"turn_key":"exec-1","data":{"direction":"workflow_to_agent",'
    + '"command_name":"add_two_numbers","response_text":"5","success":true}}\n');
  /* A replayed frame is dropped rather than rendered twice. */
  reader.push('{"type":"trace","seq":1,"turn_key":"exec-1","data":{}}\n');
  reader.push('this is not a frame\n');
  reader.push('{"type":"output","seq":2,"turn_key":"exec-1","data":{"status":"completed"}}');
  reader.end();
  assert.deepEqual(seen.map(f => f.seq), [0, 1, 2]);
  assert.deepEqual(seen.map(f => f.type), ['trace', 'trace', 'output']);
  assert.equal(seen[0].data.raw_command, 'add 2 and 3');
  assert.deepEqual(bad, ['this is not a frame']);

  /* ---- the activity panel renders those frames, and only those --------- */
  const live = w.tmBubble('agent', '…');
  const activity = w.tmActivityPanel(live);
  assert.equal(live.querySelector('.activity').open, true);
  const state = live.querySelector('.actState');
  assert.equal(state.getAttribute('role'), 'status');
  assert.equal(state.getAttribute('aria-live'), 'polite');
  assert.equal(state.textContent, '· running');

  seen.filter(f => f.type === 'trace').forEach(f => activity.step(f.data));
  const rows = live.querySelectorAll('.actRow');
  assert.equal(rows.length, 2);
  assert.match(rows[0].textContent, /agent → workflow/);
  assert.match(rows[0].textContent, /add 2 and 3/);
  assert.match(rows[1].textContent, /workflow → agent/);
  assert.match(rows[1].textContent, /add_two_numbers/);
  assert.match(rows[1].textContent, /· ok/);
  assert.equal(live.querySelector('.actCount').textContent, '2 steps');

  /* A trace carrying anything beyond the public contract is still rendered
     field by field: the panel reads named fields, so a private one cannot
     leak through it. */
  activity.step({direction: 'agent_to_workflow', raw_command: 'noop',
                 reasoning: 'SECRET internal deliberation'});
  assert.ok(!live.textContent.includes('SECRET'));

  activity.setState('waiting for your reply');
  assert.equal(state.textContent, '· waiting for your reply');

  /* ---- artifacts land beside the answer, from the real store ----------- */
  const missing = 'c'.repeat(32);
  const turnOutput = {
    turn_key: turnKey, status: 'completed', success: true,
    answer: 'The report is attached.',
    command_outputs: [
      {command_name: 'add_todo', command_response: {response: 'done', success: true,
        artifacts: {
          note: 'hello inline artifact',
          report: {__fw_artifact_ref__: htmlArtifact, size: 53, content_type: 'text/html'},
          log: {__fw_artifact_ref__: textArtifact, size: 27, content_type: 'text/plain'},
          gone: {__fw_artifact_ref__: missing, size: 10, content_type: 'text/plain'}
        }}},
      /* Same artifact name from a different command: both must stay legible. */
      {command_name: 'summarize', command_response: {response: 'ok', success: true,
        artifacts: {note: 'a different note'}}}
    ]
  };
  w.tmRenderTurn(live, turnOutput);
  assert.equal(live.querySelector('.activity').open, false,
    'activity collapses once the answer is painted');
  assert.ok(live.querySelector('.activity .cmdOut'),
    'the turn command lines fold into the activity panel');
  const order = Array.from(live.children).map(n => n.className);
  assert.ok(order.indexOf('activity') < order.indexOf('bubble'),
    'the activity reads above the answer: ' + JSON.stringify(order));
  assert.ok(order.indexOf('bubble') < order.indexOf('artifactsLinkRow'),
    'the artifacts link sits below the answer: ' + JSON.stringify(order));

  /* Artifacts wait behind one link, and open beside the answer one at a time. */
  const link = live.querySelector('.artifactsLink');
  const panel = live.querySelector('.artifacts');
  assert.match(link.textContent, /5 artifacts/);
  assert.equal(panel.hidden, true, 'artifacts start collapsed behind the link');
  assert.equal(link.getAttribute('aria-controls'), panel.id);
  link.click();
  assert.equal(panel.hidden, false);
  assert.equal(link.getAttribute('aria-expanded'), 'true');
  const shown = () => cards(live).filter(c => !c.hidden).map(c => c.querySelector('.aKey').textContent);
  const [first, prev, next, last] = Array.from(panel.querySelectorAll('.aNav button'));
  const position = () => panel.querySelector('.aPosition').textContent;
  assert.deepEqual(shown(), ['note']);
  assert.equal(position(), '1 of 5');
  assert.ok(first.disabled && prev.disabled && !next.disabled && !last.disabled);
  next.click();
  assert.deepEqual(shown(), ['report']);
  assert.equal(position(), '2 of 5');
  last.click();
  assert.equal(position(), '5 of 5');
  assert.ok(!first.disabled && !prev.disabled && next.disabled && last.disabled);
  prev.click();
  assert.deepEqual(shown(), ['gone']);
  first.click();
  assert.equal(position(), '1 of 5');
  panel.querySelector('.aClose').click();
  assert.equal(panel.hidden, true);
  assert.equal(link.getAttribute('aria-expanded'), 'false');
  assert.match(live.querySelector('.bubble').textContent, /The report is attached/);
  assert.equal(cards(live).length, 5);
  assert.deepEqual(cards(live).map(c => c.querySelector('.aKey').textContent),
    ['note', 'report', 'log', 'gone', 'note']);
  /* Duplicate names are told apart by the command that produced them. */
  const notes = cards(live).filter(c => c.querySelector('.aKey').textContent === 'note');
  assert.match(notes[0].textContent, /from add_todo #1/);
  assert.match(notes[1].textContent, /from summarize #2/);
  assert.match(notes[0].textContent, /hello inline artifact/);

  await until(() => card(live, 'log').textContent.includes('plain text artifact payload'),
    'the offloaded text artifact never loaded');
  /* HTML only ever reaches a sandboxed frame. */
  await until(() => card(live, 'report').querySelector('iframe'),
    'the offloaded HTML artifact never loaded');
  const frame = card(live, 'report').querySelector('iframe');
  assert.equal(frame.getAttribute('sandbox'), '');
  assert.match(frame.getAttribute('srcdoc'), /alert\(1\)/);
  assert.ok(!card(live, 'report').querySelector('script'));

  /* A pruned reference loses its content, not the answer. */
  await until(() => card(live, 'gone').textContent.includes('no longer in the store'),
    'the missing artifact never reported itself');
  assert.match(live.querySelector('.bubble').textContent, /The report is attached/);

  /* Every artifact offers a same-origin view/download fallback. */
  const href = card(live, 'log').querySelector('.aActions a').getAttribute('href');
  assert.ok(href.startsWith('/api/artifact/' + textArtifact + '?token='), href);

  /* ---- painting the same turn twice replaces, never stacks ------------- */
  w.tmRenderTurn(live, turnOutput);
  assert.equal(live.querySelectorAll('.meta').length, 1);
  assert.equal(live.querySelectorAll('.artifacts').length, 1);
  assert.equal(live.querySelectorAll('.artifactsLink').length, 1);
  assert.equal(live.querySelectorAll('.activity').length, 1,
    'the activity record must survive a repaint');
  assert.equal((live.querySelector('.bubble').textContent.match(/attached/g) || []).length, 1);

  /* ---- reopened history shows stored artifacts, marked as stored -------
     With a recorded experiment selected in the Debug tab: a chat turn's
     record and artifacts live in the store that produced them, so the chat
     reads must not be scoped to whatever that other tab last selected. */
  w.benchmarkExperimentSource = 'exp-selected-in-the-debug-tab';
  const before = requested.length;
  const stored = w.tmRenderStoredTurn(
    {turn_key: turnKey, status: 'completed', success: true, answer: 'stored answer',
     user_message: 'add milk'}, true);
  await until(() => stored.querySelector('.artifacts'), 'stored artifacts never loaded');
  assert.match(stored.querySelector('.artifacts').textContent,
    /from the stored record of this turn/);
  assert.ok(card(stored, 'note').textContent.includes('hello inline artifact'));
  await until(() => card(stored, 'log').textContent.includes('plain text artifact payload'),
    'the stored offloaded artifact never loaded');
  /* Nothing here claims to be live. */
  assert.ok(!stored.querySelector('.activity'));

  const chatReads = requested.slice(before);
  assert.ok(chatReads.some(p => p.startsWith('/api/turn/')), JSON.stringify(chatReads));
  assert.ok(chatReads.some(p => p.startsWith('/api/artifact/')), JSON.stringify(chatReads));
  assert.deepEqual(chatReads.filter(p => p.includes('benchmark_experiment')), [],
    'a chat read was scoped to the Debug tab selection');
  w.benchmarkExperimentSource = null;

  /* An older turn keeps a button instead of a round trip, and still loads. */
  const older = w.tmRenderStoredTurn(
    {turn_key: turnKey, status: 'completed', success: true, answer: 'older answer'}, false);
  const button = Array.from(older.querySelectorAll('button'))
    .find(b => b.textContent === 'show artifacts');
  assert.ok(button, 'an older restored turn should offer its artifacts on request');
  assert.ok(!older.querySelector('.artifacts'));
  button.click();
  await until(() => older.querySelector('.artifacts'), 'the button never loaded artifacts');

  /* The status line stays last: artifacts belong above it, not after it. */
  const kids = Array.from(stored.children).map(n => n.className);
  assert.ok(kids.indexOf('artifacts') < kids.indexOf('meta'),
    'artifacts rendered below the status line: ' + JSON.stringify(kids));

  /* ---- recovery reads the turn, it never re-runs it --------------------
     A body that died leaves tmPollTurn holding a key. Point the page's API
     base at this server, which genuinely does not have that turn, and watch
     what it does: only GET /turns reads, no second submission, and a bounded
     giving-up rather than an endless spinner. */
  w.tm.baseUrl = new URL(url).origin;
  w.tm.connected = true;
  w.tm.busy = true;
  const lost = w.tmBubble('agent', '…');
  const lostActivity = w.tmActivityPanel(lost);
  const recoveryFrom = requested.length;
  w.tmPollTurn('20260101T000000.000000Z-deadbeefdead', lost, 1500, lostActivity);
  await until(() => w.tm.busy === false, 'the recovery poll never settled');
  const recoveryReads = requested.slice(recoveryFrom);
  assert.ok(recoveryReads.length > 0, 'recovery made no request at all');
  assert.ok(recoveryReads.every(p => p.includes('/turns/')),
    'recovery did something other than read the turn: ' + JSON.stringify(recoveryReads));
  /* It ends by saying so, in the bubble and in the activity state — never by
     spinning forever, and never by sending the message again. */
  assert.equal(lost.className, 'chatMsg system');
  assert.ok(lost.querySelector('.bubble').textContent.length > 0);
  assert.match(lostActivity.node.querySelector('.actState').textContent,
    /timed out|lost|failed/);

  await new Promise(r => setTimeout(r, 300));
  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
