/* Switching evidence sources without relaunching the page (fix-9eg.7.2).

   Driven through the real page against real chatbot servers over two seeded
   workflows and two sealed workspaces. The two workflows share a turn key on
   purpose: a detail pane that survived the switch would keep resolving, and
   would be showing the wrong source's evidence.

   Every request the page makes is recorded, so "nothing from the old source"
   is checked as a property of the traffic, not only of the pixels. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], workflowB = process.argv[4], manifest1 = process.argv[5];
const sharedTurn = process.argv[6], artifactA = process.argv[7], payloadA = process.argv[8];
const secondWorkspaceSessionUrl = process.argv[9];
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
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | errors: ' + JSON.stringify(errors));
  }
  const detail = () => d.getElementById('detail').textContent;
  const composerLocked = () => d.getElementById('chatInput').disabled
    && d.getElementById('chatSend').disabled;
  const since = from => requested.slice(from);

  await until(() => w.session && w.session.workflow_path, 'the page never loaded a session');
  const serverOrigin = new URL(url).origin;

  /* ---- source A: a turn open, an experiment selected, a live chat -------
     tm is the page's own live-chat state; there is no FastAPI server in this
     harness, so the binding is set the way a connection would leave it. What
     is under test is what the switch does to it. */
  w.selectTurn(sharedTurn);
  await until(() => detail().includes('recorded in source_a'),
    'source A never rendered its turn');
  w.tm.baseUrl = serverOrigin;
  w.tm.token = 'token-minted-for-source-a';
  w.tm.channelId = 'chatbot';
  w.tm.activeConversationId = 7;
  w.tm.connected = true;
  w.tmComposerState();
  w.tmBubble('user', 'hello from source A');
  /* A's artifact really is on screen, fetched over HTTP from A's store. */
  const restored = w.tmRenderStoredTurn(
    {turn_key: sharedTurn, status: 'completed', success: true,
     answer: 'stored answer in source A', user_message: 'message recorded in source_a'},
    true);
  await until(() => restored.textContent.includes(payloadA),
    'source A never showed its offloaded artifact');
  assert.ok(requested.some(p => p.includes('/api/artifact/' + artifactA)));

  /* A search in progress owns the detail pane and walks the store over
     several round trips, and a comparison pair is keyed by experiment+task,
     which recur across archives. Both are source-scoped state. */
  d.getElementById('turnFindText').value = 'recorded';
  /* Submit rather than type: the debounce a keystroke schedules would start a
     second search behind this one and disown the first. */
  d.getElementById('turnFind').dispatchEvent(new w.Event('submit'));
  await until(() => w.turnFind.rows.length > 0,
    'the finder never found source A rows: ' + JSON.stringify({
      error: w.turnFind.error, running: w.turnFind.running}));
  assert.equal(w.turnFind.active, true);
  const findSeq = w.turnFind.seq;
  w.taskCompare.key = 'todo-list-v1\u001fadd an item';
  w.taskCompare.left = '1';
  w.taskCompare.right = '2';
  /* The Debug tab's experiment scope, which every turn and artifact read is
     filtered by while it is set. */
  w.benchmarkExperimentSource = 'exp-selected-in-source-a';

  /* ---- re-applying the SAME session changes nothing --------------------- */
  const epochBefore = w.tm.epoch;
  w.applySession();
  await new Promise(r => setTimeout(r, 150));
  assert.equal(w.tm.epoch, epochBefore, 'a same-source refresh reset the page');
  assert.equal(w.tm.connected, true, 'a same-source refresh dropped the live chat');
  assert.equal(w.tm.baseUrl, serverOrigin);
  assert.equal(w.benchmarkExperimentSource, 'exp-selected-in-source-a');
  assert.equal(w.state.turnKey, sharedTurn);
  assert.ok(d.getElementById('chatLog').textContent.includes('hello from source A'));
  assert.equal(w.turnFind.active, true, 'a same-source refresh cancelled the search');
  assert.equal(w.turnFind.seq, findSeq);
  assert.equal(w.taskCompare.left, '1');

  /* ---- switch to source B, which has no workflow server ----------------- */
  const atSwitch = requested.length;
  w.chooseWorkflow(workflowB);
  await until(() => w.session.workflow_path === workflowB, 'the switch never landed');
  await new Promise(r => setTimeout(r, 200));

  assert.notEqual(w.tm.epoch, epochBefore, 'the switch did not move the source epoch');
  /* Nothing of A's chat session survives, and the composer is shut. */
  assert.equal(w.tm.connected, false);
  assert.equal(w.tm.baseUrl, '');
  assert.equal(w.tm.token, '');
  assert.equal(w.tm.activeConversationId, null);
  assert.ok(composerLocked(), 'a viewer-only source left the composer live');
  assert.ok(!d.getElementById('chatLog').textContent.includes('hello from source A'));
  assert.match(d.getElementById('chatLog').textContent, /Switched evidence source/);
  /* Nor of A's evidence: selection, detail, artifact payload, experiment scope. */
  assert.equal(w.state.turnKey, null);
  assert.equal(w.state.storeId, null);
  assert.equal(w.benchmarkExperimentSource, null);
  assert.ok(!detail().includes('recorded in source_a'), detail().slice(0, 200));
  assert.ok(!d.body.textContent.includes(payloadA),
    "source A's artifact payload survived the switch");
  /* The search is stood down rather than left holding the pane, and its query
     and rows do not carry into a store they were not run against. */
  assert.equal(w.turnFind.active, false);
  assert.equal(w.turnFind.running, false);
  assert.equal(w.turnFind.timer, null);
  assert.equal(w.turnFind.rows.length, 0);
  assert.equal(w.turnFind.text, '');
  assert.equal(d.getElementById('turnFindText').value, '');
  assert.ok(w.turnFind.seq > findSeq, 'in-flight search pages still owned the view');
  /* The comparison pair is keyed by experiment+task, which recur across
     archives; selection_ui's onSourceSwitch() drops it at this boundary. */
  assert.equal(w.taskCompare.key, null,
    'the comparison pair survived the switch (is onSourceSwitch still defined?)');
  assert.equal(w.taskCompare.left, null);

  /* And no traffic to A's scope or A's server since the switch. */
  const afterSwitch = since(atSwitch);
  assert.deepEqual(afterSwitch.filter(p => p.includes('benchmark_experiment')), [],
    'a read after the switch was still scoped to the old experiment');
  assert.deepEqual(afterSwitch.filter(p => p.includes('/api/artifact/' + artifactA)), [],
    "the old source's artifact was re-read after the switch");

  /* ---- the composer cannot reach the server it just left ---------------- */
  const beforeSendAttempt = requested.length;
  const refusal = await w.tmFetch('/invoke_agent', {user_query: 'hi'})
    .then(() => 'the page sent a turn to the previous server', e => e.message);
  assert.match(refusal, /no workflow server is connected/);
  d.getElementById('chatInput').value = 'this must not be sent';
  w.tmSend();
  await new Promise(r => setTimeout(r, 150));
  assert.deepEqual(
    since(beforeSendAttempt).filter(p => /invoke_|\/initialize|\/turns\//.test(p)), [],
    'a chat call escaped after the source switch');

  /* ---- the colliding key now resolves to B, and only to B --------------- */
  const beforeB = requested.length;
  w.selectTurn(sharedTurn);
  await until(() => detail().includes('recorded in source_b'),
    'source B never rendered the shared turn');
  assert.ok(!detail().includes('recorded in source_a'));
  assert.deepEqual(since(beforeB).filter(p => p.includes('benchmark_experiment')), []);

  /* ---- switch again, this time to a sealed workspace --------------------
     The workspace holds the same workflow and can hold the same experiment
     and task ids, so a retained pair would show the previous store's attempts
     under this one's labels. */
  w.taskCompare.key = 'todo-list-v1\u001fadd an item';
  w.taskCompare.left = '3';
  const atWorkspace = requested.length;
  w.chooseWorkspace(manifest1);
  await until(() => w.session.workspace_mode === true, 'the workspace never opened');
  await new Promise(r => setTimeout(r, 200));
  assert.ok(!detail().includes('recorded in source_b'),
    "source B's turn survived into the workspace");
  assert.equal(w.state.turnKey, null);
  assert.equal(w.benchmarkExperimentSource, null);
  assert.equal(w.taskCompare.key, null,
    'the comparison pair followed the same task id into another source');
  assert.ok(composerLocked(), 'a read-only workspace left the composer live');
  assert.equal(d.getElementById('modeTest').style.display, 'none');
  assert.deepEqual(since(atWorkspace).filter(p => p.includes('benchmark_experiment')), []);
  const workspaceEpoch = w.tm.epoch;
  const workspaceOne = w.session;

  /* ---- a different workspace over the SAME workflow is a different source
     A chatbot in workspace mode is read-only, so this payload comes from a
     second server that opened the second manifest. It is a real session
     payload; only its delivery is out of band. */
  const second = await (await fetch(secondWorkspaceSessionUrl)).json();
  assert.equal(second.session.workspace_mode, true);
  assert.equal(second.session.workspace.workflow_folderpath,
    workspaceOne.workspace.workflow_folderpath,
    'the two workspaces were meant to share a workflow');
  assert.notEqual(w.sourceIdentity(second.session), w.sourceIdentity(workspaceOne));
  w.session = second.session;
  w.applySession();
  await new Promise(r => setTimeout(r, 150));
  assert.notEqual(w.tm.epoch, workspaceEpoch,
    'switching workspaces over one workflow was treated as the same source');

  await new Promise(r => setTimeout(r, 300));
  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
