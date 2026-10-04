/* Proof that the source-boundary assertions are load-bearing (fix-9eg.7.2).

   Runs the same switch against a PRIVATE copy of the page whose reset call
   has been removed, and asserts the defect reproduces: the old source's
   selection, experiment scoping and live-chat binding all survive into the
   new source. The shared working file is never touched — the copy is loaded
   as a string with the document URL set to the real server, so every request
   still goes to the real server. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const pageFile = process.argv[3], url = process.argv[4];
const workflowB = process.argv[5], sharedTurn = process.argv[6];
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const html = fs.readFileSync(pageFile, 'utf8');
  const dom = new JSDOM(html, {url, runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | errors: ' + JSON.stringify(errors));
  }
  const detail = () => d.getElementById('detail').textContent;

  await until(() => w.session && w.session.workflow_path, 'the page never loaded a session');
  w.selectTurn(sharedTurn);
  await until(() => detail().includes('recorded in source_a'),
    'source A never rendered its turn');
  w.benchmarkExperimentSource = 'exp-selected-in-source-a';
  w.tm.baseUrl = new URL(url).origin;
  w.tm.token = 'token-minted-for-source-a';
  w.tm.connected = true;
  w.tmComposerState();
  w.tmBubble('user', 'hello from source A');

  w.chooseWorkflow(workflowB);
  await until(() => w.session.workflow_path === workflowB, 'the switch never landed');
  await new Promise(r => setTimeout(r, 300));

  /* What the page carried into the new source without the boundary. The
     nav's own repaint eventually drops the selected turn and the detail pane
     on its next refresh, so those are not the leak; the chat binding and the
     experiment scoping are, and they persist indefinitely. */
  const carriedOver = {
    experimentScope: w.benchmarkExperimentSource,
    chatConnected: w.tm.connected,
    chatBaseUrl: w.tm.baseUrl,
    chatToken: w.tm.token,
    composerLive: !d.getElementById('chatInput').disabled,
    oldTranscript: d.getElementById('chatLog').textContent.includes('hello from source A')
  };
  assert.deepEqual(carriedOver, {
    experimentScope: 'exp-selected-in-source-a',
    chatConnected: true,
    chatBaseUrl: new URL(url).origin,
    chatToken: 'token-minted-for-source-a',
    composerLive: true,
    oldTranscript: true
  }, 'the defect no longer reproduces, so the boundary test proves nothing');

  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
