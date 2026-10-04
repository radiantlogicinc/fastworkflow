/* Reusing a recorded message as a new chat turn (fix-9eg.7.4), driven through
   the real page against a real chatbot server serving real recorded turns.

   The question this answers is not "does a string get copied" but "does
   opening and copying recorded evidence cause anything to happen": every
   request the page makes is recorded, so an execution triggered by reading
   evidence would show up here. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], multilineTurn = process.argv[4], withheldTurn = process.argv[5];
const multiline = process.argv[6];
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
    for (let i = 0; i < 150; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | errors: ' + JSON.stringify(errors));
  }
  const reuseButton = () => Array.from(d.querySelectorAll('#detail .reuseAction button'))
    .find(b => b.textContent === 'Reuse this message in chat');
  const input = d.getElementById('chatInput');

  /* ---- a recorded turn, opened from the record navigator --------------- */
  w.selectTurn(multilineTurn);
  await until(() => reuseButton(), 'the recorded turn never offered its message');
  /* The record shows the message as recorded, line breaks and all. */
  const shown = d.querySelector('#detail .msgBlock').textContent;
  assert.ok(shown.includes('second line'), shown);

  /* ---- copying it is a read, not a run --------------------------------- */
  const before = requested.length;
  reuseButton().click();
  await new Promise(r => setTimeout(r, 200));
  assert.deepEqual(requested.slice(before), [],
    'opening/copying recorded evidence made a request');

  /* Verbatim, including the line breaks an <input> would have eaten. */
  assert.equal(input.value, multiline);
  assert.ok(input.value.includes('\n'), JSON.stringify(input.value));
  assert.equal(input.tagName, 'TEXTAREA');
  /* It lands in the live chat, which is where it can be sent from. */
  assert.equal(d.getElementById('testMain').className, 'visible');
  assert.equal(w.benchmarkExperimentSource, null,
    'the live composer stayed scoped to another tab’s evidence source');
  /* And it says what sending will do, without claiming a replay. */
  const notice = d.getElementById('reuseNotice');
  assert.equal(notice.className, 'visible');
  assert.match(notice.textContent, new RegExp(multilineTurn));
  assert.match(notice.textContent, /new turn/i);
  assert.match(notice.textContent, /does not re-run/i);

  /* ---- the recorded turn is untouched ---------------------------------- */
  const reread = await (await w.fetch('/api/turn/' + encodeURIComponent(multilineTurn)
    + '?token=' + new URL(url).searchParams.get('token'))).json();
  assert.equal(reread.turn.user_message, multiline);
  assert.ok(d.querySelector('#detail .msgBlock').textContent.includes('second line'));

  /* ---- editing before sending is ordinary typing ------------------------ */
  const editedFrom = requested.length;
  input.value = input.value + '\nand one more line';
  input.dispatchEvent(new w.Event('input'));
  /* Shift+Enter composes a line; it must not send. */
  input.dispatchEvent(new w.KeyboardEvent('keydown', {key: 'Enter', shiftKey: true}));
  await new Promise(r => setTimeout(r, 150));
  assert.deepEqual(requested.slice(editedFrom), [], 'editing the copy sent something');

  /* ---- a withheld message has nothing to reuse -------------------------- */
  w.selectTurn(withheldTurn);
  await until(() => d.querySelector('#detail .msgBlock')
      && d.querySelector('#detail .msgBlock').textContent.includes('withheld by policy'),
    'the withheld turn never rendered its message block');
  assert.equal(reuseButton(), undefined,
    'a message the capture policy withheld was offered for reuse');
  assert.match(d.querySelector('#detail .reuseAction').textContent,
    /not available in full/);
  /* Reopening the reusable turn restores the action. */
  w.selectTurn(multilineTurn);
  await until(() => reuseButton(), 'the reusable turn lost its action');

  /* ---- only an explicit send sends -------------------------------------
     The composer is pointed at this server, which is not a workflow server:
     the request 404s. What matters is WHEN one is attempted at all. */
  w.tm.baseUrl = new URL(url).origin;
  w.tm.token = new URL(url).searchParams.get('token');
  w.tm.connected = true;
  w.tm.busy = false;
  w.tmComposerState();
  reuseButton().click();
  const beforeSend = requested.length;
  await new Promise(r => setTimeout(r, 150));
  assert.deepEqual(requested.slice(beforeSend).filter(p => p.includes('/invoke_')), [],
    'copying a message into a connected composer started a turn by itself');

  d.getElementById('chatSend').click();
  await until(() => requested.slice(beforeSend).some(p => p.includes('/invoke_')),
    'pressing Send did not submit the message');
  /* The turn it produced says what it is: a new one, under current settings. */
  const sent = Array.from(d.querySelectorAll('#chatLog .chatMsg.user')).pop();
  /* Sent as typed, inner line breaks intact; the composer trims the ends the
     way it does for anything else. */
  assert.equal(sent.querySelector('.bubble').textContent, multiline.trim());
  assert.ok(sent.querySelector('.bubble').textContent.includes('\n'));
  const label = sent.querySelector('.reuseNote').textContent;
  assert.match(label, /New turn, under the current settings/);
  assert.match(label, new RegExp(multilineTurn));
  /* Sending consumes the provenance: the next message is not "reused". */
  assert.equal(d.getElementById('reuseNotice').className, '');
  assert.equal(input.value, '');

  await new Promise(r => setTimeout(r, 300));
  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
