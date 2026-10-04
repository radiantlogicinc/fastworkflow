/* The page reading REAL bodies from /invoke_agent_stream, in a browser DOM.

   The bodies in argv[4] were captured from the real endpoint over the real
   hello_world workflow earlier in this test run — NDJSON, SSE, and one whose
   delivery deadline expired mid-turn. Nothing here is hand-written framing:
   the page's own reader parses the server's own bytes, split at awkward
   boundaries, and the decoded frames are applied to a real activity panel
   through the same mapping the live client uses. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const html = fs.readFileSync(process.argv[3], 'utf8');
const bodies = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

/* Chunk boundaries a real socket would give us: mid-field, mid-delimiter and
   mid-multibyte-free-ASCII alike. Prime-sized so no boundary lines up with
   the framing. */
function chunks(text, size) {
  const out = [];
  for (let i = 0; i < text.length; i += size) { out.push(text.slice(i, i + size)); }
  return out;
}

function read(w, format, body, extra) {
  const frames = [], bad = [];
  const reader = w.tmFrameReader(format, f => frames.push(f), line => bad.push(line));
  chunks(body, 7).forEach(c => reader.push(c));
  (extra || []).forEach(c => reader.push(c));
  reader.end();
  return {frames, bad};
}

(async () => {
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    url: 'http://127.0.0.1:9/?token=dom-frames',   /* discard port: no server */
    virtualConsole,
    beforeParse(window) {
      /* The page connects on load; there is deliberately nothing listening,
         so every call fails the way an unreachable server does. This test is
         about parsing bodies, not about the socket. */
      window.fetch = () => Promise.reject(new TypeError('Failed to fetch'));
    }
  });
  const w = dom.window;
  await new Promise(r => setTimeout(r, 100));

  /* The server control lives in the empty middle of the connection strip and
     reveals its fields in the section under that strip. */
  const strip = w.document.getElementById('connStrip');
  const toggle = w.document.getElementById('advToggle');
  const panel = w.document.getElementById('advPanel');
  assert.equal(toggle.parentElement, strip);
  assert.ok(strip.compareDocumentPosition(panel) & w.Node.DOCUMENT_POSITION_FOLLOWING);
  assert.equal(panel.hidden, true);
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  toggle.click();
  assert.equal(panel.hidden, false);
  assert.equal(toggle.getAttribute('aria-expanded'), 'true');
  assert.ok(panel.querySelector('#tBaseUrl'));
  toggle.click();
  assert.equal(panel.hidden, true);

  /* ---- NDJSON: the whole envelope, gapless, identity on every frame ---- */
  const nd = read(w, 'ndjson', bodies.ndjson);
  assert.deepEqual(nd.bad, [], 'a real NDJSON body failed to parse');
  assert.deepEqual(nd.frames.map(f => f.seq), nd.frames.map((_, i) => i));
  assert.equal(nd.frames[nd.frames.length - 1].type, 'output');
  assert.equal(new Set(nd.frames.map(f => f.turn_key)).size, 1);
  assert.ok(nd.frames[0].logical_turn_key, 'the durable recovery key is missing');

  /* A replayed tail (a retried read of frames already rendered) is dropped by
     seq rather than rendered twice. */
  const replayed = read(w, 'ndjson', bodies.ndjson,
    bodies.ndjson.split('\n').filter(Boolean).slice(0, 2).map(l => l + '\n'));
  assert.equal(replayed.frames.length, nd.frames.length);

  /* ---- SSE: same events, same payloads, different framing -------------- */
  const sse = read(w, 'sse', bodies.sse);
  assert.deepEqual(sse.bad, [], 'a real SSE body failed to parse');
  assert.deepEqual(sse.frames.map(f => f.type), nd.frames.map(f => f.type));
  assert.deepEqual(sse.frames.map(f => f.seq), nd.frames.map(f => f.seq));
  assert.equal(sse.frames[0].data.direction, nd.frames[0].data.direction);
  assert.equal(sse.frames[0].data.raw_command, nd.frames[0].data.raw_command);
  /* SSE payloads carry no identity by design; the header does. The reader
     says so rather than inventing a key. */
  assert.equal(sse.frames[0].turn_key, null);

  /* ---- a missed delivery deadline is not the end of the turn ----------- */
  const late = read(w, 'ndjson', bodies.ndjson_timeout);
  const lateKinds = late.frames.map(f => f.type);
  assert.ok(lateKinds.includes('timeout'), JSON.stringify(lateKinds));
  assert.equal(lateKinds[lateKinds.length - 1], 'output');

  /* ---- those frames, applied to the live UI ---------------------------- */
  for (const [format, parsed] of [['ndjson', nd], ['sse', sse], ['late', late]]) {
    const msg = w.tmBubble('agent', '…');
    const activity = w.tmActivityPanel(msg);
    assert.equal(activity.node.open, true, format + ': activity starts open so the exchange can stream');
    let finalized = null, failed = null;
    parsed.frames.forEach(f => w.tmApplyStreamFrame(f, activity, {
      output: out => { finalized = out; w.tmRenderTurn(msg, out); },
      error: detail => { failed = detail; }
    }));
    assert.ok(finalized, format + ': no answer was rendered');
    assert.equal(failed, null, format + ': a healthy turn reported an error');
    assert.ok(msg.querySelector('.bubble').textContent.includes('hello_world_workflow'),
      format + ': the workflow answer is not in the bubble');
    /* Each agent↔workflow interaction is one row, in arrival order. */
    const rows = msg.querySelectorAll('.actRow');
    assert.equal(rows.length, parsed.frames.filter(f => f.type === 'trace').length);
    assert.match(rows[0].textContent, /agent → workflow/);
    assert.match(rows[rows.length - 1].textContent, /workflow → agent/);
    assert.equal(msg.querySelector('.activity').open, false,
      format + ': activity collapsed once the answer was painted');

    if (format === 'late') {
      /* Reported as still working, not as a failure, and the answer that
         followed it still rendered above. */
      const notes = Array.from(msg.querySelectorAll('.actNote')).map(n => n.textContent);
      assert.ok(notes.some(t => /timed out/.test(t)), JSON.stringify(notes));
      assert.ok(!msg.className.includes('system'),
        'a deadline report turned the turn into an error bubble');
      assert.equal(finalized.success, true);
    }
  }

  /* A turn that is still waiting on the user has no final answer yet, so the
     exchange stays open. */
  const waiting = w.tmBubble('agent', '…');
  w.tmActivityPanel(waiting);
  w.tmRenderTurn(waiting, {
    status: 'awaiting_user', success: false, answer: 'Which one?',
    command_outputs: [{command_name: 'ask', command_response: {response: 'Which one?', success: true}}]
  });
  assert.equal(waiting.querySelector('.activity').open, true);
  assert.match(waiting.textContent, /waiting for your reply/);

  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
