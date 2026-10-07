/* "LLM input as sent" on an LLM call, on the shipped page in a real DOM.
 *
 * A call under the capture cap recorded its messages whole, so the view renders
 * them from the span with no request. An over-cap call recorded only a cut
 * envelope plus `prompt_slots_ref`, so the view fetches the rebuilt prompt on
 * expand. A cut envelope with no ref offers no view: there is nothing whole to
 * show. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const page = fs.readFileSync(process.argv[3], 'utf8');
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

const prompts = [];
const rebuilt = {prompt: {available: true, verified: true, messages: [
  {role: 'system', content: 'rebuilt system prompt'},
  {role: 'user', content: 'rebuilt question'}]}};

(async () => {
  const dom = new JSDOM(page, {url: 'http://127.0.0.1/?token=t', runScripts: 'dangerously',
    virtualConsole: console,
    beforeParse(window) {
      window.fetch = path => {
        if (String(path).startsWith('/api/prompt/')) {
          prompts.push(String(path));
          return Promise.resolve(new Response(JSON.stringify(rebuilt),
            {status: 200, headers: {'Content-Type': 'application/json'}}));
        }
        return new Promise(() => {});
      };
    }});
  const w = dom.window, d = w.document;
  const tick = () => new Promise(r => setTimeout(r, 20));
  const expand = det => { det.open = true; det.dispatchEvent(new w.Event('toggle')); };

  const whole = [{role: 'system', content: 'line one\nline two'},
                 {role: 'user', content: 'the question'}];
  for (const messages of [whole, JSON.stringify(whole)]) {
    const host = d.createElement('div');
    w.appendPromptAsSent(host, {trace_id: 't', span_id: 's', attributes: {messages}});
    const det = host.querySelector('details.promptAsSent');
    assert.ok(det, 'a call that recorded its messages whole offers the view');
    assert.match(det.querySelector('summary').textContent, /^LLM input as sent \([^,]+\)$/);
    expand(det);
    const panels = det.querySelectorAll('.promptInput');
    assert.equal(panels.length, 1, 'one panel holds the whole LLM input');
    const blocks = [...panels[0].querySelectorAll(':scope > .msgBlock')];
    assert.deepEqual(blocks.map(b => b.querySelector('.lbl').textContent), ['system', 'user']);
    assert.equal(blocks[0].querySelector('pre').textContent, 'line one\nline two');
    assert.match(det.querySelector('.promptStatus').textContent, /recorded whole/);
  }
  assert.deepEqual(prompts, [], 'a whole prompt is rendered without a request');

  const cut = d.createElement('div');
  w.appendPromptAsSent(cut, {trace_id: 't', span_id: 's',
    attributes: {messages: {truncated: true, sha256: 'abc', original_bytes: 90000}}});
  assert.equal(cut.querySelector('details.promptAsSent'), null,
    'a cut envelope without pieces offers nothing to show');

  const over = d.createElement('div');
  w.appendPromptAsSent(over, {trace_id: 'turn-1', span_id: 'span-1', attributes: {
    messages: {truncated: true}, prompt_slots_ref: {messages_bytes: 90000, slot_count: 3}}});
  const det = over.querySelector('details.promptAsSent');
  assert.match(det.querySelector('summary').textContent, /^LLM input as sent \(88 KB\)$/);
  expand(det);
  for (let i = 0; i < 50 && !det.querySelector('.msgBlock'); i++) { await tick(); }
  assert.deepEqual(prompts, ['/api/prompt/turn-1/span-1'], det.textContent);
  assert.ok(det.querySelector('.msgBlock'), det.textContent);
  assert.equal(det.querySelectorAll('.promptInput').length, 1);
  assert.deepEqual([...det.querySelectorAll('.promptInput > .msgBlock pre')].map(p => p.textContent),
    ['rebuilt system prompt', 'rebuilt question']);
  assert.match(det.querySelector('.promptStatus').textContent, /^verified/);

  assert.deepEqual(errors, []);
  w.close();
})().catch(error => {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
});
