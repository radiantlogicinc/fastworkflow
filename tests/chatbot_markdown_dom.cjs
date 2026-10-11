/* Agent answers in the chat bubble render markdown, including GFM tables.
   The page's own appendMarkdown / tmBubble / tmRenderTurn are what paint
   the bubble. Record-derived text stays text nodes: a cell that looks like
   markup must not become an element. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');

const srcDir = process.argv[3];
const names = fs.readdirSync(srcDir).filter((name) => !name.startsWith('.')).sort();
const html = names.map((name) => fs.readFileSync(path.join(srcDir, name))).join('');
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (err) => {
  if (err.type !== 'css-parsing') errors.push(String(err.message || err));
});

const SAMPLE = [
  'Angelica Schneider is a member of **24 groups**. Below is the complete list of those groups:',
  '',
  '| # | Group UID | Group Label |',
  '|---|------------------------------------|----------------------------------------------|',
  '| 1 | 8216e08435c00f85ebe819d2dd14d983 | Azure DevOps_Technical Lead |',
  '| 2 | 45a84110f346dc6b3d64710bda485d25 | Jira_Technical Lead |',
].join('\n');

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39131/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = () => new Promise(() => {});
  },
});
const w = dom.window;
const d = w.document;

function texts(nodes) {
  return Array.from(nodes).map((node) => node.textContent);
}

assert.equal(typeof w.appendMarkdown, 'function');
assert.equal(typeof w.tmBubble, 'function');
assert.equal(typeof w.tmRenderTurn, 'function');

const msg = w.tmBubble('agent', SAMPLE);
const bubble = msg.querySelector('.bubble');
assert.ok(bubble.classList.contains('md'), 'agent bubble is markdown');
assert.equal(bubble.querySelector('strong').textContent, '24 groups');
assert.equal(bubble.querySelectorAll('table').length, 1);
assert.deepEqual(texts(bubble.querySelectorAll('th')), ['#', 'Group UID', 'Group Label']);
assert.deepEqual(texts(bubble.querySelectorAll('tbody tr')[0].children), [
  '1',
  '8216e08435c00f85ebe819d2dd14d983',
  'Azure DevOps_Technical Lead',
]);
assert.equal(bubble.querySelectorAll('tbody tr').length, 2);
assert.ok(!bubble.textContent.includes('|---|'), bubble.textContent);
assert.ok(!bubble.textContent.includes('**24'), bubble.textContent);
assert.equal(bubble.querySelector('img'), null);

/* The finished turn paints through the same path, replacing the placeholder. */
w.tmRenderTurn(msg, {
  turn_key: 'tk-markdown',
  status: 'completed',
  success: true,
  answer: SAMPLE,
  command_outputs: [],
});
assert.equal(msg.querySelectorAll('table').length, 1);
assert.equal(msg.querySelector('strong').textContent, '24 groups');

/* A table on its own, with alignment, inline code, and an escaped pipe. */
const aligned = [
  '| left | center | right |',
  '| :--- | :---: | ---: |',
  '| a | **b** | `c\\|d` |',
  '| keep \\| pipe | <img src=x onerror=alert(1)> | plain |',
].join('\n');
const box = d.createElement('div');
w.appendMarkdown(box, aligned.replace(/\n/g, '\r\n'));
assert.deepEqual(
  texts(box.querySelectorAll('th')).map((text, i) => {
    return box.querySelectorAll('th')[i].className + ':' + text;
  }),
  ['align-left:left', 'align-center:center', 'align-right:right']
);
const bodyRows = box.querySelectorAll('tbody tr');
assert.equal(bodyRows[0].children[1].querySelector('strong').textContent, 'b');
assert.equal(bodyRows[0].children[2].querySelector('code').textContent, 'c\\|d');
assert.equal(bodyRows[1].children[0].textContent, 'keep | pipe');
assert.equal(bodyRows[1].querySelector('img'), null);
assert.ok(bodyRows[1].children[1].textContent.includes('<img'));

/* A fence keeps a table as source text. A bare pipe is not a table. */
const fenced = d.createElement('div');
w.appendMarkdown(fenced, '```\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```');
assert.equal(fenced.querySelector('table'), null);
assert.equal(fenced.querySelector('pre code').textContent, '| a | b |\n| --- | --- |\n| 1 | 2 |');

const prose = d.createElement('div');
w.appendMarkdown(prose, 'Use a | b when you mean either.');
assert.equal(prose.querySelector('table'), null);
assert.equal(prose.textContent, 'Use a | b when you mean either.');
assert.ok(!prose.classList.contains('md'));

const mixed = d.createElement('div');
w.appendMarkdown(mixed, '## Groups\n\n- one\n- two\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n');
assert.equal(mixed.querySelector('h2').textContent, 'Groups');
assert.deepEqual(texts(mixed.querySelectorAll('li')), ['one', 'two']);
assert.deepEqual(texts(mixed.querySelectorAll('td')), ['1', '2']);

if (errors.length) throw new Error(errors.join('\n'));
process.exit(0);
