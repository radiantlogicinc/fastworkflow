/* A step a policy verifier rule rewrote shows which rule, under which table,
   and what the intent reader saw; a step no rule touched shows none of it.
   The page's own buildRecordedStepNode renders the step. */
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
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39132/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = () => new Promise(() => {});
  },
});
const w = dom.window;
const d = w.document;
assert.equal(typeof w.buildRecordedStepNode, 'function');

function rendered(attributes) {
  const span = {span_id: 's1', name: 'fw.agent.step', status: 'ok', start_ns: 1,
                end_ns: 2, attributes: attributes};
  const node = w.buildRecordedStepNode(span, 1, 'earlier result', {}, {s1: span});
  const box = d.createElement('div');
  node.render(box);
  return box.textContent;
}

const blocked = rendered({
  step_index: 3, tool_name: 'finish', tool_args: {},
  observation: 'Nothing was closed yet: close the ticket, then answer.',
  answer: 'T-1 is closed.',
  policy_source: 'CLAIM', policy_outcome: 'rewrite', policy_mode: 'enforce',
  policy_table_version: 'helpdesk/1.0.0', policy_requests: 'R1:close[firm](T-1)',
});
assert.ok(blocked.includes('A policy rule rewrote this step'), blocked);
assert.ok(blocked.includes('CLAIM (rewrite, table helpdesk/1.0.0, enforce)'), blocked);
assert.ok(blocked.includes('T-1 is closed.'), 'the answer sent back is shown');
assert.ok(blocked.includes('R1:close[firm](T-1)'), 'the intent reader state is shown');

const plain = rendered({step_index: 1, tool_name: 'execute_workflow_query',
                        tool_args: {command: 'list_tickets'}, observation: 'T-1'});
assert.ok(!plain.includes('policy rule'), plain);

assert.deepEqual(errors, []);
process.exit(0);
