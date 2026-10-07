/* A turn's badge says whether the TURN finished, on the shipped page.
 *
 * The outcome is the lifecycle status (and why it did not finish). Whether
 * every command succeeded is a separate, neutral note: a command failing inside
 * a completed turn is ordinary, and must not read as the turn failing. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const page = fs.readFileSync(process.argv[3], 'utf8');
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

const dom = new JSDOM(page, {url: 'http://127.0.0.1/?token=t', runScripts: 'dangerously',
  virtualConsole: console,
  beforeParse(window) { window.fetch = () => new Promise(() => {}); }});
const w = dom.window;
const badges = turn => [...w.statusBadge(turn).querySelectorAll('.badge')]
  .map(b => [b.className, b.textContent]);

try {
  assert.deepEqual(badges({status: 'completed', success: 1}),
    [['badge ok', 'completed']]);
  assert.deepEqual(badges({status: 'completed', success: 0}),
    [['badge ok', 'completed'], ['badge unknown', 'a command reported failure']]);
  assert.deepEqual(badges({status: 'failed', success: 1, failure_reason: 'max_iters_exhausted'}),
    [['badge fail', 'incomplete — ran out of agent iterations']]);
  assert.deepEqual(badges({status: 'failed', success: false, failure_reason: 'TimeoutError'}),
    [['badge fail', 'incomplete — TimeoutError'],
     ['badge unknown', 'a command reported failure']]);
  assert.deepEqual(badges({status: 'abandoned', success: 1}),
    [['badge fail', 'incomplete — abandoned']]);
  assert.deepEqual(badges({status: 'awaiting_user', success: 0}),
    [['badge progress', 'awaiting your reply']]);
  const titles = turn => [...w.statusBadge(turn).querySelectorAll('.badge')].map(b => b.title);
  const [statusHint, commandHint] = titles({status: 'completed', success: 0});
  assert.match(statusHint, /^The turn's status: the agent ran to the end/);
  assert.match(commandHint, /^At least one command the agent ran returned a failure/);
  assert.match(titles({status: 'failed', success: 1})[0], /stopped before finishing/);
  assert.match(titles({status: 'awaiting_user', success: 0})[0], /asked you a question/);
  assert.deepEqual(errors, []);
  w.close();
} catch (error) {
  process.stderr.write(error.stack + '\n');
  process.exit(1);
}
