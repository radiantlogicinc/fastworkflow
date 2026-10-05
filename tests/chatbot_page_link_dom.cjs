/* The address bar names the page on screen: every move writes the fragment,
   and every fragment written opens the same page on a fresh load. Inside a
   turn a level is named by its span, or by position where it has none. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole, CookieJar} = require(process.argv[2] + '/node_modules/jsdom');
const base = process.argv[3];
const [experimentId, benchmarkId, taskId] = process.argv.slice(4);

async function open(hash, options) {
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });
  const address = (options && options.address) || base;
  const dom = await JSDOM.fromURL(address + hash, {runScripts: 'dangerously', virtualConsole,
    cookieJar: options && options.cookieJar,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, base), options);
    }});
  const w = dom.window, d = w.document, detail = d.getElementById('detail');
  const page = {
    w, d, detail, errors,
    hash: () => w.location.hash,
    heading: () => (detail.querySelector('.levelHead h2') || {}).textContent || '',
    crumbs: () => Array.from(detail.querySelectorAll('nav.crumbs button')).map(b => b.textContent),
    async until(fn, what) {
      for (let i = 0; i < 150; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
      throw Error('Timed out waiting for ' + what + ' at ' + w.location.hash + ': '
        + detail.textContent.slice(0, 400) + ' | errors: ' + JSON.stringify(errors));
    },
    row(title) {
      const found = Array.from(detail.querySelectorAll('.wfRow'))
        .find(r => r.getAttribute('aria-label').startsWith('Inspect ' + title));
      assert.ok(found, 'no "' + title + '" row');
      return found;
    },
    async close() {
      await new Promise(r => setTimeout(r, 200));
      assert.deepEqual(errors, []);
      w.close();
    }
  };
  return page;
}

(async () => {
  // The launch token leaves the address bar once the page has it; a later
  // load of the bare address is authenticated by the cookie it left behind.
  const jar = new CookieJar();
  let p = await open('#page=benchmarks', {cookieJar: jar});
  await p.until(() => p.detail.textContent.includes('BENCHMARK LIBRARY'), 'the library');
  const origin = new URL(base).origin;
  assert.equal(p.w.location.href, origin + '/#page=benchmarks');
  await p.close();
  p = await open('#page=benchmarks', {cookieJar: jar, address: origin + '/'});
  await p.until(() => p.detail.textContent.includes('BENCHMARK LIBRARY'),
    'the library opened without a token in the URL');
  await p.close();

  // Moving through a turn writes each level's name.
  p = await open('#turn=crumb-turn');
  await p.until(() => p.heading().includes('climb the crumbs'), 'the turn');
  assert.equal(p.hash(), '#turn=crumb-turn');
  p.row('Execution').click();
  await p.until(() => p.heading().startsWith('Execution'), 'Execution');
  assert.equal(p.hash(), '#turn=crumb-turn&span=crumb-exec');
  p.row('Step 1').click();
  await p.until(() => p.heading().startsWith('Step 1'), 'Step 1');
  assert.equal(p.hash(), '#turn=crumb-turn&span=crumb-step-1');
  Array.from(p.detail.querySelectorAll('nav.crumbs button'))
    .find(b => b.textContent === 'climb the crumbs').click();
  await p.until(() => p.heading().includes('climb the crumbs'), 'the turn again');
  assert.equal(p.hash(), '#turn=crumb-turn');
  p.row('Planning').click();
  await p.until(() => p.heading().startsWith('Planning'), 'Planning');
  assert.equal(p.hash(), '#turn=crumb-turn&level=0', 'a level with no span is named by position');

  // The copy button hands over the fragment, never the token in front of it.
  const copy = p.d.getElementById('copyLinkBtn');
  assert.equal(copy.style.display, '', 'Copy link shows in debug mode');
  copy.click();
  await p.until(() => p.d.getElementById('noticeStack').textContent.includes('#turn=crumb-turn&level=0'),
    'the copied link');
  assert.ok(!p.d.getElementById('noticeStack').textContent.includes('token'));

  // Chat names itself; coming back to debug brings the page's name back.
  p.d.getElementById('modeTest').click();
  assert.equal(p.hash(), '#test');
  assert.equal(copy.style.display, 'none');
  p.d.getElementById('modeDebug').click();
  assert.equal(p.hash(), '#turn=crumb-turn&level=0');

  // A fragment pasted into a live page opens its page.
  p.w.location.hash = 'turn=crumb-turn&span=crumb-step-2';
  await p.until(() => p.heading().startsWith('Step 2'), 'Step 2 from a pasted fragment');
  assert.equal(p.hash(), '#turn=crumb-turn&span=crumb-step-2');
  await p.close();

  // Each of those names opens its level on a fresh load.
  p = await open('#turn=crumb-turn&span=crumb-step-1');
  await p.until(() => p.heading().startsWith('Step 1'), 'Step 1 by span');
  assert.equal(p.hash(), '#turn=crumb-turn&span=crumb-step-1');
  await p.close();
  p = await open('#turn=crumb-turn&level=0');
  await p.until(() => p.heading().startsWith('Planning'), 'Planning by position');
  await p.close();
  p = await open('#turn=crumb-turn&level=7.3');
  await p.until(() => p.detail.textContent.includes('level 7.3'), 'the missing-level notice');
  assert.ok(p.heading().includes('climb the crumbs'), 'a missing level leaves the turn open at its top');
  await p.close();

  // Steps inferred from an older recording have no span: their position names them.
  p = await open('#turn=legacy-turn');
  await p.until(() => p.heading().includes('an older recording'), 'the legacy turn');
  p.row('Execution').click();
  await p.until(() => p.heading().startsWith('Execution'), 'legacy Execution');
  p.row('Step 2').click();
  await p.until(() => p.heading().startsWith('Step 2'), 'legacy Step 2');
  const legacyLink = p.hash();
  assert.match(legacyLink, /^#turn=legacy-turn&level=\d+\.\d+$/);
  await p.close();
  p = await open(legacyLink);
  await p.until(() => p.heading().startsWith('Step 2'), 'legacy Step 2 by position');
  await p.close();

  // Rail records are named the way the recording names them, not by their
  // navigation key, whose nested JSON escapes into an unreadable fragment.
  p = await open('');
  p.d.getElementById('modeDebug').click();
  await p.until(() => p.d.querySelectorAll('#convList summary').length, 'the rail');
  p.d.querySelector('#convList summary').click();
  await p.until(() => p.crumbs().includes('2026-09-08'), 'the date page');
  assert.equal(p.hash(), '#date=2026-09-08');
  const conversationRow = () => Array.from(p.d.querySelectorAll('#convList summary'))
    .find(s => s.textContent.includes('Conversation #10'));
  await p.until(conversationRow, 'the conversation row');
  conversationRow().click();
  await p.until(() => p.crumbs().includes('Conversation #10'), 'the conversation page');
  assert.equal(p.hash(), '#channel=cli:local&conversation=10', 'colons stay readable');
  await p.close();
  for (const [link, crumb] of [['#date=2026-09-08', '2026-09-08'],
                               ['#channel=cli:local&conversation=10', 'Conversation #10'],
                               ['#channel=cli%3Alocal&conversation=10', 'Conversation #10']]) {
    p = await open(link);
    await p.until(() => p.crumbs().includes(crumb), crumb + ' from ' + link);
    assert.equal(p.hash(), link.replace('%3A', ':'));
    await p.close();
  }
  p = await open('#channel=cli:local&conversation=99');
  await p.until(() => p.detail.textContent.includes('does not hold'), 'the missing-record notice');
  await p.close();

  // Benchmark-side pages.
  p = await open('#page=benchmarks');
  await p.until(() => p.detail.textContent.includes('BENCHMARK LIBRARY'), 'the library');
  assert.equal(p.hash(), '#page=benchmarks');
  await p.close();
  p = await open('#benchmark=' + encodeURIComponent(benchmarkId));
  await p.until(() => p.detail.textContent.includes('Tuning benchmark')
    && p.detail.textContent.includes('BENCHMARK ·'), 'the benchmark');
  assert.equal(p.hash(), '#benchmark=' + encodeURIComponent(benchmarkId));
  await p.close();
  p = await open('#experiment=' + encodeURIComponent(experimentId));
  await p.until(() => p.d.querySelector('#convList [aria-current="page"]')
    && p.detail.textContent.includes(experimentId.slice(-8)), 'the experiment');
  assert.equal(p.hash(), '#experiment=' + encodeURIComponent(experimentId));
  await p.close();

  // A task page carries the tab it is showing.
  const taskLink = '#experiment=' + encodeURIComponent(experimentId)
    + '&task=' + encodeURIComponent(taskId);
  p = await open(taskLink + '&view=feedback');
  await p.until(() => p.d.querySelector('[data-task-view="feedback"].active'), 'the feedback tab');
  assert.equal(p.hash(), taskLink + '&view=feedback');
  p.d.querySelector('[data-task-view="runs"]').click();
  await p.until(() => p.d.querySelector('[data-task-view="runs"].active'), 'the runs tab');
  assert.equal(p.hash(), taskLink);
  await p.close();

  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
