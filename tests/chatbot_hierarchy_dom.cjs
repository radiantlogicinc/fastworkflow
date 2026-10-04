const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], eid = process.argv[4];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });
(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  // Success notices live ~4.5s, less than the HTTP round trips between an action
  // and its assertion, so record them as emitted rather than reading the stack.
  const notices = [], notify = w.showNotice;
  w.showNotice = (message, kind, detail) => { notices.push(message); return notify.call(w, message, kind, detail); };
  async function until(fn) {
    for (let i=0; i<150; i++) { if (fn()) return; await new Promise(r=>setTimeout(r,50)); }
    throw Error('Timed out: '+ d.getElementById('detail').textContent);
  }
  // The rail names an experiment by its id: its label is the author's
  // optional description and is not shown there.
  const expName = 'Experiment \u00b7 ' + eid.slice(-8);
  const summaries = () => [...d.querySelectorAll('#convList summary')];
  const find = text => summaries().find(e => e.textContent.includes(text));
  const click = text => { const e=find(text); assert.ok(e, 'Missing node '+text); e.click(); };
  await until(()=>find('2026-09-07'));
  assert.equal(find('ad-hoc conversations'), undefined);
  assert.equal(find('Tuning benchmark'), undefined);
  click('2026-09-07');
  assert.ok(d.getElementById('detail').textContent.includes('UTC'));
  d.getElementById('navBenchmarks').click();
  await until(()=>find('Tuning benchmark'));
  assert.equal(find('2026-09-07'), undefined);
  assert.equal(d.querySelector('#experimentsBtn'),null);
  assert.equal(d.querySelector('#benchmarksBtn'),null);
  assert.equal(d.querySelector('#fStatus'),null);
  assert.equal(d.querySelector('#rail nav.crumbs'),null);
  click('Tuning benchmark');
  await until(()=>d.getElementById('detail').textContent.includes('Review this benchmark'));
  click(expName);
  await until(()=>d.getElementById('detail').textContent.includes('RECORDED EXPERIMENT'));
  // The winner card reads the contest after the page is otherwise drawn, so a
  // snapshot of the whole pane is only stable once that read has landed.
  // Comparing an arriving pane against a settled one is a coin flip, not a
  // navigation assertion.
  const contestSettled = () =>
    !d.getElementById('detail').textContent.includes('reading the contest');
  await until(contestSettled);
  const railView = d.getElementById('detail').textContent;
  assert.ok(railView.includes('Postmortem'), railView);
  assert.ok(!railView.includes('Save notes'), railView);
  assert.ok(!railView.includes('Notes'), railView);
  // Scoped to the result's own key list. The winner panel renders into the
  // same card and carries a key list of its own, so a query across the pane
  // describes whichever panels happen to have finished loading rather than
  // the result being asserted.
  assert.deepEqual(
    [...d.querySelector('#detail dl.kv').querySelectorAll('dt')].map(e=>e.textContent),
    ['status', 'declared', 'scored attempts', 'pass@1', 'pass^1', 'verdict sources']);
  assert.equal(d.querySelector('#detail details.provenance'), null);
  assert.ok(![...d.querySelectorAll('#detail h2')].some(e=>e.textContent==='Evidence runs'));
  assert.ok(railView.includes('evidence valid \u00b7 1 segment'), railView);
  const exp = find(expName).parentElement;
  assert.ok(exp.open);
  assert.equal(d.getElementById('navDistillations'), null);
  assert.ok(find(expName).parentElement.classList.contains('selected'));
  // Two routes to one experiment. The card in the benchmark's list and the rail's
  // node both open the recorded run's results; the card used to branch on
  // registration alone and open the pre-run handoff page for a finished run.
  click('Tuning benchmark');
  const cards = () => [...d.querySelectorAll('#detail .recordCard')].filter(e=>e.textContent.includes(eid));
  await until(()=>cards().length);
  cards()[0].click();
  await until(()=>d.getElementById('detail').textContent.includes('RECORDED EXPERIMENT'));
  await until(contestSettled);
  assert.equal(d.getElementById('detail').textContent, railView);
  assert.equal(w.benchmarkExperimentSource, eid);
  click('task'); // conversation node
  assert.ok(d.getElementById('detail').textContent.includes('1 turns'));
  assert.equal(w.benchmarkExperimentSource, eid);
  assert.ok(find('task').parentElement.open);
  click('experiment-turn'); // turns hang off their conversation in the rail
  await until(()=>d.getElementById('detail').textContent.includes('Feedback'));
  await until(()=>[...d.querySelectorAll('#detail .wfRow')].some(e=>e.textContent.includes('Planning')));
  const planningRow=[...d.querySelectorAll('#detail .wfRow')].find(e=>e.textContent.includes('Planning'));
  assert.equal(planningRow.getAttribute('role'),'button');
  planningRow.focus(); planningRow.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  assert.equal(find('Planning'), undefined);
  assert.ok(find('experiment-turn').parentElement.classList.contains('selected'));
  assert.ok(d.querySelector('#detail nav.crumbs').textContent.startsWith('Benchmarks'));
  const crumbs = d.querySelector('#detail nav.crumbs').textContent;
  for (const label of ['Tuning benchmark',expName,'task','experiment-turn','Planning']) assert.ok(crumbs.includes(label), crumbs);
  assert.ok(d.getElementById('detail').textContent.includes('Feedback'));
  await w.refreshConvs();
  assert.ok(d.querySelector('#detail nav.crumbs').textContent.includes('Planning'));
  assert.equal(find('Planning'), undefined);
  // Return through the full right-pane breadcrumb, then leave a trace fetch pending.
  [...d.querySelectorAll('#detail nav.crumbs button')].find(e=>e.textContent==='task').click();
  [...d.querySelectorAll('#detail button')].find(e=>e.textContent.includes('experiment-turn')).click();
  d.getElementById('navConversations').click();
  await until(()=>d.getElementById('detail').textContent.includes('2026-09-07'));
  assert.ok(find('2026-09-07').parentElement.classList.contains('selected'));
  assert.ok(d.getElementById('detail').textContent.includes('UTC'));
  const day = find('2026-09-07').parentElement;
  assert.ok(day.textContent.includes('Conversation #1'));
  assert.ok(day.textContent.includes('plain yesterday'));
  assert.ok(!day.textContent.includes('plain today'));
  await new Promise(r=>setTimeout(r,250));
  assert.ok(d.getElementById('detail').textContent.includes('2026-09-07'));
  // Authoring, feedback, busy state, focus, and deletion through real HTTP.
  d.getElementById('navBenchmarks').click();
  await until(()=>find('Tuning benchmark'));
  click('Tuning benchmark');
  const button = text => [...d.querySelectorAll('#detail button')].find(e=>e.textContent===text);
  await until(()=>button('New experiment'));
  button('New experiment').click();
  await until(()=>button('Delete empty experiment'));
  assert.equal(d.activeElement.tagName, 'H1');
  const createdId = d.querySelector('[aria-label="Experiment ID"]').value;
  assert.ok(notices.includes('Experiment created'));
  button('Delete empty experiment').click();
  await until(()=>d.getElementById('confirmDialog').hasAttribute('open'));
  assert.equal(d.activeElement.id,'cancelDelete');
  d.getElementById('cancelDelete').click();
  await until(()=>!d.getElementById('confirmDialog').hasAttribute('open'));
  assert.ok(button('Delete empty experiment'));
  assert.equal(d.activeElement,button('Delete empty experiment'));
  button('Delete empty experiment').click();
  d.getElementById('confirmDelete').click();
  await until(()=>notices.includes('Empty experiment deleted'));
  await until(()=>![...d.querySelectorAll('#convList [data-experiment-id]')].some(e=>e.dataset.experimentId===createdId));
  await until(()=>find('Tuning benchmark'));
  click('Tuning benchmark');
  await until(()=>button('New experiment'));
  assert.equal(d.activeElement.tagName,'H1');
  button('Edit benchmark').click();
  await until(()=>d.querySelector('[aria-label="Title"]'));
  const title = d.querySelector('[aria-label="Title"]');
  title.value=''; button('Save benchmark').click();
  assert.equal(d.activeElement,title);
  assert.ok(d.querySelector('.fieldError').textContent.includes('title'));
  button('+ Add task').click();
  assert.equal(d.activeElement.tagName,'TEXTAREA');
  title.value='Updated benchmark';
  button('Save benchmark').click();
  assert.ok(button('Save benchmark').disabled);
  assert.equal(button('Save benchmark').getAttribute('aria-busy'),'true');
  await until(()=>d.querySelector('#detail h1')?.textContent==='Updated benchmark');
  assert.equal(d.activeElement.tagName,'H1');
  await w.apiDelete('/api/benchmark-experiments/missing').catch(()=>{});
  assert.ok([...d.querySelectorAll('.notice.error')].some(e=>e.textContent.includes('experiment not found')));
  w.showNotice('Temporary notification');
  await until(()=>!d.querySelector('#noticeStack').textContent.includes('Temporary notification'));
  assert.deepEqual(errors,[]);
  dom.window.close();
})().catch(e=>{ process.stderr.write(e.stack+'\n'); process.exit(1); });
