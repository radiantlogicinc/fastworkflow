const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn) {
    for (let i = 0; i < 150; i++) {
      if (fn()) return;
      await new Promise(r => setTimeout(r, 50));
    }
    throw Error('Timed out: ' + d.getElementById('detail').textContent);
  }
  const button = text => [...d.querySelectorAll('#detail button')].find(node => node.textContent === text);
  const boxes = () => [...d.querySelectorAll('#detail input[type="checkbox"]')]
    .filter(node => node.getAttribute('aria-label') !== 'Select all ready to archive'
      && node.getAttribute('aria-label') !== 'Select all archived');

  await until(() => w.session && w.hierarchyRoot);
  d.getElementById('modeDebug').click();
  d.getElementById('archivedConversationsBtn').click();
  await until(() => d.getElementById('detail').textContent.includes('Risk Score Comparison')
    && d.getElementById('detail').textContent.includes('Second chat'));
  assert.equal(d.getElementById('detail').textContent.includes('Archive conversation'), false);
  const archiveSelected = button('Archive selected');
  assert.equal(archiveSelected.disabled, true);
  boxes().forEach(box => { box.click(); });
  assert.equal(archiveSelected.disabled, false);
  archiveSelected.click();
  await until(() => d.getElementById('confirmTitle').textContent === 'Archive 2 conversations?');
  d.getElementById('confirmDelete').click();
  await until(() => {
    const text = d.getElementById('detail').textContent;
    return text.includes('Nothing archived yet') === false
      && text.includes('No conversation is waiting')
      && text.includes('Risk Score Comparison')
      && text.includes('Second chat');
  });

  const deleteSelected = button('Delete selected');
  assert.equal(deleteSelected.disabled, true);
  boxes().forEach(box => { box.click(); });
  deleteSelected.click();
  await until(() => d.getElementById('confirmTitle').textContent === 'Delete 2 conversations?');
  d.getElementById('confirmDelete').click();
  await until(() => d.getElementById('detail').textContent.includes('Nothing archived yet'));
  assert.equal(errors.length, 0, errors.join('\n'));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
