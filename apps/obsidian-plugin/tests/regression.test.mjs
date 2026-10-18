import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, statSync} from 'node:fs';
const source = readFileSync(new URL('../src/main.ts', import.meta.url), 'utf8');
const manifest = JSON.parse(readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));
test('manifest and release assets', () => {
  assert.equal(manifest.id, 'pt-english-tutor');
  assert.equal(manifest.isDesktopOnly, true);
  assert.ok(statSync(new URL('../main.js', import.meta.url)).size > 1000);
  assert.ok(statSync(new URL('../styles.css', import.meta.url)).size > 20);
});
test('all note edits are confined to Apply correction click handler', () => {
  const editPattern = /this\.view\.dispatch\(\{\s*changes\s*:/g;
  const positions = [...source.matchAll(editPattern)].map(x => x.index);
  assert.equal(positions.length, 1, 'Review any additional write paths before release');
  const apply = source.indexOf('text: "Apply correction"');
  const click = source.indexOf('.addEventListener("click"', apply);
  assert.ok(apply >= 0 && click > apply && click < positions[0]);
  const nextButton = source.indexOf('text: "Ignore once"', positions[0]);
  assert.ok(nextButton > positions[0]);
  assert.match(source.slice(click, positions[0]), /sliceDoc\(item\.start, item\.end\) !== item\.original/);
});
test('history records only after an approved edit and skips identical replacements', () => {
  assert.match(source, /void this\.record\(item\)/);
  assert.match(source, /item\.original === item\.replacement/);
  assert.match(source, /historyEnabled/);
});
test('word preferences persist locally and practice cannot change note content', () => {
  assert.match(source, /await this\.saveData\(this\.preferences\)/);
  const practice = source.slice(source.indexOf('class TutorPracticeModal'));
  assert.doesNotMatch(practice, /dispatch\(\{\s*changes/);
  assert.match(practice, /grammarPracticeSentence/);
});
test('service calls remain local', () => {
  const endpoints = [...source.matchAll(/url:\s*"(https?:\/\/[^"]+)"/g)].map(m => m[1]);
  assert.ok(endpoints.length > 0);
  assert.ok(endpoints.every(url => url.startsWith('http://127.0.0.1:8765/')));
});
