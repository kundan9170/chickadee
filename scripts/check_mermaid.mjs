// Validate every ```mermaid block in the repo against the real Mermaid parser
// (the same engine GitHub renders with). A diagram that fails here shows an
// error box on GitHub instead of a picture.
//
//   cd <repo> && npm install --no-save mermaid@11 jsdom && node scripts/check_mermaid.mjs
//
// Common breakages this catches, all of which we have actually hit:
//   - semicolons anywhere in a diagram (Mermaid treats them as statement separators)
//   - parentheses in `participant X as Name (qualifier)` aliases
//   - curly braces inside sequence-diagram message text
//   - unquoted flowchart labels containing : ( ) / # , = < >
//   - literal \n instead of <br/> for line breaks in node labels
import { readFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!DOCTYPE html><body></body>', { pretendToBeVisual: true });
Object.defineProperty(global, "navigator", { value: dom.window.navigator, configurable: true });
global.window = dom.window; global.document = dom.window.document;
global.Element = dom.window.Element;
global.HTMLElement = dom.window.HTMLElement; global.SVGElement = dom.window.SVGElement;
global.DOMPurify = undefined;
const mermaid = (await import('mermaid')).default;

const files = [];
(function walk(d){ for (const e of readdirSync(d)) {
  if (e === '.git' || e === 'node_modules') continue;
  const p = join(d,e); const s = statSync(p);
  if (s.isDirectory()) walk(p); else if (e.endsWith('.md')) files.push(p);
} })('.');

mermaid.initialize({ startOnLoad:false, securityLevel:'loose' });
let total=0, bad=0;
for (const f of files.sort()) {
  const txt = readFileSync(f,'utf8').split('\n');
  let inb=false, buf=[], start=0;
  for (let i=0;i<txt.length;i++){
    const l = txt[i];
    if (!inb && l.trim().startsWith('```mermaid')) { inb=true; buf=[]; start=i+2; continue; }
    if (inb && l.trim()==='```') {
      inb=false; total++;
      try { await mermaid.parse(buf.join('\n')); }
      catch(e){ bad++; console.log(`\n✗ ${f}:${start}  ${String(e.message||e).split('\n')[0]}`);
                console.log('   ' + String(e.message||e).split('\n').slice(1,5).join('\n   ')); }
      continue;
    }
    if (inb) buf.push(l);
  }
}
console.log(`\n=== ${total} diagrams parsed, ${bad} failed ===`);
