// Playwright smoke test for Emergent Economies.
// Run: node tests/smoke.mjs   (expects the Vite dev server on :5173)
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE = process.env.BASE_URL || 'http://localhost:5173/';
const SHOTS = resolve(__dirname, 'screenshots');
mkdirSync(SHOTS, { recursive: true });

const problems = [];
const notes = [];
let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  page.on('console', (msg) => {
    const text = `[console.${msg.type()}] ${msg.text()}`;
    if (msg.type() === 'error') problems.push(text);
    else if (msg.type() === 'warning') notes.push(text);
    console.log(text);
  });
  page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`));
  page.on('requestfailed', (req) => problems.push(`[requestfailed] ${req.url()} ${req.failure()?.errorText}`));

  const resp = await page.goto(BASE, { waitUntil: 'load', timeout: 30000 });
  console.log(`HTTP ${resp.status()} ${resp.url()}`);
  if (!resp.ok()) problems.push(`[http] unexpected status ${resp.status()}`);

  await page.waitForSelector('#app canvas', { timeout: 20000 });
  console.log('canvas: present');

  await page.waitForFunction(() => document.getElementById('loading').classList.contains('hidden'), null, { timeout: 20000 });
  console.log('loading overlay: hidden');

  await page.waitForTimeout(2500);

  const fatalVisible = await page.evaluate(() => {
    const f = document.getElementById('fatal');
    return f && getComputedStyle(f).display !== 'none' ? f.textContent : null;
  });
  if (fatalVisible) problems.push(`[fatal] ${fatalVisible}`);

  const stats = await page.evaluate(() => ({
    step: document.getElementById('s-step')?.textContent,
    pop: document.getElementById('s-pop')?.textContent,
    status: document.getElementById('status-line')?.textContent,
    canvases: document.querySelectorAll('canvas').length
  }));
  console.log('HUD:', JSON.stringify(stats));

  const step1 = Number(stats.step);
  await page.waitForTimeout(1500);
  const step2 = Number(await page.evaluate(() => document.getElementById('s-step').textContent));
  console.log(`step advanced: ${step1} -> ${step2}`);
  if (!(step2 > step1)) problems.push(`[anim] simulation not ticking (${step1} -> ${step2})`);
  if (stats.canvases < 1) problems.push('[render] no canvas in #app');

  await page.screenshot({ path: resolve(SHOTS, 'smoke.png'), fullPage: false });
  console.log('screenshot: tests/screenshots/smoke.png');
} catch (err) {
  problems.push(`[exception] ${err.stack || err}`);
} finally {
  if (browser) await browser.close();
}

if (notes.length) {
  console.log(`\n${notes.length} warning(s) (non-fatal):`);
  for (const n of notes) console.log(' ! ' + n);
}
if (problems.length) {
  console.error(`\nFAIL — ${problems.length} problem(s):`);
  for (const p of problems) console.error(' - ' + p);
  process.exit(1);
}
console.log('\nPASS — smoke test clean');
