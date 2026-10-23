// Performance probe: attributes frame time to its three sources.
//
//   render-only  — simulation paused, pure Renderer3D cost
//   stepping     — world.step() + render (PPO idle between rollouts)
//   updating     — chunked PPO update + render (the reported 9 FPS regime)
//
// Method: install a rAF sampler in the page that timestamps every frame and
// records whether ppo.updating was true at that moment, then split the frame
// times by that flag. Run at a given sim speed.
//
// Run (dev server must be up): node tests/perf_probe.mjs [baseUrl]
import { chromium } from 'playwright';

const URL = process.argv[2] || 'http://localhost:5174/';

const SAMPLE = `
new Promise((res) => {
  const frames = [];
  let last = performance.now();
  function tick() {
    const now = performance.now();
    const s = window.__EE.stats();
    frames.push({ dt: now - last, updating: window.__EE.ppo.updating, step: s.step, updates: s.updates });
    last = now;
    if (frames.length < 400) requestAnimationFrame(tick); else res(frames);
  }
  requestAnimationFrame(tick);
})`;

function summarize(frames, label) {
  const upd = frames.filter(f => f.updating).map(f => f.dt);
  const idle = frames.filter(f => !f.updating).map(f => f.dt);
  const mean = a => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
  const stepDelta = frames.length ? (frames[frames.length - 1].step - frames[0].step) : 0;
  const updDelta = frames.length ? (frames[frames.length - 1].updates - frames[0].updates) : 0;
  const dur = frames.length ? frames.reduce((s, f) => s + f.dt, 0) / 1000 : 1;
  console.log(`${label}: frames=${frames.length}  overall ${(1000 / mean(frames.map(f => f.dt))).toFixed(1)} fps`);
  console.log(`   idle   frames=${idle.length}  ${idle.length ? (1000 / mean(idle)).toFixed(1) : '—'} fps  (mean ${(mean(idle) || 0).toFixed(1)} ms)`);
  console.log(`   update frames=${upd.length}  ${upd.length ? (1000 / mean(upd)).toFixed(1) : '—'} fps  (mean ${(mean(upd) || 0).toFixed(1)} ms)`);
  console.log(`   throughput: ${(stepDelta / dur).toFixed(1)} steps/s, ${(updDelta / dur).toFixed(2)} updates/s`);
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto(URL + '?pop=60&speed=1&start=1', { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => window.__EE && window.__EE.stats().step > 5, null, { timeout: 60000 });
await page.waitForTimeout(1500);

// --- render only (paused) ---
await page.evaluate(() => { window.__EE.running = false; });
await page.waitForTimeout(300);
summarize(await page.evaluate(SAMPLE), 'render-only (paused)');

// --- stepping at 1x ---
await page.evaluate(() => { window.__EE.speed = 1; window.__EE.running = true; });
await page.waitForTimeout(300);
summarize(await page.evaluate(SAMPLE), 'speed 1x (step + render)');

// --- stepping + updates at 4x ---
await page.evaluate(() => { window.__EE.speed = 4; });
await page.waitForTimeout(300);
summarize(await page.evaluate(SAMPLE), 'speed 4x (2 PPO chunks/frame + 4 steps)');

// --- stepping + updates at 8x ---
await page.evaluate(() => { window.__EE.speed = 8; });
await page.waitForTimeout(300);
summarize(await page.evaluate(SAMPLE), 'speed 8x (3 PPO chunks/frame + 8 steps)');

await browser.close();
