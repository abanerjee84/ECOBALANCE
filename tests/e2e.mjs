// Full-stack browser test: HUD, WebGL scene, PPO loop, interactions.
// Run: node tests/e2e.mjs [baseUrl]
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE = process.argv[2] || process.env.BASE_URL || 'http://localhost:5174/';
const SHOTS = resolve(__dirname, 'screenshots');
mkdirSync(SHOTS, { recursive: true });

const problems = [];
const notes = [];
const browser = await chromium.launch({ headless: true });

try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });
  page.on('console', (m) => {
    const t = `[console.${m.type()}] ${m.text()}`;
    if (m.type() === 'error') problems.push(t); else if (m.type() === 'warning') notes.push(t);
  });
  page.on('pageerror', (e) => problems.push(`[pageerror] ${e.message}`));
  page.on('requestfailed', (r) => problems.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText}`));

  const resp = await page.goto(BASE + '?pop=40&speed=4', { waitUntil: 'load', timeout: 30000 });
  console.log(`HTTP ${resp.status()}`);
  if (!resp.ok()) problems.push(`[http] ${resp.status()}`);
  await page.waitForSelector('#app canvas', { timeout: 20000 });
  await page.waitForFunction(() => document.getElementById('loading').classList.contains('hidden'), null, { timeout: 20000 });
  console.log('scene: canvas up, loader hidden');

  const fatalText = await page.evaluate(() => {
    const f = document.getElementById('fatal');
    return f && getComputedStyle(f).display !== 'none' ? f.textContent : null;
  });
  if (fatalText) problems.push(`[fatal] ${fatalText}`);

  // --- let the sim + RL loop run ---
  await page.waitForTimeout(6000);
  const s1 = await page.evaluate(() => window.__EE.stats());
  console.log('t+6s:', JSON.stringify({ step: s1.step, updates: s1.updates, loss: +s1.policyLoss.toFixed(4), nan: s1.nan, reward: +s1.meanReward.toFixed(4) }));
  if (!(s1.step > 0)) problems.push('[loop] no simulation steps taken');
  if (s1.nan) problems.push('[rl] NaN in PPO metrics');
  for (const [k, v] of Object.entries(s1)) {
    if (typeof v === 'number' && !Number.isFinite(v)) problems.push(`[nan] stats.${k} = ${v}`);
  }

  await page.screenshot({ path: resolve(SHOTS, 'e2e-1-early.png') });

  // --- advance the learner deterministically and check PPO updates accumulate ---
  const ff = await page.evaluate(() => window.__EE.fastForward(12));
  console.log('fastForward:', JSON.stringify(ff));
  if (!(ff.updates >= 5)) problems.push(`[rl] expected >=5 PPO updates, got ${ff.updates}`);
  const s2 = await page.evaluate(() => window.__EE.stats());
  console.log('after FF:', JSON.stringify({ step: s2.step, updates: s2.updates, pol: +s2.policyLoss.toFixed(4), val: +s2.valueLoss.toFixed(4), ent: +s2.entropy.toFixed(3), kl: +s2.approxKl.toFixed(4), ev: +s2.explainedVar.toFixed(3), nan: s2.nan }));
  if (s2.nan) problems.push('[rl] NaN after fast-forward');
  if (!(s2.entropy > 0.05)) problems.push(`[rl] entropy collapsed: ${s2.entropy}`);
  if (!(Math.abs(s2.approxKl) < 0.5)) problems.push(`[rl] approx KL out of range: ${s2.approxKl}`);

  // --- HUD content ---
  const hud = await page.evaluate(() => {
    const txt = (id) => document.getElementById(id)?.textContent?.trim();
    const rows = (id) => Array.from(document.querySelectorAll(`#${id} .row`)).map(r => r.textContent.trim());
    return {
      status: txt('status-line'),
      eq: txt('rm-eq'),
      sim: rows('sim-rows'),
      ppo: rows('ppo-rows'),
      reward: rows('rm-rows'),
      emerge: rows('em-rows'),
      charts: ['c-mort', 'c-res', 'c-mkt', 'c-ineq', 'c-learn'].map(id => document.getElementById(id).width),
      rollout: document.getElementById('rollout-bar').style.width,
      story: document.getElementById('story-lines')?.innerText?.trim() || '',
      cadence: Array.from(document.querySelectorAll('#ppo-rows .row')).map(r => r.textContent).find(t => t.includes('Learn every')) || '',
    };
  });
  console.log('status:', hud.status);
  console.log('reward eq:', hud.eq);
  const nonEmpty = (arr, name) => arr.forEach((r, i) => { if (r.includes('—')) problems.push(`[hud] ${name}[${i}] not populated: ${r}`); });
  nonEmpty(hud.sim, 'sim-rows');
  nonEmpty(hud.ppo, 'ppo-rows');
  nonEmpty(hud.reward, 'rm-rows');
  if (!hud.eq || !hud.eq.includes('r')) problems.push('[hud] reward equation missing');
  console.log('charts widths:', hud.charts.join(','), '| rollout bar:', hud.rollout);
  console.log('story:', JSON.stringify(hud.story.slice(0, 160)));
  if (!hud.story || hud.story.length < 40) problems.push('[hud] plain-English story panel empty');
  if (!hud.cadence.includes('32')) problems.push(`[hud] PPO cadence row missing/wrong: ${hud.cadence}`);
  else console.log('cadence:', hud.cadence);

  // --- guide overlay toggles ---
  await page.click('#btn-guide');
  const guideOpen = await page.evaluate(() => getComputedStyle(document.getElementById('p-guide')).display !== 'none');
  if (!guideOpen) problems.push('[ui] guide button did not open the guide');
  await page.keyboard.press('Escape');
  const guideClosed = await page.evaluate(() => getComputedStyle(document.getElementById('p-guide')).display === 'none');
  if (!guideClosed) problems.push('[ui] Escape did not close the guide');
  await page.screenshot({ path: resolve(SHOTS, 'e2e-2-populated.png') });

  // --- click an agent (projected screen position) then verify the inspector ---
  // pause first: at 4x speed the agent moves far enough between the position
  // probe and the physical click to make the raycast miss its small body
  await page.evaluate(() => { window.__EE.running = false; });
  await page.waitForTimeout(200);
  const box = await page.locator('#app canvas').boundingBox();
  const pick = await page.evaluate(() => {
    // choose the visible agent closest to the viewport centre whose projected
    // position would actually deliver the click to the canvas (side HUD panels
    // overlay the canvas edges and would swallow a click otherwise)
    const w = window.__EE.world;
    const cv = document.querySelector('#app canvas');
    let best = null;
    for (let i = 0; i < w.population; i++) {
      const p = window.__EE.screenPosOfAgent(i);
      if (!p || !p.visible) continue;
      const hit = document.elementFromPoint(p.x, p.y);
      if (!hit || (hit !== cv && !cv.contains(hit))) continue;
      const d = (p.x - innerWidth / 2) ** 2 + (p.y - innerHeight / 2) ** 2;
      if (!best || d < best.d) best = { i, x: p.x, y: p.y, d };
    }
    return best;
  });
  if (pick) {
    console.log(`clicking agent ${pick.i} at (${pick.x.toFixed(0)}, ${pick.y.toFixed(0)})`);
    await page.mouse.click(pick.x, pick.y);
  }
  await page.waitForTimeout(400);
  let inspector = await page.evaluate(() => {
    const p = document.getElementById('p-agent');
    return { visible: getComputedStyle(p).display !== 'none', title: document.getElementById('a-title').textContent };
  });
  if (!inspector.visible) {
    notes.push('center click missed an agent; trying a grid of clicks');
    outer: for (let fy = 0.35; fy <= 0.75; fy += 0.1) {
      for (let fx = 0.3; fx <= 0.7; fx += 0.1) {
        await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
        await page.waitForTimeout(150);
        inspector = await page.evaluate(() => {
          const p = document.getElementById('p-agent');
          return { visible: getComputedStyle(p).display !== 'none', title: document.getElementById('a-title').textContent };
        });
        if (inspector.visible) break outer;
      }
    }
  }
  if (!inspector.visible) problems.push('[ui] clicking an agent did not open the inspector');
  else {
    console.log('inspector:', inspector.title);
    const probs = await page.evaluate(() => Array.from(document.querySelectorAll('#a-probs .p')).map(e => e.textContent));
    console.log('action probs:', probs.join(' '));
    const sum = probs.map(p => parseFloat(p)).reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
    if (Math.abs(sum - 100) > 6) problems.push(`[ui] action probabilities sum to ${sum.toFixed(0)}%`);
    await page.screenshot({ path: resolve(SHOTS, 'e2e-3-agent.png') });
  }

  // resume (the click test paused the sim; the pause-button test below needs it running)
  await page.evaluate(() => { window.__EE.running = true; });

  // --- controls: pause / speed / config ---
  await page.click('#btn-pause');
  const paused = await page.evaluate(() => window.__EE.running);
  const stepA = await page.evaluate(() => window.__EE.stats().step);
  await page.waitForTimeout(700);
  const stepB = await page.evaluate(() => window.__EE.stats().step);
  console.log(`pause: running=${paused} step ${stepA} -> ${stepB}`);
  if (paused !== false || stepA !== stepB) problems.push('[ui] pause did not freeze the simulation');
  await page.click('#btn-pause');

  await page.click('#btn-cfg');
  await page.waitForTimeout(150);
  const cfg = await page.evaluate(() => {
    const p = document.getElementById('p-config');
    return { visible: getComputedStyle(p).display !== 'none', inputs: document.querySelectorAll('#cfg-rows input').length };
  });
  console.log('config:', JSON.stringify(cfg));
  if (!cfg.visible || cfg.inputs < 12) problems.push(`[ui] config panel incomplete: ${JSON.stringify(cfg)}`);

  // speed buttons
  await page.click('[data-sp="1"]');
  const spd = await page.evaluate(() => window.__EE.speed);
  if (spd !== 1) problems.push(`[ui] speed button failed (speed=${spd})`);
  await page.click('[data-sp="8"]');

  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  const escVisible = await page.evaluate(() => getComputedStyle(document.getElementById('p-agent')).display !== 'none');
  if (escVisible) problems.push('[ui] Escape did not deselect');

  // --- reset keeps things alive ---
  await page.click('#btn-reset');
  await page.waitForTimeout(800);
  const after = await page.evaluate(() => window.__EE.stats());
  console.log('after reset:', JSON.stringify({ step: after.step, updates: after.updates, pop: after.population }));
  if (after.step > 500) problems.push('[reset] simulation step did not restart');
  const fatalAfterReset = await page.evaluate(() => {
    const f = document.getElementById('fatal');
    return f && getComputedStyle(f).display !== 'none' ? f.textContent : null;
  });
  if (fatalAfterReset) problems.push(`[fatal after reset] ${fatalAfterReset}`);
  await page.screenshot({ path: resolve(SHOTS, 'e2e-4-after-reset.png') });

  // smooth FPS sample at speed 4
  await page.click('[data-sp="4"]');
  await page.waitForTimeout(1500);
  const fps = await page.evaluate(() => {
    const el = document.getElementById('sim-rows');
    const rows = Array.from(el.querySelectorAll('.row')).map(r => r.textContent);
    return rows.find(r => r.includes('FPS')) || '';
  });
  console.log('fps row:', fps);
} catch (err) {
  problems.push(`[exception] ${err.stack || err}`);
} finally {
  await browser.close();
}

if (notes.length) { console.log(`\n${notes.length} note(s):`); for (const n of notes.slice(0, 8)) console.log(' ! ' + n); }
if (problems.length) {
  console.error(`\nFAIL — ${problems.length} problem(s):`);
  for (const p of problems) console.error(' - ' + p);
  process.exit(1);
}
console.log('\nPASS — e2e');
