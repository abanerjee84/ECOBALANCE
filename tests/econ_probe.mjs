// Random-policy baseline probe: what does the economy look like *before* learning?
// Run: node tests/econ_probe.mjs [steps]
import { World } from '../src/sim/world.js';
import { RNG } from '../src/util/rng.js';

const steps = Number(process.argv[2] || 2000);
const N = Number(process.env.N || 60);
const w = new World({ seed: 123, population: N });
const rng = new RNG(5);
const mvx = new Float32Array(N), mvz = new Float32Array(N), acts = new Int32Array(N);

const marks = [250, 500, 1000, 2000, 4000, 8000].filter(s => s <= steps);
let next = 0;
for (let t = 1; t <= steps; t++) {
  for (let i = 0; i < N; i++) {
    mvx[i] = rng.gaussian(0, 0.7); mvz[i] = rng.gaussian(0, 0.7);
    acts[i] = rng.int(5);
  }
  w.step(mvx, mvz, acts);
  if (next < marks.length && t === marks[next]) {
    next++;
    const s = w.stats();
    console.log(`t=${String(t).padStart(5)} | price ${s.price.toFixed(3)} | d/s ${s.demand}/${s.supply} | trades Σ${s.tradesTotal} (${s.volumeEma.toFixed(2)}/step) | dyads ${s.activeTradeEdges} | shares Σ${s.sharesTotal} | flora ${(s.floraFraction * 100).toFixed(1)}% | ore ${(s.oreFraction * 100).toFixed(1)}% | E ${s.meanEnergy.toFixed(1)} | food ${s.meanFood.toFixed(1)} | store ${s.meanStore.toFixed(1)} | gini ${s.gini.toFixed(2)} | starve ${s.deathsStarve} | rew ${s.meanReward.toFixed(4)} | oh ${s.overharvest.toFixed(2)}`);
  }
}
console.log('reward parts EMA:', w.stats().rewardParts.map(v => v.toFixed(4)).join(' '));
