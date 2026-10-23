// Three.js presentation layer: ground heatmap, instanced resources, agents with
// energy bars, trade/share arcs, death markers, selection + follow camera.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ACTIONS, ACTION_NAMES } from '../sim/world.js';

export const ACTION_COLORS = [
  0x59e391, // gather
  0x5ad1ff, // trade
  0xa97bff, // share
  0xffcc5c, // store
  0xff945c, // migrate
];
const ACTION_HEX = ACTION_COLORS.map(c => '#' + c.toString(16).padStart(6, '0'));

const TRAIL_LEN = 44;
const ARC_LIFE = 0.75;
const MAX_ARCS = 96;
const MAX_RINGS = 16;
const MOTES = 340;

export class Renderer3D {
  constructor(container, world, opts = {}) {
    this.container = container;
    this.selected = -1;
    this.follow = false;
    this.arcs = [];        // {x1,z1,x2,z2,t0,kind}
    this.rings = [];       // {x,z,t0}
    this.onSelect = null;
    this.fxMode = ['low', 'high', 'auto'].includes(opts.fx) ? opts.fx : 'auto';

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    container.appendChild(renderer.domElement);
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x0a1626, 110, 340);

    this.camera = new THREE.PerspectiveCamera(48, container.clientWidth / container.clientHeight, 0.1, 900);
    this.camera.position.set(52, 46, 52);

    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 260;

    this.scene.add(new THREE.HemisphereLight(0xa8ccff, 0x1d2a3a, 1.15));
    const sun = new THREE.DirectionalLight(0xfff1d8, 1.7);
    sun.position.set(55, 80, 35);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const s = 110;
    sun.shadow.camera.left = -s; sun.shadow.camera.right = s;
    sun.shadow.camera.top = s; sun.shadow.camera.bottom = -s;
    sun.shadow.camera.far = 260;
    sun.shadow.bias = -0.0012;
    this.scene.add(sun);
    this.sun = sun;
    // cool rim light from the opposite side: depth without extra shadow cost
    const rim = new THREE.DirectionalLight(0x5ad1ff, 0.45);
    rim.position.set(-60, 34, -50);
    this.scene.add(rim);

    this._buildSky();

    // glow pool under each agent + soft radial texture shared by ambience
    this.glowTex = radialTexture();

    // bloom composer (skipped entirely on fx=low)
    this.useBloom = this.fxMode !== 'low';
    if (this.useBloom) this._buildComposer();

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();

    this.setWorld(world);
    this._bindPointer();

    window.addEventListener('resize', () => this.resize());
  }

  /* ---------------- atmosphere ---------------- */

  _buildSky() {
    // dusk gradient dome (world-independent: lives on the scene, not the root)
    const c = document.createElement('canvas');
    c.width = 4; c.height = 256;
    const ctx = c.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, 0, 256);
    g.addColorStop(0.00, '#04060f');
    g.addColorStop(0.42, '#0a1830');
    g.addColorStop(0.68, '#14395a');
    g.addColorStop(0.84, '#1f6280');   // horizon glow band
    g.addColorStop(1.00, '#060a12');
    ctx.fillStyle = g; ctx.fillRect(0, 0, 4, 256);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;

    const dome = new THREE.Mesh(
      new THREE.SphereGeometry(560, 32, 20),
      new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, fog: false, depthWrite: false })
    );
    dome.renderOrder = -2;
    this.scene.add(dome);
    this.skyDome = dome;

    // star field on the upper hemisphere
    const nStars = 420;
    const pos = new Float32Array(nStars * 3);
    const col = new Float32Array(nStars * 3);
    const tmp = new THREE.Color();
    for (let i = 0; i < nStars; i++) {
      const az = Math.random() * Math.PI * 2;
      const el = 0.06 + Math.random() * 1.45;          // above the horizon
      const r = 520 + Math.random() * 20;
      pos[i * 3] = r * Math.cos(el) * Math.sin(az);
      pos[i * 3 + 1] = r * Math.sin(el);
      pos[i * 3 + 2] = r * Math.cos(el) * Math.cos(az);
      tmp.setHSL(0.55 + Math.random() * 0.12, 0.5, 0.62 + Math.random() * 0.3);
      col[i * 3] = tmp.r; col[i * 3 + 1] = tmp.g; col[i * 3 + 2] = tmp.b;
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    sg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const stars = new THREE.Points(sg, new THREE.PointsMaterial({
      size: 1.7, sizeAttenuation: false, vertexColors: true, fog: false,
      transparent: true, opacity: 0.9, depthWrite: false,
    }));
    stars.renderOrder = -1;
    this.scene.add(stars);
    this.stars = stars;
  }

  _buildComposer() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    const composer = new EffectComposer(this.renderer);
    composer.addPass(new RenderPass(this.scene, this.camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.65, 0.82);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
    composer.setPixelRatio(this.renderer.getPixelRatio());
    composer.setSize(w, h);
    this.composer = composer;
    this.bloom = bloom;
  }

  /* ---------------- world → scene bindings ---------------- */

  setWorld(world) {
    this.world = world;
    this.selected = -1;
    this.follow = false;
    this.arcs = [];
    this.rings = [];
    const G = world.G, size = world.cfg.worldSize;

    if (this.root) { this.scene.remove(this.root); disposeGroup(this.root); }
    const root = new THREE.Group();
    this.root = root;
    this.scene.add(root);

    // --- ground with flora-density vertex colours ---
    if (this.ground) { root.remove(this.ground); this.ground.geometry.dispose(); this.ground.material.dispose(); }
    const geo = new THREE.PlaneGeometry(size, size, G, G);
    geo.rotateX(-Math.PI / 2);
    const nVerts = geo.attributes.position.count;
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(nVerts * 3).fill(1), 3));
    const ground = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 1, metalness: 0,
    }));
    ground.receiveShadow = true;
    ground.position.y = -0.02;
    root.add(ground);
    this.ground = ground;

    if (this.grid) { root.remove(this.grid); }
    const grid = new THREE.GridHelper(size, G, 0x4a9fd8, 0x27506f);
    grid.position.y = 0.012;
    grid.material.transparent = true;
    grid.material.opacity = 0.5;
    grid.material.blending = THREE.AdditiveBlending;
    grid.material.depthWrite = false;
    root.add(grid);
    this.grid = grid;

    // glowing world-border frame
    if (this.border) { root.remove(this.border); this.border.geometry.dispose(); this.border.material.dispose(); }
    const h2 = size / 2, bp = [];
    bp.push(-h2, 0.05, -h2, h2, 0.05, -h2, h2, 0.05, -h2, h2, 0.05, h2,
      h2, 0.05, h2, -h2, 0.05, h2, -h2, 0.05, h2, -h2, 0.05, -h2);
    const bg = new THREE.BufferGeometry();
    bg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bp), 3));
    const border = new THREE.LineSegments(bg, new THREE.LineBasicMaterial({
      color: 0x2f8fc4, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    border.renderOrder = 3;
    root.add(border);
    this.border = border;

    // --- flora instances (one per grid cell) ---
    const cell = world.cell;
    const cone = new THREE.ConeGeometry(cell * 0.34, cell * 1.15, 6);
    cone.translate(0, cell * 0.575, 0);
    const floraMat = new THREE.MeshStandardMaterial({
      roughness: 0.8, metalness: 0.0, flatShading: true,
      emissive: 0x0d3318, emissiveIntensity: 0.22,
    });
    const flora = new THREE.InstancedMesh(cone, floraMat, G * G);
    flora.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    flora.castShadow = !this.lowQuality;
    flora.receiveShadow = false;
    flora.frustumCulled = false;
    root.add(flora);
    this.floraMesh = flora;
    this.floraDummy = new THREE.Object3D();
    this.floraColor = new THREE.Color();

    // --- ore instances (one per node) ---
    this._buildOre();

    // --- ambient energy motes drifting over the land ---
    this._buildMotes();

    // --- agents ---
    this._buildAgents();

    // --- event overlays ---
    this._buildOverlays();

    this.syncResources(true);
    this.heatCounter = 0;
  }

  _buildOre() {
    const world = this.world;
    if (this.oreMesh) { this.root.remove(this.oreMesh); this.oreMesh.geometry.dispose(); this.oreMesh.material.dispose(); }
    const geo = new THREE.DodecahedronGeometry(1.5, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xd9ad76, roughness: 0.42, metalness: 0.55, flatShading: true,
      emissive: 0x7a4d1c, emissiveIntensity: 0.85,
    });
    const mesh = new THREE.InstancedMesh(geo, mat, Math.max(1, world.nodes.length));
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.castShadow = true;
    mesh.frustumCulled = false;
    this.root.add(mesh);
    this.oreMesh = mesh;
    this.oreDummy = new THREE.Object3D();
    this.oreColor = new THREE.Color();

    // soft glow disc under each ore node
    const disc = new THREE.CircleGeometry(2.6, 24);
    disc.rotateX(-Math.PI / 2);
    const om = new THREE.InstancedMesh(disc, new THREE.MeshBasicMaterial({
      map: this.glowTex, color: 0xd99a4a, transparent: true, opacity: 0.38,
      blending: THREE.AdditiveBlending, depthWrite: false, fog: false,
    }), Math.max(1, world.nodes.length));
    om.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    om.frustumCulled = false;
    om.renderOrder = 2;
    this.root.add(om);
    this.oreGlow = om;
  }

  _buildMotes() {
    if (this.motes) { this.scene.remove(this.motes); this.motes.geometry.dispose(); this.motes.material.dispose(); }
    const half = this.world.half + 6;
    const pos = new Float32Array(MOTES * 3);
    const seed = new Float32Array(MOTES);
    for (let i = 0; i < MOTES; i++) {
      pos[i * 3] = (Math.random() * 2 - 1) * half;
      pos[i * 3 + 1] = 0.4 + Math.random() * 14;
      pos[i * 3 + 2] = (Math.random() * 2 - 1) * half;
      seed[i] = Math.random() * 100;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      map: this.glowTex, color: 0x7fe3c0, size: 0.9, transparent: true, opacity: 0.55,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    });
    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    pts.renderOrder = 2;
    this.scene.add(pts);
    this.motes = pts;
    this.moteSeed = seed;
  }

  _buildAgents() {
    const world = this.world, N = world.population;
    const bodyGeo = new THREE.CapsuleGeometry(0.34, 0.78, 4, 10);
    bodyGeo.translate(0, 0.79, 0);
    const barBgGeo = new THREE.PlaneGeometry(1.0, 0.14);
    const barFgGeo = new THREE.PlaneGeometry(1.0, 0.14);
    barFgGeo.translate(0.5, 0, 0); // grow from the left edge

    this.agents = [];
    this.bodyMeshes = [];
    for (let i = 0; i < N; i++) {
      const color = ACTION_COLORS[world.lastAction[i]] ?? 0x9fb3c8;
      const mat = new THREE.MeshStandardMaterial({
        color, emissive: color, emissiveIntensity: 0.3, roughness: 0.45, metalness: 0.1,
      });
      const body = new THREE.Mesh(bodyGeo, mat);
      body.castShadow = true;
      body.userData.agent = i;

      const bg = new THREE.Mesh(barBgGeo, new THREE.MeshBasicMaterial({ color: 0x101820, transparent: true, opacity: 0.85, depthTest: false }));
      const fg = new THREE.Mesh(barFgGeo, new THREE.MeshBasicMaterial({ color: 0x59e391, depthTest: false }));
      bg.renderOrder = 5; fg.renderOrder = 6;
      bg.position.y = 1.95; fg.position.y = 1.95;
      bg.position.x = -0.5; fg.position.x = -0.5;

      const g = new THREE.Group();
      g.add(body); g.add(bg); g.add(fg);
      this.root.add(g);

      // additive light-pool on the ground under the agent (bloom feeds on this)
      const poolGeo = new THREE.CircleGeometry(1.15, 18);
      poolGeo.rotateX(-Math.PI / 2);
      const pool = new THREE.Mesh(poolGeo, new THREE.MeshBasicMaterial({
        map: this.glowTex, color, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false, fog: false,
      }));
      pool.position.y = 0.03;
      pool.renderOrder = 2;
      this.root.add(pool);

      this.agents.push({
        g, body, bg, fg, pool,
        trail: new Float32Array(TRAIL_LEN * 2),
        trailN: 0,
        trailHead: 0,
      });
      this.bodyMeshes.push(body);
    }

    // selection ring + trail line
    if (this.selRing) { this.root.remove(this.selRing); this.selRing.geometry.dispose(); }
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.8, 0.07, 6, 28),
      new THREE.MeshBasicMaterial({ color: 0x5ad1ff, transparent: true, opacity: 0.95 })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.05;
    ring.visible = false;
    this.root.add(ring);
    this.selRing = ring;

    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_LEN * 3), 3));
    const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      color: 0x5ad1ff, transparent: true, opacity: 0.55, depthWrite: false,
    }));
    trail.frustumCulled = false;
    trail.visible = false;
    this.root.add(trail);
    this.trail = trail;
  }

  _buildOverlays() {
    // trade / share arcs (rebuilt each frame, additive)
    const mkArcMesh = (color) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_ARCS * 6), 3));
      geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX_ARCS * 6), 3));
      const mat = new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
      });
      const m = new THREE.LineSegments(geo, mat);
      m.frustumCulled = false;
      m.renderOrder = 4;
      this.root.add(m);
      return m;
    };
    this.tradeMesh = mkArcMesh(0x5ad1ff);
    this.shareMesh = mkArcMesh(0xa97bff);

    // death rings
    this.ringPool = [];
    const ringGeo = new THREE.RingGeometry(0.7, 1.0, 24);
    ringGeo.rotateX(-Math.PI / 2);
    for (let i = 0; i < MAX_RINGS; i++) {
      const m = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
        color: 0xff6b6b, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide,
      }));
      m.position.y = 0.08;
      m.visible = false;
      this.root.add(m);
      this.ringPool.push({ m, t0: -1 });
    }
    this.ringCursor = 0;
  }

  _bindPointer() {
    const el = this.renderer.domElement;
    let downX = 0, downY = 0, downT = 0;
    el.addEventListener('pointerdown', (e) => { downX = e.clientX; downY = e.clientY; downT = performance.now(); });
    el.addEventListener('pointerup', (e) => {
      const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
      if (moved > 5 || performance.now() - downT > 500) return; // it was a drag
      const rect = el.getBoundingClientRect();
      this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hits = this.raycaster.intersectObjects(this.bodyMeshes, false);
      const idx = hits.length ? hits[0].object.userData.agent : -1;
      this.setSelected(idx);
      if (this.onSelect) this.onSelect(idx);
    });
  }

  setSelected(i) {
    this.selected = i;
    this.selRing.visible = i >= 0;
    this.trail.visible = i >= 0 && this.agents[i] && this.agents[i].trailN > 1;
    if (i < 0) this.follow = false;
  }

  /**
   * Viewport pixel position of an agent (projected through the camera).
   * Used by the automated tests to click precisely on an agent.
   * @returns {{x:number,y:number,visible:boolean}|null}
   */
  screenPosOfAgent(i) {
    if (!this.world || i < 0 || i >= this.agents.length) return null;
    const v = new THREE.Vector3(this.world.px[i], 0.6, this.world.pz[i]);
    v.project(this.camera);
    const rect = this.renderer.domElement.getBoundingClientRect();
    return {
      x: rect.left + (v.x * 0.5 + 0.5) * rect.width,
      y: rect.top + (-v.y * 0.5 + 0.5) * rect.height,
      visible: v.z < 1 && v.x > -1 && v.x < 1 && v.y > -1 && v.y < 1,
    };
  }

  setFollow(v) { this.follow = v && this.selected >= 0; }

  /* ---------------- per-step ingestion ---------------- */

  pushEvents(events) {
    const t0 = this.elapsed || 0;
    for (const t of events.trades) {
      if (this.arcs.length < MAX_ARCS * 2) this.arcs.push({ x1: t.x, z1: t.z, x2: t.x2, z2: t.z2, kind: 'trade', t0 });
    }
    for (const s of events.shares) {
      if (this.arcs.length < MAX_ARCS * 2) this.arcs.push({ x1: s.x, z1: s.z, x2: s.x2, z2: s.z2, kind: 'share', t0 });
    }
    for (const d of events.deaths) {
      const r = this.ringPool[this.ringCursor++ % this.ringPool.length];
      r.m.position.set(d.x, 0.08, d.z);
      r.m.visible = true;
      r.t0 = t0;
      r.cause = d.cause;
    }
    // record trails for the selected agent each sim step
    if (this.selected >= 0) {
      const a = this.agents[this.selected];
      if (a && this.world) {
        a.trail[a.trailHead * 2] = this.world.px[this.selected];
        a.trail[a.trailHead * 2 + 1] = this.world.pz[this.selected];
        a.trailHead = (a.trailHead + 1) % TRAIL_LEN;
        a.trailN = Math.min(a.trailN + 1, TRAIL_LEN);
        this.trail.visible = true;
      }
    }
  }

  /* ---------------- per-frame sync ---------------- */

  syncResources(force = false) {
    const world = this.world, G = world.G, dummy = this.floraDummy, mesh = this.floraMesh;
    const cell = world.cell, half = world.half;
    for (let cz = 0; cz < G; cz++) {
      for (let cx = 0; cx < G; cx++) {
        const idx = cz * G + cx;
        const b = world.flora[idx];
        const x = (cx + 0.5) * cell - half;
        const z = (cz + 0.5) * cell - half;
        const k = Math.min(b / world.cfg.floraK, 1);
        if (b <= 0.02) {
          dummy.position.set(x, -20, z); dummy.scale.setScalar(0.0001);
        } else {
          dummy.position.set(x, 0, z);
          const sc = 0.25 + 0.75 * k;
          dummy.scale.set(sc, 0.3 + 0.9 * k, sc);
        }
        dummy.rotation.set(0, (idx * 0.37) % 6.28, 0);
        dummy.updateMatrix();
        mesh.setMatrixAt(idx, dummy.matrix);
        this.floraColor.setRGB(0.06 + 0.15 * k, 0.22 + 0.30 * k, 0.10 + 0.13 * k);
        mesh.setColorAt(idx, this.floraColor);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    // ore nodes
    const od = this.oreDummy, om = this.oreMesh;
    for (let i = 0; i < world.nodes.length; i++) {
      const n = world.nodes[i];
      const k = Math.max(n.stock / n.max, 0.001);
      od.position.set(n.x, 0.75 + 0.5 * k, n.z);
      od.scale.setScalar(0.35 + 0.9 * k);
      od.rotation.set(n.x * 0.3, n.z * 0.2, n.x * 0.11);
      od.updateMatrix();
      om.setMatrixAt(i, od.matrix);
      const dep = 1 - k;
      this.oreColor.setRGB(0.76 - 0.25 * dep, 0.60 - 0.3 * dep, 0.42 - 0.2 * dep);
      om.setColorAt(i, this.oreColor);
    }
    om.instanceMatrix.needsUpdate = true;
    if (om.instanceColor) om.instanceColor.needsUpdate = true;

    // ore glow discs (follow node position, fade with stock)
    const og = this.oreGlow;
    if (og) {
      for (let i = 0; i < world.nodes.length; i++) {
        const n = world.nodes[i];
        const k = Math.max(n.stock / n.max, 0.001);
        od.position.set(n.x, 0.06, n.z);
        od.scale.setScalar(0.4 + 0.85 * k);
        od.rotation.set(0, 0, 0);
        od.updateMatrix();
        og.setMatrixAt(i, od.matrix);
      }
      og.instanceMatrix.needsUpdate = true;
    }

    // ground heatmap (cheap: only every few frames)
    this.heatCounter++;
    if (force || this.heatCounter % 8 === 0) {
      const pos = this.ground.geometry.attributes.position;
      const col = this.ground.geometry.attributes.color;
      const n = pos.count;
      for (let v = 0; v < n; v++) {
        const x = pos.getX(v), z = pos.getZ(v);
        let gx = Math.floor((x + half) / cell), gz = Math.floor((z + half) / cell);
        gx = Math.min(Math.max(gx, 0), G - 1); gz = Math.min(Math.max(gz, 0), G - 1);
        const k = Math.min(world.flora[gz * G + gx] / world.cfg.floraK, 1);
        col.setXYZ(v, 0.045 + 0.10 * k, 0.10 + 0.30 * k, 0.14 + 0.10 * k);
      }
      col.needsUpdate = true;
    }
  }

  syncAgents() {
    const world = this.world;
    for (let i = 0; i < this.agents.length; i++) {
      const a = this.agents[i];
      a.g.position.set(world.px[i], 0, world.pz[i]);

      // colour by most recent action
      const ci = world.lastAction[i];
      const hex = ACTION_COLORS[ci] ?? 0x9fb3c8;
      if (a.body.material.color.getHex() !== hex) {
        a.body.material.color.setHex(hex);
        a.body.material.emissive.setHex(hex);
      }

      // energy bar
      const e = Math.max(0, Math.min(world.energy[i] / world.cfg.EMAX, 1));
      a.fg.scale.x = Math.max(0.001, e);
      a.fg.material.color.setRGB(
        Math.min(1, 2 * (1 - e)) * 0.85 + 0.1,
        Math.min(1, 1.6 * e) * 0.9 + 0.05,
        0.2
      );
      a.fg.quaternion.copy(this.camera.quaternion);
      a.bg.quaternion.copy(this.camera.quaternion);

      // starvation dimming + ground light-pool follows position & action colour
      a.body.material.emissiveIntensity = 0.22 + 0.65 * e;
      a.pool.position.x = world.px[i];
      a.pool.position.z = world.pz[i];
      if (a.pool.material.color.getHex() !== hex) a.pool.material.color.setHex(hex);
      a.pool.material.opacity = 0.22 + 0.4 * e;
    }

    if (this.selected >= 0) {
      const i = this.selected;
      this.selRing.position.set(world.px[i], 0.05, world.pz[i]);
      const p = this.trail.geometry.attributes.position;
      const a = this.agents[i];
      for (let k = 0; k < TRAIL_LEN; k++) {
        const src = (a.trailHead + k) % TRAIL_LEN;
        const use = k < a.trailN;
        p.setXYZ(k, use ? a.trail[src * 2] : world.px[i], 0.12, use ? a.trail[src * 2 + 1] : world.pz[i]);
      }
      p.needsUpdate = true;
    }
  }

  update(dt, elapsed) {
    this.elapsed = elapsed;
    this.syncResources();
    this.syncAgents();

    // ambient motes drift upward with a slow sway, wrapping at the ceiling
    if (this.motes) {
      const p = this.motes.geometry.attributes.position;
      const arr = p.array;
      for (let i = 0; i < MOTES; i++) {
        const sd = this.moteSeed[i];
        arr[i * 3 + 1] += dt * (0.35 + 0.25 * Math.sin(sd));
        arr[i * 3] += Math.sin(elapsed * 0.4 + sd) * dt * 0.22;
        if (arr[i * 3 + 1] > 15) arr[i * 3 + 1] = 0.3;
      }
      p.needsUpdate = true;
      this.motes.material.opacity = 0.4 + 0.18 * Math.sin(elapsed * 0.7);
    }

    // border frame slow pulse
    if (this.border) this.border.material.opacity = 0.55 + 0.3 * Math.sin(elapsed * 1.3);

    // fade arcs
    const keep = [];
    for (const arc of this.arcs) {
      if (elapsed - arc.t0 < ARC_LIFE) keep.push(arc);
    }
    this.arcs = keep.slice(-MAX_ARCS * 2);
    this._writeArcs(this.tradeMesh, 'trade', elapsed, 0x2b8fbb);
    this._writeArcs(this.shareMesh, 'share', elapsed, 0x7d4dff);

    // death rings
    for (const r of this.ringPool) {
      if (r.t0 < 0) continue;
      const t = (elapsed - r.t0) / 1.1;
      if (t >= 1) { r.m.visible = false; r.t0 = -1; continue; }
      const s = 0.7 + 3.4 * t;
      r.m.scale.set(s, 1, s);
      r.m.material.opacity = (1 - t) * 0.9;
      r.m.material.color.setHex(r.cause === 'age' ? 0xffcc5c : 0xff6b6b);
    }

    // camera follow
    if (this.follow && this.selected >= 0) {
      const i = this.selected;
      const tx = this.world.px[i], tz = this.world.pz[i];
      const c = this.controls.target;
      const k = Math.min(1, dt * 3.5);
      this.camera.position.x += (tx - c.x) * k;
      this.camera.position.z += (tz - c.z) * k;
      c.x += (tx - c.x) * k;
      c.z += (tz - c.z) * k;
    }

    this.controls.update();
  }

  _writeArcs(mesh, kind, elapsed, baseHex) {
    const pos = mesh.geometry.attributes.position;
    const col = mesh.geometry.attributes.color;
    const base = new THREE.Color(baseHex);
    let w = 0;
    for (const arc of this.arcs) {
      if (arc.kind !== kind || w >= MAX_ARCS) continue;
      const t = (elapsed - arc.t0) / ARC_LIFE;
      const f = Math.max(0, 1 - t) * 0.9;
      const lift = 0.9 + 0.6 * Math.sin(Math.PI * Math.min(1, t + 0.01));
      pos.setXYZ(w * 2, arc.x1, lift, arc.z1);
      pos.setXYZ(w * 2 + 1, arc.x2, lift, arc.z2);
      col.setXYZ(w * 2, base.r * f, base.g * f, base.b * f);
      col.setXYZ(w * 2 + 1, base.r * f, base.g * f, base.b * f);
      w++;
    }
    for (let i = w; i < MAX_ARCS; i++) {
      pos.setXYZ(i * 2, 0, -50, 0);
      pos.setXYZ(i * 2 + 1, 0, -50, 0);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    mesh.geometry.setDrawRange(0, w * 2);
  }

  /**
   * Adaptive quality: if the first ~2.5s of rendering stay below `minFps`
   * (software rasterisers such as SwiftShader, weak integrated GPUs), drop the
   * shadow map to 1024², stop flora instances casting shadows and clamp the
   * pixel ratio to 1. Runs once — capable GPUs keep full quality.
   */
  adaptQuality(minFps = 24, windowMs = 2500) {
    if (this._qualityDone) return;
    const now = performance.now();
    if (this._qT0 === undefined) { this._qT0 = now; this._qFrames = 0; return; }
    this._qFrames++;
    const dt = now - this._qT0;
    if (dt < windowMs) return;
    this._qualityDone = true;
    this.measuredFps = (this._qFrames * 1000) / dt;
    if (this.fxMode === 'high') return;         // effects explicitly requested
    if (this.measuredFps >= minFps) return;
    this.lowQuality = true;
    // software rasterisers can't afford post-processing: drop bloom first
    if (this.useBloom) this.useBloom = false;
    this.sun.shadow.mapSize.set(1024, 1024);
    if (this.sun.shadow.map) { this.sun.shadow.map.dispose(); this.sun.shadow.map = null; }
    this.floraMesh.castShadow = false;
    this.renderer.setPixelRatio(1);
    this.resize();
  }

  render() {
    this.adaptQuality();
    if (this.useBloom && this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    if (this.composer) this.composer.setSize(w, h);
  }

  /** Colour legend for the HUD. */
  static actionColors() { return ACTION_HEX.slice(); }
  static actionNames() { return ACTION_NAMES.slice(); }
}

function disposeGroup(root) {
  root.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
      else o.material.dispose();
    }
  });
}

/** Soft radial-gradient sprite texture used by every glow in the scene. */
function radialTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0.0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.55)');
  g.addColorStop(1.0, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
