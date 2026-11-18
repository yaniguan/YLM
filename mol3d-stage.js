/* Volumetric high-DPI 3D molecular, vector, and tensor stage for YLM. */
(() => {
  'use strict';

  // Refined desaturated scientific CPK palette aligned with the obsidian-sage design system
  const ELEMENT_STYLE = {
    1:  { sym: 'H',  r: 0.30, hi: '#ffffff', mid: '#d6e1ec', lo: '#6d7f91' },
    3:  { sym: 'Li', r: 0.50, hi: '#eadff9', mid: '#b9a1de', lo: '#574378' },
    6:  { sym: 'C',  r: 0.44, hi: '#9eb1c4', mid: '#566778', lo: '#26313c' },
    7:  { sym: 'N',  r: 0.43, hi: '#d3e4f7', mid: '#91b6df', lo: '#3c5878' },
    8:  { sym: 'O',  r: 0.43, hi: '#f9c5b1', mid: '#d97757', lo: '#6e321e' },
    9:  { sym: 'F',  r: 0.42, hi: '#e8f8d5', mid: '#b5d88e', lo: '#4d6b34' },
    15: { sym: 'P',  r: 0.52, hi: '#f7d8a8', mid: '#d9a566', lo: '#6e4d22' },
    16: { sym: 'S',  r: 0.51, hi: '#f7eea8', mid: '#d8c56a', lo: '#6b5e22' },
    17: { sym: 'Cl', r: 0.49, hi: '#c9f2ef', mid: '#79c2bd', lo: '#2f6360' },
    26: { sym: 'Fe', r: 0.54, hi: '#f2bfa8', mid: '#c96f53', lo: '#5e2b1b' },
    35: { sym: 'Br', r: 0.53, hi: '#ecc6ba', mid: '#b86b58', lo: '#572a20' }
  };

  function elemStyle(z) {
    return ELEMENT_STYLE[z] || { sym: `Z${z}`, r: 0.45, hi: '#d6e1ec', mid: '#7f93a8', lo: '#334252' };
  }

  function rotMatrixEuler(rx, ry, rz, reflectX = false) {
    const cx = Math.cos(rx), sx = Math.sin(rx);
    const cy = Math.cos(ry), sy = Math.sin(ry);
    const cz = Math.cos(rz), sz = Math.sin(rz);
    const s = reflectX ? -1 : 1;
    return [
      [s * cy * cz, -cy * sz, sy],
      [s * (sx * sy * cz + cx * sz), -sx * sy * sz + cx * cz, -sx * cy],
      [s * (-cx * sy * cz + sx * sz), cx * sy * sz + sx * cz, cx * cy]
    ];
  }

  function applyMat(M, v) {
    return [
      M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
      M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
      M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]
    ];
  }

  function vecSub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function vecAdd(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function vecScale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
  function vecNorm(a) { return Math.hypot(a[0], a[1], a[2]); }

  function inferBonds(atoms) {
    const bonds = [];
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const zi = atoms[i].Z ?? atoms[i].z ?? 6;
        const zj = atoms[j].Z ?? atoms[j].z ?? 6;
        const d = vecNorm(vecSub(atoms[i].pos, atoms[j].pos));
        const maxD = (zi === 1 && zj === 1) ? 0.95 : ((zi === 1 || zj === 1) ? 1.32 : ((zi === 3 || zj === 3 || zi === 26 || zj === 26) ? 2.45 : 1.88));
        if (d > 0.32 && d <= maxD) bonds.push([i, j, d]);
      }
    }
    return bonds;
  }

  function projectPoint(p, cx, cy, scale, camDist = 11.5) {
    const persp = camDist / Math.max(3.5, camDist - p[2]);
    return { x: cx + p[0] * scale * persp, y: cy - p[1] * scale * persp, z: p[2], p: persp };
  }

  function drawShadedAtom(ctx, pt, radiusPx, style, alpha = 1.0, attnWeight = 0.0) {
    ctx.save();
    ctx.globalAlpha = alpha;

    // Optional token-to-atom routing halo
    if (attnWeight > 0.16) {
      const haloR = radiusPx * (1.55 + 0.45 * attnWeight);
      const hg = ctx.createRadialGradient(pt.x, pt.y, radiusPx * 0.65, pt.x, pt.y, haloR);
      hg.addColorStop(0, `rgba(181, 216, 142, ${(0.48 * attnWeight).toFixed(3)})`);
      hg.addColorStop(0.65, `rgba(121, 194, 189, ${(0.20 * attnWeight).toFixed(3)})`);
      hg.addColorStop(1, 'rgba(181, 216, 142, 0)');
      ctx.fillStyle = hg;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, haloR, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = `rgba(181, 216, 142, ${(0.35 + 0.55 * attnWeight).toFixed(3)})`;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radiusPx * 1.38, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Soft ambient drop shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.36)';
    ctx.beginPath();
    ctx.arc(pt.x + radiusPx * 0.12, pt.y + radiusPx * 0.18, radiusPx * 1.02, 0, Math.PI * 2);
    ctx.fill();

    // Subsurface/specular radial sphere shader
    const grad = ctx.createRadialGradient(
      pt.x - radiusPx * 0.34, pt.y - radiusPx * 0.36, radiusPx * 0.08,
      pt.x, pt.y, radiusPx
    );
    grad.addColorStop(0, style.hi);
    grad.addColorStop(0.46, style.mid);
    grad.addColorStop(1, style.lo);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radiusPx, 0, Math.PI * 2);
    ctx.fill();

    // Subtle Fresnel rim
    ctx.strokeStyle = 'rgba(237, 241, 247, 0.22)';
    ctx.lineWidth = 0.9;
    ctx.stroke();
    ctx.restore();
  }

  function drawDualBond(ctx, p1, p2, s1, s2, widthPx, dashed = false) {
    const mx = (p1.x + p2.x) * 0.5;
    const my = (p1.y + p2.y) * 0.5;
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;

    ctx.save();
    ctx.lineCap = 'round';
    if (dashed) ctx.setLineDash([4, 4]);

    for (const [a, b, col] of [[p1, { x: mx, y: my }, s1.mid], [{ x: mx, y: my }, p2, s2.mid]]) {
      ctx.strokeStyle = 'rgba(7, 11, 16, 0.65)';
      ctx.lineWidth = widthPx + 2.0;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();

      const bg = ctx.createLinearGradient(
        a.x + nx * widthPx * 0.5, a.y + ny * widthPx * 0.5,
        a.x - nx * widthPx * 0.5, a.y - ny * widthPx * 0.5
      );
      bg.addColorStop(0, '#c8d6e5');
      bg.addColorStop(0.42, col);
      bg.addColorStop(1, '#1c2631');
      ctx.strokeStyle = bg;
      ctx.lineWidth = widthPx;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function draw3DVectorArrow(ctx, pStart, pEnd, color, label, dashed = false) {
    const dx = pEnd.x - pStart.x;
    const dy = pEnd.y - pStart.y;
    const len = Math.hypot(dx, dy);
    if (len < 4) return;
    const ux = dx / len, uy = dy / len;
    const nx = -uy, ny = ux;
    const headLen = Math.min(16, Math.max(10, len * 0.24));
    const headHalf = headLen * 0.44;
    const bx = pEnd.x - ux * headLen;
    const by = pEnd.y - uy * headLen;

    ctx.save();
    // Subtle glow behind vector shaft
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.22;
    ctx.lineWidth = 7.5;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(pStart.x, pStart.y);
    ctx.lineTo(bx, by);
    ctx.stroke();

    ctx.globalAlpha = 1.0;
    if (dashed) ctx.setLineDash([5, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3.0;
    ctx.beginPath();
    ctx.moveTo(pStart.x, pStart.y);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);

    // Lit 3D cone head
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(pEnd.x, pEnd.y);
    ctx.lineTo(bx + nx * headHalf, by + ny * headHalf);
    ctx.lineTo(bx - nx * headHalf, by - ny * headHalf);
    ctx.closePath();
    ctx.fill();

    if (label) {
      ctx.font = '600 11.5px Inter, "Segoe UI", sans-serif';
      const tw = ctx.measureText(label).width;
      const lx = pEnd.x + ux * 8;
      const ly = pEnd.y + uy * 8;
      ctx.fillStyle = 'rgba(10, 15, 21, 0.82)';
      ctx.strokeStyle = 'rgba(145, 182, 223, 0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(lx - tw / 2 - 6, ly - 9, tw + 12, 18, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, lx, ly);
    }
    ctx.restore();
  }

  function drawTensorEllipsoid(ctx, center3D, axes3D, radii, cx, cy, scale, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.35;
    const pairs = [[0, 1], [1, 2], [2, 0]];
    pairs.forEach(([ia, ib], idx) => {
      ctx.globalAlpha = idx === 0 ? 0.52 : 0.32;
      ctx.beginPath();
      const steps = 64;
      for (let k = 0; k <= steps; k++) {
        const th = (2 * Math.PI * k) / steps;
        const c = Math.cos(th) * radii[ia];
        const s = Math.sin(th) * radii[ib];
        const p3 = [
          center3D[0] + axes3D[ia][0] * c + axes3D[ib][0] * s,
          center3D[1] + axes3D[ia][1] * c + axes3D[ib][1] * s,
          center3D[2] + axes3D[ia][2] * c + axes3D[ib][2] * s
        ];
        const pt = projectPoint(p3, cx, cy, scale);
        if (k === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
    });
    ctx.restore();
  }

  function renderViewport(ctx, rect, scene, viewMat, opts = {}) {
    const { x, y, w, h } = rect;
    const cx = x + w * 0.5;
    const cy = y + h * 0.52;

    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();

    // Subtle atmospheric spotlight
    const bgGrad = ctx.createRadialGradient(cx, cy - h * 0.06, 12, cx, cy, Math.max(w, h) * 0.65);
    bgGrad.addColorStop(0, opts.tintCenter || '#14202b');
    bgGrad.addColorStop(1, '#0b1118');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(x, y, w, h);

    const atoms = scene.atoms || [];
    if (!atoms.length) {
      ctx.restore();
      return;
    }

    // Center coordinates
    let mean = [0, 0, 0];
    for (const a of atoms) {
      mean[0] += a.pos[0]; mean[1] += a.pos[1]; mean[2] += a.pos[2];
    }
    mean = mean.map(v => v / atoms.length);

    let maxSpan = 1.6;
    const centered = atoms.map(a => {
      const c = vecSub(a.pos, mean);
      maxSpan = Math.max(maxSpan, vecNorm(c) + 0.75);
      return { ...a, cpos: c };
    });

    const scale = Math.min(w, h) * 0.34 / maxSpan;

    // Soft ground shadow ellipse
    ctx.fillStyle = 'rgba(0, 0, 0, 0.32)';
    ctx.beginPath();
    ctx.ellipse(cx, y + h * 0.83, Math.min(w * 0.28, 110), 11, 0, 0, Math.PI * 2);
    ctx.fill();

    // Transform atoms by viewMat
    const rotated = centered.map((a, idx) => {
      const r3 = applyMat(viewMat, a.cpos);
      const pt = projectPoint(r3, cx, cy, scale);
      const zNum = a.Z ?? a.z ?? 6;
      return { idx, zNum, style: elemStyle(zNum), r3, pt, attn: (scene.attn && scene.attn[idx]) || 0 };
    });

    // Optional tensor ellipsoid
    if (scene.ellipsoid) {
      const axesRot = scene.ellipsoid.axes.map(ax => applyMat(viewMat, ax));
      drawTensorEllipsoid(ctx, [0, 0, 0], axesRot, scene.ellipsoid.radii, cx, cy, scale, scene.ellipsoid.color || '#79c2bd');
    }

    const bonds = scene.bonds || inferBonds(centered.map(a => ({ Z: a.Z ?? a.z ?? 6, pos: a.cpos })));
    const primitives = [];

    for (const [i, j, dist] of bonds) {
      if (!rotated[i] || !rotated[j]) continue;
      primitives.push({
        kind: 'bond',
        z: (rotated[i].r3[2] + rotated[j].r3[2]) * 0.5 - 0.05,
        a: rotated[i],
        b: rotated[j],
        dashed: dist > 1.72
      });
    }

    for (const item of rotated) {
      primitives.push({ kind: 'atom', z: item.r3[2], item });
    }

    primitives.sort((p, q) => p.z - q.z);

    for (const prim of primitives) {
      if (prim.kind === 'bond') {
        const wPx = Math.max(3.2, scale * 0.13 * ((prim.a.pt.p + prim.b.pt.p) * 0.5));
        drawDualBond(ctx, prim.a.pt, prim.b.pt, prim.a.style, prim.b.style, wPx, prim.dashed);
      } else {
        const it = prim.item;
        const rPx = Math.max(6.5, it.style.r * scale * it.pt.p);
        const depthAlpha = Math.max(0.72, Math.min(1.0, 0.90 + it.r3[2] * 0.05));
        drawShadedAtom(ctx, it.pt, rPx, it.style, depthAlpha, it.attn);
      }
    }

    // Draw 3D vectors on top with clear contrast
    for (const arr of (scene.arrows || [])) {
      const v3 = arr.vec || arr.vector || [0, 0, 0];
      const mag = vecNorm(v3);
      if (mag < 1e-8) continue;
      const originC = arr.origin_atom !== undefined && centered[arr.origin_atom]
        ? centered[arr.origin_atom].cpos
        : (arr.origin ? vecSub(arr.origin, mean) : [0, 0, 0]);
      const normLen = Math.min(maxSpan * 0.88, Math.max(maxSpan * 0.52, mag * (arr.scale || 1.15)));
      const tipC = vecAdd(originC, vecScale(v3, normLen / mag));
      const p0 = projectPoint(applyMat(viewMat, originC), cx, cy, scale);
      const p1 = projectPoint(applyMat(viewMat, tipC), cx, cy, scale);
      draw3DVectorArrow(ctx, p0, p1, arr.color || '#b5d88e', arr.label || '', Boolean(arr.dashed));
    }

    ctx.restore();
  }

  // Interactive orbitable 3D canvas helper for the Studio tabs
  function mountScene(canvas, sceneData, options = {}) {
    if (!canvas) return { update() {} };
    let scene = sceneData || { atoms: [], arrows: [] };
    let yaw = options.yaw ?? 0.42;
    let pitch = options.pitch ?? -0.26;
    let dragging = false;
    let lastX = 0, lastY = 0;

    function draw() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(220, Math.round(rect.width || 360));
      const h = Math.max(180, Math.round(rect.height || 260));
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const viewMat = rotMatrixEuler(pitch, yaw, 0, false);
      renderViewport(ctx, { x: 0, y: 0, w, h }, scene, viewMat, options);
    }

    if (!canvas.__ylmBound) {
      canvas.__ylmBound = true;
      canvas.addEventListener('pointerdown', e => {
        dragging = true; lastX = e.clientX; lastY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove', e => {
        if (!dragging) return;
        yaw += (e.clientX - lastX) * 0.011;
        pitch = Math.max(-1.45, Math.min(1.45, pitch + (e.clientY - lastY) * 0.011));
        lastX = e.clientX; lastY = e.clientY;
        draw();
      });
      const stop = () => { dragging = false; };
      canvas.addEventListener('pointerup', stop);
      canvas.addEventListener('pointercancel', stop);
      window.addEventListener('resize', () => requestAnimationFrame(draw), { passive: true });
    }

    canvas.__ylmUpdate = nextScene => {
      scene = nextScene || { atoms: [], arrows: [] };
      draw();
    };
    canvas.__ylmUpdate(scene);
    return { update: canvas.__ylmUpdate, redraw: draw };
  }

  // Hero 5-Chapter Cinematic Split-Stage (#hero-stage-canvas)
  function initHeroShowcase() {
    const canvas = document.getElementById('hero-stage-canvas');
    if (!canvas) return;

    // Accurate (R)-fluoroethylene carbonate 3D geometry (Angstrom)
    const R_FEC_ATOMS = [
      { Z: 6, pos: [ 0.00,  0.96,  0.04] }, // 0: C=O carbon
      { Z: 8, pos: [ 0.00,  2.16,  0.09] }, // 1: carbonyl O
      { Z: 8, pos: [-1.11,  0.22,  0.02] }, // 2: ring O
      { Z: 8, pos: [ 1.11,  0.22, -0.02] }, // 3: ring O
      { Z: 6, pos: [-0.74, -1.16, -0.14] }, // 4: CH2 carbon
      { Z: 6, pos: [ 0.76, -1.14,  0.16] }, // 5: CHF stereocenter!
      { Z: 9, pos: [ 1.38, -1.96, -0.74] }, // 6: F substituent
      { Z: 1, pos: [ 1.04, -1.46,  1.17] }, // 7: H on stereocenter
      { Z: 1, pos: [-1.29, -1.78,  0.56] }, // 8: H
      { Z: 1, pos: [-0.96, -1.50, -1.15] }  // 9: H
    ];

    const WATER_ATOMS = [
      { Z: 8, pos: [ 0.00,  0.00, 0.00] },
      { Z: 1, pos: [ 0.76,  0.59, 0.00] },
      { Z: 1, pos: [-0.76,  0.59, 0.00] }
    ];

    const CHAPTERS = [
      {
        key: 'overview',
        title: '00 · Overview — Equivariant vs Distance-Only Readout',
        prompt: '“What is the 3D molecular dipole moment vector of (R)-fluoroethylene carbonate?”',
        ylmMetric: 'YLM O(3) error: 2.4 × 10⁻¹⁶',
        baseMetric: '0e Distance Baseline: vector frozen in lab frame'
      },
      {
        key: 'rotation',
        title: '01 · SO(3) Rotation — Per-Atom Force Vector (1o)',
        prompt: '“What is the force vector acting on the carbonyl oxygen atom?”',
        ylmMetric: 'f(X Rᵀ) = f(X) Rᵀ  (cos = +1.000)',
        baseMetric: 'd_ij unchanged → output never rotates'
      },
      {
        key: 'mirror',
        title: '02 · Mirror Parity — (R) vs (S) Enantiomers (0o)',
        prompt: '“Predict the chiral handedness pseudoscalar of this stereoisomer.”',
        ylmMetric: 'YLM 0o: +0.842 (R)  ⇄  −0.842 (S)',
        baseMetric: '0e Baseline: identical +0.193 on both twins'
      },
      {
        key: 'routing',
        title: '03 · Language-to-Atom Routing — Invariant Attention α_ti',
        prompt: 'Switching question wording shifts α_ti across O(carbonyl) → F(chiral) → C(chiral)',
        ylmMetric: 'Routing invariance drift: 0.0 × 10⁻¹⁶',
        baseMetric: 'Task-ID without text: 0.113 force cosine'
      },
      {
        key: 'tensor',
        title: '04 · Symmetric 3×3 Tensor (0e ⊕ 2e) & Curie Planar Zero',
        prompt: '“Predict the 3×3 quadrupole tensor and chiral output of planar H₂O (C₂v).”',
        ylmMetric: 'Q M Qᵀ exact · Planar |0o| = 3.9 × 10⁻¹⁶',
        baseMetric: 'Unconstrained MLP violates C₂v planar zero'
      }
    ];

    let chapterIdx = 0;
    let playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let tSec = 0;
    let lastTs = performance.now();
    let userYaw = 0, userPitch = 0;
    let dragging = false, lx = 0, ly = 0;

    const promptEl = document.getElementById('hero-stage-prompt');
    const toggleBtn = document.getElementById('hero-stage-toggle');
    const chapterBtns = Array.from(document.querySelectorAll('[data-hero-chapter]'));

    function setChapter(idx) {
      chapterIdx = idx;
      tSec = 0;
      chapterBtns.forEach((b, i) => b.classList.toggle('active', i === idx));
      if (promptEl) promptEl.textContent = CHAPTERS[idx].prompt;
    }

    chapterBtns.forEach((btn, idx) => {
      btn.addEventListener('click', () => {
        setChapter(idx);
        playing = true;
        updateToggleUI();
      });
    });

    function updateToggleUI() {
      if (!toggleBtn) return;
      toggleBtn.textContent = playing ? 'Ⅱ Pause' : '▶ Play';
      toggleBtn.setAttribute('aria-label', playing ? 'Pause live 3D simulation' : 'Play live 3D simulation');
    }

    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        playing = !playing;
        updateToggleUI();
      });
    }

    canvas.addEventListener('pointerdown', e => {
      dragging = true; lx = e.clientX; ly = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      if (!dragging) return;
      userYaw += (e.clientX - lx) * 0.01;
      userPitch = Math.max(-1.1, Math.min(1.1, userPitch + (e.clientY - ly) * 0.01));
      lx = e.clientX; ly = e.clientY;
    });
    const endDrag = () => { dragging = false; };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    function renderFrame(now) {
      const dt = Math.min(0.05, (now - lastTs) / 1000);
      lastTs = now;
      if (playing && !document.hidden) tSec += dt;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(320, Math.round(rect.width || 960));
      const h = Math.max(240, Math.round(rect.height || 520));
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }

      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const ch = CHAPTERS[chapterIdx];
      const rx = -0.28 + 0.34 * Math.sin(tSec * 0.65) + userPitch;
      const ry = tSec * 0.72 + userYaw;
      const rz = 0.22 * Math.cos(tSec * 0.48);
      const isMirrorFlip = chapterIdx === 2 && (Math.sin(tSec * 1.15) < 0);
      const R = rotMatrixEuler(rx, ry, rz, isMirrorFlip);
      const I3 = rotMatrixEuler(-0.22, 0.35, 0, false);

      const halfW = Math.floor(w / 2);
      const isWater = chapterIdx === 4;
      const baseAtoms = isWater ? WATER_ATOMS : R_FEC_ATOMS;

      // Transform molecular coordinates explicitly by R so both panels show the exact same rotated molecule!
      const transformedAtoms = baseAtoms.map(a => ({ Z: a.Z, pos: applyMat(R, a.pos) }));

      let attn = new Array(baseAtoms.length).fill(0);
      let baseVec = [0.0, -1.35, 0.22];
      let originIdx = undefined;
      let ellipsoidYLM = null;
      let ellipsoidBase = null;

      if (chapterIdx === 0) {
        baseVec = [-0.25, -1.45, 0.30];
        ellipsoidYLM = {
          axes: [applyMat(R, [1, 0, 0]), applyMat(R, [0, 1, 0]), applyMat(R, [0, 0, 1])],
          radii: [2.05, 2.55, 1.65],
          color: '#79c2bd'
        };
      } else if (chapterIdx === 1) {
        attn[1] = 0.96;
        originIdx = 1;
        baseVec = [0.12, -1.38, 0.24];
      } else if (chapterIdx === 2) {
        attn[5] = 0.94;
        attn[6] = 0.55;
        originIdx = 5;
        baseVec = [0.95, -0.75, -0.55];
      } else if (chapterIdx === 3) {
        const slot = Math.floor(tSec / 2.2) % 3;
        const pick = [1, 6, 5][slot];
        attn[pick] = 0.95;
        originIdx = pick;
        baseVec = slot === 0 ? [0.0, -1.4, 0.2] : (slot === 1 ? [-1.1, 0.85, 0.45] : [-0.6, 0.95, -0.7]);
        if (promptEl) {
          promptEl.textContent = [
            '“What is the force vector on the carbonyl oxygen atom?”  →  α(O1) = 0.95',
            '“What is the force vector on the fluorine substituent?”  →  α(F6) = 0.95',
            '“What is the force vector on the chiral carbon stereocenter?”  →  α(C5) = 0.95'
          ][slot];
        }
      } else if (chapterIdx === 4) {
        attn[0] = 0.85;
        baseVec = [0.0, 1.15, 0.0];
        ellipsoidYLM = {
          axes: [applyMat(R, [1, 0, 0]), applyMat(R, [0, 1, 0]), applyMat(R, [0, 0, 1])],
          radii: [1.55, 1.18, 0.86],
          color: '#b5d88e'
        };
        ellipsoidBase = {
          axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          radii: [1.55, 1.18, 0.86],
          color: '#d97757'
        };
      }

      const ylmRotVec = applyMat(R, baseVec);
      const chiralLabelYLM = chapterIdx === 2 ? (isMirrorFlip ? '(S) 0o = −0.842' : '(R) 0o = +0.842') : 'YLM (exact O(3))';
      const chiralLabelBase = chapterIdx === 2 ? '0e = +0.193 (unchanged)' : '0e Baseline (frozen)';

      renderViewport(ctx, { x: 0, y: 0, w: halfW, h }, {
        atoms: transformedAtoms,
        attn,
        ellipsoid: ellipsoidYLM,
        arrows: [
          {
            origin_atom: originIdx,
            vec: ylmRotVec,
            color: isMirrorFlip ? '#b9a1de' : '#b5d88e',
            label: chiralLabelYLM
          }
        ]
      }, I3, { tintCenter: '#152520' });

      renderViewport(ctx, { x: halfW, y: 0, w: w - halfW, h }, {
        atoms: transformedAtoms,
        attn: chapterIdx === 3 ? new Array(baseAtoms.length).fill(0) : attn.map(v => v * 0.35),
        ellipsoid: ellipsoidBase,
        arrows: [
          {
            origin_atom: originIdx,
            vec: ylmRotVec,
            color: 'rgba(121, 194, 189, 0.45)',
            dashed: true,
            label: ''
          },
          {
            origin_atom: originIdx,
            vec: baseVec,
            color: '#d97757',
            label: chiralLabelBase
          }
        ]
      }, I3, { tintCenter: '#23191d' });

      // Crisp center divider + top/bottom HUD labels
      ctx.strokeStyle = '#283644';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(halfW + 0.5, 0);
      ctx.lineTo(halfW + 0.5, h);
      ctx.stroke();

      ctx.font = '600 12px Inter, "Segoe UI", sans-serif';
      ctx.fillStyle = '#b5d88e';
      ctx.fillText('YLM · Irreps-to-Text (0e + 0o + 1o + 2e)', 20, 28);
      ctx.fillStyle = '#d97757';
      ctx.fillText('Distance-Only / Scalar-Bridge Baseline (0e)', halfW + 20, 28);

      ctx.font = '12px "SFMono-Regular", Consolas, monospace';
      ctx.fillStyle = '#9fb0c1';
      ctx.fillText(ch.ylmMetric, 20, h - 18);
      ctx.fillText(ch.baseMetric, halfW + 20, h - 18);

      if (chapterIdx === 2) {
        const badge = isMirrorFlip ? 'det(Q) = −1 · Mirror Reflection Active' : 'det(Q) = +1 · Proper Rotation';
        ctx.fillStyle = isMirrorFlip ? '#b9a1de' : '#79c2bd';
        ctx.fillText(badge, 20, 48);
      }

      requestAnimationFrame(renderFrame);
    }

    setChapter(0);
    updateToggleUI();
    requestAnimationFrame(renderFrame);
  }

  window.YLM3D = { mountScene, initHeroShowcase, rotMatrixEuler, applyMat };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHeroShowcase, { once: true });
  } else {
    initHeroShowcase();
  }
})();
