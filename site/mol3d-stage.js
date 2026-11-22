/* Volumetric high-DPI theme-adaptive 3D molecular, vector, and tensor stage for YLM. */
(() => {
  'use strict';

  const ELEMENT_STYLE = {
    1:  { sym: 'H',  r: 0.30, hi: '#ffffff', mid: '#cbd5e1', lo: '#64748b' },
    3:  { sym: 'Li', r: 0.50, hi: '#f3e8ff', mid: '#a855f7', lo: '#581c87' },
    6:  { sym: 'C',  r: 0.44, hi: '#94a3b8', mid: '#475569', lo: '#1e293b' },
    7:  { sym: 'N',  r: 0.43, hi: '#dbeafe', mid: '#3b82f6', lo: '#1e3a8a' },
    8:  { sym: 'O',  r: 0.43, hi: '#ffe4e6', mid: '#f43f5e', lo: '#881337' },
    9:  { sym: 'F',  r: 0.42, hi: '#ccfbf1', mid: '#14b8a6', lo: '#115e59' },
    15: { sym: 'P',  r: 0.52, hi: '#fef3c7', mid: '#f59e0b', lo: '#78350f' },
    16: { sym: 'S',  r: 0.51, hi: '#fef9c3', mid: '#eab308', lo: '#713f12' },
    17: { sym: 'Cl', r: 0.49, hi: '#cffafe', mid: '#06b6d4', lo: '#164e63' },
    26: { sym: 'Fe', r: 0.54, hi: '#ffedd5', mid: '#ea580c', lo: '#7c2d12' },
    35: { sym: 'Br', r: 0.53, hi: '#fce7f3', mid: '#db2777', lo: '#831843' }
  };

  function isDark() {
    return document.documentElement.dataset.theme === 'dark';
  }

  function elemStyle(z) {
    return ELEMENT_STYLE[z] || { sym: `Z${z}`, r: 0.45, hi: '#e2e8f0', mid: '#64748b', lo: '#334155' };
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

    if (attnWeight > 0.14) {
      const haloR = radiusPx * (1.58 + 0.45 * attnWeight);
      const hg = ctx.createRadialGradient(pt.x, pt.y, radiusPx * 0.65, pt.x, pt.y, haloR);
      hg.addColorStop(0, `rgba(13, 148, 136, ${(0.42 * attnWeight).toFixed(3)})`);
      hg.addColorStop(0.65, `rgba(37, 99, 235, ${(0.16 * attnWeight).toFixed(3)})`);
      hg.addColorStop(1, 'rgba(13, 148, 136, 0)');
      ctx.fillStyle = hg;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, haloR, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = `rgba(13, 148, 136, ${(0.45 + 0.50 * attnWeight).toFixed(3)})`;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radiusPx * 1.38, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = isDark() ? 'rgba(0, 0, 0, 0.38)' : 'rgba(15, 23, 42, 0.14)';
    ctx.beginPath();
    ctx.arc(pt.x + radiusPx * 0.10, pt.y + radiusPx * 0.16, radiusPx * 1.01, 0, Math.PI * 2);
    ctx.fill();

    const grad = ctx.createRadialGradient(
      pt.x - radiusPx * 0.34, pt.y - radiusPx * 0.36, radiusPx * 0.08,
      pt.x, pt.y, radiusPx
    );
    grad.addColorStop(0, style.hi);
    grad.addColorStop(0.48, style.mid);
    grad.addColorStop(1, style.lo);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radiusPx, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = isDark() ? 'rgba(255, 255, 255, 0.20)' : 'rgba(15, 23, 42, 0.18)';
    ctx.lineWidth = 0.85;
    ctx.stroke();
    ctx.restore();
  }

  function drawDualBond(ctx, p1, p2, s1, s2, widthPx, dashed = false) {
    const mx = (p1.x + p2.x) * 0.5;
    const my = (p1.y + p2.y) * 0.5;
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;
    const dark = isDark();

    ctx.save();
    ctx.lineCap = 'round';
    if (dashed) ctx.setLineDash([4, 4]);

    for (const [a, b, col] of [[p1, { x: mx, y: my }, s1.mid], [{ x: mx, y: my }, p2, s2.mid]]) {
      ctx.strokeStyle = dark ? 'rgba(7, 11, 16, 0.65)' : 'rgba(15, 23, 42, 0.22)';
      ctx.lineWidth = widthPx + 1.6;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();

      const bg = ctx.createLinearGradient(
        a.x + nx * widthPx * 0.5, a.y + ny * widthPx * 0.5,
        a.x - nx * widthPx * 0.5, a.y - ny * widthPx * 0.5
      );
      bg.addColorStop(0, dark ? '#d6e1ec' : '#ffffff');
      bg.addColorStop(0.45, col);
      bg.addColorStop(1, dark ? '#1e293b' : '#475569');
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
    const dark = isDark();

    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = dark ? 0.22 : 0.16;
    ctx.lineWidth = 7;
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
      ctx.fillStyle = dark ? 'rgba(15, 23, 42, 0.88)' : 'rgba(255, 255, 255, 0.92)';
      ctx.strokeStyle = dark ? 'rgba(148, 163, 184, 0.28)' : 'rgba(148, 163, 184, 0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(lx - tw / 2 - 6, ly - 9, tw + 12, 18, 5);
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
    ctx.lineWidth = 1.4;
    const pairs = [[0, 1], [1, 2], [2, 0]];
    pairs.forEach(([ia, ib], idx) => {
      ctx.globalAlpha = idx === 0 ? 0.55 : 0.32;
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
    const dark = isDark();

    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();

    const bgGrad = ctx.createRadialGradient(cx, cy - h * 0.06, 12, cx, cy, Math.max(w, h) * 0.68);
    if (dark) {
      bgGrad.addColorStop(0, opts.darkTint || '#142230');
      bgGrad.addColorStop(1, '#0b1119');
    } else {
      bgGrad.addColorStop(0, opts.lightTint || '#ffffff');
      bgGrad.addColorStop(1, '#f1f5f9');
    }
    ctx.fillStyle = bgGrad;
    ctx.fillRect(x, y, w, h);

    const atoms = scene.atoms || [];
    if (!atoms.length) {
      ctx.restore();
      return;
    }

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

    ctx.fillStyle = dark ? 'rgba(0, 0, 0, 0.28)' : 'rgba(15, 23, 42, 0.06)';
    ctx.beginPath();
    ctx.ellipse(cx, y + h * 0.83, Math.min(w * 0.26, 106), 9, 0, 0, Math.PI * 2);
    ctx.fill();

    const rotated = centered.map((a, idx) => {
      const r3 = applyMat(viewMat, a.cpos);
      const pt = projectPoint(r3, cx, cy, scale);
      const zNum = a.Z ?? a.z ?? 6;
      return { idx, zNum, style: elemStyle(zNum), r3, pt, attn: (scene.attn && scene.attn[idx]) || 0 };
    });

    if (scene.ellipsoid) {
      const axesRot = scene.ellipsoid.axes.map(ax => applyMat(viewMat, ax));
      drawTensorEllipsoid(ctx, [0, 0, 0], axesRot, scene.ellipsoid.radii, cx, cy, scale, scene.ellipsoid.color || '#0d9488');
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
        const wPx = Math.max(3.1, scale * 0.13 * ((prim.a.pt.p + prim.b.pt.p) * 0.5));
        drawDualBond(ctx, prim.a.pt, prim.b.pt, prim.a.style, prim.b.style, wPx, prim.dashed);
      } else {
        const it = prim.item;
        const rPx = Math.max(6.5, it.style.r * scale * it.pt.p);
        const depthAlpha = Math.max(0.76, Math.min(1.0, 0.92 + it.r3[2] * 0.04));
        drawShadedAtom(ctx, it.pt, rPx, it.style, depthAlpha, it.attn);
      }
    }

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
      draw3DVectorArrow(ctx, p0, p1, arr.color || '#0d9488', arr.label || '', Boolean(arr.dashed));
    }

    ctx.restore();
  }

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
      window.addEventListener('ylm:themechange', () => requestAnimationFrame(draw));
    }

    canvas.__ylmUpdate = nextScene => {
      scene = nextScene || { atoms: [], arrows: [] };
      draw();
    };
    canvas.__ylmUpdate(scene);
    return { update: canvas.__ylmUpdate, redraw: draw };
  }

  function initHeroShowcase() {
    const canvas = document.getElementById('hero-stage-canvas');
    if (!canvas) return;

    const R_FEC_ATOMS = [
      { Z: 6, pos: [ 0.00,  0.96,  0.04] },
      { Z: 8, pos: [ 0.00,  2.16,  0.09] },
      { Z: 8, pos: [-1.11,  0.22,  0.02] },
      { Z: 8, pos: [ 1.11,  0.22, -0.02] },
      { Z: 6, pos: [-0.74, -1.16, -0.14] },
      { Z: 6, pos: [ 0.76, -1.14,  0.16] },
      { Z: 9, pos: [ 1.38, -1.96, -0.74] },
      { Z: 1, pos: [ 1.04, -1.46,  1.17] },
      { Z: 1, pos: [-1.29, -1.78,  0.56] },
      { Z: 1, pos: [-0.96, -1.50, -1.15] }
    ];

    const WATER_ATOMS = [
      { Z: 8, pos: [ 0.00,  0.00, 0.00] },
      { Z: 1, pos: [ 0.76,  0.59, 0.00] },
      { Z: 1, pos: [-0.76,  0.59, 0.00] }
    ];

    const CHAPTERS = [
      {
        key: 'overview',
        prompt: '“What is the 3D molecular dipole moment vector of (R)-fluoroethylene carbonate?”',
        ylmMetric: 'YLM O(3) error: 2.4 × 10⁻¹⁶',
        baseMetric: '0e Distance Baseline: vector frozen in lab frame'
      },
      {
        key: 'rotation',
        prompt: '“What is the force vector acting on the carbonyl oxygen atom?”',
        ylmMetric: 'f(X Rᵀ) = f(X) Rᵀ  (cos = +1.000)',
        baseMetric: 'd_ij unchanged → output never rotates'
      },
      {
        key: 'mirror',
        prompt: '“Predict the chiral handedness pseudoscalar of this stereoisomer.”',
        ylmMetric: 'YLM 0o: +0.842 (R)  ⇄  −0.842 (S)',
        baseMetric: '0e Baseline: identical +0.193 on both twins'
      },
      {
        key: 'routing',
        prompt: 'Switching question wording shifts α_ti across O(carbonyl) → F(chiral) → C(chiral)',
        ylmMetric: 'Routing invariance drift: 0.0 × 10⁻¹⁶',
        baseMetric: 'Task-ID without text: 0.113 force cosine'
      },
      {
        key: 'tensor',
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
      const h = Math.max(240, Math.round(rect.height || 430));
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }

      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const dark = isDark();
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
      const transformedAtoms = baseAtoms.map(a => ({ Z: a.Z, pos: applyMat(R, a.pos) }));

      const tealCol = dark ? '#2dd4bf' : '#0d9488';
      const purpleCol = dark ? '#c084fc' : '#7c3aed';
      const coralCol = dark ? '#fb7185' : '#e11d48';

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
          color: dark ? '#38bdf8' : '#2563eb'
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
          color: tealCol
        };
        ellipsoidBase = {
          axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          radii: [1.55, 1.18, 0.86],
          color: coralCol
        };
      }

      const ylmRotVec = applyMat(R, baseVec);
      const chiralLabelYLM = chapterIdx === 2 ? (isMirrorFlip ? '(S) 0o = −0.842' : '(R) 0o = +0.842') : 'YLM (exact O(3))';
      const chiralLabelBase = chapterIdx === 2 ? '0e = +0.193 (unchanged)' : '0e Baseline (frozen)';

      renderViewport(ctx, { x: 0, y: 0, w: halfW, h }, {
        atoms: transformedAtoms,
        attn,
        ellipsoid: ellipsoidYLM,
        arrows: [{
          origin_atom: originIdx,
          vec: ylmRotVec,
          color: isMirrorFlip ? purpleCol : tealCol,
          label: chiralLabelYLM
        }]
      }, I3, { lightTint: '#f7fefa', darkTint: '#122220' });

      renderViewport(ctx, { x: halfW, y: 0, w: w - halfW, h }, {
        atoms: transformedAtoms,
        attn: chapterIdx === 3 ? new Array(baseAtoms.length).fill(0) : attn.map(v => v * 0.35),
        ellipsoid: ellipsoidBase,
        arrows: [
          {
            origin_atom: originIdx,
            vec: ylmRotVec,
            color: dark ? 'rgba(45, 212, 191, 0.40)' : 'rgba(13, 148, 136, 0.36)',
            dashed: true,
            label: ''
          },
          {
            origin_atom: originIdx,
            vec: baseVec,
            color: coralCol,
            label: chiralLabelBase
          }
        ]
      }, I3, { lightTint: '#fff9fa', darkTint: '#21161c' });

      ctx.strokeStyle = dark ? '#233040' : '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(halfW + 0.5, 0);
      ctx.lineTo(halfW + 0.5, h);
      ctx.stroke();

      ctx.font = '600 12px Inter, "Segoe UI", sans-serif';
      ctx.fillStyle = tealCol;
      ctx.fillText('YLM · Irreps-to-Text (0e + 0o + 1o + 2e)', 18, 26);
      ctx.fillStyle = coralCol;
      ctx.fillText('Distance-Only / Scalar-Bridge Baseline (0e)', halfW + 18, 26);

      ctx.font = '12px "SFMono-Regular", Consolas, monospace';
      ctx.fillStyle = dark ? '#9fb0c3' : '#475569';
      ctx.fillText(ch.ylmMetric, 18, h - 16);
      ctx.fillText(ch.baseMetric, halfW + 18, h - 16);

      if (chapterIdx === 2) {
        const badge = isMirrorFlip ? 'det(Q) = −1 · Mirror Reflection Active' : 'det(Q) = +1 · Proper Rotation';
        ctx.fillStyle = isMirrorFlip ? purpleCol : tealCol;
        ctx.fillText(badge, 18, 46);
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
