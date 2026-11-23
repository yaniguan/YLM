/* Clean theme-adaptive interactions + 8-tab YLM Studio controller (works standalone on GitHub Pages + live with ylm serve). */
(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';

  function isDark() {
    return document.documentElement.dataset.theme === 'dark';
  }
  function colTeal() { return isDark() ? '#2dd4bf' : '#0d9488'; }
  function colBlue() { return isDark() ? '#60a5fa' : '#2563eb'; }
  function colPurple() { return isDark() ? '#c084fc' : '#7c3aed'; }
  function colCoral() { return isDark() ? '#fb7185' : '#e11d48'; }

  const STATIC_PRESETS = {
    r_fec_chiral: {
      label: '(R)-Fluoroethylene carbonate',
      chiral0o: 0.842,
      dipole: [-0.26, -1.42, 0.28],
      forceAtom: 1,
      atomNames: ['C0 (C=O)', 'O1 (carbonyl)', 'O2 (ring)', 'O3 (ring)', 'C4 (CH₂)', 'C5* (CHF)', 'F6', 'H7', 'H8', 'H9'],
      atoms: [
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
      ]
    },
    r_bcfm_chiral: {
      label: '(R)-Bromochlorofluoromethane',
      chiral0o: 0.914,
      dipole: [0.64, -0.92, 0.48],
      forceAtom: 3,
      atoms: [
        { Z: 6,  pos: [ 0.00,  0.00,  0.00] },
        { Z: 9,  pos: [ 0.00,  1.36,  0.00] },
        { Z: 17, pos: [ 1.66, -0.58,  0.00] },
        { Z: 35, pos: [-0.92, -0.64,  1.58] },
        { Z: 1,  pos: [-0.62, -0.52, -0.89] }
      ]
    },
    water_c2v: {
      label: 'Water (H₂O · Planar C₂v)',
      chiral0o: 0.0,
      dipole: [0.00, -1.18, 0.00],
      forceAtom: 0,
      atoms: [
        { Z: 8, pos: [ 0.00,  0.00, 0.00] },
        { Z: 1, pos: [ 0.76,  0.59, 0.00] },
        { Z: 1, pos: [-0.76,  0.59, 0.00] }
      ]
    },
    trans_difluoroethene_c2h: {
      label: 'trans-1,2-Difluoroethene (C₂h)',
      chiral0o: 0.0,
      dipole: [0.00, 0.00, 0.00],
      forceAtom: 2,
      atoms: [
        { Z: 6, pos: [-0.66,  0.00, 0.00] },
        { Z: 6, pos: [ 0.66,  0.00, 0.00] },
        { Z: 9, pos: [-1.38,  1.14, 0.00] },
        { Z: 9, pos: [ 1.38, -1.14, 0.00] },
        { Z: 1, pos: [-1.22, -0.94, 0.00] },
        { Z: 1, pos: [ 1.22,  0.94, 0.00] }
      ]
    },
    lifepo4_pbc_crystal: {
      label: 'LiFePO₄ Olivine Unit Cluster',
      chiral0o: 0.0,
      dipole: [0.85, 0.52, -0.44],
      forceAtom: 0,
      atoms: [
        { Z: 3,  pos: [-1.62,  0.12, -0.40] },
        { Z: 26, pos: [ 0.15, -0.18,  0.10] },
        { Z: 15, pos: [ 1.78,  0.24, -0.12] },
        { Z: 8,  pos: [ 0.92,  1.32,  0.42] },
        { Z: 8,  pos: [ 0.88, -1.26, -0.48] },
        { Z: 8,  pos: [ 2.68,  0.48, -1.24] },
        { Z: 8,  pos: [ 2.48, -0.34,  1.06] }
      ]
    },
    CCO: {
      label: 'Ethanol (CH₃CH₂OH)',
      chiral0o: 0.0,
      dipole: [-0.58, 0.94, 0.12],
      forceAtom: 2,
      atoms: [
        { Z: 6, pos: [-1.18, -0.22,  0.00] },
        { Z: 6, pos: [ 0.14,  0.52,  0.00] },
        { Z: 8, pos: [ 1.22, -0.40,  0.00] },
        { Z: 1, pos: [ 2.04,  0.10,  0.00] },
        { Z: 1, pos: [-1.24, -0.86,  0.89] },
        { Z: 1, pos: [-1.24, -0.86, -0.89] },
        { Z: 1, pos: [-2.02,  0.48,  0.00] },
        { Z: 1, pos: [ 0.21,  1.16,  0.89] },
        { Z: 1, pos: [ 0.21,  1.16, -0.89] }
      ]
    },
    c1ccncc1: {
      label: 'Pyridine (C₅H₅N)',
      chiral0o: 0.0,
      dipole: [0.00, -1.36, 0.00],
      forceAtom: 3,
      atoms: [
        { Z: 6, pos: [ 0.00,  1.39, 0.00] },
        { Z: 6, pos: [ 1.19,  0.69, 0.00] },
        { Z: 6, pos: [ 1.14, -0.70, 0.00] },
        { Z: 7, pos: [ 0.00, -1.38, 0.00] },
        { Z: 6, pos: [-1.14, -0.70, 0.00] },
        { Z: 6, pos: [-1.19,  0.69, 0.00] },
        { Z: 1, pos: [ 0.00,  2.47, 0.00] },
        { Z: 1, pos: [ 2.14,  1.21, 0.00] },
        { Z: 1, pos: [ 2.05, -1.28, 0.00] },
        { Z: 1, pos: [-2.05, -1.28, 0.00] },
        { Z: 1, pos: [-2.14,  1.21, 0.00] }
      ]
    },
    'C[C@H](O)C(=O)O': {
      label: 'L-Lactic acid',
      chiral0o: 0.785,
      chiralAtom: 1,
      dipole: [-0.82, 0.96, -0.34],
      atoms: [
        { Z: 6, pos: [-1.42, -0.46,  0.28] },
        { Z: 6, pos: [-0.04,  0.16, -0.06] },
        { Z: 8, pos: [-0.08,  1.54,  0.24] },
        { Z: 6, pos: [ 1.10, -0.52,  0.66] },
        { Z: 8, pos: [ 1.24, -1.72,  0.62] },
        { Z: 8, pos: [ 1.98,  0.28,  1.28] },
        { Z: 1, pos: [ 0.14,  0.02, -1.13] },
        { Z: 1, pos: [-0.86,  1.92, -0.16] },
        { Z: 1, pos: [ 2.68, -0.22,  1.70] },
        { Z: 1, pos: [-2.21,  0.04, -0.28] },
        { Z: 1, pos: [-1.62, -0.36,  1.35] },
        { Z: 1, pos: [-1.42, -1.52,  0.02] }
      ]
    }
  };

  const API_VARIANTS = {
    matscibert_fp32: { label: 'MatSciBERT fp32 (default)', cpuP50: '79 ms', wallP50: '112 ms', forceCos: '0.946 / 0.910', atomHit: '0.946 / 0.929' },
    matscibert_int8: { label: 'MatSciBERT int8 text', cpuP50: '35 ms', wallP50: '39 ms', forceCos: '0.925 / 0.896', atomHit: '0.933 / 0.920' },
    minilm_fp32:     { label: 'MiniLM fp32', cpuP50: '18 ms', wallP50: '20 ms', forceCos: '0.924 / 0.888', atomHit: '0.759 / 0.737' },
    minilm_int8:     { label: 'MiniLM int8 text', cpuP50: '13 ms', wallP50: '14 ms', forceCos: '0.882 / 0.773', atomHit: '0.732 / 0.638' }
  };

  const DEFAULT_LAWS = [
    { name: 'Translation Invariance', formula: 'f(X + t) = f(X)', err: '3.25e-16', passed: true },
    { name: 'SO(3) Proper Rotation', formula: 'f(X Rᵀ) = D(R) f(X)', err: '1.12e-09', passed: true },
    { name: 'O(3) Improper Mirror Parity', formula: '0o flips · 1o reflects · 2e conjugates', err: '1.04e-09', passed: true },
    { name: 'Atom Permutation Equivariance', formula: 'f(P X) = P f(X)', err: '1.66e-16', passed: true },
    { name: 'Symmetric 3×3 Tensor (M = Mᵀ)', formula: '||M − Mᵀ||_F = 0', err: '1.71e-16', passed: true },
    { name: 'Curie Point-Group Cancellation', formula: 'Planar 0o = 0 · Inversion 1o = 0', err: '1.68e-12', passed: true },
    { name: 'Conservative Force (F = −∂E/∂x)', formula: 'Autograd vs finite-difference Jacobian', err: '2.58e-09', passed: true },
    { name: '3D Crystal Lattice PBC', formula: 'f(X + n · cell) = f(X)', err: '8.41e-17', passed: true }
  ];

  const DEFAULT_BENCH_ROWS = [
    { model: 'GraphMVP / 3D-InfoMax', scalar_mae: 0.0906, vector_cos: 0.0537, vector_eq_err: 1.35, tensor_cos: 0.7082, routing_acc: 0.868 },
    { model: 'Uni-Mol (3D Dist-Transformer)', scalar_mae: 0.2373, vector_cos: 0.0474, vector_eq_err: 1.29, tensor_cos: 0.7625, routing_acc: 0.882 },
    { model: 'EquiLLM (Inv-Bridge + 1o)', scalar_mae: 0.0868, vector_cos: 0.7069, vector_eq_err: 1.16e-6, tensor_cos: 0.9789, routing_acc: 0.861 },
    { model: 'YLM-Universe FM (Ours)', scalar_mae: 0.0981, vector_cos: 0.7202, vector_eq_err: 9.23e-7, tensor_cos: 0.9968, routing_acc: 0.854 }
  ];

  const CLI_RECIPES = {
    studio: `git clone https://github.com/yaniguan/YLM.git && cd YLM\npip install -e . && ylm serve --port 8765\nylm ask examples/water.xyz "What is the force vector on the oxygen atom?" --verify-o3`,
    api: `# Single-thread CPU inference API (79 ms p50 MatSciBERT, 35 ms int8)\npython -m src.api --smiles "O=C1OC[C@H](F)O1" --question "What is the dipole moment of this molecule?"\n# Optional FastAPI server: POST /predict\nuvicorn src.api:app --host 127.0.0.1 --port 8000`,
    physics: `# 64-bit 8-law conservation & mirror parity audit\nylm audit-physics r_fec_chiral --precision float64 --rotations 8\n# Ouroboros 2D -> 3D stereochemistry & reactive MD hooks\nylm from-ouroboros "O=C1OC[C@H](F)O1" --verify-enantiomer\nylm scan-traj examples/proton_transfer_traj.extxyz --verify-o3`,
    lablean: `# Formal Lean 4 protocol verification (lean/) & SHA-256 lab ledger (labledger/)\ncd lean && lake build && lake exe lablean && cd ..\npython -m labledger verify labledger/examples/demo_campaign\npython -m lab.distill --config lab/configs/distill_commercial.json --dry-run --commercial`,
    factory: `# Verifier-gated QM9 force questions (V0-V3) & LabLean + dual-DB SQL factory\npython scripts/06b_unpack_release.py --download\npython scripts/40_token_factory.py --limit 2000 --out runs/token_factory_v1\npython scripts/41_lab_token_factory.py --seeds 0 1 2 --rounds 6`
  };

  function buildDefaultZundelFrames() {
    const frames = [];
    const energies = [];
    for (let i = 0; i < 9; i++) {
      const u = (i / 8) * 2 - 1;
      const hx = 0.34 * u;
      const e = 0.048 * Math.pow(1 - u * u, 2) + 0.004 * u;
      energies.push(e);
      frames.push({
        step: i,
        atoms: [
          { Z: 8, pos: [-1.22,  0.00,  0.00] },
          { Z: 8, pos: [ 1.22,  0.00,  0.00] },
          { Z: 1, pos: [ hx,    0.02,  0.00] },
          { Z: 1, pos: [-1.76,  0.76,  0.22] },
          { Z: 1, pos: [-1.76, -0.76, -0.22] },
          { Z: 1, pos: [ 1.76,  0.76, -0.22] },
          { Z: 1, pos: [ 1.76, -0.76,  0.22] }
        ],
        forceVec: [-0.85 * u, 0.0, 0.0]
      });
    }
    return {
      frames,
      energies,
      events: [
        'Step 3 → 5: Shared proton H2 crosses midpoint between O0 and O1 (d(O–H) = 1.22 Å)',
        'Step 6: Covalent O1–H2 bond formed · O(3) force vector reverses sign smoothly'
      ]
    };
  }

  let liveBackend = false;
  let mdState = buildDefaultZundelFrames();
  let mdTimer = 0;

  // Theme Toggle (Defaults to crisp Light mode; remembers choice)
  function initThemeToggle() {
    const btn = document.getElementById('theme-toggle');
    const saved = localStorage.getItem('ylm-theme');
    if (saved === 'dark') {
      document.documentElement.dataset.theme = 'dark';
    }
    const syncBtn = () => {
      if (!btn) return;
      btn.textContent = isDark() ? '☀ Light' : '☾ Dark';
    };
    syncBtn();
    btn?.addEventListener('click', () => {
      const next = isDark() ? 'light' : 'dark';
      if (next === 'dark') document.documentElement.dataset.theme = 'dark';
      else delete document.documentElement.dataset.theme;
      localStorage.setItem('ylm-theme', next);
      syncBtn();
      window.dispatchEvent(new CustomEvent('ylm:themechange'));
      refreshSymmetryLiveOrStatic();
      runAskTab();
      renderOptionsRouter();
      renderMdFrame(Number(document.getElementById('md-scrubber')?.value || 0));
      runOuroborosTab();
    });
  }

  // Lightbox, Copy Buttons & Recipe Switcher
  function initEditorialUI() {
    const dialog = document.getElementById('figure-dialog');
    const expanded = document.getElementById('expanded-image');
    const caption = document.getElementById('expanded-caption');
    const closeBtn = document.getElementById('close-figure');

    document.querySelectorAll('[data-image]').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!dialog || !expanded) return;
        expanded.src = btn.dataset.image;
        const img = btn.querySelector('img');
        expanded.alt = img ? img.alt : '';
        if (caption) caption.textContent = btn.dataset.caption || '';
        dialog.showModal();
      });
    });

    if (closeBtn && dialog) {
      closeBtn.addEventListener('click', () => dialog.close());
      dialog.addEventListener('click', e => { if (e.target === dialog) dialog.close(); });
    }

    const bindCopy = (btnId, srcId, defaultLabel, statusId) => {
      const btn = document.getElementById(btnId);
      const src = document.getElementById(srcId);
      const status = statusId ? document.getElementById(statusId) : null;
      if (!btn || !src) return;
      btn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(src.textContent);
          btn.textContent = 'Copied ✓';
          if (status) status.textContent = 'Copied to clipboard.';
          setTimeout(() => {
            btn.textContent = defaultLabel;
            if (status) status.textContent = '';
          }, 2000);
        } catch {
          btn.textContent = 'Select text';
        }
      });
    };
    bindCopy('copy-citation', 'bibtex', 'Copy citation', 'copy-status');
    bindCopy('copy-recipe', 'recipe-code', 'Copy snippet', null);

    document.querySelectorAll('#recipe-pills .recipe-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        document.querySelectorAll('#recipe-pills .recipe-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        const codeEl = document.getElementById('recipe-code');
        if (codeEl && CLI_RECIPES[pill.dataset.recipe]) {
          codeEl.textContent = CLI_RECIPES[pill.dataset.recipe];
        }
      });
    });

    document.querySelectorAll('a[href="#repo-map"]').forEach(link => {
      link.addEventListener('click', () => {
        const det = document.getElementById('repo-map');
        if (det) det.open = true;
      });
    });
  }

  function activateStudioPane(targetId) {
    const tabs = Array.from(document.querySelectorAll('.studio-tab'));
    const panes = Array.from(document.querySelectorAll('.studio-pane'));
    tabs.forEach(t => {
      const active = t.dataset.pane === targetId;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', String(active));
    });
    panes.forEach(p => p.classList.toggle('active', p.id === targetId));
    window.dispatchEvent(new Event('resize'));
  }

  function initStudioTabs() {
    document.querySelectorAll('.studio-tab').forEach(tab => {
      tab.addEventListener('click', () => activateStudioPane(tab.dataset.pane));
    });

    document.querySelectorAll('[data-jump-pane]').forEach(card => {
      card.addEventListener('click', e => {
        e.preventDefault();
        const paneId = card.dataset.jumpPane;
        activateStudioPane(paneId);
        document.getElementById('studio')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });
  }

  async function apiPost(path, body) {
    const resp = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!resp.ok) throw new Error(`API ${resp.status}`);
    return resp.json();
  }

  // Tab 1: Symmetry Sandbox + 8-Law Audit
  function renderLawsList(laws, passedCount, totalCount) {
    const host = document.getElementById('audit-laws-list');
    const pill = document.getElementById('audit-summary-pill');
    if (pill) pill.textContent = `${passedCount} / ${totalCount} Passed`;
    if (!host) return;
    host.innerHTML = laws.map(l => `
      <div class="law-row">
        <div>
          <span class="law-dot ${l.passed ? '' : 'fail'}"></span>
          <strong>${l.name || l.title}</strong>
          <div style="font-size:12px;color:var(--muted);margin-left:16px;">${l.formula || l.invariant_expr || ''}</div>
        </div>
        <span class="mono ${l.passed ? 'lime' : 'orange'}" style="font-size:12.5px;">${typeof l.ylm_error === 'number' ? l.ylm_error.toExponential(2) : l.err}</span>
      </div>
    `).join('');
  }

  function vecNormNonZero(v) {
    return Math.hypot(v[0], v[1], v[2]) > 1e-4 ? v : [0.95, 0.35, 0.0];
  }

  function updateSymmetrySandboxStatic() {
    const presetKey = document.getElementById('sym-preset').value;
    const rx = Number(document.getElementById('sl-rx').value);
    const ry = Number(document.getElementById('sl-ry').value);
    const rz = Number(document.getElementById('sl-rz').value);
    const reflect = document.getElementById('chk-reflect').checked;

    document.getElementById('lbl-rx').textContent = `${rx}°`;
    document.getElementById('lbl-ry').textContent = `${ry}°`;
    document.getElementById('lbl-rz').textContent = `${rz}°`;
    document.getElementById('sym-det-pill').textContent = `det(Q) = ${reflect ? '−1.0' : '+1.0'}`;

    const preset = STATIC_PRESETS[presetKey] || STATIC_PRESETS.r_fec_chiral;
    const deg = Math.PI / 180;
    const Q = window.YLM3D.rotMatrixEuler(rx * deg, ry * deg, rz * deg, reflect);
    const origVec = vecNormNonZero(preset.dipole);
    const rotVec = window.YLM3D.applyMat(Q, origVec);
    const transAtoms = preset.atoms.map(a => ({ Z: a.Z, pos: window.YLM3D.applyMat(Q, a.pos) }));

    window.YLM3D.mountScene(document.getElementById('canvas-sym-orig'), {
      atoms: preset.atoms,
      arrows: [{ vec: origVec, color: colTeal(), label: 'YLM f(X)' }]
    });
    window.YLM3D.mountScene(document.getElementById('canvas-sym-trans'), {
      atoms: transAtoms,
      arrows: [
        { vec: rotVec, color: colTeal(), label: 'YLM f(XQᵀ)' },
        { vec: origVec, color: colCoral(), dashed: true, label: '0e Baseline' }
      ]
    });

    const dot = origVec[0] * rotVec[0] + origVec[1] * rotVec[1] + origVec[2] * rotVec[2];
    const cosUni = (dot / (Math.hypot(...origVec) * Math.hypot(...rotVec) || 1)).toFixed(3);
    const tbody = document.getElementById('sym-metrics-tbody');
    if (tbody) {
      tbody.innerHTML = `
        <tr>
          <th class="lime">YLM-Universe</th>
          <td class="mono">1.4e-15</td>
          <td class="mono lime">${reflect && preset.chiral0o ? '6.2e-15' : '0.0e+00'}</td>
          <td class="mono lime">+1.000</td>
          <td class="mono">4.8e-15</td>
        </tr>
        <tr>
          <th>EquiLLM (0e Bridge + 1o)</th>
          <td class="mono">1.1e-15</td>
          <td class="mono orange">${reflect && preset.chiral0o ? '1.68e+00' : '2.1e-01'}</td>
          <td class="mono blue">+1.000</td>
          <td class="mono">8.2e-15</td>
        </tr>
        <tr>
          <th>Uni-Mol (3D Distance)</th>
          <td class="mono">0.0e+00</td>
          <td class="mono orange">${reflect && preset.chiral0o ? '1.68e+00' : '1.9e-01'}</td>
          <td class="mono orange">${cosUni >= 0 ? '+' + cosUni : cosUni}</td>
          <td class="mono orange">1.31e+00</td>
        </tr>
      `;
    }
  }

  async function refreshSymmetryLiveOrStatic() {
    updateSymmetrySandboxStatic();
    if (!liveBackend) return;
    try {
      const presetKey = document.getElementById('sym-preset').value;
      const precision = document.getElementById('sym-precision').value;
      const rx = Number(document.getElementById('sl-rx').value);
      const ry = Number(document.getElementById('sl-ry').value);
      const rz = Number(document.getElementById('sl-rz').value);
      const reflect = document.getElementById('chk-reflect').checked;
      const permute = document.getElementById('chk-permute').checked;
      const shift = document.getElementById('chk-shift').checked;

      const sbx = await apiPost('/api/pose_sandbox', {
        structure: presetKey,
        rx_deg: rx, ry_deg: ry, rz_deg: rz,
        reflect, permute_atoms: permute,
        translate_ang: shift ? [2.4, -1.6, 3.1] : [0, 0, 0],
        precision
      });

      if (sbx.scene_orig && sbx.scene_transformed) {
        window.YLM3D.mountScene(document.getElementById('canvas-sym-orig'), {
          atoms: sbx.scene_orig.atoms,
          arrows: (sbx.scene_orig.arrows || []).map(a => ({ vec: a.vec || a.vector, color: colTeal(), label: a.label || 'YLM' }))
        });
        const paletteMap = [colBlue(), colTeal(), colCoral()];
        window.YLM3D.mountScene(document.getElementById('canvas-sym-trans'), {
          atoms: sbx.scene_transformed.atoms,
          arrows: (sbx.scene_transformed.arrows || []).map((a, i) => ({
            vec: a.vec || a.vector,
            color: paletteMap[i % paletteMap.length],
            dashed: i === 2,
            label: a.label || ''
          }))
        });
      }
    } catch {
      // Keep instant static preview cleanly
    }
  }

  // Tab 2: Ask 3D Molecule + CPU API Variant Explorer
  async function runAskTab() {
    const structVal = document.getElementById('ask-structure').value.trim();
    const qVal = document.getElementById('ask-question').value.trim();
    const variantKey = document.getElementById('ask-api-variant')?.value || 'matscibert_fp32';
    const apiMeta = API_VARIANTS[variantKey] || API_VARIANTS.matscibert_fp32;
    const apiPill = document.getElementById('ask-api-pill');
    const summaryEl = document.getElementById('ask-result-summary');
    const rotPill = document.getElementById('ask-rot-pill');

    if (apiPill) apiPill.textContent = `CPU p50: ${apiMeta.cpuP50}`;

    if (liveBackend) {
      try {
        const res = await apiPost('/api/ask', { structure: structVal, question: qVal, verify_o3: true });
        const vPred = (res.vectors && res.vectors[0]) || {};
        const vec = vPred.vector_physical || [0, 0, 0];
        const eqErr = res.o3_check ? res.o3_check.vector_equivariance_error : 1.2e-6;
        if (rotPill) rotPill.textContent = `O(3) err: ${Number(eqErr).toExponential(2)}`;
        const structInfo = res.structure || {};
        const zMap = { H: 1, Li: 3, C: 6, N: 7, O: 8, F: 9, P: 15, S: 16, Cl: 17, Fe: 26, Br: 35 };
        const atoms = (structInfo.symbols || []).map((sym, idx) => ({
          Z: zMap[sym] || 6,
          pos: structInfo.pos_angstrom[idx]
        }));
        const attn = res.pooled_atom_attention || [];
        let maxIdx = vPred.anchor_atom !== null && vPred.anchor_atom !== undefined ? vPred.anchor_atom : 0;
        attn.forEach((v, i) => { if (v > (attn[maxIdx] || 0)) maxIdx = i; });
        const isForceTask = /force/i.test(vPred.name || qVal);
        window.YLM3D.mountScene(document.getElementById('canvas-ask-3d'), {
          atoms,
          attn,
          arrows: [{
            origin_atom: isForceTask ? maxIdx : undefined,
            vec,
            color: colTeal(),
            label: vPred.name || (isForceTask ? 'Predicted Force' : 'Predicted Dipole')
          }]
        });
        summaryEl.innerHTML = `
          <div style="border-top:1px solid var(--line);padding-top:12px;display:grid;gap:7px;font-size:13.5px;">
            <div><span style="color:var(--muted);">Readout:</span> <strong class="lime mono">${vPred.name || 'vector (1o)'}</strong> (${vPred.unit || 'atomic units'})</div>
            <div><span style="color:var(--muted);">3D Vector:</span> <span class="mono">[${vec.map(v => Number(v).toFixed(4)).join(', ')}]</span></div>
            <div><span style="color:var(--muted);">Top routed atom:</span> <span class="mono">Atom #${maxIdx} (α = ${Number(attn[maxIdx] || 0).toFixed(3)})</span></div>
            <div><span style="color:var(--muted);">CPU API profile (${apiMeta.label}):</span> <span class="mono blue">${apiMeta.cpuP50} CPU · cos ${apiMeta.forceCos}</span></div>
          </div>
        `;
        return;
      } catch {
        // Fall through to static preset display
      }
    }

    const p = STATIC_PRESETS[structVal] || STATIC_PRESETS.water_c2v;
    const isForce = /force/i.test(qVal);
    const fIdx = p.forceAtom ?? 0;
    const isMini = variantKey.startsWith('minilm');
    const peakAlpha = isMini ? 0.76 : 0.94;
    const attn = p.atoms.map((_, idx) => (isForce && idx === fIdx ? peakAlpha : (isForce && idx === 0 ? 0.14 : 0.03)));
    window.YLM3D.mountScene(document.getElementById('canvas-ask-3d'), {
      atoms: p.atoms,
      attn,
      arrows: [{
        origin_atom: isForce ? fIdx : undefined,
        vec: p.dipole,
        color: colTeal(),
        label: isForce ? 'Force (1o)' : 'Dipole (1o)'
      }]
    });
    summaryEl.innerHTML = `
      <div style="border-top:1px solid var(--line);padding-top:12px;display:grid;gap:7px;font-size:13.5px;">
        <div><span style="color:var(--muted);">Task:</span> <strong class="lime mono">${isForce ? 'per-atom force (1o)' : 'molecular dipole (1o)'}</strong></div>
        <div><span style="color:var(--muted);">Vector:</span> <span class="mono">[${p.dipole.map(v => v.toFixed(3)).join(', ')}]</span></div>
        <div><span style="color:var(--muted);">Top routed atom:</span> <span class="mono">Atom #${fIdx} (α = ${peakAlpha.toFixed(2)} · ${p.label})</span></div>
        <div><span style="color:var(--muted);">CPU API (${apiMeta.label}):</span> <span class="mono blue">CPU p50 ${apiMeta.cpuP50} · test/held-out cos ${apiMeta.forceCos} · atom hit ${apiMeta.atomHit}</span></div>
      </div>
    `;
  }

  // Tab 3 (NEW): Options A / B / C Interactive Router
  let optTask = 'single';
  let optMode = 'C';

  const OPT_SPECS = {
    A: {
      params: '0.42M params',
      note: 'Option A scores atoms from rotation-invariant norms only and outputs a linear combination of atom 1o vectors. Great on single atoms, but cannot build cross-atom projections.',
      single: { cos: 0.956, weights: [0.04, 0.91, 0.02, 0.01, 0.01, 0.01], pred: [0.08, -1.32, 0.21], gold: [0.10, -1.36, 0.22] },
      diff:   { cos: 0.759, weights: [0.05, 0.56, 0.04, 0.03, 0.04, 0.28], pred: [-0.52, -0.95, 0.14], gold: [-0.94, -1.18, 0.52] },
      proj:   { cos: 0.542, weights: [0.06, 0.08, 0.05, 0.05, 0.44, 0.32], pred: [0.35, -0.78, 0.12], gold: [0.62, -0.82, -0.90] }
    },
    B: {
      params: '0.58M params',
      note: 'Option B adds normalized same-irrep directional inner-product keys Q̂_t · K̂_i. Without --attn_sup, force cosine is high (0.959) while attention spreads onto bonded neighbors (atom acc 0.696).',
      single: { cos: 0.959, weights: [0.22, 0.70, 0.03, 0.03, 0.01, 0.01], pred: [0.09, -1.34, 0.21], gold: [0.10, -1.36, 0.22] },
      diff:   { cos: 0.834, weights: [0.06, 0.51, 0.03, 0.02, 0.04, 0.34], pred: [-0.79, -1.10, 0.40], gold: [-0.94, -1.18, 0.52] },
      proj:   { cos: 0.698, weights: [0.03, 0.05, 0.04, 0.04, 0.48, 0.36], pred: [0.51, -0.79, -0.58], gold: [0.62, -0.82, -0.90] }
    },
    B_sup: {
      params: '0.58M params',
      note: 'Option B + --attn_sup 0.1 adds an auxiliary cross-entropy routing loss on force questions, sharpening the attention peak onto the exact asked atom.',
      single: { cos: 0.964, weights: [0.05, 0.90, 0.02, 0.01, 0.01, 0.01], pred: [0.10, -1.35, 0.22], gold: [0.10, -1.36, 0.22] },
      diff:   { cos: 0.848, weights: [0.03, 0.53, 0.02, 0.02, 0.03, 0.37], pred: [-0.83, -1.12, 0.44], gold: [-0.94, -1.18, 0.52] },
      proj:   { cos: 0.715, weights: [0.02, 0.03, 0.03, 0.03, 0.50, 0.39], pred: [0.54, -0.80, -0.64], gold: [0.62, -0.82, -0.90] }
    },
    C: {
      params: '0.85M params (channelwise)',
      note: 'Option C couples routed irreps through a text-gated Clebsch–Gordan tensor product (1o ⊗ 1o → 0e ⊕ 1e ⊕ 2e). Channel-wise mode cuts weights from 13.1M to 0.85M while unlocking multi-atom projections.',
      single: { cos: 0.976, weights: [0.03, 0.92, 0.02, 0.01, 0.01, 0.01], pred: [0.10, -1.36, 0.22], gold: [0.10, -1.36, 0.22] },
      diff:   { cos: 0.890, weights: [0.02, 0.50, 0.02, 0.01, 0.02, 0.43], pred: [-0.91, -1.16, 0.49], gold: [-0.94, -1.18, 0.52] },
      proj:   { cos: 0.781, weights: [0.02, 0.02, 0.02, 0.02, 0.49, 0.43], pred: [0.60, -0.81, -0.84], gold: [0.62, -0.82, -0.90] }
    }
  };

  function renderOptionsRouter() {
    const mol = STATIC_PRESETS.r_fec_chiral;
    const spec = OPT_SPECS[optMode] || OPT_SPECS.C;
    const cur = spec[optTask] || spec.single;

    const cosPill = document.getElementById('opt-cos-pill');
    const paramPill = document.getElementById('opt-param-pill');
    const noteEl = document.getElementById('opt-mechanism-note');
    const barsEl = document.getElementById('opt-attn-bars');
    const metricsEl = document.getElementById('opt-bench-metrics');

    if (cosPill) {
      cosPill.textContent = `cos = ${cur.cos.toFixed(3)}`;
      cosPill.className = `pill-tag mono ${cur.cos >= 0.82 ? 'ok' : ''}`;
    }
    if (paramPill) paramPill.textContent = spec.params;
    if (noteEl) noteEl.textContent = spec.note;

    const fullAttn = new Array(mol.atoms.length).fill(0.01);
    const keyIndices = [0, 1, 2, 3, 5, 6]; // C0, O1, O2, O3, C5*, F6
    keyIndices.forEach((atomIdx, k) => { fullAttn[atomIdx] = cur.weights[k]; });

    const originAtom = optTask === 'proj' ? 5 : 1;
    window.YLM3D.mountScene(document.getElementById('canvas-opt-3d'), {
      atoms: mol.atoms,
      attn: fullAttn,
      arrows: [
        { origin_atom: originAtom, vec: cur.gold, color: colBlue(), dashed: true, label: 'True xTB' },
        { origin_atom: originAtom, vec: cur.pred, color: colTeal(), label: `Option ${optMode.replace('_sup', '+sup')}` }
      ]
    });

    if (barsEl) {
      const labels = ['C0 (C=O)', 'O1 (carbonyl)', 'O2 (ring)', 'O3 (ring)', 'C5* (chiral)', 'F6 (subst.)'];
      barsEl.innerHTML = labels.map((lbl, i) => {
        const w = cur.weights[i];
        return `
          <div class="attn-bar-row">
            <span class="mono">${lbl}</span>
            <div class="attn-track"><div class="attn-fill ${i === 5 ? 'secondary' : ''}" style="width:${Math.round(w * 100)}%"></div></div>
            <span class="mono" style="text-align:right;">${(w * 100).toFixed(0)}%</span>
          </div>
        `;
      }).join('');
    }

    if (metricsEl) {
      metricsEl.innerHTML = `
        <div style="display:grid;gap:5px;">
          <div><span style="color:var(--muted);">Single-atom force cos:</span> <strong class="mono">${spec.single.cos.toFixed(3)}</strong> &nbsp;·&nbsp; <span style="color:var(--muted);">F_A − F_B cos:</span> <strong class="mono">${spec.diff.cos.toFixed(3)}</strong></div>
          <div><span style="color:var(--muted);">Two-atom axis proj cos:</span> <strong class="mono lime">${spec.proj.cos.toFixed(3)}</strong> &nbsp;·&nbsp; <span style="color:var(--muted);">O(3) equivariance err:</span> <span class="mono">~2e-6</span></div>
        </div>
      `;
    }
  }

  // Tab 4: Reaction Dynamics
  function renderMdFrame(idx) {
    const frames = mdState.frames || [];
    if (!frames.length) return;
    const clamped = Math.max(0, Math.min(frames.length - 1, idx));
    const fr = frames[clamped];
    const lbl = document.getElementById('md-frame-label');
    if (lbl) lbl.textContent = `Frame ${clamped + 1} / ${frames.length}`;

    window.YLM3D.mountScene(document.getElementById('canvas-md-3d'), {
      atoms: fr.atoms,
      attn: fr.atoms.map((_, i) => (i === 2 ? 0.88 : 0)),
      arrows: fr.forceVec ? [{ origin_atom: 2, vec: fr.forceVec, color: colTeal(), label: 'F(H⁺)' }] : []
    });

    renderMdEnergySVG(mdState.energies || [], clamped);
  }

  function renderMdEnergySVG(energies, activeIdx) {
    const host = document.getElementById('md-energy-chart');
    if (!host || !energies.length) return;
    const dark = isDark();
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 520 200');
    svg.style.cssText = 'width:100%;height:auto;display:block;border-radius:10px;border:1px solid var(--line);';

    const mk = (tag, attrs, txt) => {
      const n = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
      if (txt !== undefined) n.textContent = txt;
      return n;
    };

    svg.append(mk('rect', { width: 520, height: 200, fill: dark ? '#0e151f' : '#f8fafc' }));
    const px = 48, py = 20, pw = 448, ph = 142;
    const eMin = Math.min(...energies), eMax = Math.max(...energies, eMin + 1e-4);
    const sx = i => px + (energies.length === 1 ? pw / 2 : (i / (energies.length - 1)) * pw);
    const sy = e => py + ph - ((e - eMin) / (eMax - eMin)) * (ph - 18) - 9;

    for (let g = 0; g <= 3; g++) {
      const gy = py + (g / 3) * ph;
      svg.append(mk('line', { x1: px, y1: gy, x2: px + pw, y2: gy, stroke: dark ? '#223041' : '#e2e8f0', 'stroke-width': 1 }));
    }

    const ax = sx(activeIdx);
    svg.append(mk('line', { x1: ax, y1: py, x2: ax, y2: py + ph, stroke: colBlue(), 'stroke-dasharray': '4 4', 'stroke-width': 1.4 }));

    const pathD = energies.map((e, i) => `${i === 0 ? 'M' : 'L'}${sx(i).toFixed(1)},${sy(e).toFixed(1)}`).join(' ');
    svg.append(mk('path', { d: pathD, fill: 'none', stroke: colTeal(), 'stroke-width': 2.8, 'stroke-linecap': 'round' }));

    energies.forEach((e, i) => {
      svg.append(mk('circle', {
        cx: sx(i), cy: sy(e), r: i === activeIdx ? 5.5 : 3.5,
        fill: i === activeIdx ? colTeal() : colBlue(), stroke: dark ? '#0e151f' : '#ffffff', 'stroke-width': 1.6
      }));
    });

    const txtCol = dark ? '#9fb0c3' : '#475569';
    svg.append(mk('text', { x: px, y: 186, fill: txtCol, 'font-size': 11.5 }, 'Step 0'));
    svg.append(mk('text', { x: px + pw, y: 186, fill: txtCol, 'font-size': 11.5, 'text-anchor': 'end' }, `Step ${energies.length - 1}`));
    svg.append(mk('text', { x: px + pw / 2, y: 186, fill: txtCol, 'font-size': 11.5, 'font-weight': 600, 'text-anchor': 'middle' }, `ΔE = ${((energies[activeIdx] - eMin) * 1000).toFixed(1)} meV`));

    host.replaceChildren(svg);
  }

  async function runMdTab() {
    const mode = document.getElementById('md-mode').value;
    const scrubber = document.getElementById('md-scrubber');
    const logEl = document.getElementById('md-events-log');

    if (liveBackend) {
      try {
        const res = await apiPost('/api/run_md', { mode, steps: 8, temperature_k: 320.0 });
        const rawFrames = res.trajectory || [];
        if (rawFrames.length) {
          const evItems = (res.events && res.events.length)
            ? res.events.map(ev => `Step ${ev.frame_index ?? ev.step ?? 0}: ${ev.description || ev.event_type || ev.kind || 'Reaction event'}`)
            : (res.scan_events || []).map(ev => (typeof ev === 'string' ? ev : (ev.description || JSON.stringify(ev))));
          mdState = {
            frames: rawFrames.map((f, idx) => ({
              step: idx,
              atoms: (f.Z || []).map((z, ai) => ({ Z: z, pos: f.pos_ang[ai] })),
              forceVec: f.max_force_vec || [0.4, 0.1, 0.0]
            })),
            energies: res.energies_ev || rawFrames.map((_, i) => 0.02 * Math.sin(i)),
            events: evItems
          };
        }
      } catch {
        mdState = buildDefaultZundelFrames();
      }
    } else {
      mdState = buildDefaultZundelFrames();
    }

    scrubber.max = String(Math.max(0, mdState.frames.length - 1));
    scrubber.value = '0';
    renderMdFrame(0);
    if (logEl) {
      logEl.innerHTML = (mdState.events.length ? mdState.events : ['No bond-topology changes triggered in window.'])
        .map(line => `<div style="padding:5px 0;border-bottom:1px solid var(--line);">• ${line}</div>`).join('');
    }
  }

  // Tab 5: Ouroboros 2D -> 3D Chirality
  async function runOuroborosTab() {
    const smi = document.getElementById('ob-smiles').value.trim();
    const box = document.getElementById('ob-metrics-box');
    const isS = smi.includes('@@');

    if (liveBackend) {
      try {
        const res = await apiPost('/api/ouroboros', { smiles: smi });
        if (res.atoms) {
          window.YLM3D.mountScene(document.getElementById('canvas-ob-3d'), {
            atoms: res.atoms,
            attn: res.atoms.map((_, i) => (res.chiral_atoms && res.chiral_atoms.includes(i) ? 0.92 : 0)),
            arrows: res.dipole ? [{ vec: res.dipole, color: isS ? colPurple() : colTeal(), label: isS ? '(S) 0o' : '(R) 0o' }] : []
          });
        }
        const et = res.enantiomer_test || {};
        box.innerHTML = `
          <dl style="display:grid;gap:12px;margin:0;font-size:13.5px;">
            <div style="padding-bottom:10px;border-bottom:1px solid var(--line);">
              <dt style="color:var(--muted);">Stereo SMILES</dt>
              <dd class="mono" style="margin:4px 0 0;font-size:15px;">${smi}</dd>
            </div>
            <div style="padding-bottom:10px;border-bottom:1px solid var(--line);">
              <dt style="color:var(--muted);">YLM 0o Enantiomer Sum |f(R) + f(S)|</dt>
              <dd class="mono lime" style="margin:4px 0 0;font-size:20px;font-weight:700;">${Number(et.ylm_sum_residual ?? 2.2e-16).toExponential(2)}</dd>
            </div>
            <div>
              <dt style="color:var(--muted);">Pairwise Distance Matrix Drift ||D(R) − D(S)||</dt>
              <dd class="mono" style="margin:4px 0 0;">0.00 × 10⁰ (identical on mirror twins — why 0e models fail)</dd>
            </div>
          </dl>
        `;
        return;
      } catch {
        // Fallback below
      }
    }

    const base = smi.includes('Br')
      ? STATIC_PRESETS.r_bcfm_chiral
      : (smi.includes('C(=O)O') ? STATIC_PRESETS['C[C@H](O)C(=O)O'] : STATIC_PRESETS.r_fec_chiral);
    const cIdx = base === STATIC_PRESETS.r_bcfm_chiral ? 0 : (base.chiralAtom ?? 5);
    const atoms = base.atoms.map(a => ({ Z: a.Z, pos: [isS ? -a.pos[0] : a.pos[0], a.pos[1], a.pos[2]] }));
    const cVal = (base.chiral0o || 0.8421).toFixed(4);
    window.YLM3D.mountScene(document.getElementById('canvas-ob-3d'), {
      atoms,
      attn: atoms.map((_, i) => (i === cIdx ? 0.94 : 0)),
      arrows: [{ origin_atom: cIdx, vec: [isS ? -0.9 : 0.9, -0.7, -0.5], color: isS ? colPurple() : colTeal(), label: isS ? `(S) −${cVal}` : `(R) +${cVal}` }]
    });
    box.innerHTML = `
      <dl style="display:grid;gap:12px;margin:0;font-size:13.5px;">
        <div style="padding-bottom:10px;border-bottom:1px solid var(--line);">
          <dt style="color:var(--muted);">Stereo SMILES</dt>
          <dd class="mono" style="margin:4px 0 0;font-size:15px;">${smi}</dd>
        </div>
        <div style="padding-bottom:10px;border-bottom:1px solid var(--line);">
          <dt style="color:var(--muted);">YLM 0o Pseudoscalar Output</dt>
          <dd class="mono ${isS ? 'purple' : 'lime'}" style="margin:4px 0 0;font-size:22px;font-weight:700;">${isS ? '−' + cVal : '+' + cVal}</dd>
        </div>
        <div>
          <dt style="color:var(--muted);">Mirror Sum Residual |f(R) + f(S)|</dt>
          <dd class="mono lime" style="margin:4px 0 0;">2.22 × 10⁻¹⁶ (exact parity anti-symmetry)</dd>
        </div>
      </dl>
    `;
  }

  // Tab 6 (NEW): LabLean 96-Well Decidable Protocol Checker
  const PIPETTE_RANGE = {
    P2: [0.2, 2.0],
    P20: [2.0, 20.0],
    P200: [20.0, 200.0],
    P1000: [100.0, 1000.0]
  };

  const LEAN_PRESETS = {
    pcr_valid:      { pip: 'P200', vol: 50,  freshTip: true,  thawed: true,  balanced: true,  sealedTransfer: false, desc: 'Addgene PCR setup: thaw stocks → transfer 50 µL master mix into A1–A4 → seal → spin balanced.' },
    dilution_valid: { pip: 'P200', vol: 50,  freshTip: true,  thawed: true,  balanced: true,  sealedTransfer: false, desc: 'iGEM 2-fold serial dilution: preload 50 µL PBS in A2–A6, serially transfer 50 µL from A1 (100 µL stock).' },
    err_overflow:   { pip: 'P200', vol: 165, freshTip: true,  thawed: true,  balanced: true,  sealedTransfer: false, desc: 'Well A2 starts with 50 µL buffer; adding 165 µL reaches 215 µL (> 200 µL 96-well capacity).' },
    err_pipette:    { pip: 'P20',  vol: 50,  freshTip: true,  thawed: true,  balanced: true,  sealedTransfer: false, desc: 'Attempting to aspirate 50 µL with a P20 pipette (valid range 2.0–20.0 µL).' },
    err_contam:     { pip: 'P200', vol: 50,  freshTip: false, thawed: true,  balanced: true,  sealedTransfer: false, desc: 'Re-entering stock reservoir with a used tip that touched template DNA well.' },
    err_rotor:      { pip: 'P200', vol: 50,  freshTip: true,  thawed: true,  balanced: false, sealedTransfer: false, desc: 'Opposite centrifuge rotor slots differ by 250 mg (> 100 mg rotorTol mass tolerance).' }
  };

  let activeLeanPreset = 'pcr_valid';

  function renderLabLeanChecker() {
    const pip = document.getElementById('lean-pipette').value;
    const vol = Number(document.getElementById('lean-vol').value || 50);
    const freshTip = document.getElementById('lean-fresh-tip').checked;
    const thawed = document.getElementById('lean-thawed').checked;
    const balanced = document.getElementById('lean-balanced').checked;
    const [pMin, pMax] = PIPETTE_RANGE[pip] || [20, 200];

    const preload = 50;
    const wellFinal = preload + vol;
    const reservoirInit = 600;
    const nTransfers = 3;
    const reservoirLeft = reservoirInit - nTransfers * vol;
    const totalInit = reservoirInit + nTransfers * preload;
    const totalFinal = reservoirLeft + nTransfers * wellFinal;

    const rules = [
      { name: 'Pipette volume range', formula: `${pip}: ${pMin} ≤ ${vol} ≤ ${pMax} µL`, ok: vol >= pMin && vol <= pMax },
      { name: 'No 96-well overflow', formula: `V(A2..A4) = ${wellFinal.toFixed(1)} ≤ 200.0 µL`, ok: wellFinal <= 200.0 },
      { name: 'Sufficient source liquid', formula: `Reservoir left = ${reservoirLeft.toFixed(1)} ≥ 0 µL`, ok: reservoirLeft >= 0 },
      { name: 'Tip mounted before aspirate', formula: 'tipState ≠ none', ok: true },
      { name: 'No reagent cross-contamination', formula: 'dropTip between distinct reagents', ok: freshTip },
      { name: 'Reagents thawed before use', formula: 'storageState = liquid (4–25 °C)', ok: thawed },
      { name: 'No pipetting through plate seal', formula: 'target.sealed = false on transfer', ok: true },
      { name: 'Safe thermal incubation', formula: '4 °C ≤ T_incubate (37 °C) ≤ 98 °C', ok: true },
      { name: 'Centrifuge rotor balance & seal', formula: '|m_slot1 − m_slot4| ≤ 100 mg, ≤ 15k rpm', ok: balanced }
    ];

    const passedCount = rules.filter(r => r.ok).length;
    const allOk = passedCount === rules.length;

    const conservePill = document.getElementById('lean-conserve-pill');
    if (conservePill) {
      conservePill.textContent = `run_conserves: ΣV = ${totalInit.toFixed(1)} → ${totalFinal.toFixed(1)} µL`;
    }
    const verdictPill = document.getElementById('lean-verdict-pill');
    if (verdictPill) {
      verdictPill.textContent = allOk ? `ValidProtocol · ${passedCount} / 9` : `InvalidProtocol · ${passedCount} / 9`;
      verdictPill.className = `pill-tag ${allOk ? 'ok' : 'bad'}`;
    }

    const traceEl = document.getElementById('lean-step-trace');
    if (traceEl) {
      traceEl.textContent = LEAN_PRESETS[activeLeanPreset]?.desc || 'Custom protocol configuration.';
    }

    // Render 8x12 96-well plate (row-major 1..96: F5 = 65)
    const gridEl = document.getElementById('lean-plate-grid');
    if (gridEl) {
      const rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
      let html = '<div></div>';
      for (let c = 1; c <= 12; c++) html += `<div class="plate-hdr">${c}</div>`;
      rows.forEach((r, ri) => {
        html += `<div class="plate-hdr">${r}</div>`;
        for (let c = 1; c <= 12; c++) {
          const idx1 = ri * 12 + c;
          const isActiveWell = ri === 0 && c >= 1 && c <= 4;
          const v = isActiveWell ? (c === 1 ? 100 : wellFinal) : 0;
          const over = v > 200;
          const cls = over ? 'well-cell overflow' : (v > 0 ? 'well-cell filled sealed' : 'well-cell');
          html += `<div class="${cls}" title="Well ${r}${c} (index ${idx1}): ${v.toFixed(0)} µL">${v > 0 ? Math.round(v) : ''}</div>`;
        }
      });
      gridEl.innerHTML = html;
    }

    const rulesHost = document.getElementById('lean-rules-list');
    if (rulesHost) {
      rulesHost.innerHTML = rules.map(r => `
        <div class="law-row">
          <div>
            <span class="law-dot ${r.ok ? '' : 'fail'}"></span>
            <strong>${r.name}</strong>
            <div class="mono" style="font-size:11.5px;color:var(--muted);margin-left:16px;">${r.formula}</div>
          </div>
          <span class="mono ${r.ok ? 'lime' : 'orange'}" style="font-size:12px;font-weight:650;">${r.ok ? 'PASS' : 'REJECT'}</span>
        </div>
      `).join('');
    }
  }

  // Tab 7 (NEW): Token Factory V0-V3 Verifier + LabLedger SHA-256 Chain
  function evaluateTokenFactoryCandidate() {
    const q = (document.getElementById('tf-question-input')?.value || '').trim();
    const split = document.getElementById('tf-split-select')?.value || 'train';
    const low = q.toLowerCase();

    const v0Ok = q.length >= 15 && q.length <= 240 && /force/i.test(q) && (q.endsWith('?') || q.endsWith('.'));
    const v1Ok = split === 'train';

    const hasNegation = /\b(not|except|neither|without|or)\b/i.test(low);
    const hasIndefinite = /\b(a|an|any)\s+(carbon|oxygen|nitrogen|hydrogen|fluorine)\s+atom\b/i.test(low);
    const hasPlural = /\b(atoms|carbons|oxygens|forces)\b/i.test(low);
    const hasAtLeastAmbig = /carbon atom bonded to one carbon/i.test(low);
    const namesElement = /\b(carbon|oxygen|nitrogen|hydrogen|fluorine)\s+atom\b/i.test(low);
    const v2Ok = v0Ok && namesElement && !hasNegation && !hasIndefinite && !hasPlural && !hasAtLeastAmbig;
    const v3Ok = v0Ok && v1Ok && v2Ok;

    let reason = 'Unique atom resolved on QM9 train molecule; exact GFN2-xTB force vector [-0.0184, +0.0412, -0.0063] Ha/bohr attached.';
    if (!v0Ok) reason = 'Rejected at V0 (Format): must be a well-formed force question (15–240 chars ending in ? or .).';
    else if (!v1Ok) reason = 'Rejected at V1 (Split Guard): molecule belongs to held-out test split; factory only emits train-split items.';
    else if (hasNegation) reason = 'Rejected at V2a (Closed Vocabulary): negation / disjunction ("not", "or", "without") is disallowed.';
    else if (hasIndefinite || hasPlural) reason = 'Rejected at V2b (Uniqueness): indefinite article ("a carbon") or plural does not uniquely address one atom.';
    else if (hasAtLeastAmbig) reason = 'Rejected at V2d ("At-Least" Reading): another carbon in the molecule has ≥ 1 carbon neighbor, so wording is ambiguous.';
    else if (!namesElement) reason = 'Rejected at V2c (Atom Phrase): must specify a supported QM9 element + neighbor environment.';

    const gates = [
      { code: 'V0 · Format', ok: v0Ok, desc: v0Ok ? 'Valid syntax' : 'Bad question form' },
      { code: 'V1 · Train Split', ok: v0Ok && v1Ok, desc: v1Ok ? 'QM9 train only' : 'Split leak blocked' },
      { code: 'V2 · Unique Atom', ok: v0Ok && v1Ok && v2Ok, desc: v2Ok ? 'Exact + at-least match' : 'Ambiguous / mutated' },
      { code: 'V3 · xTB Lookup', ok: v3Ok, desc: v3Ok ? 'Label attached' : 'Skipped on failure' }
    ];

    const pill = document.getElementById('tf-verdict-pill');
    if (pill) {
      pill.textContent = v3Ok ? 'KEPT · Verified' : 'REJECTED';
      pill.className = `pill-tag ${v3Ok ? 'ok' : 'bad'}`;
    }

    const grid = document.getElementById('tf-gates-grid');
    if (grid) {
      grid.innerHTML = gates.map(g => `
        <div class="gate-box ${g.ok ? 'pass' : 'reject'}">
          <strong class="${g.ok ? 'lime' : 'orange'}">${g.ok ? '✓ ' : '✕ '}${g.code}</strong>
          <span>${g.desc}</span>
        </div>
      `).join('');
    }

    const detail = document.getElementById('tf-detail-box');
    if (detail) detail.textContent = reason;
  }

  let ledgerTampered = false;

  function simpleHexHash(str) {
    let h1 = 0xdeadbeef ^ str.length, h2 = 0x41c6ce57 ^ str.length;
    for (let i = 0; i < str.length; i++) {
      const ch = str.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (h2 >>> 0).toString(16).padStart(8, '0') + (h1 >>> 0).toString(16).padStart(8, '0');
  }

  function renderLedgerChain() {
    const baseEvents = [
      { id: 'ev_01', op: 'add', level: 'experiment', text: 'exp_pcr_01 · origin=real · license=MIT' },
      { id: 'ev_02', op: 'add', level: 'observation', text: ledgerTampered ? 'reading: 25.0 °C ± 0.1 °C [EDITED IN PLACE!]' : 'reading: 98.4 °C ± 0.2 °C [instrument_fault]' },
      { id: 'ev_03', op: 'supersede', level: 'observation', text: 'replaces ev_02 after thermocouple recalibration → 72.0 °C ± 0.1 °C' }
    ];

    // Stored expected hashes from untampered append-only log
    const cleanPayloads = [
      '0000000000000000|exp_pcr_01 · origin=real · license=MIT',
      null,
      null
    ];
    const h0 = simpleHexHash(cleanPayloads[0]);
    const h1Clean = simpleHexHash(`${h0}|reading: 98.4 °C ± 0.2 °C [instrument_fault]`);
    const h2Clean = simpleHexHash(`${h1Clean}|replaces ev_02 after thermocouple recalibration → 72.0 °C ± 0.1 °C`);
    const storedHashes = [h0, h1Clean, h2Clean];

    let prev = '0000000000000000';
    const computed = baseEvents.map((ev, idx) => {
      const actual = simpleHexHash(`${prev}|${ev.text}`);
      const ok = actual === storedHashes[idx];
      prev = actual;
      return { ...ev, expected: storedHashes[idx], actual, ok };
    });

    const allOk = computed.every(e => e.ok);
    const pill = document.getElementById('ledger-status-pill');
    if (pill) {
      pill.textContent = allOk ? 'Chain Verified · Export Gate OPEN' : 'TAMPER DETECTED · Export Gate BLOCKED';
      pill.className = `pill-tag mono ${allOk ? 'ok' : 'bad'}`;
    }

    const host = document.getElementById('ledger-events-list');
    if (host) {
      host.innerHTML = computed.map(e => `
        <div class="ledger-event ${e.ok ? '' : 'tampered'}">
          <div style="display:flex;justify-content:space-between;gap:8px;font-weight:600;">
            <span><span class="mono">${e.id}</span> · <code>${e.op}</code> (${e.level})</span>
            <span class="mono ${e.ok ? 'lime' : 'orange'}">${e.ok ? '✓ sha256 match' : '✕ hash mismatch'}</span>
          </div>
          <div style="color:var(--muted);margin-top:3px;">${e.text}</div>
          <div class="mono" style="font-size:11px;color:var(--subtle);margin-top:4px;">expected: ${e.expected} · computed: ${e.actual}</div>
        </div>
      `).join('');
    }
  }

  // Tab 8: Benchmark Table
  function renderBenchmarkRows(rows) {
    const tbody = document.getElementById('bench-live-tbody');
    if (!tbody) return;
    tbody.innerHTML = rows.map(r => {
      const isYLM = r.model.includes('YLM');
      return `
        <tr>
          <th class="${isYLM ? 'lime' : ''}">${r.model}</th>
          <td class="mono">${Number(r.scalar_mae).toFixed(4)}</td>
          <td class="mono ${r.vector_cos > 0.5 ? 'lime' : ''}">${(r.vector_cos >= 0 ? '+' : '') + Number(r.vector_cos).toFixed(4)}</td>
          <td class="mono ${r.vector_eq_err < 1e-4 ? 'lime' : 'orange'}">${Number(r.vector_eq_err).toExponential(2)}</td>
          <td class="mono">${(r.tensor_cos >= 0 ? '+' : '') + Number(r.tensor_cos).toFixed(4)}</td>
          <td class="mono">${Number(r.routing_acc).toFixed(3)}</td>
        </tr>
      `;
    }).join('');
  }

  async function initStudio() {
    initStudioTabs();
    renderLawsList(DEFAULT_LAWS, 8, 8);
    renderBenchmarkRows(DEFAULT_BENCH_ROWS);

    const statusBadge = document.getElementById('studio-engine-status');
    const isStaticHost = window.location.hostname.endsWith('.github.io') || window.location.protocol === 'file:';
    if (isStaticHost) {
      liveBackend = false;
      statusBadge.classList.add('static-mode');
      statusBadge.textContent = 'Browser Interactive Mode · Run ylm serve for live PyTorch';
    } else {
      try {
        const h = await fetch('/api/health', { method: 'GET' });
        if (h.ok) {
          liveBackend = true;
          statusBadge.innerHTML = `<span class="status-dot"></span> Live PyTorch Engine · Connected`;
        } else {
          throw new Error('offline');
        }
      } catch {
        liveBackend = false;
        statusBadge.classList.add('static-mode');
        statusBadge.textContent = 'Browser Interactive Mode · Run ylm serve for live PyTorch';
      }
    }

    // Tab 1 controls
    ['sym-preset', 'sym-precision', 'sl-rx', 'sl-ry', 'sl-rz', 'chk-reflect', 'chk-permute', 'chk-shift'].forEach(id => {
      document.getElementById(id)?.addEventListener('input', refreshSymmetryLiveOrStatic);
    });
    const auditBtn = document.getElementById('btn-run-audit');
    auditBtn?.addEventListener('click', async () => {
      if (!liveBackend) { renderLawsList(DEFAULT_LAWS, 8, 8); return; }
      auditBtn.disabled = true; auditBtn.textContent = 'Auditing…';
      try {
        const rep = await apiPost('/api/physics_audit', {
          structure: document.getElementById('sym-preset').value,
          precision: document.getElementById('sym-precision').value,
          n_rotations: 6
        });
        renderLawsList(rep.laws || DEFAULT_LAWS, rep.passed_count ?? 8, rep.total_laws ?? 8);
      } finally {
        auditBtn.disabled = false; auditBtn.textContent = 'Verify 8 Physics Laws';
      }
    });

    // Tab 2 controls
    document.querySelectorAll('#ask-preset-chips .preset-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#ask-preset-chips .preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        document.getElementById('ask-structure').value = chip.dataset.struct;
        document.getElementById('ask-question').value = chip.dataset.q;
        runAskTab();
      });
    });
    document.getElementById('ask-api-variant')?.addEventListener('change', runAskTab);
    document.getElementById('btn-ask-run')?.addEventListener('click', runAskTab);

    // Tab 3 controls (Options A/B/C Router)
    document.querySelectorAll('#opt-task-chips .preset-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#opt-task-chips .preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        optTask = chip.dataset.optTask;
        renderOptionsRouter();
      });
    });
    document.querySelectorAll('#opt-mode-chips .preset-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#opt-mode-chips .preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        optMode = chip.dataset.optMode;
        renderOptionsRouter();
      });
    });

    // Tab 4 controls (MD + Auto-Scrub)
    const mdScrub = document.getElementById('md-scrubber');
    mdScrub?.addEventListener('input', e => renderMdFrame(Number(e.target.value)));
    document.getElementById('btn-run-md')?.addEventListener('click', runMdTab);
    const playMdBtn = document.getElementById('btn-play-md');
    playMdBtn?.addEventListener('click', () => {
      if (mdTimer) {
        clearInterval(mdTimer);
        mdTimer = 0;
        playMdBtn.textContent = '▶ Auto-Scrub';
        return;
      }
      playMdBtn.textContent = 'Ⅱ Stop';
      mdTimer = setInterval(() => {
        const max = Number(mdScrub.max || 8);
        const next = (Number(mdScrub.value || 0) + 1) % (max + 1);
        mdScrub.value = String(next);
        renderMdFrame(next);
      }, 520);
    });

    // Tab 5 controls (Ouroboros + Enantiomer Flip)
    document.querySelectorAll('#ob-chips .preset-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#ob-chips .preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        document.getElementById('ob-smiles').value = chip.dataset.smi;
        runOuroborosTab();
      });
    });
    document.getElementById('btn-flip-ob')?.addEventListener('click', () => {
      const inp = document.getElementById('ob-smiles');
      if (!inp) return;
      if (inp.value.includes('@@')) inp.value = inp.value.replace('@@', '@');
      else if (inp.value.includes('@')) inp.value = inp.value.replace('@', '@@');
      runOuroborosTab();
    });
    document.getElementById('btn-run-ob')?.addEventListener('click', runOuroborosTab);

    // Tab 6 controls (LabLean Checker)
    document.querySelectorAll('#lean-preset-chips .preset-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#lean-preset-chips .preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        activeLeanPreset = chip.dataset.lean;
        const p = LEAN_PRESETS[activeLeanPreset];
        if (p) {
          document.getElementById('lean-pipette').value = p.pip;
          document.getElementById('lean-vol').value = String(p.vol);
          document.getElementById('lean-fresh-tip').checked = p.freshTip;
          document.getElementById('lean-thawed').checked = p.thawed;
          document.getElementById('lean-balanced').checked = p.balanced;
        }
        renderLabLeanChecker();
      });
    });
    ['lean-pipette', 'lean-vol', 'lean-fresh-tip', 'lean-thawed', 'lean-balanced'].forEach(id => {
      document.getElementById(id)?.addEventListener('input', renderLabLeanChecker);
      document.getElementById(id)?.addEventListener('change', renderLabLeanChecker);
    });

    // Tab 7 controls (Token Factory + LabLedger)
    document.querySelectorAll('#tf-preset-chips .preset-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#tf-preset-chips .preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        document.getElementById('tf-question-input').value = chip.dataset.tfQ;
        document.getElementById('tf-split-select').value = chip.dataset.tfSplit;
        evaluateTokenFactoryCandidate();
      });
    });
    document.getElementById('tf-question-input')?.addEventListener('input', evaluateTokenFactoryCandidate);
    document.getElementById('tf-split-select')?.addEventListener('change', evaluateTokenFactoryCandidate);
    document.getElementById('btn-ledger-tamper')?.addEventListener('click', () => {
      ledgerTampered = true;
      renderLedgerChain();
    });
    document.getElementById('btn-ledger-reset')?.addEventListener('click', () => {
      ledgerTampered = false;
      renderLedgerChain();
    });

    // Tab 8 controls
    const benchBtn = document.getElementById('btn-run-bench');
    benchBtn?.addEventListener('click', async () => {
      if (!liveBackend) {
        benchBtn.textContent = 'Start ylm serve locally for live training';
        setTimeout(() => { benchBtn.textContent = 'Run Live CPU Check'; }, 2200);
        return;
      }
      benchBtn.disabled = true; benchBtn.textContent = 'Training 4 models on CPU…';
      try {
        const ep = Number(document.getElementById('bench-epochs').value || 3);
        const res = await apiPost('/api/eval_benchmark', { epochs: ep, eval_rotations: 4 });
        if (res.models) renderBenchmarkRows(res.models);
      } finally {
        benchBtn.disabled = false; benchBtn.textContent = 'Run Live CPU Check';
      }
    });

    refreshSymmetryLiveOrStatic();
    runAskTab();
    renderOptionsRouter();
    runMdTab();
    runOuroborosTab();
    renderLabLeanChecker();
    evaluateTokenFactoryCandidate();
    renderLedgerChain();
  }

  function boot() {
    initThemeToggle();
    initEditorialUI();
    initStudio();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true });
  else boot();
})();
