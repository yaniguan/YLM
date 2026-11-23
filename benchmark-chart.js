/* Dependency-free responsive theme-aware SVG benchmark explorer for YLM. */
(() => {
  'use strict';

  const scriptURL = document.currentScript ? document.currentScript.src : document.baseURI;
  const dataURL = new URL('assets/benchmark-data.json', scriptURL);
  const NS = 'http://www.w3.org/2000/svg';

  function getPalette() {
    const dark = document.documentElement.dataset.theme === 'dark';
    return dark
      ? { bg: '#111924', plot: '#0d141e', text: '#edf2f7', muted: '#9fb0c3', grid: '#223041', axis: '#5c7087', whisker: '#e2e8f0' }
      : { bg: '#ffffff', plot: '#f8fafc', text: '#0f172a', muted: '#475569', grid: '#e2e8f0', axis: '#94a3b8', whisker: '#334155' };
  }

  function el(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function init() {
    const host = document.getElementById('benchmark-chart');
    const select = document.getElementById('benchmark-select');
    const sdToggle = document.getElementById('benchmark-show-sd');
    const legend = document.getElementById('benchmark-chart-legend');
    const caption = document.getElementById('benchmark-chart-caption');
    if (!host || !select || !legend || !caption) return;

    try {
      const resp = await fetch(dataURL);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const views = data.views || [];
      const methodsByName = new Map((data.methods || []).map(m => [m.name, m]));

      select.replaceChildren();
      for (const v of views) {
        const opt = document.createElement('option');
        opt.value = v.key;
        opt.textContent = v.title;
        select.append(opt);
      }
      select.value = views[0].key;

      function renderLegend(items, isLine) {
        legend.replaceChildren();
        for (const item of items) {
          const wrap = document.createElement('span');
          wrap.className = 'legend-item';
          const sw = el('svg', { viewBox: '0 0 44 12', width: 44, height: 12, 'aria-hidden': 'true' });
          if (isLine) {
            sw.append(el('line', {
              x1: 2, y1: 6, x2: 42, y2: 6,
              stroke: item.color,
              'stroke-width': item.primary ? 3.4 : 2.3,
              'stroke-linecap': 'round',
              'stroke-dasharray': item.dash && item.dash.length ? item.dash.join(' ') : 'none'
            }));
            sw.append(el('circle', { cx: 22, cy: 6, r: item.primary ? 3.8 : 3.0, fill: item.color }));
          } else {
            sw.append(el('rect', { x: 8, y: 2, width: 28, height: 8, rx: 3, fill: item.color }));
          }
          const lbl = document.createElement('span');
          lbl.textContent = item.name;
          wrap.append(sw, lbl);
          legend.append(wrap);
        }
      }

      function render() {
        const PALETTE = getPalette();
        const view = views.find(v => v.key === select.value) || views[0];
        const showSD = sdToggle ? sdToggle.checked : true;
        caption.textContent = view.caption || '';

        const width = host.getBoundingClientRect().width || 980;
        const compact = width < 580;
        const fSize = compact ? 17 : 14;
        const axSize = compact ? 18 : 14.5;
        const plot = { x: 90, y: 28, w: 882, h: 344 };
        const yMin = view.yRange[0];
        const yMax = Math.max(view.yRange[1], yMin + 0.2);
        const sy = val => plot.y + plot.h - ((val - yMin) / (yMax - yMin)) * plot.h;

        const svg = el('svg', {
          viewBox: '0 0 1000 450',
          role: 'img',
          'aria-label': view.title,
          preserveAspectRatio: 'xMidYMid meet'
        });
        svg.append(el('rect', { width: 1000, height: 450, fill: PALETTE.bg, rx: 8 }));
        svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, fill: PALETTE.plot, rx: 6 }));

        for (const yv of view.yTicks) {
          const y = sy(yv);
          svg.append(el('line', {
            x1: plot.x, y1: y, x2: plot.x + plot.w, y2: y,
            stroke: PALETTE.grid, 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke'
          }));
          svg.append(el('text', {
            x: plot.x - 12, y, fill: PALETTE.muted,
            'font-size': fSize, 'text-anchor': 'end', 'dominant-baseline': 'middle'
          }, yv.toFixed(2)));
        }

        if (view.kind === 'line') {
          const logMin = Math.log10(view.xTicks[0]);
          const logMax = Math.log10(view.xTicks[view.xTicks.length - 1]);
          const sx = xv => plot.x + 42 + ((Math.log10(xv) - logMin) / (logMax - logMin)) * (plot.w - 84);

          for (const xv of view.xTicks) {
            const x = sx(xv);
            svg.append(el('line', {
              x1: x, y1: plot.y + plot.h, x2: x, y2: plot.y + plot.h + 6,
              stroke: PALETTE.axis, 'stroke-width': 1.2
            }));
            svg.append(el('text', {
              x, y: plot.y + plot.h + 25, fill: PALETTE.muted,
              'font-size': fSize, 'text-anchor': 'middle'
            }, `${xv}%`));
          }

          const legendList = [];
          for (const s of view.series) {
            const meta = methodsByName.get(s.name) || { name: s.name, color: '#0d9488', dash: [], primary: false };
            legendList.push(meta);

            if (showSD) {
              const upper = s.points.map(([x, y, sd]) => [sx(x), sy(Math.min(yMax, y + (sd || 0)))]);
              const lower = [...s.points].reverse().map(([x, y, sd]) => [sx(x), sy(Math.max(yMin, y - (sd || 0)))]);
              const bandD = [...upper, ...lower].map((p, idx) => `${idx === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ') + ' Z';
              svg.append(el('path', { d: bandD, fill: meta.color, 'fill-opacity': meta.primary ? 0.16 : 0.09 }));
            }

            const d = s.points.map(([x, y], idx) => `${idx === 0 ? 'M' : 'L'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');
            svg.append(el('path', {
              d, fill: 'none', stroke: meta.color,
              'stroke-width': meta.primary ? 3.2 : 2.3,
              'stroke-linecap': 'round', 'stroke-linejoin': 'round',
              'stroke-dasharray': meta.dash && meta.dash.length ? meta.dash.join(' ') : 'none'
            }));

            for (const [x, y, sd] of s.points) {
              const cx = sx(x), cy = sy(y);
              if (showSD && sd > 0) {
                const yTop = sy(y + sd), yBot = sy(y - sd);
                svg.append(el('line', { x1: cx, y1: yTop, x2: cx, y2: yBot, stroke: meta.color, 'stroke-width': 1.8 }));
                svg.append(el('line', { x1: cx - 5, y1: yTop, x2: cx + 5, y2: yTop, stroke: meta.color, 'stroke-width': 1.8 }));
                svg.append(el('line', { x1: cx - 5, y1: yBot, x2: cx + 5, y2: yBot, stroke: meta.color, 'stroke-width': 1.8 }));
              }
              const dot = el('circle', {
                cx, cy, r: meta.primary ? 5.2 : 4.2,
                fill: meta.color, stroke: PALETTE.bg, 'stroke-width': 1.8
              });
              dot.append(el('title', {}, `${s.name} (${x}%): ${y.toFixed(3)}${sd ? ` ± ${sd.toFixed(3)}` : ''}`));
              svg.append(dot);
              if (x === 100) {
                svg.append(el('text', {
                  x: cx + 10, y: cy + (s.name.includes('MiniLM') ? 13 : -8),
                  fill: meta.color, 'font-size': 13, 'font-weight': 650
                }, y.toFixed(3)));
              }
            }
          }
          svg.append(el('text', {
            x: plot.x + plot.w / 2, y: 428, fill: PALETTE.muted,
            'font-size': axSize, 'text-anchor': 'middle'
          }, view.xLabel));
          renderLegend(legendList, true);
        } else {
          const cats = view.categories;
          const bars = view.bars;
          const groupW = plot.w / cats.length;
          const barW = Math.min(36, (groupW * 0.68) / bars.length);

          cats.forEach((cat, ci) => {
            const gx = plot.x + groupW * (ci + 0.5);
            svg.append(el('text', {
              x: gx, y: plot.y + plot.h + 26, fill: PALETTE.muted,
              'font-size': compact ? 13.5 : 13, 'text-anchor': 'middle'
            }, cat));

            const totalW = bars.length * barW + (bars.length - 1) * 6;
            const startX = gx - totalW / 2;
            bars.forEach((b, bi) => {
              const val = b.values[ci];
              if (val === null || val === undefined) return;
              const bx = startX + bi * (barW + 6);
              const by = sy(val);
              const bh = Math.max(2, plot.y + plot.h - by);
              const rect = el('rect', {
                x: bx, y: by, width: barW, height: bh, rx: 4,
                fill: b.color, 'fill-opacity': 0.90
              });
              const sd = (b.sds && b.sds[ci]) || 0;
              rect.append(el('title', {}, `${b.name} — ${cat}: ${val.toFixed(3)}${sd ? ` ± ${sd.toFixed(3)}` : ''}`));
              svg.append(rect);

              let labelAnchorY = by - 7;
              if (showSD && sd > 0) {
                const cx = bx + barW / 2;
                const yTop = sy(Math.min(yMax, val + sd));
                const yBot = sy(Math.max(yMin, val - sd));
                svg.append(el('line', { x1: cx, y1: yTop, x2: cx, y2: yBot, stroke: PALETTE.whisker, 'stroke-width': 1.5 }));
                svg.append(el('line', { x1: cx - 4.5, y1: yTop, x2: cx + 4.5, y2: yTop, stroke: PALETTE.whisker, 'stroke-width': 1.5 }));
                labelAnchorY = Math.min(by, yTop) - 7;
              }

              svg.append(el('text', {
                x: bx + barW / 2, y: labelAnchorY,
                fill: PALETTE.text, 'font-size': 11, 'font-weight': 650, 'text-anchor': 'middle'
              }, val.toFixed(2)));
            });
          });
          renderLegend(bars, false);
        }

        svg.append(el('path', {
          d: `M${plot.x},${plot.y}V${plot.y + plot.h}H${plot.x + plot.w}`,
          fill: 'none', stroke: PALETTE.axis, 'stroke-width': 1.2, 'vector-effect': 'non-scaling-stroke'
        }));
        svg.append(el('text', {
          transform: `translate(24 ${plot.y + plot.h / 2}) rotate(-90)`,
          fill: PALETTE.muted, 'font-size': axSize, 'text-anchor': 'middle'
        }, view.yLabel));

        host.replaceChildren(svg);
      }

      select.addEventListener('change', render);
      if (sdToggle) sdToggle.addEventListener('change', render);
      window.addEventListener('ylm:themechange', render);
      render();
      let raf = 0;
      const onResize = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(render); };
      if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(host);
      else window.addEventListener('resize', onResize, { passive: true });
    } catch (err) {
      host.textContent = 'Benchmark figure data could not be loaded.';
      console.error(err);
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
