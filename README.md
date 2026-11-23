# YLM Project Website & Interactive Studio

Static research website for **YLM** (`https://yaniguan.github.io/YLM/`), also served locally with live PyTorch inference by `ylm serve`.

## Two ways to view

1. **Static GitHub Pages (no backend or build step required)**:
   Publish `site/` directly (via `.github/workflows/pages.yml`). `.nojekyll` keeps plain static assets available as-is.
2. **Local live PyTorch mode**:
   Run `ylm serve --port 8765` from the repository root and open `http://127.0.0.1:8765/` to connect the interactive Studio tabs to live `YLM` checkpoints, 64-bit 8-law physics audits, reactive MD hooks, and RDKit/Ouroboros 2D-to-3D stereo embedding.
