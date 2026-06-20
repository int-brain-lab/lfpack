Local development
-----------------

1. Install dependencies (once):
     uv sync --group dev

2. Regenerate API reference from docstrings (run after editing _core.py):
     uv run quartodoc build --config docs/_quarto.yml

3. Live-preview the site:
     quarto preview docs/

The preview server reloads automatically when .qmd files change.
Re-run step 2 manually whenever docstrings change — quartodoc output
is not watched by the preview server.


Production build
----------------

The site is built and deployed automatically by GitHub Actions on every
push to main (.github/workflows/docs.yml).  The workflow runs the same
three steps above (quartodoc build → quarto render → deploy to GitHub Pages).

Do not commit the generated _site/ directory; it is built in CI.