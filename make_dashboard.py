"""
Steps 3, 4, 7 -- dashboard generator.

Reads the saved scoring output (data/step6_balanced.jsonl, optionally
data/step2_first100.jsonl) and emits ONE self-contained, offline HTML file
(dashboard.html) with:

  * headline metric cards
  * an "imbalanced vs balanced" comparison strip (uses the step-2 run when present)
  * a star-rating distribution
  * answer-vs-prediction grouped bars and per-class accuracy
  * a 3x3 confusion matrix
  * LLM-emotion vs NRC-word-list-emotion distributions + agreement
  * an interactive, filterable review table (correct/mismatch, class, emotion,
    text search) with a live count

Everything the page shows is computed in the browser FROM the saved JSONL
records -- the same file the summary numbers were derived from -- so the on-page
figures cannot drift from the saved output. Charts are pure HTML/CSS (no
canvas, no external libraries) so the file works fully offline and nothing can
collapse to zero width.

The theme is driven by CSS custom properties in :root; a small palette picker
lets the host recolor the whole page in one click.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT_HTML = HERE / "dashboard.html"


def _load_rows(path: Path):
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_HTML)
    ap.add_argument("--step6", default=HERE / "data" / "step6_balanced.jsonl")
    ap.add_argument("--step2", default=HERE / "data" / "step2_first100.jsonl")
    args = ap.parse_args()

    step6 = _load_rows(Path(args.step6))
    step2 = _load_rows(Path(args.step2))
    if not step6:
        raise SystemExit("No step6 data found -- run `python classify.py step6` first.")

    payload = {
        "step6": step6,
        "step2": step2 or [],
        "step6_summary": _load_summary(HERE / "data" / "step6_summary.json"),
        "step2_summary": _load_summary(HERE / "data" / "step2_summary.json"),
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    html_template = (HERE / "_dashboard_template.html").read_text(encoding="utf-8")
    html = html_template.replace("__DATA_JSON__", data_json)
    Path(args.out).write_text(html, encoding="utf-8")
    out = Path(args.out)
    print(f"Dashboard written to {out} ({out.stat().st_size/1024:.0f} kB)")


def _load_summary(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


if __name__ == "__main__":
    build()
