"""Generate source-backed, dependency-free SVG Pareto diagrams for Task 2.

Each diagram uses metrics from a completed experiment.json. The test scatter
is DESCRIPTIVE only: test results are never used for checkpoint selection.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path


def group_of(name: str) -> str:
    if name == "gpt2":
        return "GPT-2"
    if name.startswith("rl_"):
        return "RL"
    if name.startswith("rerank_"):
        return "Rerank"
    return "Simple baseline"


COLORS = {"GPT-2": "#0c7a66", "RL": "#3b5cc4",
          "Rerank": "#be6f16", "Simple baseline": "#8a4ab1"}


def draw_tradeoff(metrics: dict, source_kind: str, split: str,
                  selected: str, output: Path) -> None:
    data = []
    for name, info in sorted(metrics.items()):
        x, y = float(info["ndcg_at_10"]), float(info["auditor_completion_at_10"])
        if math.isfinite(x) and math.isfinite(y):
            data.append((name, x, y, group_of(name)))
    if not data:
        raise ValueError("Cannot plot a split without finite NDCG and reward observations")
    width, height = 1080, max(560, 140 + 22 * len(data))
    l, t, w, h = 95, 92, 620, height - 185
    xx = [p[1] for p in data]
    yy = [p[2] for p in data]

    def bounds(values):
        minimum, maximum = min(values), max(values)
        delta = max(maximum - minimum, 0.01)
        pad = .10 * delta
        return minimum - pad, maximum + pad

    x0, x1 = bounds(xx)
    y0, y1 = bounds(yy)

    def sx(value):
        return l + (value - x0) / (x1 - x0) * w

    def sy(value):
        return t + (1 - (value - y0) / (y1 - y0)) * h

    title = f"T-Lab 2026 · {split.upper()} tradeoff"
    subtitle = (
        "SYNTHETIC SOFTWARE SMOKE — NOT RESEARCH DATA" if source_kind == "synthetic"
        else "Yambda-50M · offline model-based reward, not causal user retention"
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{l}" y="34" font-size="24" font-family="sans-serif" fill="#162033">{escape(title)}</text>',
        f'<text x="{l}" y="59" font-size="13" font-family="sans-serif" fill="#556078">{escape(subtitle)}</text>',
    ]
    for i in range(5):
        frac = i / 4
        x, y = l + frac * w, t + frac * h
        lines += [
            f'<line x1="{x:.1f}" y1="{t}" x2="{x:.1f}" y2="{t+h}" stroke="#e2e6ec"/>',
            f'<line x1="{l}" y1="{y:.1f}" x2="{l+w}" y2="{y:.1f}" stroke="#e2e6ec"/>',
            f'<text x="{x:.1f}" y="{t+h+23}" text-anchor="middle" font-size="12" font-family="sans-serif">{x0+frac*(x1-x0):.4f}</text>',
            f'<text x="{l-11}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif">{y1-frac*(y1-y0):.3f}</text>',
        ]
    lines += [
        f'<line x1="{l}" y1="{t+h}" x2="{l+w}" y2="{t+h}" stroke="#333d4e" stroke-width="1.5"/>',
        f'<line x1="{l}" y1="{t}" x2="{l}" y2="{t+h}" stroke="#333d4e" stroke-width="1.5"/>',
        f'<text x="{l+w/2}" y="{height-48}" text-anchor="middle" font-size="16" font-family="sans-serif">NDCG@10 — includes out-of-vocabulary misses</text>',
        f'<text transform="translate(22 {t+h/2}) rotate(-90)" text-anchor="middle" font-size="15" font-family="sans-serif">Auditor predicted completion</text>',
        '<text x="748" y="99" font-size="16" font-family="sans-serif" font-weight="bold">All compared policies</text>',
    ]
    for i, (name, x, y, group) in enumerate(data):
        px, py, color = sx(x), sy(y), COLORS[group]
        selected_point = name == selected
        lines += [
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{8 if selected_point else 5}" fill="{color}" fill-opacity=".75" stroke="#ffffff" stroke-width="1.5">',
            f'<title>{escape(name)}: NDCG={x:.5f}; RM={y:.5f}</title></circle>',
            f'<circle cx="760" cy="{123+i*22}" r="5" fill="{color}"/>',
            f'<text x="774" y="{127+i*22}" font-size="12" font-family="monospace" fill="#263040">{escape(name[:43])}</text>',
        ]
        if selected_point:
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="11" fill="none" stroke="{color}" stroke-width="2"/>')
    lines += [
        f'<text x="750" y="{height-28}" font-size="12" font-family="sans-serif" fill="#526078">'
        + escape("Circled = selected on validation (never on test)") + '</text>',
        '</svg>',
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    source = payload.get("source_kind")
    if source not in ("synthetic", "yambda"):
        raise ValueError("Unknown source-kind; refusing ungrounded research visualizations")
    for split in ("validation", "test"):
        draw_tradeoff(payload[split], source, split,
                      payload["selected_on_validation"],
                      args.outdir / f"tradeoff_{split}.svg")


if __name__ == "__main__":
    main()
