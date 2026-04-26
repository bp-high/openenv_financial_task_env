#!/usr/bin/env python3
"""Overlay multiple SFT training-loss curves on a single plot for ablation.

Reads the per-step ``training_metrics.jsonl`` files produced by
``analyze_sft_logs.py`` and emits a single PNG with all runs overlaid.

Usage:
    python data_pipeline/compare_sft_runs.py \\
        --runs runs/sft_plots/qwen3b_kimi:"max-seq=4K" \\
               runs/sft_plots/qwen3b_kimi_long:"max-seq=8K" \\
        --output runs/sft_plots/comparison.png \\
        --title "Qwen2.5-Coder-3B + LoRA — context length ablation"
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_metrics(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def rolling(xs, win):
    return [
        statistics.fmean(xs[max(0, i - win):i + 1])
        for i in range(len(xs))
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="entries of the form '<run_dir>:<label>'")
    p.add_argument("--output", default="runs/sft_plots/comparison.png")
    p.add_argument("--title", default="SFT loss comparison")
    p.add_argument("--rolling-window", type=int, default=3)
    args = p.parse_args()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, spec in enumerate(args.runs):
        run_dir, _, label = spec.partition(":")
        if not label:
            label = run_dir
        metrics_path = Path(run_dir) / "training_metrics.jsonl"
        if not metrics_path.exists():
            print(f"WARNING: {metrics_path} not found, skipping")
            continue
        metrics = load_metrics(metrics_path)
        losses = [m.get("loss") for m in metrics if "loss" in m]
        steps = list(range(1, len(losses) + 1))
        color = palette[i % len(palette)]
        ax.plot(steps, losses, color=color, linewidth=1.0, alpha=0.4)
        ax.plot(steps, rolling(losses, args.rolling_window),
                color=color, linewidth=2.0, label=f"{label}  (final={losses[-1]:.3f})")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(alpha=0.3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())