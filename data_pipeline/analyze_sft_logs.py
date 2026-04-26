#!/usr/bin/env python3
"""Pull HF Jobs logs for an SFT training run, parse training metrics, plot.

Two input modes:

  1. ``--job-id <id>`` — pulls fresh logs via huggingface_hub.fetch_job_logs.
                         Requires HF auth.
  2. ``--logs-file <path>`` — read pre-saved logs from a file.

The transformers ``Trainer`` logs metrics as Python-dict-formatted lines on
stdout, e.g.::

    {'loss': 1.4523, 'grad_norm': 0.82, 'learning_rate': 2e-05, 'epoch': 0.04}

The end-of-training summary looks like::

    {'train_runtime': 312.4, 'train_samples_per_second': ..., 'train_loss': 0.78, 'epoch': 4.0}

We parse both forms, save a clean JSONL of step-level metrics, and emit a
2-panel PNG (loss curve + LR schedule).

Outputs land in ``--output-dir`` (default ``runs/sft_plots/``):

    raw_logs.txt           # the full job stdout (so future replays are offline)
    training_metrics.jsonl # one record per step
    summary.json           # final-epoch summary metrics
    sft_loss_curve.png     # 2-panel: loss + LR vs step

Usage:
    python data_pipeline/analyze_sft_logs.py \\
        --job-id 69ed74aed70108f37acdf4fc \\
        --output-dir runs/sft_plots/qwen3b_kimi
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# Match a Python-dict-printed log line.  We only catch the lines that contain
# 'loss' or 'train_loss' so we don't slurp every {…} the script ever printed.
_DICT_LINE_RE = re.compile(r"^\{.*?(?:'loss'|'train_loss').*\}$")


_NUMERIC_KEYS = {
    "loss", "learning_rate", "epoch", "grad_norm",
    "train_loss", "train_runtime", "train_samples_per_second",
    "train_steps_per_second", "step",
}


def _coerce_numeric(d: dict[str, Any]) -> dict[str, Any]:
    """Cast known-numeric keys to float so downstream formatting/plotting works."""
    out = {}
    for k, v in d.items():
        if k in _NUMERIC_KEYS:
            try:
                out[k] = float(v)
                continue
            except (TypeError, ValueError):
                pass
        out[k] = v
    return out


def parse_logs(text: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return (per_step_metrics, final_summary).

    ``per_step_metrics`` is a list of dicts with keys like loss/lr/epoch/grad_norm.
    ``final_summary`` is the train_runtime/train_loss/etc. dict if present.
    All numeric fields are coerced to float (the JSON path can leave them as
    strings if Python repr was unusual).
    """
    metrics: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not _DICT_LINE_RE.match(line):
            continue
        # Trainer prints with single quotes; convert to JSON-friendly double quotes
        # carefully (avoid touching strings that legitimately contain quotes).
        # The Trainer output is simple enough that a wholesale replace works.
        try:
            d = json.loads(line.replace("'", '"'))
        except json.JSONDecodeError:
            # Fallback for floats with weird repr or NaN
            try:
                d = eval(line, {"__builtins__": {}}, {"nan": float("nan"), "inf": float("inf")})
            except Exception:
                continue

        d = _coerce_numeric(d)

        if "train_loss" in d and "train_runtime" in d:
            final = d
        elif "loss" in d:
            metrics.append(d)

    return metrics, final


def fetch_logs(job_id: str) -> str:
    """Use huggingface_hub Python API to stream all logs for a job."""
    from huggingface_hub import fetch_job_logs
    return "\n".join(line for line in fetch_job_logs(job_id=job_id))


def make_plot(metrics: list[dict], out_path: Path, title_suffix: str = "") -> None:
    import matplotlib.pyplot as plt

    steps = [i + 1 for i in range(len(metrics))]
    losses = [m.get("loss") for m in metrics]
    lrs = [m.get("learning_rate") for m in metrics]
    grad_norms = [m.get("grad_norm") for m in metrics if "grad_norm" in m]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))

    # Loss
    axs[0].plot(steps, losses, color="#1f77b4", linewidth=1.6, label="train loss")
    # Optional rolling average for readability
    if len(losses) >= 5:
        import statistics
        win = max(3, len(losses) // 20)
        rolled = [
            statistics.fmean([x for x in losses[max(0, i - win):i + 1] if x is not None])
            for i in range(len(losses))
        ]
        axs[0].plot(steps, rolled, color="#ff7f0e", linewidth=1.8, label=f"rolling avg (window={win})")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("loss")
    axs[0].set_title(f"SFT training loss{title_suffix}")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # LR
    axs[1].plot(steps, lrs, color="#2ca02c", linewidth=1.6)
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("learning rate")
    axs[1].set_title("LR schedule")
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--job-id", help="HF Job ID (e.g. 69ed74aed70108f37acdf4fc)")
    src.add_argument("--logs-file", help="Pre-saved logs file (skip API fetch)")
    p.add_argument("--output-dir", default="runs/sft_plots",
                   help="where to write training_metrics.jsonl, sft_loss_curve.png, etc.")
    p.add_argument("--title-suffix", default="",
                   help="appended to plot title (e.g. ' — Qwen2.5-Coder-3B + LoRA')")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Get raw logs ----
    if args.job_id:
        print(f"Fetching logs for job {args.job_id} via HF API …")
        text = fetch_logs(args.job_id)
        (out_dir / "raw_logs.txt").write_text(text)
        print(f"  saved to {out_dir / 'raw_logs.txt'} ({len(text)} chars)")
    else:
        text = Path(args.logs_file).read_text()
        print(f"Read {len(text)} chars from {args.logs_file}")

    # ---- 2. Parse ----
    metrics, final = parse_logs(text)
    print(f"\nParsed {len(metrics)} per-step metrics records")
    if final:
        def _fmt(x, spec=".4f"):
            try:
                return format(float(x), spec)
            except (TypeError, ValueError):
                return str(x)
        print(f"Final summary: train_loss={_fmt(final.get('train_loss'))}, "
              f"runtime={_fmt(final.get('train_runtime'), '.0f')}s, "
              f"epoch={final.get('epoch')}")
    else:
        print("(no end-of-training summary line found)")

    if not metrics:
        print("ERROR: no training metrics parsed.  "
              "Either the run failed early or the log format changed.",
              file=sys.stderr)
        return 1

    # ---- 3. Persist parsed records ----
    with open(out_dir / "training_metrics.jsonl", "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    if final:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(final, f, indent=2)

    # Headline numbers in stdout
    losses = [m["loss"] for m in metrics if "loss" in m]
    if losses:
        print(f"\nLoss trajectory:  start={losses[0]:.3f}  →  end={losses[-1]:.3f}  "
              f"(min={min(losses):.3f})")

    # ---- 4. Plot ----
    plot_path = out_dir / "sft_loss_curve.png"
    try:
        make_plot(metrics, plot_path, title_suffix=args.title_suffix)
        print(f"\nPlot saved: {plot_path}")
    except ImportError:
        print("\nmatplotlib not installed — skipped plot. "
              "Run `pip install matplotlib` to enable.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
