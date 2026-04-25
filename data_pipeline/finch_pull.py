"""Stratified puller for the FinWorkBench/Finch dataset.

Selects 50 tasks across the most-frequent task_type tags, downloads source
and reference xlsx files, and emits a manifest.jsonl row for each task.

Usage:
    python data_pipeline/finch_pull.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "finch_50"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"

# Per-tag pick budgets.  Sum = 50.  Web Search tasks have non-xlsx sources
# (web/PDF), so we drop them and reallocate slots to denser tags.
TAG_BUDGET = {
    "Calculation": 16,
    "Structuring / Formatting": 11,
    "Data Entry / Import": 6,
    "Validation / Review": 5,
    "Cross-sheet/file Retrieval": 5,
    "Summary / Visualization": 4,
    "Financial Modeling": 3,
}

# Eval holdout per tag (rest go to train).  Sum = 10.
EVAL_HOLDOUT = {
    "Calculation": 3,
    "Structuring / Formatting": 2,
    "Data Entry / Import": 1,
    "Validation / Review": 1,
    "Cross-sheet/file Retrieval": 1,
    "Summary / Visualization": 1,
    "Financial Modeling": 1,
}


def primary_tag(task_type: str) -> str:
    """Return the first tag in the comma-separated task_type field."""
    return task_type.split(",")[0].strip()


def is_xlsx_task(row) -> bool:
    """Pure-xlsx tasks: source files are all xlsx, reference output is xlsx."""
    srcs = row["source_files"]
    if not srcs or any(not s.lower().endswith(".xlsx") for s in srcs):
        return False
    refs = row["reference_outputs"].get("files") or []
    if refs and any(not r.lower().endswith(".xlsx") for r in refs):
        return False
    return True


def download(url: str, dest: Path, timeout: float = 30.0, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
                f.write(r.read())
            return
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"download failed after {retries} retries: {last_exc}")


def select(ds, seed: int = 17) -> dict:
    """Return {tag: [row, ...]} sized per TAG_BUDGET, xlsx-only, single-source."""
    rng = random.Random(seed)
    by_primary: dict[str, list] = defaultdict(list)
    for row in ds:
        if not is_xlsx_task(row):
            continue
        if len(row["source_files"]) != 1:
            continue
        if not row["reference_outputs"].get("files"):
            # Skip pure-QA for now; MODIFY tasks dominate and grade cleanly.
            continue
        by_primary[primary_tag(row["task_type"])].append(row)

    picked: dict[str, list] = {}
    for tag, budget in TAG_BUDGET.items():
        pool = by_primary.get(tag, [])
        rng.shuffle(pool)
        picked[tag] = pool[:budget]
        if len(picked[tag]) < budget:
            print(f"  ⚠ tag {tag!r}: wanted {budget}, got {len(picked[tag])}")
    return picked


def emit_manifest(picked: dict) -> list[dict]:
    """Download files and build manifest rows.  Returns the list of rows."""
    rows: list[dict] = []
    rng = random.Random(31)

    for tag, items in picked.items():
        rng.shuffle(items)
        eval_n = EVAL_HOLDOUT.get(tag, 0)
        for i, row in enumerate(items):
            split = "eval" if i < eval_n else "train"
            tid = f"finch_{row['id']}"
            task_dir = DATA_DIR / row["id"]

            src_name = row["source_files"][0]
            src_url = row["source_files_urls"][0]
            ref_name = row["reference_outputs"]["files"][0]
            ref_url = row["reference_file_urls"][0]

            src_path = task_dir / src_name
            ref_path = task_dir / ref_name
            try:
                download(src_url, src_path)
                download(ref_url, ref_path)
            except Exception as e:
                print(f"  ✗ {tid}: download failed: {e}")
                continue

            rows.append({
                "id": tid,
                "family": "xlsx",
                "origin": "finch",
                "orig_id": row["id"],
                "split": split,
                "primary_tag": tag,
                "all_tags": [t.strip() for t in row["task_type"].split(",")],
                "business_type": row["business_type"],
                "instruction": row["instruction_en"],
                "constraints": row.get("task_constraints", "") or "",
                "source_file": str(src_path.relative_to(REPO_ROOT)),
                "reference_file": str(ref_path.relative_to(REPO_ROOT)),
                "task_type": "MODIFY",
                "max_steps": 15,
            })
            print(f"  ✓ {tid:14s}  {split:5s}  {tag}")

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Don't download, just print picks")
    args = p.parse_args()

    import sys

    print("Loading FinWorkBench/Finch …", flush=True)
    ds = load_dataset("FinWorkBench/Finch", split="test")
    print(f"  {len(ds)} rows", flush=True)

    picked = select(ds)
    total = sum(len(v) for v in picked.values())
    print(f"\nSelected {total} tasks across {len(picked)} tags", flush=True)

    if args.dry_run:
        for tag, items in picked.items():
            print(f"  {tag}: {[r['id'] for r in items]}", flush=True)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)
    rows = emit_manifest(picked)
    rows.sort(key=lambda r: (r["split"], r["primary_tag"], r["orig_id"]))

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    train_n = sum(1 for r in rows if r["split"] == "train")
    eval_n = sum(1 for r in rows if r["split"] == "eval")
    print(f"\nManifest written: {MANIFEST_PATH}", flush=True)
    print(f"  train: {train_n} | eval: {eval_n}", flush=True)


if __name__ == "__main__":
    main()