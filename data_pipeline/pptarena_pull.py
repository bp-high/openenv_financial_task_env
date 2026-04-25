"""Puller for PPTArena (https://github.com/michaelofengend/PPTArena).

Reads `src/evaluation_pairs_refined.json` from a local PPTArena checkout, picks
40 tasks stratified by `edit_type`, copies the original + ground_truth pptx
files into our data/ tree, and appends docx-style manifest rows with
family="pptx".

Usage:
    PPTARENA_ROOT=/path/to/PPTArena-main python data_pipeline/pptarena_pull.py
    python data_pipeline/pptarena_pull.py --root /path/to/PPTArena-main [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "pptarena"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"

# Per edit_type pick budgets (sum = 40).
EDIT_TYPE_BUDGET = {
    "Text & Typography": 6,
    "Charts": 4,
    "Images & Pictures": 4,
    "Theme & Background": 3,
    "Alignment, Distribution & Z-order": 3,
    "Slide/Section Management & Footers": 3,
    "Tables": 3,
    "Shapes & Drawing": 2,
    "SmartArt & Diagrams": 2,
    "Slide Layout & Placeholders": 2,
    "Accessibility & Semantics": 1,
    # long-tail singletons — include each for maximum edit-type coverage
    "Slide Transitions": 1,
    "Hyperlinks & Action Settings": 1,
    "Template & Master-Level Edits": 1,
    "Audio & Video": 1,
    "Object Animations": 1,
}

# Eval holdout per edit_type (sum = 8).  Singletons can't be held out (only 1
# sample), so they go to train.
EVAL_HOLDOUT = {
    "Text & Typography": 2,
    "Charts": 1,
    "Images & Pictures": 1,
    "Theme & Background": 1,
    "Tables": 1,
    "Slide/Section Management & Footers": 1,
    "Alignment, Distribution & Z-order": 1,
}


def slugify(name: str) -> str:
    """Slug from a task name — used as a stable filename-safe id."""
    out = []
    for c in name.lower():
        if c.isalnum():
            out.append(c)
        elif c in (" ", "-", "_", ":"):
            out.append("_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s[:60]


def select(pairs: list[dict], seed: int = 23) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_edit_type: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        et = p.get("edit_type", "")
        by_edit_type[et].append(p)

    picked: dict[str, list[dict]] = {}
    for et, budget in EDIT_TYPE_BUDGET.items():
        pool = list(by_edit_type.get(et, []))
        rng.shuffle(pool)
        picked[et] = pool[:budget]
        if len(picked[et]) < budget:
            print(f"  ⚠ edit_type {et!r}: wanted {budget}, got {len(picked[et])}")
    return picked


def emit_manifest(picked: dict, pptarena_root: Path) -> list[dict]:
    rows: list[dict] = []
    rng = random.Random(41)
    seen_slugs: set[str] = set()

    for et, items in picked.items():
        rng.shuffle(items)
        eval_n = EVAL_HOLDOUT.get(et, 0)
        for i, p in enumerate(items):
            split = "eval" if i < eval_n else "train"

            # Build a unique task id from the slug; suffix with a counter on collision
            base = slugify(p.get("name", "task")) or "task"
            tid = f"pptarena_{base}"
            n = 2
            while tid in seen_slugs:
                tid = f"pptarena_{base}_{n}"
                n += 1
            seen_slugs.add(tid)

            orig_src = pptarena_root / p["original"]
            orig_ref = pptarena_root / p["ground_truth"]
            if not orig_src.exists() or not orig_ref.exists():
                print(f"  ✗ {tid}: missing pair files — skip", flush=True)
                continue

            task_dir = DATA_DIR / base
            task_dir.mkdir(parents=True, exist_ok=True)
            src_dest = task_dir / f"{base}_src.pptx"
            ref_dest = task_dir / f"{base}_ref.pptx"
            try:
                shutil.copy2(orig_src, src_dest)
                shutil.copy2(orig_ref, ref_dest)
            except Exception as e:
                print(f"  ✗ {tid}: copy failed: {e}", flush=True)
                continue

            cats = p.get("category", [])
            if not isinstance(cats, list):
                cats = [cats]

            # Compose the agent-facing instruction: prompt is the user-style ask;
            # style_target adds the explicit constraints.  Keep both — prompt is
            # the headline, style_target is a "hidden but visible" spec.
            prompt = (p.get("prompt") or "").strip()
            style = (p.get("style_target") or "").strip()
            if style and style not in prompt:
                instruction = f"{prompt}\n\nDetails:\n{style}"
            else:
                instruction = prompt

            rows.append({
                "id": tid,
                "family": "pptx",
                "origin": "pptarena",
                "orig_id": p.get("name", ""),
                "split": split,
                "primary_tag": et,
                "all_tags": [et] + cats,
                "business_type": "presentation",
                "instruction": instruction,
                "constraints": (
                    "You will be given a PowerPoint file as input. Modify it "
                    "in-place using python-pptx. Preserve any content not "
                    "explicitly required to change. Return the full updated file."
                ),
                "source_file": str(src_dest.relative_to(REPO_ROOT)),
                "reference_file": str(ref_dest.relative_to(REPO_ROOT)),
                "task_type": "MODIFY",
                "max_steps": 15,
            })
            print(f"  ✓ {tid:55s} {split:5s} {et}", flush=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("PPTARENA_ROOT", ""))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.root:
        print("usage: --root /path/to/PPTArena-main  (or set PPTARENA_ROOT)")
        return 2
    pptarena_root = Path(args.root).expanduser().resolve()
    pairs_json = pptarena_root / "src" / "evaluation_pairs_refined.json"
    if not pairs_json.exists():
        print(f"missing: {pairs_json}")
        return 2

    print(f"Reading {pairs_json} …", flush=True)
    with open(pairs_json) as f:
        pairs = json.load(f)
    print(f"  {len(pairs)} pairs", flush=True)

    picked = select(pairs)
    total = sum(len(v) for v in picked.values())
    print(f"\nSelected {total} tasks across {len(picked)} edit_types", flush=True)

    if args.dry_run:
        for et, items in picked.items():
            print(f"  {et}: {[i.get('name', '')[:40] for i in items]}")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = emit_manifest(picked, pptarena_root)
    rows.sort(key=lambda r: (r["split"], r["primary_tag"], r["id"]))

    # Append (don't overwrite — Finch + OSWorld rows already in manifest)
    existing = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            existing = [line for line in f if line.strip()]
    new_ids = {r["id"] for r in rows}
    keep = [line for line in existing if json.loads(line)["id"] not in new_ids]
    with open(MANIFEST_PATH, "w") as f:
        for line in keep:
            f.write(line)
        for r in rows:
            f.write(json.dumps(r) + "\n")

    train_n = sum(1 for r in rows if r["split"] == "train")
    eval_n = sum(1 for r in rows if r["split"] == "eval")
    print(f"\nManifest updated: {MANIFEST_PATH}", flush=True)
    print(f"  pptx rows added: {len(rows)}  (train: {train_n} | eval: {eval_n})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())