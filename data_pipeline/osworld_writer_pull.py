"""Puller for the OSWorld-Verified libreoffice_writer subset.

Pulls 21 strict-docx tasks (skipping 1 .odt and 1 .pdf input) from the
xlang-ai/OSWorld GitHub repo and the xlangai/ubuntu_osworld_file_cache HF
dataset.  Emits manifest rows with `family: docx`.

Usage:
    python data_pipeline/osworld_writer_pull.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "osworld_writer"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"

GH_RAW = (
    "https://raw.githubusercontent.com/xlang-ai/OSWorld/main/"
    "evaluation_examples/examples/libreoffice_writer/{uuid}.json"
)

# 23 OSWorld-Verified writer UUIDs.  Two are excluded:
#   6a33f9b9 → .odt input (not strict docx)
#   4bcb1253 → .pdf input (PDF→docx conversion task)
ALL_UUIDS = [
    "0810415c-bde4-4443-9047-d5f70165a697",
    "0a0faba3-5580-44df-965d-f562a99b291c",
    "0b17a146-2934-46c7-8727-73ff6b6483e8",
    "0e47de2a-32e0-456c-a366-8c607ef7a9d2",
    "0e763496-b6bb-4508-a427-fad0b6c3e195",
    "3ef2b351-8a84-4ff2-8724-d86eae9b842e",
    "4bcb1253-a636-4df4-8cb0-a35c04dfef31",  # PDF input — exclude
    "66399b0d-8fda-4618-95c4-bfc6191617e9",
    "6a33f9b9-0a56-4844-9c3f-96ec3ffb3ba2",  # .odt — exclude
    "6ada715d-3aae-4a32-a6a7-429b2e43fb93",
    "6f81754e-285d-4ce0-b59e-af7edb02d108",
    "72b810ef-4156-4d09-8f08-a0cf57e7cefe",
    "8472fece-c7dd-4241-8d65-9b3cd1a0b568",
    "88fe4b2d-3040-4c70-9a70-546a47764b48",
    "936321ce-5236-426a-9a20-e0e3c5dc536f",
    "adf5e2c3-64c7-4644-b7b6-d2f0167927e7",
    "b21acd93-60fd-4127-8a43-2f5178f4a830",
    "bb8ccc78-479f-4a2f-a71e-d565e439436b",
    "d53ff5ee-3b1a-431e-b2be-30ed2673079b",
    "e246f6d8-78d7-44ac-b668-fcf47946cb50",
    "e528b65e-1107-4b8c-8988-490e4fece599",
    "ecc2413d-8a48-416e-a3a2-d30106ca36cb",
    "f178a4a9-d090-4b56-bc4c-4b72a61a035d",
]
EXCLUDE_UUIDS = {
    "4bcb1253-a636-4df4-8cb0-a35c04dfef31",
    "6a33f9b9-0a56-4844-9c3f-96ec3ffb3ba2",
}
STRICT_DOCX_UUIDS = [u for u in ALL_UUIDS if u not in EXCLUDE_UUIDS]

# Eval holdout — 4 of 21.  Picked deterministically by index so the split is
# reproducible.  Hold out a mix of evaluator types after we see them.
EVAL_FRACTION = 4 / 21


def fetch_json(url: str, timeout: float = 20.0, retries: int = 3) -> dict:
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"fetch failed after {retries} retries: {last_exc}")


def download(url: str, dest: Path, timeout: float = 60.0, retries: int = 3) -> None:
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


def normalize_filename(url: str) -> str:
    """Pull out a clean filename from the HF cache URL (URL-decoded)."""
    name = urllib.parse.unquote(url.rsplit("/", 1)[-1].split("?")[0])
    return name


def is_docx(name: str) -> bool:
    return name.lower().endswith(".docx")


def derive_split(idx: int, total: int, evaluator_func: str, holdout_evals: set[str]) -> str:
    """Hold out 4-ish tasks for eval, biased to cover distinct evaluator funcs."""
    if evaluator_func in holdout_evals:
        return "eval"
    return "train"


def normalize_evaluator(evaluator: dict) -> tuple[str, list[dict]]:
    """Coerce single/compound evaluator into (conj, [{func, options, expected}]).

    Single form:    evaluator = {func: str, options: dict|None, expected: dict}
    Compound form:  evaluator = {func: list[str], options: list[dict]|None,
                                 expected: list[dict], conj: "or"|"and"}
    Returns ("and"|"or", [check_dict, …]) where each check_dict has
    keys: func (str), options (dict), expected (dict).
    """
    func = evaluator.get("func")
    options = evaluator.get("options")
    expected = evaluator.get("expected")
    conj = evaluator.get("conj") or "and"

    if isinstance(func, list):
        n = len(func)
        opt_list = options if isinstance(options, list) else [options or {}] * n
        exp_list = expected if isinstance(expected, list) else [expected] * n
        checks = []
        for i in range(n):
            checks.append({
                "func": func[i],
                "options": opt_list[i] if i < len(opt_list) and opt_list[i] is not None else {},
                "expected": exp_list[i] if i < len(exp_list) else {},
            })
        return conj, checks
    else:
        return "and", [{
            "func": func or "",
            "options": options or {},
            "expected": expected or {},
        }]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"Fetching {len(STRICT_DOCX_UUIDS)} OSWorld-Verified writer task JSONs …", flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Pass 1: fetch all task JSONs and inspect evaluators
    tasks: list[tuple[str, dict]] = []
    func_counter: Counter[str] = Counter()
    conj_counter: Counter[str] = Counter()

    for uid in STRICT_DOCX_UUIDS:
        try:
            t = fetch_json(GH_RAW.format(uuid=uid))
        except Exception as e:
            print(f"  ✗ {uid}: {e}", flush=True)
            continue
        tasks.append((uid, t))
        conj, checks = normalize_evaluator(t["evaluator"])
        conj_counter[conj if len(checks) > 1 else "single"] += 1
        for c in checks:
            func_counter[c["func"]] += 1

    print(f"\nFetched {len(tasks)} task JSONs", flush=True)
    print("Evaluator-func distribution (per check):", flush=True)
    for fn, c in func_counter.most_common():
        print(f"  {c:3d}  {fn}", flush=True)
    print("Compound conj distribution (per task):", flush=True)
    for k, c in conj_counter.most_common():
        print(f"  {c:3d}  {k}", flush=True)

    if args.dry_run:
        return 0

    # Pick one task per distinct evaluator-func for eval (up to 4); the rest train.
    eval_funcs: list[str] = []
    for fn, _ in func_counter.most_common():
        if len(eval_funcs) >= 4:
            break
        eval_funcs.append(fn)
    holdout_evals = set(eval_funcs[:4])
    print(f"\nEval split funcs (1 task each): {sorted(holdout_evals)}", flush=True)

    rows: list[dict] = []
    seen_eval_funcs: set[str] = set()
    skipped_non_docx = 0

    for idx, (uid, t) in enumerate(tasks):
        conj, checks = normalize_evaluator(t["evaluator"])
        primary_func = checks[0]["func"]

        # Initial file URL — first download config block
        init_url: str | None = None
        for cfg in t.get("config", []):
            if cfg.get("type") == "download":
                files = cfg.get("parameters", {}).get("files", [])
                if files:
                    init_url = files[0]["url"]
                    break
        if init_url is None:
            print(f"  ✗ {uid[:8]}: no init file in config", flush=True)
            continue
        init_name = normalize_filename(init_url)
        if not is_docx(init_name):
            print(f"  ⊘ {uid[:8]}: non-docx init ({init_name}) — skip", flush=True)
            skipped_non_docx += 1
            continue

        task_dir = DATA_DIR / uid
        src_path = task_dir / init_name
        try:
            download(init_url, src_path)
        except Exception as e:
            print(f"  ✗ {uid[:8]}: src download failed: {e}", flush=True)
            continue

        # Resolve & download every gold variant for the checks list.
        # `expected` may be:
        #   - a dict with path: str           (single gold file)
        #   - a dict with path: list[str]     (multi-gold: pass list to evaluator)
        normalized_checks: list[dict] = []
        for c_i, c in enumerate(checks):
            exp = c.get("expected", {}) or {}
            paths = exp.get("path", "") if isinstance(exp, dict) else ""
            if isinstance(paths, str):
                paths = [paths] if paths else []
            elif not isinstance(paths, list):
                paths = []

            dests = exp.get("dest")
            if isinstance(dests, str):
                dests = [dests]
            elif not isinstance(dests, list):
                dests = [None] * len(paths)

            expected_files: list[str] = []
            ok = True
            for p_i, gold_url in enumerate(paths):
                gold_name = normalize_filename(gold_url)
                # Tolerate non-docx auxiliaries (some tasks pull a reference data
                # file alongside the docx gold) — keep all that download cleanly.
                dest_name = dests[p_i] if p_i < len(dests) and dests[p_i] else gold_name
                gold_path = task_dir / f"gold_{c_i}_{p_i}__{dest_name}"
                try:
                    download(gold_url, gold_path)
                except Exception as e:
                    print(f"  ⚠ {uid[:8]}: gold[{c_i}/{p_i}] download failed: {e}", flush=True)
                    ok = False
                    break
                expected_files.append(str(gold_path.relative_to(REPO_ROOT)))

            if not ok:
                continue

            normalized_checks.append({
                "func": c["func"],
                "options": c["options"],
                "expected_files": expected_files,  # always a list, possibly empty
            })

        if not normalized_checks:
            print(f"  ⊘ {uid[:8]}: no usable checks", flush=True)
            continue

        # Pick a primary reference_file: first check with at least one gold file
        # ending in .docx (used by the generic diff layer + progress signal).
        primary_ref = ""
        for c in normalized_checks:
            for f in c.get("expected_files") or []:
                if f.lower().endswith(".docx"):
                    primary_ref = f
                    break
            if primary_ref:
                break

        # Bias eval split to cover distinct evaluator funcs
        if primary_func in holdout_evals and primary_func not in seen_eval_funcs:
            split = "eval"
            seen_eval_funcs.add(primary_func)
        else:
            split = "train"

        tid = f"osworld_{uid[:8]}"
        rows.append({
            "id": tid,
            "family": "docx",
            "origin": "osworld",
            "orig_id": uid,
            "split": split,
            "primary_tag": primary_func,
            "all_tags": list({c["func"] for c in normalized_checks}),
            "business_type": "writer",
            "instruction": t.get("instruction", "").strip(),
            "constraints": "",
            "source_file": str(src_path.relative_to(REPO_ROOT)),
            "reference_file": primary_ref,
            "task_type": "MODIFY",
            "max_steps": 15,
            "evaluator": {
                "conj": conj,
                "checks": normalized_checks,
            },
            "source": t.get("source", ""),
        })
        compound = "" if len(normalized_checks) == 1 else f" [{conj}×{len(normalized_checks)}]"
        print(f"  ✓ {tid:25s} {split:5s} {primary_func}{compound}", flush=True)

    if not rows:
        print("No rows emitted; aborting.", flush=True)
        return 1

    # Append (don't overwrite — Finch rows are already there)
    existing: list[str] = []
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
    print(f"  docx rows added:  {len(rows)}  (train: {train_n} | eval: {eval_n})", flush=True)
    if skipped_non_docx:
        print(f"  non-docx skipped: {skipped_non_docx}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
