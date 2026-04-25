"""Task definitions for the Financial Task Environment.

Contains 10 tasks backed by real Excel workbooks covering diverse enterprise
finance & accounting workflows (QA, calculation, validation, data entry,
formatting, modeling, consolidation).  Each task ships a source .xlsx that
the agent must read or modify via Python code execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

# Base directory where xlsx files live (data/<task_id>/)
DATA_DIR = Path(os.environ.get("FINANCIAL_ENV_DATA_DIR", Path(__file__).parent / "data"))

TASKS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Helper to build source / reference paths
# ---------------------------------------------------------------------------

def _paths(task_id: str, src: str, ref: str | None = None):
    """Return dict with resolved source and optional reference paths."""
    d: Dict[str, Any] = {
        "source_file": str(DATA_DIR / task_id / src),
    }
    if ref:
        d["reference_file"] = str(DATA_DIR / task_id / ref)
    return d

# ── EASY ──────────────────────────────────────────────────────────────────

# Task 1 — QA: count rows (Calculation)
TASKS["task_1"] = {
    "id": "task_1",
    "orig_id": "119",
    "title": "Count Plants in Spreadsheet",
    "difficulty": "easy",
    "task_type": "QA",
    "category": "Calculation",
    "instruction": "How many plants are recorded in the spreadsheet?",
    "constraints": "",
    "reference_answer": "85",
    **_paths("119", "119_src_0.xlsx"),
}

# Task 2 — QA: value retrieval (Cross-sheet Retrieval)
TASKS["task_2"] = {
    "id": "task_2",
    "orig_id": "118",
    "title": "Retrieve TW EOL Charge",
    "difficulty": "easy",
    "task_type": "QA",
    "category": "Cross-sheet/file Retrieval",
    "instruction": "What is the TW EOL charge for 2002? Please provide just the amount.",
    "constraints": "",
    "reference_answer": "113291",
    **_paths("118", "118_src_0.xlsx"),
}

# Task 3 — QA: multi-step calculation (Calculation)
TASKS["task_3"] = {
    "id": "task_3",
    "orig_id": "34",
    "title": "Portfolio Mark-to-Market Change",
    "difficulty": "easy",
    "task_type": "QA",
    "category": "Calculation",
    "instruction": (
        "Assume the following changes occur in the Jul\u2013Dec 2002 market: "
        "Flat curve prices increase uniformly by $2/MWh; Peak 6x16 curve prices "
        "increase uniformly by $5/MWh; monthly contract volumes (Flat and Peak "
        "Total MWh) remain unchanged. Based on the 2002 table, calculate: "
        "(1) the total added value (mark-to-market change) for the combined "
        "Flat + Peak portfolio; and (2) what percentage of this added value "
        "comes from the Peak 6x16 contracts rather than the Flat contracts."
    ),
    "constraints": "",
    "reference_answer": (
        "The total added value of the July\u2013December 2002 portfolio is "
        "$1,989,600 (in absolute terms). Of this amount, approximately 27.9% "
        "(about 28%) comes from the Peak 6x16 contracts, with the remaining "
        "~72.1% coming from the Flat contracts."
    ),
    **_paths("34", "34_src_0.xlsx"),
}

# ── MEDIUM ────────────────────────────────────────────────────────────────

# Task 4 — Modify: summarise imbalances (Calculation + modify)
TASKS["task_4"] = {
    "id": "task_4",
    "orig_id": "35",
    "title": "Summarize Pipeline Imbalances",
    "difficulty": "medium",
    "task_type": "MODIFY",
    "category": "Calculation",
    "instruction": (
        "Summarize the volume and dollar imbalances that exist between the "
        "various pipeline operators (Operators) and Transwestern."
    ),
    "constraints": (
        "You will be given an Excel file as input. Perform all required "
        "operations by modifying the existing workbook. You may add new sheets "
        "if necessary, but you must preserve all original sheets and their "
        "contents. Return the full updated workbook."
    ),
    **_paths("35", "35_src_0.xlsx", "35_ref_0.xlsx"),
}

# Task 5 — Modify: audit & fix formulas (Validation / Review)
TASKS["task_5"] = {
    "id": "task_5",
    "orig_id": "40",
    "title": "Audit and Correct Formula Errors",
    "difficulty": "medium",
    "task_type": "MODIFY",
    "category": "Validation / Review, Calculation",
    "instruction": (
        "Audit the workbook and correct the formula errors in place so numbers "
        "calculate properly."
    ),
    "constraints": (
        "You will be given an Excel file as input. Perform all required "
        "operations by modifying the existing workbook. You may add new sheets "
        "if necessary, but you must preserve all original sheets and their "
        "contents. Return the full updated workbook."
    ),
    **_paths("40", "40_src_0.xlsx", "40_ref_0.xlsx"),
}

# Task 6 — Modify: create table + filter (Structuring / Formatting)
TASKS["task_6"] = {
    "id": "task_6",
    "orig_id": "60",
    "title": "Create Table and Apply Filter",
    "difficulty": "medium",
    "task_type": "MODIFY",
    "category": "Structuring / Formatting",
    "instruction": (
        "On the All Natural Gas sheet, create an Excel table and filter to "
        "show only the COUNTERPARTY entries highlighted in red."
    ),
    "constraints": (
        "You will be given an Excel file as input. Perform all required "
        "operations by modifying the existing workbook. You may add new sheets "
        "if necessary, but you must preserve all original sheets and their "
        "contents. Return the full updated workbook."
    ),
    **_paths("60", "60_src_0.xlsx", "60_ref_0.xlsx"),
}

# Task 7 — Modify: data entry + formatting (Data Entry / Import)
TASKS["task_7"] = {
    "id": "task_7",
    "orig_id": "21",
    "title": "Add Weekday Row and Data Entry",
    "difficulty": "medium",
    "task_type": "MODIFY",
    "category": "Data Entry / Import, Structuring / Formatting",
    "instruction": (
        "Add a weekday line directly below the date headers and update the "
        "12/31/2001 (Mon) column. For that day, there are no \"Receipts\"; "
        "record disbursements of $1,980,800 to Calpine (Power Purchases) and "
        "$100,000 to an unspecified vendor (Gas Purchases). Under Enron Facility "
        "Services, enter $3,101,855 for \"$2.5 per day\" and -$2,081,386 for "
        "\"estimate receipt\"; in Personnel, EES is $584,500; leave all other "
        "items as \"-\"."
    ),
    "constraints": (
        "You will be given an Excel file as input. Perform all required "
        "operations by modifying the existing workbook. You may add new sheets "
        "if necessary, but you must preserve all original sheets and their "
        "contents. Return the full updated workbook."
    ),
    **_paths("21", "21_src_0.xlsx", "21_ref_0.xlsx"),
}

# ── HARD ──────────────────────────────────────────────────────────────────

# Task 8 — Modify: balance-sheet validation + indicator calcs
TASKS["task_8"] = {
    "id": "task_8",
    "orig_id": "0",
    "title": "Balance Sheet Validation and Indicators",
    "difficulty": "hard",
    "task_type": "MODIFY",
    "category": "Validation / Review, Calculation, Structuring / Formatting",
    "instruction": (
        "Complete the validation and indicator calculations as follows: on the "
        "Balance Sheet, add a control to ensure TOTAL ASSETS equals TOTAL "
        "LIABILITIES AND EQUITY; on the Income Statement (Revenue & Expenses), "
        "add an Equity Roll Forward Test to reconcile equity movement and "
        "highlight any differences."
    ),
    "constraints": (
        "You will be given an Excel file as input. Perform all required "
        "operations by modifying the existing workbook. You may add new sheets "
        "if necessary, but you must preserve all original sheets and their "
        "contents. Return the full updated workbook."
    ),
    **_paths("0", "0_src_0.xlsx", "0_ref_0.xlsx"),
}

# Task 9 — Modify: add new sheet mirroring structure (Financial Modeling)
TASKS["task_9"] = {
    "id": "task_9",
    "orig_id": "24",
    "title": "Create Scenario3 Worksheet",
    "difficulty": "hard",
    "task_type": "MODIFY",
    "category": "Structuring / Formatting, Financial Modeling",
    "instruction": (
        'Add a new worksheet named "Scenario3" to the same workbook, mirroring '
        "the structure, row/column layout, monthly detail table, and chart area "
        'of "Scenario1". For Scenario3, update the hedging assumptions to a '
        "balanced allocation: 10-Yr 25%, 5-Yr 20%, 1-Yr 15%, May-Sep 20%, "
        "Q3 15%. Keep the note \"Maximum Monthly Average Short Position to "
        'Cover (July Peak) = 30,508 MW" unchanged; only the new sheet should '
        "be added, and formulas may be used within it."
    ),
    "constraints": (
        "You will be given an Excel file as input. Perform all required "
        "operations by modifying the existing workbook. You may add new sheets "
        "if necessary, but you must preserve all original sheets and their "
        "contents. Return the full updated workbook."
    ),
    **_paths("24", "24_src_0.xlsx", "24_ref_0.xlsx"),
}

# Task 10 — Modify: cross-sheet consolidation (multi-type)
TASKS["task_10"] = {
    "id": "task_10",
    "orig_id": "67",
    "title": "Consolidate by Type and Area",
    "difficulty": "hard",
    "task_type": "MODIFY",
    "category": "Structuring / Formatting, Calculation, Validation / Review, Cross-sheet Retrieval",
    "instruction": (
        "Create a new 'by type_area' worksheet based on the Summary and the "
        "other tabs. It should present two separate tables summarized by "
        "Imbal Type; within each table, consolidate by area, include Volume, "
        "Value and Date, and calculate totals. Finally, confirm that the value "
        "and volume totals tie to the totals shown on the Summary."
    ),
    "constraints": (
        "You will be given an Excel file as input. Perform all required "
        "operations by modifying the existing workbook. You may add new sheets "
        "if necessary, but you must preserve all original sheets and their "
        "contents. Return the full updated workbook."
    ),
    **_paths("67", "67_src_0.xlsx", "67_ref_0.xlsx"),
}

# ---------------------------------------------------------------------------
# Manifest loader — pulls additional tasks (Finch-50, OSWorld docx, PPTArena,
# TSBench) from data/manifest.jsonl.  Each manifest row already has the same
# shape as the hand-written TASKS dict above, plus a `family` and `split` field.
# ---------------------------------------------------------------------------

import json

_MANIFEST_PATH = Path(__file__).parent / "data" / "manifest.jsonl"


def _difficulty_for(primary_tag: str) -> str:
    """Heuristic difficulty bucket from a Finch primary_tag."""
    easy = {"Cross-sheet/file Retrieval", "Summary / Visualization"}
    hard = {"Financial Modeling", "Validation / Review"}
    if primary_tag in easy:
        return "easy"
    if primary_tag in hard:
        return "hard"
    return "medium"


def _load_manifest() -> None:
    """Load manifest.jsonl rows into TASKS in-place.  No-op if missing."""
    if not _MANIFEST_PATH.exists():
        return
    with open(_MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row["id"]
            if tid in TASKS:
                continue  # don't overwrite hand-curated tasks
            tags = row.get("all_tags", [])
            primary = row.get("primary_tag", "")
            TASKS[tid] = {
                "id": tid,
                "orig_id": row.get("orig_id", ""),
                "title": f"{primary}: {row['instruction'][:60]}…",
                "difficulty": _difficulty_for(primary),
                "task_type": row.get("task_type", "MODIFY"),
                "category": ", ".join(tags) if tags else primary,
                "family": row.get("family", "xlsx"),
                "split": row.get("split", "train"),
                "primary_tag": primary,
                "all_tags": tags,
                "instruction": row["instruction"],
                "constraints": row.get("constraints", ""),
                "source_file": str(Path(__file__).parent / row["source_file"]),
            }
            if row.get("reference_file"):
                TASKS[tid]["reference_file"] = str(Path(__file__).parent / row["reference_file"])
            # Pass through the docx evaluator block, resolving expected_files
            # to ABSOLUTE paths so the gold-stash dedup-by-string can match
            # them against the (already absolute) reference_file.
            if row.get("evaluator"):
                ev = dict(row["evaluator"])
                if ev.get("checks"):
                    repo_root = Path(__file__).parent
                    new_checks = []
                    for c in ev["checks"]:
                        nc = dict(c)
                        nc["expected_files"] = [
                            str(repo_root / f) if f and not Path(f).is_absolute() else f
                            for f in (c.get("expected_files") or [])
                        ]
                        new_checks.append(nc)
                    ev["checks"] = new_checks
                TASKS[tid]["evaluator"] = ev


_load_manifest()

# ---------------------------------------------------------------------------
# Helper accessors
# ---------------------------------------------------------------------------

# Hand-curated tasks come first (sorted numerically), then manifest-loaded ones.
def _sort_key(tid: str):
    parts = tid.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        return (0, int(parts[1]))  # task_<n>
    return (1, tid)                # everything else (finch_<id>, osworld_<uuid>, …)


TASK_IDS: List[str] = sorted(TASKS.keys(), key=_sort_key)


def get_task(task_id: str) -> Dict[str, Any]:
    """Return a task dict by ID or raise KeyError."""
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task_id '{task_id}'. Available: {len(TASK_IDS)} tasks."
        )
    return TASKS[task_id]


def list_tasks(split: str | None = None, family: str | None = None) -> List[Dict[str, str]]:
    """Return a summary list of tasks, optionally filtered by split or family."""
    out = []
    for tid in TASK_IDS:
        t = TASKS[tid]
        if split is not None and t.get("split", "train") != split:
            continue
        if family is not None and t.get("family", "xlsx") != family:
            continue
        out.append({
            "id": t["id"],
            "title": t["title"],
            "difficulty": t["difficulty"],
            "task_type": t["task_type"],
            "category": t["category"],
            "family": t.get("family", "xlsx"),
            "split": t.get("split", "train"),
        })
    return out


def split_ids(split: str) -> List[str]:
    """Return task IDs in a given split (train/eval).  Unmarked tasks count as train."""
    return [tid for tid in TASK_IDS if TASKS[tid].get("split", "train") == split]
