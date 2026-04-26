# Edits log — Round-2 environment extension

This file tracks every change made on top of the Round-1 submission, in order.
Useful both as a journal and as a re-deploy checklist.

**Round-1 baseline:** commit `bf77949` ("Update readme") on `main`. Single
family (xlsx), 10 hand-curated Finch tasks, monolithic `graders.py`,
heuristic step-rewards.

**Round-2 target:** unified office-document RL environment — xlsx + docx + pptx,
real enterprise artifacts, gaming-resistant multi-layer grading, manifest-driven.

---

## State at Round-1 (baseline)

| Area | What was there |
|---|---|
| Task families | xlsx only |
| Number of tasks | 10 hand-curated, all from Finch |
| Task definitions | Hardcoded `TASKS = {...}` dict in [`tasks.py`](tasks.py) |
| Source data | `data/<orig_id>/{src,ref}_0.xlsx` — 10 dirs |
| Grading | One `graders.py` module, two functions: `grade_qa` (text) and `grade_xlsx` (cell-diff) |
| Step rewards | `_compute_code_reward` in [`server/financial_environment.py`](server/financial_environment.py): heuristics on code string (regex `save(`, count substantive lines, length of stdout). Cap 0.10/step. |
| Sandboxing | None — agent's subprocess has full filesystem access |
| Reward components | 4 signals, all heuristic, partly gameable |
| Train/eval split | None |
| Deps | `openpyxl` only |

### Known weaknesses identified before changes

1. `save(` string match misses `prs.save()`, `Document.save()` — wouldn't generalize past xlsx.
2. No measurement of whether file *actually* changed; just whether code mentioned save.
3. No "moving toward gold" signal.
4. Hardcoded task table — can't scale past ~30 tasks without bloat.
5. Gold files reachable from sandbox via `glob(data/**)` → reward hacking.

---

## Phase 1 — Manifest loader + 50 stratified Finch tasks

**Goal:** scale beyond 10 hand-curated tasks; introduce a manifest the env
loads at startup so future task families (docx, pptx) plug in cleanly.

### New files
- [`data_pipeline/finch_pull.py`](data_pipeline/finch_pull.py) — stratified
  puller for the `FinWorkBench/Finch` HF dataset (172 tasks). Picks **50
  xlsx-only MODIFY tasks** across 7 tag buckets:

  | Tag | Picked | of total |
  |---|---|---|
  | Calculation | 16 | of 119 |
  | Structuring / Formatting | 11 | of 86 |
  | Data Entry / Import | 6 | of 44 |
  | Validation / Review | 5 | of 37 |
  | Cross-sheet/file Retrieval | 5 | of 36 |
  | Summary / Visualization | 4 | of 33 |
  | Financial Modeling | 3 | of 15 |

  Web Search dropped — all such tasks have non-xlsx sources. Slots reallocated
  to Calculation + Structuring.

- [`data/manifest.jsonl`](data/manifest.jsonl) — 50 rows, schema:
  ```json
  {"id": "finch_10", "family": "xlsx", "origin": "finch", "orig_id": "10",
   "split": "eval", "primary_tag": "Calculation",
   "all_tags": ["Calculation", "Financial Modeling"],
   "business_type": "Predictive Modeling",
   "instruction": "...", "constraints": "...",
   "source_file": "data/finch_50/10/10_src_0.xlsx",
   "reference_file": "data/finch_50/10/10_ref_0.xlsx",
   "task_type": "MODIFY", "max_steps": 15}
  ```

- [`data/finch_50/<id>/{src,ref}.xlsx`](data/finch_50/) — ~42 MB, 50 tasks × 2 files.

### Train/eval split
- 40 train / 10 eval (stratified — at least 1 holdout per tag).
- Driven by per-tag `EVAL_HOLDOUT` budget in the puller.

### Modified files
- [`tasks.py`](tasks.py) — added `_load_manifest()` that reads
  `data/manifest.jsonl` and merges rows into `TASKS` (skipping any whose ID
  already exists, so the original 10 hand-curated tasks remain). Added
  `list_tasks(split=, family=)`, `split_ids()` filters.

### Resulting task counts
- 60 total (10 original + 50 Finch), 50 train / 10 eval.

---

## Phase 2 — Unified `RewardTracker`

**Goal:** replace heuristic code-string scoring with **real file-state**
signals, generalizable across xlsx/pptx/docx.

### New file
- [`rewards.py`](rewards.py) — `RewardTracker` class, one instance per episode.

### Reward components (all per-step, summed and clamped to 0.10)

| Component | Range | What it actually checks |
|---|---|---|
| `exec_health` | 0–0.020 | Subprocess return code; bonus if stdout non-empty |
| `lib_engagement` | 0–0.010 | Code matches `_LIB_PATTERNS[family]` regex (xlsx → openpyxl/load_workbook/Workbook; pptx → Presentation; docx → Document) |
| `mutation` | 0–0.030 | SHA-256 of working file changed since last step |
| `validity` | 0–0.020 | Mutated file still parses with the family's loader |
| `progress` | 0–0.040 | Structural distance to gold *decreased* this step (gated by `enable_progress`) |

### Per-family structural distance (in `rewards.py`)
- `_xlsx_distance` — fraction of gold cells matched (mirrors final grader)
- `_pptx_distance` — fraction of gold (slide_idx, shape_idx) text-frames matched
- `_docx_distance` — fraction of gold paragraphs matched at same index

### Modified files
- [`server/financial_environment.py`](server/financial_environment.py):
  - Replaced `_compute_code_reward` with a delegate to `RewardTracker`
  - `_compute_code_reward` now returns `(total, breakdown_dict)` instead of just `float`
  - Per-episode tracker stood up in `reset()` after copying source to workdir
  - `FINANCIAL_ENV_PROGRESS=0` env var disables the progress signal (for clean eval)
  - Reward decomposition surfaced in feedback for debugging

### Smoke test results
- Read-only step: 0.030 (exec_health 0.020 + lib_engagement 0.010)
- Save+modify step: 0.080 (+ mutation 0.030 + validity 0.020)
- Failed code: 0.005 (exec_health_fail only)
- Decomposition logged in feedback, e.g.: `Reward: total=0.080 (exec_health=0.020, lib_engagement=0.010, mutation=0.030, validity=0.020, progress=0.000)`

---

## Phase 3 — DOCX family (OSWorld-Verified writer subset)

**Goal:** add Microsoft Word (.docx) tasks alongside xlsx, with real
property-checking evaluators ported from OSWorld.

### New files

- [`data_pipeline/osworld_writer_pull.py`](data_pipeline/osworld_writer_pull.py)
  — pulls 21 strict-docx tasks from `xlang-ai/OSWorld` (GitHub) and
  `xlangai/ubuntu_osworld_file_cache` (HF). Of the 23 published writer
  UUIDs, drops 2 (one `.odt`, one `.pdf` source) leaving **21 strict-docx**.

  Schema normalization: OSWorld evaluators come in two shapes (single-string
  `func` vs. compound `func: list[str]` with `conj: "or"|"and"` and parallel
  `expected`/`options` lists). The puller normalizes everything to
  `evaluator: {conj, checks: [{func, options, expected_files: [...]}, ...]}`.
  Multi-gold (`multi: true`) tasks have `expected_files` as a list per check.

- [`graders/docx_metrics.py`](graders/docx_metrics.py) — port of 16 evaluator
  functions from
  [OSWorld's `desktop_env/evaluators/metrics/docs.py`](https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/evaluators/metrics/docs.py)
  (Apache-2.0). Heavy deps (`skimage`, `easyocr`) imported lazily; one
  function (`find_default_font`) stubbed because it operates on a LibreOffice
  config XML that doesn't exist in our headless sandbox.

  Added `infeasible` handler: passes iff agent didn't modify the source
  (the agent should refuse). The `bb8ccc78` task ("Share this document with
  my team and let us edit it together in real-time") uses this — it's
  genuinely impossible from a code-execution sandbox.

  Dispatcher: `run_evaluator(conj, checks, working_file, source_file)` —
  `and` = `min(scores)`, `or` = `max(scores)`.

  | Evaluator | Tasks | Style |
  |---|---|---|
  | `compare_docx_files` | 7× | Content diff (with options: ignore_blanks, ignore_case, fuzzy_match, …) |
  | `compare_line_spacing` | 3× | Property |
  | `compare_docx_tables` | 3× | Structure |
  | `check_tabstops` | 1× | Property + position-distance |
  | `compare_subscript_contains` | 1× | Property |
  | `has_page_numbers_in_footers` | 1× | Single-file property |
  | `compare_font_names` | 1× | Single-file property |
  | `is_first_line_centered` | 1× | Single-file property |
  | `compare_docx_images` | 1× | Pixel-byte diff |
  | `compare_unique_train_records` | 1× | Multi-file domain logic |
  | `evaluate_strike_through_last_paragraph` | 1× | Property |
  | `evaluate_colored_words_in_tables` | 1× | Skimage CIE delta-E |
  | `infeasible` | 1× | Sentinel (file-unchanged check) |
  | `check_italic_font_size_14` | 1× | Property |
  | `contains_page_break` | 1× | Property |
  | `find_default_font` | 1× | **Stubbed** (LO-config-dependent) |

### File reorganization (mid-phase)

- Renamed `graders.py` (root module) → `graders/__init__.py` (package).
  Forced because `graders/` (new dir for `docx_metrics.py`) collides with
  `graders.py` (old root file) — Python won't accept both. Existing
  `from graders import grade_task` imports still work transparently.

### New 3-layer DOCX grader

In [`graders/__init__.py`](graders/__init__.py):

```python
def grade_docx(task, output_path):
    if not _docx_validity(output_path):     # layer 1 — validity gate
        return 0.001
    diff_score    = _docx_diff(output_path, task["reference_file"])    # layer 2
    primary_score = run_evaluator(...)                                 # layer 3
    return 0.4 * diff_score + 0.6 * primary_score
```

The dispatcher (`grade_task`) routes by `task["family"]` — xlsx still uses
the cell-diff path, docx uses the new 3-layer path.

### Modified files

- [`tasks.py`](tasks.py) — manifest loader now passes through `evaluator`,
  `primary_tag`, `all_tags`. Resolves evaluator's `expected_files` to absolute
  paths (matters for the gold-stash dedup in Phase 4).

- [`pyproject.toml`](pyproject.toml) + [`Dockerfile`](Dockerfile) — added
  `python-docx>=1.1.0`, `rapidfuzz>=3.0.0`, `Pillow>=10.0.0`.

### Resulting task counts
- 81 total (10 original + 50 Finch xlsx + 21 OSWorld docx).
- 17 docx train, 4 docx eval (stratified to cover 4 distinct evaluator funcs).

### Smoke test results
- Submit gold to compound `and×2` task → **0.999** ✓
- Submit corrupted bytes → **0.001** (validity gate rejects) ✓
- Submit unmodified source → **0.400** (diff layer says similar, per-task says no-edit)

### OSWorld quirk noted
- `osworld_0a0faba3` (`check_tabstops` task): the gold itself doesn't satisfy
  `word_number_split_by_tabstop=3` for paragraph [2] (`"Make payment\t..."` has
  only 2 words before the tab). This is a faithful port of OSWorld's
  behavior, not a bug in our code. May want to relax the rule for training
  or move that task to eval-only.

---

## Phase 4 — Reward-hacking defenses

**Goal:** plug the two biggest hacking surfaces identified in the Q2 audit.

### Defense 1 — Gold file moved out of sandbox at episode start

**Threat:** `glob('/app/env/data/**/*Gold*.docx')` or `glob('**/*_ref_*.xlsx')`
finds the gold; agent submits it for an instant 0.999.

**Fix:** at `reset()`:

1. Make a per-episode COPY of the global `TASKS[id]` dict (so episode-time
   path mutations don't pollute the global)
2. Create a tmpdir at `/tmp/oe_gold_<random>/`
3. **Move (rename)** every gold file from `data/...` into the tmpdir with a
   generic name (`gold_ref<ext>`, `check_<i>_<j>_<random><ext>`)
4. Track the moves in `self._gold_originals` so `close()` can restore
5. Rewrite the episode-task's `reference_file` and
   `evaluator.checks[*].expected_files` to point at the tmpdir paths

**De-dup**: when the same path appears as both `reference_file` and an
evaluator `expected_files` entry (common — the puller sets reference_file =
first check's first expected_file), the stasher uses a `path_map` to ensure
both new paths point to the same stashed location.

**Restore**: `close()` renames stashed files back to their original `data/`
locations. `reset()` calls `close()` at the start of each episode in case
the prior episode didn't end cleanly.

### Defense 2 — Per-task evaluator as 6th reward signal

**Threat:** the previous 5 components rewarded "moved closer to gold via
generic structural distance", which an agent could optimize without
satisfying the actual property check the task is testing.

**Fix:** new `eval_check` component (0–0.020). Computes the per-task
evaluator at episode start, then on each mutating step. Rewards
*increases* in spec-aligned score.

```python
# rewards.py
if self.task_evaluator is not None and file_valid:
    cur_eval = self._safe_task_eval()
    if self._prev_eval is not None and cur_eval > self._prev_eval:
        delta = cur_eval - self._prev_eval
        sig.eval_check = min(EVAL_CHECK_MAX, EVAL_CHECK_MAX * delta)
    self._prev_eval = cur_eval
```

For docx, the env passes `task_evaluator = run_evaluator(conj, checks, ...)`
into the tracker. xlsx/pptx pass `None`.

### Modified files

- [`rewards.py`](rewards.py):
  - Added `task_evaluator` param to `RewardTracker.__init__`
  - Added `eval_check` field to `StepSignals` + recomputed `total`
  - Added `EVAL_CHECK_MAX = 0.020` constant
  - Added `_safe_task_eval()` helper

- [`server/financial_environment.py`](server/financial_environment.py):
  - `__init__`: added `_gold_stash_dir`, `_gold_originals` fields
  - `reset()`: copies task dict, creates stash dir, calls `_stash_gold_files`,
    builds `task_evaluator` callable for docx, passes it to `RewardTracker`
  - New methods: `_stash_gold_files(task, stash_dir)`, `_make_task_evaluator()`
  - `close()`: restores moved gold files to data/, removes stash dir

### Smoke test results

| Scenario | Score | Expected | Result |
|---|---|---|---|
| Compound and×2 docx, submit stashed gold | 0.999 | ~0.999 | ✓ |
| Single-check docx, submit stashed gold | 0.999 | ~0.999 | ✓ |
| Submit corrupted bytes | 0.001 | 0.001 | ✓ (validity gate) |
| Submit source (unmodified) | 0.400 | partial | ✓ (diff 1.0, per-task 0) |
| xlsx (no per-task evaluator), submit gold | 0.999 | ~0.999 | ✓ |
| Code step copies stashed gold to working file | total=0.090 with `eval_check=0.020` | should fire | ✓ |
| Original gold file present in data/ during episode | False on disk | False | ✓ (moved out) |
| Original restored after `close()` | True on disk | True | ✓ |

---

## Phase 5 — PPTX family (PPTArena ingest)

**Goal:** add Microsoft PowerPoint (.pptx) tasks. PPTArena chosen over TSBench
because PPTArena ships actual gold .pptx files; TSBench only has
`ideal_description` text and would need an LLM judge.

### Source

Local checkout of [PPTArena](https://github.com/michaelofengend/PPTArena)
unpacked at `~/Downloads/PPTArena-main`. The repo's
`src/evaluation_pairs_refined.json` has 100 well-curated task pairs:

```json
{
  "name": "Case 31: Fix Text Overflow",
  "prompt": "...",
  "style_target": "<detailed expected output spec>",
  "original": "Original/<file>.pptx",
  "ground_truth": "GroundTruth/<file>.pptx",
  "category": ["Content", "Layout"],
  "edit_type": "Text & Typography"
}
```

Distribution across the 100:

| edit_type | count |
|---|---|
| Text & Typography | 29 |
| Charts | 10 |
| Images & Pictures | 10 |
| Theme & Background | 9 |
| Alignment, Distribution & Z-order | 8 |
| Slide/Section Management & Footers | 8 |
| Tables | 8 |
| Shapes & Drawing | 4 |
| SmartArt & Diagrams | 4 |
| Slide Layout & Placeholders | 3 |
| Accessibility & Semantics | 2 |
| Long-tail singletons (Transitions, Hyperlinks, Master, Audio/Video, Animations) | 1 each |

### New file
- [`data_pipeline/pptarena_pull.py`](data_pipeline/pptarena_pull.py) — reads
  `evaluation_pairs_refined.json`, picks **38 tasks** stratified by
  `edit_type`. Sub-budget below; sum is 38 (close to the 40 target — the
  gap is from the long-tail edit_types having only 1 sample each).

  | edit_type | picked | of total |
  |---|---|---|
  | Text & Typography | 6 | of 29 |
  | Charts | 4 | of 10 |
  | Images & Pictures | 4 | of 10 |
  | Theme & Background | 3 | of 9 |
  | Alignment, Distribution & Z-order | 3 | of 8 |
  | Slide/Section Management & Footers | 3 | of 8 |
  | Tables | 3 | of 8 |
  | Shapes & Drawing | 2 | of 4 |
  | SmartArt & Diagrams | 2 | of 4 |
  | Slide Layout & Placeholders | 2 | of 3 |
  | Accessibility & Semantics | 1 | of 2 |
  | Long-tail singletons | 5 × 1 | of 5 |

  Long-tail singletons all go to train (only 1 sample each — can't hold out).
  Eval holdout = 8: 2 from Text & Typography, 1 each from {Charts, Images,
  Theme, Alignment, Slide Mgmt, Tables}.

  The agent-facing instruction is `prompt + "\n\nDetails:\n" + style_target`
  — `style_target` carries the explicit spec PPTArena uses internally for
  evaluation, exposed to the agent as a "hidden but visible" constraint.

### Data layout

```
data/pptarena/<slug>/
   <slug>_src.pptx   # copied from PPTArena-main/Original/
   <slug>_ref.pptx   # copied from PPTArena-main/GroundTruth/
```

Total disk: ~244 MB for 38 tasks (pptx files are larger than docx/xlsx —
they contain embedded images and themes).

### Grader — `grade_pptx` (2-layer, no per-task evaluator)

In [`graders/__init__.py`](graders/__init__.py):

```python
def grade_pptx(task, output_path):
    if not _pptx_validity(output_path):  # layer 1
        return 0.001
    # layer 2: structural diff
    #   slide-count match (30%) + per-shape text-equality (70%, fuzzy 90%+ allowed)
    ...
```

Per-task evaluator is **intentionally not wired**. PPTArena's published
evaluator is a VLM-as-judge pipeline (instruction-following + visual quality)
which is expensive and non-deterministic. Skipping for v1; wiring it as an
optional `RENDER_FOR_VLM=1` flag is in the Open Issues list.

### Modified files

- [`graders/__init__.py`](graders/__init__.py): added `_pptx_validity`,
  `_pptx_load_shape_text`, `grade_pptx`. Dispatcher now routes pptx → grade_pptx.
- [`pyproject.toml`](pyproject.toml) + [`Dockerfile`](Dockerfile): added
  `python-pptx>=1.0.0`.

### Resulting task counts (cumulative)

| Family | Origin | Train | Eval | Total |
|---|---|---|---|---|
| xlsx | hand-curated | 10 | 0 | 10 |
| xlsx | Finch | 40 | 10 | 50 |
| docx | OSWorld | 17 | 4 | 21 |
| pptx | PPTArena | 30 | 8 | 38 |
| **total** | | **97** | **22** | **119** |

### Smoke test results

| Scenario | Score | Expected | Result |
|---|---|---|---|
| Submit stashed gold (eval task) | 0.999 | ~0.999 | ✓ |
| Submit corrupted .pptx bytes | 0.001 | 0.001 | ✓ (validity gate) |
| Code step that mutates + saves (add blank slide) | total=0.080 | ≥0.06 | ✓ (exec=0.020, lib=0.010, mutation=0.030, validity=0.020) |
| Gold-stash works for pptx (file moves out of `data/`) | True | True | ✓ |
| `close()` restores gold to `data/` | True | True | ✓ |

### Known limitation: text-only diff is weak for layout tasks

For an Alignment / Layout task (e.g. *Case 60: Fix Text Placement*), source
and ground-truth have near-identical text content — only shape positions
differ. Our diff layer scores 0.999 on the unmodified source for this case,
which is not what we want. Two paths to fix:

1. **Extend `grade_pptx` with position+size diff** (cheap; ~30 lines): for
   each (slide_idx, shape_idx) pair, compare `(left, top, width, height)`
   within tolerance. Recompose the score as `0.2 * slide_count + 0.8 * avg(
   0.5 * text_match + 0.25 * position_match + 0.25 * size_match)`.

2. **Wire VLM judge** behind `PPTX_VLM_JUDGE=1` env var — render slides via
   headless LibreOffice → PNG, send (instruction, before, after, ref) to a
   VLM. Matches PPTArena's published methodology but is expensive.

Recommended: (1) before any RL training; (2) for the final eval scoreboard.

### Phase 5 follow-up: layout-aware diff (delivered)

Implemented option (1) above. The grader now loads every shape's
`(left, top, width, height)` (in EMU) and computes a per-shape composite
score:

- **Text** (50%) — exact match → 1.0; rapidfuzz partial credit otherwise.
- **Position** (25%) — `_coord_match(left, denom=slide_w)` averaged with
  same for `top`. Tolerance: `delta ≤ 2%` of slide dim → 1.0; `delta ≥ 20%`
  → 0.0; linear in between. Both sides None (placeholder inheriting from
  layout) is treated as a match.
- **Size** (25%) — same `_coord_match` for width/height.

Final score reweighted: `0.2 * slide_count + 0.8 * avg(per-shape composite)`.

#### Smoke results on all 8 pptx eval tasks (source-vs-gold)

| Task | Before fix | After fix | Notes |
|---|---|---|---|
| `case_36_add_speaker_notes` | 0.999 | **0.683** | Big drop — entire shapes added in gold |
| `case_32_arrange_image_and_text` | 0.999 | **0.824** | Position diff captured |
| `case_7_update_quarter_two_data_b` | 0.999 | **0.948** | Chart text + size diff |
| `case_60_fix_text_placement` | 0.999 | **0.981** | Modest — positions in tolerance band |
| `case_35_structural_fix` | 0.999 | 0.971 | Modest |
| `case_49_normalize_thousand_separators` | 0.999 | 0.992 | Tiny text edit, no layout change |
| `case_40_hindu_center_titles` | 0.999 | 0.997 | Title-alignment only — small px shift |
| `case_26_match_slide_colors_to_theme` | 0.999 | 0.999 | Pure color/theme — geometry unchanged |

5 of 8 eval tasks now show meaningful drop. The remaining 3 (`case_40`,
`case_49`, `case_26`) still score ~0.99 because their edits are
**styling-only** — color, font, fill — which our geometry-only diff
doesn't see.

#### Remaining gap: styling-only tasks (29 of 100 PPTArena tasks)

Styling tasks edit shape `fill`, `line`, font `name/size/bold/italic/color`,
or theme — none of which are captured by text + geometry. Two ways to
close the gap, both filed as new follow-ups:

a. **Per-shape style diff**: for each shape, compare
   `fill.solid().fore_color.rgb`, `line.color.rgb`, and for the first run
   in each text frame: `font.name, font.size, font.bold, font.italic,
   font.color.rgb`. Add as a 4th component in `_shape_match_score`. ~50 lines.

b. **VLM judge** (option 2 above) — catches styling for free since it
   compares rendered images. Defer to eval-time only because of cost.

For training, (a) is sufficient. For the final scoreboard, (b) is nicer.

### Phase 5 follow-up #2: style-aware diff (delivered)

Implemented option (a) above. New `_shape_style()` extractor pulls 7
attributes per shape (all None-tolerant — failures during read become
`None`, which counts as a match against another `None`):

| Attribute | Weight | Source |
|---|---|---|
| `fill_rgb` | 0.30 | `shape.fill.fore_color.rgb` (solid fills only) |
| `font_rgb` | 0.20 | first-run `font.color.rgb` |
| `font_size_pt` | 0.15 | first-run `font.size.pt` |
| `font_name` | 0.10 | first-run `font.name` |
| `line_rgb` | 0.10 | `shape.line.color.rgb` |
| `font_bold` | 0.075 | first-run `font.bold` |
| `font_italic` | 0.075 | first-run `font.italic` |

Per-shape composite reweighted from `50% text + 25% pos + 25% size` to:

> **40% text + 20% style + 20% position + 20% size**

Why these weights? Text is still dominant because most edits affect text
content. Style gets equal weight to position/size, reflecting that styling
edits are common in PPTArena (~29 tasks).

#### Smoke results across all 8 pptx eval tasks (source-vs-gold)

| Task | Phase-5 layout-only | Phase-5+style | Discrimination (gold − source) |
|---|---|---|---|
| `case_26_match_slide_colors_to_theme` | 0.999 | **0.971** | 0.000 → **0.028** ✓ unblocked |
| `case_36_add_speaker_notes` | 0.683 | 0.715 | 0.316 → 0.284 |
| `case_32_arrange_image_and_text` | 0.824 | 0.855 | 0.175 → 0.144 |
| `case_60_fix_text_placement` | 0.981 | 0.985 | 0.018 → 0.014 |
| `case_35_structural_fix` | 0.971 | 0.975 | 0.028 → 0.024 |
| `case_7_update_quarter_two_data_b` | 0.948 | 0.951 | 0.051 → 0.048 |
| `case_40_hindu_center_titles` | 0.997 | 0.998 | tiny |
| `case_49_normalize_thousand_separators` | 0.992 | 0.994 | tiny |

Gold-vs-gold remained 0.999 on all 8 (no regression).

**Trade-off observed:** the styling task discrimination went from 0 → 0.028,
but text/layout-heavy tasks lost a few percentage points of discrimination
because the text weight dropped from 50% → 40%. Net positive but not
dramatic.

#### The dilution problem (now the binding limitation)

For tasks where only a few shapes out of many are edited (e.g.
`case_40_hindu_center_titles` edits 1 title shape per slide), the diff
averages across **all** shapes — the un-edited majority dominates and
the score barely moves between source and gold. This is structural to
average-based diff and not a bug.

Two follow-ups to consider:

a. **Edit-zone masking** — score only shapes whose attributes differ
   between source and gold (using `task.source_file` as the baseline).
   Changes scoring semantics: instead of "how close to gold", you measure
   "did the agent fix the parts that were supposed to change". ~30 lines,
   but more invasive than (b) below.

b. **VLM judge** — compares rendered images, naturally focuses on visible
   differences. The right long-term answer; expensive — defer to eval-time
   behind a flag.

---

## Phase 6 — Inference script v2 (manifest-aware benchmarking)

**Goal:** Round-1's [`inference.py`](inference.py) was hardcoded to 5 xlsx
tasks and produced stdout-only output. Round-2 needs a script that:

1. Selects tasks from the manifest (filterable by split/family/ids)
2. Picks the right system prompt per family (openpyxl / python-docx / python-pptx)
3. Persists results to disk so we can produce reward curves and before/after
   plots for the judging story

### CLI (new)

```
python inference.py [--split eval|train|all]
                    [--family xlsx|docx|pptx|all]
                    [--limit N]
                    [--task-ids id1,id2,…]
                    [--output-dir runs/<custom>]
                    [--model <name>]
                    [--api-base <url>] [--env-url <http://…>]
                    [--max-steps 15] [--task-timeout 360]
                    [--temperature 0.0] [--max-tokens 12000]
```

`--task-ids` overrides `--split`/`--family`. Selection is sorted
deterministically by (family, primary_tag, id).

### Output structure (new)

Each run writes a `runs/<timestamp>_<model_slug>/` directory:

```
results.json           # summary + per-task records
summary.csv            # flat table for plotting
trajectories/<id>.jsonl # full step trace per task (action, reward, feedback)
log.txt                # mirrors stdout
```

`results.json` shape:
```json
{
  "model": "...",
  "split": "eval", "family": "all",
  "n_tasks": 22, "avg_score": 0.456, "success_rate": 0.318,
  "total_elapsed_s": 1840.5,
  "by_family": {
    "xlsx": {"n": 10, "avg": 0.521},
    "docx": {"n": 4, "avg": 0.402},
    "pptx": {"n": 8, "avg": 0.388}
  },
  "results": [{ "task_id":..., "score":..., "step_rewards":[...], ...}]
}
```

`summary.csv` columns: `task_id, family, primary_tag, split, score, success,
steps, elapsed_s, error` — feeds straight into matplotlib/seaborn for the
hero plot in the README.

### Family-aware system prompts (new)

The single prompt mentioning `openpyxl` is replaced by three:

| Family | Prompt mentions |
|---|---|
| xlsx | `openpyxl.load_workbook`, `wb.save(path)` |
| docx | `from docx import Document`, `doc.save(path)`, common imports for shared/enum |
| pptx | `from pptx import Presentation`, `prs.save(path)`, color/util imports |

Selection is by `obs["family"]` (env-provided, with fallback to the
manifest's `family` field).

### Other changes

- `MAX_STEPS` default raised from 10 → 15 to match the env's actual cap
  (was undercutting agents on hard tasks)
- `TASK_TIMEOUT` raised from 240s → 360s — pptx tasks have larger files
  and need more inspection time
- Task selection auto-injects the 10 hand-curated `task_1..task_10` (which
  live in `tasks.py`, not the manifest) so they remain runnable via
  `--task-ids`
- Action extractor now also recognizes `docx`/`pptx` strings as code-block
  hints (was openpyxl-only)
- Trajectory persistence: every (action, reward, feedback) tuple is saved
  per task — useful as **input to SFT warm-start** in the eventual training
  loop

### Smoke validation

- `--help` prints clean usage
- Loads 119 tasks from manifest + injects 10 hand-curated; selects:
  - `--split eval` → 22 tasks (10 xlsx + 4 docx + 8 pptx) ✓
  - `--task-ids finch_10,osworld_0a0faba3,pptarena_case_60_fix_text_placement` → 3 tasks ✓
- Output writers (json/csv/jsonl) round-trip cleanly via synthetic test

A full live benchmark (with model API + env server) is the user's next
action — costs ~$0.50-2 in API tokens for a 22-task eval depending on model.

### Modified files

- [`inference.py`](inference.py) — full rewrite (~400 lines, was ~350)

### Files unchanged in Phase 6

- All env-server code, graders, manifest, data, deps

---

## Phase 7 — Live-discovered exploit + anti-exploit fix

**Trigger:** during Kimi-K2.5 eval (Apr 25, 2026), the model submitted the
**unmodified source file in step 1** for two tasks and scored very high:

| Task | Edit type | Score on src-unchanged submit | Why it worked |
|---|---|---|---|
| `pptarena_case_40_hindu_center_titles` | Title alignment | 0.998 | Paragraph-level `alignment` wasn't in `_shape_style`; everything else (text, position, size, font attrs) was identical between source and gold |
| `pptarena_case_26_match_slide_colors_to_theme` | Theme color | 0.971 | Gold uses theme-color references (None RGB); source uses explicit RGB. The mismatch dilutes across 30 shapes for only ~3% drop |

This is genuine reward hacking by an inference-time agent, exactly what the
"hard to game" criterion in the judging guide warns about. Two fixes
delivered:

### Fix 1: extended `_shape_style` (catches the per-attribute gaps)

Added two new attributes to the per-shape style extractor:

| Attribute | Source | Catches |
|---|---|---|
| `para_alignment` | `shape.text_frame.paragraphs[0].alignment` | "Center the title" / "right-align" tasks |
| `fill_theme` | `shape.fill.fore_color.theme_color` (when fill is solid but `.rgb` raises) | "Match colors to theme" tasks where gold uses theme refs and source uses explicit RGB |

Reweighted `_STYLE_WEIGHTS` from 7 attrs → 9 attrs:

```
fill_rgb 0.22 | fill_theme 0.08 | font_rgb 0.17 | para_alignment 0.15
font_size_pt 0.12 | line_rgb 0.08 | font_name 0.08
font_bold 0.05 | font_italic 0.05
```

Status: improves shape-level discrimination, but the **dilution problem
still wins** when only 2 of 55 shapes change (case_40 src-vs-gold went
from 0.998 → 0.997 — basically unchanged because of averaging). This is
why we need Fix 2.

### Fix 2: byte-equality anti-exploit at grade time (the actual fix)

Added in [`graders/__init__.py`](graders/__init__.py)'s `grade_task`:
**if the agent's submitted file is byte-identical to the source AND the
task isn't OSWorld's `infeasible` sentinel, return 0.001 immediately.**

```python
if src_file_exists and not is_infeasible_task:
    if same_bytes(output_path, source_file):
        return 0.001  # agent didn't actually do anything
```

This kills the entire class of "submit source unchanged" exploits across
all three families, regardless of which specific attribute the diff
misses. Validation:

| Test | Before fix | After fix |
|---|---|---|
| Submit unmodified source on `case_40` | 0.998 | **0.001** ✓ |
| Submit unmodified source on `case_26` | 0.971 | **0.001** ✓ |
| Submit gold on `case_40` | 0.999 | 0.999 ✓ no regression |
| Submit gold on `case_26` | 0.999 | 0.999 ✓ no regression |
| All 8 pptx eval tasks, gold-vs-gold | 0.999 | 0.999 ✓ no regression |

The OSWorld `infeasible` task (where not modifying *is* the correct
answer) is correctly excluded — that path uses the existing `infeasible`
evaluator function which already does its own equality check and credits
the agent.

### Important implication for SFT corpus building

When we eventually filter trajectories for the SFT corpus, **drop any
trajectory where `n_steps == 1` and the only action was `submit_file`**
even after this fix. Reasons:
1. Defense in depth — if a future grader gap appears, we don't want the
   student model trained on "submit unchanged" wins
2. A real solve takes at least one code step; 1-step `submit_file` is
   structurally suspicious

This filter is documented as a TODO for the SFT collection script.

### Re-eval needed

The Kimi-K2.5 baseline numbers from `runs/baseline_kimi_k25_eval/` were
collected with the pre-fix grader. The two exploited tasks are now
correctly graded at 0.001 instead of 0.998/0.971, lowering the run's
average. Either re-run Kimi on those two tasks with `--resume`, or
recompute the average locally:

```bash
# Quick local recompute (no re-inference) — assumes you already pushed
# updated graders. The OLD numbers are inflated; the NEW numbers reflect
# what Kimi actually solved.
```

(Recommendation: re-run with `--resume --task-ids pptarena_case_40_hindu_center_titles,pptarena_case_26_match_slide_colors_to_theme`. Costs <$0.10.)

---

## Phase 8 — SFT corpus builder (trajectory → messages-format JSONL)

**Goal:** turn teacher trajectories (collected on the train split via
`inference.py --split train`) into an SFT-ready corpus for warm-starting
a small student model (Qwen2.5-Coder-3B-Instruct) before GRPO.

### New file
- [`data_pipeline/build_sft_corpus.py`](data_pipeline/build_sft_corpus.py)
  — reads a `runs/<dir>/{summary.csv, trajectories/*.jsonl}` produced by
  `inference.py`, applies six filters, and emits a JSONL where each row
  is one accepted episode in the
  [TRL `SFTTrainer` `messages` format](https://huggingface.co/docs/trl/main/en/sft_trainer):

  ```jsonl
  {"task_id": "...", "family": "xlsx", "primary_tag": "Calculation",
   "split": "train", "score": 0.94, "n_steps": 6,
   "messages": [
     {"role": "system",    "content": <SYSTEM_PROMPTS[family]>},
     {"role": "user",      "content": <task instruction + source path + family>},
     {"role": "assistant", "content": "```python\n…\n```"},
     {"role": "user",      "content": "Code execution result (step 1/15):\n…"},
     {"role": "assistant", "content": "SUBMIT_FILE: /…"},
     ...
   ]}
  ```

### Filters (in order)

| # | Filter | What it drops | Why |
|---|---|---|---|
| 1 | `error` column non-empty | Failed runs (timeouts, model crashes) | No useful signal |
| 2 | `n_steps < --min-steps` (default 2) | Trivial 1-step runs | Real solves take ≥1 code step |
| 3 | **1-step `submit_file`** | Trajectories where the only action is `submit_file` | **Defense in depth against grader exploits** — Phase 7 proved a model can submit source unchanged and beat the diff threshold; even with the byte-equality check, future grader gaps could re-open this. A real solve takes ≥1 code step; we never want to teach the student "skip the work". Always dropped, regardless of score. |
| 4 | `final_score < --score-threshold` (default 0.4) | Low-quality solves | Don't train on partial-fail patterns |
| 5 | Malformed action types | Action types outside `{code, submit, submit_file}` | Schema enforcement |
| 6 | No real work | Trajectories with no successful code step (`reward > 0.005`) | Drops "model only made syntax errors" cases |

The `--min-steps 2` and the explicit 1-step-submit-file check are
**redundant by design** — both catch the same exploit class so a future
refactor that loosens one doesn't open the door.

### Message reconstruction details

- **System prompt:** imported verbatim from `inference.SYSTEM_PROMPTS[family]`
  so the SFT corpus matches what the model sees at deployment.
- **First user message:** task instruction + constraints + source-file
  path (extracted from the trajectory's first code action via regex,
  falls back to manifest's `source_file`) + family + task type. The
  env's xlsx-summary section is intentionally skipped to avoid re-opening
  files at corpus-build time.
- **Assistant turns:** action content wrapped in the format the
  `extract_action()` parser expects:
  - `code` → ` ```python\n{content}\n``` `
  - `submit` → `SUBMIT_ANSWER: {content}`
  - `submit_file` → `SUBMIT_FILE: {content}`
- **User turns:** mirror inference.py's per-step feedback message:
  ```
  Code execution result (step {n}/{max_steps}):
  {feedback}

  Source file: {path}
  ```

### Smoke test (against the MiniMax-M2.1 eval run)

```
Input rows    : 22
Accepted      : 10
Drops:
  low_score                    12

Accepted breakdown:
  docx      2
  pptx      4
  xlsx      4
Avg steps   : 10.8
Avg score   : 0.794
```

For the actual SFT corpus we'll use **train-split teacher trajectories
from Kimi-K2.5**, not the eval baseline. With 97 train tasks at
~30–50% retention rate that's ~30–50 high-quality episodes — enough for
a meaningful SFT warm-start before GRPO.

### Modified files

- None (new file only)

### Files unchanged in Phase 8

- env server, graders, manifest, data, deps

---

## Phase 9 — Hard early-submit gate at the env layer

**Trigger:** during Phase-2 trajectory collection on the train split,
Kimi-K2.5 was *still* trying to submit the unmodified source file at
step 1 (e.g., `pptarena_case_91_add_qr_code`), even though the Phase-7
grader correctly scored it 0.001. Post-grading defense alone wasn't
enough — every wasted "submit at step 1" episode was lost training data
and burned API budget.

### Fix: refuse the action before grading

[`server/financial_environment.py`](server/financial_environment.py) now
tracks `_code_steps_taken` (incremented in `_handle_code` regardless of
success — even a failed code attempt counts). Both submit handlers
(`_handle_submit_file`, `_handle_submit_text`) check
`_code_steps_taken >= _min_code_steps_before_submit` (default 1) and
return early with explanatory feedback if not.

Crucially, **the rejection does NOT end the episode**:

- The agent gets back a feedback message: `❌ Submit rejected: you must
  execute at least 1 code step before submitting...`
- The reward for the rejected step is `0.001`
- `done=False` — the agent has its remaining steps (15 - n_used) to recover

This shape is exactly right for an RL agent: ending the episode would
make a single bad attempt catastrophic; keeping it open turns it into
a corrective signal.

The minimum is overridable via `FINANCIAL_ENV_MIN_CODE_STEPS` env var.
Set to `0` to disable the gate (useful only for debugging).

### Belt-and-suspenders: prompt also tells the model

[`inference.py`](inference.py)'s `_BASE_RULES` now includes:

> 6. **You MUST execute at least one code step before submitting.** The
>    environment will reject SUBMIT_ANSWER and SUBMIT_FILE on step 1 — you
>    need to read or modify the file with code first. Submitting the source
>    file unchanged is never a correct solve and will be rejected.

Defense in depth: the prompt prevents wasted retries on models that
follow instructions; the env layer enforces the rule on models that
don't.

### Smoke test results

```
Reset:          code_steps_taken = 0, min_required = 1

Step 1: submit_file (early)       → reward=0.001, done=False  ✓ rejected
Step 2: code (any code)           → counter increments to 1    ✓
Step 3: submit_file (after code)  → reward=normal, done=True   ✓ allowed
Step 1: submit (QA, early)        → reward=0.001, done=False   ✓ same gate
Disabled (env var=0)              → submit goes through        ✓
```

### Stack of defenses against the "submit unchanged" exploit class

This is now the third independent defense, all targeting the same
exploit class:

| Layer | Phase | What it does |
|---|---|---|
| **Env action gate** | **9** (this one) | **Refuse the submit action itself if no code step has been taken** |
| Grader byte-equality | 7 | If submit happens AND output is byte-identical to source → 0.001 |
| SFT corpus filter | 8 | Drop trajectories with `n_steps==1` and `submit_file` even at high score |

Layer 9 prevents the trajectory from existing in the first place.
Layer 7 catches it if Layer 9 is somehow bypassed (e.g.,
`FINANCIAL_ENV_MIN_CODE_STEPS=0`).
Layer 8 prevents future grader gaps from leaking into SFT training data.

### Modified files

- [`server/financial_environment.py`](server/financial_environment.py) —
  added `_code_steps_taken`, `_min_code_steps_before_submit`,
  `_early_submit_rejected()`. Both submit handlers gated.
- [`inference.py`](inference.py) — added rule #6 to `_BASE_RULES`.

### Files unchanged in Phase 9

- graders, manifest, data, deps

### Phase 9.1 — `--skip-completed` for cheap re-runs

After Phase 9 landed, the natural question was: "do I just run with `--resume`
and the env will sort it out?"  Answer: no — `--resume` alone re-runs every
selected task and merges. To save API spend on already-good trajectories,
added a `--skip-completed` flag to [`inference.py`](inference.py).

When set with `--resume`, drops tasks whose prior result is **clean**:
- `error` column empty
- `score >= --skip-completed-threshold` (default `0.05`)
- `steps > 1` — single-step results are the Phase-7 exploit pattern; always retried regardless of score

Re-runs only tasks that errored, scored low, or were single-step.  Concretely
for the existing MiniMax baseline run: 13 skipped (clean), 9 retried (low
score). For a Kimi train-split run with 1-step submit_file exploits, those
all fall into the "steps ≤ 1" bucket and get correctly re-tried under the
new Phase-9 env gate.

Usage:
```bash
python3 inference.py \
  --split train \
  --resume --skip-completed \
  --output-dir runs/teacher_kimi_k25_train \
  --model moonshotai/Kimi-K2.5 ...
```

If everything's already clean, the script prints "Nothing to do" and exits
without spending a cent.

---

## Phase 10 — SFT training script

**Goal:** warm-start `Qwen2.5-Coder-3B-Instruct` on the SFT corpus built
in Phase 8, before GRPO. Per the $45 budget plan (1× A100 80GB on HF Jobs
@ $2.50/hr), SFT runs ~6h ≈ $15 leaving ~$30 for GRPO + eval.

### New file
- [`train_sft.py`](train_sft.py) — TRL `SFTTrainer` driver. Loads the
  `messages`-format JSONL, applies the model's chat template, masks loss
  on user/system tokens (assistant-only loss), trains a LoRA adapter,
  optionally pushes to HF Hub.

### Key choices

| Decision | Why |
|---|---|
| **`assistant_only_loss=True`** | Multi-turn agent SFT — we don't want to train on env-generated user feedback, only on assistant turns (the things the model produces) |
| **LoRA r=32, alpha=64, all-linear targets** | Sweet spot for 3B+ models; full-FT memory cost is unjustified for a $45 budget |
| **bf16 + gradient checkpointing + 8K seq len** | Fits a 3B model + 32-rank LoRA + 8K context comfortably on A100 80GB; can be dropped to 4K + r=16 for L40S 48GB |
| **`packing=False`** | Multi-turn examples are too varied to pack cleanly; each episode is its own sample |
| **CLI: `--push-to-hub`** | Optional push for the GRPO step to pull the SFT adapter from Hub instead of local disk |
| **CLI: `--use-qlora`** | 4-bit quantization fallback for tighter VRAM (e.g. consumer GPU dev) |

### Command (HF Jobs)

```bash
hf jobs run \
  --hardware "Nvidia A100 - large" \
  --timeout 8h \
  --image "huggingface/transformers-pytorch-gpu:latest" \
  --secrets HF_TOKEN \
  -- \
  bash -c "pip install -U 'trl>=0.11' peft accelerate bitsandbytes && \
           python train_sft.py \
             --dataset data/sft_kimi_k25.jsonl \
             --output-dir /tmp/qwen3b-sft \
             --push-to-hub bpHigh/qwen3b-office-sft"
```

### Local smoke test

The argparse layer imports cleanly without GPU. The full training requires
a GPU + the trl/peft/accelerate stack — not run locally as part of CI; the
real validation is the HF Jobs run.

### Modified files

- None (new file only)

### Files unchanged in Phase 10

- env server, graders, manifest, data, deps

---

## Current state (post-Phase 10)

### Repo layout

```
openenv_financial_task_env/
├── data/
│   ├── manifest.jsonl                # 109 rows: 50 Finch + 21 OSWorld + 38 PPTArena
│   ├── 0/, 21/, 24/, …               # original 10 hand-curated task dirs (xlsx)
│   ├── finch_50/<orig_id>/{src,ref}.xlsx
│   ├── osworld_writer/<uuid>/<src + N gold files>.docx
│   └── pptarena/<slug>/{<slug>_src,<slug>_ref}.pptx
├── data_pipeline/
│   ├── finch_pull.py                 # Phase 1
│   ├── osworld_writer_pull.py        # Phase 3
│   └── pptarena_pull.py              # Phase 5
├── graders/
│   ├── __init__.py                   # grade_xlsx + grade_docx + grade_pptx + dispatcher
│   └── docx_metrics.py               # 16 OSWorld evaluator functions
├── rewards.py                        # Phase 2; updated in Phase 4
├── server/financial_environment.py   # gold stash + per-task eval signal wired in
├── tasks.py                          # manifest loader; absolute-path resolution
├── models.py                         # unchanged
├── client.py                         # unchanged
├── inference.py                      # unchanged
├── pyproject.toml                    # +python-docx, +python-pptx, +rapidfuzz, +Pillow
├── Dockerfile                        # +python-docx, +python-pptx, +rapidfuzz, +Pillow
├── openenv.yaml                      # unchanged from Round 1
└── edits.md                          # this file
```

### Task inventory

| Family | Source | Train | Eval | Total |
|---|---|---|---|---|
| xlsx | hand-curated | 10 | 0 | 10 |
| xlsx | Finch | 40 | 10 | 50 |
| docx | OSWorld writer | 17 | 4 | 21 |
| pptx | PPTArena | 30 | 8 | 38 |
| **total** | | **97** | **22** | **119** |

### Reward signal stack

| Layer | Purpose | Mode |
|---|---|---|
| Per-step `RewardTracker` | Dense process reward (6 components) | Always on |
| `progress` | Structural distance to gold ↓ | On for training, off for eval (`FINANCIAL_ENV_PROGRESS=0`) |
| `eval_check` | Per-task evaluator score ↑ | Auto-enabled when task has an evaluator block (currently docx only) |
| Final grade — xlsx | 30% sheet-name + 70% cell-level diff | Submit-only |
| Final grade — docx | Validity gate + 40% diff + 60% per-task evaluator | Submit-only |
| Final grade — pptx | Validity gate + 20% slide-count + 80% avg(40% text + 20% style + 20% position + 20% size) | Submit-only |

### Defenses against reward hacking

| Vector | Status | Details |
|---|---|---|
| Persistent globals | ✅ Each step is fresh `subprocess.run` |
| Time runaway | ✅ 30s subprocess timeout |
| Memory runaway | ⚠️ No `ulimit` yet (TODO) |
| Glob the gold via `data/` | ✅ Gold moved out of `data/` for the episode |
| Read manifest.jsonl to find gold path | ⚠️ Still reachable; would need full sandbox isolation (TODO) |
| Generic-distance gaming | ✅ `eval_check` rewards spec-aligned progress |
| **Submit-source-unchanged** (Phase 7) | ✅ Byte-equality check at grade time → 0.001 |
| **1-step-submit-file in SFT corpus** (Phase 8) | ✅ Builder drops these even at high score |
| **Early submit before any code step** (Phase 9) | ✅ Env refuses the action itself; episode stays open for recovery |
| `lib_engagement` regex gaming | 🟡 Trivial cap (0.010); AST-based check would harden (TODO) |
| `mutation` spam | 🟡 Capped per-step but could spam-save garbage; could couple to progress (TODO) |

---

## Open issues / next steps (not yet done)

1. ~~**Layout-aware pptx diff**~~ — **DONE** in Phase 5 follow-up. Position
   + size matching with tolerance now active. 5 of 8 eval tasks meaningfully
   degrade source-vs-gold; 3 styling-only tasks still don't (see #2).

2. ~~**Style-aware pptx diff**~~ — **DONE** (Phase 5 follow-up #2). 7-attribute
   style match (fill/line color, first-run font name/size/bold/italic/color).
   Unblocked the pure-styling task `case_26` (discrimination 0 → 0.028).

3. **Edit-zone masking for pptx** — current diff averages over all shapes,
   so small targeted edits get diluted. Mask the score to shapes whose
   attributes differ between source and gold. Changes semantics: "did the
   agent fix the parts that were supposed to change" instead of "how close
   to gold overall". ~30 lines. **Priority:** medium — biggest improvement
   on tasks where edit surface is <5% of the deck.

4. **PPTX VLM judge** (optional, behind `PPTX_VLM_JUDGE=1`): render slides
   via headless LibreOffice → PNG, send (instruction, before, after, ref)
   to a VLM. Matches PPTArena's published methodology. Expensive — defer
   to final eval-time only, not training inner loop.

3. **TSBench** — skipped this round because it ships only `ideal_description`
   text (no gold files). Could add later as an LLM-judge family. Would
   need a separate grader; structurally similar to a per-task evaluator
   that calls Claude/GPT-4o with `(diff_summary, ideal_description)`.

4. **Memory cgroup** on agent subprocess: prevent OOM-bomb step from killing
   the env server.

5. **AST-based library check** in rewards.py: replace regex with real call
   detection so `import openpyxl  # decoy` doesn't earn the bonus.

6. **Couple mutation reward to progress**: only credit `mutation` if
   `progress > 0` in the same step OR last N steps — kills the spam-save
   strategy while preserving exploration credit.

7. **Manifest hiding for full sandbox isolation**: at server startup, also
   move/redact `data/manifest.jsonl` so a determined agent can't read it
   from the subprocess. Better: deploy with the data tree mounted at a
   path the agent's cwd subtree can't reach (bwrap, or docker bind-mounts
   to e.g. `/var/lib/openenv_data`).

8. **Test on more docx evaluator types** end-to-end. Currently smoke-tested
   `compare_docx_files` (single + compound `and`) and `compare_docx_tables`.
   Should sweep all 16 evaluators with synthetic agent outputs.

9. **`osworld_0a0faba3` quirk** — gold doesn't self-pass `check_tabstops`
   constraint due to a 2-words-before-tab paragraph. Either move to eval-only
   or relax the constraint.

10. **Inference baseline** — re-run the Round-1 inference script across all
    119 tasks (or a stratified subset) to refresh the README scoreboard.

11. **README rewrite** — current README is Round-1. Needs the cross-format
    pitch (xlsx + docx + pptx), the multi-layer grader story, the
    gaming-resistance angle.

12. **Training script** — TRL/Unsloth GRPO with LoRA on Qwen2.5-Coder-3B,
    trajectory-collection from a teacher (Claude Haiku 4.5), + SFT warm-start.
    Per the earlier $100-budget plan.

---

## Re-deploy checklist

If a fresh contributor wants to reproduce the current state from
commit `bf77949`:

1. `pip install -e ".[dev]"` (now pulls python-docx, python-pptx, rapidfuzz, Pillow)
2. `python data_pipeline/finch_pull.py` — ~3 min, downloads ~42 MB
3. `python data_pipeline/osworld_writer_pull.py` — ~30 s, downloads ~10 MB
4. Download/clone PPTArena to a local path (e.g. `~/Downloads/PPTArena-main`),
   then `python data_pipeline/pptarena_pull.py --root ~/Downloads/PPTArena-main`
   — copies ~244 MB
5. Check `data/manifest.jsonl` has 109 lines (50 + 21 + 38)
6. `python -c "from tasks import TASKS; print(len(TASKS))"` should print 119
7. Smoke test: `python -c "from server.financial_environment import FinancialEnvironment; e = FinancialEnvironment(); o = e.reset(task_id='finch_10'); print(o.task_id)"`
8. Docker build: `docker build -t financial-task-env:latest .` — should complete cleanly with the new deps

For training (RL):
- Set `FINANCIAL_ENV_PROGRESS=1` (default) for dense gradient
- Ensure each rollout worker uses its own `FinancialEnvironment` instance — gold-stash is single-tenant per task