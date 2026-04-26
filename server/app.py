"""FastAPI application for the Office Document Task Environment.

Wraps OpenEnv's standard create_app() and mounts a Gradio dashboard at
``/dashboard``.  The OpenEnv playground continues to live at ``/web``;
the dashboard is a project summary view with leaderboard, training
plots, and a file-upload preview for ad-hoc tasks.

Routes:
  /web         OpenEnv playground (baked-in)
  /dashboard   Gradio Blocks app — leaderboard, plots, upload demo
  /docs        FastAPI docs (baked-in)
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from openenv.core.env_server.http_server import create_app

from models import FinancialAction, FinancialObservation
from server.financial_environment import FinancialEnvironment


app = create_app(
    FinancialEnvironment,
    FinancialAction,
    FinancialObservation,
    env_name="financial_task_env",
)


# ---------------------------------------------------------------------------
# Dashboard data helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
PLOTS_DIR = RUNS_DIR / "sft_plots"


def _load_results(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "results.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _leaderboard_rows() -> List[List[Any]]:
    """Each row: [Model, Kind, Avg, Success, xlsx, docx, pptx, n].  Pending
    runs render as '—' so judges can see what's still missing."""
    candidates = [
        ("MiniMaxAI/MiniMax-M2.1", "frontier baseline", "baseline_minimax_m21_eval"),
        ("moonshotai/Kimi-K2.5", "teacher", "baseline_kimi_k25_eval"),
        ("Qwen/Qwen2.5-Coder-3B-Instruct", "student baseline (vanilla)",
         "baseline_qwen25coder3b_eval"),
        ("Qwen3B + LoRA SFT (4K)", "student trained, 4K context",
         "sft_eval_v2/bpHigh_qwen3b-office-sft-kimi"),
        ("Qwen3B + LoRA SFT (8K)", "student trained, 8K context",
         "sft_eval_v2/bpHigh_qwen3b-office-sft-kimi-long"),
    ]

    def _f(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    def _pct(v):
        if v is None:
            return "—"
        return f"{v * 100:.0f}%"

    rows = []
    for label, kind, dirname in candidates:
        d = RUNS_DIR / dirname
        r = _load_results(d)
        if r is None:
            rows.append([label, kind, "—", "—", "—", "—", "—", "—"])
            continue
        bf = r.get("by_family", {})
        rows.append([
            label, kind,
            _f(r.get("avg_score")),
            _pct(r.get("success_rate")),
            _f(bf.get("xlsx", {}).get("avg")),
            _f(bf.get("docx", {}).get("avg")),
            _f(bf.get("pptx", {}).get("avg")),
            r.get("n_tasks", 0) or "—",
        ])
    return rows


def _sft_summary_rows() -> List[List[Any]]:
    """Per-SFT-run table: [Run, Final Loss, Runtime, Epochs]."""
    out = []
    for sub, label in (("qwen3b_kimi", "4K context"),
                       ("qwen3b_kimi_long", "8K context")):
        p = PLOTS_DIR / sub / "summary.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        loss = data.get("train_loss")
        rt = data.get("train_runtime")
        out.append([
            label,
            f"{loss:.4f}" if isinstance(loss, (int, float)) else "—",
            f"{rt:.0f}s" if isinstance(rt, (int, float)) else "—",
            f"{data.get('epoch')}",
        ])
    return out


def _task_inventory() -> Dict[str, int]:
    manifest = REPO_ROOT / "data" / "manifest.jsonl"
    counts = {"xlsx": 10, "docx": 0, "pptx": 0}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                fam = row.get("family", "xlsx")
                counts[fam] = counts.get(fam, 0) + 1
            except Exception:
                pass
    return counts


def _comparison_plot_path() -> Optional[str]:
    p = PLOTS_DIR / "comparison_4k_vs_8k.png"
    return str(p) if p.exists() else None


def _per_run_plot(run: str) -> Optional[str]:
    p = PLOTS_DIR / run / "sft_loss_curve.png"
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Trajectory replay — find Kimi's best run per family, render step-by-step
# with reward decomposition explained.
# ---------------------------------------------------------------------------

import csv  # noqa: E402
import re   # noqa: E402

REWARD_EXPLANATIONS = {
    "exec_health":    "Subprocess returned 0; bonus for non-empty stdout",
    "lib_engagement": "Code uses the family's expected library (openpyxl / python-docx / python-pptx)",
    "mutation":       "Working file's SHA-256 changed since last step (real edit, not just a re-save)",
    "validity":       "Mutated file still parses with the family's loader (no corruption)",
    "progress":       "Structural distance to the gold reference decreased",
    "eval_check":     "Per-task evaluator score went up (docx-only — runs the OSWorld check function)",
}

REWARD_LINE_RE = re.compile(r"Reward:\s*total=([\d.]+)\s*\((.*?)\)")


def _parse_reward_components(feedback: str) -> Tuple[Optional[float], Dict[str, float]]:
    m = REWARD_LINE_RE.search(feedback or "")
    if not m:
        return None, {}
    total = float(m.group(1))
    comps: Dict[str, float] = {}
    for kv in m.group(2).split(","):
        kv = kv.strip()
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                comps[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return total, comps


def _strip_reward_block(feedback: str) -> str:
    """Remove the 'Reward: total=...' tail from a feedback string so we can
    render the env's actual stdout/stderr separately from the breakdown."""
    if not feedback:
        return ""
    idx = feedback.find("\n\nReward: total=")
    if idx == -1:
        idx = feedback.find("Reward: total=")
    if idx == -1:
        return feedback
    return feedback[:idx].rstrip()


def _find_best_kimi_per_family() -> Dict[str, Dict[str, Any]]:
    """Top Kimi-K2.5 score per family from the training-set teacher run.
    Returns {family: {task_id, score, primary_tag, instruction}}."""
    teacher_dir = RUNS_DIR / "teacher_kimi_k25_train"
    summary = teacher_dir / "summary.csv"
    if not summary.exists():
        return {}

    best: Dict[str, Tuple[float, str, str]] = {}
    with open(summary) as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            fam = row.get("family", "")
            try:
                score = float(row.get("score", 0))
            except ValueError:
                continue
            if score > best.get(fam, (0.0,))[0]:
                best[fam] = (score, row["task_id"], row.get("primary_tag", ""))

    # Pull the instruction text from the manifest so we can show what the
    # task actually asked for.
    manifest_path = REPO_ROOT / "data" / "manifest.jsonl"
    instruction_by_id: Dict[str, str] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                instruction_by_id[row["id"]] = row.get("instruction", "")
            except Exception:
                pass

    out = {}
    for fam, (score, tid, tag) in best.items():
        out[fam] = {
            "task_id": tid,
            "score": score,
            "primary_tag": tag,
            "instruction": instruction_by_id.get(tid, ""),
        }
    return out


def _load_trajectory(task_id: str, run_dir_name: str = "teacher_kimi_k25_train") -> List[Dict[str, Any]]:
    p = RUNS_DIR / run_dir_name / "trajectories" / f"{task_id}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _render_step_html(step: Dict[str, Any], step_idx: int) -> str:
    """Pretty-print a single step: action + env stdout + reward breakdown."""
    action_type = step.get("action_type", "")
    action_content = step.get("action_content", "")
    feedback = step.get("feedback", "") or ""
    reward = float(step.get("reward") or 0)
    done = step.get("done", False)

    total, comps = _parse_reward_components(feedback)
    body_feedback = _strip_reward_block(feedback)
    if len(action_content) > 2400:
        action_content = action_content[:2400] + "\n# … (truncated for display)"
    if len(body_feedback) > 2000:
        body_feedback = body_feedback[:2000] + "\n... (truncated for display)"

    # Reward-breakdown table for code steps; final-grade callout for submits.
    if comps:
        rows_html = ""
        for key, val in comps.items():
            checkmark = "✅" if val > 0 else "—"
            why = REWARD_EXPLANATIONS.get(key, "")
            rows_html += (
                f'<tr>'
                f'<td style="text-align:center;width:30px">{checkmark}</td>'
                f'<td><code>{key}</code></td>'
                f'<td style="text-align:right;font-variant-numeric:tabular-nums">'
                f'{val:.3f}</td>'
                f'<td style="color:#64748b;font-size:.88em">{why}</td>'
                f'</tr>'
            )
        breakdown_html = (
            '<table style="width:100%;border-collapse:collapse;margin-top:.5rem">'
            '<thead><tr style="border-bottom:1px solid #e5e7eb">'
            '<th></th><th style="text-align:left">component</th>'
            '<th style="text-align:right">value</th>'
            '<th style="text-align:left;color:#64748b;font-weight:500">why this fired</th>'
            f'</tr></thead><tbody>{rows_html}'
            '<tr style="border-top:1px solid #e5e7eb;font-weight:600">'
            '<td></td><td>total (capped at 0.10)</td>'
            f'<td style="text-align:right">{total:.3f}</td><td></td></tr>'
            '</tbody></table>'
        )
    else:
        breakdown_html = (
            '<div style="padding:.5rem .75rem;background:#fef3c7;'
            'border-left:3px solid #d97706;margin-top:.5rem">'
            f'<strong>Final grade: {reward:.3f}</strong> — '
            "computed by the family's grade_xxx() function "
            '(validity gate + diff + per-task evaluator).'
            '</div>'
        )

    action_color = {
        "code": "#2563eb",
        "submit": "#059669",
        "submit_file": "#059669",
    }.get(action_type, "#64748b")
    done_badge = (
        '<span style="background:#10b981;color:white;padding:.1rem .4rem;'
        'border-radius:.25rem;font-size:.7em;margin-left:.4rem">DONE</span>'
        if done else ''
    )

    # Pre-build the chunks that involve quotes / backslashes so the f-string
    # below stays simple (Python 3.10 disallows backslashes in f-string exprs).
    action_html = _html_escape(action_content)
    feedback_html = (
        _html_escape(body_feedback) if body_feedback
        else '<em style="color:#9ca3af">(no output)</em>'
    )

    return f"""
<div style="border:1px solid #e5e7eb;border-radius:6px;padding:1rem;margin-bottom:1rem;background:white">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem">
    <strong>Step {step_idx}</strong>
    <div>
      <span style="color:{action_color};font-family:monospace;font-size:.9em">
        action_type=&quot;{action_type}&quot;</span>
      {done_badge}
      <span style="margin-left:1rem;color:#64748b">reward = <strong>{reward:.3f}</strong></span>
    </div>
  </div>

  <div style="color:#64748b;font-size:.85em;margin-top:.7rem">Agent action:</div>
  <pre style="background:#1f2937;color:#e5e7eb;padding:.75rem;border-radius:4px;
              overflow-x:auto;font-size:.82rem;margin:.3rem 0;max-height:300px">{action_html}</pre>

  <div style="color:#64748b;font-size:.85em;margin-top:.7rem">Env feedback:</div>
  <pre style="background:#f3f4f6;color:#1f2937;padding:.75rem;border-radius:4px;
              overflow-x:auto;font-size:.82rem;margin:.3rem 0;max-height:200px">{feedback_html}</pre>

  {breakdown_html}
</div>
"""


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


def _render_replay_section(family: str, info: Dict[str, Any]) -> str:
    """Full replay HTML block for one family's best Kimi run."""
    task_id = info["task_id"]
    score = info["score"]
    tag = info["primary_tag"]
    instruction = info["instruction"]
    traj = _load_trajectory(task_id)
    if not traj:
        return f'<p class="muted">No trajectory available for {task_id}.</p>'

    fam_label = {"xlsx": "Excel", "docx": "Word", "pptx": "PowerPoint"}.get(family, family)
    header = f"""
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;
            padding:1rem;margin-bottom:1rem">
  <div style="display:flex;justify-content:space-between">
    <div>
      <strong>📄 {fam_label} ({family})</strong> · task <code>{task_id}</code>
      <span style="color:#64748b">· {tag}</span>
    </div>
    <div><strong>Final score: {score:.3f}</strong> · {len(traj)} steps</div>
  </div>
  <div style="color:#475569;font-size:.92em;margin-top:.5rem">
    <strong>Instruction:</strong> {_html_escape(instruction[:600])}
    {"..." if len(instruction) > 600 else ""}
  </div>
</div>
"""
    steps_html = ""
    for i, step in enumerate(traj, start=1):
        steps_html += _render_step_html(step, i)
    return header + steps_html


# ---------------------------------------------------------------------------
# File upload handler
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path(tempfile.gettempdir()) / "openenv_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_EXTS = {".xlsx", ".docx", ".pptx"}


def handle_upload(file_obj, instruction: str):
    """Accept a user-uploaded office document + task description.  We don't
    run an agent (no hosted LoRA), but we save the file and emit a copy-pasteable
    inference command the user can run locally — making the demo useful without
    needing a model endpoint live in the Space.
    """
    if file_obj is None:
        return ("⚠️ Please upload a .xlsx / .docx / .pptx file first.", "", None)

    src_path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    ext = src_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        return (f"⚠️ Unsupported extension `{ext}`. "
                f"Need one of: {', '.join(sorted(SUPPORTED_EXTS))}", "", None)

    family = ext.lstrip(".")
    instruction = (instruction or "").strip()
    if not instruction:
        return ("⚠️ Add a task instruction (e.g., 'center the title on slide 1').",
                "", None)

    # Save to upload dir with original-ish filename
    dest = UPLOAD_DIR / f"user_{src_path.stem}{ext}"
    try:
        shutil.copy2(src_path, dest)
    except Exception as e:
        return (f"⚠️ Couldn't save upload: {e}", "", None)

    # Tell the user how to use it.  Since the Space can't run their fine-tuned
    # model live (no hosted LoRA endpoint), we produce instructions for two
    # paths: (a) hit the Space's HTTP API with their own model, or (b) run
    # eval_lora.py locally against an uploaded one-off task.
    summary = (
        f"✅ Saved `{dest.name}` ({dest.stat().st_size:,} bytes)\n\n"
        f"**Detected family**: `{family}`  ·  **Task**: \"{instruction[:120]}\""
    )

    api_snippet = f"""# Option A — hit the env API directly with your own LLM
# (the Space's playground at /web works for one-off interactive runs)

# Option B — register as an ad-hoc task locally + run inference.py
git clone https://github.com/bp-high/openenv_financial_task_env.git
cd openenv_financial_task_env
mkdir -p data/user_uploads/my_task
cp /path/to/your/{src_path.name} data/user_uploads/my_task/source{ext}

# Add this row to data/manifest.jsonl:
echo '{{"id": "user_my_task", "family": "{family}", "origin": "user_upload",
"split": "eval", "primary_tag": "user", "all_tags": ["user"],
"business_type": "user_upload",
"instruction": "{instruction}",
"constraints": "",
"source_file": "data/user_uploads/my_task/source{ext}",
"reference_file": "",
"task_type": "MODIFY", "max_steps": 15}}' >> data/manifest.jsonl

# Run inference against your task with any HF Router model:
python inference.py --task-ids user_my_task \\
  --model moonshotai/Kimi-K2.5 \\
  --api-base https://api.tokenfactory.us-central1.nebius.com/v1/

# OR run our SFT'd Qwen3B against it (no API needed):
python eval_lora.py --task-ids user_my_task \\
  --adapters bpHigh/qwen3b-office-sft-kimi-long
"""
    return summary, api_snippet, str(dest)


# ---------------------------------------------------------------------------
# Build the Gradio dashboard
# ---------------------------------------------------------------------------

INTRO_MD = """\
# 📊 Office Document Task Environment

Cross-format RL environment — **Excel · Word · PowerPoint** —
119 tasks, real enterprise artifacts, gaming-resistant grading,
SFT'd Qwen3-3B student.

[🎮 OpenEnv Playground](/web) ·
[📦 GitHub](https://github.com/bp-high/openenv_financial_task_env) ·
[🤖 SFT Adapter (8K)](https://huggingface.co/bpHigh/qwen3b-office-sft-kimi-long) ·
[🤖 SFT Adapter (4K)](https://huggingface.co/bpHigh/qwen3b-office-sft-kimi) ·
[📜 API docs](/docs)
"""

REWARD_MD = """\
Every per-step reward measured from **real file state**, not regex on the
agent's code:

| Signal | Range | What it actually checks |
|---|---|---|
| `exec_health` | 0–0.020 | Subprocess exit code 0; bonus if stdout non-empty |
| `lib_engagement` | 0–0.010 | Code uses the family's expected library |
| `mutation` | 0–0.030 | SHA-256 of working file changed since last step |
| `validity` | 0–0.020 | Mutated file still parses |
| `progress` | 0–0.040 | Structural distance to gold *decreased* |
| `eval_check` | 0–0.020 | Per-task evaluator score *increased* (docx-only) |
"""

DEFENSES_MD = """\
A model trying to game the grader (e.g., submit unmodified source on step 1)
has to defeat **four independent defenses**:

| Layer | Phase | What it does |
|---|---|---|
| Env action gate | 9 | Refuses `submit_file` if no code step has been taken |
| Per-episode gold stash | 4 | Gold files moved to `/tmp/oe_gold_<random>/` at episode start |
| Grader byte-equality | 7 | If submit's bytes match source bytes → score=0.001 |
| SFT corpus filter | 8 | Builder drops `n_steps==1 + submit_file` trajectories at any score |

Live story of how Kimi-K2.5 found this exploit during eval and the 3 fixes
that followed: [edits.md Phase 7](https://github.com/bp-high/openenv_financial_task_env/blob/main/edits.md#phase-7--live-discovered-exploit--anti-exploit-fix).
"""

PIPELINE_MD = """\
The student SFT corpus came from **Kimi-K2.5** (the teacher) running on the
97 train tasks. Trajectories were filtered through 6 layers before landing
in the corpus:

1. error column non-empty → drop
2. `n_steps < 2` → drop
3. **1-step `submit_file`** → drop *(defense-in-depth — even at high score)*
4. `final_score < 0.4` → drop
5. malformed action_type → drop
6. no successful code step → drop

**97 raw → 53 SFT examples** (avg score 0.841, avg 7.7 steps).
Builder: `data_pipeline/build_sft_corpus.py`.
"""


def build_dashboard() -> gr.Blocks:
    counts = _task_inventory()
    total_tasks = sum(counts.values())

    with gr.Blocks(title="Office Document Task Env — Dashboard") as demo:

        gr.Markdown(INTRO_MD)

        # ---- Task inventory stats ----
        gr.Markdown("## Task inventory")
        with gr.Row():
            for label, n in [
                ("total tasks", total_tasks),
                (".xlsx (Finch + curated)", counts.get("xlsx", 0)),
                (".docx (OSWorld-Verified)", counts.get("docx", 0)),
                (".pptx (PPTArena)", counts.get("pptx", 0)),
            ]:
                gr.HTML(
                    f'<div style="text-align:center;padding:1rem;'
                    f'border:1px solid #e5e7eb;border-radius:8px;background:white">'
                    f'<div style="font-size:1.8rem;font-weight:700;color:#2563eb">{n}</div>'
                    f'<div style="color:#64748b;font-size:.85rem">{label}</div>'
                    f'</div>'
                )

        # ---- Leaderboard ----
        gr.Markdown("## Leaderboard — 22-task eval split")
        gr.DataFrame(
            value=_leaderboard_rows(),
            headers=["Model", "Kind", "Avg", "Success", "xlsx", "docx", "pptx", "n"],
            interactive=False,
            wrap=True,
        )
        gr.Markdown(
            "Rows showing `—` haven't been published yet. Reproduce: see "
            "`inference.py` (API-served) and `eval_lora.py` (in-process LoRA) "
            "in the README."
        )

        # ---- SFT runs ----
        gr.Markdown("## SFT training runs")
        gr.DataFrame(
            value=_sft_summary_rows() or [["—", "—", "—", "—"]],
            headers=["Run", "Final train_loss", "Runtime", "Epochs"],
            interactive=False,
        )
        gr.Markdown(
            "Hardware: 1× L40S 48GB on HF Jobs ($1.80/hr). "
            "Total SFT cost: **under $2**."
        )

        # ---- Plot ----
        gr.Markdown("## 4K vs 8K context length ablation")
        plot = _comparison_plot_path()
        if plot:
            gr.Image(value=plot, label="SFT loss — 4K vs 8K")
        else:
            gr.Markdown(
                "_Comparison plot not yet generated. Run "
                "`data_pipeline/compare_sft_runs.py` after the second SFT job finishes._"
            )

        # ---- Reward design ----
        gr.Markdown("## Reward design")
        gr.Markdown(REWARD_MD)

        # ---- Anti-hacking ----
        gr.Markdown("## Anti-hacking — 4 independent defenses")
        gr.Markdown(DEFENSES_MD)

        # ---- Trajectory pipeline ----
        gr.Markdown("## Trajectory collection pipeline")
        gr.Markdown(PIPELINE_MD)

        # ---- Replay: see Kimi solve a task ----
        gr.Markdown("## 🎬 Replay — see Kimi-K2.5 solve a task end-to-end")
        gr.Markdown(
            "Below: Kimi-K2.5's **best run per file family** from the training "
            "set, replayed step by step. Each step shows the agent's code, the "
            "env's stdout/stderr, and the per-component reward decomposition "
            "with explanations of *why* each component fired. Final score for "
            "all three: **0.999** — the env grader's near-max."
        )
        best = _find_best_kimi_per_family()
        # Order: xlsx → docx → pptx (story flows from simplest format to richest)
        for fam in ["xlsx", "docx", "pptx"]:
            info = best.get(fam)
            if not info:
                continue
            fam_label = {"xlsx": "Excel (xlsx)", "docx": "Word (docx)",
                         "pptx": "PowerPoint (pptx)"}[fam]
            with gr.Accordion(
                f"{fam_label} — task {info['task_id']} (score {info['score']:.3f})",
                open=(fam == "docx"),  # default-open the shortest one
            ):
                gr.HTML(_render_replay_section(fam, info))

        # ---- File upload demo ----
        gr.Markdown("## 🗂️ Try your own task")
        gr.Markdown(
            "Upload a `.xlsx` / `.docx` / `.pptx`, give it a task instruction, "
            "and we'll generate the inference command you can run against the "
            "env locally with any HF-Router model or our SFT'd Qwen3B adapter. "
            "Live agent runs aren't hosted in this Space (no inference endpoint "
            "for the LoRA), but the env + model run cleanly anywhere with a GPU."
        )
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(
                    label="Upload office doc",
                    file_types=[".xlsx", ".docx", ".pptx"],
                )
                instruction = gr.Textbox(
                    label="Task instruction",
                    placeholder="e.g., 'Center the title on slide 1' or "
                                "'Add a SUM formula in row 50 column F'",
                    lines=2,
                )
                submit = gr.Button("Generate inference command", variant="primary")
            with gr.Column(scale=2):
                upload_status = gr.Markdown()
                command_block = gr.Code(label="Run this locally", language="shell")
                saved_path = gr.Textbox(label="Saved upload path", visible=False)

        submit.click(
            fn=handle_upload,
            inputs=[upload, instruction],
            outputs=[upload_status, command_block, saved_path],
        )

        gr.Markdown(
            "---\n\n"
            "**Full project journey across 11 phases:** "
            "[edits.md](https://github.com/bp-high/openenv_financial_task_env/blob/main/edits.md) "
            "(~2,000 lines of phase-by-phase changelog)."
        )

    return demo


# Build + mount.  Gradio wraps itself as a sub-app inside FastAPI; the
# OpenEnv playground at /web is unaffected.
#
# `root_path="/dashboard"` is critical when mounted at a sub-path: Gradio
# generates static-asset and websocket URLs relative to `root_path`, so
# without it the iframe loads but every CSS/JS/WS request 404s and the
# page renders blank.
demo = build_dashboard()
app = gr.mount_gradio_app(app, demo, path="/dashboard", root_path="/dashboard")


# Redirect bare / to the dashboard so visitors landing on the root URL
# (anyone clicking the Space link) get the rich UI by default.  /web still
# serves the OpenEnv playground untouched.
from fastapi.responses import RedirectResponse  # noqa: E402


@app.get("/")
def _root():
    return RedirectResponse(url="/dashboard")


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
