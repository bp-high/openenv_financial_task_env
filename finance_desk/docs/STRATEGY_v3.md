# Finance Desk v3: why terminal mode

Written after v2 scored 3/40 (industrial 2, office 1) and the account reached 4/40 best-per-domain (rank 14/30).

## What the evidence says (leaderboard JSON, 81 evaluated heldout-v2 runs, and board posts #196–#232)

1. **Our banked domains are effectively capped.** Industrial has never exceeded 2 in 81 runs; office 3 happened
   twice, and office scores dropped board-wide after 2026-10-14 (dnakhla #232: 32/60 runs had office >= 2 before,
   2/22 after). The account holds industrial 2 (v2) + office 2 (railsim-v1). A lower run never lowers a best.
2. **Every remaining point is in rare domains.** Share of runs that solved >= 1 private task: software 21%
   (always exactly one task), finance 11%, media 4%, science / math / security ~1% each.
3. **So v3 risks nothing by abandoning industrial/office** and should maximise P(software >= 1 or finance >= 1).
4. **Training reaches the private score through generic habits, not domain content.** No-update runs score 0–2
   (mean 1.25) vs 2.97 for trained runs (naklitechie #229); a first-action collapse trained to saturation carried
   to the eval (rca-umb #223); domain-matched training has not shown up for anyone (dnakhla #230, Profa-QE #228).
5. **The private tasks are most likely SkillsBench-shaped tasks served through OpenEnv's `tbench2_env`**
   (dnakhla #196, rca-umb #197): files in a container, a prose instruction, `exec` / `write_file` / `evaluate`,
   pytest-graded output, budgets 4096 completion / 8192 context / 32 calls. The untrained model loses mostly
   before anything is graded: prose before the JSON, broken JSON escaping, dumping whole files until the budget
   ends the episode.
6. Code/tool environments roughly double the rare-domain hit rate (software 26% vs 15%, finance 14% vs 8% by a
   rough keyword split; small samples).

## What v3 is

- **The exact `tbench2_env` contract.** `fdesk/terminal/models.py` copies `Tbench2Action` / `Tbench2Observation`
  field for field; `/schema` is byte-identical to the operator's own TB2 smoke test
  (`benchflow/openenv-smoke-fix-git`). The model trains on the same system message, the same JSON observations
  (instruction repeated in each, like tbench2_env) and the same budgets it most likely meets on the private tasks.
- **Software + finance only.** Software: fix seeded bugs in a module (visible failing test / bug report / both),
  implement a log summarizer to a docstring spec, repair a report script and regenerate its output. All graded by
  hidden tests run against the agent's code (generation proves shipped code fails and the reference passes).
  Finance: the 14 v2 finance families, with the answer written to `answer.json`.
- **The habits it rewards are the transferable ones:** one clean JSON action per reply, inspect files
  economically, write/run code, produce the artifact, stay inside 4096 tokens.
- **Robust sandbox.** Commands run as `nobody` when the node allows switching users and fall back to the server
  user when SETUID/SETGID are dropped (#212/#213), tested both ways.
- **Calibrated on live Qwen3.8-27B** with an arena-faithful loop (strict single-JSON replies, `{"finish": true}`
  scored via `evaluate`, completion budget counting observations after the first). See docs/calibration_v3.md.

## Honest expectation

The draw model (#232) predicts best-per-domain totals of 2.86 / 3.90 / 4.56 / 5.02 / 5.37 after 1–5 runs; we are
at 4 after 3. dnakhla's terminal-desk-v3c (office/software/finance on the same contract) scored 2/40 with no new
point. v3 improves the odds of a software or finance point (estimate 30–40% per run); it does not guarantee one.
Submitting every 24 h remains the most reliable lever.
