# Finance Desk — a procedural multi-domain OpenEnv environment

Finance Desk is a single [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
environment for the [OpenEnv Arena](https://openenvarena-arena.hf.space). It is
built around **finance and economics** — the deepest part of the environment —
and adds one or two families for **each of the other seven arena domains**, so a
single policy trains across all eight at once.

Every episode is a small, freshly generated workspace. The agent reads a few
short CSV/JSON/memo files, computes the answer with a **sandboxed Python tool**
(run as `nobody`, CPU/memory/time limited, no access to the environment source),
and submits **one JSON object**. The reward is **all-or-nothing and exact**:
`1.0` only if every documented field is correct, otherwise `0.0`.

## Why it is built this way (lessons from the Arena board)

- **All-or-nothing grading.** Partial credit taught earlier entrants' policies
  to guess a field and submit in ~90 tokens; exact grading removes that collapse.
- **Deterministic verifiers.** Each answer is recomputed from the files the
  agent can see, so there is a unique correct answer and no reward hacking.
- **Hidden corrections (level 3).** Higher levels add a `notice_*.txt` file that
  overrides part of the data. An agent that stops before reading everything gets
  a different, wrong answer — directly training the "read the late caveat" habit.
- **Short observations.** Briefs and files are kept well under the context
  budget; the task instruction is given once, not repeated each turn.
- **Instance sharing.** Unseeded resets inside a 10-minute window return the
  same instance, so the parallel episodes of one GRPO group compare answers to
  the same problem.

## Action schema

One generic action serves every task:

```json
{"tool": "list" | "read" | "python" | "submit", "args": { ... }}
```

- `list {}` — file names and sizes.
- `read {"path": "x.csv"}` or `{"paths": [...]}` — file contents.
- `python {"code": "..."}` — run Python 3 in the workspace (numpy, pandas
  available; 10 s limit; prints are the output).
- `submit {"answer": { ...documented keys... }}` — grade once and end the
  episode.

Observations are `{"output": str, "steps_left": int, "reward": float, "done": bool}`.

## Domains and families (29 families, 3 difficulty levels each unless noted)

| Domain | Families |
|---|---|
| Finance & economics | loan amortization · bond pricing & accrued interest · NPV/XNPV/IRR · FX consolidation · depreciation & disposals · receivables aging · inventory FIFO/moving-average COGS · bank reconciliation · payroll withholding · cap table & SAFE conversion · PE distribution waterfall · credit covenants · Modified-Dietz/TWR returns · real-GDP growth & rebasing |
| Office & white-collar | expense-report audit · cross-timezone meeting scheduling |
| Industrial & physical | line OEE from shift/downtime logs · storage-tank simulation |
| Software engineering | release impact & min-cost test selection · dependency resolution |
| Natural science | first-order kinetics fit · Beer–Lambert calibration |
| Math & formal reasoning | toll-budget shortest route · bounded coin combinations |
| Cybersecurity | first-match firewall evaluation · SSH brute-force forensics |
| Media & content | subtitle QC/retiming · edit-decision-list timecodes · ABC-notation composition |

The ABC-notation (music) family is derived from
[`XiaomiMiMo/MiMo-V2.6-RL-oss`](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)
(Apache-2.0); everything else is original procedural generators with
independent deterministic verifiers.

## Reproduce the local validation

```sh
pip install "openenv @ git+https://github.com/meta-pytorch/OpenEnv.git@86a180ede21e044f7929b9a7783ad83aa67d83a3" pytest numpy pandas
pytest -q                           # 177 tests: oracle==1, every wrong field==0, finish/budget==0, caveats bite
docker build -t finance-desk .      # linux/amd64
docker run --rm -d -p 8000:8000 finance-desk
openenv validate --url http://localhost:8000
python scripts/ws_replay.py         # oracle & finish episodes over the WebSocket
```

## License

Apache-2.0. See `LICENSE`.
