---
license: apache-2.0
task_categories:
  - reinforcement-learning
language:
  - en
tags:
  - openenv
  - rl-environment
  - agent-environment
  - finance
pretty_name: Finance Desk
configs:
  - config_name: default
    data_files: tasks.jsonl
---

# Finance Desk — OpenEnv environment tasks

Public task preview and source for the **Finance Desk** OpenEnv Arena
environment: deep finance & economics plus one or two families for each of the
other seven arena domains, trained as one policy.

- **Image (public, linux/amd64):** `ghcr.io/bp-high/finance-desk:v2`
- **Source:** https://github.com/bp-high/openenv_financial_task_env/tree/dev/pensive-lovelace-bt5rsn/finance_desk

Each episode is a freshly generated workspace of short CSV/JSON/memo files. The
agent reads them, computes with a sandboxed Python tool, and submits one JSON
object. **v2 is single-step**: every file's full contents are embedded in the first observation (aligned with the top MetaHarness env), so a task can be answered in one submit; a `python` tool stays available for heavy arithmetic. Reward is **all-or-nothing**: `1.0` only if every documented field is
exactly right, else `0.0`. Level-3 tasks hide a `notice_*.txt` that overrides
part of the data, so skipping a file yields a wrong answer.

`tasks.jsonl` has three illustrative instances per submitted task id
(`prompt`, `files`, `answer`, `domain`, `level`, `seed`). The container
regenerates fresh instances at training time; this preview is illustrative, not
the evaluated set. The ABC-notation (music) family is derived from
[`XiaomiMiMo/MiMo-V2.6-RL-oss`](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)
(Apache-2.0).

`source/fdesk/` is the exact environment package (generators + verifiers).
