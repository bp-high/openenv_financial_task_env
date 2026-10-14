import copy
import json
import os
import subprocess
import sys

import pytest

from fdesk.environment import MAX_STEPS, FinanceDeskEnvironment
from fdesk.families import FAMILIES, task_ids
from fdesk.models import DeskAction

SEEDS = range(12)


def play(env, task_id, seed, action):
    env.reset(seed=seed, task_id=task_id)
    return env.step(DeskAction(**action))


def oracle(env):
    return json.loads(json.dumps(env._instance.answer))


@pytest.mark.parametrize("task_id", task_ids())
def test_oracle_scores_one_and_wrong_scores_zero(task_id):
    env = FinanceDeskEnvironment()
    for seed in SEEDS:
        env.reset(seed=seed, task_id=task_id)
        ans = oracle(env)
        obs = env.step(DeskAction(tool="submit", args={"answer": ans}))
        assert obs.done and obs.reward == 1.0, (task_id, seed, ans)
        for key in ans:  # break one key at a time
            env.reset(seed=seed, task_id=task_id)
            bad = copy.deepcopy(ans)
            v = bad[key]
            if isinstance(v, bool):
                bad[key] = not v
            elif isinstance(v, (int, float)):
                bad[key] = v + max(1.0, abs(v) * 0.01)
            elif isinstance(v, list):
                bad[key] = v[:-1] if v else ["X-NONE"]
            else:
                bad[key] = str(v) + "9"
            obs = env.step(DeskAction(tool="submit", args={"answer": bad}))
            assert obs.done and obs.reward == 0.0, (task_id, seed, key)


@pytest.mark.parametrize("task_id", task_ids())
def test_finish_and_budget_score_zero(task_id):
    env = FinanceDeskEnvironment()
    obs = play(env, task_id, 0, {"tool": "submit", "args": {"answer": {}}})
    assert obs.done and obs.reward == 0.0
    env.reset(seed=1, task_id=task_id)
    for i in range(MAX_STEPS):
        obs = env.step(DeskAction(tool="list", args={}))
    assert obs.done and obs.reward == 0.0


def test_lenient_action_shapes():
    env = FinanceDeskEnvironment()
    env.reset(seed=3, task_id="loan_amort-l1")
    obs = env.step(DeskAction(**{"tool": "read", "path": "loan.json"}))
    assert "principal" in obs.output and not obs.done
    ans = oracle(env)
    obs = env.step(DeskAction(tool="submit", args={k: str(v) for k, v in ans.items()}))
    assert obs.reward == 1.0


def test_python_tool_reads_workspace():
    env = FinanceDeskEnvironment()
    env.reset(seed=5, task_id="fx_consolidation-l2")
    obs = env.step(DeskAction(tool="python", args={"code": "import pandas as pd\nprint(len(pd.read_csv('entities.csv')))"}))
    assert obs.output.strip().split("\n")[0].isdigit(), obs.output
    obs = env.step(DeskAction(tool="python", args={"code": "while True: pass"}))
    assert "killed" in obs.output or "exit code" in obs.output


def test_notices_change_the_answer():
    for name, fam in FAMILIES.items():
        if 3 not in fam.levels:
            continue
        env = FinanceDeskEnvironment()
        env.reset(seed=0, task_id=f"{name}-l3")
        assert any(n.startswith("notice") for n in env._instance.files), name


def test_reset_is_deterministic_across_processes():
    code = ("import json,sys; sys.path.insert(0,'.'); from fdesk.environment import FinanceDeskEnvironment as E; "
            "from fdesk.families import task_ids; e=E(); out={}\n"
            "for t in task_ids():\n e.reset(seed=7, task_id=t); out[t]=e._instance.answer\n"
            "print(json.dumps(out, sort_keys=True))")
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONHASHSEED": "123"}).stdout
    assert a == b


def test_brief_is_short():
    env = FinanceDeskEnvironment()
    for t in task_ids():
        obs = env.reset(seed=0, task_id=t)
        assert len(obs.output) < 8000, (t, len(obs.output))  # inline files make the first observation larger by design
        assert sum(len(v) for v in env._instance.files.values()) < 9000, t
