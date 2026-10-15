import json
import os
import subprocess
import sys

import pytest

from fdesk.terminal.environment import TerminalDeskEnvironment
from fdesk.terminal.models import Tbench2Action
from fdesk.terminal.tasks import TASKS, task_ids

SEEDS = range(6)
A = Tbench2Action


@pytest.mark.parametrize("task_id", task_ids())
def test_oracle_passes_and_untouched_fails(task_id):
    env = TerminalDeskEnvironment()
    for seed in SEEDS:
        env.reset(seed=seed, task_id=task_id)
        oracle = env._inst.oracle
        obs = env.step(A(action_type="evaluate"))
        assert obs.done and obs.reward == 0.0, (task_id, seed, obs.output)
        env.reset(seed=seed, task_id=task_id)
        for path, content in oracle.items():
            assert env.step(A(action_type="write_file", file_path=path, content=content)).success
        obs = env.step(A(action_type="evaluate"))
        assert obs.done and obs.reward == 1.0, (task_id, seed, obs.output)


def test_every_observation_carries_the_instruction():
    env = TerminalDeskEnvironment()
    first = env.reset(seed=1, task_id="sw_stats_fix-l1")
    assert first.instruction and first.reward == 0.0 and not first.done
    obs = env.step(A(action_type="exec", command="ls"))
    assert obs.instruction == first.instruction and "tests" in obs.output and obs.reward is None


def test_wrong_finance_answer_scores_zero():
    env = TerminalDeskEnvironment()
    env.reset(seed=2, task_id="loan_amort-l2")
    ans = dict(env._inst.answer)
    ans["balance"] = ans["balance"] + 1
    env.step(A(action_type="write_file", file_path="answer.json", content=json.dumps(ans)))
    assert env.step(A(action_type="evaluate")).reward == 0.0
    env.reset(seed=2, task_id="loan_amort-l2")
    env.step(A(action_type="write_file", file_path="answer.json", content="{not json"))
    assert env.step(A(action_type="evaluate")).reward == 0.0


def test_write_file_stays_in_workdir_and_exec_reports_exit_codes():
    env = TerminalDeskEnvironment()
    env.reset(seed=0, task_id="sw_money_fix-l1")
    assert not env.step(A(action_type="write_file", file_path="../../etc/x", content="no")).success
    obs = env.step(A(action_type="exec", command="exit 3"))
    assert "[exit code 3]" in obs.output
    obs = env.step(A(action_type="wait", session_id="s"))
    assert not obs.success and obs.error


def test_hand_written_output_does_not_pass_report_task():
    # the expected out/summary.csv alone is not enough: hidden tests re-run build() on other inputs
    env = TerminalDeskEnvironment()
    env.reset(seed=3, task_id="sw_report_fix-l2")
    env.step(A(action_type="write_file", file_path="out/summary.csv", content=env._inst.oracle["out/summary.csv"]))
    assert env.step(A(action_type="evaluate")).reward == 0.0


def test_close_and_step_limit_end_with_a_grade():
    env = TerminalDeskEnvironment()
    env.reset(seed=0, task_id="bank_recon-l1")
    obs = env.step(A(action_type="close"))
    assert obs.done and obs.reward == 0.0
    from fdesk.terminal.environment import MAX_STEPS
    env.reset(seed=0, task_id="bank_recon-l1")
    for _ in range(MAX_STEPS):
        obs = env.step(A(action_type="exec", command="true"))
    assert obs.done and obs.reward == 0.0


def test_reset_is_deterministic_across_processes():
    code = ("import json,sys; sys.path.insert(0,'.'); from fdesk.terminal.environment import TerminalDeskEnvironment as E; "
            "from fdesk.terminal.tasks import task_ids; e=E(); out={}\n"
            "for t in task_ids():\n e.reset(seed=7, task_id=t); out[t]=e._inst.files\n"
            "print(json.dumps(out, sort_keys=True))")
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONHASHSEED": "123"}).stdout
    assert a == b


def test_instructions_fit_the_budget():
    env = TerminalDeskEnvironment()
    for t in task_ids():
        obs = env.reset(seed=0, task_id=t)
        assert len(obs.instruction) < 2000, (t, len(obs.instruction))
        assert sum(len(v) for v in env._inst.files.values()) < 12000, t
    assert {t.domain for t in TASKS.values()} == {"software engineering", "finance and economics"}
