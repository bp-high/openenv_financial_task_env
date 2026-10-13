"""Replay tasks against a running server over the OpenEnv WebSocket, the way the arena drives an episode.

Usage: python scripts/ws_replay.py [ws://host:port/ws] [n_per_family]
For each sampled (task_id, seed) it: reset -> reads files via the oracle-known answer -> submit oracle (expect
reward 1) and, separately, finish (expect reward 0). Reports pass/fail counts.
"""

import json
import random
import sys

from websockets.sync.client import connect

sys.path.insert(0, ".")
from fdesk.environment import FinanceDeskEnvironment  # noqa: E402  (to compute oracle answers)
from fdesk.families import FAMILIES, task_ids  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def rpc(ws, typ, **payload):
    ws.send(json.dumps({"type": typ, **payload}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("type") in ("reset_response", "step_response", "error", "state_response"):
            return msg


def oracle_answer(task_id, seed):
    env = FinanceDeskEnvironment()
    env.reset(seed=seed, task_id=task_id)
    return json.loads(json.dumps(env._instance.answer))


def main():
    ok = bad = 0
    details = []
    for task_id in task_ids():
        for seed in range(N):
            ans = oracle_answer(task_id, seed)
            with connect(URL, max_size=None) as ws:
                r = rpc(ws, "reset", data={"seed": seed, "task_id": task_id})
                obs = r.get("observation", {})
                assert "steps_left" in obs, (task_id, r)
                # submit the oracle answer
                s = rpc(ws, "step", data={"tool": "submit", "args": {"answer": ans}})
                so = s.get("observation", s)
                rew = so.get("reward")
                done = so.get("done")
            if rew == 1.0 and done:
                ok += 1
            else:
                bad += 1
                details.append((task_id, seed, "oracle", rew, done, so.get("output", "")[:80]))
            # a bare finish must score 0
            with connect(URL, max_size=None) as ws:
                rpc(ws, "reset", data={"seed": seed, "task_id": task_id})
                s = rpc(ws, "step", data={"tool": "submit", "args": {}})
                so = s.get("observation", s)
                if not (so.get("reward") == 0.0 and so.get("done")):
                    bad += 1
                    details.append((task_id, seed, "finish", so.get("reward"), so.get("done"), ""))
    print(f"oracle episodes correct: {ok}  problems: {bad}")
    for d in details[:40]:
        print("  FAIL", d)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
