"""Calibrate Finance Desk v3 terminal tasks on live Qwen3.8-27B (thinking off), arena-style, billed to DeepPPT.

Episode loop as AGENTS.md describes it: system = action schema; first user turn = reset observation as JSON;
each reply must be exactly ONE JSON object (an action, or {"finish": true}); any other reply ends the episode,
which is then scored through finish_action (evaluate). Budgets per task: completion_tokens 4096 covers the model's
tokens AND every observation after the first; context_tokens 8192 the whole conversation; tool_calls_total 32.
Drives the real v3 container over WebSocket. Usage: calibrate_v3.py OUT.jsonl N_EPISODES id1,id2,... [ws_urls]
"""
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

from websockets.sync.client import connect

TOK = os.environ["HUGGING_FACE_API_KEY"]
MODEL = "Qwen/Qwen3.8-27B:fastest"
COMPLETION_BUDGET, CONTEXT_BUDGET, CALLS = 4096, 8192, 32
SCHEMA = json.load(open(os.environ.get("SCHEMA_JSON", "schema_v3.json")))["action"]
SYSTEM = ("You are an agent acting in an environment. Each of your replies must be exactly one JSON object "
          "and nothing else: either an action matching this JSON schema, or {\"finish\": true} to end the episode.\n"
          "Action schema:\n" + json.dumps(SCHEMA))


def chat(messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 1.0, "reasoning_effort": "none"}
    req = urllib.request.Request("https://router.huggingface.co/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + TOK, "Content-Type": "application/json",
                                          "X-HF-Bill-To": "DeepPPT"})
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=180))
            return d["choices"][0]["message"].get("content") or "", d.get("usage", {})
        except Exception as e:
            if attempt == 4:
                return None, {"error": str(e)[:200]}
            time.sleep(3 * (attempt + 1))


def strict_json(text):
    t = re.sub(r"^\s*<think>\s*</think>\s*", "", text or "").strip()
    try:
        obj = json.loads(t)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def ntok(s):
    return max(1, int(len(s) / 3.3))


def rpc(ws, typ, **p):
    ws.send(json.dumps({"type": typ, **p}))
    m = json.loads(ws.recv(timeout=300))
    if m.get("type") == "error":
        raise RuntimeError(str(m)[:300])
    return m.get("data", m)


def episode(url, task_id, seed):
    log = {"task_id": task_id, "seed": seed, "calls": 0, "end": "", "reward": 0.0, "actions": []}
    with connect(url, max_size=None, open_timeout=20) as ws:
        first = rpc(ws, "reset", data={"seed": seed, "task_id": task_id})["observation"]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(first)}]
        used = 0  # completion budget: model tokens + observations after the first
        ctx = ntok(SYSTEM) + ntok(msgs[1]["content"])
        while True:
            room = min(COMPLETION_BUDGET - used, CONTEXT_BUDGET - ctx)
            if room < 32:
                log["end"] = "budget"
                break
            text, usage = chat(msgs, room)
            if text is None:
                log["end"] = "api_error"
                break
            ct = usage.get("completion_tokens", ntok(text))
            used += ct
            if "prompt_tokens" in usage:
                ctx = usage["prompt_tokens"] + ct
            else:
                ctx += ct
            act = strict_json(text)
            if act is None:
                log["end"] = "format"
                log["bad_reply"] = (text or "")[:300]
                break
            if act.get("finish") is True or (set(act) == {"action"} and isinstance(act["action"], dict) and False):
                log["end"] = "finish"
                break
            if set(act) == {"action"} and isinstance(act["action"], dict):
                act = act["action"]
            if log["calls"] >= CALLS:
                log["end"] = "calls"
                break
            log["calls"] += 1
            log["actions"].append((act.get("action_type"), (act.get("command") or act.get("file_path") or "")[:80]))
            try:
                d = rpc(ws, "step", data=act)
            except Exception as e:
                log["end"] = "bad_action"
                log["bad_reply"] = str(e)[:200]
                break
            obs, done = d["observation"], d.get("done")
            if done:
                log["end"] = "evaluate"
                log["reward"] = float(d.get("reward") or 0.0)
                return log
            otxt = json.dumps(obs)
            ot = ntok(otxt)
            if used + ot > COMPLETION_BUDGET or ctx + ot > CONTEXT_BUDGET:
                log["end"] = "budget_obs"
                break
            used += ot
            ctx += ot
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": otxt}]
        # scored through finish_action
        d = rpc(ws, "step", data={"action_type": "evaluate"})
        log["reward"] = float(d.get("reward") or 0.0)
        log["used"] = used
    return log


def main():
    out, n, ids = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
    urls = (sys.argv[4] if len(sys.argv) > 4 else "ws://localhost:8003/ws").split(",")
    jobs = [(tid, 1000 + k) for tid in ids for k in range(n)]
    random.Random(0).shuffle(jobs)
    lock = threading.Lock()
    workers = int(os.environ.get("WORKERS", "12"))

    def run(i_job):
        i, (tid, seed) = i_job
        try:
            r = episode(urls[i % len(urls)], tid, seed)
        except Exception as e:
            r = {"task_id": tid, "seed": seed, "end": "harness_error", "reward": 0.0, "err": str(e)[:200]}
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
        return r

    with cf.ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(run, enumerate(jobs)):
            pass


if __name__ == "__main__":
    main()
