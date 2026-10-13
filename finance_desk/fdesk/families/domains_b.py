"""Math and formal reasoning, cybersecurity, and media production families."""

import ipaddress
import json
from datetime import datetime, timedelta

from .base import (COMMON_RULES, Family, Spec, dump, q2, qn, read_csv, to_csv)


def _between(s, a, b):
    return s.split(a, 1)[1].split(b, 1)[0]


# ============================================================================= math and formal reasoning

class ConstrainedRoute(Family):
    name = "math_route"
    title = "Shortest route under a toll budget"
    domain = "math and formal reasoning"
    specs = {"minutes": Spec(tol=0.0), "route": Spec(kind="seq"), "tolls": Spec(tol=0.0)}

    def files(self, rng, level):
        n = rng.randint(7, 10)
        nodes = [chr(65 + i) for i in range(n)]
        edges = {}
        for i in range(1, n):  # spanning tree, then extra roads
            j = rng.randrange(i)
            edges[tuple(sorted((nodes[i], nodes[j])))] = None
        while len(edges) < n + rng.randint(3, 7):
            a, b = rng.sample(nodes, 2)
            edges[tuple(sorted((a, b)))] = None
        rows = [[a, b, rng.randint(5, 60), rng.choice([0, 0, 2, 3, 5, 8])] for a, b in edges]
        src, dst = rng.sample(nodes, 2)
        q = {"from": src, "to": dst}
        if level >= 2:
            q["toll_budget"] = rng.choice([0, 2, 3, 5, 6])
        files = {"roads.csv": to_csv(["a", "b", "minutes", "toll"], rows), "trip.json": dump(q)}
        if level >= 3:
            base = self.solve(files, level)
            if base and len(base["route"]) > 1:
                i = rng.randrange(len(base["route"]) - 1)
                a, b = sorted(base["route"][i:i + 2])
                files["notice_traffic.txt"] = f"Traffic alert: the road between {a} and {b} is closed for the whole trip.\n"
        prompt = (
            "Roads in roads.csv are two-way. Find the route from trip 'from' to 'to' with the least total minutes"
            + (" among routes whose total toll is at most toll_budget" if level >= 2 else "")
            + ". Ties: fewer roads, then the alphabetically smaller node sequence. "
            'Answer: {"minutes": integer, "route": [nodes from start to end], "tolls": integer total toll}. '
            + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        trip = json.loads(files["trip.json"])
        closed = None
        if "notice_traffic.txt" in files:
            t = files["notice_traffic.txt"]
            closed = tuple(sorted((_between(t, "between ", " and"), _between(t, " and ", " is closed"))))
        adj = {}
        for r in read_csv(files["roads.csv"]):
            if closed and tuple(sorted((r["a"], r["b"]))) == closed:
                continue
            for x, y in ((r["a"], r["b"]), (r["b"], r["a"])):
                adj.setdefault(x, []).append((y, int(r["minutes"]), int(r["toll"])))
        budget = trip.get("toll_budget", 10 ** 9)
        best = None
        stack = [([trip["from"]], 0, 0)]
        while stack:  # exhaustive over simple paths; graphs have at most 10 nodes
            path, m, toll = stack.pop()
            if path[-1] == trip["to"]:
                key = (m, len(path), path)
                if best is None or key < best[0]:
                    best = (key, toll)
                continue
            for y, mm, tt in adj.get(path[-1], []):
                if y not in path and toll + tt <= budget:
                    stack.append((path + [y], m + mm, toll + tt))
        if best is None:
            return None
        return {"minutes": best[0][0], "route": best[0][2], "tolls": best[1]}


class BoundedChange(Family):
    name = "math_coins"
    title = "Counting bounded coin combinations and the fewest-coins solution"
    domain = "math and formal reasoning"
    specs = {"ways": Spec(tol=0.0), "fewest_coins": Spec(tol=0.0)}

    def files(self, rng, level):
        k = rng.randint(3, 5)
        den = sorted(rng.sample([1, 2, 3, 5, 7, 10, 12, 20, 25, 50], k))
        lim = {str(d): rng.randint(2, 12) for d in den} if level >= 2 else {str(d): None for d in den}
        target = rng.randint(40, 160)
        files = {"coins.json": dump({"denominations": den, "max_count": lim, "target": target})}
        if level >= 3:
            d = rng.choice(den)
            files["notice_mint.txt"] = f"Mint notice: the {d}-unit coin is withdrawn and may not be used.\n"
        prompt = (
            "coins.json lists coin denominations"
            + (" and the maximum count usable of each" if level >= 2 else " (unlimited supply; max_count null)")
            + ". ways = the number of distinct multisets of coins summing exactly to target (order does not "
              "matter). fewest_coins = the smallest number of coins summing to target, or -1 if impossible. "
              'Answer: {"ways": integer, "fewest_coins": integer}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        c = json.loads(files["coins.json"])
        den = list(c["denominations"])
        if "notice_mint.txt" in files:
            den.remove(int(_between(files["notice_mint.txt"], "the ", "-unit")))
        T = c["target"]
        ways = [1] + [0] * T
        few = [0] + [10 ** 9] * T
        for d in den:
            lim = c["max_count"][str(d)] or T // d
            nw, nf = [0] * (T + 1), [10 ** 9] * (T + 1)
            for s in range(T + 1):
                for j in range(0, lim + 1):
                    if s - j * d < 0:
                        break
                    nw[s] += ways[s - j * d]
                    nf[s] = min(nf[s], few[s - j * d] + j)
            ways, few = nw, nf
        if ways[T] == 0:
            return None
        return {"ways": ways[T], "fewest_coins": few[T]}


# ============================================================================= cybersecurity

class FirewallAudit(Family):
    name = "sec_firewall"
    title = "First-match firewall evaluation over CIDR and port ranges"
    domain = "cybersecurity"
    specs = {"allowed": Spec(kind="ids"), "denied_count": Spec(tol=0.0)}

    def _ip(self, rng, nets):
        net = ipaddress.ip_network(rng.choice(nets))
        return str(net[rng.randrange(1, min(net.num_addresses - 1, 4000))])

    def files(self, rng, level):
        nets = ["10.0.0.0/8", "10.20.0.0/16", "10.20.5.0/24", "192.168.1.0/24", "172.16.0.0/12", "203.0.113.0/24", "198.51.100.0/24"]
        rules = []
        for i in range(rng.randint(5, 8)):
            proto = rng.choice(["tcp", "udp", "any"])
            port = rng.choice(["22", "80", "443", "53", "8000-8100", "any", "1024-65535"])
            rules.append(f"{rng.choice(['allow', 'allow', 'deny'])} {proto} from {rng.choice(nets + ['any'])} to "
                         f"{rng.choice(nets + ['any'])} port {port}")
        rules.append("deny any from any to any port any")
        conns = []
        for i in range(rng.randint(8, 12)):
            conns.append([f"c{i + 1:02d}", rng.choice(["tcp", "udp"]), self._ip(rng, nets), self._ip(rng, nets),
                          rng.choice([22, 80, 443, 53, 8080, 8001, 3306, 5000, 65000])])
        files = {"rules.txt": "# evaluated top to bottom; the first matching rule decides\n" + "\n".join(rules) + "\n",
                 "connections.csv": to_csv(["conn_id", "proto", "src", "dst", "dst_port"], conns)}
        if level >= 2:
            files["rules.txt"] = files["rules.txt"].replace(f"\n{rules[1]}\n", f"\n# {rules[1]}  (disabled)\n")
        if level >= 3:
            base = self.solve(files, level)
            if base and base["allowed"]:
                cid = rng.choice(base["allowed"])
                ip = next(c[2] for c in conns if c[0] == cid)
                src = rng.choice([n for n in nets if ipaddress.ip_address(ip) in ipaddress.ip_network(n)])
                files["notice_incident.txt"] = (f"Incident response: an emergency rule was pushed above all others: "
                                                f"'deny any from {src} to any port any'.\n")
        prompt = (
            "Evaluate each connection in connections.csv against rules.txt. A rule matches when the protocol "
            "matches (any = both), src and dst addresses are inside the rule's networks (any = all), and dst_port "
            "is the rule's port or inside its inclusive range (any = all). The first matching rule decides. "
            + ("Lines starting with # are comments. " if level >= 2 else "")
            + 'Answer: {"allowed": [conn ids], "denied_count": integer}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        rules = []
        for ln in files["rules.txt"].splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            rules.append((p[0], p[1], p[3], p[5], p[7]))
        if "notice_incident.txt" in files:
            p = _between(files["notice_incident.txt"], "'", "'").split()
            rules.insert(0, (p[0], p[1], p[3], p[5], p[7]))
        inside = lambda ip, net: net == "any" or ipaddress.ip_address(ip) in ipaddress.ip_network(net)

        def port_ok(port, spec):
            if spec == "any":
                return True
            if "-" in spec:
                a, b = spec.split("-")
                return int(a) <= port <= int(b)
            return port == int(spec)

        allowed, denied = [], 0
        for c in read_csv(files["connections.csv"]):
            for act, proto, s, d, port in rules:
                if proto in ("any", c["proto"]) and inside(c["src"], s) and inside(c["dst"], d) and port_ok(int(c["dst_port"]), port):
                    if act == "allow":
                        allowed.append(c["conn_id"])
                    else:
                        denied += 1
                    break
        if level < 3 and len(allowed) < 2:
            return None
        return {"allowed": sorted(allowed), "denied_count": denied}


class AuthForensics(Family):
    name = "sec_auth"
    title = "SSH brute-force detection and compromised accounts from auth logs"
    domain = "cybersecurity"
    specs = {"attacker_ips": Spec(kind="ids"), "compromised_accounts": Spec(kind="ids")}

    def files(self, rng, level):
        users = ["alice", "bob", "carol", "dave", "erin", "frank", "root", "deploy"]
        t0 = datetime(2025, 10, 12, 1, 0, 0)
        ev = []
        ips = [f"198.51.100.{rng.randint(2, 250)}" for _ in range(3)] + [f"203.0.113.{rng.randint(2, 250)}" for _ in range(3)]
        for ip in ips:
            if rng.random() < 0.6:  # burst
                start = t0 + timedelta(minutes=rng.randint(0, 200))
                n = rng.randint(3, 9)
                tgt = rng.choice(users)
                for k in range(n):
                    kind = "Failed password" if (level < 2 or rng.random() < 0.7) else "Invalid user"
                    ev.append((start + timedelta(seconds=k * rng.randint(20, 150)), kind, rng.choice([tgt, tgt, rng.choice(users)]), ip))
                if rng.random() < 0.5:
                    ev.append((start + timedelta(seconds=n * 160 + rng.randint(1, 400)), "Accepted password", tgt, ip))
            else:
                for _ in range(rng.randint(1, 3)):
                    ev.append((t0 + timedelta(minutes=rng.randint(0, 300)), rng.choice(["Failed password", "Accepted password"]), rng.choice(users), ip))
        for _ in range(rng.randint(3, 6)):
            ev.append((t0 + timedelta(minutes=rng.randint(0, 300)), "Accepted publickey", rng.choice(users), f"10.0.{rng.randint(0, 9)}.{rng.randint(2, 99)}"))
        ev.sort()
        lines = []
        for t, kind, u, ip in ev:
            who = f"invalid user {u}" if kind == "Invalid user" else u
            verb = "Failed password for" if kind == "Invalid user" else kind + " for"
            lines.append(f"{t:%Y-%m-%dT%H:%M:%S}Z bastion sshd[{rng.randint(1000, 9999)}]: {verb} {who} from {ip} port {rng.randint(30000, 65000)} ssh2")
        policy = {"failures_threshold": rng.choice([3, 4, 5]), "window_minutes": rng.choice([5, 10, 15])}
        files = {"auth.log": "\n".join(lines) + "\n", "detection_policy.json": dump(policy)}
        if level >= 3:
            files["notice_soc.txt"] = (f"SOC note: {rng.choice(ips)} is our authorised external vulnerability scanner. "
                                       "Exclude it from all findings.\n")
        prompt = (
            "Analyse auth.log with detection_policy.json. Every 'Failed password for ...' line is a failure "
            "(including 'invalid user'). An IP is an attacker if it has at least failures_threshold failures inside "
            "some window of window_minutes (inclusive of both ends). An account is compromised if an attacker IP "
            "logs in to it with 'Accepted password' within window_minutes after that IP's last failure before the "
            "login, and that IP had at least failures_threshold failures in the window_minutes before the login. "
            'Answer: {"attacker_ips": [ips], "compromised_accounts": [usernames]}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        pol = json.loads(files["detection_policy.json"])
        N, W = pol["failures_threshold"], timedelta(minutes=pol["window_minutes"])
        skip = set()
        if "notice_soc.txt" in files:
            skip.add(_between(files["notice_soc.txt"], "SOC note: ", " is our"))
        fails, accepts = {}, []
        for ln in files["auth.log"].splitlines():
            t = datetime.strptime(ln[:19], "%Y-%m-%dT%H:%M:%S")
            ip = _between(ln, " from ", " port")
            if ip in skip:
                continue
            if "Failed password for" in ln:
                fails.setdefault(ip, []).append(t)
            elif "Accepted password for" in ln:
                accepts.append((t, _between(ln, "Accepted password for ", " from"), ip))
        attackers = sorted(ip for ip, ts in fails.items() if any(sum(1 for u in ts if s <= u <= s + W) >= N for s in ts))
        comp = set()
        for t, user, ip in accepts:
            if ip not in attackers:
                continue
            before = [u for u in fails[ip] if u <= t]
            if before and t - max(before) <= W and sum(1 for u in before if t - W <= u) >= N:
                comp.add(user)
        if not attackers:
            return None
        return {"attacker_ips": attackers, "compromised_accounts": sorted(comp)}


# ============================================================================= media and content production

def _srt_t(ms):
    h, r = divmod(int(round(ms)), 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _srt_ms(s):
    hms, ms = s.strip().split(",")
    h, m, sec = hms.split(":")
    return (int(h) * 3600 + int(m) * 60 + int(sec)) * 1000 + int(ms)


WORDS = ("we never planned to stay this long but the river kept rising and the road out washed away "
         "so we built the radio from parts and waited for anyone listening on the other side").split()


class SubtitleQc(Family):
    name = "media_srt"
    title = "Subtitle retiming and reading-speed QC"
    domain = "media and content production"
    specs = {"cue_start": Spec(kind="str"), "fast_cues": Spec(kind="ids"), "total_display_s": Spec(tol=0.0011)}

    def files(self, rng, level):
        cues, t = [], rng.randint(1000, 20000)
        for i in range(rng.randint(8, 14)):
            dur = rng.randint(700, 4500)
            text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 12)))
            if len(text) > 40:
                cut = text.rfind(" ", 0, len(text) // 2 + 5)
                text = text[:cut] + "\n" + text[cut + 1:]
            end = t + dur
            if level >= 2 and rng.random() < 0.25:
                end += rng.randint(100, 600)  # overlaps the next cue
            cues.append((i + 1, t, end, text))
            t += dur + rng.randint(80, 900)
        srt = "\n".join(f"{n}\n{_srt_t(a)} --> {_srt_t(b)}\n{tx}\n" for n, a, b, tx in cues)
        req = {"offset_ms": rng.choice([-1500, -800, -250, 400, 1200, 2500]), "max_chars_per_second": 17,
               "report_cue": rng.randint(2, len(cues))}
        files = {"episode.srt": srt, "qc_request.json": dump(req)}
        if level >= 3:
            files["notice_post.txt"] = ("Post-production note: this file was timed against the 25 fps master; the "
                                        "delivery is 23.976 fps. Before applying offset_ms, multiply every timestamp by "
                                        "25/23.976 (keep fractional milliseconds until the final rounding).\n")
        prompt = (
            "QC episode.srt with qc_request.json. Shift every timestamp by offset_ms. "
            + ("If a cue ends after the next cue starts, set its end to the next cue's start. " if level >= 2 else "")
            + "Duration = end - start. Reading speed = characters of the cue text excluding line breaks (spaces "
              "count) / duration in seconds; a cue is fast if its speed is above max_chars_per_second. "
              'Answer: {"cue_start": start of cue number report_cue as HH:MM:SS,mmm (rounded to the ms), '
              '"fast_cues": [cue numbers], "total_display_s": sum of durations in seconds rounded to 3 dp}. '
              + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        req = json.loads(files["qc_request.json"])
        scale = 25 / 23.976 if "notice_post.txt" in files else 1.0
        cues = []
        for block in files["episode.srt"].strip().split("\n\n"):
            ls = block.split("\n")
            a, b = ls[1].split(" --> ")
            cues.append([int(ls[0]), _srt_ms(a) * scale + req["offset_ms"], _srt_ms(b) * scale + req["offset_ms"], "".join(ls[2:])])
        if level >= 2:
            for i in range(len(cues) - 1):
                cues[i][2] = min(cues[i][2], cues[i + 1][1])
        if any(c[1] < 0 for c in cues):
            return None
        fast, total = [], 0.0
        for n, a, b, tx in cues:
            d = (b - a) / 1000
            total += d
            sp = len(tx) / d
            if abs(sp - req["max_chars_per_second"]) < 0.05:
                return None
            if sp > req["max_chars_per_second"]:
                fast.append(str(n))
        start = next(c[1] for c in cues if c[0] == req["report_cue"])
        return {"cue_start": _srt_t(start), "fast_cues": fast, "total_display_s": qn(total, 3)}


def _tc_frames(tc, fps):
    h, m, s, f = (int(x) for x in tc.split(":"))
    return ((h * 60 + m) * 60 + s) * fps + f


def _frames_tc(n, fps):
    s, f = divmod(n, fps)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


class EditDecisionList(Family):
    name = "media_edl"
    title = "Edit decision list: record timecodes with dissolves"
    domain = "media and content production"
    specs = {"program_duration": Spec(kind="str"), "record_in": Spec(kind="str")}

    def files(self, rng, level):
        fps = rng.choice([24, 25, 30])
        rows = []
        for i in range(rng.randint(6, 10)):
            a = rng.randint(0, 3600 * fps)
            ln = rng.randint(2 * fps, 40 * fps)
            tr = "cut" if level == 1 or rng.random() < 0.6 else f"dissolve {rng.choice([8, 12, 16, 24])}"
            rows.append([f"{i + 1:03d}", f"A{rng.randint(1, 40):03d}", _frames_tc(a, fps), _frames_tc(a + ln, fps), "cut" if i == 0 else tr])
        meta = {"fps": fps, "record_start": "01:00:00:00", "report_event": rows[rng.randint(2, len(rows) - 1)][0]}
        files = {"edl.csv": to_csv(["event", "reel", "source_in", "source_out", "transition_in"], rows), "sequence.json": dump(meta)}
        if level >= 3:
            ev = rng.choice(rows[1:])
            if ev[0] != meta["report_event"] and rng.random() < 0.5:
                files["notice_editor.txt"] = f"Editor's note: event {ev[0]} was cut from the final version; remove it (its transition goes with it).\n"
            else:
                files["notice_editor.txt"] = (f"Editor's note: event {ev[0]} was trimmed; its source_out is now "
                                              f"{_frames_tc(_tc_frames(ev[2], fps) + fps * rng.randint(1, 2), fps)}.\n")
        prompt = (
            "Lay the events of edl.csv end to end on the record timeline starting at record_start (non-drop-frame "
            "timecode HH:MM:SS:FF at sequence fps). A clip lasts source_out - source_in frames (out is exclusive). "
            + ("transition_in 'dissolve N' means the event starts N frames before the previous event ends "
               "(they overlap by N frames). " if level >= 2 else "")
            + 'Answer: {"program_duration": total length as HH:MM:SS:FF, "record_in": record timecode where '
              "report_event starts}. " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        meta = json.loads(files["sequence.json"])
        fps = meta["fps"]
        rows = read_csv(files["edl.csv"])
        n = files.get("notice_editor.txt", "")
        if "was cut" in n:
            rows = [r for r in rows if r["event"] != _between(n, "event ", " was cut")]
        elif "trimmed" in n:
            for r in rows:
                if r["event"] == _between(n, "event ", " was trimmed"):
                    r["source_out"] = _between(n, "source_out is now ", ".")
        pos, start0, rec_in, end = 0, _tc_frames(meta["record_start"], fps), None, 0
        for i, r in enumerate(rows):
            ln = _tc_frames(r["source_out"], fps) - _tc_frames(r["source_in"], fps)
            ov = int(r["transition_in"].split()[1]) if r["transition_in"].startswith("dissolve") and i > 0 else 0
            if ov >= ln:
                return None
            pos = end - ov
            if r["event"] == meta["report_event"]:
                rec_in = start0 + pos
            end = pos + ln
        if rec_in is None:
            return None
        return {"program_duration": _frames_tc(end, fps), "record_in": _frames_tc(rec_in, fps)}


FAMILIES = [ConstrainedRoute(), BoundedChange(), FirewallAudit(), AuthForensics(), SubtitleQc(), EditDecisionList()]
