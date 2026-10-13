"""Software engineering, industrial systems, natural science and office work families."""

import itertools
import json
import math
from datetime import date, datetime, timedelta

from .base import (COMMON_RULES, Family, Spec, dump, iso, pdate, q2, qn, read_csv, to_csv)


def _between(s, a, b):
    return s.split(a, 1)[1].split(b, 1)[0]


# ============================================================================= software engineering

class ChangeImpact(Family):
    name = "sw_impact"
    title = "Release impact: affected modules, build order, minimum-cost regression suites"
    domain = "software engineering"
    specs = {"affected": Spec(kind="ids"), "build_order": Spec(kind="seq"), "suites": Spec(kind="ids"),
             "suite_cost": Spec(tol=0.0)}

    def files(self, rng, level):
        n = rng.randint(7, 11)
        names = sorted(rng.sample(["auth", "billing", "cache", "catalog", "checkout", "config", "events", "gateway",
                                   "inventory", "ledger", "metrics", "notify", "orders", "payments", "search",
                                   "session", "shipping", "storage", "users"], n))
        order = names[:]
        rng.shuffle(order)
        deps = {m: sorted(rng.sample(order[:i], min(i, rng.randint(0, 3)))) for i, m in enumerate(order)}
        changed = sorted(rng.sample(order[: max(2, n // 2)], rng.randint(1, 2)))
        files = {"modules.json": dump(dict(sorted(deps.items()))), "change.json": dump({"changed": changed})}
        if level >= 2:
            suites = {}
            for i in range(rng.randint(5, 8)):
                suites[f"suite_{i:02d}"] = {"cost": rng.randint(2, 15), "covers": sorted(rng.sample(names, rng.randint(1, 4)))}
            suites["suite_full"] = {"cost": rng.randint(25, 45), "covers": names}
            files["test_suites.json"] = dump(suites)
        if level >= 3:
            cands = [(m, d) for m, ds in deps.items() for d in ds]
            if cands:
                m, d = rng.choice(cands)
                files["notice_release.txt"] = (f"Release note: in this release {m} drops its dependency on {d} "
                                               "(the import was removed). Use the updated dependency graph.\n")
        prompt = (
            "modules.json maps each module to the modules it depends on directly. Affected = every changed module "
            "plus every module that depends on an affected module, directly or transitively. build_order = the "
            "affected modules in dependency order (a module after the affected modules it depends on), always "
            "taking the alphabetically smallest module that is ready. "
            + ("suites = the set of test suites whose 'covers' together include every affected module with minimum "
               "total cost (ties: fewer suites, then the alphabetically smaller sorted list). " if level >= 2 else "")
            + 'Answer: {"affected": [modules], "build_order": [modules in order]'
            + (', "suites": [suite ids], "suite_cost": integer' if level >= 2 else "") + "}. " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        deps = {m: set(d) for m, d in json.loads(files["modules.json"]).items()}
        if "notice_release.txt" in files:
            t = files["notice_release.txt"]
            deps[_between(t, "release ", " drops")].discard(_between(t, "dependency on ", " (the"))
        aff = set(json.loads(files["change.json"])["changed"])
        grew = True
        while grew:
            new = {m for m, d in deps.items() if d & aff} - aff
            grew = bool(new)
            aff |= new
        order, done = [], set()
        while len(order) < len(aff):
            ready = sorted(m for m in aff - done if not ((deps[m] & aff) - done))
            order.append(ready[0])
            done.add(ready[0])
        out = {"affected": sorted(aff), "build_order": order}
        if level >= 2:
            suites = json.loads(files["test_suites.json"])
            best = None
            ids = sorted(suites)
            for r in range(1, len(ids) + 1):
                for combo in itertools.combinations(ids, r):
                    cov = set().union(*(suites[s]["covers"] for s in combo))
                    if aff <= cov:
                        key = (sum(suites[s]["cost"] for s in combo), r, list(combo))
                        if best is None or key < best:
                            best = key
            out["suites"], out["suite_cost"] = best[2], best[0]
        return out


def _ver(v):
    return tuple(int(x) for x in v.split("."))


def _ok(v, cons):
    for c in [c.strip() for c in cons.split(",") if c.strip()]:
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if c.startswith(op):
                x = _ver(c[len(op):].strip())
                a = _ver(v)
                if not {">=": a >= x, "<=": a <= x, "==": a == x, "!=": a != x, ">": a > x, "<": a < x}[op]:
                    return False
                break
    return True


class DependencyResolver(Family):
    name = "sw_resolver"
    title = "Package version resolution with conflicting constraints"
    domain = "software engineering"
    specs = {}

    def files(self, rng, level):
        k = 3 if level == 1 else rng.randint(4, 5)
        pkgs = sorted(rng.sample(["acorn", "bramble", "cobalt", "dynamo", "ember", "fjord", "gizmo", "halcyon"], k))
        index = {}
        for p in pkgs:
            vs = sorted({f"{rng.randint(1, 3)}.{rng.randint(0, 6)}.{rng.randint(0, 3)}" for _ in range(rng.randint(3, 5))}, key=_ver)
            index[p] = {}
            for v in vs:
                req = {}
                for q in rng.sample([x for x in pkgs if x != p], rng.randint(0, 2)):
                    m = rng.randint(1, 3)
                    req[q] = rng.choice([f">={m}.0.0", f"<{m + 1}.0.0", f">={m}.2.0,<{m + 1}.0.0", f"!={m}.0.0"])
                index[p][v] = req
        root = {p: rng.choice(["", f">={rng.randint(1, 2)}.0.0", f"<{rng.randint(2, 4)}.0.0"]) for p in pkgs}
        files = {"index.json": dump(index), "project_requirements.json": dump(root)}
        if level >= 3:
            p = rng.choice(pkgs)
            v = max(index[p], key=_ver)
            files["notice_security.txt"] = f"Security advisory: {p} {v} is yanked. It must never be selected.\n"
        prompt = (
            "Resolve versions for EVERY package in index.json (all are required). index.json gives, per package "
            "and version, constraints that version places on other packages; project_requirements.json gives the "
            "project's own constraints (empty = any). Constraints are comma-separated comparisons, all of which "
            "must hold; versions compare numerically by part. A resolution picks one version per package so that "
            "every constraint from the project and from each picked version holds. Choose the valid resolution "
            "that is greatest when compared package by package in alphabetical order (highest version of the "
            "first package, then the second, ...). "
            'Answer: an object mapping each package name to its version string, e.g. {"acorn": "2.1.0", ...}. '
            + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        index = json.loads(files["index.json"])
        root = json.loads(files["project_requirements.json"])
        yanked = set()
        if "notice_security.txt" in files:
            t = files["notice_security.txt"]
            yanked.add(tuple(_between(t, "advisory: ", " is yanked").split(" ")))
        pkgs = sorted(index)
        choices = [sorted((v for v in index[p] if (p, v) not in yanked), key=_ver, reverse=True) for p in pkgs]
        for combo in itertools.product(*choices):
            pick = dict(zip(pkgs, combo))
            if all(_ok(pick[p], root[p]) for p in pkgs) and all(
                    _ok(pick[q], c) for p in pkgs for q, c in index[p][pick[p]].items()):
                return pick
        return None


# ============================================================================= industrial and physical systems

class LineOee(Family):
    name = "ind_oee"
    title = "Overall equipment effectiveness from shift and downtime logs"
    domain = "industrial and physical systems"
    specs = {"best_line": Spec(kind="str"), "plant_oee_pct": Spec(tol=0.005), "worst_availability_line": Spec(kind="str")}

    def files(self, rng, level):
        lines = [f"L{i}" for i in range(1, rng.randint(3, 5) + 1)]
        shifts, downs = [], []
        n = 0
        for ln in lines:
            cycle = rng.choice([12, 15, 18, 20, 24, 30])
            for sh in ["A", "B", "C"][: rng.randint(2, 3)]:
                planned = 480
                dt = []
                for _ in range(rng.randint(1, 4)):
                    n += 1
                    reason = rng.choice(["jam", "changeover", "material shortage", "electrical fault"]
                                        + (["planned maintenance"] if level >= 2 else []))
                    dt.append([f"DT{n:03d}", ln, sh, rng.randint(5, 60), reason])
                run = planned - sum(d[3] for d in dt if d[4] != "planned maintenance") - sum(d[3] for d in dt if d[4] == "planned maintenance")
                total = int(run * 60 / cycle * rng.uniform(0.7, 0.97))
                shifts.append([ln, sh, planned, cycle, total, int(total * rng.uniform(0.005, 0.06))])
                downs += dt
        files = {"shifts.csv": to_csv(["line", "shift", "planned_minutes", "ideal_cycle_s", "total_count", "reject_count"], shifts),
                 "downtime.csv": to_csv(["event_id", "line", "shift", "minutes", "reason"], downs)}
        if level >= 3:
            d = rng.choice(downs)
            downs.append([f"DT{n + 1:03d}", d[1], d[2], d[3], d[4]])
            files["downtime.csv"] = to_csv(["event_id", "line", "shift", "minutes", "reason"], downs)
            files["notice_mes.txt"] = (f"MES note: event DT{n + 1:03d} is a duplicate upload of {d[0]}; "
                                       "count that stop only once.\n")
        prompt = (
            "Compute OEE per production line from shifts.csv and downtime.csv, aggregating a line's shifts. "
            + ("Planned maintenance is not a loss: subtract its minutes from planned time instead of counting it as "
               "downtime. " if level >= 2 else "")
            + "run time = planned - downtime. Availability = sum run / sum planned; Performance = sum(ideal_cycle_s * "
              "total_count) / sum(run time in seconds); Quality = sum(total - reject) / sum total; OEE = A*P*Q. "
              "The plant OEE uses the same sums over all lines. "
              'Answer: {"best_line": line with the highest OEE, "plant_oee_pct": percent rounded to 2 dp, '
              '"worst_availability_line": line with the lowest availability}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        downs = read_csv(files["downtime.csv"])
        if "notice_mes.txt" in files:
            dup = _between(files["notice_mes.txt"], "event ", " is a duplicate")
            downs = [d for d in downs if d["event_id"] != dup]
        agg = {}
        for s in read_csv(files["shifts.csv"]):
            key = (s["line"], s["shift"])
            pm = sum(int(d["minutes"]) for d in downs if (d["line"], d["shift"]) == key and d["reason"] == "planned maintenance")
            dt = sum(int(d["minutes"]) for d in downs if (d["line"], d["shift"]) == key and d["reason"] != "planned maintenance")
            planned = int(s["planned_minutes"]) - pm
            run = planned - dt
            a = agg.setdefault(s["line"], [0, 0, 0.0, 0, 0])
            a[0] += planned; a[1] += run; a[2] += int(s["ideal_cycle_s"]) * int(s["total_count"])
            a[3] += int(s["total_count"]); a[4] += int(s["total_count"]) - int(s["reject_count"])
        oee = lambda a: (a[1] / a[0]) * (a[2] / (a[1] * 60)) * (a[4] / a[3])
        lines = sorted(agg, key=lambda l: -oee(agg[l]))
        if oee(agg[lines[0]]) - oee(agg[lines[1]]) < 0.002:
            return None
        av = sorted(agg, key=lambda l: agg[l][1] / agg[l][0])
        if agg[av[1]][1] / agg[av[1]][0] - agg[av[0]][1] / agg[av[0]][0] < 0.002:
            return None
        tot = [sum(a[i] for a in agg.values()) for i in range(5)]
        return {"best_line": lines[0], "plant_oee_pct": qn(oee(tot) * 100, 2), "worst_availability_line": av[0]}


class TankSimulation(Family):
    name = "ind_tank"
    title = "Minute-step storage tank simulation with level-controlled pumps"
    domain = "industrial and physical systems"
    specs = {"final_level_m3": Spec(tol=0.01), "overflow_m3": Spec(tol=0.01), "unmet_demand_m3": Spec(tol=0.01)}

    def files(self, rng, level):
        cap = rng.randrange(40, 160)
        tank = {"capacity_m3": cap, "initial_m3": round(cap * rng.uniform(0.2, 0.8), 1), "minutes": rng.choice([120, 180, 240])}
        rows, t = [], 0
        while t < tank["minutes"]:
            d = rng.choice([15, 20, 30, 45])
            rows.append([t, min(t + d, tank["minutes"]), round(rng.uniform(0, 4.0), 2), round(rng.uniform(0, 3.5), 2)])
            t += d
        files = {"tank.json": dump(tank), "schedule.csv": to_csv(["from_min", "to_min", "inflow_m3_per_min", "demand_m3_per_min"], rows)}
        if level >= 2:
            files["pump.json"] = dump({"pump": "P2", "rate_m3_per_min": round(rng.uniform(1.0, 3.0), 2),
                                       "start_when_level_at_least": round(cap * rng.uniform(0.75, 0.9), 1),
                                       "stop_when_level_at_most": round(cap * rng.uniform(0.4, 0.6), 1), "initially_on": False})
        if level >= 3:
            a = rng.randrange(0, tank["minutes"] - 30)
            files["notice_operations.txt"] = (f"Operations log: the inflow feed was isolated from minute {a} to minute "
                                              f"{a + rng.choice([20, 30, 40])} (inflow 0 in [from, to)).\n")
        prompt = (
            "Simulate the tank minute by minute for minutes 0..minutes-1. Rows of schedule.csv cover [from_min, to_min). "
            "Each minute: level += inflow; "
            + ("then, if pump P2 is on, it removes min(rate, level) as transfer (not demand); " if level >= 2 else "")
            + "then demand is served from the tank up to the available level, any shortfall is unmet demand; then any "
              "level above capacity overflows and is lost. "
            + ("P2 switches at the end of a minute: off->on when level >= start_when_level_at_least, on->off when "
               "level <= stop_when_level_at_most. " if level >= 2 else "")
            + 'Answer: {"final_level_m3", "overflow_m3", "unmet_demand_m3"} rounded to 2 dp. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        tank = json.loads(files["tank.json"])
        sched = [(int(r["from_min"]), int(r["to_min"]), float(r["inflow_m3_per_min"]), float(r["demand_m3_per_min"]))
                 for r in read_csv(files["schedule.csv"])]
        iso_ = None
        if "notice_operations.txt" in files:
            t = files["notice_operations.txt"]
            iso_ = (int(_between(t, "from minute ", " to")), int(_between(t, "to minute ", " (")))
        pump = json.loads(files["pump.json"]) if "pump.json" in files else None
        on = pump["initially_on"] if pump else False
        lvl, over, unmet = tank["initial_m3"], 0.0, 0.0
        for m in range(tank["minutes"]):
            inflow, demand = next((i, d) for a, b, i, d in sched if a <= m < b)
            if iso_ and iso_[0] <= m < iso_[1]:
                inflow = 0.0
            lvl += inflow
            if pump and on:
                lvl -= min(pump["rate_m3_per_min"], lvl)
            served = min(demand, lvl)
            unmet += demand - served
            lvl -= served
            if lvl > tank["capacity_m3"]:
                over += lvl - tank["capacity_m3"]
                lvl = tank["capacity_m3"]
            if pump:
                if not on and lvl >= pump["start_when_level_at_least"]:
                    on = True
                elif on and lvl <= pump["stop_when_level_at_most"]:
                    on = False
        return {"final_level_m3": qn(lvl, 2), "overflow_m3": qn(over, 2), "unmet_demand_m3": qn(unmet, 2)}


# ============================================================================= natural science

def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return m, my - m * mx


class Kinetics(Family):
    name = "sci_kinetics"
    title = "First-order rate constant from a decay series"
    domain = "natural science"
    specs = {"k_per_min": Spec(tol=0.000011), "half_life_min": Spec(tol=0.011), "predicted_mM": Spec(tol=0.0011)}

    def files(self, rng, level):
        k, c0 = rng.uniform(0.01, 0.15), rng.uniform(2, 20)
        off = round(rng.uniform(0.05, 0.3), 3) if level >= 3 else 0.0
        rows = []
        for i, t in enumerate(sorted(rng.sample(range(0, 90, 3), rng.randint(7, 11)))):
            c = c0 * math.exp(-k * t) * (1 + rng.gauss(0, 0.02))
            flag = "ok"
            if level >= 2 and rng.random() < 0.2:
                c *= rng.choice([0.6, 1.5])
                flag = rng.choice(["bubble", "saturated"])
            rows.append([f"S{i + 1:02d}", t, round(c + off, 4), flag])
        t_pred = int(round(rng.uniform(1.0, 3.0) * math.log(2) / k))
        files = {"decay_run.csv": to_csv(["sample", "t_min", "conc_mM", "qc_flag"], rows),
                 "request.json": dump({"compound": rng.choice(["NB-17", "QX-4", "TR-202"]), "predict_at_min": t_pred})}
        if level >= 3:
            files["notice_instrument.txt"] = (f"Instrument note: the detector had a constant baseline offset of +{off} "
                                              "mM during this run. Subtract it from every reading before fitting.\n")
        prompt = (
            "Fit first-order decay C = C0*exp(-k t) to decay_run.csv by ordinary least squares of ln(C) on t "
            + ("using only rows with qc_flag 'ok' " if level >= 2 else "")
            + "(k = -slope, C0 = exp(intercept)). half-life = ln 2 / k. predicted_mM = C0*exp(-k*predict_at_min). "
              'Answer: {"k_per_min": rounded to 5 dp, "half_life_min": rounded to 2 dp, "predicted_mM": rounded to '
              "3 dp} (compute each from the unrounded fit). " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        rows = read_csv(files["decay_run.csv"])
        off = 0.0
        if "notice_instrument.txt" in files:
            off = float(_between(files["notice_instrument.txt"], "offset of +", " mM"))
        if level >= 2:
            rows = [r for r in rows if r["qc_flag"] == "ok"]
        pts = [(float(r["t_min"]), float(r["conc_mM"]) - off) for r in rows]
        if len(pts) < 4 or any(c <= 0 for _, c in pts):
            return None
        m, b = _ols([t for t, _ in pts], [math.log(c) for _, c in pts])
        k = -m
        if k <= 0:
            return None
        tp = json.loads(files["request.json"])["predict_at_min"]
        return {"k_per_min": qn(k, 5), "half_life_min": qn(math.log(2) / k, 2), "predicted_mM": qn(math.exp(b - k * tp), 3)}


class Calibration(Family):
    name = "sci_calibration"
    title = "Beer-Lambert calibration curve and diluted unknowns"
    domain = "natural science"
    specs = {"slope": Spec(tol=0.000011), "out_of_range": Spec(kind="ids")}

    def files(self, rng, level):
        eps, b0 = rng.uniform(0.004, 0.02), rng.uniform(0.0, 0.03)
        concs = [0, 10, 25, 50, 75, 100, 150][: rng.randint(5, 7)]
        std = []
        for c in concs:
            for _ in range(2 if level >= 2 else 1):
                std.append([c, round(b0 + eps * c * (1 + rng.gauss(0, 0.015)) + rng.gauss(0, 0.002), 4)])
        top = max(a for _, a in std)
        samples = []
        for i in range(rng.randint(3, 5)):
            a = round(rng.uniform(0.05, top * (1.25 if level >= 2 else 0.95)), 4)
            samples.append([f"U{i + 1}", a, rng.choice([1, 2, 5, 10])])
        files = {"standards.csv": to_csv(["conc_uM", "absorbance"], std),
                 "samples.csv": to_csv(["sample_id", "absorbance", "dilution_factor"], samples)}
        if level >= 3:
            s = rng.choice(samples)
            files["notice_lab.txt"] = (f"Lab note: sample {s[0]} was actually diluted {rng.choice([x for x in [2, 4, 20, 25] if x != s[2]])}x, "
                                       "not as recorded in samples.csv.\n")
        prompt = (
            "Fit absorbance = slope * conc_uM + intercept by ordinary least squares over all rows of standards.csv. "
            "Each sample's original concentration = (absorbance - intercept) / slope * dilution_factor. "
            + ("A sample whose absorbance is above the highest standard absorbance is out of range: list it and do "
               "not report a concentration for it. " if level >= 2 else "")
            + 'Answer: {"slope": rounded to 5 dp, one key per in-range sample id with its concentration in uM rounded '
              'to 2 dp' + (', "out_of_range": [sample ids]' if level >= 2 else "") + "}. " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        std = read_csv(files["standards.csv"])
        m, b = _ols([float(r["conc_uM"]) for r in std], [float(r["absorbance"]) for r in std])
        top = max(float(r["absorbance"]) for r in std)
        fix = {}
        if "notice_lab.txt" in files:
            t = files["notice_lab.txt"]
            fix[_between(t, "sample ", " was")] = float(_between(t, "diluted ", "x"))
        out = {"slope": qn(m, 5)}
        oor = []
        for r in read_csv(files["samples.csv"]):
            a = float(r["absorbance"])
            if level >= 2 and a > top:
                oor.append(r["sample_id"])
                continue
            out[r["sample_id"]] = q2((a - b) / m * fix.get(r["sample_id"], float(r["dilution_factor"])))
        if level >= 2:
            out["out_of_range"] = sorted(oor)
        return out


# ============================================================================= office and white-collar work

class MeetingScheduler(Family):
    name = "office_meeting"
    title = "Earliest common meeting slot across time zones"
    domain = "office and white-collar work"
    specs = {"start_utc": Spec(kind="str"), "room": Spec(kind="str")}

    def files(self, rng, level):
        base = rng.choice([-8, -5, -3, 0, 1, 3, 5, 8])
        names = rng.sample(["Ana", "Bo", "Chidi", "Devi", "Emre", "Farah", "Gus"], rng.randint(3, 4))
        people = [(n, base + rng.choice([0, 1, 2, 3, 4])) for n in names]
        day0 = date(2025, 10, 13)
        busy = []
        for name, tz in people:
            for d in range(3):
                for _ in range(rng.randint(2, 5)):
                    st = datetime(day0.year, day0.month, day0.day) + timedelta(days=d, hours=rng.randint(8, 16) - tz, minutes=rng.choice([0, 15, 30, 45]))
                    busy.append([name, st.strftime("%Y-%m-%dT%H:%M"), (st + timedelta(minutes=rng.choice([30, 45, 60, 90, 120]))).strftime("%Y-%m-%dT%H:%M")])
        rng.shuffle(busy)
        dur = rng.choice([30, 45, 60])
        req = {"duration_minutes": dur, "search_from_utc": "2025-10-13T00:00", "search_to_utc": "2025-10-16T00:00",
               "required": [p[0] for p in people]}
        files = {"people.csv": to_csv(["name", "utc_offset_hours", "work_start_local", "work_end_local"],
                                      [[n, tz, "09:00", "17:00"] for n, tz in people]),
                 "busy_utc.csv": to_csv(["name", "start", "end"], busy), "request.json": dump(req)}
        if level >= 2:
            rooms = []
            for r in ["Atlas", "Birch", "Cove"]:
                for _ in range(rng.randint(1, 4)):
                    st = datetime(2025, 10, 13) + timedelta(days=rng.randint(0, 2), hours=rng.randint(6, 18), minutes=rng.choice([0, 30]))
                    rooms.append([r, st.strftime("%Y-%m-%dT%H:%M"), (st + timedelta(minutes=rng.choice([60, 120, 180]))).strftime("%Y-%m-%dT%H:%M")])
            files["room_bookings_utc.csv"] = to_csv(["room", "start", "end"], rooms)
        if level >= 3:
            n, tz = rng.choice(people)
            files["notice_email.txt"] = (f"Email from {n}: I can't do anything before 13:00 my local time on "
                                         f"{iso(day0 + timedelta(days=rng.randint(0, 1)))}.\n")
        prompt = (
            "Find the earliest meeting start (UTC, on a :00/:15/:30/:45 boundary) in [search_from_utc, search_to_utc) "
            "such that the whole meeting lies inside every required person's working hours (local time = UTC + "
            "utc_offset_hours) and overlaps none of their busy intervals (intervals are [start, end); touching is "
            "fine). "
            + ("It also needs a room from Atlas, Birch, Cove with no overlapping booking; pick the first such room "
               "alphabetically. " if level >= 2 else "")
            + 'Answer: {"start_utc": "YYYY-MM-DDTHH:MM"' + (', "room": name' if level >= 2 else "") + "}. " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        req = json.loads(files["request.json"])
        ppl = {r["name"]: int(r["utc_offset_hours"]) for r in read_csv(files["people.csv"])}
        P = lambda s: datetime.fromisoformat(s)
        busy = [(r["name"], P(r["start"]), P(r["end"])) for r in read_csv(files["busy_utc.csv"])]
        rooms = [(r["room"], P(r["start"]), P(r["end"])) for r in read_csv(files.get("room_bookings_utc.csv", "room,start,end\n"))]
        late = None
        if "notice_email.txt" in files:
            t = files["notice_email.txt"]
            late = (_between(t, "Email from ", ":"), date.fromisoformat(_between(t, "local time on ", ".")))
        dur = timedelta(minutes=req["duration_minutes"])
        t, end = P(req["search_from_utc"]), P(req["search_to_utc"])
        while t + dur <= end:
            ok = True
            for n in req["required"]:
                ls, le = t + timedelta(hours=ppl[n]), t + dur + timedelta(hours=ppl[n])
                if ls.date() != (le - timedelta(minutes=1)).date() or ls.time() < datetime.strptime("09:00", "%H:%M").time() \
                        or (le - timedelta(seconds=1)).time() >= datetime.strptime("17:00", "%H:%M").time():
                    ok = False
                if late and n == late[0] and ls.date() == late[1] and ls.hour < 13:
                    ok = False
                if any(b[0] == n and b[1] < t + dur and t < b[2] for b in busy):
                    ok = False
                if not ok:
                    break
            room = None
            if ok and level >= 2:
                room = next((r for r in ["Atlas", "Birch", "Cove"] if not any(x[0] == r and x[1] < t + dur and t < x[2] for x in rooms)), None)
                ok = room is not None
            if ok:
                out = {"start_utc": t.strftime("%Y-%m-%dT%H:%M")}
                if level >= 2:
                    out["room"] = room
                return out
            t += timedelta(minutes=15)
        return None


class ExpenseAudit(Family):
    name = "office_expense"
    title = "Expense report audit against travel policy"
    domain = "office and white-collar work"
    specs = {"total_reimbursable": Spec(tol=0.01), "rejected_lines": Spec(kind="ids")}

    def files(self, rng, level):
        emps = rng.sample(["kim", "lopez", "mehta", "novak", "okafor"], rng.randint(2, 3))
        rows, n = [], 0
        for e in emps:
            for _ in range(rng.randint(4, 8)):
                n += 1
                d = date(2025, 9, rng.randint(1, 30))
                cat = rng.choice(["meals", "meals", "lodging", "airfare", "mileage", "alcohol", "taxi"] + (["gift"] if level >= 2 else []))
                miles = rng.randint(10, 300) if cat == "mileage" else ""
                amt = {"meals": rng.uniform(8, 70), "lodging": rng.uniform(120, 320), "airfare": rng.uniform(150, 900),
                       "mileage": rng.uniform(5, 150), "alcohol": rng.uniform(10, 60), "taxi": rng.uniform(12, 80),
                       "gift": rng.uniform(20, 120)}[cat]
                rows.append([f"X{n:03d}", e, iso(d), cat, round(amt, 2), miles, rng.choice(["Y", "Y", "Y", "N"])])
        pol = {"meals_cap_per_employee_per_day": rng.choice([60, 75, 90]), "lodging_cap_per_night": rng.choice([200, 250]),
               "mileage_rate_per_mile": 0.70, "non_reimbursable": ["alcohol"] + (["gift"] if level >= 2 else []),
               "receipt_required_at_or_above": 75}
        files = {"expense_lines.csv": to_csv(["line_id", "employee", "date", "category", "amount", "miles", "receipt"], rows),
                 "travel_policy.json": dump(pol)}
        if level >= 3:
            files["notice_policy_update.txt"] = (f"Policy update: for expenses dated 2025-09-{rng.randint(10, 25)} or later, the meals cap "
                                                 f"is {pol['meals_cap_per_employee_per_day'] - 25} per employee per day.\n")
        prompt = (
            "Audit September expenses against travel_policy.json, line by line, then apply caps. A line is "
            "rejected (pays 0) if its category is non-reimbursable, or if it has no receipt (receipt N) and its "
            "amount is at or above receipt_required_at_or_above. Mileage pays miles * mileage_rate_per_mile "
            "(the amount column is ignored and no receipt is needed). Lodging pays min(amount, cap). An employee's "
            "accepted meals on one date together pay at most the daily meals cap. Everything else pays its amount. "
            'Answer: {"total_reimbursable": dollars rounded to cents, "rejected_lines": [line ids]}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        pol = json.loads(files["travel_policy.json"])
        cut = None
        if "notice_policy_update.txt" in files:
            t = files["notice_policy_update.txt"]
            cut = (pdate(_between(t, "dated ", " or later")), float(_between(t, "meals cap is ", " per")))
        total, rejected, meals = 0.0, [], {}
        for r in read_csv(files["expense_lines.csv"]):
            amt, cat = float(r["amount"]), r["category"]
            if cat in pol["non_reimbursable"] or (cat != "mileage" and r["receipt"] == "N" and amt >= pol["receipt_required_at_or_above"]):
                rejected.append(r["line_id"])
                continue
            if cat == "mileage":
                total += int(r["miles"]) * pol["mileage_rate_per_mile"]
            elif cat == "lodging":
                total += min(amt, pol["lodging_cap_per_night"])
            elif cat == "meals":
                meals[(r["employee"], r["date"])] = meals.get((r["employee"], r["date"]), 0) + amt
            else:
                total += amt
        for (e, d), v in meals.items():
            cap = pol["meals_cap_per_employee_per_day"]
            if cut and pdate(d) >= cut[0]:
                cap = cut[1]
            total += min(v, cap)
        return {"total_reimbursable": q2(total), "rejected_lines": sorted(rejected)}


FAMILIES = [ChangeImpact(), DependencyResolver(), LineOee(), TankSimulation(), Kinetics(), Calibration(),
            MeetingScheduler(), ExpenseAudit()]
