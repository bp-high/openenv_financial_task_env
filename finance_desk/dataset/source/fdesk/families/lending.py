"""Lending and fixed income: loan amortization, bond pricing, discounted cash flows."""

import json
import math
import random
from datetime import date, timedelta

from .base import (COMMON_RULES, Family, Spec, dump, iso, pdate, q2, qn, read_csv, to_csv)


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, min(d.day, 28))


class LoanAmortization(Family):
    name = "loan_amort"
    title = "Loan amortization with prepayments and a rate reset"
    specs = {"balance": Spec(tol=0.05), "total_interest": Spec(tol=0.05)}

    def files(self, rng, level):
        principal = rng.randrange(80, 900) * 1000
        rate = rng.choice([3.25, 3.875, 4.5, 5.125, 5.75, 6.25, 6.875, 7.4, 8.1])
        term = rng.choice([60, 84, 120, 180, 240, 360])
        target = rng.randint(14, min(term - 12, 72))
        files = {"loan.json": dump({"loan_id": f"LN-{rng.randrange(10000, 99999)}", "principal": principal,
                                    "annual_rate_pct": rate, "term_months": term, "first_payment": "2025-02-01",
                                    "report_after_payment": target})}
        if level >= 2:
            rows = []
            for m in sorted(rng.sample(range(2, target + 1), rng.randint(2, 4))):
                rows.append([iso(add_months(date(2025, 2, 1), m - 1)), m, rng.randrange(1, 20) * 500])
            # a prepayment after the reporting point, which must not be counted
            rows.append([iso(add_months(date(2025, 2, 1), target + 1)), target + 2, rng.randrange(1, 20) * 500])
            files["prepayments.csv"] = to_csv(["date", "payment_number", "extra_principal"], rows)
        if level >= 3:
            n = rng.randint(4, target - 2)
            new_rate = round(rate + rng.choice([-1.5, -1.0, -0.75, 0.5, 1.0, 1.25, 2.0]), 3)
            files["notice_rate_reset.txt"] = (
                f"Servicer notice: the loan is variable-rate. Starting with payment number {n} the annual rate is "
                f"{new_rate}%. At that payment, the scheduled payment is recomputed with the same annuity formula "
                f"on the balance outstanding just before payment {n}, at the new rate, over the remaining "
                f"term (term_months - {n} + 1 payments), rounded half-up to cents.\n")
        prompt = (
            "Fixed-payment monthly loan in loan.json. Monthly rate r = annual_rate_pct / 100 / 12. Scheduled payment = "
            "P*r/(1-(1+r)^-n) on principal P over term_months n, rounded half-up to cents. Each payment: interest = "
            "balance*r rounded half-up to cents; principal part = payment - interest; balance decreases by the "
            "principal part."
            + (" Extra principal in prepayments.csv is applied right after the scheduled payment with the same "
               "payment_number; it does not change the scheduled payment amount." if level >= 2 else "")
            + " Report the balance right after payment number report_after_payment (including any extra principal "
              "of that payment) and the total interest paid over payments 1..report_after_payment. "
            'Answer: {"balance": number, "total_interest": number} in dollars. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        loan = json.loads(files["loan.json"])
        bal, r, n = float(loan["principal"]), loan["annual_rate_pct"] / 1200, loan["term_months"]
        target = loan["report_after_payment"]
        pay = _annuity(bal, r, n)
        extra = {}
        if "prepayments.csv" in files:
            for row in read_csv(files["prepayments.csv"]):
                k = int(row["payment_number"])
                extra[k] = extra.get(k, 0) + float(row["extra_principal"])
        reset = None
        if "notice_rate_reset.txt" in files:
            t = files["notice_rate_reset.txt"]
            reset = (int(_between(t, "Starting with payment number ", " ")), float(_between(t, "the annual rate is ", "%")) / 1200)
        total_int = 0.0
        for k in range(1, target + 1):
            if reset and k == reset[0]:
                r = reset[1]
                pay = _annuity(bal, r, n - k + 1)
            interest = q2(bal * r)
            total_int = q2(total_int + interest)
            bal = q2(bal - (pay - interest))
            bal = q2(bal - extra.get(k, 0))
            if bal <= 0:
                return None
        return {"balance": bal, "total_interest": total_int}


def _annuity(p, r, n):
    return q2(p * r / (1 - (1 + r) ** -n)) if r else q2(p / n)


def _between(s, a, b):
    return s.split(a, 1)[1].split(b, 1)[0]


class BondPricing(Family):
    name = "bond_price"
    title = "Bond dirty/clean price and accrued interest between coupon dates"
    specs = {"dirty_price": Spec(tol=0.002), "clean_price": Spec(tol=0.002), "accrued_interest": Spec(tol=0.01)}

    def files(self, rng, level):
        freq = rng.choice([1, 2]) if level >= 2 else 2
        coupon = rng.choice([2.5, 3.0, 3.75, 4.25, 5.0, 5.5, 6.125, 7.0])
        mat = date(rng.randint(2027, 2040), rng.choice([3, 6, 9, 12]) if freq == 2 else rng.randint(1, 12), 15)
        settle = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 360))
        y = round(coupon + rng.uniform(-2.0, 2.5), 3)
        face = rng.choice([10, 25, 50, 100, 250]) * 1000
        bond = {"isin": f"XS{rng.randrange(10**9, 10**10)}", "coupon_pct": coupon, "coupons_per_year": freq,
                "maturity": iso(mat), "settlement": iso(settle), "yield_pct": y, "face_holding": face}
        files = {"bond.json": dump(bond)}
        if level >= 3:
            shift = rng.choice([1, 2, 3])
            files["notice_trade_correction.txt"] = (
                f"Trade correction: the settlement date in bond.json is wrong. The trade settles {shift} calendar "
                f"day(s) later than stated, and the agreed yield is {round(y + rng.choice([-0.35, -0.2, 0.15, 0.3]), 3)}%."
                " Use the corrected settlement date and yield.\n")
        prompt = (
            "Price the bond holding in bond.json. Coupon dates step back from maturity by 12/coupons_per_year months "
            "(same day of month). Let prev and next be the coupon dates with prev <= settlement < next. w = actual days(settlement, next) / actual "
            "days(prev, next). Per 100 face: c = coupon_pct/coupons_per_year, y = yield_pct/100/coupons_per_year, "
            "N = number of coupons from next to maturity inclusive. dirty = sum_{k=1..N} c/(1+y)^(k-1+w) + "
            "100/(1+y)^(N-1+w). accrued per 100 = c*(1-w). clean = dirty - accrued per 100. "
            "accrued_interest = accrued per 100 * face_holding / 100, in dollars. "
            'Answer: {"dirty_price": number per 100, rounded to 4 dp, "clean_price": number per 100, rounded to 4 dp, '
            '"accrued_interest": dollars rounded to cents}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        b = json.loads(files["bond.json"])
        settle, y_pct = pdate(b["settlement"]), b["yield_pct"]
        if "notice_trade_correction.txt" in files:
            t = files["notice_trade_correction.txt"]
            settle += timedelta(days=int(_between(t, "settles ", " calendar")))
            y_pct = float(_between(t, "agreed yield is ", "%"))
        f, mat = b["coupons_per_year"], pdate(b["maturity"])
        step = 12 // f
        dates = [mat]
        while dates[-1] > settle:
            dates.append(add_months_exact(dates[-1], -step))
        prev = dates[-1]
        coupons_after = [d for d in dates if d > settle]
        nxt = coupons_after[-1]
        n = len(coupons_after)
        w = (nxt - settle).days / (nxt - prev).days
        c, y = b["coupon_pct"] / f, y_pct / 100 / f
        dirty = sum(c / (1 + y) ** (k - 1 + w) for k in range(1, n + 1)) + 100 / (1 + y) ** (n - 1 + w)
        acc = c * (1 - w)
        return {"dirty_price": qn(dirty, 4), "clean_price": qn(dirty - acc, 4),
                "accrued_interest": q2(acc * b["face_holding"] / 100)}


def add_months_exact(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, d.day)


class CashFlowValuation(Family):
    name = "dcf_npv"
    title = "NPV / XNPV / IRR project selection"
    specs = {"best_project": Spec(kind="str"), "best_value": Spec(tol=0.01), "irr_pct": Spec(tol=0.01)}

    def files(self, rng, level):
        projects = ["ALPHA", "BRAVO", "CEDAR", "DELTA"][: rng.randint(2, 4) if level > 1 else 3]
        rate = rng.choice([6.0, 7.5, 8.0, 9.25, 10.0, 11.5, 12.0])
        rows = []
        start = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 200))
        for p in projects:
            outlay = -rng.randrange(200, 2000) * 1000
            years = rng.randint(4, 8)
            if level == 1:
                rows.append([p, 0, outlay])
                for t in range(1, years + 1):
                    rows.append([p, t, rng.randrange(40, 600) * 1000])
            else:
                rows.append([p, iso(start), outlay])
                d = start
                for _ in range(years + rng.randint(0, 4)):
                    d = d + timedelta(days=rng.randrange(120, 420))
                    amt = rng.randrange(-80, 600) * 1000
                    rows.append([p, iso(d), amt or 1000])
        rng.shuffle(rows)
        if level == 1:
            files = {"cashflows.csv": to_csv(["project", "year", "amount"], rows)}
        else:
            files = {"cashflows.csv": to_csv(["project", "date", "amount"], rows)}
        files["policy.json"] = dump({"discount_rate_pct": rate})
        if level >= 3:
            p = rng.choice(projects)
            mine = [r for r in rows if r[0] == p and r[2] > 0]
            r0 = rng.choice(mine)
            new_amt = rng.randrange(-50, 900) * 1000 or 2000
            files["notice_revision.txt"] = (
                f"Revision from FP&A: project {p}'s cash flow dated {r0[1]} is restated to {new_amt}. "
                f"Also the hurdle rate was raised to {rate + 1.5}%.\n")
        if level == 1:
            how = ("NPV = sum amount/(1+r)^year with r = discount_rate_pct/100 (year 0 undiscounted). "
                   'Answer: {"best_project": name with the highest NPV, "best_value": its NPV rounded to cents}.')
        else:
            how = ("XNPV = sum amount/(1+r)^(days/365) where days = actual days from that project's earliest date. "
                   "Pick the project with the highest XNPV. Also compute that project's XIRR: the rate x with "
                   "sum amount/(1+x)^(days/365) = 0 (it has exactly one in (-0.99, 10)). "
                   'Answer: {"best_project": name, "best_value": its XNPV rounded to cents, '
                   '"irr_pct": its XIRR in percent rounded to 2 dp}.')
        return "Evaluate the capital projects in cashflows.csv using policy.json. " + how + " " + COMMON_RULES, files

    def solve(self, files, level):
        rate = json.loads(files["policy.json"])["discount_rate_pct"]
        rows = read_csv(files["cashflows.csv"])
        key = "year" if level == 1 else "date"
        flows = {}
        for r in rows:
            flows.setdefault(r["project"], []).append([r[key], float(r["amount"])])
        if "notice_revision.txt" in files:
            t = files["notice_revision.txt"]
            p = _between(t, "project ", "'s")
            d = _between(t, "dated ", " is restated")
            amt = float(_between(t, "restated to ", ".").strip())
            for f in flows[p]:
                if f[0] == d:
                    f[1] = amt
            rate = float(_between(t, "raised to ", "%"))
        r = rate / 100
        vals, series = {}, {}
        for p, fl in flows.items():
            if level == 1:
                vals[p] = sum(a / (1 + r) ** int(t) for t, a in fl)
            else:
                d0 = min(pdate(t) for t, _ in fl)
                s = [((pdate(t) - d0).days / 365, a) for t, a in fl]
                series[p] = s
                vals[p] = sum(a / (1 + r) ** x for x, a in s)
        ranked = sorted(vals.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 500:
            return None  # avoid near-ties
        best, val = ranked[0]
        out = {"best_project": best, "best_value": q2(val)}
        if level >= 2:
            irr = _xirr(series[best])
            if irr is None:
                return None
            out["irr_pct"] = qn(irr * 100, 2)
        return out


def _xirr(series):
    f = lambda x: sum(a / (1 + x) ** t for t, a in series)
    lo, hi = -0.99, 10.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    # require a single sign change on a coarse grid
    grid = [lo + i * (hi - lo) / 400 for i in range(401)]
    vals = [f(g) for g in grid]
    if sum(1 for a, b in zip(vals, vals[1:]) if a * b < 0) != 1:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


FAMILIES = [LoanAmortization(), BondPricing(), CashFlowValuation()]
