"""Payroll, venture/PE finance, credit covenants, performance measurement and macroeconomics."""

import json
import math
from datetime import date, timedelta

from .base import (COMMON_RULES, Family, Spec, dump, iso, pdate, q2, qn, read_csv, to_csv)


def _between(s, a, b):
    return s.split(a, 1)[1].split(b, 1)[0]


# ----------------------------------------------------------------------------- payroll

BRACKETS = [[0, 10], [11925, 12], [48475, 22], [103350, 24], [197300, 32], [250525, 35], [626350, 37]]


class Payroll(Family):
    name = "payroll"
    title = "Monthly payroll withholding with a social-security wage base"
    specs = {"total_net_pay": Spec(tol=0.03), "total_income_tax": Spec(tol=0.03), "total_social_security": Spec(tol=0.03)}

    def files(self, rng, level):
        rows = []
        for i in range(rng.randint(3, 6)):
            sal = rng.randrange(40, 420) * 1000
            ytd = q2(sal / 12 * 10)
            rows.append([f"E{rng.randrange(100, 999)}{i}", sal, rng.choice([0, 3, 5, 6, 8, 10]), ytd])
        policy = {"pay_period": "2025-11", "periods_per_year": 12, "standard_deduction_annual": 15000,
                  "federal_brackets_annual": [{"over": o, "rate_pct": r} for o, r in BRACKETS],
                  "social_security_rate_pct": 6.2, "social_security_wage_base": 176100, "medicare_rate_pct": 1.45}
        if level >= 2:
            policy["additional_medicare_rate_pct"] = 0.9
            policy["additional_medicare_threshold_ytd"] = 200000
        files = {"employees.csv": to_csv(["emp_id", "annual_salary", "pretax_401k_pct", "ytd_gross_before_period"], rows),
                 "payroll_policy.json": dump(policy)}
        if level >= 3:
            e = rng.choice(rows)
            if rng.random() < 0.5:
                files["notice_hr.txt"] = (f"HR notice: effective this pay period, {e[0]}'s annual salary is "
                                          f"{e[1] + rng.randrange(5, 60) * 1000}. YTD figures are unchanged.\n")
            else:
                files["notice_hr.txt"] = (f"HR notice: effective this pay period, {e[0]}'s 401k election changes to "
                                          f"{rng.choice([0, 2, 4, 12, 15])}% of gross.\n")
        prompt = (
            "Run the 2025-11 payroll (one of 12 monthly periods) from employees.csv and payroll_policy.json. Per "
            "employee, every amount rounded half-up to cents: gross = annual_salary/12; 401k = gross*pretax_401k_pct/100; "
            "income tax = (progressive tax on max(0, (gross-401k)*12 - standard_deduction_annual), each bracket's "
            "rate applying to income above its 'over' threshold up to the next) / 12; social security = rate * the "
            "part of gross that keeps YTD gross (ytd_gross_before_period + gross) within the wage base; medicare = "
            "rate * gross"
            + ("; additional medicare = its rate * the part of this period's gross that takes YTD gross above the "
               "threshold" if level >= 2 else "")
            + ". 401k does not reduce social security or medicare wages. net = gross - 401k - income tax - social "
              "security - medicare" + (" - additional medicare" if level >= 2 else "") + ". "
            'Answer: {"total_net_pay", "total_income_tax", "total_social_security"}: sums over employees in dollars. '
            + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        pol = json.loads(files["payroll_policy.json"])
        emps = read_csv(files["employees.csv"])
        t = files.get("notice_hr.txt", "")
        net = tax = ss = 0.0
        for e in emps:
            sal, pct, ytd = float(e["annual_salary"]), float(e["pretax_401k_pct"]), float(e["ytd_gross_before_period"])
            if t and e["emp_id"] + "'s" in t:
                if "annual salary is" in t:
                    sal = float(_between(t, "annual salary is ", ". "))
                else:
                    pct = float(_between(t, "changes to ", "%"))
            gross = q2(sal / 12)
            k = q2(gross * pct / 100)
            taxable = max(0.0, (gross - k) * 12 - pol["standard_deduction_annual"])
            br = pol["federal_brackets_annual"]
            annual = 0.0
            for i, b in enumerate(br):
                top = br[i + 1]["over"] if i + 1 < len(br) else float("inf")
                if taxable > b["over"]:
                    annual += (min(taxable, top) - b["over"]) * b["rate_pct"] / 100
            it = q2(annual / 12)
            base = pol["social_security_wage_base"]
            s = q2(max(0.0, min(gross, base - ytd)) * pol["social_security_rate_pct"] / 100)
            med = q2(gross * pol["medicare_rate_pct"] / 100)
            add = 0.0
            if "additional_medicare_rate_pct" in pol:
                thr = pol["additional_medicare_threshold_ytd"]
                add = q2(max(0.0, ytd + gross - max(thr, ytd)) * pol["additional_medicare_rate_pct"] / 100)
            net += gross - k - it - s - med - add
            tax += it
            ss += s
        return {"total_net_pay": q2(net), "total_income_tax": q2(tax), "total_social_security": q2(ss)}


# ----------------------------------------------------------------------------- cap table

class CapTable(Family):
    name = "cap_table"
    title = "Priced round with SAFE conversion"
    specs = {"price_per_share": Spec(tol=0.00005), "new_investor_shares": Spec(tol=0.0),
             "safe_shares": Spec(tol=0.0), "founders_pct": Spec(tol=0.01)}

    def files(self, rng, level):
        f = rng.randint(2, 3)
        rows = [[f"Founder {chr(65 + i)}", "common", rng.randrange(1500, 5000) * 1000] for i in range(f)]
        rows.append(["Employees (granted)", "options", rng.randrange(300, 1500) * 1000])
        rows.append(["Unallocated pool", "options", rng.randrange(100, 800) * 1000])
        rows.append(["Angel", "common", rng.randrange(50, 400) * 1000])
        pre = rng.randrange(8, 60) * 1_000_000
        inv = rng.randrange(2, 20) * 1_000_000
        files = {"cap_table.csv": to_csv(["holder", "class", "shares"], rows),
                 "term_sheet.json": dump({"round": "Series A", "pre_money_valuation": pre, "investment": inv})}
        if level >= 2:
            safes = []
            for i in range(rng.randint(1, 3)):
                safes.append([f"SAFE-{i + 1}", rng.randrange(1, 12) * 250_000, rng.randrange(4, 50) * 1_000_000,
                              rng.choice([0, 10, 15, 20, 25])])
            files["safes.csv"] = to_csv(["safe_id", "amount", "valuation_cap", "discount_pct"], safes)
        if level >= 3:
            sid = rng.choice(read_csv(files["safes.csv"]))["safe_id"]
            if rng.random() < 0.5:
                files["notice_counsel.txt"] = f"Counsel note: {sid} was repurchased and cancelled before the round; it does not convert.\n"
            else:
                files["notice_counsel.txt"] = (f"Counsel note: {sid} was amended; its valuation_cap is "
                                               f"{rng.randrange(4, 50) * 1_000_000} and its discount_pct is 0.\n")
        prompt = (
            "Close the Series A in term_sheet.json. Pre-money fully diluted shares S = all shares in cap_table.csv "
            "(common plus all options, granted and unallocated)"
            + (", excluding SAFE conversion shares" if level >= 2 else "")
            + ". price_per_share = pre_money_valuation / S, rounded half-up to 4 dp. new_investor_shares = "
              "floor(investment / price_per_share). "
            + ("Each SAFE converts at conversion price = min(valuation_cap / S, price_per_share * (1 - "
               "discount_pct/100)), rounded half-up to 4 dp; its shares = floor(amount / conversion price). "
               if level >= 2 else "")
            + "founders_pct = shares held by holders named 'Founder ...' / post-money fully diluted shares * 100, "
              "rounded to 2 dp; post-money = S + new investor shares"
            + (" + SAFE shares" if level >= 2 else "") + ". "
            'Answer: {"price_per_share": number, "new_investor_shares": integer'
            + (', "safe_shares": integer total' if level >= 2 else "")
            + ', "founders_pct": number}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        rows = read_csv(files["cap_table.csv"])
        ts = json.loads(files["term_sheet.json"])
        S = sum(int(r["shares"]) for r in rows)
        founders = sum(int(r["shares"]) for r in rows if r["holder"].startswith("Founder"))
        price = qn(ts["pre_money_valuation"] / S, 4)
        new = math.floor(ts["investment"] / price + 1e-9)
        safe_sh = 0
        if level >= 2:
            safes = read_csv(files["safes.csv"])
            t = files.get("notice_counsel.txt", "")
            for s in safes:
                cap, disc = float(s["valuation_cap"]), float(s["discount_pct"])
                if t and s["safe_id"] + " " in t:
                    if "cancelled" in t:
                        continue
                    cap = float(_between(t, "valuation_cap is ", " and"))
                    disc = 0.0
                cp = qn(min(cap / S, price * (1 - disc / 100)), 4)
                if cp <= 0:
                    return None
                safe_sh += math.floor(float(s["amount"]) / cp + 1e-9)
        post = S + new + safe_sh
        out = {"price_per_share": price, "new_investor_shares": new, "founders_pct": qn(founders / post * 100, 2)}
        if level >= 2:
            out["safe_shares"] = safe_sh
        return out


# ----------------------------------------------------------------------------- PE waterfall

class PeWaterfall(Family):
    name = "pe_waterfall"
    title = "European distribution waterfall with preferred return and GP catch-up"
    specs = {"lp_total": Spec(tol=0.02), "gp_carry": Spec(tol=0.02)}

    def files(self, rng, level):
        exit_d = date(2025, 6, 30)
        n = 1 if level == 1 else rng.randint(2, 4)
        contribs = []
        d = date(rng.randint(2018, 2021), rng.randint(1, 12), 1)
        for _ in range(n):
            contribs.append([iso(d), rng.randrange(5, 60) * 100_000])
            d += timedelta(days=rng.randrange(90, 500))
        C = sum(c[1] for c in contribs)
        proceeds = round(C * rng.uniform(0.85, 3.2), 2)
        terms = {"preferred_return_pct": rng.choice([6, 7, 8, 9]), "carried_interest_pct": rng.choice([20, 20, 25]),
                 "gp_catch_up": "none" if level == 1 else "full", "exit_date": iso(exit_d), "exit_proceeds": proceeds}
        files = {"lpa_terms.json": dump(terms), "capital_calls.csv": to_csv(["call_date", "amount"], contribs)}
        if level >= 3:
            if rng.random() < 0.5:
                files["notice_side_letter.txt"] = (f"Side letter (supersedes LPA): the preferred return is "
                                                   f"{terms['preferred_return_pct'] - rng.choice([1, 2])}%.\n")
            else:
                c = rng.choice(contribs)
                files["notice_side_letter.txt"] = (f"Administrator correction: the capital call dated {c[0]} was "
                                                   f"{c[1] + rng.randrange(1, 10) * 50_000}, not {c[1]}.\n")
        prompt = (
            "Split exit_proceeds between LPs and the GP. Tier 1: LPs get back total called capital. Tier 2: LPs get "
            "the preferred return: for each call, amount * ((1 + pref/100)^(days/365) - 1), days = actual days from "
            "call_date to exit_date. "
            + ("Tier 3 (full catch-up): the GP gets 100% until it holds carried_interest_pct of tiers 2+3, i.e. "
               "pref_paid * carry/(100-carry). " if level >= 2 else "")
            + "Last tier: the remainder is split (100-carry)% LP / carry% GP. Each tier is paid only from what is "
              "left. "
              'Answer: {"lp_total": number, "gp_carry": number} in dollars rounded to cents. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        t = json.loads(files["lpa_terms.json"])
        calls = [[r["call_date"], float(r["amount"])] for r in read_csv(files["capital_calls.csv"])]
        pref = t["preferred_return_pct"]
        n = files.get("notice_side_letter.txt", "")
        if "preferred return is" in n:
            pref = float(_between(n, "preferred return is ", "%"))
        elif "capital call dated" in n:
            d = _between(n, "dated ", " was")
            for c in calls:
                if c[0] == d:
                    c[1] = float(_between(n, " was ", ", not"))
        ex = pdate(t["exit_date"])
        carry = t["carried_interest_pct"] / 100
        rem = t["exit_proceeds"]
        C = sum(a for _, a in calls)
        P = sum(a * ((1 + pref / 100) ** ((ex - pdate(d)).days / 365) - 1) for d, a in calls)
        roc = min(rem, C); rem -= roc
        pp = min(rem, P); rem -= pp
        cu = 0.0
        if t["gp_catch_up"] == "full":
            cu = min(rem, pp * carry / (1 - carry)); rem -= cu
        lp = roc + pp + rem * (1 - carry)
        gp = cu + rem * carry
        return {"lp_total": q2(lp), "gp_carry": q2(gp)}


# ----------------------------------------------------------------------------- covenants

class Covenant(Family):
    name = "covenant"
    title = "Credit-agreement covenant test on LTM financials"
    specs = {"net_leverage": Spec(tol=0.005), "interest_coverage": Spec(tol=0.005), "compliant": Spec(kind="bool")}

    def files(self, rng, level):
        qs = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3"]
        test_q = rng.choice(qs[3:]) if level >= 2 else qs[-1]
        rows = []
        for q in (qs if level >= 2 else qs[-4:]):
            rev = rng.randrange(40, 200) * 100_000
            rows.append([q, rev, round(rev * rng.uniform(0.4, 0.6)), round(rev * rng.uniform(0.15, 0.3)),
                         round(rev * rng.uniform(0.03, 0.06)), round(rev * rng.uniform(0.03, 0.10)),
                         rng.choice([0, 0, round(rev * rng.uniform(0.01, 0.08))]), round(rev * rng.uniform(0.01, 0.04))])
        i = [r[0] for r in rows].index(test_q)
        pre = sum(r[1] - r[2] - r[3] for r in rows[i - 3:i + 1])
        cash = rng.randrange(5, 120) * 100_000
        bal = {"test_quarter": test_q, "total_debt": round(max(pre, 1e6) * rng.uniform(1.5, 6.0) / 100_000) * 100_000 + cash,
               "cash": cash}
        ca = {"addback_cap_pct": rng.choice([10, 15, 20]), "cash_netting_cap": rng.choice([2_500_000, 5_000_000, 7_500_000]),
              "max_net_leverage": rng.choice([3.0, 3.5, 4.0, 4.5]), "min_interest_coverage": rng.choice([2.0, 2.5, 3.0])}
        files = {"quarterly_pnl.csv": to_csv(["quarter", "revenue", "cogs", "opex_incl_restructuring", "d_and_a",
                                              "interest_expense", "restructuring_charges", "tax_expense"], rows),
                 "balance_sheet.json": dump(bal),
                 "credit_agreement.txt": (
                     "Section 1.1 Definitions (summary).\n"
                     "'LTM' means the four fiscal quarters ending with the test quarter.\n"
                     "'EBITDA' means LTM revenue - cogs - opex_incl_restructuring (so before D&A, interest and tax), "
                     "plus an add-back of LTM restructuring_charges limited to addback_cap_pct% of EBITDA before "
                     "that add-back.\n"
                     "'Net Debt' means total_debt minus cash, where the cash deducted may not exceed cash_netting_cap.\n"
                     "Section 7.1: Net Leverage = Net Debt / EBITDA must not exceed max_net_leverage. Interest "
                     "Coverage = EBITDA / LTM interest_expense must be at least min_interest_coverage.\n"
                     "Terms: " + json.dumps(ca) + "\n")}
        if level >= 3:
            if rng.random() < 0.5:
                files["notice_amendment.txt"] = (f"Amendment No. 2: addback_cap_pct is "
                                                 f"{ca['addback_cap_pct'] + rng.choice([5, 10])} for all tests.\n")
            else:
                files["notice_amendment.txt"] = (f"Amendment No. 2: total_debt for the test excludes a "
                                                 f"{rng.randrange(5, 40) * 100_000} non-recourse facility.\n")
        prompt = (
            "Test the borrower's covenants per credit_agreement.txt at the test quarter in balance_sheet.json. "
            "Compliant means both covenant tests pass on the unrounded ratios. "
            'Answer: {"net_leverage": ratio rounded to 2 dp, "interest_coverage": ratio rounded to 2 dp, '
            '"compliant": true/false}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        bal = json.loads(files["balance_sheet.json"])
        ca = json.loads(files["credit_agreement.txt"].split("Terms: ")[1])
        debt = bal["total_debt"]
        n = files.get("notice_amendment.txt", "")
        if "addback_cap_pct is" in n:
            ca["addback_cap_pct"] = float(_between(n, "addback_cap_pct is ", " for"))
        elif "excludes a" in n:
            debt -= float(_between(n, "excludes a ", " non-recourse"))
        rows = read_csv(files["quarterly_pnl.csv"])
        qs = [r["quarter"] for r in rows]
        i = qs.index(bal["test_quarter"])
        if i < 3:
            return None
        ltm = rows[i - 3:i + 1]
        s = lambda k: sum(float(r[k]) for r in ltm)
        pre = s("revenue") - s("cogs") - s("opex_incl_restructuring")
        if pre <= 0:
            return None
        e = pre + min(s("restructuring_charges"), pre * ca["addback_cap_pct"] / 100)
        nd = debt - min(bal["cash"], ca["cash_netting_cap"])
        lev, cov = nd / e, e / s("interest_expense")
        if abs(lev - ca["max_net_leverage"]) < 0.05 or abs(cov - ca["min_interest_coverage"]) < 0.05:
            return None
        ok = lev <= ca["max_net_leverage"] and cov >= ca["min_interest_coverage"]
        return {"net_leverage": qn(lev, 2), "interest_coverage": qn(cov, 2), "compliant": ok}


# ----------------------------------------------------------------------------- performance measurement

class PortfolioReturn(Family):
    name = "portfolio_return"
    title = "Modified Dietz and time-weighted returns with external flows"
    specs = {"modified_dietz_pct": Spec(tol=0.005), "twr_pct": Spec(tol=0.005)}

    def files(self, rng, level):
        start, end = date(2025, 1, 1), date(2025, 6, 30)
        v = round(rng.uniform(0.5, 5) * 1e6, 2)
        rows_v = [[iso(start), "start", v]]
        flows = []
        dts = sorted(rng.sample(range(10, 170), rng.randint(2, 4)))
        for k in dts:
            d = start + timedelta(days=k)
            v = round(v * (1 + rng.uniform(-0.06, 0.08)), 2)
            rows_v.append([iso(d), "before_flow", v])
            f = round(rng.choice([1, 1, -1]) * rng.uniform(0.05, 0.4) * v, 2)
            flows.append([iso(d), f, rng.choice(["client contribution", "client withdrawal"]) if True else ""])
            flows[-1][2] = "client contribution" if f > 0 else "client withdrawal"
            v = round(v + f, 2)
        v = round(v * (1 + rng.uniform(-0.06, 0.08)), 2)
        rows_v.append([iso(end), "end", v])
        files = {"valuations.csv": to_csv(["date", "point", "market_value"], rows_v),
                 "cash_flows.csv": to_csv(["date", "amount", "description"], flows)}
        if level >= 3:
            fl = rng.choice(flows)
            files["notice_ops.txt"] = (f"Operations note: the {fl[1]} flow on {fl[0]} was a transfer between two "
                                       "sleeves of this same portfolio, not an external flow; the market values "
                                       "already reflect it. Exclude it from external flows.\n")
        prompt = (
            "Measure the portfolio's return from 2025-01-01 to 2025-06-30. External flows are in cash_flows.csv "
            "(positive = money in). Modified Dietz = (EMV - BMV - sum F) / (BMV + sum F_i * (D - d_i)/D), D = actual "
            "days from start to end, d_i = actual days from start to the flow. "
            + ("Time-weighted return: chain the sub-period returns between flow dates, where each sub-period starts "
               "at the previous point's value plus that point's external flow and ends at the next 'before_flow' or "
               "'end' value; TWR = product of (1 + r) - 1. " if level >= 2 else "")
            + 'Answer: {"modified_dietz_pct": percent rounded to 2 dp'
            + (', "twr_pct": percent rounded to 2 dp' if level >= 2 else "") + "}. " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        vals = read_csv(files["valuations.csv"])
        flows = {r["date"]: float(r["amount"]) for r in read_csv(files["cash_flows.csv"])}
        if "notice_ops.txt" in files:
            flows[_between(files["notice_ops.txt"], " flow on ", " was")] = 0.0
        s, e = pdate(vals[0]["date"]), pdate(vals[-1]["date"])
        D = (e - s).days
        bmv, emv = float(vals[0]["market_value"]), float(vals[-1]["market_value"])
        F = sum(flows.values())
        den = bmv + sum(f * (D - (pdate(d) - s).days) / D for d, f in flows.items())
        out = {"modified_dietz_pct": qn((emv - bmv - F) / den * 100, 2)}
        if level >= 2:
            g, start_v = 1.0, bmv
            for r in vals[1:]:
                g *= float(r["market_value"]) / start_v
                start_v = float(r["market_value"]) + flows.get(r["date"], 0.0)
            out["twr_pct"] = qn((g - 1) * 100, 2)
        return out


# ----------------------------------------------------------------------------- macroeconomics

class RealGrowth(Family):
    name = "real_growth"
    title = "Real GDP growth, CAGR and rebasing from nominal GDP and a deflator"
    specs = {"real_growth_pct": Spec(tol=0.005), "real_cagr_pct": Spec(tol=0.005), "real_level_rebased": Spec(tol=0.01)}

    def files(self, rng, level):
        years = list(range(2014, 2025))
        nom, defl = [], []
        n, dfl = rng.uniform(800, 25000), rng.uniform(88, 98)
        for y in years:
            rows = None
            n *= 1 + rng.uniform(-0.03, 0.09)
            dfl *= 1 + rng.uniform(-0.005, 0.05)
            nom.append(round(n, 1))
            defl.append(round(dfl if y != 2017 else 100.0, 2))
        country = rng.choice(["Aurelia", "Borovia", "Cascadia", "Dravnia", "Elbonia"])
        y1 = rng.choice(years[3:])
        y0 = rng.choice([y for y in years if y <= y1 - 3])
        files = {"national_accounts.csv": to_csv(["year", "nominal_gdp_bn", "gdp_deflator_2017_100"],
                                                 [[y, a, b] for y, a, b in zip(years, nom, defl)]),
                 "request.json": dump({"country": country, "growth_year": y1, "cagr_from": y0, "cagr_to": y1,
                                       **({"rebase_year": rng.choice([y for y in years if y != 2017])} if level >= 2 else {})})}
        if level >= 3:
            yy = rng.choice([y0, y1, y1 - 1])
            files["notice_statistics_office.txt"] = (
                f"Statistics office revision: nominal GDP for {yy} is revised to "
                f"{round(nom[years.index(yy)] * rng.uniform(1.01, 1.04), 1)} bn. Deflators are unchanged.\n")
        prompt = (
            "From national_accounts.csv and request.json: real GDP (2017 prices) = nominal / deflator * 100. "
            "real_growth_pct = real GDP growth in growth_year over the prior year; real_cagr_pct = compound annual "
            "growth of real GDP from cagr_from to cagr_to. "
            + ("real_level_rebased = real GDP of growth_year in rebase_year prices = nominal / (deflator / "
               "deflator of rebase_year), in bn rounded to 2 dp. " if level >= 2 else "")
            + 'Answer: {"real_growth_pct", "real_cagr_pct"' + (', "real_level_rebased"' if level >= 2 else "")
            + "} with percents rounded to 2 dp. " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        req = json.loads(files["request.json"])
        data = {int(r["year"]): [float(r["nominal_gdp_bn"]), float(r["gdp_deflator_2017_100"])]
                for r in read_csv(files["national_accounts.csv"])}
        n = files.get("notice_statistics_office.txt", "")
        if n:
            data[int(_between(n, "nominal GDP for ", " is"))][0] = float(_between(n, "revised to ", " bn"))
        real = {y: a / b * 100 for y, (a, b) in data.items()}
        y1, y0 = req["growth_year"], req["cagr_from"]
        out = {"real_growth_pct": qn((real[y1] / real[y1 - 1] - 1) * 100, 2),
               "real_cagr_pct": qn(((real[req["cagr_to"]] / real[y0]) ** (1 / (req["cagr_to"] - y0)) - 1) * 100, 2)}
        if level >= 2:
            out["real_level_rebased"] = q2(data[y1][0] / (data[y1][1] / data[req["rebase_year"]][1]))
        return out


FAMILIES = [Payroll(), CapTable(), PeWaterfall(), Covenant(), PortfolioReturn(), RealGrowth()]
