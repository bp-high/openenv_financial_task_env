"""Accounting close work: FX consolidation, fixed-asset depreciation, receivables aging, inventory costing,
bank reconciliation."""

import json
from datetime import date, timedelta

from .base import (COMMON_RULES, Family, Spec, dump, iso, pdate, q2, read_csv, to_csv)


def _between(s, a, b):
    return s.split(a, 1)[1].split(b, 1)[0]


# ----------------------------------------------------------------------------- FX consolidation

CCY = {"EUR": (1.02, 1.16), "GBP": (1.18, 1.34), "JPY": (0.0062, 0.0074), "CAD": (0.70, 0.78),
       "INR": (0.0115, 0.0124), "BRL": (0.16, 0.21), "CHF": (1.05, 1.18), "AUD": (0.62, 0.70)}


class FxConsolidation(Family):
    name = "fx_consolidation"
    title = "Translate subsidiary trial balances to USD and consolidate"
    specs = {"revenue_usd": Spec(tol=0.02), "net_income_usd": Spec(tol=0.02), "cash_usd": Spec(tol=0.02)}

    def files(self, rng, level):
        n = rng.randint(3, 5)
        ccys = rng.sample(sorted(CCY), n - 1)
        ents = [("US-PARENT", "USD")] + [(f"SUB-{c}", c) for c in ccys]
        tb, rates = [], []
        for e, c in ents:
            scale = 1 / CCY[c][0] if c != "USD" else 1
            for acct in ["revenue", "expenses", "cash", "receivables"]:
                amt = round(rng.uniform(0.2, 9.0) * 1e6 * scale, 2)
                if acct == "expenses":
                    amt = round(amt * 0.6, 2)
                tb.append([e, acct, amt])
        for c in ccys:
            lo, hi = CCY[c]
            for m in range(1, 13):
                rates.append([f"2025-{m:02d}", c, "average", round(rng.uniform(lo, hi), 6)])
            rates.append(["2025-12-31", c, "closing", round(rng.uniform(lo, hi), 6)])
        files = {"entities.csv": to_csv(["entity", "functional_currency"], [list(x) for x in ents]),
                 "trial_balance_2025.csv": to_csv(["entity", "account", "amount_local"], tb),
                 "fx_rates_usd_per_unit.csv": to_csv(["period", "currency", "type", "usd_per_unit"], rates)}
        rules = ("revenue and expenses translate at the currency's simple mean of the 12 monthly 'average' rates; "
                 "cash and receivables translate at the 'closing' rate. USD amounts need no translation. "
                 "net income = revenue - expenses.")
        if level >= 2:
            seller, buyer = rng.sample(ents, 2)
            amt = round(rng.uniform(0.05, 0.5) * 1e6 / (CCY[seller[1]][0] if seller[1] != "USD" else 1), 2)
            files["intercompany_sales.csv"] = to_csv(["seller", "buyer", "amount_seller_currency"], [[seller[0], buyer[0], amt]])
            rules += (" Eliminate intercompany sales: subtract the translated amount (seller's average rate) from "
                      "consolidated revenue and the same USD amount from consolidated expenses.")
        if level >= 3:
            e, c = rng.choice(ents[1:])
            if rng.random() < 0.5:
                files["notice_controller.txt"] = (
                    f"Controller note: {e} was sold on 2025-12-31 before the close and is excluded from the "
                    "consolidated balance sheet (cash and receivables), but its 2025 revenue and expenses stay in "
                    "the consolidated income statement.\n")
            else:
                newr = round(rng.uniform(*CCY[c]), 6)
                files["notice_controller.txt"] = (
                    f"Controller note: the {c} closing rate in the rates file is a typo; the correct 2025-12-31 "
                    f"closing rate is {newr} USD per {c}.\n")
        prompt = ("Consolidate the group's 2025 trial balances into USD. " + rules +
                  " Do not round intermediate values. "
                  'Answer: {"revenue_usd": number, "net_income_usd": number, "cash_usd": number}, each rounded to '
                  "cents. " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        ents = {r["entity"]: r["functional_currency"] for r in read_csv(files["entities.csv"])}
        avg, close = {}, {}
        for r in read_csv(files["fx_rates_usd_per_unit.csv"]):
            if r["type"] == "average":
                avg.setdefault(r["currency"], []).append(float(r["usd_per_unit"]))
            else:
                close[r["currency"]] = float(r["usd_per_unit"])
        avg = {c: sum(v) / len(v) for c, v in avg.items()}
        avg["USD"] = close["USD"] = 1.0
        excluded = set()
        t = files.get("notice_controller.txt", "")
        if "was sold" in t:
            excluded.add(_between(t, "Controller note: ", " was sold"))
        elif "closing rate is" in t:
            c = _between(t, "the ", " closing rate")
            close[c] = float(_between(t, "closing rate is ", " USD"))
        rev = exp = cash = 0.0
        for r in read_csv(files["trial_balance_2025.csv"]):
            c, a, v = ents[r["entity"]], r["account"], float(r["amount_local"])
            if a == "revenue":
                rev += v * avg[c]
            elif a == "expenses":
                exp += v * avg[c]
            elif a == "cash" and r["entity"] not in excluded:
                cash += v * close[c]
        if "intercompany_sales.csv" in files:
            for r in read_csv(files["intercompany_sales.csv"]):
                usd = float(r["amount_seller_currency"]) * avg[ents[r["seller"]]]
                rev -= usd
                exp -= usd
        return {"revenue_usd": q2(rev), "net_income_usd": q2(rev - exp), "cash_usd": q2(cash)}


# ----------------------------------------------------------------------------- depreciation

class Depreciation(Family):
    name = "depreciation"
    title = "Fixed-asset register: FY depreciation, net book value, disposal gain/loss"
    specs = {"fy_depreciation": Spec(tol=0.05), "year_end_nbv": Spec(tol=0.05), "disposal_gain_loss": Spec(tol=0.02)}

    def files(self, rng, level):
        fy = 2025
        rows = []
        for i in range(rng.randint(4, 7)):
            cost = rng.randrange(5, 400) * 250
            life = rng.choice([3, 5, 7, 10])
            salvage = rng.choice([0, 0, round(cost * 0.1, 2)])
            method = "SL" if level == 1 else rng.choice(["SL", "DDB"])
            placed = date(rng.randint(fy - 3, fy), rng.randint(1, 12), 1)
            rows.append([f"FA-{rng.randrange(1000, 9999)}", cost, salvage, life, method, iso(placed)])
        files = {"asset_register.csv": to_csv(["asset_id", "cost", "salvage", "life_years", "method", "in_service_month"], rows)}
        if level >= 3:
            a = rng.choice([r for r in rows if pdate(r[5]) < date(fy, 1, 1)])
            m = rng.randint(2, 12)
            files["notice_disposal.txt"] = (
                f"Disposal notice: asset {a[0]} was sold on {fy}-{m:02d}-15 for proceeds of "
                f"{round(a[1] * rng.uniform(0.05, 0.7), 2)}. Depreciate it through the month before the sale month, "
                "then remove it from the register. Gain/loss = proceeds - NBV at sale.\n")
        prompt = (
            f"Compute fiscal-year {fy} (calendar) depreciation from asset_register.csv. Depreciation is monthly: an "
            "asset depreciates starting in its in_service_month. SL: monthly amount = (cost - salvage) / life_years / 12 "
            "until NBV reaches salvage. "
            + ("DDB: for each 12-month asset year counted from in_service_month, the asset-year depreciation = NBV at "
               "the start of that asset year * 2 / life_years, capped so NBV never goes below salvage, spread evenly "
               "over that asset year's 12 months. " if level >= 2 else "")
            + f"Sum each asset's {fy} months; round each asset's {fy} total to cents. NBV = cost - accumulated "
            "depreciation (NBV starts at cost). "
            'Answer: {"fy_depreciation": total for all assets, "year_end_nbv": total NBV at 2025-12-31 of assets '
            "still held"
            + (', "disposal_gain_loss": number (negative for a loss)' if level >= 3 else "")
            + "}, rounded to cents. " + COMMON_RULES)
        return prompt, files

    @staticmethod
    def _monthly(cost, salvage, life, method, n_months):
        """Depreciation for months 1..n_months since in service, as a list."""
        out, nbv = [], cost
        for k in range(n_months):
            if method == "SL":
                d = min((cost - salvage) / life / 12, max(nbv - salvage, 0))
            else:
                if k % 12 == 0:
                    year_amt = min(nbv * 2 / life, max(nbv - salvage, 0))
                    per = year_amt / 12
                d = per
            out.append(d)
            nbv -= d
        return out

    def solve(self, files, level):
        fy = 2025
        disposal = None
        t = files.get("notice_disposal.txt")
        if t:
            disposal = (_between(t, "asset ", " was sold"), pdate(_between(t, "sold on ", " for")),
                        float(_between(t, "proceeds of ", ". Depreciate")))
        tot = nbv_sum = 0.0
        gl = None
        for r in read_csv(files["asset_register.csv"]):
            cost, salv, life = float(r["cost"]), float(r["salvage"]), int(r["life_years"])
            placed = pdate(r["in_service_month"])
            start_idx = (fy - placed.year) * 12 - (placed.month - 1)  # months in service before FY start
            end_month = 12
            if disposal and disposal[0] == r["asset_id"]:
                end_month = disposal[1].month - 1
            months_total = start_idx + end_month
            sched = self._monthly(cost, salv, life, r["method"], max(months_total, 0))
            fy_part = sum(sched[max(start_idx, 0):])
            fy_dep = q2(fy_part)
            tot += fy_dep
            nbv = cost - sum(sched)
            if disposal and disposal[0] == r["asset_id"]:
                gl = q2(disposal[2] - nbv)
            else:
                nbv_sum += nbv
        out = {"fy_depreciation": q2(tot), "year_end_nbv": q2(nbv_sum)}
        if level >= 3:
            out["disposal_gain_loss"] = gl if gl is not None else 0.0
        return out


# ----------------------------------------------------------------------------- receivables aging

class ReceivablesAging(Family):
    name = "ar_aging"
    title = "Apply payments and credit notes, then age open receivables"
    specs = {k: Spec(tol=0.01) for k in ["current", "d1_30", "d31_60", "d61_90", "d90_plus", "unapplied_cash"]}

    def files(self, rng, level):
        asof = date(2025, 9, 30)
        custs = [f"C{rng.randrange(100, 999)}" for _ in range(rng.randint(3, 5))]
        inv, pays = [], []
        n = 0
        for c in custs:
            for _ in range(rng.randint(2, 5)):
                n += 1
                d = asof - timedelta(days=rng.randrange(0, 200))
                inv.append([f"INV-{5000 + n * 7}", c, iso(d), iso(d + timedelta(days=rng.choice([15, 30, 45, 60]))),
                            round(rng.randrange(20, 1500) * 10.25, 2)])
        for c in custs:
            mine = [i for i in inv if i[1] == c]
            for _ in range(rng.randint(1, 3)):
                ref = rng.choice(mine)[0] if (level >= 2 and rng.random() < 0.5) else ""
                pd = asof - timedelta(days=rng.randrange(-20, 120)) if level >= 2 else asof - timedelta(days=rng.randrange(0, 120))
                pays.append([f"PMT-{rng.randrange(10000, 99999)}", c, iso(pd), round(rng.randrange(10, 1800) * 10.25, 2), ref])
        files = {"invoices.csv": to_csv(["invoice_id", "customer", "invoice_date", "due_date", "amount"], inv),
                 "payments.csv": to_csv(["payment_id", "customer", "received", "amount", "applies_to"], pays)}
        if level >= 2:
            c = rng.choice(inv)
            files["credit_notes.csv"] = to_csv(["credit_id", "customer", "applies_to", "amount"],
                                               [[f"CN-{rng.randrange(100, 999)}", c[1], c[0], round(c[4] * rng.choice([0.1, 0.25, 0.5]), 2)]])
        if level >= 3:
            p = rng.choice([x for x in pays if pdate(x[2]) <= asof])
            files["notice_treasury.txt"] = f"Treasury notice: payment {p[0]} was returned unpaid (NSF). Treat it as never received.\n"
        prompt = (
            f"Age the receivables as of {iso(asof)}. "
            + ("Ignore payments received after that date. " if level >= 2 else "")
            + ("Credit notes reduce the invoice in applies_to first. " if level >= 2 else "")
            + "Process payments by received date (ties by payment_id). A payment with applies_to pays that invoice "
              "first; any remainder, and every payment without applies_to, pays the customer's open invoices "
              "oldest due_date first (ties by invoice_id). Money left after all of a customer's invoices are paid is "
              "unapplied cash. Days past due = as-of date - due_date. Buckets for open balances: current (<= 0 days), "
              "d1_30, d31_60, d61_90, d90_plus. "
              'Answer: {"current", "d1_30", "d31_60", "d61_90", "d90_plus", "unapplied_cash"}, each a total in dollars '
              "rounded to cents. " + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        asof = date(2025, 9, 30)
        inv = read_csv(files["invoices.csv"])
        open_ = {i["invoice_id"]: float(i["amount"]) for i in inv}
        meta = {i["invoice_id"]: i for i in inv}
        for cn in read_csv(files.get("credit_notes.csv", "credit_id,customer,applies_to,amount\n")):
            open_[cn["applies_to"]] = max(0.0, open_[cn["applies_to"]] - float(cn["amount"]))
        bounced = set()
        if "notice_treasury.txt" in files:
            bounced.add(_between(files["notice_treasury.txt"], "payment ", " was returned"))
        pays = [p for p in read_csv(files["payments.csv"]) if p["payment_id"] not in bounced]
        if level >= 2:
            pays = [p for p in pays if pdate(p["received"]) <= asof]
        pays.sort(key=lambda p: (p["received"], p["payment_id"]))
        unapplied = 0.0
        for p in pays:
            amt = float(p["amount"])
            if p["applies_to"]:
                take = min(amt, open_[p["applies_to"]])
                open_[p["applies_to"]] -= take
                amt -= take
            for iid in sorted((i for i in open_ if meta[i]["customer"] == p["customer"]),
                              key=lambda i: (meta[i]["due_date"], i)):
                if amt <= 1e-9:
                    break
                take = min(amt, open_[iid])
                open_[iid] -= take
                amt -= take
            unapplied += amt
        b = dict.fromkeys(["current", "d1_30", "d31_60", "d61_90", "d90_plus"], 0.0)
        for iid, bal in open_.items():
            if bal <= 1e-9:
                continue
            dpd = (asof - pdate(meta[iid]["due_date"])).days
            key = "current" if dpd <= 0 else "d1_30" if dpd <= 30 else "d31_60" if dpd <= 60 else "d61_90" if dpd <= 90 else "d90_plus"
            b[key] += bal
        out = {k: q2(v) for k, v in b.items()}
        out["unapplied_cash"] = q2(unapplied)
        return out


# ----------------------------------------------------------------------------- inventory costing

class InventoryCosting(Family):
    name = "inventory_cogs"
    title = "Perpetual FIFO / moving-average COGS and ending inventory"
    specs = {"cogs": Spec(tol=0.02), "ending_value": Spec(tol=0.02), "ending_units": Spec(tol=0.0)}

    def files(self, rng, level):
        method = "FIFO" if level == 1 else rng.choice(["FIFO", "MOVING_AVERAGE"])
        d = date(2025, 1, 2)
        on_hand = 0
        rows = []
        n = rng.randint(8, 14)
        for i in range(n):
            d += timedelta(days=rng.randrange(1, 12))
            if on_hand < 50 or rng.random() < 0.45:
                q = rng.randrange(40, 400)
                rows.append([f"T{i + 1:03d}", iso(d), "purchase", q, round(rng.uniform(8, 30), 2)])
                on_hand += q
            else:
                q = rng.randrange(10, on_hand)
                rows.append([f"T{i + 1:03d}", iso(d), "sale", q, ""])
                on_hand -= q
        files = {"inventory_ledger.csv": to_csv(["txn_id", "date", "type", "units", "unit_cost"], rows),
                 "policy.json": dump({"sku": f"SKU-{rng.randrange(1000, 9999)}", "cost_method": method})}
        if level >= 3:
            buys = [r for r in rows if r[2] == "purchase"]
            b = rng.choice(buys[1:] if len(buys) > 1 else buys)
            files["notice_receiving.txt"] = (
                f"Receiving correction: purchase {b[0]} was invoiced at {round(b[4] + rng.choice([-2.5, -1.25, 1.5, 3.0]), 2)} "
                f"per unit, not {b[4]}. Use the invoiced cost.\n")
        prompt = (
            "Cost the SKU's 2025 inventory ledger perpetually in txn order, using policy.json's cost_method. Start "
            "with zero units. FIFO: sales consume the oldest remaining purchase layers. MOVING_AVERAGE: after each "
            "purchase, unit cost = total inventory value / units (unrounded); a sale costs units * that unit cost. "
            'Answer: {"cogs": total cost of all sales, "ending_value": value of units left, both rounded to cents, '
            '"ending_units": integer}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        method = json.loads(files["policy.json"])["cost_method"]
        rows = read_csv(files["inventory_ledger.csv"])
        fix = {}
        if "notice_receiving.txt" in files:
            t = files["notice_receiving.txt"]
            fix[_between(t, "purchase ", " was invoiced")] = float(_between(t, "invoiced at ", " per unit"))
        layers, cogs, units, value = [], 0.0, 0, 0.0
        for r in rows:
            q = int(r["units"])
            if r["type"] == "purchase":
                c = fix.get(r["txn_id"], float(r["unit_cost"]))
                layers.append([q, c])
                units += q
                value += q * c
            else:
                if method == "FIFO":
                    need = q
                    while need:
                        take = min(need, layers[0][0])
                        cogs += take * layers[0][1]
                        value -= take * layers[0][1]
                        layers[0][0] -= take
                        need -= take
                        if not layers[0][0]:
                            layers.pop(0)
                else:
                    uc = value / units
                    cogs += q * uc
                    value -= q * uc
                units -= q
        return {"cogs": q2(cogs), "ending_value": q2(value), "ending_units": units}


# ----------------------------------------------------------------------------- bank reconciliation

class BankReconciliation(Family):
    name = "bank_recon"
    title = "Month-end bank reconciliation"
    specs = {"adjusted_balance": Spec(tol=0.01), "outstanding_checks": Spec(kind="ids"),
             "deposits_in_transit": Spec(tol=0.01), "book_adjustments": Spec(kind="ids")}

    def files(self, rng, level):
        opening = round(rng.uniform(20000, 90000), 2)
        book, bank = [], []
        chk = rng.randrange(1000, 5000)
        d0 = date(2025, 8, 1)
        for i in range(rng.randint(7, 12)):
            d = d0 + timedelta(days=rng.randrange(0, 31))
            if rng.random() < 0.55:
                chk += 1
                amt = -round(rng.uniform(150, 9000), 2)
                book.append([iso(d), f"CHK{chk}", f"Check {chk}", amt])
                if rng.random() < 0.7:
                    bank.append([iso(d + timedelta(days=rng.randrange(1, 6))), f"CHK{chk}", f"CHECK {chk}", amt])
            else:
                amt = round(rng.uniform(500, 15000), 2)
                ref = f"DEP{rng.randrange(100, 999)}"
                book.append([iso(d), ref, "Customer deposit", amt])
                if rng.random() < 0.75:
                    bank.append([iso(d + timedelta(days=rng.randrange(0, 3))), ref, "DEPOSIT", amt])
        extra = [("SVC FEE", -round(rng.uniform(10, 60), 2)), ("INTEREST", round(rng.uniform(2, 40), 2)),
                 ("NSF RETURN", -round(rng.uniform(200, 2500), 2))] if level >= 2 else [("SVC FEE", -round(rng.uniform(10, 60), 2))]
        for i, (desc, amt) in enumerate(extra):
            bank.append([iso(date(2025, 8, 31)), f"BNK{rng.randrange(10, 99)}{i}", desc, amt])
        bank.sort()
        book.sort()
        notice = None
        cleared = [r for r in bank if r[1].startswith("CHK")]
        if level >= 3 and cleared:
            r = rng.choice(cleared)
            true = round(r[3] - rng.choice([9.0, 90.0, 18.0, 45.0]), 2)
            for b in book:
                if b[1] == r[1]:
                    b[3] = true
            notice = (f"Bank letter: check {r[1]} was encoded wrongly. The bank posted {r[3]} but the check is for "
                      f"{true}; the bank will correct it next month. Treat the difference as a bank-side adjustment, "
                      "not a book adjustment.\n")
        bank_end = q2(opening + sum(r[3] for r in bank))
        book_end = q2(opening + sum(r[3] for r in book))
        files = {"bank_statement_aug.csv": to_csv(["date", "ref", "description", "amount"], bank),
                 "book_cash_ledger_aug.csv": to_csv(["date", "ref", "description", "amount"], book),
                 "balances.json": dump({"opening_balance_both": opening, "bank_statement_ending": bank_end,
                                        "book_ending": book_end})}
        if notice:
            files["notice_bank.txt"] = notice
        prompt = (
            "Reconcile August cash. Match bank and book items by ref. Book items missing from the bank are "
            "outstanding checks (negative) or deposits in transit (positive). Bank items missing from the book "
            "must be booked (book adjustments). adjusted_balance = bank_statement_ending + deposits in transit "
            "- outstanding checks (+ any bank-side correction); it must equal book_ending + book adjustments. "
            'Answer: {"adjusted_balance": number, "outstanding_checks": [refs], "deposits_in_transit": total '
            'dollars, "book_adjustments": [bank refs]}. ' + COMMON_RULES)
        return prompt, files

    def solve(self, files, level):
        bank = read_csv(files["bank_statement_aug.csv"])
        book = read_csv(files["book_cash_ledger_aug.csv"])
        bal = json.loads(files["balances.json"])
        bank_refs = {r["ref"] for r in bank}
        book_refs = {r["ref"] for r in book}
        oc = [r for r in book if r["ref"] not in bank_refs and float(r["amount"]) < 0]
        dit = [r for r in book if r["ref"] not in bank_refs and float(r["amount"]) > 0]
        adj = [r for r in bank if r["ref"] not in book_refs]
        adjusted = bal["bank_statement_ending"] + sum(float(r["amount"]) for r in dit) + sum(float(r["amount"]) for r in oc)
        if "notice_bank.txt" in files:
            t = files["notice_bank.txt"]
            posted = float(_between(t, "bank posted ", " but"))
            true = float(_between(t, "check is for ", ";"))
            adjusted += true - posted
        book_side = bal["book_ending"] + sum(float(r["amount"]) for r in adj)
        if abs(adjusted - book_side) > 0.005:
            return None
        return {"adjusted_balance": q2(adjusted), "outstanding_checks": sorted(r["ref"] for r in oc),
                "deposits_in_transit": q2(sum(float(r["amount"]) for r in dit)),
                "book_adjustments": sorted(r["ref"] for r in adj)}


FAMILIES = [FxConsolidation(), Depreciation(), ReceivablesAging(), InventoryCosting(), BankReconciliation()]
