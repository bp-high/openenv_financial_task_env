# Finance Desk v3 calibration (live Qwen3.8-27B)

Arena-style loop (`calibrate_v3.py`, scratchpad harness): system message = the action schema; first user turn =
the reset observation as JSON; every reply must be exactly one JSON object (an action or `{"finish": true}`), any
other reply ends the episode and it is scored through `evaluate`; 32 tool calls; 4096 completion tokens counting
every observation after the first; 8192 context; temperature 1.0; HF router `:fastest`. 4 episodes per id per pass.

- **no-think**: `reasoning_effort: none` (provider reasoning off). Primary pass.
- **/no_think**: a `/no_think` system prefix; the provider still reasons in a hidden field, and that reasoning
  counts against the budget (many empty replies). Kept as a second view.

No-think pass: 228 episodes, mean reward 0.57. Zero-reward format endings: prose before the JSON 42, empty 28, invalid JSON 10, code fence 1. Solved episodes used 3.9 tool calls on average.

Selection: mixed outcome across the two passes (pooled solve rate roughly 0.15-0.85), software first (the most
reachable private point), then finance families for variety. Saturated (8/8) and dead (0/8) ids are left out.

| task id | no-think | /no_think | pooled | selected |
|---|---|---|---|---|
| `sw_dates_fix-l1` | 2/4 | 1/4 | 0.38 | **yes** |
| `sw_dates_fix-l2` | 1/4 | 1/4 | 0.25 | **yes** |
| `sw_dates_fix-l3` | 2/4 | 2/4 | 0.50 | **yes** |
| `sw_logstats_impl-l1` | 3/4 | 2/4 | 0.62 | **yes** |
| `sw_logstats_impl-l2` | 0/3 | 0/4 | 0.00 |  |
| `sw_logstats_impl-l3` | 0/4 | 0/4 | 0.00 |  |
| `sw_money_fix-l1` | 0/4 | 0/4 | 0.00 |  |
| `sw_money_fix-l2` | 1/4 | 0/4 | 0.12 | **yes** |
| `sw_money_fix-l3` | 1/4 | 0/4 | 0.12 |  |
| `sw_report_fix-l1` | 3/4 | 2/4 | 0.62 | **yes** |
| `sw_report_fix-l2` | 1/4 | 3/4 | 0.50 | **yes** |
| `sw_report_fix-l3` | 1/4 | 2/4 | 0.38 | **yes** |
| `sw_stats_fix-l1` | 2/4 | 3/4 | 0.62 | **yes** |
| `sw_stats_fix-l2` | 2/4 | 4/4 | 0.75 | **yes** |
| `sw_stats_fix-l3` | 0/3 | 4/4 | 0.57 | **yes** |
| `ar_aging-l1` | 3/4 | 0/4 | 0.38 |  |
| `ar_aging-l2` | 2/4 | 0/4 | 0.25 |  |
| `ar_aging-l3` | 2/4 | 0/4 | 0.25 |  |
| `bank_recon-l1` | 3/4 | 3/4 | 0.75 |  |
| `bank_recon-l2` | 3/4 | 3/4 | 0.75 |  |
| `bank_recon-l3` | 2/4 | 2/4 | 0.50 | **yes** |
| `bond_price-l1` | 2/4 | 3/4 | 0.62 |  |
| `bond_price-l2` | 2/4 | 3/4 | 0.62 |  |
| `bond_price-l3` | 2/4 | 3/4 | 0.62 | **yes** |
| `cap_table-l1` | 4/4 | 4/4 | 1.00 |  |
| `cap_table-l2` | 4/4 | 1/4 | 0.62 |  |
| `cap_table-l3` | 3/4 | 1/4 | 0.50 | **yes** |
| `covenant-l1` | 2/4 | 4/4 | 0.75 |  |
| `covenant-l2` | 3/4 | 3/4 | 0.75 |  |
| `covenant-l3` | 2/4 | 2/4 | 0.50 | **yes** |
| `dcf_npv-l1` | 4/4 | 4/4 | 1.00 |  |
| `dcf_npv-l2` | 3/3 | 4/4 | 1.00 |  |
| `dcf_npv-l3` | 2/4 | 3/3 | 0.71 |  |
| `depreciation-l1` | 2/4 | 1/4 | 0.38 |  |
| `depreciation-l2` | 0/4 | 1/4 | 0.12 |  |
| `depreciation-l3` | 1/4 | 0/4 | 0.12 |  |
| `fx_consolidation-l1` | 3/4 | 4/4 | 0.88 |  |
| `fx_consolidation-l2` | 3/4 | 2/4 | 0.62 | **yes** |
| `fx_consolidation-l3` | 3/4 | 0/4 | 0.38 |  |
| `inventory_cogs-l1` | 4/4 | 4/4 | 1.00 |  |
| `inventory_cogs-l2` | 4/4 | 4/4 | 1.00 |  |
| `inventory_cogs-l3` | 4/4 | 3/4 | 0.88 |  |
| `loan_amort-l1` | 3/4 | 4/4 | 0.88 |  |
| `loan_amort-l2` | 3/4 | 4/4 | 0.88 |  |
| `loan_amort-l3` | 1/3 | 2/4 | 0.43 |  |
| `payroll-l1` | 2/3 | 1/4 | 0.43 |  |
| `payroll-l2` | 2/4 | 1/4 | 0.38 | **yes** |
| `payroll-l3` | 3/3 | 0/4 | 0.43 |  |
| `pe_waterfall-l1` | 4/4 | 4/4 | 1.00 |  |
| `pe_waterfall-l2` | 2/4 | 3/4 | 0.62 |  |
| `pe_waterfall-l3` | 3/4 | 2/4 | 0.62 | **yes** |
| `portfolio_return-l1` | 3/3 | 3/3 | 1.00 |  |
| `portfolio_return-l2` | 3/4 | 1/4 | 0.50 | **yes** |
| `portfolio_return-l3` | 1/4 | 0/3 | 0.14 |  |
| `real_growth-l1` | 3/4 | 4/4 | 0.88 |  |
| `real_growth-l2` | 2/4 | 4/4 | 0.75 |  |
| `real_growth-l3` | 2/4 | 4/4 | 0.75 |  |

Selected 19 ids: 11 software, 8 finance.

## v3b (42 ids)

v3's run failed on the trainer side (TRAINER_REPORTED_FAILURE at step 116, no evaluation). v3b resubmits the same
image with every id whose pooled solve rate is 0.12-0.80: the 19 above plus 23 more (`tasks_v3b.txt`): 12 software,
30 finance. Left out: 7 saturated ids (8/8), 3 dead (0/8) and 5 near-saturated (0.88).
