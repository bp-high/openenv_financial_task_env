# Qwen3.8-27B calibration (thinking off, Arena-style loop, billed via DeepPPT)

6 episodes per task against the live eval model; lenient first-JSON parse exactly like the arena.
Kept tasks with 0.20 < mean < 0.95 and <=3/6 format failures (mixed-outcome, gradient-bearing).

| task | mean solve | format-fail /6 | status |
|---|---|---|---|
| ind_oee-l2 | 1.00 | 0 | saturated |
| ind_tank-l2 | 1.00 | 0 | saturated |
| sci_calibration-l3 | 1.00 | 0 | saturated |
| sci_kinetics-l2 | 1.00 | 0 | saturated |
| sec_firewall-l1 | 1.00 | 0 | saturated |
| sw_impact-l2 | 1.00 | 0 | saturated |
| ar_aging-l2 | 0.83 | 0 | KEEP |
| bank_recon-l2 | 0.83 | 1 | KEEP |
| bond_price-l1 | 0.83 | 1 | KEEP |
| math_coins-l1 | 0.83 | 1 | KEEP |
| math_coins-l2 | 0.83 | 1 | KEEP |
| math_route-l1 | 0.83 | 1 | KEEP |
| media_edl-l1 | 0.83 | 1 | KEEP |
| media_music-l1 | 0.83 | 1 | KEEP |
| office_expense-l3 | 0.83 | 1 | KEEP |
| office_meeting-l1 | 0.83 | 1 | KEEP |
| office_meeting-l2 | 0.83 | 1 | KEEP |
| office_meeting-l3 | 0.83 | 1 | KEEP |
| pe_waterfall-l1 | 0.83 | 1 | KEEP |
| portfolio_return-l1 | 0.83 | 0 | KEEP |
| real_growth-l1 | 0.83 | 1 | KEEP |
| sci_calibration-l2 | 0.83 | 1 | KEEP |
| sec_auth-l1 | 0.83 | 1 | KEEP |
| sw_impact-l1 | 0.83 | 1 | KEEP |
| cap_table-l1 | 0.67 | 1 | KEEP |
| covenant-l1 | 0.67 | 2 | KEEP |
| dcf_npv-l1 | 0.67 | 2 | KEEP |
| depreciation-l1 | 0.67 | 2 | KEEP |
| ind_oee-l1 | 0.67 | 1 | KEEP |
| ind_tank-l1 | 0.67 | 2 | KEEP |
| ind_tank-l3 | 0.67 | 2 | KEEP |
| inventory_cogs-l1 | 0.67 | 2 | KEEP |
| media_srt-l1 | 0.67 | 2 | KEEP |
| payroll-l1 | 0.67 | 2 | KEEP |
| sci_kinetics-l1 | 0.67 | 2 | KEEP |
| sec_auth-l2 | 0.67 | 2 | KEEP |
| sw_resolver-l1 | 0.67 | 1 | KEEP |
| bank_recon-l1 | 0.50 | 3 | KEEP |
| fx_consolidation-l1 | 0.50 | 3 | KEEP |
| media_srt-l2 | 0.50 | 3 | KEEP |
| office_expense-l1 | 0.50 | 3 | KEEP |
| office_expense-l2 | 0.50 | 3 | KEEP |
| ind_oee-l3 | 0.33 | 4 | too-hard/format |
| loan_amort-l1 | 0.33 | 4 | too-hard/format |
| sw_impact-l3 | 0.33 | 4 | too-hard/format |
| ar_aging-l1 | 0.17 | 5 | too-hard/format |
| media_edl-l2 | 0.17 | 5 | too-hard/format |
| media_music-l2 | 0.17 | 4 | too-hard/format |
| sci_calibration-l1 | 0.17 | 0 | too-hard/format |
| sw_resolver-l2 | 0.17 | 5 | too-hard/format |
