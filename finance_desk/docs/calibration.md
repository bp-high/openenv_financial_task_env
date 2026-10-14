# Qwen3.8-27B calibration (thinking off, Arena-style loop, billed via DeepPPT)

6 episodes/task against the live eval model; lenient first-JSON parse as the arena does.
KEEP gate: 0.20 < mean solve < 0.95 and <=3/6 format failures. 86 task ids measured across two passes.
Final submission = 50 KEEP tasks (all 8 domains).

| task | mean | fmt/6 | in_submission |
|---|---|---|---|
| cap_table-l2 | 1.00 | 0 |  |
| ind_oee-l2 | 1.00 | 0 |  |
| ind_tank-l2 | 1.00 | 0 |  |
| loan_amort-l2 | 1.00 | 0 |  |
| math_coins-l3 | 1.00 | 0 |  |
| sci_calibration-l3 | 1.00 | 0 |  |
| sci_kinetics-l2 | 1.00 | 0 |  |
| sec_firewall-l1 | 1.00 | 0 |  |
| sw_impact-l2 | 1.00 | 0 |  |
| ar_aging-l2 | 0.83 | 0 | yes |
| bank_recon-l2 | 0.83 | 1 | yes |
| bond_price-l1 | 0.83 | 1 | yes |
| cap_table-l3 | 0.83 | 1 |  |
| covenant-l3 | 0.83 | 1 |  |
| inventory_cogs-l2 | 0.83 | 1 |  |
| math_coins-l1 | 0.83 | 1 | yes |
| math_coins-l2 | 0.83 | 1 | yes |
| math_route-l1 | 0.83 | 1 | yes |
| media_edl-l1 | 0.83 | 1 | yes |
| media_music-l1 | 0.83 | 1 | yes |
| office_expense-l3 | 0.83 | 1 | yes |
| office_meeting-l1 | 0.83 | 1 | yes |
| office_meeting-l2 | 0.83 | 1 | yes |
| office_meeting-l3 | 0.83 | 1 | yes |
| pe_waterfall-l1 | 0.83 | 1 | yes |
| portfolio_return-l1 | 0.83 | 0 | yes |
| real_growth-l1 | 0.83 | 1 | yes |
| sci_calibration-l2 | 0.83 | 1 | yes |
| sec_auth-l1 | 0.83 | 1 | yes |
| sec_firewall-l2 | 0.83 | 1 | yes |
| sec_firewall-l3 | 0.83 | 1 | yes |
| sw_impact-l1 | 0.83 | 1 | yes |
| bond_price-l3 | 0.67 | 2 | yes |
| cap_table-l1 | 0.67 | 1 | yes |
| covenant-l1 | 0.67 | 2 | yes |
| covenant-l2 | 0.67 | 2 | yes |
| dcf_npv-l1 | 0.67 | 2 | yes |
| dcf_npv-l2 | 0.67 | 2 | yes |
| depreciation-l1 | 0.67 | 2 | yes |
| fx_consolidation-l3 | 0.67 | 2 | yes |
| ind_oee-l1 | 0.67 | 1 | yes |
| ind_tank-l1 | 0.67 | 2 | yes |
| ind_tank-l3 | 0.67 | 2 | yes |
| inventory_cogs-l1 | 0.67 | 2 | yes |
| inventory_cogs-l3 | 0.67 | 2 | yes |
| loan_amort-l3 | 0.67 | 2 | yes |
| math_route-l2 | 0.67 | 2 | yes |
| media_srt-l1 | 0.67 | 2 | yes |
| media_srt-l3 | 0.67 | 2 | yes |
| payroll-l1 | 0.67 | 2 | yes |
| payroll-l2 | 0.67 | 2 | yes |
| pe_waterfall-l2 | 0.67 | 2 |  |
| sci_kinetics-l1 | 0.67 | 2 | yes |
| sec_auth-l2 | 0.67 | 2 | yes |
| sw_resolver-l1 | 0.67 | 1 | yes |
| bank_recon-l1 | 0.50 | 3 | yes |
| bond_price-l2 | 0.50 | 3 |  |
| depreciation-l2 | 0.50 | 3 |  |
| fx_consolidation-l1 | 0.50 | 3 | yes |
| fx_consolidation-l2 | 0.50 | 3 |  |
| math_route-l3 | 0.50 | 3 | yes |
| media_srt-l2 | 0.50 | 3 | yes |
| office_expense-l1 | 0.50 | 3 | yes |
| office_expense-l2 | 0.50 | 3 | yes |
| portfolio_return-l2 | 0.50 | 3 |  |
| real_growth-l2 | 0.50 | 3 |  |
| sci_kinetics-l3 | 0.50 | 3 | yes |
| sec_auth-l3 | 0.50 | 3 | yes |
| bank_recon-l3 | 0.33 | 4 |  |
| dcf_npv-l3 | 0.33 | 4 |  |
| ind_oee-l3 | 0.33 | 4 |  |
| loan_amort-l1 | 0.33 | 4 |  |
| payroll-l3 | 0.33 | 4 |  |
| portfolio_return-l3 | 0.33 | 4 |  |
| sw_impact-l3 | 0.33 | 4 |  |
| ar_aging-l1 | 0.17 | 5 |  |
| media_edl-l2 | 0.17 | 5 |  |
| media_edl-l3 | 0.17 | 5 |  |
| media_music-l2 | 0.17 | 4 |  |
| pe_waterfall-l3 | 0.17 | 5 |  |
| real_growth-l3 | 0.17 | 5 |  |
| sci_calibration-l1 | 0.17 | 0 |  |
| sw_resolver-l2 | 0.17 | 5 |  |
| sw_resolver-l3 | 0.17 | 5 |  |
| ar_aging-l3 | 0.00 | 6 |  |
| depreciation-l3 | 0.00 | 6 |  |
