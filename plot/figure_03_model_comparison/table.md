# Refreshed model-comparison table

Evaluation window: 2020-01-02 to 2025-11-27, stride 7.

| model | target_space | samples | pm_rmse_h1 | pm_rmse_h7 | pm_rmse_h14 | pm_rmse_h30 | pm_mae_h30 | run_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chiformer | chi2pm | 309 | 0.101223 | 1.235490 | 2.335093 | 4.288893 | 2.962715 | chiformer_joint_d128_doy8_EAM_a-o-h-s_fObsEAM_fEAMQ0_TIN_varEmb_tgtQ_axisBias_KV_sh_1_curH8-16-24-30_P_8_360_30_fi_0_pm_7 |
| LSTM | chi2pm | 309 | 0.270937 | 1.616309 | 2.650670 | 4.439917 | 3.150899 | lstm_chi2pm_d128_doy8_EAM_a-o-h-s_RevIN_S360_H30_pm5 |
| DLinear | chi2pm | 309 | 0.097772 | 1.285445 | 2.448191 | 4.505020 | 3.052596 | dlinear_chi2pm_d64_doy8_EAM_a-o-h-s_RevIN_S720_H30_pm5_K61_ind1 |
| PatchTST | chi2pm | 309 | 0.145956 | 1.340619 | 2.484460 | 4.556401 | 3.125501 | patchtst_chi2pm_d128_doy8_EAM_a-o-h-s_RevIN_P_16_8_S720_H30_pm5 |
| BulletinA | pm | 309 | 0.059796 | 1.228538 | 2.464223 | 4.761389 | 3.262212 | bulletina |
| DLinear | pm | 309 | 0.400181 | 1.736405 | 2.993742 | 5.416403 | 3.809657 | dlinear_pm_d64_doy8_EAM_a-o-h-s_RevIN_S720_H30_pm5_K7_ind1 |
| PatchTST | pm | 309 | 1.493477 | 2.450810 | 3.479554 | 5.639719 | 4.196140 | patchtst_pm_d64_doy8_EAM_a-o-h-s_RevIN_P_16_8_S720_H30_pm5 |
| LSTM | pm | 309 | 6.749933 | 6.604839 | 6.995674 | 8.431197 | 6.686013 | lstm_pm_d128_doy8_EAM_a-o-h-s_RevIN_S720_H30_pm5 |
