# Chiformer pm7 EAM ablation table

Baseline is the full pm7 d128 Chiformer run. Positive delta means worse PM RMSE@30.

| variant | label | group | pm_rmse_h1 | pm_rmse_h7 | pm_rmse_h10 | pm_rmse_h14 | pm_rmse_h30 | delta_pm_rmse_h30 | delta_pm_rmse_pct_h30 | pm_mae_h30 | raw_chi_rmse_h30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Full baseline | baseline | 0.101 | 1.235 | 1.753 | 2.335 | 4.289 | 0.000 | 0.000 | 2.963 | 32.110 |
| forecast_off | Future off | forecast | 0.094 | 1.241 | 1.773 | 2.373 | 4.384 | 0.095 | 2.224 | 2.997 | 32.408 |
| forecast_7d | Future 7 d | forecast | 0.102 | 1.241 | 1.768 | 2.372 | 4.445 | 0.156 | 3.640 | 3.041 | 32.834 |
| no_eam | No EAM | forecast | 0.113 | 1.270 | 1.816 | 2.453 | 4.505 | 0.216 | 5.035 | 3.057 | 32.277 |
| no_aam | No AAM | leave_one_out | 0.098 | 1.236 | 1.756 | 2.346 | 4.250 | -0.039 | -0.909 | 2.937 | 32.336 |
| no_oam | No OAM | leave_one_out | 0.095 | 1.214 | 1.732 | 2.318 | 4.245 | -0.044 | -1.023 | 2.920 | 32.019 |
| no_ham | No HAM | leave_one_out | 0.103 | 1.256 | 1.783 | 2.375 | 4.375 | 0.086 | 2.005 | 2.979 | 32.081 |
| no_slam | No SLAM | leave_one_out | 0.106 | 1.238 | 1.756 | 2.331 | 4.250 | -0.039 | -0.908 | 2.936 | 32.221 |
| aam_only | AAM only | single_family | 0.098 | 1.228 | 1.751 | 2.347 | 4.295 | 0.006 | 0.133 | 2.949 | 31.863 |
| oam_only | OAM only | single_family | 0.099 | 1.233 | 1.750 | 2.344 | 4.266 | -0.023 | -0.538 | 2.935 | 31.778 |
| ham_only | HAM only | single_family | 0.095 | 1.226 | 1.745 | 2.336 | 4.270 | -0.019 | -0.432 | 2.926 | 31.989 |
| slam_only | SLAM only | single_family | 0.106 | 1.263 | 1.803 | 2.429 | 4.507 | 0.218 | 5.075 | 3.072 | 32.268 |
