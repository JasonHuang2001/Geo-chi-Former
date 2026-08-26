# Figure 3 notes

- Core conclusion: Geo-$\chi$-Former gives the lowest 30-day PM RMSE on the refreshed 2020-2025 fixed weekly test windows.
- Data lineage: `results/frozen/baselines/model_comparison_summary.csv` contains the frozen 309-issue-date model-comparison metrics.
- Panels: a) horizon-wise PM RMSE curves, with forecast domain encoded by line style and model family encoded by marker shape; b) ranked PM RMSE@30 bars; c) PM domain versus $\chi$ domain gain for shared architectures.
- Display transform: no smoothing; panel c uses PM domain H30 minus $\chi$ domain H30, so positive values indicate that forecasting in the excitation domain improves H30 RMSE.
- Interpretation note: this figure compares saved best checkpoints only, not the full historical grid-search population.

## Draft caption

Model comparison over fixed weekly forecast starts from 2020-01-02 to 2025-11-27. (a) Horizon-wise PM RMSE shows error growth from 1 to 30 days, with line style separating PM domain and $\chi$ domain forecasting and marker shape identifying model families. (b) Ranked 30-day PM RMSE summarizes the main operational comparison. (c) The 30-day RMSE difference between PM domain and $\chi$ domain variants quantifies the benefit of forecasting in the excitation domain before Wilson reconstruction; positive values indicate lower error for the $\chi$ domain formulation.
