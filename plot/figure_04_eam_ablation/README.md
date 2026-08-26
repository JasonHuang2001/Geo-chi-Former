# Figure 4 notes

- Core conclusion: EAM inputs and forecast-side EAM availability improve 30-day PM forecasts; HAM shows the most robust marginal contribution among the tested EAM families.
- Data lineage: `results/frozen/ablations/model_ablation_summary.csv` supplies the full pm7 baseline; `results/frozen/ablations/eam_forecast_ablation_summary.csv` supplies EAM and forecast-data ablation metrics.
- Evaluation target: Wilson-integrated PM errors from Chiformer chi forecasts.
- Panels: a) relative PM RMSE@30 changes; b) horizon-wise PM RMSE changes for every non-baseline EAM ablation, with marker shape and line style helping distinguish curves; c) remove-family versus family-only comparison.
- Display transform: PM RMSE deltas are reported relative to the full pm7 baseline. Panel b uses percentage change at each horizon. No smoothing or temporal aggregation is applied.
- Interpretation note: EAM-family ablations are model-performance diagnostics and should not be worded as physical causal proof.
