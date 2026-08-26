# Figure 5 notes

- Core conclusion: the full Chiformer has the lowest PM RMSE at 30 d among the clean model-ablation variants; removing the cross-variate branch and future-aware KV shift yields the clearest degradation, while the No RoPE setting causes smaller but consistent degradation.
- Data lineage: Figure 5 uses the bundled summary and source-data tables in this directory. Historical per-run metric paths are provenance records and are not distributed.
- Panel map: panel a is the main H30 PM RMSE percentage-change ranking; panel b is the lead-wise PM RMSE percentage-change profile over H1/H7/H10/H14/H30, with the day-1 noCVA spike retained in the data and briefly labeled because it exceeds the displayed y-range.
- Display transforms: no smoothing or temporal aggregation is applied; all deltas are direct arithmetic relative to the full baseline, and percentage changes use the corresponding full-baseline metric as denominator. Panel b uses a display-only y-axis upper limit of 11% so the main lead-wise differences remain readable.
- Layout archetype: full-width two-panel main-text figure, matching the restrained AGU/JGR comparison style used by the pm7 EAM ablation and refreshed model-comparison figures.
- Interpretation note: noCVA disables the cross-variate branch together with variate embedding and should be described as branch removal; the legacy learnable noRoPE diagnostic run is intentionally excluded from this clean ablation figure.
- Uncertainty note: no 95% confidence intervals are drawn because the current metric JSON files do not contain paired bootstrap or per-issue uncertainty estimates.

## Draft caption

Model-structure ablation for the fixed d128 pm7 Chiformer checkpoint. (a) Percentage change in 30-day Wilson-integrated PM RMSE relative to the full model; positive values indicate degradation after removing or replacing a module. The No RoPE variant replaces RoPE with sinusoidal positional encoding rather than removing positional information entirely. (b) Lead-wise PM RMSE percentage changes at 1, 7, 10, 14, and 30 days, using the same color mapping as panel (a). The day-1 No attention across variables value exceeds the displayed y-range and is labeled in the panel; the full value is retained in the source data. Removing attention across variables and the KV shift shows the largest 30-day gains, while positional encoding and axis-aware bias provide horizon-dependent contributions.
