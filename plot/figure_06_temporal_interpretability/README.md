# Figure 6 notes

Core conclusion: Variable-wise autocorrelation provides a data-only temporal persistence reference for the fixed d128 pm7 checkpoint's variable-specific Cross-Time attention enrichment across short, intraseasonal, half-year, and near-annual visible lags.
Figure archetype: two-panel temporal interpretability main figure
Target journal/output: JGR/Nature-style manuscript figures with editable vector text
Backend: Python/matplotlib
Combined figure size: 183 mm x 96 mm

Data lineage:
- Attention reusable source table: `plot/figure_06_temporal_interpretability/source_data/panel_b_variable_lag_enrichment.csv`.
- Autocorrelation reusable source table: `plot/figure_06_temporal_interpretability/source_data/panel_a_variable_autocorrelation.csv`.
- The bundled source tables contain the fixed author-generated values shown in Figure 6.

Figure map:
- a: variable-wise absolute autocorrelation for GAM/chi and EAM-family x and y components.
- b: variable-wise Cross-Time lag-attention enrichment for chi and EAM x/y channels.

Display transforms:
- The attention figure averages attention_weight over sample, rollout, layer, and head for each variable and key patch, then normalizes each variable's lag profile by its own mean lag attention.
- The attention figure uses patch_len=8; max visible lag is 352 d for seq_len=360. The 352 (~1 yr) label is near-annual, not a complete 365.25 d annual-cycle claim.
- The autocorrelation source table computes same-variable autocorrelation over lags from 0 to +365 d at 5 d spacing.
- Panel a plots |autocorr_x| and |autocorr_y| separately. Absolute autocorrelation describes temporal persistence only and does not distinguish positive from negative phase relationships.
- No temporal smoothing was applied to the autocorrelation curves.
- The autocorrelation figure is retrospective physical context and is not an operational future-input claim.

Interpretation limits:
- These figures support trained-model temporal attention diagnostics and retrospective variable-memory context, not Earth-system physical causality.
- The near-annual wording is restricted to the visible patch-token history edge.
- Future PM/GAM are never used as Chiformer inputs; the autocorrelation figure is a separate data-only diagnostic.
- Suggested text interpretation: the autocorrelation panel provides a variable-internal statistical reference for the attention heatmap. Differences among GAM/chi and EAM-family autocorrelation curves indicate which variables retain stronger memory at short, intraseasonal, half-year, and near-annual lags. Agreement between these persistence scales and the Cross-Time attention enrichment supports a temporal-memory interpretation, but it does not imply learned physical causality.

JGR-style draft captions:

Figure X. Temporal interpretability diagnostics for the fixed d128 pm7 Chiformer checkpoint. (a) Variable-wise autocorrelation over 0-365 d in the test-period physical source data, with the upper and lower axes corresponding to the x and y components, respectively. No temporal smoothing was applied, and absolute autocorrelations quantify each variable's own temporal persistence. (b) Cross-time attention by variable and lag. Attention weights were averaged over test issue dates, rollout blocks, layers, and heads and then normalized within each variable, so colors emphasize each variable's lag profile rather than total attention magnitude. Values greater than one indicate above-average attention enrichment. Vertical dashed lines mark 30, 90, 182.6 d, and the maximum visible lag (352 (~1 yr); seq_len=360, patch_len=8). Panel (a) provides retrospective data-only temporal-persistence context for the model attention diagnostic in panel (b), but the figure does not establish physical causality.

Source data:
- panel_a_variable_autocorrelation.csv
- panel_b_variable_lag_enrichment.csv
