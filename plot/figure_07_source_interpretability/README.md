# Figure 7 notes

Core conclusion: The fixed d128 pm7 checkpoint uses EAM components selectively, the axis-aware model strengthens same-axis source preference, and fixed-checkpoint masking shows the largest PM response for the components emphasized by attention.
Figure archetype: three-panel source-use diagnostic with SI-only normalized-importance comparison
Target journal/output: JGR manuscript figure with editable vector text
Backend: Python/matplotlib
Final size: 183 mm x 126 mm

Data lineage:
- Attention source table: `plot/figure_07_source_interpretability/source_data/panel_a_attention_enrichment.csv`.
- Axis-contrast source table: `plot/figure_07_source_interpretability/source_data/panel_b_same_axis_contrast.csv`.
- Masking-response source table: `plot/figure_07_source_interpretability/source_data/panel_c_masking_response.csv`.
- Supplementary comparison table: `plot/figure_07_source_interpretability/source_data/supplement_attention_masking_importance.csv`.
- Full checkpoint run: `chiformer_joint_d128_doy8_EAM_a-o-h-s_fObsEAM_fEAMQ0_TIN_varEmb_tgtQ_axisBias_KV_sh_1_curH8-16-24-30_P_8_360_30_fi_0_pm_7`.
- No-axis-aware checkpoint run: `chiformer_joint_d128_doy8_EAM_a-o-h-s_fObsEAM_fEAMQ0_TIN_varEmb_tgtQ_noAxisBias_KV_sh_1_curH8-16-24-30_P_8_360_30_fi_0_pm_7_abmodel_no_axis_bias_only`.
- Test issue dates: n=309.

Panel map:
- a: channel-count-corrected EAM component attention enrichment for chi_x and chi_y target queries.
- b: same-axis minus cross-axis source preference comparing the full model with the no-axis-aware ablation.
- c: sample-level distribution of fixed-checkpoint EAM-component masking effects on Wilson-reconstructed PM.

Display transforms:
- Panel a sums axis-resolved attention_share and expected_uniform_share within each EAM component and target, then plots observed share divided by expected channel-count share.
- Panel b uses target-specific axis_preference_delta values; each horizontal segment connects no-axis-aware and full-model values for one target-component pair.
- Panel c uses vertical boxplots from sample-level masking effects; boxes show 25th-75th percentiles, whiskers show 5th-95th percentiles, median lines show sample medians, and filled points show mean absolute PM changes.
- The normalized attention-versus-masking comparison is exported as SI-only source data and should not be duplicated in the main figure.

Interpretation limits:
- These are trained-model attention and masking diagnostics, not proof of Earth-system physical causality.
- Masking effects explain the fixed trained checkpoint and are not Earth-system interventions.
- Attention and masking need not be perfectly rank-equivalent because they probe different model surfaces.

JGR-style draft caption:

Figure X. Cross-variate attention and fixed-checkpoint source sensitivity diagnostics. (a) Channel-count-corrected attention enrichment for EAM components in target excitation queries. (b) Same-axis enrichment contrast for the full model and the no-axis-aware variant. (c) Distribution of absolute PM responses after masking each EAM component.

Source data:
- panel_a_attention_enrichment.csv
- panel_b_same_axis_contrast.csv
- panel_c_masking_response.csv
- supplement_attention_masking_importance.csv
