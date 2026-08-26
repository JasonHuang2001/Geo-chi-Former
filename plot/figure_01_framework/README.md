# Figure 1 notes

- Core conclusion: The proposed framework shifts the learning target from the PM response domain to the excitation domain, while retaining PM-domain evaluation through Wilson forward reconstruction.
- Data lineage: Schematic only; variables follow Dataset_EOP_ULGap semantics: PM is converted to GAM by Wilson inversion, chi^S is forecast by Chiformer, and chi^G is reconstructed before Wilson forward PM evaluation.
- Panel map: one horizontal schematic with observed PM, Wilson inversion, excitation-domain forecasting, and Wilson reconstruction.
- Display transforms: schematic geometry only; no quantitative data transformation is plotted.
- Interpretation note: Future GAM and future PM must not appear as model inputs; future EAM is only a forecast-side covariate in true-forecast or pseudo-forecast settings.
