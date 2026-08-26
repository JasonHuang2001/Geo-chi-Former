# Figure 2 notes

Core conclusion: History-only low-frequency background removal and target-anchored normalization reduce preprocessing-driven scale shifts in the excitation target.
Figure archetype: quantitative grid
Target journal/output: AGU/JGR-ready full-width manuscript figure plus SI raw-comparison panel
Backend: Python/matplotlib
Final main size: 183 mm x 118 mm
Final supplement size: 183 mm x 68 mm

Data lineage:
- Source file: data/eop_data_xy_EAM.csv.
- GAM is Wilson-derived from PM in mas.
- Summed EAM follows the loader aggregation of AAM, OAM, HAM, and SLAM components.
- The low-frequency background is fitted from historical gap values only.
- Distribution panels use train, validation, and test history windows from Dataset_EOP_ULGap.

Panel map:
- a: full-record gap and history-only background; upper/lower rows show x/y components.
- b: raw GAM and short-to-seasonal target distributions across train, validation, and test splits; upper/lower rows show x/y components.
- c: normalized target distributions across train, validation, and test splits; upper/lower rows show x/y components.
- Supplement: raw GAM and summed EAM during the test period for x and y components.

Display transforms:
- Gap curves use a 30-day running mean.
- Displayed background uses a 365-day centred median; unsmoothed values remain in source data.
- Distribution violins use deterministic subsamples clipped to q01-q99 for display; exact statistics use all values.
- Within each violin, the thin white vertical line marks the interquartile range and the short white horizontal line marks the median.

Interpretation notes:
- No future target samples are used for target-anchored normalization.
- The figure documents preprocessing and scale behavior, not physical causality.
- Main and supplement figures avoid figure-level captions inside the graphics.

Draft caption:
History-only background removal and target-anchored normalization stabilize the excitation target. (a) Full-record gap and fitted low-frequency background. (b) Raw GAM and short-to-seasonal target distributions across data splits. (c) Normalized target distributions after sample-wise target anchoring. Upper and lower rows show x and y components.

Source data:
- Gap/source rows: 12052.
- Distribution-source rows: 126000.
- Statistics rows: 18.

Exact split statistics:
- x/raw_gam: test-train mean shift +75.984, std ratio 0.691
- x/short_gam: test-train mean shift -15.994, std ratio 0.904
- x/normalized_target: test-train mean shift +0.000, std ratio 1.000
- y/raw_gam: test-train mean shift -16.195, std ratio 1.037
- y/short_gam: test-train mean shift +35.746, std ratio 1.007
- y/normalized_target: test-train mean shift -0.000, std ratio 1.000

Exported files:
- gap_timeseries.csv.gz
- distribution.csv.gz
- distribution_stats.csv
- split_metadata.csv
- figure_02_distribution_shift.png
- figure_02_distribution_shift.jpg
- figure_02_distribution_shift.pdf
- figure_02_distribution_shift.svg
- standalone panel files prefixed with figure_02_distribution_shift_panel_

Supplement exported files:
- supplement_raw_gam_eam.png
- supplement_raw_gam_eam.jpg
- supplement_raw_gam_eam.pdf
- supplement_raw_gam_eam.svg
- standalone panel files prefixed with supplement_raw_gam_eam_panel_
