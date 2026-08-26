# Frozen numerical-regression intermediate

`paper_ul_gap_test_L360_P30.npz` stores the 309 history-only UL-gap fits used by the saved paper evaluation. It is an author-generated derived intermediate, not an IERS, GFZ, or ETH raw product and not a substitute for either prepared checkpoint-input CSV.

The file is required because refitting `sklearn.linear_model.HuberRegressor` did not reproduce the previously saved values exactly even under the nominally pinned environment. The cache filename used by the legacy loader did not encode the numerical-library build, and regenerated values differed by as much as 1.431 mas. Freezing this approximately 0.97 MB intermediate makes the checkpoint regression auditable without silently widening the acceptance threshold.

SHA-256: `B8D0DE60E6514DC2AB0813D0A3C13987B4DA13423F96135CAC1B551B7A67D04C`

The prediction command verifies this checksum and loads the file read-only. For a single issue date, it selects the matching row from the complete 309-sample cache. The underlying prepared PM/EAM CSV files remain local and are still required through `--data-dir`.
