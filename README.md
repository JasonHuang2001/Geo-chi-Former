# Geo-chi-Former

Code and checkpoint for:

> Junhai Huang, Boyang Zhu, and Xiaodong Liu. “Geo-chi-Former: A Geodynamically Constrained Excitation-Domain Transformer for Short-Term Polar Motion Prediction.” Manuscript prepared for *Journal of Geophysical Research: Solid Earth*.

Geo-chi-Former forecasts short-term polar motion in the geophysical angular-momentum excitation domain. This repository contains the model, checkpoint inference code, paper checkpoint, figure source data, and static figure previews.

## Included scope

| Path | Contents |
|---|---|
| `models/chiformer.py` | Core Geo-chi-Former architecture |
| `data/` | EOP/EAM loader, input schemas, paper split, and UL-gap intermediate |
| `utils/integrator.py` | Wilson inversion and polar-motion reconstruction |
| `scripts/`, `utils/` | Checkpoint prediction, input validation, and evaluation |
| `checkpoints/paper_main/` | Paper checkpoint, immutable configuration, and SHA-256 checksums |
| `plot/figure_*` | Figure source data and explanatory notes; no plotting programs |
| `figures/previews/` | Static previews of manuscript Figures 1–7 |
| `tests/` | Strict checkpoint loading, CPU forward pass, physics, CLI, and file checks |

Training experiments, baseline models, ablation runners, interpretability pipelines, paper-table builders, and plotting programs are not included.

## Installation

The frozen environment uses Windows, Python 3.12, PyTorch 2.7.1, and CUDA 12.8.

```powershell
git clone https://github.com/JasonHuang2001/Geo-chi-Former.git
cd Geo-chi-Former
conda env create -f environment.yml
conda activate geo-chi-former
```

Verify that the checkpoint loads into the model and completes a CPU forward pass:

```powershell
python -B run.py test
```

This smoke test does not require the restricted prepared inputs.

## Run the paper checkpoint

Checkpoint prediction requires two prepared files in a local data directory:

- `eop_data_xy_EAM.csv`
- `eam14forecast_daily.csv`

They combine third-party IERS, GFZ, and ETH Zurich products and are not redistributed here. Expected hashes, schemas, date ranges, and preparation boundaries are documented in [`data/README.md`](data/README.md).

Validate authorized local copies:

```powershell
python -B run.py validate --data-dir D:\path\to\prepared_data
```

Run one issue date:

```powershell
python -B run.py predict `
  --data-dir D:\path\to\prepared_data `
  --output-dir outputs\single_issue `
  --issue-date 2020-01-02
```

Run the frozen 309-issue-date paper protocol:

```powershell
python -B run.py predict `
  --data-dir D:\path\to\prepared_data `
  --output-dir outputs\paper_protocol `
  --all-paper-dates
```

The default checkpoint, evaluation overlay, and UL-gap intermediate are selected automatically and verified by SHA-256. To evaluate an exported prediction table independently:

```powershell
python -B run.py evaluate `
  --predictions outputs\paper_protocol\test_predictions.csv
```

## Figure source data

All retained files below `plot/figure_*` are source data or explanatory notes, not executable plotting code. The two largest Figure 2 tables are stored as reproducible gzip streams:

```powershell
gzip -dk plot\figure_02_distribution_shift\distribution.csv.gz
gzip -dk plot\figure_02_distribution_shift\gap_timeseries.csv.gz
```

Python and pandas can also read them directly with `pandas.read_csv("file.csv.gz")`.

## Figure previews

### Figure 1 — Geo-chi-Former architecture

![Figure 1](figures/previews/figure_01_framework.png)

### Figure 2 — Distribution-shift treatment

![Figure 2](figures/previews/figure_02_distribution_shift.png)

### Figure 3 — Model comparison

![Figure 3](figures/previews/figure_03_model_comparison.png)

### Figure 4 — EAM ablation

![Figure 4](figures/previews/figure_04_eam_ablation.png)

### Figure 5 — Structural ablation

![Figure 5](figures/previews/figure_05_model_ablation.png)

### Figure 6 — Temporal interpretability

![Figure 6](figures/previews/figure_06_temporal_interpretability.png)

### Figure 7 — Source interpretability

![Figure 7](figures/previews/figure_07_source_interpretability.png)

## Data, citation, and license

The Apache License 2.0 covers the repository’s original source code and author-generated project files. It does not relicense upstream Earth-orientation or angular-momentum products. See [`docs/data_availability.md`](docs/data_availability.md) and [`docs/third_party_notices.md`](docs/third_party_notices.md).

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The article DOI and archived software DOI should be added after publication.
