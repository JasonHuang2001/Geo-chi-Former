# Geo-chi-Former

Core source code and figure previews for:

> Junhai Huang, Boyang Zhu, and Xiaodong Liu. “Geo-chi-Former: A Geodynamically Constrained Excitation-Domain Transformer for Short-Term Polar Motion Prediction.” Manuscript prepared for *Journal of Geophysical Research: Solid Earth*.

Geo-chi-Former forecasts short-term polar motion in the geophysical angular-momentum excitation domain. This compact repository presents the proposed architecture, comparison models, training and validation code, the data loader, the Wilson excitation–polar-motion conversion, and the figures generated for the manuscript.

## Contents

| Path | Description |
|---|---|
| `models/chiformer.py` | Geo-chi-Former architecture |
| `models/DLinear.py` | DLinear comparison model |
| `models/LSTM.py` | LSTM comparison model |
| `models/PatchTST.py` | PatchTST comparison model |
| `data/EOP_loader.py` | PM/EAM dataset construction and preprocessing |
| `data/eop_core.py` | Shared excitation-domain data functions |
| `utils/integrator.py` | Wilson inversion and polar-motion reconstruction |
| `utils/*_trainer.py` | Training loops, validation, early stopping, and checkpoint selection |
| `workflows/` | Geo-chi-Former and unified baseline training entry points |
| `configs/baseline_runs.json` | PM-domain and excitation-domain comparison settings |
| `data/example_eop_eam.csv` | Illustrative prepared-data CSV layout |
| `figures/previews/` | Generated manuscript figures |

Model checkpoints, plotting programs, grid searches, ablation runners, and intermediate results are intentionally omitted. The retained Python files require PyTorch, NumPy, pandas, and scikit-learn; these dependencies are listed in `requirements.txt`.

## Data

The model uses daily polar motion from IERS and Earth angular-momentum products from GFZ/ESMGFZ and the ETH Zurich prediction comparison service. These third-party data are not redistributed here.

Download links, required units, column names, and preparation notes are provided in [`data/README.md`](data/README.md). The included [`data/example_eop_eam.csv`](data/example_eop_eam.csv) shows the daily column layout expected by the loader; its three rows and numeric values are illustrative only.

## Training and validation

Place the prepared daily data in `data/eop_data_xy_EAM.csv`. Geo-chi-Former also uses `data/eam14forecast_daily.csv` when forecast-side EAM is enabled. The three-row example documents columns only and is not long enough for training.

Each entry point builds the training and validation date splits, validates once per epoch, applies early stopping, and restores the checkpoint with the lowest validation loss:

```powershell
python -m workflows.train_chiformer
python -m workflows.train_baseline --model dlinear --target-space pm
python -m workflows.train_baseline --model lstm --target-space chi
python -m workflows.train_baseline --model patchtst --target-space chi
```

The unified baseline entry point accepts `--target-space pm` for direct polar-motion prediction and `--target-space chi` for excitation-domain prediction followed by Wilson reconstruction. It reads the exact settings for all six paper comparison runs from [`configs/baseline_runs.json`](configs/baseline_runs.json).

## Figures

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

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The source code is released under the [Apache License 2.0](LICENSE). Upstream IERS, GFZ, and ETH Zurich data remain subject to their providers’ terms.
