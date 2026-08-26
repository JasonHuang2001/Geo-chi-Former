# Checkpoint input files

The published checkpoint requires two prepared CSV files. They are not committed because they combine third-party Earth-orientation and angular-momentum products whose redistribution terms are separate from the repository’s Apache-2.0 software license.

Place authorized copies in one local directory and pass it to `python run.py validate --data-dir ...` or `python run.py predict --data-dir ...`.

| Filename | Role | Frozen paper snapshot | SHA-256 | Schema |
|---|---|---|---|---|
| `eop_data_xy_EAM.csv` | Daily PM and analyzed AAM/OAM/HAM/SLAM | 18,263 rows; 1976-01-01 to 2025-12-31 | `CEC7BB2B64C66C9F9E852381187A5620036EA01E03D37010ECEE7AC7293470F2` | `schemas/eop_eam_daily.schema.json` |
| `eam14forecast_daily.csv` | Issue-date/lead-indexed 14-day EAM forecasts | 19,012 rows; 1,358 issue dates from 2021-05-20 to 2025-12-08 | `0705FA92124484292A75B09E7545D7D562CB4B0D0CFD77F19E9EA98B93BEBA1A` | `schemas/eam_14day_forecast.schema.json` |

The daily file stores `xpole` and `ypole` in arcseconds and EAM excitation values in dimensionless form. The forecast file uses zero-based `lead_day`, with `date = issue_date + lead_day` and 14 rows per complete issue date. The complete machine-readable manifest is `paper_input_manifest.json`; the frozen split is `splits/paper_split.json`.

Use `--allow-unverified-inputs` only for intentional non-paper experiments. It skips snapshot hashes while retaining structural validation.

## Upstream product families

- IERS EOP 20 C04: <https://datacenter.iers.org/data/latestVersion/EOP_20_C04_one_file_1962-now.txt>
- IERS Bulletin A: <https://maia.usno.navy.mil/products/bulletin-a>
- GFZ ESMGFZ EAM products: <https://rz-vm480.gfz.de/files/ESMGFZ/EAM/>
- ETH Zurich GPC EAM products: <https://gpc.ethz.ch/products/EAM/>

Consult each provider’s current attribution, citation, access, and redistribution terms. The bundled `reference/paper_ul_gap_test_L360_P30.npz` is an author-generated numerical-regression intermediate, not a replacement for either prepared CSV.
