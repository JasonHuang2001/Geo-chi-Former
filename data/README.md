# Data preparation

The repository does not redistribute the Earth-orientation or angular-momentum records used in the manuscript. Obtain them from the official services and follow each provider's citation and usage terms.

## Download sources

| Data | Official source | Use |
|---|---|---|
| Polar motion | [IERS EOP 20 C04](https://datacenter.iers.org/data/latestVersion/EOP_20_C04_one_file_1962-now.txt) | Daily `xpole` and `ypole` at 0 h UTC |
| Atmospheric angular momentum | [GFZ operational AAM](https://rz-vm480.gfz.de/files/ESMGFZ/EAM/operational_AAM/) | AAM mass and motion terms |
| Oceanic angular momentum | [GFZ operational OAM](https://rz-vm480.gfz.de/files/ESMGFZ/EAM/operational_OAM/) | OAM mass and motion terms |
| Hydrological angular momentum | [GFZ operational HAM](https://rz-vm480.gfz.de/files/ESMGFZ/EAM/operational_HAM/) | HAM mass and motion terms |
| Sea-level angular momentum | [GFZ operational SLAM](https://rz-vm480.gfz.de/files/ESMGFZ/EAM/operational_SLAM/) | SLAM mass terms |
| Optional forecast EAM | [ETH Zurich GPC EAM archive](https://gpc.ethz.ch/products/EAM/) | Archived component forecasts |

The GFZ directories organize analysis files by component and year. Download the years needed for the study period. The ETH archive organizes forecasts by component and forecast product.

## Daily CSV layout

Prepare one continuous, date-sorted row per UTC day and merge the component records by date. The minimum loader-compatible columns are:

```text
date,xpole,ypole,aam_x,aam_y,aam_vx,aam_vy,oam_x,oam_y,oam_vx,oam_vy,ham_x,ham_y,ham_vx,ham_vy,slam_x,slam_y
```

- `date`: ISO calendar date, `YYYY-MM-DD`.
- `xpole`, `ypole`: polar motion in arcseconds.
- `*_x`, `*_y`: dimensionless mass excitation terms.
- `*_vx`, `*_vy`: dimensionless motion excitation terms.
- Values must be numeric and finite, with no duplicated or missing dates.
- Convert subdaily products to a consistent daily epoch or daily statistic before the date merge; record the selected rule with the experiment.

The loader converts polar motion from arcseconds to milliarcseconds. For AAM, OAM, and HAM it adds the mass and motion terms; SLAM contributes its mass terms. The resulting excitation components are converted from radians to milliarcseconds internally.

[`example_eop_eam.csv`](example_eop_eam.csv) demonstrates the required structure with three synthetic rows. It is a format example only and must not be used as scientific input.

## Optional forecast CSV

Forecast-aware experiments additionally require an issue-date/lead table. Each complete issue date should contain leads 0–13 with columns `issue_date`, `date`, `lead_day`, followed by the same AAM/OAM/HAM/SLAM component fields. The relationship must satisfy `date = issue_date + lead_day`.

The original preparation utilities updated an existing combined table and were not a verified end-to-end reconstruction from all upstream files. They are therefore not presented as a reproducible download pipeline in this compact repository.
