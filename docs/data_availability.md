# Data availability

This repository includes the source code, trained paper checkpoint, model configuration, evaluation settings, author-generated figure source data, static previews, and the author-generated UL-gap regression intermediate.

Checkpoint inference additionally requires local prepared copies of `eop_data_xy_EAM.csv` and `eam14forecast_daily.csv`. These files combine products from IERS, GFZ, and ETH Zurich and are not redistributed or relicensed by this repository. Their expected hashes, schemas, snapshots, and upstream access points are documented in `data/README.md`.

The repository can therefore be checked at two levels:

1. `python run.py test` strictly loads the checkpoint and completes a deterministic CPU forward pass without restricted data.
2. `python run.py predict ...` performs scientific checkpoint inference when authorized prepared inputs are supplied locally.

The software and article DOI fields should be updated after archival deposit and manuscript publication.
