# Figure source data

This directory contains the author-generated source data and notes for manuscript Figures 1–7. It contains no executable plotting code.

Most tables are plain CSV. The two Figure 2 tables larger than 1 MB are stored as `.csv.gz`; pandas reads them directly, or they can be decompressed with `gzip -dk FILE.csv.gz`.
