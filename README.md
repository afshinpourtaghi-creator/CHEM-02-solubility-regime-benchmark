# CHEM-02 — Reproducible numerical analyses (v1.0.0 MIT source-release candidate)

## Source data
Download **MixtureSolDB v3** from https://doi.org/10.5281/zenodo.18660057 as `MixtureSolDB.csv`.
Check the file's MD5 is `597e31685fe36c78d568f3d37b6e1336`. The raw dataset is **not** redistributed here pending confirmation of its reuse license.

## Python environment
Python 3.13.5; numpy 2.3.5; pandas 2.2.3; rdkit 2025.9.4; lightgbm 4.6.0; scipy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8. `requirements.txt` pins the analysis dependencies.

## First run from a fresh checkout

```bash
python -m pip install -r requirements.txt
python run_reproduction.py --raw /path/to/MixtureSolDB.csv --out ./local_run --stage prepare
python smoke_test.py --metadata ./local_run/harmonized_metadata.csv
python run_reproduction.py --raw /path/to/MixtureSolDB.csv --out ./local_run --stage raw_audit
python verify_aggregates.py
```

## Rebuild primary comparisons (computationally expensive)

```bash
python run_reproduction.py --raw /path/to/MixtureSolDB.csv --out ./local_run --stage phase13
python run_reproduction.py --raw /path/to/MixtureSolDB.csv --out ./local_run --stage phase12
python run_reproduction.py --raw /path/to/MixtureSolDB.csv --out ./local_run --stage phase14
python run_reproduction.py --raw /path/to/MixtureSolDB.csv --out ./local_run --stage classical
```

The phase-13 invocation reuses the exact original source-disjoint and joint-chemical-OOD seeds 1001–1003 and each model's original nested hyperparameter selection. These runs can take hours and should not be described as re-executed in this package unless their logs/results are actually obtained. **Phase 14 is exploratory with unequal computational budgets.** In phase 16, the classical model is given *measured endpoint and three interior calibration values from the test curve*; it cannot be compared directly to zero-shot ML.

## What has been tested in this release
- Original raw CSV checksum validated and preprocessing rerun end-to-end: **172,880 records** reproduced; all historical columns verified on 172,880 rows (numeric tolerances ≤1e-12).
- Phase-16 classical analysis rerun from newly generated processed data; the entire local three-point summary agrees **exactly** with the archived historical CSV.
- `verify_aggregates.py` passed on bundled aggregate results; this checks previously saved values, **not** ML retraining.
- `smoke_test.py` passed (RDKit descriptors + LightGBM training on a small subset; not a scientific holdout).
- Python source files pass syntax compilation. A clean-machine end-to-end retraining of every costly phase-12/13/14 model has **not** been carried out in this release.

## Scientific restrictions
- The external Montenegro et al. (2026) `Table S3` numeric dataset has **not** been scored; do not report quantitative external validation metrics.
- Archived experiments are retrospective; some experimental systems have been visited during earlier development.
- Do **not** publish the raw CSV/cleaned row-level derivative until reuse rights are resolved.
- Data-source MD5 and output hashes should be retained with any public tagged release; the public repository / Zenodo DOI remains pending.

## v24.1 author handoff (18 October 2026)
- The `raw_audit` stage uses a different implementation to compare raw-to-clean results with the current pipeline. To compare against historical independently retained metadata, pass `--reference-metadata /path/to/historical.csv`. Without this argument, the label *independent historical replication* must not be used.
- An exploratory full retrain of Phase 12 / seed 901 was attempted in the current execution environment. It completed part of inner hyperparameter selection but exceeded the runtime limit prior to outer-model testing. No new full-retrain MAE is claimed. `run_reproduction.py --stage verify_archive --out ./local_run` now works without `--raw`.
- Script execution and aggregate reproducibility do not constitute external, prospectively independent validation. Authors must confirm their responsibility for all claims before submitting.

## Public-release status (18 October 2026)

The files in this directory are READY FOR AUTHOR REVIEW AND DEPOSITION but have **not** been published to GitHub or Zenodo by the assistant. No DOI has been assigned. Publication requires the author's account access, approval of a software reuse license, and a public GitHub Release (or a manual Zenodo software deposit). Do not substitute this directory's version label for a repository or DOI.

The primary publication integrity limit is unchanged: the 2026-10-18 clean software smoke test, archived aggregate checks, and independent data/classical checks DO NOT independently reproduce all computationally expensive nested ML fits. The release includes executable scripts and complete instructions but a clean-environment full retrain still merits verification before submission.

**No raw, harmonized row-level source data, or third-party training CSV is authorized for redistribution**; obtain MixtureSolDB from its original DOI and check the MD5.


## Software licence and data rights

The original CHEM-02 research code in this repository is licensed under
**MIT** (see `LICENSE`, copyright 2026 Afshin Pourtaghi). This does **not**
relicense any external data, including MixtureSolDB. The data are deliberately
excluded from this code repository. Users must download the dataset from its
original archival DOI and abide by its own terms.

This release candidate is **not yet publicly deposited**; replace this status
only after publishing the GitHub release and verifying the Zenodo DOI. Full
independent retraining of computationally intensive ML experiments has not yet
been completed.
