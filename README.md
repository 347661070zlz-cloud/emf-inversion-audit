# Rethinking the Reported Accuracy of Smartphone-Based Electric-Field Inversion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23758133.svg)](https://doi.org/10.5281/zenodo.23758133)

Code and data pipeline for the paper:

> Lianzhong Zheng and Juan Pan, "Rethinking the Reported Accuracy of Smartphone-Based
> Electric-Field Inversion for Crowdsourced EMF Monitoring," *IEEE Access*
> (under review).

Starting from a public smartphone measurement campaign, we reproduce a
gradient-boosting electric-field inversion model and then vary three factors that
are usually fixed by convention: the evaluation protocol, the feature set, and the
device calibration.

## What the code shows

| Finding | Script |
|---|---|
| The evaluation protocol moves the error estimate by 0.39 dB — more than twice the spread among six regressors | `protocol_experiment.py` |
| Adding the reported EARFCN removes a band-dependent bias of up to 0.5 dB and lowers MAE by 15–31 % depending on how the label is defined | `band_model_experiment.py`, `label_confound_experiment.py` |
| Input-space recalibration of the same handset is absorbed by the tree ensemble; only cross-device offsets matter, and 100 labelled samples recover 65–75 % of the loss | `device_shift_experiment.py` |
| Pooled R² hides the effect that MAE reveals, and per-fold R² is meaningless when a fold contains a single band | `protocol_experiment.py` |

## Data

The dataset is **not** redistributed here. It is the supporting dataset of the
original study and is available from figshare under CC-BY 4.0:

* DOI: <https://doi.org/10.6084/m9.figshare.25034537.v1>

Download it and unpack the archive into a folder named `ds01/` at the repository
root, so that the following paths exist:

```
ds01/samples/                  # released sample table (46,306 rows, 23 features)
ds01/data for table 1/
ds01/data for figure 4/
ds01/data for figure 6/
ds01/readme.txt
```

`ds01/` is listed in `.gitignore` and is never committed.

## Quick start

```bash
git clone https://github.com/347661070zlz-cloud/emf-inversion-audit.git
cd emf-inversion-audit

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place the dataset in ds01/ (see above)

python run_all.py                  # reproduction of the original study
```

All scripts write to `outputs/` and are idempotent: re-running overwrites the
previous results.

## Which script produces what

| Paper item | Command | Output |
|---|---|---|
| Table I — reproduction of the baseline | `python run_all.py` | `outputs/tables/table5_gbdt.csv` |
| Table II — effect of the protocol | `python protocol_experiment.py` | `outputs/protocol/protocol_aggregate.csv` |
| Table III — protocol × feature design | `python band_model_experiment.py` | `outputs/band_model/band_model_summary_random_stpbcv.csv` |
| Table IV — within-band R² | `python band_model_experiment.py` | same file, columns `b*_R2` |
| Table V — model comparison | `python compare_models.py` | `outputs/compare/model_summary.csv` |
| Table VI — inter-device offset | `python device_shift_experiment.py` | `outputs/protocol/device_shift.csv` |
| Fig. 1 — study area and folds | `python make_figures.py` | `outputs/figures/fig3_study_area.pdf` |
| Fig. 2 — test MAE by protocol | `python make_figures.py` | `outputs/figures/fig1_protocols.pdf` |
| Fig. 3 — predicted vs. measured | `python make_figures_diagnostics.py` | `outputs/figures/fig_pred_scatter.pdf` |
| Fig. 4 — field-strength CDF | `python make_figures_diagnostics.py` | `outputs/figures/fig_field_cdf.pdf` |
| Fig. 5 — protocol × feature | `python make_figures.py` | `outputs/figures/fig2_protocol_feature.pdf` |
| Fig. 6 — residual distribution per band | `python make_figures_diagnostics.py` | `outputs/figures/fig_residual_band.pdf` |
| Fig. 7 — cost of an inter-device offset | `python make_figures.py` | `outputs/figures/fig4_device.pdf` |

Additional analyses referenced in the text:

| Analysis | Command | Output |
|---|---|---|
| Three label definitions (15.4 / 22.6 / 31.5 % gain) | `python label_confound_experiment.py` | `outputs/label_confound/summary.csv` |
| Campaign-level hold-out check (1.33 dB) | `python campaign_logo_check.py` | `outputs/protocol/campaign_logo_summary.json` |
| Embargo sensitivity (0/20/50 m, 0/60/300 s) | `python protocol_experiment.py --schemes stpbcv:0:0 stpbcv:20:60 stpbcv:50:300` | `outputs/protocol/protocol_aggregate.csv` |
| Recovery of the unreported EWMA coefficient | `python sensitivity.py` | `outputs/tables/alpha_sensitivity.csv` |
| Single-fit timing (Table V) | `python benchmark_speed.py` | `outputs/compare/timing.csv` |

`make_figures_diagnostics.py` caches the leave-one-group-out predictions in
`outputs/tables/predictions_logo.csv`; delete that file to recompute them
(about 2–3 min on 8 cores).

## Reproducing the full pipeline

```bash
python run_all.py
python protocol_experiment.py
python band_model_experiment.py --seeds 8 --jobs 8
python label_confound_experiment.py
python device_shift_experiment.py
python deep_calibration_experiment.py
python compare_models.py
python campaign_logo_check.py
python make_figures_diagnostics.py
python make_figures.py
```

Runtime is roughly 20–40 min on 8 CPU cores; `band_model_experiment.py` and
`protocol_experiment.py` dominate.

## Requirements

Python 3.9 or newer. Dependencies are listed in `requirements.txt`:

```
numpy, pandas, scikit-learn, scipy, matplotlib, joblib
```

No GPU is needed, and no step requires network access once the dataset is in
place.

## Implementation notes

* The dataset contains 46,306 samples from five measurement campaigns over four
  LTE downlink bands; every sample carries a GPS fix, which is what makes the
  blocked cross-validation schemes possible.
* Coordinates are matched to the sample table by nearest timestamp within 3 s
  (`src/data_io.py`); the result is cached in
  `outputs/cv/samples_with_coords.csv`.
* Random seeds are fixed (`RANDOM_STATE = 42` in `src/config.py`); the eight-seed
  protocol sweep uses seeds 0–7.
* `src/env_check.py` runs at import time and prints an actionable message if a
  dependency is missing.

## Citation

If you use this code, please cite the paper above and the original dataset:

> S. Zeng, W. Chen, Y. Ji, L. Yan, and X. Zhao, "Measurement and calibration of
> EMF: A study using phone and GBDT for mobile communication signals," *Radio
> Science*, vol. 59, no. 2, p. e2023RS007890, 2024,
> doi: 10.1029/2023RS007890.

## License

The code in this repository is released under the MIT license (see `LICENSE`).
The dataset is distributed under its own terms; see the figshare record.
