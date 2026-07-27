# Solar Radiation Imputation Report

Generated with fixed random seed `20260728`.

## Inputs and units

Six hourly anomaly-marked station files were checked against their corresponding
prewashed source files. `Rso` was detected from its observed range as
`MJ m-2 h-1` and converted using `Rso_wm2 = Rso × 277.7778`.  Astronomical
day/night status used station coordinates, the `America/Chicago` timezone, and a
30-minute sunrise/sunset buffer.

| station | rows | missing_Srad | Rso_min | Rso_max | Rso_detected_unit | raw_timestamp_match | raw_Srad_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CB01 | 15597 | 0 | 0.000 | 3.679 | MJ m-2 h-1 | True | True |
| CB04 | 98907 | 232 | 0.000 | 3.968 | MJ m-2 h-1 | True | True |
| CB06 | 74596 | 4 | 0.000 | 3.931 | MJ m-2 h-1 | True | True |
| FD02 | 99914 | 1 | 0.000 | 3.958 | MJ m-2 h-1 | True | True |
| FD03 | 86211 | 0 | 0.000 | 3.976 | MJ m-2 h-1 | True | True |
| WC05 | 98902 | 2020 | 0.000 | 3.965 | MJ m-2 h-1 | True | True |

## Imputation outcomes

Observed `Srad` is retained in both `Srad` and `Srad_original`; only
`Srad_filled` contains a replacement. Weather-explained low radiation is
retained. Long-zero, sudden-drop, and spike records are replaced only when at
least two calibrated peer predictions agree and the observation is clearly
deviant; all others remain unchanged and are marked for manual review.

| station | imputed | corrected | unfilled | manual_review_retained | clipped |
| --- | --- | --- | --- | --- | --- |
| CB01 | 0 | 100 | 0 | 14 | 0 |
| CB04 | 169 | 690 | 63 | 679 | 0 |
| CB06 | 0 | 434 | 4 | 31 | 0 |
| FD02 | 1 | 463 | 0 | 175 | 0 |
| FD03 | 0 | 390 | 0 | 62 | 0 |
| WC05 | 1568 | 558 | 452 | 133 | 0 |

![Imputation outcomes](figures/imputation_outcomes_by_station.png)

## Validation

Normal records were masked in reproducible continuous segments of 1 hour, 2–6
hours, 7–24 hours, and 48–96 hours. Calibration and climatology fitting excluded
all target-station validation timestamps. The table below aggregates all gap
lengths and stations.

| method | n_expected | n_predicted | coverage | MAE | RMSE | bias | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi_peer | 5454 | 4957 | 0.909 | 26.080 | 66.932 | 2.317 | 0.955 |
| single_peer | 5454 | 4992 | 0.915 | 26.932 | 68.850 | 1.455 | 0.952 |
| month_hour_climatology | 5454 | 4987 | 0.914 | 58.414 | 121.731 | 8.714 | 0.851 |
| clear_sky_index_interpolation | 5454 | 4993 | 0.915 | 140.640 | 247.057 | 27.871 | 0.387 |
| linear_Srad_interpolation | 5454 | 5454 | 1.000 | 274.590 | 388.352 | 10.131 | -0.595 |

Best overall RMSE: **multi_peer** (66.932 W/m²).

![Validation RMSE](figures/validation_rmse_by_method.png)

Detailed station-by-gap metrics are in `validation_metrics.csv`, and all masked
predictions are in `validation_predictions.csv`.

## Manual review retained

| station | reason | records |
| --- | --- | --- |
| CB01 | long_zero_run_retained_insufficient_peer_consensus | 3 |
| CB01 | long_zero_run_retained_not_clearly_deviant | 1 |
| CB01 | sudden_drop_retained_insufficient_peer_consensus | 5 |
| CB01 | sudden_spike_retained_insufficient_peer_consensus | 2 |
| CB01 | sudden_spike_retained_not_clearly_deviant | 3 |
| CB04 | long_zero_run_retained_insufficient_peer_consensus | 366 |
| CB04 | long_zero_run_retained_not_clearly_deviant | 192 |
| CB04 | sudden_drop_retained_insufficient_peer_consensus | 81 |
| CB04 | sudden_drop_retained_not_clearly_deviant | 3 |
| CB04 | sudden_spike_retained_insufficient_peer_consensus | 1 |
| CB04 | sudden_spike_retained_not_clearly_deviant | 36 |
| CB06 | long_zero_run_retained_insufficient_peer_consensus | 3 |
| CB06 | long_zero_run_retained_not_clearly_deviant | 1 |
| CB06 | sudden_drop_retained_insufficient_peer_consensus | 4 |
| CB06 | sudden_drop_retained_not_clearly_deviant | 3 |
| CB06 | sudden_spike_retained_insufficient_peer_consensus | 3 |
| CB06 | sudden_spike_retained_not_clearly_deviant | 17 |
| FD02 | long_zero_run_retained_insufficient_peer_consensus | 68 |
| FD02 | long_zero_run_retained_not_clearly_deviant | 46 |
| FD02 | sudden_drop_retained_insufficient_peer_consensus | 22 |
| FD02 | sudden_drop_retained_not_clearly_deviant | 2 |
| FD02 | sudden_spike_retained_insufficient_peer_consensus | 10 |
| FD02 | sudden_spike_retained_not_clearly_deviant | 27 |
| FD03 | long_zero_run_retained_insufficient_peer_consensus | 3 |
| FD03 | long_zero_run_retained_not_clearly_deviant | 1 |
| FD03 | sudden_drop_retained_insufficient_peer_consensus | 14 |
| FD03 | sudden_drop_retained_not_clearly_deviant | 2 |
| FD03 | sudden_spike_retained_insufficient_peer_consensus | 10 |
| FD03 | sudden_spike_retained_not_clearly_deviant | 32 |
| WC05 | long_zero_run_retained_insufficient_peer_consensus | 42 |
| WC05 | sudden_drop_retained_insufficient_peer_consensus | 47 |
| WC05 | sudden_drop_retained_not_clearly_deviant | 1 |
| WC05 | sudden_spike_retained_insufficient_peer_consensus | 5 |
| WC05 | sudden_spike_retained_not_clearly_deviant | 38 |

## Integrity checks

| station | input_rows | output_rows | row_count_match | timestamp_match | Srad_original_preserved | normal_unmarked_records | normal_unmarked_unchanged | filled_outside_physical_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CB01 | 15597 | 15597 | True | True | True | 15074 | True | 0 |
| CB04 | 98907 | 98907 | True | True | True | 94216 | True | 0 |
| CB06 | 74596 | 74596 | True | True | True | 72238 | True | 0 |
| FD02 | 99914 | 99914 | True | True | True | 95776 | True | 0 |
| FD03 | 86211 | 86211 | True | True | True | 84028 | True | 0 |
| WC05 | 98902 | 98902 | True | True | True | 93301 | True | 0 |

## Method notes

- Monthly robust affine relationships map peer-station clear-sky index to the
  target station; global relationships are used only when a month lacks enough
  paired normal records.
- Priority is multi-peer, single-peer, clear-sky-index interpolation for gaps
  no longer than three hours, then month-hour climatology.
- No raw-`Srad` interpolation is used in production imputation, and unreliable
  values remain `NA`.
- Generated values are constrained to 0–1300 W/m²; clipping is recorded in
  `Srad_value_was_clipped`.
