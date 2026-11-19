#!/usr/bin/env python3
"""Command-line runner for the held-out station-year CNN experiments.

Thin wrapper over ``src`` so experiments run outside notebooks (e.g. overnight
on a UT CS Linux machine, CPU or GPU). Resumable, seeded, with CSV + checkpoint
logging. GPU is auto-detected and **not** assumed.

Examples
--------
  python main.py --horizon 72 --features swc,existing_rain,lagged_rain
  python main.py --horizon 48 --features cumulative_rain --stations FD02 --test-years 2020
  # overnight, detached:
  nohup python -u main.py --horizon 72 --features swc > experiment.log 2>&1 < /dev/null &
"""
from __future__ import annotations
import argparse, os, sys, json, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C
from src.data_processing import load_dataframe, build_station_year_store
from src.evaluation import run_fold


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=72)
    ap.add_argument("--features", default="swc,existing_rain,lagged_rain,cumulative_rain,temperature,combined_met",
                    help="comma list of groups: " + ", ".join(C.FEATURE_GROUPS))
    ap.add_argument("--stations", default=",".join(C.SELECTED_STATIONS))
    ap.add_argument("--test-years", default="2023,2020")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", choices=["small", "larger", "configurable"], default="larger")
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--filters", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--output-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    from src.training import get_device
    import torch
    device = get_device(args.device)
    print(f"[env] device={device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    res_csv = os.path.join(args.output_dir, "experiments.csv")
    loss_csv = os.path.join(args.output_dir, "loss_histories.csv")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    stations = args.stations.split(",")
    test_years = [int(y) for y in args.test_years.split(",")]
    groups = [g.strip() for g in args.features.split(",")]
    for g in groups:
        if g not in C.FEATURE_GROUPS:
            sys.exit(f"unknown feature group {g!r}; choices: {list(C.FEATURE_GROUPS)}")

    store = build_station_year_store(load_dataframe(), stations)

    rows, done, loss_rows = [], set(), []
    if os.path.exists(res_csv) and not args.no_resume:
        prev = pd.read_csv(res_csv)
        rows = prev.to_dict("records")
        for _, r in prev.iterrows():
            done.add((str(r["features"]), str(r["test_station"]),
                      str(r["test_year"]), str(r["horizon"])))
    if os.path.exists(loss_csv) and not args.no_resume:
        loss_rows = pd.read_csv(loss_csv).to_dict("records")

    model_kw = dict(kernel_size=args.kernel_size, n_layers=args.layers,
                    base_filters=args.filters, dropout=args.dropout)
    for st in stations:
        for yr in test_years:
            if (st, yr) not in store:
                continue
            for grp in groups:
                fcols = C.feature_columns(grp)
                key = ("|".join(fcols), st, str(yr), str(args.horizon))
                if key in done:
                    print(f"[skip] {grp} {st}-{yr} H{args.horizon}", flush=True)
                    continue
                try:
                    row, hist = run_fold(
                        store, stations, st, yr, fcols,
                        window=args.window, horizon=args.horizon, arch=args.arch,
                        seed=args.seed, epochs=args.epochs, patience=args.patience,
                        batch=args.batch, lr=args.lr, device=device, **model_kw)
                except Exception as e:  # keep going; log and move on
                    print(f"[error] {grp} {st}-{yr} H{args.horizon}: {e}", flush=True)
                    continue
                row["group"] = grp
                rows.append(row)
                pd.DataFrame(rows).to_csv(res_csv, index=False)
                for ep, (tl, vl) in enumerate(zip(hist["train_loss"], hist["val_loss"]), 1):
                    loss_rows.append(dict(group=grp, test_station=st, test_year=yr,
                                          horizon=args.horizon, epoch=ep,
                                          train_loss=tl, val_loss=vl))
                pd.DataFrame(loss_rows).to_csv(loss_csv, index=False)
                print(f"[ok] {grp:16s} {st}-{yr} H{args.horizon}: "
                      f"RMSE={row['RMSE']:.4f} vsPers={row['impr_vs_persist_pct']:+.1f}% "
                      f"({row['train_seconds']:.0f}s)", flush=True)

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[done] results -> {res_csv}", flush=True)


if __name__ == "__main__":
    main()
