"""Visualize missing-data patterns for the TxSON 33-station cleanup outputs.

Examples:
    python data_visualization/visualize_txson_33_gaps.py
    python data_visualization/visualize_txson_33_gaps.py --station CB01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PARAM_ORDER = [
    "Ppt",
    "Tair",
    "Srad",
    "RH",
    "Wind speed",
    "Wind direction",
    "SWC_5",
    "SWC_10",
    "SWC_20",
    "SWC_50",
    "T_5",
    "T_10",
    "T_20",
    "T_50",
    "Flag",
]

GAP_BUCKETS = [
    ("Short <24h", 0, 24, "Already filled by Shortgaps.py"),
    ("Medium 24h-7d", 24, 168, "Use Mediumgaps.py / SARIMAX"),
    ("Long 7-30d", 168, 720, "Use Longgaps.py / XGBoost"),
    ("Very long >=30d", 720, None, "Use VeryLongGaps.py / donor stations"),
]


def site_codes(cleaned_dir: Path) -> list[str]:
    return sorted(
        path.name.removeprefix("Station").removesuffix("_cleaned_data.csv")
        for path in cleaned_dir.glob("Station*_cleaned_data.csv")
        if not path.name.removeprefix("Station").startswith(tuple("123456"))
    )


def read_cleaned(cleaned_dir: Path, site: str) -> pd.DataFrame:
    return pd.read_csv(
        cleaned_dir / f"Station{site}_cleaned_data.csv",
        index_col=0,
        parse_dates=True,
    )


def read_short_output(output_dir: Path, site: str) -> pd.DataFrame | None:
    path = output_dir / f"Station{site}_filled_shortgaps.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def read_missing(missing_dir: Path, site: str) -> pd.DataFrame:
    path = missing_dir / f"Station{site}_missing_data.csv"
    return pd.read_csv(path, parse_dates=["Start Timestamp", "End Timestamp"])


def ordered_params(columns) -> list[str]:
    known = [param for param in PARAM_ORDER if param in columns]
    extra = sorted(set(columns) - set(known))
    return known + extra


def missing_percent_table(cleaned_dir: Path, sites: list[str], use_short: bool, output_dir: Path) -> pd.DataFrame:
    rows = []
    all_params = []
    for site in sites:
        df = read_short_output(output_dir, site) if use_short else None
        if df is None:
            df = read_cleaned(cleaned_dir, site)
        params = ordered_params(df.columns)
        all_params.extend(params)
        pct = (df[params].isna().mean() * 100).round(3)
        rows.append(pd.Series(pct, name=site))

    params = ordered_params(sorted(set(all_params)))
    return pd.DataFrame(rows).reindex(columns=params)


def gap_bucket_table(missing_dir: Path, sites: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_rows = []
    detail_rows = []
    for site in sites:
        missing = read_missing(missing_dir, site)
        station_row = {"site": site}
        if missing.empty:
            for bucket, *_ in GAP_BUCKETS:
                station_row[bucket] = 0
            station_rows.append(station_row)
            continue

        missing["Number Missing"] = pd.to_numeric(missing["Number Missing"], errors="coerce").fillna(0)
        for bucket, min_hours, max_hours, method in GAP_BUCKETS:
            mask = missing["Number Missing"] >= min_hours
            if max_hours is not None:
                mask &= missing["Number Missing"] < max_hours
            hours = int(missing.loc[mask, "Number Missing"].sum())
            station_row[bucket] = hours
            for _, row in missing.loc[mask].iterrows():
                detail_rows.append({
                    "site": site,
                    "parameter": row["Parameter"],
                    "start": row["Start Timestamp"],
                    "end": row["End Timestamp"],
                    "hours": int(row["Number Missing"]),
                    "bucket": bucket,
                    "recommended_fill": method,
                })
        station_rows.append(station_row)
    return pd.DataFrame(station_rows), pd.DataFrame(detail_rows)


def station_metadata(cleaned_dir: Path, missing_dir: Path, sites: list[str]) -> pd.DataFrame:
    rows = []
    for site in sites:
        df = read_cleaned(cleaned_dir, site)
        missing = read_missing(missing_dir, site)
        met_cols = {"Tair", "Srad", "RH", "Wind speed", "Wind direction"}
        rows.append({
            "site": site,
            "rows": len(df),
            "cols": len(df.columns),
            "start": df.index.min(),
            "end": df.index.max(),
            "has_met_columns": bool(met_cols & set(df.columns)),
            "missing_pct_stage0": round(float(df.isna().sum().sum() / df.size * 100), 3),
            "max_gap_hours": int(pd.to_numeric(missing.get("Number Missing", pd.Series(dtype=float)), errors="coerce").max()) if len(missing) else 0,
        })
    return pd.DataFrame(rows)


def write_overview(cleaned_dir: Path, missing_dir: Path, output_dir: Path, report_dir: Path) -> Path:
    sites = site_codes(cleaned_dir)
    if not sites:
        raise SystemExit(f"No new station cleaned files found in {cleaned_dir}")

    stage0_pct = missing_percent_table(cleaned_dir, sites, use_short=False, output_dir=output_dir)
    bucket_hours, detail = gap_bucket_table(missing_dir, sites)
    meta = station_metadata(cleaned_dir, missing_dir, sites)

    report_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(report_dir / "txson_33_station_missing_metadata.csv", index=False)
    detail.to_csv(report_dir / "txson_33_gap_details.csv", index=False)

    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.10,
        specs=[[{"type": "heatmap"}], [{"type": "bar"}], [{"type": "table"}]],
        row_heights=[0.40, 0.32, 0.28],
        subplot_titles=(
            "Stage 0 missing percentage by station and parameter",
            "Missing hours by gap length category",
            "Station-level cleanup summary",
        ),
    )

    unavailable = stage0_pct.isna()
    unavailable_z = unavailable.where(unavailable, other=None).astype(float)
    unavailable_text = unavailable.where(unavailable, other=None).replace({True: "Not available"})
    fig.add_trace(
        go.Heatmap(
            z=unavailable_z.values,
            x=stage0_pct.columns,
            y=stage0_pct.index,
            text=unavailable_text.values,
            colorscale=[[0, "#d9dee7"], [1, "#d9dee7"]],
            showscale=False,
            hovertemplate="Station=%{y}<br>Parameter=%{x}<br>%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=stage0_pct.values,
            x=stage0_pct.columns,
            y=stage0_pct.index,
            colorscale="Viridis",
            zmin=0,
            zmax=100,
            colorbar={"title": "% missing", "x": 1.02, "len": 0.35, "y": 0.87},
            hovertemplate="Station=%{y}<br>Parameter=%{x}<br>Missing=%{z:.3f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for bucket, *_ in GAP_BUCKETS:
        fig.add_trace(
            go.Bar(
                x=bucket_hours["site"],
                y=bucket_hours[bucket],
                name=bucket,
                hovertemplate="Station=%{x}<br>Missing hours=%{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    table = meta.copy()
    table["start"] = table["start"].astype(str)
    table["end"] = table["end"].astype(str)
    fig.add_trace(
        go.Table(
            header={"values": list(table.columns), "align": "left"},
            cells={"values": [table[col] for col in table.columns], "align": "left"},
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title="TxSON 33-Station Missing Data Overview",
        height=1500,
        barmode="stack",
        template="plotly_white",
        margin={"l": 70, "r": 170, "t": 90, "b": 50},
        legend={
            "title": {"text": "Gap length"},
            "x": 1.04,
            "y": 0.47,
            "xanchor": "left",
            "yanchor": "middle",
        },
    )
    fig.update_yaxes(tickmode="array", tickvals=sites, ticktext=sites, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)

    out = report_dir / "txson_33_gap_overview.html"
    fig.write_html(out, include_plotlyjs=True)
    return out


def write_station_detail(cleaned_dir: Path, output_dir: Path, report_dir: Path, station: str) -> Path:
    df = read_short_output(output_dir, station)
    source_label = "after Shortgaps.py"
    if df is None:
        df = read_cleaned(cleaned_dir, station)
        source_label = "Stage 0 cleaned"

    params = ordered_params(df.columns)
    missing_mask = df[params].isna().astype(int).T
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.12,
        specs=[[{"type": "heatmap"}], [{"type": "bar"}]],
        row_heights=[0.70, 0.30],
        subplot_titles=(
            f"{station}: missing timeline ({source_label})",
            f"{station}: missing percentage by parameter",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=missing_mask.values,
            x=df.index,
            y=missing_mask.index,
            colorscale=[[0, "#f7fbff"], [1, "#b2182b"]],
            showscale=False,
            hovertemplate="Time=%{x}<br>Parameter=%{y}<br>Missing=%{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    pct = (df[params].isna().mean() * 100).round(3)
    fig.add_trace(
        go.Bar(
            x=pct.index,
            y=pct.values,
            hovertemplate="Parameter=%{x}<br>Missing=%{y:.3f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"TxSON Station {station} Missing Data Detail",
        height=900,
        template="plotly_white",
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    fig.update_xaxes(tickangle=45, row=2, col=1)

    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"txson_{station}_missing_timeline.html"
    fig.write_html(out, include_plotlyjs=True)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Create TxSON 33-station missing-data visualizations.")
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path("data-cleanup/imputation_pipeline"),
        help="Directory containing cleaned_data, missing_data, and output.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data_visualization/txson_33_gap_reports"),
        help="Directory where HTML and CSV reports are written.",
    )
    parser.add_argument("--station", type=str, default=None, help="Optional site code for a station detail HTML.")
    return parser.parse_args()


def main():
    args = parse_args()
    cleaned_dir = args.pipeline_dir / "cleaned_data"
    missing_dir = args.pipeline_dir / "missing_data"
    output_dir = args.pipeline_dir / "output"

    overview = write_overview(cleaned_dir, missing_dir, output_dir, args.report_dir)
    print(f"Wrote overview: {overview}")

    if args.station:
        detail = write_station_detail(cleaned_dir, output_dir, args.report_dir, args.station)
        print(f"Wrote station detail: {detail}")


if __name__ == "__main__":
    main()
