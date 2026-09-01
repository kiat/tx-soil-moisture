from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import HBox, VBox, interactive_output


SOIL_MOISTURE_COLS = ["SWC_5", "SWC_10", "SWC_20", "SWC_50"]
SOIL_TEMPERATURE_COLS = ["T_5", "T_10", "T_20", "T_50"]
MET_COLS = ["Ppt", "Tair", "RH", "Wind speed", "Wind direction", "Srad"]

DEPTH_COLORS = {
    "SWC_5": "#4682B4",
    "SWC_10": "#FFA500",
    "SWC_20": "#3CB371",
    "SWC_50": "#DC143C",
    "T_5": "#4682B4",
    "T_10": "#FFA500",
    "T_20": "#3CB371",
    "T_50": "#DC143C",
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "data-cleanup").exists() and (parent / "datasets").exists():
            return parent
    return here.parents[1]


def pipeline_dir() -> Path:
    return repo_root() / "data-cleanup" / "imputation_pipeline"


def discover_stations() -> list[str]:
    base = pipeline_dir()
    soil_stations = {
        path.name.removeprefix("Station").removesuffix("_filled_final.csv")
        for path in (base / "output").glob("Station*_filled_final.csv")
    }
    met_stations = {
        path.name.removeprefix("Station").removesuffix("_met_filled_complete.csv")
        for path in (base / "met_output").glob("Station*_met_filled_complete.csv")
    }
    if not soil_stations and not met_stations:
        raise FileNotFoundError(
            "No final Soil or MET station files were found. Run the final Soil and MET stages first."
        )
    if soil_stations != met_stations:
        missing_soil = sorted(met_stations - soil_stations)
        missing_met = sorted(soil_stations - met_stations)
        raise FileNotFoundError(
            "Final visualization requires paired Soil and MET outputs for every station. "
            f"Missing Soil: {missing_soil or 'none'}; missing MET: {missing_met or 'none'}."
        )
    return sorted(soil_stations)


def read_time_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[df.index.notna()]
    return df.loc[~df.index.duplicated(keep="last")].sort_index()


def load_station_data(station_id: str) -> pd.DataFrame:
    """Merge the final Soil and final MET products by timestamp."""
    base = pipeline_dir()
    soil_path = base / "output" / f"Station{station_id}_filled_final.csv"
    met_path = base / "met_output" / f"Station{station_id}_met_filled_complete.csv"
    missing = [str(path) for path in (soil_path, met_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Final visualization inputs are missing for Station{station_id}: "
            + ", ".join(missing)
        )

    combined = read_time_series(soil_path).drop(columns=MET_COLS, errors="ignore")
    met = read_time_series(met_path)
    missing_met_columns = [column for column in MET_COLS if column not in met.columns]
    if missing_met_columns:
        raise ValueError(
            f"Final MET file for Station{station_id} is missing columns: "
            + ", ".join(missing_met_columns)
        )
    merged_index = combined.index.union(met.index).sort_values()
    combined = combined.reindex(merged_index)
    for column in MET_COLS:
        combined[column] = met[column].reindex(merged_index)
    return combined


def station_years(station_id: str) -> list[int]:
    df = load_station_data(station_id)
    return sorted(df.index.year.unique().tolist())


def station_months(station_id: str, year: int) -> list[int]:
    df = load_station_data(station_id)
    months = sorted(df.loc[df.index.year == year].index.month.unique().tolist())
    return months or list(range(1, 13))


def filter_time(df: pd.DataFrame, year: int, month: int, time_resolution: str) -> pd.DataFrame:
    if time_resolution == "Monthly":
        return df[(df.index.year == year) & (df.index.month == month)]
    if time_resolution == "Yearly":
        return df[df.index.year == year]
    return df


def add_missing_markers(fig: go.Figure, df: pd.DataFrame, cols: list[str]) -> None:
    marker_y = []
    marker_x = []
    marker_text = []
    for offset, col in enumerate(cols):
        if col not in df.columns:
            continue
        missing_index = df.index[df[col].isna()]
        if len(missing_index) == 0:
            continue
        marker_x.extend(missing_index)
        marker_y.extend([offset] * len(missing_index))
        marker_text.extend([col] * len(missing_index))
    if marker_x:
        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                name="Missing timestamps",
                marker={"size": 5, "color": "#B22222", "opacity": 0.45, "symbol": "line-ns-open"},
                yaxis="y2",
                text=marker_text,
                hovertemplate="Missing %{text}<br>%{x}<extra></extra>",
            )
        )


def make_line_figure(
    df: pd.DataFrame,
    station_id: str,
    columns: list[str],
    title: str,
    yaxis_title: str,
    show_missing: bool = True,
) -> go.Figure:
    available = [col for col in columns if col in df.columns]
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data available for {title}")
        fig.add_annotation(
            text="No data for this station/time selection",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 18},
        )
        return fig
    if not available:
        fig = go.Figure()
        fig.update_layout(title=f"{station_id}: selected variables are not available")
        fig.add_annotation(
            text="Selected variables are not available for this station",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 18},
        )
        return fig

    fig = go.Figure()
    for col in available:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                connectgaps=False,
                line={"width": 1.5, "color": DEPTH_COLORS.get(col)},
                hovertemplate=f"{col}<br>%{{x}}<br>%{{y}}<extra></extra>",
            )
        )

    if show_missing:
        add_missing_markers(fig, df, available)
        fig.update_layout(
            yaxis2={
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "showticklabels": False,
                "range": [-0.5, max(len(available) - 0.5, 0.5)],
            }
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
        height=650,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def plot_soil_moisture(station_id: str, year: int, month: int, time_resolution: str):
    df = filter_time(load_station_data(station_id), year, month, time_resolution)
    period = period_label(year, month, time_resolution)
    fig = make_line_figure(
        df,
        station_id,
        SOIL_MOISTURE_COLS,
        f"Soil Moisture for Station {station_id} {period}",
        "Soil Moisture (m3/m3)",
    )
    fig.show()
    return fig


def plot_soil_temperature(station_id: str, year: int, month: int, time_resolution: str):
    df = filter_time(load_station_data(station_id), year, month, time_resolution)
    period = period_label(year, month, time_resolution)
    fig = make_line_figure(
        df,
        station_id,
        SOIL_TEMPERATURE_COLS,
        f"Soil Temperature for Station {station_id} {period}",
        "Temperature (C)",
    )
    fig.show()
    return fig


def plot_met_variable(station_id: str, variable: str, year: int, month: int, time_resolution: str):
    df = filter_time(load_station_data(station_id), year, month, time_resolution)
    period = period_label(year, month, time_resolution)
    fig = make_line_figure(
        df,
        station_id,
        [variable],
        f"{variable} for Station {station_id} {period}",
        variable,
    )
    fig.show()
    return fig


def period_label(year: int, month: int, time_resolution: str) -> str:
    if time_resolution == "Monthly":
        return f"in {year}-{month:02d}"
    if time_resolution == "Yearly":
        return f"in {year}"
    return "for full record"


def update_plot(plot_type, station_id, year, month, met_variable, time_resolution):
    if plot_type == "Soil Moisture":
        return plot_soil_moisture(station_id, year, month, time_resolution)
    if plot_type == "Soil Temperature":
        return plot_soil_temperature(station_id, year, month, time_resolution)
    if plot_type == "MET Variable":
        return plot_met_variable(station_id, met_variable, year, month, time_resolution)
    fig = go.Figure()
    fig.show()
    return fig


def display_dashboard():
    stations = discover_stations()
    if not stations:
        raise RuntimeError("No TxSON 33-station cleaned files found. Run datacleaning.py first.")

    first_station = stations[0]
    years = station_years(first_station)
    months = station_months(first_station, years[0])

    plot_type_selector = widgets.Dropdown(
        options=["Soil Moisture", "Soil Temperature", "MET Variable"],
        value="Soil Moisture",
        description="Plot Type:",
    )
    station_selector = widgets.Dropdown(
        options=stations,
        value=first_station,
        description="Station:",
    )
    year_selector = widgets.Dropdown(
        options=years,
        value=years[0],
        description="Year:",
    )
    month_selector = widgets.Dropdown(
        options=months,
        value=months[0],
        description="Month:",
    )
    met_variable_selector = widgets.Dropdown(
        options=MET_COLS,
        value="Ppt",
        description="MET Var:",
    )
    time_resolution_selector = widgets.Dropdown(
        options=["Monthly", "Yearly", "Total"],
        value="Monthly",
        description="Time Type:",
    )

    def update_year_options(*args):
        years_for_station = station_years(station_selector.value)
        year_selector.options = years_for_station
        if year_selector.value not in years_for_station:
            year_selector.value = years_for_station[0]
        update_month_options()

    def update_month_options(*args):
        months_for_selection = station_months(station_selector.value, year_selector.value)
        month_selector.options = months_for_selection
        if month_selector.value not in months_for_selection:
            month_selector.value = months_for_selection[0]

    def update_month_visibility(*args):
        month_selector.disabled = time_resolution_selector.value in {"Yearly", "Total"}

    def update_year_visibility(*args):
        year_selector.disabled = time_resolution_selector.value == "Total"

    station_selector.observe(update_year_options, names="value")
    year_selector.observe(update_month_options, names="value")
    time_resolution_selector.observe(update_month_visibility, names="value")
    time_resolution_selector.observe(update_year_visibility, names="value")
    update_month_visibility()
    update_year_visibility()

    out_fig = interactive_output(update_plot, {
        "plot_type": plot_type_selector,
        "station_id": station_selector,
        "year": year_selector,
        "month": month_selector,
        "met_variable": met_variable_selector,
        "time_resolution": time_resolution_selector,
    })

    ui = VBox([
        HBox([
            plot_type_selector,
            station_selector,
            year_selector,
            month_selector,
            met_variable_selector,
            time_resolution_selector,
        ])
    ])
    display(ui, out_fig)
    return ui, out_fig
