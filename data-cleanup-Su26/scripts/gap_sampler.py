"""Draw a random sample of synthetic gaps across all stations (soil or met),
for testing imputation methods. Library-only for now (no CLI)."""
from pathlib import Path

import numpy as np
import pandas as pd

try:                                        # imported from the repo root (notebooks)
    from .synthetic_gap import synthetic_gap
except ImportError:                         # run from inside scripts/
    from synthetic_gap import synthetic_gap


def _default_columns(df):
    """The columns a gap may be punched into when the user doesn't pick any:
    every measurement column. 'Flag' is a QC flag, not a measurement."""
    return [c for c in df.columns if c != "Flag"]


def _gap_lengths(df, columns, max_rows):
    """Length in rows of every real NaN run shorter than max_rows in the given columns."""
    out = []
    for col in columns:
        missing = df[col].isna()
        run_id = (missing != missing.shift()).cumsum()          # label each run of equal values
        run_len = missing.groupby(run_id).sum()                 # NaN runs sum to their length,
        out += [int(n) for n in run_len if 0 < n < max_rows]    # runs of real data sum to 0
    return out


def _plan_slots(length_pool, n_gaps, strata, rain_frac, rng):
    """Decide every draw's gap length and rain requirement up front.

    strata=True splits n_gaps evenly across the pool's distinct lengths;
    strata=False draws each gap's length from the pool at random. Returns a
    list of n_gaps (hours, want_rain) pairs, where want_rain is
    True / False / None (None = don't care).
    """
    if strata:
        # even split: any remainder goes to the shortest lengths
        choices = np.unique(length_pool)                # distinct lengths, sorted
        base, extra = divmod(n_gaps, len(choices))
        quotas = [base + 1 if i < extra else base for i in range(len(choices))]
        slot_hours = np.repeat(choices, quotas)
    else:
        # natural mix: common lengths in the pool appear more often
        slot_hours = rng.choice(length_pool, size=n_gaps)
    slot_hours = slot_hours.astype(int)

    if rain_frac is None:
        return [(int(h), None) for h in slot_hours]

    # require rain in rain_frac of the slots at every distinct length. 'owed'
    # carries the fractional remainder from one length to the next, so the
    # overall share stays exact even when a length's count doesn't divide evenly.
    slots, owed = [], 0.0
    for h in np.unique(slot_hours):
        k = int((slot_hours == h).sum())                # slots at this length
        owed += k * rain_frac                           # rainy slots they should add
        n_rain = min(k, max(0, int(owed + 0.5)))        # round to whole slots
        owed -= n_rain                                  # carry the remainder onward
        slots += [(int(h), True)] * n_rain + [(int(h), False)] * (k - n_rain)
    return slots


def _station_files(data_dir, met):
    """Map station name -> file path for the soil (or met) csvs in data_dir.

    Trusts the prewash naming convention: met files end in "_met", soil files
    don't, and the folder holds nothing but prewashed station csvs.
    """
    station_files = {}
    for path in sorted(data_dir.iterdir()):
        if not path.name.endswith(".csv"):
            continue
        is_met_file = path.stem.endswith("_met")
        if is_met_file == met:
            station_files[path.stem] = path
    if not station_files:
        kind = "met" if met else "soil"
        raise FileNotFoundError(f"no {kind} station csvs found in {data_dir.resolve()}")
    return station_files


def _print_tally(samples, attempts, n_gaps, strata, rain_frac):
    """Report how the draw went, after the fact."""
    if len(samples) < n_gaps:
        print(f"warning: only {len(samples)} of {n_gaps} slots filled after {attempts} attempts")
    if not samples:
        return                                  # nothing drawn -> nothing to tally
    if not strata and rain_frac is None:
        return                                  # plain draw -> nothing more to report
    drawn = pd.DataFrame([{"length_h": s["length_h"], "rain": s["rain_in_gap"]}
                          for s in samples])
    print(f"{len(samples)} gaps drawn ({attempts} attempts)")
    if rain_frac is not None:                   # show the length x rain balance
        print(pd.crosstab(drawn["length_h"], drawn["rain"]).to_string())
    else:                                       # show the count at each length
        print(drawn["length_h"].value_counts().sort_index().to_string())


def sample_gaps(data_dir="prewashed_data", n_gaps=250, seed=None, column=None,
                lengths=None, strata=False, rain_frac=None, context="1h",
                pool_max="24h", met=False):
    """Return a list of synthetic-gap samples drawn randomly across all stations.

    data_dir  : path to the prewashed station csvs.
    n_gaps    : how many synthetic gaps to draw.
    seed      : master seed, for a reproducible sample.
    column    : None -> a random measurement column per gap (any column except
                the QC 'Flag'); "SWC_50" fixes one column; or a menu like
                ["SWC_5", "SWC_10"] to draw from. 'Flag' cannot be gapped.
    lengths   : the exact gap lengths to draw, e.g. ["1h", "6h", "23h"] (not bin
                edges); None -> draw lengths that match the durations of real
                gaps in the data (only their durations are used — every sampled
                gap is synthetic, punched into fully clean data).
    strata    : False -> each gap's length is drawn from the pool at random
                (common lengths appear more often); True -> n_gaps is split
                evenly across the pool's distinct lengths.
    rain_frac : None -> don't control rain; or a fraction like 0.5 -> that share
                of gaps must contain rain (Ppt > 0 inside the gap window),
                enforced separately at every distinct gap length.
    context   : data kept on each side of each gap (rows or a span like "2D");
                None keeps the full station frame — but beware: the strictly-
                synthetic rule then requires the WHOLE record to be gap-free,
                which only a few stations are, so None quietly restricts the
                draw to those stations. Prefer a finite context.
    pool_max  : when lengths is None, only real-gap durations shorter than this
                span are matched; "24h" (the default) fits the project's <24h
                imputation goal, "7D" would admit multi-day durations too.
    met       : False -> sample the soil stations (files not ending in "_met");
                True -> sample the met stations (files ending in "_met").

    Returns a list of n_gaps sample dicts (fewer only if the draw budget runs
    out first — a warning is printed). Each dict describes one synthetic gap:

    station     : str — station name, e.g. "CB01" or "CB01_met" (the file's stem).
    column      : str — the one column the gap was punched into, e.g. "SWC_5".
    length_h    : int — gap length in hours (= rows; the data are hourly).
    start       : pd.Timestamp — the first missing hour of the gap.
    rain_in_gap : bool — True if Ppt > 0 at any hour inside the gap window.
    gapped      : pd.DataFrame — the station frame with the synthetic gap punched
                  into `column`, trimmed to `context` on each side of the gap.
                  The synthetic gap is the ONLY missing data in the frame:
                  windows that touch a real gap in any column are rejected, so
                  every other value is real data, and at least one real row
                  sits on each side of the gap.
    answer      : pd.DataFrame — the ground truth: the true values of `column`
                  over the gap window, one column indexed by the missing hours.
                  Score an imputation with
                  `imputed.loc[answer.index, answer.columns]` against `answer`.
    """
    if column is not None:
        wanted = [column] if isinstance(column, str) else list(column)
        if "Flag" in wanted:
            raise ValueError("'Flag' is a TxSON QC flag, not a measurement")

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir.resolve()} is not a directory — "
                                "point data_dir at the prewashed_data folder")

    rng = np.random.default_rng(seed)

    station_files = _station_files(data_dir, met)   # station name -> its file path
    stations = sorted(station_files)

    _cache, _rain_cache = {}, {}

    def load(station):
        if station not in _cache:
            _cache[station] = pd.read_csv(station_files[station],
                                          index_col="Date", parse_dates=["Date"],
                                          dtype={"Flag": str})   # Flag is a string, not a number
        return _cache[station]

    def rain_rows(station):
        """Integer positions of hours with Ppt > 0 at a station."""
        if station not in _rain_cache:
            _rain_cache[station] = np.flatnonzero((load(station)["Ppt"] > 0).to_numpy())
        return _rain_cache[station]

    # gap lengths in hours: the real gap pool measured from the data
    # (capped at pool_max), or the user's fixed menu
    if lengths is None:
        max_rows = int(pd.Timedelta(pool_max) / pd.Timedelta("1h"))
        length_pool = np.array([n for st in stations
                                for n in _gap_lengths(load(st), _default_columns(load(st)),
                                                      max_rows)])
        if len(length_pool) == 0:
            raise ValueError(f"no real gaps shorter than {pool_max} found in the data to draw "
                             "lengths from — pass a fixed menu via lengths")
    else:
        length_pool = np.array([int(pd.Timedelta(l) / pd.Timedelta("1h")) for l in lengths])

    # every draw's gap length and rain requirement, decided up front
    slots = _plan_slots(length_pool, n_gaps, strata, rain_frac, rng)

    samples, attempts = [], 0
    while len(samples) < n_gaps and attempts < n_gaps * 40:   # * 40: give up eventually
        attempts += 1

        # a random station, and a random measurement column it actually has
        station = str(rng.choice(stations))
        df = load(station)
        if column is None:
            menu = _default_columns(df)
        elif isinstance(column, str):
            menu = [column]
        else:
            menu = list(column)
        menu = [c for c in menu if c in df.columns]      # e.g. six stations have no SWC_50
        if not menu:
            continue                                     # station lacks the requested column -> redraw
        col = str(rng.choice(menu))

        # this draw's gap length and rain requirement (next unfilled slot)
        hours, want_rain = slots[len(samples)]
        if want_rain:                                    # anchor the window on a random rainy hour
            wet = rain_rows(station)
            if len(wet) == 0:
                continue
            start_pos = int(rng.choice(wet)) - int(rng.integers(0, hours))
            if start_pos < 1 or start_pos + hours > len(df) - 1:   # keep an anchor row each side
                continue
            start = df.index[start_pos]
        else:                                            # let synthetic_gap place it randomly
            start = None

        # punch the gap; synthetic_gap refuses windows that overlap real gaps
        try:
            gapped, answer = synthetic_gap(df, f"{hours}h", column=col, start=start,
                                           seed=int(rng.integers(2**32)),
                                           context=context)
        except ValueError:
            continue

        # strictly synthetic: the punched hole must be the only missing data in
        # the window, and a real row must sit on each side of it
        if gapped.isna().sum().sum() != len(answer):     # any extra NaN is a real gap
            continue
        if not (gapped.index[0] < answer.index[0] and gapped.index[-1] > answer.index[-1]):
            continue                                     # gap flush with the record edge

        has_rain = bool(df.loc[answer.index, "Ppt"].gt(0).any())   # from the original df,
        if want_rain is False and has_rain:              # so it's correct even when Ppt
            continue                                     # itself is the gapped column

        samples.append({"station": station, "column": col, "length_h": hours,
                        "start": answer.index[0], "rain_in_gap": has_rain,
                        "gapped": gapped, "answer": answer})

    _print_tally(samples, attempts, n_gaps, strata, rain_frac)
    return samples
