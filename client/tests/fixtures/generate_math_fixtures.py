"""
generate_math_fixtures.py

Generates the ground-truth fixtures used by client/tests/timeseriesMath.test.js.

The client-side timeseries extraction has to reproduce a handful of very
specific pandas/scipy/numpy behaviours (banker's rounding, the masking rules of
Series.interpolate(method='time'), ddof=1 rolling statistics, and the
polynomial edge handling of savgol_filter's mode='interp'). Rather than trust a
reading of the docs, we capture what those libraries actually do here and
assert the JavaScript port matches.

Run from the repository root:

    python client/tests/fixtures/generate_math_fixtures.py

Requires numpy, pandas and scipy. The output JSON is committed, so this only
needs re-running if the reference implementation in
server/utils/extract_multi_zarr_ts.py changes.
"""

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

NAN = float("nan")
OUT_PATH = pathlib.Path(__file__).parent / "math_fixtures.json"


def encode(values):
    """JSON has no NaN, so missing values travel as None and are restored in JS."""
    return [None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
            for v in values]


def build_interpolation_cases():
    """
    pandas Series.interpolate(method='time') cases.

    Covers the three behaviours the port has to get right: flat (not linear)
    extrapolation past the last observation, leading gaps that stay empty under
    the default forward direction, and `limit` counting from the start of each
    gap rather than across the whole series.
    """
    cases = []

    def add(name, times, values, **kwargs):
        idx = pd.to_datetime(times)
        series = pd.Series(values, index=idx)
        result = series.interpolate(method="time", **kwargs)
        cases.append({
            "name": name,
            "times": [int(t.value // 1_000_000) for t in idx],   # ms since epoch
            "values": encode(values),
            "options": {
                # Infinity is not representable in JSON; the JS side treats a
                # missing limit as unlimited, matching the pandas default.
                "limit": kwargs.get("limit"),
                "limitDirection": kwargs.get("limit_direction", "forward"),
            },
            "expected": encode(result.values),
        })

    daily = pd.date_range("2020-01-01", periods=10, freq="D")
    gapped = [NAN, NAN, 10.0, NAN, NAN, NAN, 40.0, NAN, NAN, NAN]

    add("gap_limit_2", daily, gapped, limit=2)
    add("gap_limit_24", daily, gapped, limit=24)
    add("gap_both", daily, gapped, limit_direction="both")
    add("gap_limit_2_both", daily, gapped, limit=2, limit_direction="both")

    # Runs long enough to separate the forward fill from the backward one. A
    # run is only fillable in a direction it is anchored in, so a leading run
    # never fills forwards and a trailing run never fills backwards - even
    # under limit_direction='both'.
    add("interior_run_7_forward", pd.date_range("2020-01-01", periods=9, freq="D"),
        [10.0] + [NAN] * 7 + [80.0], limit=2)
    add("interior_run_7_both", pd.date_range("2020-01-01", periods=9, freq="D"),
        [10.0] + [NAN] * 7 + [80.0], limit=2, limit_direction="both")
    add("leading_run_5_forward", pd.date_range("2020-01-01", periods=7, freq="D"),
        [NAN] * 5 + [10.0, 20.0], limit=2)
    add("leading_run_5_both", pd.date_range("2020-01-01", periods=7, freq="D"),
        [NAN] * 5 + [10.0, 20.0], limit=2, limit_direction="both")
    add("trailing_run_5_forward", pd.date_range("2020-01-01", periods=7, freq="D"),
        [10.0, 20.0] + [NAN] * 5, limit=2)
    add("trailing_run_5_both", pd.date_range("2020-01-01", periods=7, freq="D"),
        [10.0, 20.0] + [NAN] * 5, limit=2, limit_direction="both")

    # Irregular spacing: interpolation must be weighted by real elapsed time,
    # not by position in the array.
    irregular = pd.to_datetime([
        "2020-01-01 00:00", "2020-01-01 06:00", "2020-01-05 00:00",
        "2020-01-05 12:00", "2020-01-20 00:00", "2020-01-21 00:00",
    ])
    add("irregular_spacing", irregular, [1.0, NAN, NAN, 9.0, NAN, 12.0], limit=24)

    # Degenerate inputs that must not throw.
    add("all_missing", daily[:4], [NAN, NAN, NAN, NAN], limit=24)
    add("single_observation", daily[:5], [NAN, 7.0, NAN, NAN, NAN], limit=2)
    add("no_gaps", daily[:4], [1.0, 2.0, 3.0, 4.0], limit=24)

    return cases


def build_rolling_cases():
    """pandas .rolling(window, center=True, min_periods=1) mean and std."""
    cases = []

    def add(name, values, window):
        series = pd.Series(values)
        rolling = series.rolling(window, center=True, min_periods=1)
        cases.append({
            "name": name,
            "values": encode(values),
            "window": window,
            "expectedMean": encode(rolling.mean().values),
            "expectedStd": encode(rolling.std().values),
            "expectedOverallStd": None if not np.isfinite(series.std()) else float(series.std()),
        })

    add("sparse_with_spike", [1.0, NAN, 3.0, 100.0, 5.0, NAN, NAN, 8.0, 9.0, 10.0], 5)
    add("dense", [float(v) for v in range(12)], 5)
    # An isolated observation gives a window of one, where ddof=1 std is NaN.
    add("isolated_points", [NAN, NAN, 5.0, NAN, NAN, NAN, 9.0, NAN, NAN], 5)
    add("all_missing", [NAN] * 6, 5)
    add("constant", [4.0] * 8, 5)

    return cases


def build_savgol_cases():
    """
    scipy.signal.savgol_filter with the default mode='interp'.

    The edge samples are the interesting part: scipy fits one polynomial across
    the leading window and one across the trailing window rather than padding
    the signal, so a naive implementation diverges at exactly the first and
    last halfwidth samples.
    """
    rng = np.random.default_rng(42)
    cases = []

    def add(name, values, window, poly):
        values = np.asarray(values, dtype=float)
        cases.append({
            "name": name,
            "values": encode(values),
            "window": window,
            "poly": poly,
            "expectedCoeffs": encode(
                __import__("scipy.signal", fromlist=["savgol_coeffs"]).savgol_coeffs(window, poly)
            ),
            "expected": encode(savgol_filter(values, window, poly)),
        })

    add("short_w5_p2", [1., 3., 2., 8., 5., 4., 9., 7., 6., 10., 12., 11., 14., 13., 15.], 5, 2)
    add("short_w7_p2", [1., 3., 2., 8., 5., 4., 9., 7., 6., 10., 12., 11., 14., 13., 15.], 7, 2)
    # The production defaults: window 25, polyorder 2.
    add("noisy_w25_p2", rng.normal(3000, 250, 400), 25, 2)
    add("noisy_w25_p3", rng.normal(3000, 250, 400), 25, 3)
    # Window equal to the series length exercises the edge fits over everything.
    add("window_equals_length", rng.normal(100, 10, 25), 25, 2)
    add("large_values_w51_p2", rng.normal(18000, 4000, 900), 51, 2)

    return cases


def build_rounding_cases():
    """numpy.round, including the exact halves where it rounds to even."""
    rng = np.random.default_rng(7)
    values = np.concatenate([
        np.array([2.345, 0.125, 0.135, 1.005, 2.5, 0.25, 0.35, -0.25, -2.345, 0.0]),
        rng.uniform(-5000, 25000, 2000),
    ])
    return [
        {"decimals": 1, "values": encode(values), "expected": encode(np.round(values, 1))},
        {"decimals": 2, "values": encode(values), "expected": encode(np.round(values, 2))},
    ]


def main():
    fixtures = {
        "_comment": "Generated by tests/fixtures/generate_math_fixtures.py - do not edit by hand.",
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "interpolateTime": build_interpolation_cases(),
        "rolling": build_rolling_cases(),
        "savgol": build_savgol_cases(),
        "rounding": build_rounding_cases(),
    }

    OUT_PATH.write_text(json.dumps(fixtures, indent=1))
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
