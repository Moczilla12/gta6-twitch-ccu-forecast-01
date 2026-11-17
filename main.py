"""Forecast Twitch concurrent viewers for a launch day's hours 2..24.

Idea: a big launch's day-1 curve looks a lot like other big launches' day-1
curves once you anchor on the first hour. For each past launch we know how
its hourly viewers moved away from its own hour 1 (in log space); the
forecast for a new launch takes the median of those moves across a filtered
set of similar launches, then widens the outer quantiles by a fixed factor
so the band stays honest about how different launches really are.

The filter keeps only launches that behaved like a clean console launch:
no early-access contamination, no weak tier, a late peak (hour 10 or
later), a peak a few times the first hour, and a big jump from the
pre-release baseline. If too few launches survive, the filter is relaxed
step by step; if almost none survive, the forecast falls back to
persistence (stay at hour 1, spread widening like a random walk).

Only the Python standard library is used, there is no network access and
nothing is random, so two runs on the same case write identical bytes.
Every case is handled on its own: a broken case still gets a persistence
forecast and the program keeps going with exit code 0.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Log-space spread of one hour's move in the persistence fallback.
FALLBACK_SIGMA = 0.15
# How much wider than the raw cross-launch spread the outer quantiles sit.
SPREAD_FACTOR = 1.2
# Hours of a launch's day, 1-indexed.
DAY_HOURS = range(1, 25)
FORECAST_HOURS = range(2, 25)
# Pre-release baseline window, in hours before day_start: [-336, -24).
BASELINE_START = -336
BASELINE_END = -24


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round_half_up_2(x):
    """Round to 2 decimals, half up; return an int when the result is whole."""
    whole = math.floor(x * 100 + 0.5) / 100
    return int(whole) if whole.is_integer() else whole


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def quantile_interp(values, tau):
    """Linear-interpolation quantile of a non-empty list of numbers."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = tau * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def persistence_row(p1, hour):
    """Five quantiles around hour 1's print, widening with the square root of time."""
    width = FALLBACK_SIGMA * math.sqrt(hour - 1)
    return [round_half_up_2(p1 * math.exp(z * width)) for z in Z_SCORES]


def clean_row(p1, hour, values):
    """Clamp, order and round one hour's five quantiles."""
    out = []
    prev = 1.0
    for value in values:
        clipped = min(max(value, 1.0), 1e9)
        clipped = max(clipped, prev)
        out.append(round_half_up_2(clipped))
        prev = clipped
    return out


def hour_offset(point_time, day_start):
    """Whole hours between a point's timestamp and the launch day's start."""
    return int((point_time - day_start).total_seconds() // 3600)


def launch_shape(comp):
    """Summarise one comparable launch, or return None when unusable.

    Returns (release-day hourly viewers by hour, surge from baseline).
    The baseline is the median log viewers over the pre-release window
    [-336, -24) hours; the surge is log(hour 1) minus that baseline.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hourly = comp.get("hourly") or []
    release = {}
    baseline_logs = []
    for point in hourly:
        try:
            viewers = point["viewers"]
            offset = hour_offset(parse_time(point["from"]), day_start)
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        if 0 <= offset < 24:
            release[offset + 1] = viewers
        elif BASELINE_START <= offset < BASELINE_END and viewers > 0:
            baseline_logs.append(math.log(viewers))
    first = release.get(1)
    if not first or first <= 0:
        return None
    surge = None
    if baseline_logs:
        surge = math.log(first) - median(baseline_logs)
    return release, surge


def is_clean_launch(comp):
    """True unless the unlock notes mark this launch as contaminated or weak."""
    unlock = comp.get("unlock") or ""
    return not (
        "early-access-contaminated" in unlock
        or "Tier: weak" in unlock
        or "target-history" in unlock
    )


def launch_peak(release):
    """(peak viewers, hour of the peak) over the release day's 24 hours."""
    best_value, best_hour = 0, 0
    for hour in DAY_HOURS:
        value = release.get(hour)
        if value and (value, hour) > (best_value, best_hour):
            best_value, best_hour = value, hour
    return best_value, best_hour


def pick_training_set(clean, target_surge):
    """Choose the launches whose day-1 shapes anchor the forecast.

    `clean` is a list of (release-day dict, surge) for tier-clean launches.
    Tight first; loosen the peak-ratio cap when too few survive; fall back
    to every clean launch; None means persistence for the whole case.
    A target whose first hour shows no launch yet (small surge) uses the
    loose set straight away.
    """
    tight, loose = [], []
    for release, surge in clean:
        peak, peak_hour = launch_peak(release)
        first = release[1]
        ratio = peak / max(1, first)
        if surge is not None and surge <= 4:
            continue
        if peak_hour < 10 or ratio < 1.2:
            continue
        if ratio <= 5.0:
            tight.append(release)
        if ratio <= 15.0:
            loose.append(release)
    pool = loose if (target_surge is not None and target_surge < 2.0) else tight
    if len(pool) < 8 and pool is tight:
        pool = loose
    if len(pool) < 5:
        pool = [release for release, _surge in clean]
    if len(pool) < 3:
        return None
    return pool


def target_surge(case, day_start, p1):
    """The target's own surge from its pre-release baseline, if computable."""
    points = case.get("history") or case.get("history_hourly") or []
    baseline_logs = []
    for point in points:
        try:
            stamp = point.get("at") or point.get("from")
            viewers = point["viewers"]
            offset = hour_offset(parse_time(stamp), day_start)
        except (KeyError, TypeError, ValueError):
            continue
        if BASELINE_START <= offset < BASELINE_END and viewers and viewers > 0:
            baseline_logs.append(math.log(viewers))
    if not baseline_logs or p1 <= 0:
        return None
    return math.log(p1) - median(baseline_logs)


def requested_hours(case):
    """The hours a case asks to be forecast, in order, defaulting to 2..24."""
    try:
        hours = sorted({int(row["hour"]) for row in case.get("forecast") or []})
    except (KeyError, TypeError, ValueError):
        hours = []
    return hours or list(FORECAST_HOURS)


def forecast_case(case):
    """Build the 23-row forecast for one case dict."""
    day_start = parse_time(case["day_start"])
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None:
        try:
            p1 = max(1.0, float(first["viewers"]))
        except (KeyError, TypeError, ValueError):
            p1 = 1.0

    clean = []
    for comp in case.get("comparables") or []:
        if not is_clean_launch(comp):
            continue
        shape = launch_shape(comp)
        if shape is not None:
            clean.append(shape)
    surge = target_surge(case, day_start, p1)
    training = pick_training_set(clean, surge)

    log_p1 = math.log(p1)
    rows = []
    for hour in requested_hours(case):
        if training is None:
            rows.append({"hour": hour, "q": persistence_row(p1, hour)})
            continue
        moves = []
        for release in training:
            first_hour = release.get(1)
            later = release.get(hour)
            if first_hour and first_hour > 0 and later and later > 0:
                moves.append(math.log(later / first_hour))
        if len(moves) < 3:
            rows.append({"hour": hour, "q": persistence_row(p1, hour)})
            continue
        centre = quantile_interp(moves, 0.5)
        values = []
        for tau in LEVELS:
            edge = quantile_interp(moves, tau)
            values.append(math.exp(log_p1 + centre + SPREAD_FACTOR * (edge - centre)))
        rows.append({"hour": hour, "q": clean_row(p1, hour, values)})
    return {"hours": rows}


def fallback_forecast(case):
    """A persistence forecast built defensively from whatever the case holds."""
    p1 = 1.0
    try:
        known = case.get("known") or []
        first = next((k for k in known if k.get("hour") == 1), None)
        if first is not None:
            p1 = max(1.0, float(first["viewers"]))
    except (KeyError, TypeError, ValueError):
        p1 = 1.0
    hours = requested_hours(case)
    return {"hours": [{"hour": h, "q": persistence_row(p1, h)} for h in hours]}


def run_case(path):
    """Read one case file and return its forecast; never raise."""
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
        return forecast_case(case)
    except Exception:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            return fallback_forecast(case)
        except Exception:
            return {"hours": [{"hour": h, "q": persistence_row(1.0, h)} for h in FORECAST_HOURS]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        forecast = run_case(path)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
