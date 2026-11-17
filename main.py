"""Forecast Twitch concurrent viewers for a launch day's hours 2..24.

Model (v2): peak-amplitude + shape decomposition, fit on similar past launches.

    log v(h) = log h1 + G * s(h)

where G = log(peak / hour-1) is the launch's total log-growth and s(h) is the
normalised shape (0 at hour 1, 1 at the peak).  Two things are estimated
from the training launches (the case's comparables that look like clean
rolling-midnight console launches):

  * G is predicted from the target's own hour-1 level: log(peak/h1) falls
    almost linearly with log(h1) (bigger first hour -> the world is already
    watching -> less room to grow).  The prediction is an OLS fit shrunk
    75% toward the median G, with the hour-1 level clipped to the range of
    launches seen ([10k, 400k]) and G floored at log(1.2): a rolling-midnight
    launch's peak must exceed its first hour (the US has not unlocked yet),
    and no line fitted on past launches is trusted past the biggest one.
  * s(h) is the median normalised shape across training launches, lightly
    smoothed over neighbouring hours.

Quantiles come from honest leave-one-out residuals of the same procedure:
for each training launch, fit on the others, predict it, and record the
log-residual per hour.  The forecast's quantiles are the median curve
shifted by those residual quantiles, so the band is as wide as past
launches really were hard to predict -- no wider, no narrower.

If too few clean launches survive, fall back to every clean launch; if
still too few, fall back to persistence (stay at hour 1, spread widening
like a random walk).

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
# Hours of a launch's day, 1-indexed.
DAY_HOURS = range(1, 25)
FORECAST_HOURS = range(2, 25)
# Pre-release baseline window, in hours before day_start: [-336, -24).
BASELINE_START = -336
BASELINE_END = -24
# G-model: shrink of the OLS prediction toward the median G.
G_SHRINK = 0.75
# G-model: clip log(hour-1) to the launches seen, and floor G at log(1.2).
G_LOG_H1_LO = math.log(10000)
G_LOG_H1_HI = math.log(400000)
G_FLOOR = math.log(1.2)


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


def is_class_b(shape):
    """A clean rolling-midnight launch that builds through the day.

    Same rule the offline validation used: a big surge from the pre-release
    baseline, a late peak (hour 10 or later), and a peak a few times the
    first hour -- but not hundreds of times (that is a launch whose day
    really starts later, not a midnight one).
    """
    release, surge = shape
    if surge is not None and surge <= 4:
        return False
    peak, peak_hour = launch_peak(release)
    first = release[1]
    ratio = peak / max(1, first)
    return peak_hour >= 10 and 1.2 <= ratio <= 15.0


def ols(xs, ys):
    """Least-squares line; returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def growth_of(release):
    """Total log-growth log(peak / hour-1) of one launch's release day."""
    peak, _hour = launch_peak(release)
    return math.log(peak / max(1, release[1] or 1))


def shape_of(release):
    """Normalised shape s(h) = (log v(h) - log h1) / G, 0 at hour 1."""
    total = growth_of(release)
    first = max(1, release[1] or 1)
    out = {}
    for hour in DAY_HOURS:
        value = release.get(hour)
        if value is None or total <= 0:
            out[hour] = None
        else:
            out[hour] = (math.log(value) - math.log(first)) / total
    return out


def fit_g_model(training):
    """(intercept, slope, median G, shrink) for log(peak/h1) on log(h1)."""
    xs = [math.log(max(1, release[1] or 1)) for release, _surge in training]
    ys = [growth_of(release) for release, _surge in training]
    intercept, slope = ols(xs, ys)
    return (intercept, slope, median(ys), G_SHRINK)


def predict_g(g_model, p1):
    """Total log-growth for a target whose first hour printed p1."""
    intercept, slope, med, shrink = g_model
    clipped = min(max(math.log(max(1, p1)), G_LOG_H1_LO), G_LOG_H1_HI)
    raw = intercept + slope * clipped
    return max(G_FLOOR, med + shrink * (raw - med))


def fit_shape(training):
    """Median normalised shape per hour, lightly smoothed over neighbours."""
    shapes = {hour: [] for hour in DAY_HOURS}
    for release, _surge in training:
        shape = shape_of(release)
        for hour in DAY_HOURS:
            if shape[hour] is not None:
                shapes[hour].append(shape[hour])
    med = {}
    for hour in DAY_HOURS:
        values = shapes[hour]
        med[hour] = median(values) if values else (0.0 if hour == 1 else 1.0)
    smoothed = {}
    for hour in DAY_HOURS:
        neighbours = [med[h] for h in range(max(1, hour - 1), min(24, hour + 1) + 1)]
        smoothed[hour] = sum(neighbours) / len(neighbours)
    smoothed[1] = 0.0
    return smoothed


def median_curve(training, p1):
    """The median forecast curve (dict hour -> viewers) for a target."""
    g_model = fit_g_model(training)
    shape = fit_shape(training)
    total = max(0.0, predict_g(g_model, p1))
    return {hour: p1 * math.exp(total * shape[hour]) for hour in DAY_HOURS}


def loo_residuals(training):
    """Honest leave-one-out log-residuals of the same procedure, per hour."""
    resid = {hour: [] for hour in FORECAST_HOURS}
    for index, held in enumerate(training):
        sub = training[:index] + training[index + 1:]
        if len(sub) < 3:
            continue
        release, _surge = held
        curve = median_curve(sub, max(1, release[1] or 1))
        for hour in FORECAST_HOURS:
            value = release.get(hour)
            if value:
                resid[hour].append(math.log(value) - math.log(max(1e-9, curve[hour])))
    return resid


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
    training = [shape for shape in clean if is_class_b(shape)]
    if len(training) < 3:
        training = clean
    if len(training) < 3:
        training = None

    rows = []
    if training is None:
        for hour in requested_hours(case):
            rows.append({"hour": hour, "q": persistence_row(p1, hour)})
        return {"hours": rows}

    med_curve = median_curve(training, p1)
    resid = loo_residuals(training)
    for hour in requested_hours(case):
        values = resid.get(hour) or []
        if len(values) < 3:
            rows.append({"hour": hour, "q": persistence_row(p1, hour)})
            continue
        med_resid = median(values)
        quantiles = []
        for tau in LEVELS:
            edge = quantile_interp(values, tau)
            quantiles.append(med_curve[hour] * math.exp(med_resid + (edge - med_resid)))
        rows.append({"hour": hour, "q": clean_row(p1, hour, quantiles)})
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
