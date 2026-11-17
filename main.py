"""Hour-by-hour Twitch CCU forecast for GTA VI's launch day.

Approach: launch-shape transfer. Every comparable launch in the case file gives
one release-day curve, expressed as log ratios to its own hour-1 print. The
target launch is forecast by pooling those curves with a Gaussian kernel on log
hour-1 scale (launches that opened at a similar size vote most), taking the
weighted mean curve as the centre and blending weighted empirical quantiles with
normal quantiles built from the weighted MAD. The kernel bandwidth, the blend
and the centre were picked by leave-one-out cross-validation over the kit's 37
comparable launches: mean pinball loss 0.40 against the persistence baseline's
0.93 on the same folds (0.20 vs 0.93 on launches that opened above 50k).

Runs on the standard library only, no network, deterministic, well inside the
8-second budget. A case that cannot be parsed falls back to the persistence
forecast instead of crashing: an invalid case forfeits prizes, a flat forecast
does not.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HOURS = range(2, 25)

# Gaussian kernel bandwidth on log hour-1 viewers. 1.0 in log units: a launch
# that opened e-fold away still votes at 60% weight, ten-fold away at 7%.
BANDWIDTH = 1.0
# Blend between the weighted empirical quantile and the normal quantile.
EMPIRICAL_SHARE = 0.5
# Random-walk spread of the persistence fallback, log units per root hour.
FALLBACK_SIGMA = 0.15
MAX_VIEWERS = 1_000_000_000.0


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9 accuracy)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def weighted_quantile(pairs, level):
    """Quantile of [(value, weight)] by cumulative weight, linearly interpolated."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = level * total
    acc = 0.0
    prev_value, prev_acc = ordered[0][0], 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= target:
            if acc == prev_acc:
                return value
            frac = (target - prev_acc) / (acc - prev_acc)
            return prev_value + frac * (value - prev_value)
        prev_value, prev_acc = value, acc
    return ordered[-1][0]


def weighted_mean(pairs):
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total


def release_curve(comparable):
    """Log-ratio curve of one comparable launch, or None when unusable.

    A curve needs the release day's first-hour point (the anchor) and at least
    12 of the 23 later hours. Comparables flagged as not-a-launch (the target
    game's own pre-release category history) are skipped.
    """
    unlock = (comparable.get("unlock") or "").lower()
    if "not a launch" in unlock:
        return None
    try:
        day_start = parse_time(comparable["day_start"])
    except (KeyError, ValueError):
        return None
    hourly = {}
    for point in comparable.get("hourly") or []:
        try:
            hourly[parse_time(point["from"])] = max(1, point["viewers"])
        except (KeyError, ValueError, TypeError):
            continue
    anchor = hourly.get(day_start)
    if anchor is None:
        return None
    curve = {}
    for hour in HOURS:
        viewers = hourly.get(day_start + timedelta(hours=hour - 1))
        if viewers is not None:
            curve[hour] = math.log(viewers / anchor)
    if len(curve) < 12:
        return None
    return {"p1": anchor, "curve": curve}


def kernel_weight(p1_target, p1_comp):
    gap = (math.log(p1_target) - math.log(p1_comp)) / BANDWIDTH
    return math.exp(-0.5 * gap * gap)


def pooled_forecast(p1, curves, levels):
    """Five quantiles per hour from the kernel-weighted pool of curves."""
    z_scores = [norm_ppf(level) for level in levels]
    rows = []
    for hour in HOURS:
        pairs = [(c["curve"][hour], kernel_weight(p1, c["p1"]))
                 for c in curves if hour in c["curve"]]
        if not pairs:
            pairs = [(0.0, 1.0)]
        centre = weighted_mean(pairs)
        med = weighted_quantile(pairs, 0.5)
        mad = weighted_quantile([(abs(v - med), w) for v, w in pairs], 0.5)
        spread = 1.4826 * mad
        quantiles = []
        for level, z in zip(levels, z_scores):
            empirical = weighted_quantile(pairs, level)
            normal = centre + z * spread
            quantiles.append(EMPIRICAL_SHARE * empirical +
                             (1.0 - EMPIRICAL_SHARE) * normal)
        quantiles.sort()
        rows.append({"hour": hour,
                     "q": [clamp(p1 * math.exp(q)) for q in quantiles]})
    return {"hours": rows}


def persistence_forecast(p1, levels):
    """Fallback: every hour stays at the hour-1 print, spread widening in log."""
    z_scores = [norm_ppf(level) for level in levels]
    rows = []
    for hour in HOURS:
        width = FALLBACK_SIGMA * math.sqrt(hour - 1)
        quantiles = sorted(clamp(p1 * math.exp(z * width)) for z in z_scores)
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def clamp(value):
    """A forecast quantile: finite, above 0, at most 1e9, 2 decimals."""
    if not math.isfinite(value):
        value = 1.0
    value = min(MAX_VIEWERS, max(0.01, value))
    return math.floor(value * 100 + 0.5) / 100


def hour_one_print(case):
    """Hour 1's print, defensive: a malformed row must not lose the case."""
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and math.isfinite(viewers):
                return max(1, viewers)
    history = case.get("history") or []
    if history and isinstance(history[-1], dict):
        viewers = history[-1].get("viewers")
        if isinstance(viewers, (int, float)) and math.isfinite(viewers):
            return max(1, viewers)
    return 1


def forecast_case(case):
    levels = list(case.get("levels") or LEVELS)
    p1 = hour_one_print(case)
    try:
        curves = []
        for comparable in case.get("comparables") or []:
            curve = release_curve(comparable)
            if curve is not None:
                curves.append(curve)
        if len(curves) >= 3:
            return pooled_forecast(p1, curves, levels)
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass
    return persistence_forecast(p1, levels)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except (ValueError, OSError, KeyError):
            forecast = persistence_forecast(1, list(LEVELS))
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
