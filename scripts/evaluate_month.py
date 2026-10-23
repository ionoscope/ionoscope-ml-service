"""Train on a historical year and replay a held-out month at +15 minutes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ionoscope-data"))
from data_preparation.normalize_time_grid import read_giro_raw, normalize_giro

STEP = pd.Timedelta(minutes=15)


def build_examples(normalized: pd.DataFrame, train_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.DataFrame:
    """An interval's values become available at its end, never at its start."""
    grid = pd.date_range(train_start - pd.Timedelta(days=7), test_end - STEP, freq=STEP)
    frame = normalized.set_index("time_utc").reindex(grid).rename_axis("source_interval_start").reset_index()
    frame["origin"] = frame.source_interval_start + STEP
    frame["target_start"] = frame.origin
    frame["target_end"] = frame.origin + STEP
    frame["actual"] = frame.foF2.shift(-1)
    frame["persistence"] = frame.foF2
    for steps in [1, 2, 4, 12, 24, 48, 96, 192, 672]:
        frame[f"foF2_lag_{steps}"] = frame.foF2.shift(steps)
    for steps in [1, 4, 12, 96]:
        frame[f"foF2_diff_{steps}"] = frame.foF2 - frame.foF2.shift(steps)
    for steps in [4, 12, 96]:
        frame[f"foF2_mean_{steps}"] = frame.foF2.rolling(steps, min_periods=1).mean()
    for name, values, period in [
        ("hour", frame.origin.dt.hour + frame.origin.dt.minute / 60, 24),
        ("year", frame.origin.dt.dayofyear, 365.25),
    ]:
        frame[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    return frame


def split_examples(frame: pd.DataFrame, train_start: pd.Timestamp,
                   test_start: pd.Timestamp, test_end: pd.Timestamp):
    train = frame[(frame.target_start >= train_start) & (frame.target_end <= test_start)].copy()
    test = frame[(frame.target_start >= test_start) & (frame.target_end <= test_end)].copy()
    return train, test


def metrics(actual, predicted):
    valid = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if valid.empty:
        return {"n": 0, "mae_mhz": None, "rmse_mhz": None}
    return {"n": len(valid), "mae_mhz": float(mean_absolute_error(valid.actual, valid.predicted)),
            "rmse_mhz": float(np.sqrt(mean_squared_error(valid.actual, valid.predicted)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--station", default="TR169")
    parser.add_argument("--train-start", default="2024-09-01T00:00:00Z")
    parser.add_argument("--test-start", default="2025-09-01T00:00:00Z")
    parser.add_argument("--test-end", default="2025-10-01T00:00:00Z")
    parser.add_argument("--cs-filter", choices=["off", "threshold"], default="off")
    parser.add_argument("--min-cs", type=float, default=50)
    args = parser.parse_args()
    start, cutoff, end = [pd.Timestamp(v).tz_convert("UTC") for v in (args.train_start, args.test_start, args.test_end)]
    if not start < cutoff < end:
        raise ValueError("Require train-start < test-start < test-end")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = read_giro_raw(Path(args.raw_dir), include_stations={args.station})
    if raw.empty:
        raise ValueError("No station observations found")
    # Broad physical bounds; no interpolation, proxy substitution or future filling.
    raw.loc[~raw.foF2.between(.5, 25), "foF2"] = np.nan
    normalized = normalize_giro(raw, "15min", args.min_cs if args.cs_filter == "threshold" else None, "30min")
    frame = build_examples(normalized, start, end)
    train, test = split_examples(frame, start, cutoff, end)
    features = [c for c in frame.columns if c == "foF2" or c.startswith(("foF2_lag_", "foF2_diff_", "foF2_mean_"))
                or c in {"hour_sin", "hour_cos", "year_sin", "year_cos"}]
    valid_train = train.foF2.notna() & train.actual.notna()
    frame["partition"] = "warmup_or_outside"
    frame.loc[train.index, "partition"] = "train"
    frame.loc[test.index, "partition"] = "test"
    frame.to_csv(output / "examples.csv", index=False)
    if valid_train.sum() < 100:
        raise ValueError("Fewer than 100 usable training pairs; inspect coverage")
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05,
        l2_regularization=.01, early_stopping=False, random_state=42)
    model.fit(train.loc[valid_train, features], train.loc[valid_train, "actual"])
    eligible = test.foF2.notna()
    test["prediction"] = np.nan
    if eligible.any():
        test.loc[eligible, "prediction"] = model.predict(test.loc[eligible, features])
    test["forecast_available"] = eligible
    test["scored"] = eligible & test.actual.notna()
    test["reason"] = np.where(~eligible, "missing_current_foF2", np.where(test.actual.isna(), "missing_actual", "scored"))
    test[["source_interval_start", "foF2_source_timestamp", "origin", "target_start", "target_end",
          "foF2", "actual", "prediction", "persistence", "forecast_available", "scored", "reason"]].to_csv(output / "predictions.csv", index=False)
    common = test[test.scored]
    model_metrics = metrics(common.actual, common.prediction)
    baseline = metrics(common.actual, common.persistence)
    report = {
        "station": args.station, "train_start": str(start), "train_end_exclusive": str(cutoff),
        "test_start": str(cutoff), "test_end_exclusive": str(end), "cs_filter": args.cs_filter,
        "min_cs": args.min_cs if args.cs_filter == "threshold" else None,
        "target": "Mean foF2 over the next 15-minute interval; issue at previous interval end",
        "train_slots": len(train), "train_pairs": int(valid_train.sum()), "test_slots": len(test),
        "forecasts": int(eligible.sum()), "scored_pairs": len(common),
        "forecast_coverage": float(eligible.mean()), "scoring_coverage": float(test.scored.mean()),
        "model": model_metrics, "persistence": baseline,
        "feature_columns": features,
        "mae_improvement_percent": (100 * (1 - model_metrics["mae_mhz"] / baseline["mae_mhz"])
                                    if baseline["mae_mhz"] else None),
        "assumptions": ["Interval observations assumed available at interval end; publication latency unknown",
                        "Historical archive may contain later revisions; this is not an exact live replay",
                        "No test-month retraining or tuning; no external drivers used",
                        "CS filtering disabled only in experimental off mode; missing foF2 blocks a forecast"],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    joblib.dump({"model": model, "feature_columns": features, "metadata": report}, output / "model.joblib")
    monthly = frame.groupby(frame.source_interval_start.dt.strftime("%Y-%m")).agg(
        slots=("foF2", "size"), fof2_present=("foF2", "count"))
    monthly["coverage"] = monthly.fof2_present / monthly.slots
    monthly.to_csv(output / "monthly_coverage.csv")
    daily = []
    for day, group in test.groupby(test.origin.dt.strftime("%Y-%m-%d")):
        scored = group[group.scored]
        daily.append({"date": day, "slots": len(group), "forecasts": int(group.forecast_available.sum()),
                      "scored": len(scored), "model_mae": metrics(scored.actual, scored.prediction)["mae_mhz"],
                      "persistence_mae": metrics(scored.actual, scored.persistence)["mae_mhz"]})
    pd.DataFrame(daily).to_csv(output / "daily_metrics.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(test.target_start, test.actual, lw=.65, label="Observed interval mean")
    ax.plot(test.target_start, test.prediction, lw=.65, alpha=.75, label="Forecast +15 min")
    ax.set(xlabel="Target interval start (UTC)", ylabel="foF2 (MHz)", title=f"{args.station}: held-out month, fixed model")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "forecast_month.png", dpi=160); plt.close(fig)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
