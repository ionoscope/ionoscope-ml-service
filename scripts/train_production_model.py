#!/usr/bin/env python3
"""Train and export the first production foF2 +15min model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FUTURE_DERIVED_MARKERS = ("_target_", "_pred", "_savgol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, help="Directory with stations/*_features.csv.")
    parser.add_argument("--output", default="production_models/fof2_next_15min.joblib")
    parser.add_argument("--horizon", default="15min")
    parser.add_argument("--target", default="foF2")
    parser.add_argument("--test-start", default="2025-10-01T00:00:00Z")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional deterministic row cap for smoke runs.")
    parser.add_argument("--max-stations", type=int, default=0, help="Optional station cap for smoke runs.")
    parser.add_argument("--rows-per-station", type=int, default=0, help="Optional CSV row cap per station for smoke runs.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_dir = Path(args.features_dir)
    station_dir = features_dir / "stations"
    paths = sorted(station_dir.glob("*_features.csv"))
    if args.max_stations > 0:
        paths = paths[: args.max_stations]
    if not paths:
        raise FileNotFoundError(f"No station feature files found in {station_dir}.")

    target_column = f"{args.target}_target_{safe_name(args.horizon)}"
    frame = load_training_frame(paths, target_column, args.max_rows, args.rows_per_station, args.random_state)
    feature_columns = numeric_feature_columns(frame, args.target)
    train, test = split_train_test(frame, args.test_start)
    if train.empty or test.empty:
        raise ValueError("Train/test split is empty. Check --test-start and feature coverage.")

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=args.random_state,
    )
    y_train = pd.to_numeric(train[target_column], errors="coerce")
    valid_train = y_train.notna()
    model.fit(train.loc[valid_train, feature_columns], y_train.loc[valid_train])

    y_test = pd.to_numeric(test[target_column], errors="coerce")
    valid_test = y_test.notna()
    predictions = pd.Series(model.predict(test.loc[valid_test, feature_columns]), index=y_test.loc[valid_test].index)
    metrics = metric_record(y_test.loc[valid_test], predictions)

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "metadata": {
            "target": args.target,
            "horizon": args.horizon,
            "model": "HistGradientBoostingRegressor",
            "model_version": "0.1.0",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_start": args.test_start,
            "train_rows": int(valid_train.sum()),
            "test_rows": int(valid_test.sum()),
            "metrics": metrics,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    print(json.dumps(artifact["metadata"], indent=2))


def safe_name(value: str) -> str:
    return value.replace(" ", "").replace("/", "_").replace("-", "_")


def load_training_frame(
    paths: list[Path],
    target_column: str,
    max_rows: int,
    rows_per_station: int,
    random_state: int,
) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, nrows=rows_per_station if rows_per_station > 0 else None)
        if target_column not in frame.columns:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"No feature files contain target column {target_column!r}.")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=[target_column])
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=random_state).sort_values("time_utc")
    return frame.reset_index(drop=True)


def split_train_test(frame: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    time = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    start = pd.Timestamp(test_start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    return frame.loc[time < start].copy(), frame.loc[time >= start].copy()


def numeric_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    excluded = {"hour", "month", "doy"}
    target_prefix = f"{target}_target_"
    columns = []
    for column in frame.select_dtypes(include=np.number).columns:
        if column in excluded:
            continue
        if column.startswith(target_prefix) or any(marker in column for marker in FUTURE_DERIVED_MARKERS):
            continue
        columns.append(column)
    return columns


def metric_record(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    valid = pd.DataFrame({"actual": y_true, "predicted": y_pred}).dropna()
    if valid.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "corr": np.nan}
    error = valid["predicted"] - valid["actual"]
    return {
        "n": int(len(valid)),
        "mae": float(mean_absolute_error(valid["actual"], valid["predicted"])),
        "rmse": float(np.sqrt(mean_squared_error(valid["actual"], valid["predicted"]))),
        "r2": float(r2_score(valid["actual"], valid["predicted"])) if len(valid) > 1 else np.nan,
        "corr": float(valid["actual"].corr(valid["predicted"])) if len(valid) > 1 else np.nan,
        "bias": float(error.mean()),
    }


if __name__ == "__main__":
    main()
