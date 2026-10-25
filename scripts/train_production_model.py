#!/usr/bin/env python3
"""Train and export the first production foF2 +15min model."""

from __future__ import annotations

import argparse
import json
import hashlib
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
    parser.add_argument("--dataset-step", default="15min", help="Duration of each aggregated feature/target interval.")
    parser.add_argument("--target", default="foF2")
    parser.add_argument("--test-start", default="2025-10-01T00:00:00Z")
    parser.add_argument("--train-start", help="Inclusive first training target interval start.")
    parser.add_argument("--test-end", help="Exclusive test target interval end boundary.")
    parser.add_argument("--feature-schema", help="Trusted model artifact whose feature list must be preserved.")
    parser.add_argument("--data-manifest", help="Historical preparation manifest to embed in training provenance.")
    parser.add_argument("--model-version", default="0.1.0")
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
    if args.feature_schema:
        from sklearn._loss import _loss
        import sys
        sys.modules.setdefault('_loss', _loss)
        feature_columns = list(joblib.load(args.feature_schema)['feature_columns'])
        forbidden = [c for c in feature_columns if c in {'actual','prediction','persistence','partition'}
                     or any(marker in c for marker in FUTURE_DERIVED_MARKERS)]
        if forbidden:
            raise ValueError(f'Future-derived columns in feature schema: {forbidden}')
    else:
        feature_columns = None
    frame = load_training_frame(paths, target_column, args.max_rows, args.rows_per_station, args.random_state, feature_columns)
    if feature_columns is not None:
        missing = set(feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required schema features: {sorted(missing)}")
    else:
        feature_columns = numeric_feature_columns(frame, args.target)
    station_codes = sorted(frame.station.dropna().unique().tolist())
    train, test = split_train_test(frame, args.test_start, args.horizon, args.dataset_step)
    del frame
    delta, step = pd.Timedelta(args.horizon), pd.Timedelta(args.dataset_step)
    if args.train_start:
        train = train[pd.to_datetime(train.time_utc, utc=True) + delta >= pd.Timestamp(args.train_start)]
    if args.test_end:
        test = test[pd.to_datetime(test.time_utc, utc=True) + delta + step <= pd.Timestamp(args.test_end)]
    if train.empty or test.empty:
        raise ValueError("Train/test split is empty. Check --test-start and feature coverage.")

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=args.random_state,
        early_stopping=False,
    )
    y_train = pd.to_numeric(train[target_column], errors="coerce")
    valid_train = np.isfinite(y_train) & np.isfinite(pd.to_numeric(train[args.target], errors='coerce'))
    if not valid_train.any():
        raise ValueError('No usable training pairs')
    model.fit(train.loc[valid_train, feature_columns], y_train.loc[valid_train])

    y_test = pd.to_numeric(test[target_column], errors="coerce")
    eligible_test = np.isfinite(pd.to_numeric(test[args.target], errors='coerce'))
    valid_test = np.isfinite(y_test) & eligible_test
    if not valid_test.any():
        raise ValueError('No usable test pairs')
    predictions = pd.Series(model.predict(test.loc[valid_test, feature_columns]), index=y_test.loc[valid_test].index)
    metrics = metric_record(y_test.loc[valid_test], predictions)
    baseline = metric_record(y_test.loc[valid_test], test.loc[valid_test, args.target])
    per_station = []
    for station, subset in test.groupby('station'):
        valid = valid_test.loc[subset.index]
        ids = subset.index[valid]
        per_station.append({'station': station, 'slots': len(subset),
            'train_slots': int(train.station.eq(station).sum()),
            'train_pairs': int((valid_train & train.station.eq(station)).sum()),
            'forecast_slots': int(eligible_test.loc[subset.index].sum()),
            'model': metric_record(y_test.loc[ids], predictions.loc[ids]),
            'persistence': metric_record(y_test.loc[ids], test.loc[ids, args.target])})

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "metadata": {
            "target": args.target,
            "horizon": args.horizon,
            "model": "HistGradientBoostingRegressor",
            "model_version": args.model_version,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_start": args.test_start,
            "train_start": args.train_start,
            "test_end": args.test_end,
            "station_codes": station_codes,
            "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "feature_schema_sha256": hashlib.sha256(Path(args.feature_schema).read_bytes()).hexdigest() if args.feature_schema else None,
            "data_preparation": json.loads(Path(args.data_manifest).read_text()) if args.data_manifest else None,
            "train_target_start_min": str((pd.to_datetime(train.time_utc,utc=True)+delta).min()),
            "train_target_end_max": str((pd.to_datetime(train.time_utc,utc=True)+delta+step).max()),
            "dataset_step": args.dataset_step,
            "split_policy": "train target interval ends <= test-start; test target interval starts >= test-start",
            "train_rows": int(valid_train.sum()),
            "test_rows": int(valid_test.sum()),
            "metrics": metrics,
            "persistence": baseline,
            "test_slots": len(test),
            "forecast_slots": int(eligible_test.sum()),
            "per_station": per_station,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    output.with_suffix('.metrics.json').write_text(json.dumps(artifact['metadata'], indent=2, allow_nan=False), encoding='utf-8')
    exported = test[['station', 'time_utc', args.target, target_column]].copy()
    exported['prediction'] = np.nan
    if eligible_test.any():
        exported.loc[eligible_test, 'prediction'] = model.predict(test.loc[eligible_test, feature_columns])
    exported.to_csv(output.with_suffix('.predictions.csv'), index=False)
    print(json.dumps(artifact["metadata"], indent=2))


def safe_name(value: str) -> str:
    return value.replace(" ", "").replace("/", "_").replace("-", "_")


def load_training_frame(
    paths: list[Path],
    target_column: str,
    max_rows: int,
    rows_per_station: int,
    random_state: int,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for path in paths:
        wanted = set(feature_columns or []) | {target_column, 'time_utc', 'station'}
        frame = pd.read_csv(path, nrows=rows_per_station if rows_per_station > 0 else None,
            usecols=(lambda c: c in wanted) if feature_columns else None,
            dtype={c:'float32' for c in [*(feature_columns or []),target_column]} if feature_columns else None)
        if target_column not in frame.columns:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"No feature files contain target column {target_column!r}.")
    frame = pd.concat(frames, ignore_index=True)
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=random_state).sort_values("time_utc")
    return frame.reset_index(drop=True)


def split_train_test(frame: pd.DataFrame, test_start: str, horizon: str = "15min",
                     dataset_step: str = "15min") -> tuple[pd.DataFrame, pd.DataFrame]:
    time = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    start = pd.Timestamp(test_start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    # CSV timestamps label interval starts. A shifted target is not available
    # until the END of its interval, even if its feature row predates the split.
    step, delta = pd.Timedelta(dataset_step), pd.Timedelta(horizon)
    if step <= pd.Timedelta(0) or delta < step or delta % step != pd.Timedelta(0):
        raise ValueError("horizon must be a positive multiple of dataset-step")
    target_start = time + delta
    target_end = target_start + step
    return frame.loc[target_end <= start].copy(), frame.loc[target_start >= start].copy()


def numeric_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    excluded = {"hour", "month", "doy", "actual", "prediction", "persistence", "partition"}
    target_prefix = f"{target}_target_"
    columns = []
    for column in frame.select_dtypes(include=np.number).columns:
        if column in excluded:
            continue
        if column.startswith(target_prefix) or any(marker in column for marker in FUTURE_DERIVED_MARKERS):
            continue
        columns.append(column)
    return columns


def metric_record(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | None]:
    valid = pd.DataFrame({"actual": y_true, "predicted": y_pred}).dropna()
    if valid.empty:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "corr": None, "bias": None}
    error = valid["predicted"] - valid["actual"]
    return {
        "n": int(len(valid)),
        "mae": float(mean_absolute_error(valid["actual"], valid["predicted"])),
        "rmse": float(np.sqrt(mean_squared_error(valid["actual"], valid["predicted"]))),
        "r2": float(r2_score(valid["actual"], valid["predicted"])) if len(valid) > 1 else None,
        "corr": float(valid["actual"].corr(valid["predicted"])) if len(valid) > 1 and valid.actual.std() > 0 and valid.predicted.std() > 0 else None,
        "bias": float(error.mean()),
    }


if __name__ == "__main__":
    main()
