from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "production_models" / "fof2_next_15min.joblib"


@lru_cache(maxsize=1)
def load_production_model(path: str | None = None) -> dict[str, Any] | None:
    model_path = Path(path) if path else DEFAULT_MODEL_PATH
    if not model_path.exists():
        return None
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Unsupported model artifact type: {type(artifact)!r}")
    required_keys = {"model", "feature_columns", "metadata"}
    missing = required_keys - set(artifact)
    if missing:
        raise KeyError(f"Model artifact is missing keys: {sorted(missing)}")
    return artifact
