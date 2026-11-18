"""Synthetic kiln data and reusable feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "pottery_load_kg",
    "target_temperature_c",
    "initial_moisture_pct",
    "planned_duration_hours",
    "insulation_rating",
    "kiln_volume_m3",
    "ambient_temperature_c",
]
ENGINEERED_FEATURES = ["thermal_load_index", "moisture_removal_load"]
MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
TARGET = "energy_consumption_kwh"


def generate_dataset(n_records: int = 300, version: int = 1, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic dataset for a kiln fleet version."""
    rng = np.random.default_rng(seed + version)
    load = rng.uniform(20, 250, n_records)
    temperature = rng.uniform(750, 1250, n_records)
    moisture = rng.uniform(2, 18, n_records)
    duration = rng.uniform(4, 18, n_records)
    insulation = rng.uniform(0.75, 0.98, n_records)
    volume = rng.uniform(1.0, 5.0, n_records)
    ambient = rng.uniform(15, 38, n_records)
    if version == 2:
        insulation = np.clip(insulation - 0.12, 0.35, 0.98)
        efficiency_penalty = 1.18
    else:
        efficiency_penalty = 1.0

    thermal = load * (temperature - ambient) / 1000
    moisture_load = load * moisture / 100
    energy = (
        7
        + 2.2 * thermal
        + 0.85 * moisture_load
        + 1.6 * duration
        + 0.7 * volume
        + 0.018 * (temperature - 700)
    ) * (1 + (1 - insulation) * 1.8) * efficiency_penalty
    energy += rng.normal(0, 2.0, n_records)

    frame = pd.DataFrame(
        {
            "dataset_version": f"v{version}",
            "pottery_load_kg": load,
            "target_temperature_c": temperature,
            "initial_moisture_pct": moisture,
            "planned_duration_hours": duration,
            "insulation_rating": insulation,
            "kiln_volume_m3": volume,
            "ambient_temperature_c": ambient,
            TARGET: np.maximum(energy, 0.1),
        }
    )
    return add_engineered_features(frame)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add and return the shared thermal and moisture load features."""
    result = frame.copy()
    result["thermal_load_index"] = (
        result["pottery_load_kg"]
        * (result["target_temperature_c"] - result["ambient_temperature_c"])
        / 1000
    )
    result["moisture_removal_load"] = (
        result["pottery_load_kg"] * result["initial_moisture_pct"] / 100
    )
    return result


def build_feature_store(output_dir: str = "artifacts") -> pd.DataFrame:
    """Create both dataset versions and persist reusable engineered features."""
    data_dir = Path("data")
    Path(output_dir).mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    v1 = generate_dataset(version=1)
    v2 = generate_dataset(version=2)
    v1.to_csv(data_dir / "kiln_batches_v1.csv", index=False)
    v2.to_csv(data_dir / "kiln_batches_v2.csv", index=False)
    feature_store = pd.concat([v1, v2], ignore_index=True)
    feature_store.to_csv(Path(output_dir) / "feature_store.csv", index=False)
    return feature_store
