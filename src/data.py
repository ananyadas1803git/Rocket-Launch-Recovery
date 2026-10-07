"""Launch data integration, synthetic demo data and input validation.

The generated rows are educational simulations, not historical launch records.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "payload_mass_kg",
    "air_temp_c",
    "wind_speed_m_s",
    "precipitation_mm",
    "humidity_pct",
    "engine_temp_c",
    "chamber_pressure_mpa",
    "vibration_mm_s",
    "engine_health_score",
    "rocket_family",
    "launch_site",
]
NUMERIC_FEATURES = FEATURE_COLUMNS[:9]
CATEGORICAL_FEATURES = FEATURE_COLUMNS[9:]
TARGET_COLUMN = "outcome"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def integrate_sources(launches: pd.DataFrame, weather: pd.DataFrame, engine: pd.DataFrame) -> pd.DataFrame:
    """Join launch, weather and engine tables on one unique launch_id per table."""
    sources = {"launches": launches, "weather": weather, "engine": engine}
    for name, frame in sources.items():
        if "launch_id" not in frame.columns:
            raise ValueError(f"{name} source must include a launch_id column.")
        if frame["launch_id"].isna().any() or frame["launch_id"].duplicated().any():
            raise ValueError(f"{name} source must have exactly one non-empty row per launch_id.")
    launch_ids = set(launches["launch_id"])
    weather_ids = set(weather["launch_id"])
    engine_ids = set(engine["launch_id"])
    retained_ids = launch_ids & weather_ids & engine_ids
    dropped = len(launch_ids - retained_ids)
    if dropped:
        warnings.warn(
            f"Integration retained {len(retained_ids)} matched launches and excluded {dropped} launches without both weather and engine rows.",
            stacklevel=2,
        )
    merged = launches.merge(weather, on="launch_id", how="inner", validate="one_to_one")
    merged = merged.merge(engine, on="launch_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No matching launch_id values across all three source files.")
    return merged


def generate_demo_data(rows: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible, plausible-looking dataset for demonstrating a workflow."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2008-01-01", periods=rows, freq="5D")
    sites = rng.choice(["Coastal", "Inland", "Island"], rows, p=[0.45, 0.30, 0.25])
    families = rng.choice(["Nova-I", "Aster-II", "Vega-M"], rows, p=[0.42, 0.36, 0.22])
    payload = np.clip(rng.lognormal(7.9, 0.72, rows), 250, 24000)
    temp = np.clip(rng.normal(18, 12, rows), -22, 47)
    wind = np.clip(rng.gamma(2.2, 4.1, rows), 0, 34)
    precip = np.where(rng.random(rows) < 0.76, 0, rng.gamma(1.7, 2.4, rows))
    humidity = np.clip(rng.normal(56, 19, rows), 8, 100)
    engine_temp = np.clip(rng.normal(720, 54, rows), 560, 900)
    pressure = np.clip(rng.normal(10.4, 0.82, rows), 7.2, 13.2)
    vibration = np.clip(rng.lognormal(1.6, 0.48, rows), 1, 18)
    health = np.clip(rng.beta(10, 2, rows) * 100, 40, 100)

    # A transparent simulated failure mechanism gives the demo learnable signal.
    log_odds = (
        -3.15
        + 0.085 * np.maximum(wind - 12, 0)
        + 0.18 * np.maximum(precip - 1, 0)
        + 0.012 * np.abs(temp - 18)
        + 0.025 * np.maximum(engine_temp - 770, 0)
        + 0.75 * np.abs(pressure - 10.4)
        + 0.19 * np.maximum(vibration - 5, 0)
        + 0.075 * np.maximum(78 - health, 0)
        + 0.000018 * payload
    )
    failure_probability = 1 / (1 + np.exp(-log_odds))
    # Reduce probabilities to keep a rare-event classification challenge.
    failure_probability = np.clip(failure_probability * 0.48, 0.015, 0.72)
    failed = rng.binomial(1, failure_probability)

    return pd.DataFrame(
        {
            "launch_date": dates,
            "payload_mass_kg": payload.round(1),
            "air_temp_c": temp.round(1),
            "wind_speed_m_s": wind.round(1),
            "precipitation_mm": precip.round(1),
            "humidity_pct": humidity.round(1),
            "engine_temp_c": engine_temp.round(1),
            "chamber_pressure_mpa": pressure.round(2),
            "vibration_mm_s": vibration.round(2),
            "engine_health_score": health.round(1),
            "rocket_family": families,
            "launch_site": sites,
            "outcome": np.where(failed == 1, "failure", "success"),
        }
    )


def ensure_demo_csv(path: Path, rows: int = 2400, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data(rows, seed).to_csv(path, index=False)
    return path


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the model input contract and return a safe, normalized copy."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    clean = frame[REQUIRED_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        original = clean[col]
        clean[col] = pd.to_numeric(original, errors="coerce")
        malformed = original.notna() & clean[col].isna()
        if malformed.any():
            raise ValueError(f"{col} contains non-numeric values; use blank cells for missing measurements.")
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(str).str.strip().str.lower()
    unexpected = sorted(set(clean[TARGET_COLUMN]) - {"success", "failure"})
    if unexpected:
        raise ValueError("outcome must contain only 'success' or 'failure'.")
    if clean[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Both success and failure examples are required to train the model.")
    if not clean["engine_health_score"].dropna().between(0, 100).all():
        raise ValueError("engine_health_score must be between 0 and 100.")
    if not clean["humidity_pct"].dropna().between(0, 100).all():
        raise ValueError("humidity_pct must be between 0 and 100.")
    return clean
