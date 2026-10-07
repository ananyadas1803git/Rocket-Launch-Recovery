"""Join public Falcon 9 launch records to pre-launch ERA5 weather features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_PATH = DATA / "spacex_falcon9_launches.csv"
WEATHER_PATH = DATA / "spacex_launch_weather_era5.csv"
OUTPUT_PATH = DATA / "spacex_falcon9_training.csv"

FEATURES = [
    "payload_mass_kg", "launch_site", "orbit", "booster_category",
    "booster_prior_flights", "booster_reused", "flight_number",
    "launch_month", "launch_hour_utc", "air_temp_c_tminus_1h",
    "humidity_pct_tminus_1h", "wind_speed_m_s_tminus_1h",
    "wind_gust_m_s_tminus_1h", "precipitation_mm_tminus_1h",
    "pressure_hpa_tminus_1h", "cloud_cover_pct_tminus_1h",
]
TARGET = "recovery_success"


def prepare_spaceflight_launches(raw: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Flight No.", "Date", "Time", "Launch site", "Payload mass", "Orbit",
        "Version Booster", "Booster landing", "Launch outcome",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Launch data is missing columns: " + ", ".join(missing))
    if "Flight No." not in weather:
        raise ValueError("Weather data must include Flight No. for the launch-time join.")

    frame = raw.copy()
    frame["flight_number"] = pd.to_numeric(frame["Flight No."], errors="coerce")
    frame["launch_time_utc"] = pd.to_datetime(
        frame["Date"].astype(str) + " " + frame["Time"].astype(str),
        format="mixed", dayfirst=True, utc=True, errors="coerce",
    )
    frame["launch_month"] = frame["launch_time_utc"].dt.month
    frame["launch_hour_utc"] = frame["launch_time_utc"].dt.hour
    frame["payload_mass_kg"] = pd.to_numeric(
        frame["Payload mass"].astype(str).str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce",
    )
    version = frame["Version Booster"].fillna("").astype(str)
    frame["booster_category"] = version.str.extract(
        r"(v1\.0|v1\.1|FT|B4|B5)", expand=False,
    ).fillna("unknown")
    frame["booster_prior_flights"] = pd.to_numeric(
        version.str.extract(r"\.(\d+)", expand=False), errors="coerce",
    ).sub(1).clip(lower=0)
    frame["booster_reused"] = (
        version.str.contains("♺", regex=False)
        | version.str.contains(r"\.(?:[2-9]|[1-9]\d)", regex=True)
    ).astype(int)
    frame["launch_site"] = frame["Launch site"].fillna("Unknown").astype(str).str.strip()
    frame["orbit"] = frame["Orbit"].fillna("Unknown").astype(str).str.strip()
    # Recovery success means the first stage was reported as successfully landed.
    # No-attempt and precluded missions are retained as non-recoveries.
    frame[TARGET] = frame["Booster landing"].fillna("").astype(str).str.strip().eq("Success").astype(int)

    weather_columns = [c for c in weather.columns if c not in {"Flight No.", "launch_time_utc"}]
    frame = frame.merge(weather[["Flight No.", *weather_columns]], on="Flight No.", how="left", validate="one_to_one")
    weather_features = [c for c in FEATURES if "tminus_1h" in c]
    if frame[weather_features].isna().any().any():
        missing_counts = frame[weather_features].isna().sum()
        nonzero = missing_counts[missing_counts > 0]
        if not nonzero.empty:
            raise ValueError("Missing model inputs after joining weather: " + ", ".join(f"{key}={value}" for key, value in nonzero.items()))
    result = frame[FEATURES + [TARGET]].copy()
    result["launch_time_utc"] = frame["launch_time_utc"].astype(str)
    result["weather_time_utc"] = frame.get("weather_time_utc", "")
    result["landing_outcome_raw"] = frame["Booster landing"].fillna("").astype(str).str.strip()
    result["mission_outcome_raw"] = frame.get("Launch outcome", "").astype(str).str.strip()
    return result.sort_values("launch_time_utc").reset_index(drop=True)


def main() -> None:
    launches = pd.read_csv(RAW_PATH)
    weather = pd.read_csv(WEATHER_PATH)
    prepared = prepare_spaceflight_launches(launches, weather)
    prepared.to_csv(OUTPUT_PATH, index=False)
    print(f"Prepared {len(prepared)} launches with {len(FEATURES)} recorded/derived inputs: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
