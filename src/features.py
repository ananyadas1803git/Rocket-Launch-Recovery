"""Leakage-safe engineering of a few interpretable interaction features."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LaunchFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds scale/stress indicators while retaining the original input columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        result["log_payload_mass"] = np.log1p(result["payload_mass_kg"].clip(lower=0))
        result["wind_precipitation_interaction"] = result["wind_speed_m_s"] * result["precipitation_mm"]
        result["engine_stress_index"] = (
            (100 - result["engine_health_score"]).clip(lower=0)
            + result["vibration_mm_s"] * 2
        )
        result["thermal_pressure_interaction"] = (
            (result["engine_temp_c"] - 720).abs()
            * (result["chamber_pressure_mpa"] - 10.4).abs()
        )
        return result

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + [
            "log_payload_mass", "wind_precipitation_interaction",
            "engine_stress_index", "thermal_pressure_interaction",
        ], dtype=object)
