"""Integrate, engineer, compare, calibrate and explain launch classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .data import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES,
    ensure_demo_csv, integrate_sources, validate_dataset,
)
from .features import LaunchFeatureEngineer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "spacex_falcon9_training.csv"
SYNTHETIC_DATA_PATH = ROOT / "data" / "launches_demo.csv"
ARTIFACTS = ROOT / "artifacts"
ENGINEERED_NUMERIC = NUMERIC_FEATURES + [
    "log_payload_mass", "wind_precipitation_interaction",
    "engine_stress_index", "thermal_pressure_interaction",
]


def build_model(classifier) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, ENGINEERED_NUMERIC),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("feature_engineering", LaunchFeatureEngineer()),
        ("preprocess", preprocess),
        ("feature_selection", SelectPercentile(score_func=f_classif, percentile=80)),
        ("classifier", classifier),
    ])


def model_candidates(failure_rate: float) -> dict:
    scale_pos_weight = (1 - failure_rate) / max(failure_rate, 1e-6)
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1500, solver="liblinear", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=320, min_samples_leaf=3, class_weight="balanced_subsample",
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=320, max_depth=3, learning_rate=0.04, min_child_weight=3,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, tree_method="hist",
            random_state=42, n_jobs=4,
        ),
    }


def _metrics(y_true, probabilities, threshold=0.5):
    predictions = probabilities >= threshold
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 4),
        "failure_precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "failure_recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "failure_f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
    }


def _positive_shap_values(explanation):
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def _explain_model(fitted_pipeline, X_train, X_test):
    """Calculate global SHAP importance for the selected classifier."""
    engineer = fitted_pipeline.named_steps["feature_engineering"]
    prep = fitted_pipeline.named_steps["preprocess"]
    selector = fitted_pipeline.named_steps["feature_selection"]
    classifier = fitted_pipeline.named_steps["classifier"]
    transformed_train = prep.transform(engineer.transform(X_train))
    transformed_test = prep.transform(engineer.transform(X_test))
    selected = selector.get_support()
    feature_names = prep.get_feature_names_out()[selected].tolist()
    background = transformed_train[: min(80, transformed_train.shape[0])][:, selected]
    samples = transformed_test[: min(120, transformed_test.shape[0])][:, selected]
    if hasattr(background, "toarray"):
        background = background.toarray()
    if hasattr(samples, "toarray"):
        samples = samples.toarray()
    if isinstance(classifier, XGBClassifier):
        # XGBoost exposes its TreeSHAP contributions through pred_contribs.
        import xgboost as xgb
        values = classifier.get_booster().predict(xgb.DMatrix(samples), pred_contribs=True)[:, :-1]
    else:
        import shap
        explainer = shap.Explainer(classifier, background, feature_names=feature_names)
        explanation = explainer(samples)
        values = _positive_shap_values(explanation)
    mean_abs = np.abs(values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda item: item[1], reverse=True)
    global_summary = [
        {"feature": str(name), "mean_abs_shap": round(float(score), 6)}
        for name, score in ranked
    ]
    sample_rows = []
    for sample_i in range(min(20, len(values))):
        sample_rows.append({
            "row": int(sample_i),
            "attributions": [
                {"feature": str(name), "shap_value": round(float(value), 6)}
                for name, value in sorted(
                    zip(feature_names, values[sample_i]), key=lambda item: abs(item[1]), reverse=True
                )[:8]
            ],
        })
    return global_summary, sample_rows


def train(data_path: Path | None = None) -> dict:
    if data_path is None:
        data_path = DATA_PATH if DATA_PATH.exists() else SYNTHETIC_DATA_PATH
        if not data_path.exists():
            ensure_demo_csv(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Launch dataset not found: {data_path}")
    headers = set(pd.read_csv(data_path, nrows=0).columns)
    if {"recovery_success", "payload_mass_kg", "air_temp_c_tminus_1h"}.issubset(headers):
        from .spacex_landing import train_spacex_landing
        return train_spacex_landing(data_path)
    if {"Organisation", "Location", "Date", "Detail", "Mission_Status"}.issubset(headers):
        from .public_history import train_spaceflight_history
        return train_spaceflight_history(data_path)
    clean = validate_dataset(pd.read_csv(data_path))
    X = clean[FEATURE_COLUMNS]
    y = clean["outcome"].map({"success": 0, "failure": 1})
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development, y_development, test_size=0.25,
        stratify=y_development, random_state=43,
    )

    failure_rate = float(y_train.mean())
    fitted = {}
    comparison = []
    for name, estimator in model_candidates(failure_rate).items():
        candidate = build_model(estimator)
        candidate.fit(X_train, y_train)
        validation_probability = candidate.predict_proba(X_validation)[:, 1]
        fitted[name] = candidate
        comparison.append({
            "model": name,
            "validation": _metrics(y_validation, validation_probability),
        })

    # Choose on validation ROC AUC, then reserve test data for final reporting.
    winner = max(comparison, key=lambda row: row["validation"]["roc_auc"])["model"]
    champion_base = fitted[winner]
    calibrated = CalibratedClassifierCV(
        estimator=build_model(model_candidates(failure_rate)[winner]),
        method="sigmoid", cv=3,
    )
    calibrated.fit(X_train, y_train)
    validation_probability = calibrated.predict_proba(X_validation)[:, 1]
    threshold_candidates = np.unique(np.r_[0.0, validation_probability, 1.0])
    threshold = max(
        threshold_candidates,
        key=lambda cut: (
            fbeta_score(y_validation, validation_probability >= cut, beta=2, zero_division=0),
            cut,
        ),
    )
    test_probability = calibrated.predict_proba(X_test)[:, 1]
    final_metrics = _metrics(y_test, test_probability, threshold)
    for row in comparison:
        base_test_probability = fitted[row["model"]].predict_proba(X_test)[:, 1]
        row["test"] = _metrics(y_test, base_test_probability)

    fraction_positive, mean_predicted = calibration_curve(
        y_test, test_probability, n_bins=8, strategy="quantile"
    )
    final_metrics.update({
        "dataset_rows": int(len(clean)),
        "test_rows": int(len(y_test)),
        "failure_rate": round(float(y.mean()), 4),
        "threshold": round(float(threshold), 4),
        "selected_model": winner,
        "accuracy": round(float(accuracy_score(y_test, test_probability >= threshold)), 4),
        "confusion_matrix_labels": ["success", "failure"],
        "confusion_matrix": confusion_matrix(y_test, test_probability >= threshold, labels=[0, 1]).tolist(),
        "calibration_bins": [
            {"mean_predicted_probability": round(float(p), 4), "observed_failure_rate": round(float(o), 4)}
            for p, o in zip(mean_predicted, fraction_positive)
        ],
        "threshold_selection": "Maximize F2 on validation data; test data remains held out.",
        "data_note": "Synthetic educational demo data; metrics are not operational evidence.",
    })

    # SHAP is computed for the validation-selected base estimator on its test inputs.
    try:
        shap_global, shap_examples = _explain_model(champion_base, X_train, X_test)
        shap_status = {"available": True, "message": "SHAP explanations generated."}
    except ImportError as exc:
        # Keep model training usable in a misconfigured local environment, while
        # making the missing explainability dependency visible in the dashboard.
        shap_global, shap_examples = [], []
        shap_status = {"available": False, "message": str(exc)}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, ARTIFACTS / "launch_risk_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
    (ARTIFACTS / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    (ARTIFACTS / "shap_importance.json").write_text(json.dumps(shap_global, indent=2))
    (ARTIFACTS / "shap_examples.json").write_text(json.dumps(shap_examples, indent=2))
    (ARTIFACTS / "shap_status.json").write_text(json.dumps(shap_status, indent=2))
    eda = {
        "rows": int(len(clean)),
        "failure_rate": round(float(y.mean()), 4),
        "outcome_counts": clean["outcome"].value_counts().to_dict(),
        "missing_values": clean.isna().sum().to_dict(),
        "numeric_summary": clean[NUMERIC_FEATURES].describe().round(2).to_dict(),
        "correlation": clean[NUMERIC_FEATURES].corr().round(2).to_dict(),
    }
    clean.to_csv(ARTIFACTS / "training_data.csv", index=False)
    (ARTIFACTS / "eda_summary.json").write_text(json.dumps(eda, indent=2))
    return {"metrics": final_metrics, "comparison": comparison, "shap_importance": shap_global}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train launch outcome model comparison")
    parser.add_argument("--data", type=Path, help="Public historical launch CSV or integrated launch-feature CSV")
    parser.add_argument("--launches", type=Path, help="Launch/outcome CSV for integration on launch_id")
    parser.add_argument("--weather", type=Path, help="Weather CSV for integration on launch_id")
    parser.add_argument("--engine", type=Path, help="Engine condition CSV for integration on launch_id")
    args = parser.parse_args()
    has_sources = any((args.launches, args.weather, args.engine))
    if has_sources:
        if not all((args.launches, args.weather, args.engine)):
            parser.error("--launches, --weather, and --engine must be supplied together")
        data_path = args.data or ROOT / "data" / "integrated.csv"
        integrated = integrate_sources(
            pd.read_csv(args.launches), pd.read_csv(args.weather), pd.read_csv(args.engine)
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        integrated.to_csv(data_path, index=False)
    else:
        data_path = args.data or DATA_PATH
    result = train(data_path)
    print(json.dumps(result["metrics"], indent=2))
