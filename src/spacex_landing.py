"""Train classifiers on SpaceX Falcon 9 booster-recovery outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .prepare_spacex import DATA, FEATURES, TARGET
from .train import ARTIFACTS

DATA_PATH = DATA / "spacex_falcon9_training.csv"
CATEGORICAL = ["launch_site", "orbit", "booster_category"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]


def _pipeline(classifier):
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", prep),
        ("feature_selection", SelectPercentile(f_classif, percentile=85)),
        ("classifier", classifier),
    ])


def _metric_row(y, probability, threshold=0.5):
    predicted = probability >= threshold
    return {
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predicted)), 4),
        "failure_precision": round(float(precision_score(y, predicted, zero_division=0)), 4),
        "failure_recall": round(float(recall_score(y, predicted, zero_division=0)), 4),
        "failure_f1": round(float(f1_score(y, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "brier_score": round(float(brier_score_loss(y, probability)), 4),
    }


def _shap_values(model, X_train, X_test):
    prep = model.named_steps["preprocess"]
    selector = model.named_steps["feature_selection"]
    classifier = model.named_steps["classifier"]
    train_matrix = prep.transform(X_train)
    test_matrix = prep.transform(X_test)
    keep = selector.get_support()
    names = prep.get_feature_names_out()[keep].tolist()
    background = train_matrix[: min(60, train_matrix.shape[0])][:, keep]
    sample = test_matrix[: min(60, test_matrix.shape[0])][:, keep]
    if hasattr(background, "toarray"):
        background = background.toarray()
        sample = sample.toarray()
    if isinstance(classifier, XGBClassifier):
        import xgboost as xgb
        contribution = classifier.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)[:, :-1]
    else:
        import shap
        explanation = shap.Explainer(classifier, background, feature_names=names)(sample)
        contribution = np.asarray(explanation.values)
        if contribution.ndim == 3:
            contribution = contribution[:, :, 1]
    importance = np.abs(contribution).mean(axis=0)
    ranked = sorted(zip(names, importance), key=lambda item: item[1], reverse=True)
    global_rows = [{"feature": str(name), "mean_abs_shap": round(float(value), 6)} for name, value in ranked]
    examples = [{
        "row": index,
        "attributions": [
            {"feature": str(name), "shap_value": round(float(value), 6)}
            for name, value in sorted(zip(names, contribution[index]), key=lambda item: abs(item[1]), reverse=True)[:8]
        ],
    } for index in range(min(20, len(contribution)))]
    return global_rows, examples


def train_spacex_landing(path: Path = DATA_PATH) -> dict:
    frame = pd.read_csv(path)
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError("Prepared Falcon 9 data is missing: " + ", ".join(missing))
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)
    X = frame[FEATURES]
    # Positive class is a failed/unrecovered first stage.
    y = 1 - pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 10:
        raise ValueError("Need at least 10 successful-recovery and 10 non-recovery examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42,
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced_subsample", max_features="sqrt", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=180, max_depth=3, learning_rate=0.04, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=4),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison, fitted = [], {}
    scorers = {"roc_auc": "roc_auc", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"}
    for name, estimator in candidates.items():
        pipe = _pipeline(estimator)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scorers, n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        comparison.append({
            "model": name,
            "cross_validation": {
                f"{metric}_mean": round(float(scores[f"test_{metric}"].mean()), 4)
                for metric in scorers
            } | {
                f"{metric}_std": round(float(scores[f"test_{metric}"].std()), 4)
                for metric in scorers
            },
            "test": _metric_row(y_test, pipe.predict_proba(X_test)[:, 1]),
        })

    winner = max(comparison, key=lambda row: row["cross_validation"]["roc_auc_mean"])["model"]
    calibrated = CalibratedClassifierCV(estimator=_pipeline(candidates[winner]), method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    probability = calibrated.predict_proba(X_test)[:, 1]
    final = _metric_row(y_test, probability)
    observed, predicted = calibration_curve(y_test, probability, n_bins=5, strategy="quantile")
    final.update({
        "selected_model": winner,
        "threshold": 0.5,
        "threshold_selection": "Fixed 0.50 probability cutoff; not tuned on the held-out test set.",
        "dataset_rows": int(len(frame)),
        "test_rows": int(len(y_test)),
        "failure_rate": round(float(y.mean()), 4),
        "always_success_test_accuracy": round(float((y_test == 0).mean()), 4),
        "confusion_matrix_labels": ["recovered", "not_recovered"],
        "confusion_matrix": confusion_matrix(y_test, probability >= 0.5, labels=[0, 1]).tolist(),
        "calibration_bins": [
            {"mean_predicted_probability": round(float(p), 4), "observed_failure_rate": round(float(o), 4)}
            for p, o in zip(predicted, observed)
        ],
        "split_method": "Stratified random 75/25 holdout; models compared by 5-fold stratified CV on training rows. This small dataset is not sufficient for a strong future-launch generalization claim.",
        "data_note": "Falcon 9 first-stage landing/recovery classification. Reanalysis weather is sampled at the launch site one hour before launch. Reuse fields are proxies; no engine telemetry is public in this dataset.",
    })

    try:
        shap_global, shap_examples = _shap_values(fitted[winner], X_train, X_test)
        shap_status = {"available": True, "message": "SHAP explanations generated."}
    except Exception as exc:
        shap_global, shap_examples = [], []
        shap_status = {"available": False, "message": str(exc)}

    outcome = np.where(y == 1, "failure", "success")
    training_data = frame[FEATURES].copy()
    training_data["outcome"] = outcome
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, ARTIFACTS / "launch_risk_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(final, indent=2))
    (ARTIFACTS / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    (ARTIFACTS / "shap_importance.json").write_text(json.dumps(shap_global, indent=2))
    (ARTIFACTS / "shap_examples.json").write_text(json.dumps(shap_examples, indent=2))
    (ARTIFACTS / "shap_status.json").write_text(json.dumps(shap_status, indent=2))
    training_data.to_csv(ARTIFACTS / "training_data.csv", index=False)
    numeric_data = training_data[NUMERIC]
    eda = {
        "rows": int(len(frame)), "failure_rate": round(float(y.mean()), 4),
        "outcome_counts": pd.Series(outcome).value_counts().to_dict(),
        "missing_values": training_data.isna().sum().to_dict(),
        "numeric_summary": numeric_data.describe().round(2).to_dict(),
        "correlation": numeric_data.corr().round(2).to_dict(),
    }
    (ARTIFACTS / "eda_summary.json").write_text(json.dumps(eda, indent=2))
    medians = numeric_data.median(numeric_only=True).fillna(0).to_dict()
    metadata = {
        "dataset_type": "spacex_landing",
        "source_file": "spacex_falcon9_training.csv",
        "source_url": "https://github.com/Roderic19/IBM-Applied-Data-Science-Capstone/blob/main/spacex_web_scraped.csv",
        "weather_source_url": "https://open-meteo.com/en/docs/historical-weather-api",
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "categories": {name: sorted(training_data[name].dropna().astype(str).unique().tolist()) for name in CATEGORICAL},
        "numeric_defaults": {name: float(medians.get(name, 0)) for name in NUMERIC},
        "target": "First-stage recovery success (landing and recovery recorded as Success)",
        "target_counts": pd.Series(outcome).value_counts().to_dict(),
        "years": [int(pd.to_datetime(frame["launch_time_utc"], utc=True).dt.year.min()), int(pd.to_datetime(frame["launch_time_utc"], utc=True).dt.year.max())],
        "engine_proxy_note": "Booster block, reuse marker, and previously flown count are public condition proxies, not engine telemetry.",
        "weather_note": "ERA5 hourly reanalysis sampled at the launch site at the previous full UTC hour; this is historical reconstructed weather, not a launch forecast.",
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"metrics": final, "comparison": comparison, "shap_importance": shap_global}
