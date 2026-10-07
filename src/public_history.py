"""Model the published global space-mission launch history dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .train import ARTIFACTS, _metrics

RAW_FEATURES = ["organisation", "launch_site", "rocket_family", "launch_year", "launch_month"]
CATEGORICAL = ["organisation", "launch_site", "rocket_family"]
NUMERIC = ["launch_year", "launch_month"]


def prepare_public_history(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"Organisation", "Location", "Date", "Detail", "Mission_Status"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Historical launch file is missing columns: " + ", ".join(missing))
    frame = raw.copy()
    frame["launch_datetime"] = pd.to_datetime(frame["Date"], format="mixed", utc=True, errors="coerce")
    frame["outcome_raw"] = frame["Mission_Status"].astype(str).str.strip().str.casefold()
    status_map = {
        "success": "success",
        "failure": "failure",
        "partial failure": "failure",
        "prelaunch failure": "failure",
    }
    frame["outcome"] = frame["outcome_raw"].map(status_map)
    frame["organisation"] = frame["Organisation"].fillna("Unknown").astype(str).str.strip()
    frame["launch_site"] = frame["Location"].fillna("Unknown").astype(str).str.strip()
    detail = frame["Detail"].fillna("Unknown").astype(str)
    frame["rocket_family"] = detail.str.split("|", n=1, regex=False).str[0].str.strip().replace("", "Unknown")
    frame["launch_year"] = frame["launch_datetime"].dt.year
    frame["launch_month"] = frame["launch_datetime"].dt.month
    frame = frame.dropna(subset=["launch_datetime", "outcome"])
    frame = frame.sort_values("launch_datetime", kind="stable").reset_index(drop=True)
    if frame["outcome"].nunique() != 2 or len(frame) < 50:
        raise ValueError("Need at least 50 dated launch rows with both success and failure outcomes.")
    return frame


def _build_model(classifier):
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", prep),
        ("feature_selection", SelectPercentile(f_classif, percentile=80)),
        ("classifier", classifier),
    ])


def _explain(model, X_train, X_test):
    prep = model.named_steps["preprocess"]
    selector = model.named_steps["feature_selection"]
    classifier = model.named_steps["classifier"]
    train_matrix = prep.transform(X_train)
    test_matrix = prep.transform(X_test)
    selected = selector.get_support()
    names = prep.get_feature_names_out()[selected].tolist()
    samples = test_matrix[: min(150, test_matrix.shape[0])][:, selected]
    if hasattr(samples, "toarray"):
        samples = samples.toarray()
    if isinstance(classifier, XGBClassifier):
        import xgboost as xgb
        contributions = classifier.get_booster().predict(xgb.DMatrix(samples), pred_contribs=True)[:, :-1]
    else:
        import shap
        background = train_matrix[: min(80, train_matrix.shape[0])][:, selected]
        if hasattr(background, "toarray"):
            background = background.toarray()
        explanation = shap.Explainer(classifier, background, feature_names=names)(samples)
        contributions = np.asarray(explanation.values)
        if contributions.ndim == 3:
            contributions = contributions[:, :, 1]
    importance = np.abs(contributions).mean(axis=0)
    ranked = sorted(zip(names, importance), key=lambda row: row[1], reverse=True)
    global_values = [
        {"feature": str(name), "mean_abs_shap": round(float(value), 6)} for name, value in ranked
    ]
    examples = [{
        "row": index,
        "attributions": [
            {"feature": str(name), "shap_value": round(float(value), 6)}
            for name, value in sorted(zip(names, contributions[index]), key=lambda row: abs(row[1]), reverse=True)[:8]
        ],
    } for index in range(min(20, len(contributions)))]
    return global_values, examples


def train_spaceflight_history(path: Path) -> dict:
    raw = pd.read_csv(path, encoding="utf-8")
    history = prepare_public_history(raw)
    X = history[RAW_FEATURES]
    y = history["outcome"].map({"success": 0, "failure": 1})
    n = len(history)
    # Allocate more records to fitting so the model includes post-1980 launch
    # history, while reserving distinct later periods for calibration,
    # threshold choice, and the final out-of-time test.
    fit_end, calibration_end, validation_end = int(n * 0.65), int(n * 0.75), int(n * 0.85)
    X_train, X_calibration = X.iloc[:fit_end], X.iloc[fit_end:calibration_end]
    X_validation, X_test = X.iloc[calibration_end:validation_end], X.iloc[validation_end:]
    y_train, y_calibration = y.iloc[:fit_end], y.iloc[fit_end:calibration_end]
    y_validation, y_test = y.iloc[calibration_end:validation_end], y.iloc[validation_end:]
    splits = [y_train, y_calibration, y_validation, y_test]
    if any(min(part.sum(), (part == 0).sum()) < 2 for part in splits):
        raise ValueError("Chronological split does not contain enough examples of each outcome in each period.")

    failure_rate = float(y_train.mean())
    scale = (1 - failure_rate) / max(failure_rate, 1e-6)
    candidates = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", max_features="sqrt", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss", scale_pos_weight=scale, tree_method="hist", random_state=42, n_jobs=4),
    }
    fitted, comparison = {}, []
    for name, classifier in candidates.items():
        model = _build_model(classifier)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_validation)[:, 1]
        fitted[name] = model
        comparison.append({"model": name, "validation": _metrics(y_validation, probability)})

    winner = max(comparison, key=lambda row: row["validation"]["roc_auc"])["model"]
    # Fit calibration on its own later-in-time period; preserve the trained model's ranking.
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted[winner]), method="sigmoid"
    )
    calibrated.fit(X_calibration, y_calibration)
    val_prob = calibrated.predict_proba(X_validation)[:, 1]
    thresholds = np.unique(np.r_[0.0, val_prob, 1.0])
    # Keep both operating points: use the accuracy-maximizing threshold for the
    # user's stated objective, and report the recall-oriented alternative too.
    accuracy_threshold = max(
        thresholds,
        key=lambda cut: (accuracy_score(y_validation, val_prob >= cut), cut),
    )
    f2_threshold = max(
        thresholds,
        key=lambda cut: (fbeta_score(y_validation, val_prob >= cut, beta=2, zero_division=0), cut),
    )
    test_prob = calibrated.predict_proba(X_test)[:, 1]
    test_metrics = _metrics(y_test, test_prob, accuracy_threshold)
    recall_focused_metrics = _metrics(y_test, test_prob, f2_threshold)
    test_success_baseline = float((y_test == 0).mean())
    for row in comparison:
        row["test"] = _metrics(y_test, fitted[row["model"]].predict_proba(X_test)[:, 1])
    observed, predicted = calibration_curve(y_test, test_prob, n_bins=8, strategy="quantile")
    test_metrics.update({
        "dataset_rows": int(len(history)), "test_rows": int(len(y_test)),
        "failure_rate": round(float(y.mean()), 4), "threshold": round(float(accuracy_threshold), 4),
        "threshold_objective": "Maximize accuracy on the chronological validation period; ties favor fewer failure flags.",
        "validation_selected_accuracy": round(float(accuracy_score(y_validation, val_prob >= accuracy_threshold)), 4),
        "always_success_test_accuracy": round(test_success_baseline, 4),
        "recall_focused_threshold": round(float(f2_threshold), 4),
        "recall_focused_test_metrics": recall_focused_metrics,
        "selected_model": winner,
        "accuracy": round(float(accuracy_score(y_test, test_prob >= accuracy_threshold)), 4),
        "confusion_matrix_labels": ["success", "failure"],
        "confusion_matrix": confusion_matrix(y_test, test_prob >= accuracy_threshold, labels=[0, 1]).tolist(),
        "calibration_bins": [
            {"mean_predicted_probability": round(float(p), 4), "observed_failure_rate": round(float(o), 4)}
            for p, o in zip(predicted, observed)
        ],
        "threshold_selection": "The deployed threshold maximizes accuracy on a chronological validation period after a separate chronological calibration period; the latest 15% is held out for testing. A separate F2-optimized threshold is reported to show the recall tradeoff.",
        "data_note": "Real mission history through 2020. The source has no payload mass, weather, or engine telemetry.",
    })

    try:
        shap_values, shap_examples = _explain(fitted[winner], X_train, X_test)
        shap_status = {"available": True, "message": "SHAP explanations generated."}
    except ImportError as exc:
        shap_values, shap_examples = [], []
        shap_status = {"available": False, "message": str(exc)}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, ARTIFACTS / "launch_risk_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (ARTIFACTS / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    (ARTIFACTS / "shap_importance.json").write_text(json.dumps(shap_values, indent=2))
    (ARTIFACTS / "shap_examples.json").write_text(json.dumps(shap_examples, indent=2))
    (ARTIFACTS / "shap_status.json").write_text(json.dumps(shap_status, indent=2))
    training_frame = history[RAW_FEATURES + ["outcome"]]
    training_frame.to_csv(ARTIFACTS / "training_data.csv", index=False)
    eda = {
        "rows": int(len(history)), "failure_rate": round(float(y.mean()), 4),
        "outcome_counts": history["outcome"].value_counts().to_dict(),
        "missing_values": training_frame.isna().sum().to_dict(),
        "numeric_summary": history[NUMERIC].describe().round(2).to_dict(),
        "correlation": history[NUMERIC].corr().round(2).to_dict(),
    }
    (ARTIFACTS / "eda_summary.json").write_text(json.dumps(eda, indent=2))
    metadata = {
        "dataset_type": "historical_missions",
        "source_file": path.name,
        "source_url": "https://github.com/salmanadnan2257/space-race-analysis/blob/main/mission_launches.csv",
        "features": RAW_FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "years": [int(history["launch_year"].min()), int(history["launch_year"].max())],
        "fit_end_year": int(history["launch_year"].iloc[fit_end - 1]),
        "calibration_end_year": int(history["launch_year"].iloc[calibration_end - 1]),
        "validation_end_year": int(history["launch_year"].iloc[validation_end - 1]),
        "organisations": sorted(history["organisation"].unique().tolist()),
        "launch_sites": sorted(history["launch_site"].unique().tolist()),
        "rocket_families": sorted(history["rocket_family"].unique().tolist()),
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"metrics": test_metrics, "comparison": comparison}
