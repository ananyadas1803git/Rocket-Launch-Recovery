"""Interactive launch risk demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data import FEATURE_COLUMNS
from src.rag import generate_answer, retrieve
from src.train import ARTIFACTS, train

st.set_page_config(page_title="Rocket Launch Classifier", page_icon="🚀", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACTS / "launch_risk_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: Path):
    return json.loads(path.read_text())


model = load_model()
metadata = load_json(ARTIFACTS / "model_metadata.json")
metrics = load_json(ARTIFACTS / "metrics.json")
historical_mode = metadata.get("dataset_type") == "historical_missions"
landing_mode = metadata.get("dataset_type") == "spacex_landing"
if landing_mode:
    st.title("🚀 Rocket Launch Classifier")
    st.caption("A predictive engineering project for first-stage booster recovery outcomes.")
    st.warning(
        "This small public dataset predicts whether a Falcon 9 first stage was recovered. "
        "Weather comes from ERA5 reanalysis; booster generation/reuse are condition proxies. "
        "It contains no engine telemetry and is not a launch safety tool."
    )
elif historical_mode:
    st.title("🚀 Rocket Launch Classifier")
    st.caption("A predictive engineering demo for exploring historical mission outcomes.")
    st.warning(
        "This model is trained on historical mission outcomes through 2020. The source has "
        "no payload mass, pre-launch weather, or engine telemetry. "
        + ("The accuracy-optimized threshold missed every failure in the held-out period. " if metrics["failure_recall"] == 0 else "Its held-out results are not reliable launch guidance. ")
        + "Treat it as a retrospective learning demo, not launch guidance."
    )
else:
    st.title("🚀 Rocket Launch Classifier")
    st.caption("A predictive engineering demo for exploring launch outcome risk factors.")
    st.warning(
        "The bundled fallback training data is synthetic and for education only. Predictions "
        "are not validated for real launch decisions."
    )
default_threshold = float(metrics["threshold"])
with st.sidebar:
    st.header("Decision threshold")
    threshold = st.slider(
        "Flag a launch as higher risk at",
        min_value=0.001, max_value=1.0, value=min(1.0, max(0.001, round(default_threshold, 3))), step=0.001,
        help="Lower thresholds flag more launches for review and usually catch more failures, with more false alarms.",
    )
    st.caption(("Fixed default cutoff: " if landing_mode else "Validation-selected default: ") + f"{default_threshold:.1%}")
st.subheader("Score a launch using available pre-launch features")
with st.form("prediction_form"):
    if historical_mode:
        left, middle, right = st.columns(3)
        with left:
            organisation = st.selectbox("Launch organisation", metadata["organisations"])
            launch_year = st.number_input("Launch year", min_value=metadata["years"][0], max_value=metadata["years"][1], value=metadata["years"][1], step=1)
        with middle:
            launch_site = st.selectbox("Launch site", metadata["launch_sites"])
            launch_month = st.selectbox("Launch month", list(range(1, 13)), index=0)
        with right:
            rocket_family = st.selectbox("Rocket / vehicle family", metadata["rocket_families"])
    elif landing_mode:
        values = {}
        left, middle, right = st.columns(3)
        with left:
            values["launch_site"] = st.selectbox("Launch site", metadata["categories"]["launch_site"])
            values["orbit"] = st.selectbox("Target orbit", metadata["categories"]["orbit"])
            values["booster_category"] = st.selectbox("Booster generation", metadata["categories"]["booster_category"])
            values["payload_mass_kg"] = st.number_input("Payload mass (kg)", min_value=0.0, value=float(metadata["numeric_defaults"]["payload_mass_kg"]), step=100.0)
            values["flight_number"] = st.number_input("Falcon 9 flight number", min_value=1.0, value=float(metadata["numeric_defaults"]["flight_number"]), step=1.0)
        with middle:
            values["booster_prior_flights"] = st.number_input("Known prior booster flights", min_value=0.0, value=float(metadata["numeric_defaults"]["booster_prior_flights"]), step=1.0, help="Parsed from the public booster designation where available; missing records were imputed during training.")
            values["booster_reused"] = st.checkbox("Booster marked as reused", value=bool(metadata["numeric_defaults"]["booster_reused"] >= 0.5))
            values["launch_month"] = st.number_input("Launch month", min_value=1, max_value=12, value=int(round(metadata["numeric_defaults"]["launch_month"])), step=1)
            values["launch_hour_utc"] = st.number_input("Launch hour (UTC)", min_value=0, max_value=23, value=int(round(metadata["numeric_defaults"]["launch_hour_utc"])), step=1)
            values["air_temp_c_tminus_1h"] = st.number_input("Air temperature 1h before launch (°C)", value=float(metadata["numeric_defaults"]["air_temp_c_tminus_1h"]), step=1.0)
            values["humidity_pct_tminus_1h"] = st.number_input("Humidity 1h before launch (%)", min_value=0.0, max_value=100.0, value=float(metadata["numeric_defaults"]["humidity_pct_tminus_1h"]), step=1.0)
        with right:
            values["wind_speed_m_s_tminus_1h"] = st.number_input("Wind speed 1h before launch (m/s)", min_value=0.0, value=float(metadata["numeric_defaults"]["wind_speed_m_s_tminus_1h"]), step=0.5)
            values["wind_gust_m_s_tminus_1h"] = st.number_input("Wind gust 1h before launch (m/s)", min_value=0.0, value=float(metadata["numeric_defaults"]["wind_gust_m_s_tminus_1h"]), step=0.5)
            values["precipitation_mm_tminus_1h"] = st.number_input("Precipitation (mm)", min_value=0.0, value=float(metadata["numeric_defaults"]["precipitation_mm_tminus_1h"]), step=0.1)
            values["pressure_hpa_tminus_1h"] = st.number_input("Sea-level pressure (hPa)", value=float(metadata["numeric_defaults"]["pressure_hpa_tminus_1h"]), step=0.5)
            values["cloud_cover_pct_tminus_1h"] = st.number_input("Cloud cover (%)", min_value=0.0, max_value=100.0, value=float(metadata["numeric_defaults"]["cloud_cover_pct_tminus_1h"]), step=1.0)
    else:
        left, middle, right = st.columns(3)
        with left:
            payload = st.number_input("Payload mass (kg)", min_value=0.0, max_value=100000.0, value=8500.0, step=250.0)
            air_temp = st.number_input("Air temperature (°C)", min_value=-60.0, max_value=60.0, value=18.0)
            wind = st.number_input("Wind speed (m/s)", min_value=0.0, max_value=100.0, value=7.0)
            precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=200.0, value=0.0)
        with middle:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=55.0)
            engine_temp = st.number_input("Engine temperature (°C)", min_value=0.0, max_value=1500.0, value=720.0)
            pressure = st.number_input("Chamber pressure (MPa)", min_value=0.0, max_value=50.0, value=10.4)
            vibration = st.number_input("Vibration (mm/s)", min_value=0.0, max_value=100.0, value=3.5)
        with right:
            health = st.slider("Engine health score", 0, 100, 90)
            family = st.selectbox("Rocket family", ["Nova-I", "Aster-II", "Vega-M", "Other"])
            site = st.selectbox("Launch site", ["Coastal", "Inland", "Island", "Other"])
    submitted = st.form_submit_button("Estimate risk", type="primary")
if submitted:
    if historical_mode:
        row = pd.DataFrame([{
            "organisation": organisation, "launch_site": launch_site,
            "rocket_family": rocket_family, "launch_year": launch_year,
            "launch_month": launch_month,
        }])[metadata["features"]]
    elif landing_mode:
        row = pd.DataFrame([{**values, "booster_reused": int(values["booster_reused"])}])[metadata["features"]]
    else:
        row = pd.DataFrame([{
            "payload_mass_kg": payload, "air_temp_c": air_temp,
            "wind_speed_m_s": wind, "precipitation_mm": precipitation,
            "humidity_pct": humidity, "engine_temp_c": engine_temp,
            "chamber_pressure_mpa": pressure, "vibration_mm_s": vibration,
            "engine_health_score": health,
            "rocket_family": "Nova-I" if family == "Other" else family,
            "launch_site": "Coastal" if site == "Other" else site,
        }])[FEATURE_COLUMNS]
    risk = float(model.predict_proba(row)[0, 1])
    prediction_name = "estimated non-recovery probability" if landing_mode else "estimated failure probability"
    st.session_state["last_prediction_context"] = (
        f"{prediction_name}: {risk:.1%}; decision threshold: {threshold:.1%}; "
        f"inputs: {json.dumps(row.iloc[0].to_dict(), default=str)}"
    )
    risk_label = "Estimated non-recovery probability" if landing_mode else "Estimated failure probability"
    st.metric(risk_label, f"{risk:.1%}")
    st.progress(min(1.0, max(0.0, risk)))
    flag_label = "Recovery-risk flag" if landing_mode else "Higher risk flag"
    st.write(f"**{flag_label}:** " + ("Yes — review the input conditions" if risk >= threshold else "No — below the selected threshold"))
    st.caption("Model output only; this is not a go/no-go recommendation.")

st.divider()
st.subheader("Ask about the project or prediction")
st.caption(
    "Answers are grounded in the project documentation and source notes. "
    "This assistant does not change the classifier or its prediction."
)

try:
    configured_api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    configured_api_key = ""
api_key = configured_api_key or os.getenv("GEMINI_API_KEY", "")
generation_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []

for message in st.session_state["rag_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    label = source["label"]
                    url = source.get("url")
                    st.markdown(f"- [{label}]({url})" if url else f"- {label}")

with st.form("rag_question_form", clear_on_submit=True):
    question = st.text_input("Ask about the data, features, or this prediction")
    ask_question = st.form_submit_button("Ask", type="primary")

if ask_question and question.strip():
    question = question.strip()
    st.session_state["rag_messages"].append({"role": "user", "content": question})
    chunks = retrieve(question)
    sources = []
    for index, chunk in enumerate(chunks, start=1):
        sources.extend({"label": f"[{index}] {name}", "url": name if name.startswith("https://") else None} for name in chunk.sources)
    prediction_context = st.session_state.get("last_prediction_context")
    if not chunks:
        answer = "I couldn't find relevant information for that question in the project notes. Try asking about the prediction target, launch features, weather data, reuse proxy, or model evaluation."
    elif not api_key:
        answer = "I found relevant project notes, but generated answers are not enabled yet. Set `GEMINI_API_KEY` in your environment or Streamlit secrets to enable the cited Q&A assistant. The retrieved source notes are listed below."
    else:
        try:
            answer = generate_answer(
                question,
                chunks,
                api_key=api_key,
                model=generation_model,
                prediction_context=prediction_context,
            )
        except Exception as exc:
            answer = f"I couldn't generate an answer right now ({exc.__class__.__name__}). The retrieved project sources are listed below."
    if prediction_context and "[P]" in answer:
        sources.append({"label": "[P] Current prediction shown in the app", "url": None})
    st.session_state["rag_messages"].append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()

st.divider()
st.subheader("Score multiple candidate launches")
st.write(f"Upload a CSV containing these model input columns: {', '.join(metadata['features'])}.")
upload = st.file_uploader("Candidate launches CSV", type="csv", key="batch")
if upload:
    try:
        batch = pd.read_csv(upload)
        missing = sorted(set(metadata["features"]) - set(batch.columns))
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            scored = batch.copy()
            score_column = "non_recovery_probability" if landing_mode else "failure_probability"
            scored[score_column] = model.predict_proba(batch[metadata["features"]])[:, 1]
            scored["higher_risk_flag"] = scored[score_column] >= threshold
            scored["risk_band"] = scored["higher_risk_flag"].map({
                True: "Flag for review", False: "Below selected threshold",
            })
            st.dataframe(scored.sort_values(score_column, ascending=False), use_container_width=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False), "launch_risk_scores.csv", "text/csv")
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
