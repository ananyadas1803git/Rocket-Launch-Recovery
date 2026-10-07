# Rocket Launch Recovery App

An end-to-end predictive engineering project comparing Logistic Regression, Random Forest, and XGBoost for the probability that a Falcon 9 first-stage booster is **not recovered successfully**. It includes launch and payload data, archived weather, booster reuse proxies, model comparison, probability calibration, SHAP explainability, a Streamlit prediction page, and a documentation-grounded RAG assistant.

## Dataset and target

The default dataset contains **121 Falcon 9 launches from 2010–2021**. Launch, payload, booster and landing fields come from the public [`spacex_web_scraped.csv` table](https://github.com/Roderic19/IBM-Applied-Data-Science-Capstone/blob/main/spacex_web_scraped.csv). The model target is first-stage recovery: `Success` is a recovered booster; failed, uncontrolled, controlled-but-not-recovered, no-attempt, and precluded outcomes are non-recoveries. This target is distinct from primary mission success.

Weather inputs are historical ERA5 reanalysis from [Open-Meteo's Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), joined by launch region and sampled at the previous full UTC hour. These are reconstructed regional weather estimates, not sensor readings or archived forecasts. Payload mass, booster generation, reuse markers, prior-flight count where listed, orbit, launch site, flight number, UTC month/hour, and weather are the model inputs. Reuse history is a rough hardware-condition proxy; **actual engine temperature, chamber pressure, vibration, and engine-health telemetry are not public in this source**.

The prepared dataset is [`data/spacex_falcon9_training.csv`](data/spacex_falcon9_training.csv); raw launch and matched weather inputs are in `data/spacex_falcon9_launches.csv` and `data/spacex_launch_weather_era5.csv`. `src/prepare_spacex.py` rebuilds the model-ready table from those two files.

## Current held-out evaluation

The current run selects Random Forest by 5-fold stratified cross-validation on the training rows, then reports a separate stratified 25% holdout using a fixed 0.50 threshold. On this small 31-launch test set it achieved **80.65% accuracy**, **0.768 ROC AUC**, **74.77% balanced accuracy**, **54.55% non-recovery recall**, **85.71% non-recovery precision**, and **0.155 Brier score**. The always-recovered baseline scored 64.52% accuracy. Confusion matrix (actual rows; predicted columns): `[[19, 1], [5, 6]]` for recovered / not recovered.

This is a more relevant project dataset than the worldwide catalog, but 121 records and 11 test failures are still too few for a strong generalization claim. The random holdout is useful for a class project and is not a future-launch validation. Do not use these predictions for safety or launch decisions. The worldwide historical catalog remains available as `data/mission_launches_raw.csv` for the separate mission-outcome workflow, but it is no longer the default training dataset.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.prepare_spacex
python -m src.train
streamlit run app.py
```

Training writes the calibrated model, candidate comparison, held-out metrics, EDA summary, SHAP outputs/status, and metadata to `artifacts/`. To use a different integrated dataset instead:

```bash
python -m src.train --data path/to/integrated.csv
```

For actual engine-condition analysis, integrate launch IDs with quality-controlled engine measurements captured before launch. Do not substitute booster reuse for measured engine health, and do not fabricate missing telemetry.

## Deploy with Docker

```bash
docker build -t launch-risk-lab .
docker run --rm -p 8501:8501 launch-risk-lab
```

Open `http://localhost:8501`. The container trains from the checked-in prepared SpaceX dataset. For Streamlit Community Cloud, set `app.py` as the main file and install dependencies from `requirements.txt`.

The prediction page also includes a RAG assistant that retrieves relevant project notes from `docs/rag_knowledge_base.md` using TF-IDF, then generates a concise answer with citations. It uses Google's Gemini Interactions API only for answer generation; retrieved context and the current prediction do not alter classifier inputs or scores. Set `GEMINI_API_KEY` as an environment variable locally, or add it to the app's Streamlit secrets when deployed. `GEMINI_MODEL` optionally selects a different Gemini model; it defaults to `gemini-3.8-flash`. Without an API key, the app still retrieves the relevant notes and explains how to enable generated answers.

```toml
# Streamlit app secrets (do not commit this file)
GEMINI_API_KEY = "your-gemini-api-key"
```

## Project layout

```text
app.py                             Streamlit prediction page and RAG assistant
src/rag.py                        Local TF-IDF retrieval and grounded answer generation
docs/rag_knowledge_base.md        Curated, cited project facts for the RAG assistant
src/prepare_spacex.py              Launch/weather join and feature preparation
src/spacex_landing.py              Model selection, calibration, evaluation, SHAP
src/train.py                       Dataset routing and integrated-data workflow
src/data.py                        Synthetic fallback and source integration helpers
data/spacex_falcon9_launches.csv    Public launch source table
data/spacex_launch_weather_era5.csv Matched hourly ERA5 weather snapshots
data/spacex_falcon9_training.csv   Model-ready real-world dataset
artifacts/                         Trained model and evaluation outputs
Dockerfile                         Container deployment
```
