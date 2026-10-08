# Rocket Recovery Predictor knowledge base

## Prediction target
The classifier estimates the probability that a Falcon 9 first-stage booster was not successfully recovered. The positive class is non-recovery; a recorded landing outcome of `Success` is a successful recovery. This is not the same as whether the payload reached orbit or whether the overall mission succeeded. The data contains 121 launches from 2010 through 2021, so results are uncertain and should not be treated as launch guidance.
Sources: README.md; src/prepare_spacex.py; https://github.com/Roderic19/IBM-Applied-Data-Science-Capstone/blob/main/spacex_web_scraped.csv

## Weather inputs
Weather variables are historical ERA5 reanalysis from Open-Meteo, joined to each launch and sampled at the previous full UTC hour. They are reconstructed regional observations, not archived launch forecasts or measurements from sensors on the rocket. Weather features include temperature, humidity, wind, gust, precipitation, pressure, and cloud cover.
Sources: README.md; src/prepare_spacex.py; https://open-meteo.com/en/docs/historical-weather-api

## Booster reuse and missing telemetry
Booster generation, the reuse marker, and the number of known prior booster flights are public-data proxies for hardware condition. They are not measurements of engine health. This dataset does not include engine temperature, chamber pressure, vibration, or engine-health telemetry. Do not interpret a reuse feature as a direct measurement of engine condition.
Sources: README.md; src/prepare_spacex.py; src/spacex_landing.py

## Model selection and evaluation
The project compares Logistic Regression, Random Forest, and XGBoost. The selected model is chosen by stratified five-fold cross-validation on the training portion, then sigmoid probability calibration is applied. Evaluation uses a separate stratified random 25% holdout. The dataset and holdout are small; this is not a future-launch validation or a strong generalization claim.
Sources: README.md; src/spacex_landing.py

## Probability and decision threshold
The displayed probability is the model's estimated probability of non-recovery, not a certainty. The sidebar threshold controls when the app marks an estimate as higher risk for review. Lowering the threshold flags more cases; it does not change the model probability or retrain the model. The app is an educational project and not a go/no-go or safety tool.
Sources: README.md; app.py; src/spacex_landing.py

## Feature definitions
Inputs include payload mass, launch site, orbit, booster category, known prior booster flights, reuse marker, flight number, UTC month and hour, and weather values from one hour before launch. The project derives booster category and reuse proxies from the public booster designation, and joins weather using the launch flight number.
Sources: README.md; src/prepare_spacex.py; src/spacex_landing.py
