# WeatherNauts 🌤️

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Model](https://img.shields.io/badge/Model-HistGradientBoosting-8A2BE2?style=flat-square)
![Optimizer](https://img.shields.io/badge/Optimizer-Optuna%20TPE-orange?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-scikit--learn-red?style=flat-square)

WeatherNauts is a competitive ML agent built for the ML Arena competition. It predicts temperature, wind speed and rainfall for Paris 6 hours ahead, using 24 hours of historical data from 20 European cities.


## What it does

Given 24 hours of meteorological observations across 20 European cities, the agent predicts three targets for Paris 6 hours into the future: temperature, wind speed, and rainfall.

Rather than predicting absolute values directly, the model predicts the variation Δy and reconstructs the output as ŷ = y_t + Δy. This reduces target variance significantly and improves convergence across all three regressors.

Final leaderboard score: **L = −0.11**


## Pipeline

**Data cleaning.** Physical outlier clipping followed by cascaded NaN imputation — missing values are filled using correlated city readings before falling back to temporal interpolation.

**Feature engineering.** 98 structured variables across three families: temporal features (hour, day, lag statistics), geographic features (city distances, coastal proximity), and physical features (pressure gradients, wind vectors, humidity deltas).

**Modeling.** Three independent `HistGradientBoostingRegressor` models, one per target. Hyperparameters tuned via Optuna TPE Bayesian search with time-series aware cross-validation.


## Stack

```
scikit-learn    HistGradientBoostingRegressor
optuna          TPE Bayesian hyperparameter search
xgboost         baseline and ensemble reference
pandas / numpy  feature engineering pipeline
```


## Results

| Target | Metric | Score |
|---|---|---|
| Temperature | MAE | competitive |
| Wind speed | MAE | competitive |
| Rainfall | MAE | competitive |
| Overall | L score | **−0.11** |

Full technical report available [here](https://drive.google.com/file/d/12kuwHxD5po4elmCxtaDkZ85lAboj_9jD/view?usp=sharing).


## Team

Adnane GARAB · Massy MERAKEB · Hakim ROHIMUN
