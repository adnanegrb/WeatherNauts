# WeatherNauts 🌤️

WeatherNauts is a competitive ML agent built for the ML Arena competition. It predicts temperature, wind speed and rainfall for Paris 6 hours ahead, using 24 hours of historical data from 20 European cities. The pipeline covers three stages : physical data cleaning with outlier clipping and cascaded NaN imputation, feature engineering producing 98 structured variables across temporal, geographic and physical families, and three independent HistGradientBoostingRegressor models optimized via Optuna TPE Bayesian search. Rather than predicting absolute values, the agent predicts the variation Δy and reconstructs the output as ŷ = y_t + Δy, reducing target variance and improving convergence. Final score achieved : L = -0.28, competitive at leaderboard level. Full technical report available [here](https://drive.google.com/file/d/12kuwHxD5po4elmCxtaDkZ85lAboj_9jD/view?usp=sharing).

**Team :** Adnane GARAB · Massy Merakeb · Hakim Rohimun
