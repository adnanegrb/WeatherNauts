import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Features utilisées pour l'entraînement
feature_cols = [
    'temperature', 'humidity', 'wind_u', 'wind_v',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'paris_temperature_lag1', 'paris_temperature_lag3',
    'paris_temperature_lag6', 'paris_temperature_lag12',
    'paris_rain_lag1', 'paris_rain_lag3',
    'paris_rain_lag6', 'paris_rain_lag12',
    'paris_wind_speed_lag1', 'paris_wind_speed_lag6',
    'delta_temperature_1h', 'delta_temperature_6h',
    'delta_rain_1h', 'delta_rain_6h',
    'delta_humidity_1h', 'delta_humidity_6h',
    'upwind_temp', 'upwind_rain', 'upwind_wind',
    'dew_depression'
]

# Std donnés par le concours
STD_TEMP = 7.49
STD_WIND = 5.05
STD_RAIN = 0.40

# ─────────────────────────────────────────
# 1. CHARGER TOUTES LES ANNÉES 2020-2026
# ─────────────────────────────────────────

dfs = []
for year in range(2020, 2027):
    try:
        df_year = pd.read_csv(f"weather_{year}_features.csv")
        dfs.append(df_year)
        print(f"✅ {year} chargé — {len(df_year)} lignes")
    except FileNotFoundError:
        print(f"⚠️ weather_{year}_features.csv non trouvé")

df = pd.concat(dfs, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# Garder seulement Paris
paris = df[df['city_name'] == 'Paris'].copy()
paris = paris.sort_values('timestamp').reset_index(drop=True)

# ─────────────────────────────────────────
# 2. CRÉER LES TARGETS T+6H
# ─────────────────────────────────────────

# shift(-6) = valeur 6 heures dans le futur
paris['target_temp'] = paris['temperature'].shift(-6)
paris['target_wind'] = paris['wind_speed'].shift(-6)
paris['target_rain'] = paris['rain'].shift(-6)

# Supprimer les lignes sans target ou sans features
paris = paris.dropna(subset=['target_temp', 'target_wind', 'target_rain'])
paris = paris.dropna(subset=feature_cols)

# ─────────────────────────────────────────
# 3. SPLIT TRAIN / TEST
# ─────────────────────────────────────────

# Train : 2020-2024
train = paris[paris['timestamp'].dt.year <= 2024]

# Test : 2025 + 2026 (3 mois disponibles)
test = paris[paris['timestamp'].dt.year >= 2025]

X_train = train[feature_cols]
X_test  = test[feature_cols]

y_train_temp = train['target_temp']
y_train_wind = train['target_wind']
y_train_rain = train['target_rain']

y_test_temp  = test['target_temp']
y_test_wind  = test['target_wind']
y_test_rain  = test['target_rain']

print(f"\nTrain : {len(train)} lignes (2020-2024)")
print(f"Test  : {len(test)} lignes (2025-2026)")

# ─────────────────────────────────────────
# 4. SCALING → Ridge seulement
# ─────────────────────────────────────────

# Fit sur train seulement → jamais sur test
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')
print("\nscaler.pkl sauvegardé ✅")

# ─────────────────────────────────────────
# 5. MODÈLE TEMPÉRATURE → Ridge
# ─────────────────────────────────────────

# GridSearch pour trouver le meilleur alpha
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_gs     = GridSearchCV(
    Ridge(),
    ridge_params,
    cv=5,
    scoring='neg_mean_absolute_error'
)
ridge_gs.fit(X_train_scaled, y_train_temp)
model_temp = ridge_gs.best_estimator_

print(f"Ridge best alpha : {ridge_gs.best_params_}")

joblib.dump(model_temp, 'model_temp.pkl')
print("model_temp.pkl sauvegardé ✅")

# ─────────────────────────────────────────
# 6. MODÈLE VENT → XGBoost Regressor
# ─────────────────────────────────────────

# XGBoost → pas besoin de scaling
model_wind = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model_wind.fit(X_train, y_train_wind)

joblib.dump(model_wind, 'model_wind.pkl')
print("model_wind.pkl sauvegardé ✅")

# ─────────────────────────────────────────
# 7. MODÈLE PLUIE → Classifieur + Régresseur
# ─────────────────────────────────────────

# Seuil 0.1mm → il pleut (1) ou pas (0)
y_train_rain_clf = (y_train_rain > 0.1).astype(int)

# Classifieur → va-t-il pleuvoir ?
model_rain_clf = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
model_rain_clf.fit(X_train, y_train_rain_clf)

joblib.dump(model_rain_clf, 'model_rain_clf.pkl')
print("model_rain_clf.pkl sauvegardé ✅")

# Régresseur → combien de mm ?
# Entraîné seulement sur les heures où il pleut
rain_mask      = y_train_rain > 0.1
model_rain_reg = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
model_rain_reg.fit(X_train[rain_mask], y_train_rain[rain_mask])

joblib.dump(model_rain_reg, 'model_rain_reg.pkl')
print("model_rain_reg.pkl sauvegardé ✅")

# ─────────────────────────────────────────
# 8. ÉVALUATION SUR TEST
# ─────────────────────────────────────────

# Prédictions température
pred_temp = model_temp.predict(X_test_scaled)
mae_temp  = np.mean(np.abs(pred_temp - y_test_temp))

# Prédictions vent
pred_wind = model_wind.predict(X_test)
mae_wind  = np.mean(np.abs(pred_wind - y_test_wind))

# Prédictions pluie
rain_proba = model_rain_clf.predict_proba(X_test)[:, 1]
pred_rain  = np.where(
    rain_proba > 0.5,
    model_rain_reg.predict(X_test),
    0.0
)
mae_rain = np.mean(np.abs(pred_rain - y_test_rain))

# ─────────────────────────────────────────
# 9. RÉSULTATS DÉTAILLÉS
# ─────────────────────────────────────────

print(f"\n{'='*40}")
print(f"RÉSULTATS SUR TEST (2025 + 2026)")
print(f"{'='*40}")
print(f"MAE température  : {mae_temp:.2f}°C  (std={STD_TEMP})")
print(f"MAE vent         : {mae_wind:.2f} m/s (std={STD_WIND})")
print(f"MAE pluie        : {mae_rain:.2f} mm  (std={STD_RAIN})")

# Score normalisé comme le concours
score_final = -np.mean([
    mae_temp / STD_TEMP,
    mae_wind / STD_WIND,
    mae_rain / STD_RAIN
])

print(f"\nScore final  : {score_final:.4f}")
print(f"Baseline     : -0.3000")
print(f"{'='*40}")

# Détail par année
print(f"\nDÉTAIL PAR ANNÉE :")
print(f"{'='*40}")
for year in [2025, 2026]:
    mask = test['timestamp'].dt.year == year
    if mask.sum() == 0:
        continue
    mae_t = np.mean(np.abs(pred_temp[mask.values] - y_test_temp[mask]))
    mae_w = np.mean(np.abs(pred_wind[mask.values] - y_test_wind[mask]))
    mae_r = np.mean(np.abs(pred_rain[mask.values] - y_test_rain[mask]))
    s     = -np.mean([mae_t/STD_TEMP, mae_w/STD_WIND, mae_r/STD_RAIN])
    print(f"{year} → temp:{mae_t:.2f}°C | vent:{mae_w:.2f}m/s | pluie:{mae_r:.2f}mm | score:{s:.4f}")
