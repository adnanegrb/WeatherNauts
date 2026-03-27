import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

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

STD_TEMP = 7.49
STD_WIND = 5.05
STD_RAIN = 0.40

# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────
print("⏳ Chargement des données...")

dfs = []
for year in range(2020, 2027):
    try:
        df_year = pd.read_csv(f"weather_{year}_features.csv")
        dfs.append(df_year)
        print(f"   ✅ {year} — {len(df_year)} lignes")
    except FileNotFoundError:
        print(f"   ⚠️ {year} non trouvé")

df = pd.concat(dfs, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
print(f"✅ Total chargé — {len(df)} lignes\n")

# ─────────────────────────────────────────
# 2. PRÉPARATION PARIS
# ─────────────────────────────────────────
print("⏳ Préparation des données Paris...")

paris = df[df['city_name'] == 'Paris'].copy()
paris = paris.sort_values('timestamp').reset_index(drop=True)
print(f"   Paris brut — {len(paris)} lignes")

paris['target_temp'] = paris['temperature'].shift(-6)
paris['target_wind'] = paris['wind_speed'].shift(-6)
paris['target_rain'] = paris['rain'].shift(-6)

paris = paris.dropna(subset=['target_temp', 'target_wind', 'target_rain'])
paris = paris.dropna(subset=feature_cols)
print(f"   Paris après dropna — {len(paris)} lignes")
print(f"   Features disponibles — {len(feature_cols)} colonnes\n")

# ─────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────
print("⏳ Split Train / Test...")

train = paris[paris['timestamp'].dt.year <= 2024]
test  = paris[paris['timestamp'].dt.year >= 2025]

X_train = train[feature_cols]
X_test  = test[feature_cols]

y_train_temp = train['target_temp']
y_train_wind = train['target_wind']
y_train_rain = train['target_rain']

y_test_temp  = test['target_temp']
y_test_wind  = test['target_wind']
y_test_rain  = test['target_rain']

print(f"   ✅ Train — {len(train)} lignes (2020-2024)")
print(f"   ✅ Test  — {len(test)} lignes (2025-2026)\n")

# ─────────────────────────────────────────
# 4. SCALING
# ─────────────────────────────────────────
print("⏳ Scaling StandardScaler...")

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')
print("   ✅ scaler.pkl sauvegardé\n")

# ─────────────────────────────────────────
# 5. MODÈLE TEMPÉRATURE
# ─────────────────────────────────────────
print("⏳ Entraînement Ridge (température)...")
print("   GridSearch en cours — patience ⏳")

ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_gs     = GridSearchCV(
    Ridge(),
    ridge_params,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
ridge_gs.fit(X_train_scaled, y_train_temp)
model_temp = ridge_gs.best_estimator_

print(f"   ✅ Meilleur alpha : {ridge_gs.best_params_}")
joblib.dump(model_temp, 'model_temp.pkl')
print("   ✅ model_temp.pkl sauvegardé\n")

# ─────────────────────────────────────────
# 6. MODÈLE VENT
# ─────────────────────────────────────────
print("⏳ Entraînement XGBoost (vent)...")
print("   GridSearch en cours — patience ⏳")

xgb_params_wind = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample'    : [0.8, 1.0]
}
xgb_gs_wind = GridSearchCV(
    XGBRegressor(random_state=42),
    xgb_params_wind,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
xgb_gs_wind.fit(X_train, y_train_wind)
model_wind = xgb_gs_wind.best_estimator_

print(f"   ✅ Meilleurs params : {xgb_gs_wind.best_params_}")
joblib.dump(model_wind, 'model_wind.pkl')
print("   ✅ model_wind.pkl sauvegardé\n")

# ─────────────────────────────────────────
# 7. MODÈLE PLUIE
# ─────────────────────────────────────────
print("⏳ Entraînement XGBoost Classifieur (pluie)...")
print("   GridSearch en cours — patience ⏳")

y_train_rain_clf = (y_train_rain > 0.1).astype(int)
print(f"   Heures avec pluie : {y_train_rain_clf.sum()} / {len(y_train_rain_clf)}")

xgb_params_clf = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}
xgb_gs_clf = GridSearchCV(
    XGBClassifier(random_state=42),
    xgb_params_clf,
    cv=5,
    scoring='neg_log_loss',
    n_jobs=-1,
    verbose=1
)
xgb_gs_clf.fit(X_train, y_train_rain_clf)
model_rain_clf = xgb_gs_clf.best_estimator_

print(f"   ✅ Meilleurs params clf : {xgb_gs_clf.best_params_}")
joblib.dump(model_rain_clf, 'model_rain_clf.pkl')
print("   ✅ model_rain_clf.pkl sauvegardé\n")

print("⏳ Entraînement XGBoost Régresseur (pluie)...")

rain_mask = y_train_rain > 0.1
print(f"   Lignes avec pluie pour régresseur : {rain_mask.sum()}")

xgb_params_reg = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}
xgb_gs_reg = GridSearchCV(
    XGBRegressor(random_state=42),
    xgb_params_reg,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
xgb_gs_reg.fit(X_train[rain_mask], y_train_rain[rain_mask])
model_rain_reg = xgb_gs_reg.best_estimator_

print(f"   ✅ Meilleurs params reg : {xgb_gs_reg.best_params_}")
joblib.dump(model_rain_reg, 'model_rain_reg.pkl')
print("   ✅ model_rain_reg.pkl sauvegardé\n")

# ─────────────────────────────────────────
# 8. ÉVALUATION
# ─────────────────────────────────────────
print("⏳ Évaluation sur test...")

pred_temp  = model_temp.predict(X_test_scaled)
mae_temp   = np.mean(np.abs(pred_temp - y_test_temp))
print(f"   ✅ Température prédite")

pred_wind  = model_wind.predict(X_test)
mae_wind   = np.mean(np.abs(pred_wind - y_test_wind))
print(f"   ✅ Vent prédit")

rain_proba = model_rain_clf.predict_proba(X_test)[:, 1]
pred_rain  = np.where(
    rain_proba > 0.5,
    model_rain_reg.predict(X_test),
    0.0
)
mae_rain = np.mean(np.abs(pred_rain - y_test_rain))
print(f"   ✅ Pluie prédite\n")

# ─────────────────────────────────────────
# 9. RÉSULTATS FINAUX
# ─────────────────────────────────────────

score_final = -np.mean([
    mae_temp / STD_TEMP,
    mae_wind / STD_WIND,
    mae_rain / STD_RAIN
])

print(f"{'='*40}")
print(f"RÉSULTATS FINAUX")
print(f"{'='*40}")
print(f"MAE température  : {mae_temp:.2f}°C")
print(f"MAE vent         : {mae_wind:.2f} m/s")
print(f"MAE pluie        : {mae_rain:.2f} mm")
print(f"\nScore final  : {score_final:.4f}")
print(f"Baseline     : -0.3000")
print(f"{'='*40}")

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

print(f"\n🏆 Tous les modèles sauvegardés et prêts pour agent.py !")
