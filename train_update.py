import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────

STD_TEMP = 7.49
STD_WIND = 5.05
STD_RAIN = 0.40

# Villes voisines triées par distance
NEIGHBORS = {
    'Brussels':          264,
    'London':            344,
    'Rotterdam':         373,
    'Köln':              403,
    'Düsseldorf':        411,
    'Amsterdam':         431,
    'Essen':             440,
    'Dortmund':          469,
    'Frankfurt am Main': 479,
}

# Médiane des distances pour normalisation
_dists  = np.array(list(NEIGHBORS.values()), dtype=np.float32)
_sigma  = float(np.median(_dists))
WEIGHTS = {city: 1.0 / (1.0 + d / _sigma)
           for city, d in NEIGHBORS.items()}


# ─────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 1 — Chargement des CSV clean...")
print("=" * 50)

dfs = []
for year in range(2020, 2027):
    try:
        df_year = pd.read_csv(f"weather_{year}_clean.csv")
        df_year['timestamp'] = pd.to_datetime(
            df_year['timestamp'], utc=True)
        dfs.append(df_year)
        print(f"   ✅ {year} — {len(df_year):,} lignes")
    except FileNotFoundError:
        print(f"   ⚠️  {year} non trouvé — ignoré")

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(['city_name', 'timestamp']).reset_index(drop=True)
print(f"\n✅ Total chargé : {len(df):,} lignes\n")


# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 2 — Feature Engineering...")
print("=" * 50)

# ── 2a. Features Paris ──────────────────
print("   ⏳ Features Paris (lags, deltas, rolling, cyclical)...")

paris = df[df['city_name'] == 'Paris'].copy()
paris = paris.sort_values('timestamp').reset_index(drop=True)

# Lags température
for h in [1, 3, 6, 12, 24]:
    paris[f'lag_temp_{h}h'] = paris['temperature'].shift(h)

# Lags pluie
for h in [1, 3, 6, 12]:
    paris[f'lag_rain_{h}h'] = paris['rain'].shift(h)

# Lags vent
for h in [1, 6, 12]:
    paris[f'lag_wind_{h}h'] = paris['wind_speed'].shift(h)

# Lags humidité
for h in [6, 12]:
    paris[f'lag_hum_{h}h'] = paris['humidity'].shift(h)

# Deltas température
for h in [1, 3, 6, 12]:
    paris[f'delta_temp_{h}h'] = paris['temperature'].diff(h)

# Deltas pluie
for h in [1, 3, 6]:
    paris[f'delta_rain_{h}h'] = paris['rain'].diff(h)

# Deltas vent
for h in [1, 6]:
    paris[f'delta_wind_{h}h'] = paris['wind_speed'].diff(h)

# Deltas humidité
paris['delta_hum_6h'] = paris['humidity'].diff(6)

# Rolling stats températures
for w in [3, 6, 12, 24]:
    paris[f'roll_mean_temp_{w}h'] = paris['temperature'].rolling(w).mean()
    paris[f'roll_std_temp_{w}h']  = paris['temperature'].rolling(w).std()
    paris[f'roll_min_temp_{w}h']  = paris['temperature'].rolling(w).min()
    paris[f'roll_max_temp_{w}h']  = paris['temperature'].rolling(w).max()

# Rolling stats pluie
for w in [3, 6, 12, 24]:
    paris[f'roll_mean_rain_{w}h'] = paris['rain'].rolling(w).mean()
    paris[f'roll_sum_rain_{w}h']  = paris['rain'].rolling(w).sum()

# Rolling stats vent
for w in [6, 12]:
    paris[f'roll_mean_wind_{w}h'] = paris['wind_speed'].rolling(w).mean()
    paris[f'roll_std_wind_{w}h']  = paris['wind_speed'].rolling(w).std()

# Rolling humidité
paris['roll_mean_hum_6h']  = paris['humidity'].rolling(6).mean()
paris['roll_mean_cloud_6h'] = paris['clouds'].rolling(6).mean()

# Encodage cyclique heure et mois
paris['sin_hour']  = np.sin(2 * np.pi * paris['hour']  / 24)
paris['cos_hour']  = np.cos(2 * np.pi * paris['hour']  / 24)
paris['sin_month'] = np.sin(2 * np.pi * paris['month'] / 12)
paris['cos_month'] = np.cos(2 * np.pi * paris['month'] / 12)

# Décomposition vent U,V
paris['wind_u'] = (paris['wind_speed']
                   * np.sin(np.deg2rad(paris['wind_direction'])))
paris['wind_v'] = (paris['wind_speed']
                   * np.cos(np.deg2rad(paris['wind_direction'])))

# Dew point depression → proxy probabilité pluie
T      = paris['temperature']
RH     = paris['humidity'].clip(1, 100)
a, b   = 17.27, 237.7
gamma  = (a * T) / (b + T) + np.log(RH / 100)
paris['dew_depression'] = T - (b * gamma) / (a - gamma)

# Rain persistence → combien d'heures consécutives de pluie
paris['rain_binary']      = (paris['rain'] > 0.1).astype(int)
paris['rain_persistence'] = (paris['rain_binary']
                             .groupby(
                                 (paris['rain_binary'] == 0)
                                 .cumsum()
                             ).cumsum())

print("   ✅ Features Paris calculées")

# ── 2b. Features villes voisines ────────
print("   ⏳ Features villes voisines...")

for city, dist in NEIGHBORS.items():
    w    = WEIGHTS[city]
    city_df = df[df['city_name'] == city][
        ['timestamp', 'temperature', 'rain',
         'wind_speed', 'wind_direction', 'humidity']
    ].copy()
    city_df = city_df.sort_values('timestamp')

    # Renommer les colonnes
    prefix = city.replace(' ', '_').replace('ü', 'u').replace('ö', 'o')
    city_df = city_df.rename(columns={
        'temperature': f'{prefix}_temp',
        'rain':        f'{prefix}_rain',
        'wind_speed':  f'{prefix}_wind',
        'humidity':    f'{prefix}_hum',
        'wind_direction': f'{prefix}_dir',
    })

    # Merger avec Paris
    paris = paris.merge(
        city_df[['timestamp',
                 f'{prefix}_temp', f'{prefix}_rain',
                 f'{prefix}_wind', f'{prefix}_hum']],
        on='timestamp', how='left'
    )

    # Pondérer par distance
    paris[f'{prefix}_temp_w']  = paris[f'{prefix}_temp'] * w
    paris[f'{prefix}_rain_w']  = paris[f'{prefix}_rain'] * w
    paris[f'{prefix}_wind_w']  = paris[f'{prefix}_wind'] * w

    # Gradient spatial Paris - voisin
    paris[f'{prefix}_grad_temp'] = (
        paris['temperature'] - paris[f'{prefix}_temp']) * w
    paris[f'{prefix}_grad_rain'] = (
        paris['rain'] - paris[f'{prefix}_rain']) * w

    # Lag 6h du voisin → upwind signal
    paris[f'{prefix}_temp_lag6'] = (
        paris[f'{prefix}_temp'].shift(6) * w)

print("   ✅ Features voisines calculées")

# ── 2c. Target T+6h ─────────────────────
print("   ⏳ Création des targets T+6h...")

paris['target_temp'] = paris['temperature'].shift(-6)
paris['target_wind'] = paris['wind_speed'].shift(-6)
paris['target_rain'] = paris['rain'].shift(-6)

# Supprimer les NaN
paris = paris.dropna().reset_index(drop=True)

print(f"   ✅ Dataset final : {len(paris):,} lignes\n")


# ─────────────────────────────────────────
# 3. DÉFINIR LES FEATURES
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 3 — Définition des features...")
print("=" * 50)

# Features pour Ridge (température)
FEATURES_RIDGE = [
    # Valeurs actuelles
    'temperature', 'humidity', 'clouds', 'rain',
    'wind_u', 'wind_v', 'dew_depression',
    # Cyclical
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    # Lags température
    'lag_temp_1h', 'lag_temp_3h', 'lag_temp_6h',
    'lag_temp_12h', 'lag_temp_24h',
    # Deltas
    'delta_temp_1h', 'delta_temp_3h',
    'delta_temp_6h', 'delta_temp_12h',
    # Rolling stats temp
    'roll_mean_temp_3h', 'roll_std_temp_3h',
    'roll_mean_temp_6h', 'roll_std_temp_6h',
    'roll_min_temp_6h',  'roll_max_temp_6h',
    'roll_mean_temp_12h','roll_std_temp_12h',
    'roll_mean_temp_24h',
    # Rolling humidité
    'roll_mean_hum_6h', 'roll_mean_cloud_6h',
    # Voisins
    'Brussels_temp_w', 'Brussels_grad_temp', 'Brussels_temp_lag6',
    'London_temp_w',   'London_grad_temp',   'London_temp_lag6',
    'Rotterdam_temp_w','Rotterdam_grad_temp','Rotterdam_temp_lag6',
    'K_ln_temp_w',     'K_ln_grad_temp',
    'D_sseldorf_temp_w','D_sseldorf_grad_temp',
    'Amsterdam_temp_w', 'Amsterdam_grad_temp',
]

# Features pour GradientBoosting (vent + pluie)
FEATURES_GB = [
    # Valeurs actuelles
    'temperature', 'rain', 'wind_speed', 'wind_direction',
    'humidity', 'clouds', 'snow',
    'wind_u', 'wind_v', 'dew_depression',
    # Cyclical
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    # Lags temp
    'lag_temp_1h', 'lag_temp_3h', 'lag_temp_6h', 'lag_temp_12h',
    # Lags pluie
    'lag_rain_1h', 'lag_rain_3h', 'lag_rain_6h', 'lag_rain_12h',
    # Lags vent
    'lag_wind_1h', 'lag_wind_6h', 'lag_wind_12h',
    # Lags humidité
    'lag_hum_6h', 'lag_hum_12h',
    # Deltas
    'delta_temp_1h', 'delta_temp_3h', 'delta_temp_6h',
    'delta_rain_1h', 'delta_rain_3h', 'delta_rain_6h',
    'delta_wind_1h', 'delta_wind_6h',
    'delta_hum_6h',
    # Rolling stats temp
    'roll_mean_temp_3h', 'roll_std_temp_3h',
    'roll_mean_temp_6h', 'roll_std_temp_6h',
    'roll_min_temp_6h',  'roll_max_temp_6h',
    'roll_mean_temp_12h','roll_mean_temp_24h',
    # Rolling pluie
    'roll_mean_rain_3h', 'roll_sum_rain_3h',
    'roll_mean_rain_6h', 'roll_sum_rain_6h',
    'roll_mean_rain_12h','roll_sum_rain_12h',
    'roll_mean_rain_24h','roll_sum_rain_24h',
    # Rolling vent
    'roll_mean_wind_6h', 'roll_std_wind_6h',
    'roll_mean_wind_12h',
    # Humidité rolling
    'roll_mean_hum_6h', 'roll_mean_cloud_6h',
    # Rain persistence
    'rain_persistence',
    # Voisins temp
    'Brussels_temp_w', 'Brussels_grad_temp', 'Brussels_temp_lag6',
    'London_temp_w',   'London_grad_temp',   'London_temp_lag6',
    'Rotterdam_temp_w','Rotterdam_grad_temp',
    # Voisins pluie
    'Brussels_rain_w', 'Brussels_grad_rain',
    'London_rain_w',   'London_grad_rain',
    'Rotterdam_rain_w',
    # Voisins vent
    'Brussels_wind_w', 'London_wind_w', 'Rotterdam_wind_w',
    'Amsterdam_wind_w','D_sseldorf_wind_w',
]

# Garder seulement les colonnes disponibles
FEATURES_RIDGE = [f for f in FEATURES_RIDGE if f in paris.columns]
FEATURES_GB    = [f for f in FEATURES_GB    if f in paris.columns]

print(f"   ✅ Features Ridge : {len(FEATURES_RIDGE)}")
print(f"   ✅ Features GB    : {len(FEATURES_GB)}\n")


# ─────────────────────────────────────────
# 4. SPLIT TRAIN / TEST
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 4 — Split Train / Test...")
print("=" * 50)

train = paris[paris['timestamp'].dt.year <= 2024]
test  = paris[paris['timestamp'].dt.year >= 2025]

X_train_ridge = train[FEATURES_RIDGE].values.astype(np.float32)
X_test_ridge  = test[FEATURES_RIDGE].values.astype(np.float32)
X_train_gb    = train[FEATURES_GB].values.astype(np.float32)
X_test_gb     = test[FEATURES_GB].values.astype(np.float32)

y_train_temp  = train['target_temp'].values
y_train_wind  = train['target_wind'].values
y_train_rain  = train['target_rain'].values
y_test_temp   = test['target_temp'].values
y_test_wind   = test['target_wind'].values
y_test_rain   = test['target_rain'].values

print(f"   ✅ Train : {len(train):,} samples (2020-2024)")
print(f"   ✅ Test  : {len(test):,} samples (2025-2026)\n")


# ─────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 5 — Scaling (Ridge seulement)...")
print("=" * 50)

scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train_ridge)
X_test_scaled   = scaler.transform(X_test_ridge)

joblib.dump(scaler, 'scaler.pkl')
print("   ✅ scaler.pkl sauvegardé\n")


# ─────────────────────────────────────────
# 6. MODÈLE TEMPÉRATURE → Ridge
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 6 — Ridge (température)...")
print("=" * 50)

model_temp = Ridge(alpha=1.0)
model_temp.fit(X_train_scaled, y_train_temp)

pred_temp = model_temp.predict(X_test_scaled)
mae_temp  = np.mean(np.abs(pred_temp - y_test_temp))
print(f"   ✅ MAE température : {mae_temp:.3f}°C")

joblib.dump(model_temp, 'model_temp.pkl')
print("   ✅ model_temp.pkl sauvegardé\n")


# ─────────────────────────────────────────
# 7. MODÈLE VENT → GradientBoosting
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 7 — GradientBoosting (vent)...")
print("=" * 50)

model_wind = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=8,
    max_features=0.8,
    random_state=42,
)
model_wind.fit(X_train_gb, y_train_wind)

pred_wind = model_wind.predict(X_test_gb)
mae_wind  = np.mean(np.abs(pred_wind - y_test_wind))
print(f"   ✅ MAE vent : {mae_wind:.3f} m/s")

joblib.dump(model_wind, 'model_wind.pkl')
print("   ✅ model_wind.pkl sauvegardé\n")


# ─────────────────────────────────────────
# 8. MODÈLE PLUIE → Classifieur + Régresseur
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 8 — GradientBoosting (pluie)...")
print("=" * 50)

# Seuil 0.1mm → il pleut
y_train_rain_clf = (y_train_rain > 0.1).astype(int)
pct_rain = 100 * y_train_rain_clf.mean()
print(f"   ℹ️  {pct_rain:.1f}% des heures avec pluie")

# Classifieur
model_rain_clf = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=8,
    max_features=0.8,
    random_state=42,
)
model_rain_clf.fit(X_train_gb, y_train_rain_clf)

joblib.dump(model_rain_clf, 'model_rain_clf.pkl')
print("   ✅ model_rain_clf.pkl sauvegardé")

# Régresseur → entraîné seulement sur heures pluvieuses
rain_mask      = y_train_rain > 0.1
print(f"   ℹ️  {rain_mask.sum():,} heures pluvieuses pour régresseur")

model_rain_reg = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=5,
    max_features=0.8,
    random_state=42,
)
model_rain_reg.fit(X_train_gb[rain_mask], y_train_rain[rain_mask])

joblib.dump(model_rain_reg, 'model_rain_reg.pkl')
print("   ✅ model_rain_reg.pkl sauvegardé\n")


# ─────────────────────────────────────────
# 9. ÉVALUATION FINALE
# ─────────────────────────────────────────

print("=" * 50)
print("⏳ ÉTAPE 9 — Évaluation sur test (2025-2026)...")
print("=" * 50)

# Prédictions
rain_proba = model_rain_clf.predict_proba(X_test_gb)[:, 1]
pred_rain  = np.where(
    rain_proba > 0.5,
    model_rain_reg.predict(X_test_gb),
    0.0
)
pred_rain = np.clip(pred_rain, 0, None)
pred_wind = np.clip(pred_wind, 0, None)

# MAE
mae_temp = np.mean(np.abs(pred_temp - y_test_temp))
mae_wind = np.mean(np.abs(pred_wind - y_test_wind))
mae_rain = np.mean(np.abs(pred_rain - y_test_rain))

# Score final
score_temp  = mae_temp / STD_TEMP
score_wind  = mae_wind / STD_WIND
score_rain  = mae_rain / STD_RAIN
score_final = -np.mean([score_temp, score_wind, score_rain])

print(f"\n{'=' * 50}")
print(f"  RÉSULTATS FINAUX — TEST 2025-2026")
print(f"{'=' * 50}")
print(f"  MAE température : {mae_temp:.3f}°C  → score : {-score_temp:.4f}")
print(f"  MAE vent        : {mae_wind:.3f} m/s → score : {-score_wind:.4f}")
print(f"  MAE pluie       : {mae_rain:.3f} mm  → score : {-score_rain:.4f}")
print(f"{'=' * 50}")
print(f"  Score final  : {score_final:.4f}")
print(f"  Baseline     : -0.3000")
print(f"  Amélioration : {score_final - (-0.30):+.4f}")
print(f"{'=' * 50}")

# Détail par année
print(f"\n  DÉTAIL PAR ANNÉE :")
print(f"  {'─' * 45}")
years_test = test['timestamp'].dt.year.values
for year in [2025, 2026]:
    mask  = years_test == year
    if mask.sum() == 0:
        continue
    mae_t = np.mean(np.abs(pred_temp[mask] - y_test_temp[mask]))
    mae_w = np.mean(np.abs(pred_wind[mask] - y_test_wind[mask]))
    mae_r = np.mean(np.abs(pred_rain[mask] - y_test_rain[mask]))
    s     = -np.mean([mae_t/STD_TEMP, mae_w/STD_WIND, mae_r/STD_RAIN])
    print(f"  {year} → temp:{mae_t:.2f}°C | "
          f"vent:{mae_w:.2f}m/s | "
          f"pluie:{mae_r:.2f}mm | "
          f"score:{s:.4f}")

print(f"\n🏆 Tous les .pkl sauvegardés et prêts pour agent.py !")
print(f"   scaler.pkl, model_temp.pkl, model_wind.pkl,")
print(f"   model_rain_clf.pkl, model_rain_reg.pkl")
