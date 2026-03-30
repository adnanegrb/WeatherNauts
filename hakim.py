import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

STD_TEMP = 7.49
STD_WIND = 5.05
STD_RAIN = 0.40
DATA_DIR = r"C:\Users\adnan\Downloads"

NEIGHBORS = {
    'Brussels': 264, 'London': 344, 'Rotterdam': 373,
    'Köln': 403, 'Düsseldorf': 411, 'Amsterdam': 431,
    'Essen': 440, 'Dortmund': 469, 'Frankfurt am Main': 479,
}
_sigma  = float(np.median(list(NEIGHBORS.values())))
WEIGHTS = {c: 1.0 / (1.0 + d / _sigma) for c, d in NEIGHBORS.items()}

# ── Chargement CSV clean ─────────────────────────────
dfs = []
for year in range(2020, 2027):
    df = pd.read_csv(f"{DATA_DIR}/weather_{year}_clean.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True).sort_values(
    ['city_name', 'timestamp']).reset_index(drop=True)

df_train = df_all[df_all['timestamp'].dt.year <= 2024].copy()
df_test  = df_all[df_all['timestamp'].dt.year >= 2025].copy()

# ── Feature Engineering ──────────────────────────────
# Calcule toutes les features depuis données brutes Paris
# + villes voisines pondérées par distance
# Même fonction utilisée pour train et test

def build_features(df, df_ref):
    paris = df[df['city_name'] == 'Paris'].copy().sort_values('timestamp').reset_index(drop=True)
    X     = pd.DataFrame(index=paris.index)

    # Valeurs actuelles
    for col in ['temperature', 'wind_speed', 'rain', 'humidity', 'clouds', 'snow']:
        X[col] = paris[col]

    # Cyclical encoding
    X['sin_hour']  = np.sin(2 * np.pi * paris['hour']  / 24)
    X['cos_hour']  = np.cos(2 * np.pi * paris['hour']  / 24)
    X['sin_month'] = np.sin(2 * np.pi * paris['month'] / 12)
    X['cos_month'] = np.cos(2 * np.pi * paris['month'] / 12)

    # Wind U,V decomposition
    X['wind_u'] = paris['wind_speed'] * np.sin(np.deg2rad(paris['wind_direction']))
    X['wind_v'] = paris['wind_speed'] * np.cos(np.deg2rad(paris['wind_direction']))

    # Dew point depression
    T, RH = paris['temperature'], paris['humidity'].clip(1, 100)
    gamma = (17.27 * T) / (237.7 + T) + np.log(RH / 100)
    X['dew_depression'] = T - (237.7 * gamma) / (17.27 - gamma)

    # Lags temporels
    for h in [1, 3, 6, 12, 24]:
        X[f'lag_temp_{h}h'] = paris['temperature'].shift(h)
        X[f'lag_wind_{h}h'] = paris['wind_speed'].shift(h)
    for h in [1, 3, 6, 12]:
        X[f'lag_rain_{h}h'] = paris['rain'].shift(h)
    for h in [6, 12]:
        X[f'lag_hum_{h}h']  = paris['humidity'].shift(h)
    for h in [1, 6, 12]:
        X[f'lag_dir_{h}h']    = paris['wind_direction'].shift(h)
        X[f'lag_wind_u_{h}h'] = X['wind_u'].shift(h)
        X[f'lag_wind_v_{h}h'] = X['wind_v'].shift(h)

    # Deltas
    for h in [1, 3, 6, 12]:
        X[f'delta_temp_{h}h'] = paris['temperature'].diff(h)
        X[f'delta_wind_{h}h'] = paris['wind_speed'].diff(h)
        X[f'delta_dir_{h}h']  = paris['wind_direction'].diff(h)
    for h in [1, 3, 6]:
        X[f'delta_rain_{h}h'] = paris['rain'].diff(h)

    # Stats fenêtres (mean, std, min, max)
    for w in [3, 6, 12, 24]:
        for col, alias in [('temperature', 'temp'), ('wind_speed', 'wind')]:
            X[f'mean_{alias}_{w}h'] = paris[col].rolling(w).mean()
            X[f'std_{alias}_{w}h']  = paris[col].rolling(w).std()
            X[f'min_{alias}_{w}h']  = paris[col].rolling(w).min()
            X[f'max_{alias}_{w}h']  = paris[col].rolling(w).max()
    for w in [3, 6, 12]:
        X[f'mean_rain_{w}h'] = paris['rain'].rolling(w).mean()
        X[f'sum_rain_{w}h']  = paris['rain'].rolling(w).sum()
    for w in [6, 12]:
        X[f'mean_hum_{w}h']   = paris['humidity'].rolling(w).mean()
        X[f'mean_cloud_{w}h'] = paris['clouds'].rolling(w).mean()

    # Pentes linéaires
    for w in [3, 6, 12]:
        X[f'slope_temp_{w}h'] = (paris['temperature'] - paris['temperature'].shift(w)) / w
        X[f'slope_wind_{w}h'] = (paris['wind_speed']  - paris['wind_speed'].shift(w))  / w

    # Villes voisines pondérées par distance
    for city, dist in NEIGHBORS.items():
        w      = WEIGHTS[city]
        prefix = city.replace(' ', '_').replace('ü', 'u').replace('ö', 'o')
        cdf    = df_ref[df_ref['city_name'] == city][
            ['timestamp', 'temperature', 'rain', 'wind_speed', 'wind_direction', 'humidity']
        ].copy().sort_values('timestamp').rename(columns={
            'temperature':    f'{prefix}_temp',
            'rain':           f'{prefix}_rain',
            'wind_speed':     f'{prefix}_wind',
            'wind_direction': f'{prefix}_dir',
            'humidity':       f'{prefix}_hum',
        })
        m = paris.merge(cdf, on='timestamp', how='left')
        X[f'{prefix}_temp_w']    = m[f'{prefix}_temp'].values * w
        X[f'{prefix}_wind_w']    = m[f'{prefix}_wind'].values * w
        X[f'{prefix}_rain_w']    = m[f'{prefix}_rain'].values * w
        X[f'{prefix}_grad_temp'] = (paris['temperature'].values - m[f'{prefix}_temp'].values) * w
        X[f'{prefix}_grad_wind'] = (paris['wind_speed'].values  - m[f'{prefix}_wind'].values) * w
        X[f'{prefix}_temp_lag6'] = m[f'{prefix}_temp'].shift(6).values * w
        X[f'{prefix}_wind_lag6'] = m[f'{prefix}_wind'].shift(6).values * w
        X[f'{prefix}_wind_u']    = m[f'{prefix}_wind'].values * np.sin(np.deg2rad(m[f'{prefix}_dir'].values)) * w
        X[f'{prefix}_wind_v']    = m[f'{prefix}_wind'].values * np.cos(np.deg2rad(m[f'{prefix}_dir'].values)) * w

    # Targets : DELTA pour temp et vent, valeur absolue pour pluie
    X['y_delta_temp'] = paris['temperature'].shift(-6).values - paris['temperature'].values
    X['y_temp_t0']    = paris['temperature'].values
    X['y_delta_wind'] = paris['wind_speed'].shift(-6).values  - paris['wind_speed'].values
    X['y_wind_t0']    = paris['wind_speed'].values
    X['y_rain']       = paris['rain'].shift(-6).values
    X['timestamp']    = paris['timestamp'].values

    X = X.dropna().reset_index(drop=True)

    feat_cols = [c for c in X.columns if c not in [
        'y_delta_temp', 'y_temp_t0', 'y_delta_wind',
        'y_wind_t0', 'y_rain', 'timestamp'
    ]]
    return (
        X[feat_cols].values.astype(np.float32),
        X['y_delta_temp'].values.astype(np.float32),
        X['y_temp_t0'].values.astype(np.float32),
        X['y_delta_wind'].values.astype(np.float32),
        X['y_wind_t0'].values.astype(np.float32),
        X['y_rain'].values.astype(np.float32),
        pd.to_datetime(X['timestamp']),
        feat_cols
    )

print("Feature engineering...")
(X_tr, y_dtemp_tr, y_temp0_tr,
 y_dwind_tr, y_wind0_tr,
 y_rain_tr, _, feat_cols)    = build_features(df_train, df_all)

(X_te, y_dtemp_te, y_temp0_te,
 y_dwind_te, y_wind0_te,
 y_rain_te, ts_te, _)        = build_features(df_test, df_all)

y_temp_te = y_temp0_te + y_dtemp_te
y_wind_te = y_wind0_te + y_dwind_te
print(f"X_train {X_tr.shape} | X_test {X_te.shape} | {len(feat_cols)} features\n")

# ── Scaling pour KNN ─────────────────────────────────
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

# ── Entrainement des modèles ─────────────────────────
# GBR prédit le DELTA (changement à T+6h)
# KNN prédit la valeur absolue de la pluie

print("Training GBR température (delta)...")
gbr_temp = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=4,
    subsample=0.8, min_samples_leaf=8, max_features=0.8, random_state=42
)
gbr_temp.fit(X_tr, y_dtemp_tr)

print("Training GBR vent (delta)...")
gbr_wind = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=4,
    subsample=0.8, min_samples_leaf=8, max_features=0.8, random_state=42
)
gbr_wind.fit(X_tr, y_dwind_tr)

print("Training KNN pluie...")
knn_rain = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
knn_rain.fit(X_tr_sc, y_rain_tr)

# ── Sauvegarde des modèles ───────────────────────────
joblib.dump(gbr_temp, f"{DATA_DIR}/model_temp.pkl")
joblib.dump(gbr_wind, f"{DATA_DIR}/model_wind.pkl")
joblib.dump(knn_rain, f"{DATA_DIR}/model_rain.pkl")
joblib.dump(scaler,   f"{DATA_DIR}/scaler.pkl")
joblib.dump(feat_cols,f"{DATA_DIR}/feat_cols.pkl")
print("Modèles sauvegardés.\n")

# ── Evaluation sur 2025-2026 ─────────────────────────
pred_temp = y_temp0_te + gbr_temp.predict(X_te)
pred_wind = np.clip(y_wind0_te + gbr_wind.predict(X_te), 0, None)
pred_rain = np.clip(knn_rain.predict(X_te_sc), 0, None)
pred_rain = np.where(pred_rain < 1.0, 0.0, pred_rain)  # seuil optimal = 1.0

mae_temp = np.mean(np.abs(pred_temp - y_temp_te))
mae_wind = np.mean(np.abs(pred_wind - y_wind_te))
mae_rain = np.mean(np.abs(pred_rain - y_rain_te))
score    = np.mean([-mae_temp/STD_TEMP, -mae_wind/STD_WIND, -mae_rain/STD_RAIN])

print(f"Score température : {-mae_temp/STD_TEMP:.4f}  (MAE = {mae_temp:.4f}°C)")
print(f"Score vent        : {-mae_wind/STD_WIND:.4f}  (MAE = {mae_wind:.4f} m/s)")
print(f"Score pluie       : {-mae_rain/STD_RAIN:.4f}  (MAE = {mae_rain:.4f} mm)")
print(f"Score final       : {score:.4f}  (baseline = -0.3000)")
