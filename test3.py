import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import joblib
import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────

STD_WIND  = 5.05
DATA_DIR  = r"C:\Users\Massy\Documents\ProjetAi\Données\clean data"
N_LOOPS   = 5

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

_dists  = np.array(list(NEIGHBORS.values()), dtype=np.float32)
_sigma  = float(np.median(_dists))
WEIGHTS = {city: 1.0 / (1.0 + d / _sigma) for city, d in NEIGHBORS.items()}

PKL_SCALER = os.path.join(DATA_DIR, 'scaler_wind.pkl')
PKL_RIDGE  = os.path.join(DATA_DIR, 'model_wind_ridge.pkl')
PKL_GB     = os.path.join(DATA_DIR, 'model_wind_gb.pkl')
PKL_RF     = os.path.join(DATA_DIR, 'model_wind_rf.pkl')


# ═══════════════════════════════════════════════════════
# FONCTION : Chargement + Feature Engineering
# ═══════════════════════════════════════════════════════

def load_and_engineer() -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("  ETAPE 1 — Chargement des fichiers CSV")
    print("=" * 60)

    dfs = []
    for year in range(2020, 2027):
        path = os.path.join(DATA_DIR, f"weather_{year}_clean.csv")
        try:
            t0 = time.time()
            df_year = pd.read_csv(path)
            df_year['timestamp'] = pd.to_datetime(df_year['timestamp'], utc=True)
            dfs.append(df_year)
            villes = df_year['city_name'].nunique() if 'city_name' in df_year.columns else '?'
            print(f"  [OK] {year} charge  — {len(df_year):>7,} lignes | "
                  f"{villes} villes | {time.time()-t0:.2f}s")
        except FileNotFoundError:
            print(f"  [!]  {year} — fichier introuvable dans {DATA_DIR}, ignore")

    if not dfs:
        raise FileNotFoundError(f"Aucun CSV trouve dans : {DATA_DIR}")

    print(f"\n  -> Concatenation de {len(dfs)} fichiers...")
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['city_name', 'timestamp']).reset_index(drop=True)
    print(f"  [OK] Dataset brut total : {len(df):,} lignes | "
          f"{df['city_name'].nunique()} villes\n")

    # Isolation Paris
    print("  -> Isolation des donnees Paris...")
    paris = df[df['city_name'] == 'Paris'].copy()
    paris = paris.sort_values('timestamp').reset_index(drop=True)
    print(f"  [OK] Paris : {len(paris):,} lignes  "
          f"({paris['timestamp'].dt.year.min()}-{paris['timestamp'].dt.year.max()})")

    # ── Feature Engineering ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETAPE 2 — Feature Engineering (Paris) — cible = VENT T+6h")
    print("=" * 60)

    print("  -> Lags vent (1h 3h 6h 12h 24h)...")
    for h in [1, 3, 6, 12, 24]:
        paris[f'lag_wind_{h}h'] = paris['wind_speed'].shift(h)

    print("  -> Lags direction vent (1h 6h 12h)...")
    for h in [1, 6, 12]:
        paris[f'lag_dir_{h}h'] = paris['wind_direction'].shift(h)

    print("  -> Lags temperature (1h 3h 6h 12h)...")
    for h in [1, 3, 6, 12]:
        paris[f'lag_temp_{h}h'] = paris['temperature'].shift(h)

    print("  -> Lags pluie (1h 3h 6h)...")
    for h in [1, 3, 6]:
        paris[f'lag_rain_{h}h'] = paris['rain'].shift(h)

    print("  -> Lags humidite (6h 12h) et pression (si dispo)...")
    for h in [6, 12]:
        paris[f'lag_hum_{h}h'] = paris['humidity'].shift(h)

    print("  -> Deltas vent (1h 3h 6h 12h)...")
    for h in [1, 3, 6, 12]:
        paris[f'delta_wind_{h}h'] = paris['wind_speed'].diff(h)

    print("  -> Deltas temperature (1h 6h) et pluie (1h 3h)...")
    for h in [1, 6]:
        paris[f'delta_temp_{h}h'] = paris['temperature'].diff(h)
    for h in [1, 3]:
        paris[f'delta_rain_{h}h'] = paris['rain'].diff(h)

    print("  -> Rolling stats VENT (3h 6h 12h 24h) : mean/std/min/max...")
    for w in [3, 6, 12, 24]:
        paris[f'roll_mean_wind_{w}h'] = paris['wind_speed'].rolling(w).mean()
        paris[f'roll_std_wind_{w}h']  = paris['wind_speed'].rolling(w).std()
        paris[f'roll_min_wind_{w}h']  = paris['wind_speed'].rolling(w).min()
        paris[f'roll_max_wind_{w}h']  = paris['wind_speed'].rolling(w).max()

    print("  -> Rolling stats temperature (6h 12h 24h) : mean/std...")
    for w in [6, 12, 24]:
        paris[f'roll_mean_temp_{w}h'] = paris['temperature'].rolling(w).mean()
        paris[f'roll_std_temp_{w}h']  = paris['temperature'].rolling(w).std()

    print("  -> Rolling stats pluie (3h 6h 12h) : mean/sum...")
    for w in [3, 6, 12]:
        paris[f'roll_mean_rain_{w}h'] = paris['rain'].rolling(w).mean()
        paris[f'roll_sum_rain_{w}h']  = paris['rain'].rolling(w).sum()

    print("  -> Rolling humidite + nuages (6h 12h)...")
    for w in [6, 12]:
        paris[f'roll_mean_hum_{w}h']   = paris['humidity'].rolling(w).mean()
        paris[f'roll_mean_cloud_{w}h'] = paris['clouds'].rolling(w).mean()

    print("  -> Encodage cyclique heure/mois (sin/cos)...")
    paris['sin_hour']  = np.sin(2 * np.pi * paris['hour']  / 24)
    paris['cos_hour']  = np.cos(2 * np.pi * paris['hour']  / 24)
    paris['sin_month'] = np.sin(2 * np.pi * paris['month'] / 12)
    paris['cos_month'] = np.cos(2 * np.pi * paris['month'] / 12)

    print("  -> Composantes vent U/V (decomposition vectorielle)...")
    paris['wind_u'] = paris['wind_speed'] * np.sin(np.deg2rad(paris['wind_direction']))
    paris['wind_v'] = paris['wind_speed'] * np.cos(np.deg2rad(paris['wind_direction']))

    # Lags U/V — signal de direction upwind
    for h in [1, 6, 12]:
        paris[f'lag_wind_u_{h}h'] = paris['wind_u'].shift(h)
        paris[f'lag_wind_v_{h}h'] = paris['wind_v'].shift(h)

    print("  -> Dew point depression + persistance de vent fort (>5 m/s)...")
    T     = paris['temperature']
    RH    = paris['humidity'].clip(1, 100)
    a, b  = 17.27, 237.7
    gamma = (a * T) / (b + T) + np.log(RH / 100)
    paris['dew_depression'] = T - (b * gamma) / (a - gamma)

    paris['wind_strong']       = (paris['wind_speed'] > 5).astype(int)
    paris['wind_persistence']  = (
        paris['wind_strong']
        .groupby((paris['wind_strong'] == 0).cumsum())
        .cumsum()
    )

    # ── Features villes voisines ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ETAPE 3 — Features villes voisines (vent upwind)")
    print("=" * 60)

    for city, dist in NEIGHBORS.items():
        w      = WEIGHTS[city]
        prefix = (city.replace(' ', '_')
                      .replace('\xfc', 'u')
                      .replace('\xf6', 'o'))
        print(f"  -> {city:<22} dist={dist}km  poids={w:.4f}  "
              f"-> merge + gradients vent + lag6h...")

        city_df = df[df['city_name'] == city][
            ['timestamp', 'temperature', 'rain',
             'wind_speed', 'wind_direction', 'humidity']
        ].copy().sort_values('timestamp')

        city_df = city_df.rename(columns={
            'temperature':    f'{prefix}_temp',
            'rain':           f'{prefix}_rain',
            'wind_speed':     f'{prefix}_wind',
            'humidity':       f'{prefix}_hum',
            'wind_direction': f'{prefix}_dir',
        })

        paris = paris.merge(
            city_df[['timestamp',
                      f'{prefix}_temp', f'{prefix}_rain',
                      f'{prefix}_wind', f'{prefix}_hum']],
            on='timestamp', how='left'
        )

        paris[f'{prefix}_wind_w']    = paris[f'{prefix}_wind'] * w
        paris[f'{prefix}_grad_wind'] = (paris['wind_speed'] - paris[f'{prefix}_wind']) * w
        paris[f'{prefix}_wind_lag6'] = paris[f'{prefix}_wind'].shift(6) * w
        paris[f'{prefix}_temp_w']    = paris[f'{prefix}_temp'] * w
        paris[f'{prefix}_grad_temp'] = (paris['temperature'] - paris[f'{prefix}_temp']) * w

    # ── Cible T+6h ───────────────────────────────────────────────────────
    print("\n  -> Creation de la cible : wind_speed T+6h...")
    paris['target_wind'] = paris['wind_speed'].shift(-6)

    n_before = len(paris)
    paris    = paris.dropna().reset_index(drop=True)
    n_after  = len(paris)
    print(f"  -> Suppression NaN : {n_before:,} -> {n_after:,} lignes "
          f"({n_before - n_after:,} supprimees)")

    years_ok = sorted(paris['timestamp'].dt.year.unique())
    print(f"  [OK] Dataset final : {n_after:,} lignes | annees : {years_ok}\n")
    return paris


# ═══════════════════════════════════════════════════════
# FEATURES pour chaque modele
# ═══════════════════════════════════════════════════════

# Ridge : features lineaires — vent + context meteo
FEATURES_RIDGE_BASE = [
    # Vent actuel
    'wind_speed', 'wind_direction', 'wind_u', 'wind_v',
    # Context meteo
    'temperature', 'humidity', 'clouds', 'rain', 'dew_depression',
    # Cyclique
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    # Lags vent
    'lag_wind_1h', 'lag_wind_3h', 'lag_wind_6h', 'lag_wind_12h', 'lag_wind_24h',
    # Lags U/V
    'lag_wind_u_1h', 'lag_wind_v_1h', 'lag_wind_u_6h', 'lag_wind_v_6h',
    # Deltas vent
    'delta_wind_1h', 'delta_wind_3h', 'delta_wind_6h', 'delta_wind_12h',
    # Rolling vent
    'roll_mean_wind_3h', 'roll_std_wind_3h',
    'roll_mean_wind_6h', 'roll_std_wind_6h',
    'roll_min_wind_6h',  'roll_max_wind_6h',
    'roll_mean_wind_12h','roll_std_wind_12h',
    'roll_mean_wind_24h',
    # Rolling temp et humidite
    'roll_mean_temp_6h', 'roll_std_temp_6h', 'roll_mean_temp_12h',
    'roll_mean_hum_6h',  'roll_mean_cloud_6h',
    # Voisins vent
    'Brussels_wind_w',   'Brussels_grad_wind', 'Brussels_wind_lag6',
    'London_wind_w',     'London_grad_wind',   'London_wind_lag6',
    'Rotterdam_wind_w',  'Rotterdam_grad_wind',
    'Amsterdam_wind_w',  'Amsterdam_grad_wind',
    'D_sseldorf_wind_w', 'D_sseldorf_grad_wind',
]

# GBR + RF : features completes (non-lineaires)
FEATURES_GB_BASE = [
    # Vent actuel
    'wind_speed', 'wind_direction', 'wind_u', 'wind_v',
    # Context meteo
    'temperature', 'humidity', 'clouds', 'rain', 'snow', 'dew_depression',
    # Cyclique
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    # Lags vent
    'lag_wind_1h', 'lag_wind_3h', 'lag_wind_6h', 'lag_wind_12h', 'lag_wind_24h',
    # Lags direction
    'lag_dir_1h', 'lag_dir_6h', 'lag_dir_12h',
    # Lags U/V
    'lag_wind_u_1h', 'lag_wind_v_1h',
    'lag_wind_u_6h', 'lag_wind_v_6h',
    'lag_wind_u_12h','lag_wind_v_12h',
    # Lags temp, pluie, humidite
    'lag_temp_1h', 'lag_temp_3h', 'lag_temp_6h', 'lag_temp_12h',
    'lag_rain_1h', 'lag_rain_3h', 'lag_rain_6h',
    'lag_hum_6h',  'lag_hum_12h',
    # Deltas vent
    'delta_wind_1h', 'delta_wind_3h', 'delta_wind_6h', 'delta_wind_12h',
    # Deltas temp et pluie
    'delta_temp_1h', 'delta_temp_6h',
    'delta_rain_1h', 'delta_rain_3h',
    # Rolling vent
    'roll_mean_wind_3h', 'roll_std_wind_3h',
    'roll_mean_wind_6h', 'roll_std_wind_6h',
    'roll_min_wind_6h',  'roll_max_wind_6h',
    'roll_mean_wind_12h','roll_std_wind_12h',
    'roll_mean_wind_24h',
    # Rolling temp
    'roll_mean_temp_6h', 'roll_std_temp_6h',
    'roll_mean_temp_12h','roll_mean_temp_24h',
    # Rolling pluie
    'roll_mean_rain_3h', 'roll_sum_rain_3h',
    'roll_mean_rain_6h', 'roll_sum_rain_6h',
    'roll_mean_rain_12h',
    # Rolling humidite et nuages
    'roll_mean_hum_6h',   'roll_mean_hum_12h',
    'roll_mean_cloud_6h', 'roll_mean_cloud_12h',
    # Persistance vent fort
    'wind_persistence',
    # Voisins vent
    'Brussels_wind_w',   'Brussels_grad_wind', 'Brussels_wind_lag6',
    'London_wind_w',     'London_grad_wind',   'London_wind_lag6',
    'Rotterdam_wind_w',  'Rotterdam_grad_wind','Rotterdam_wind_lag6',
    'Amsterdam_wind_w',  'Amsterdam_grad_wind',
    'D_sseldorf_wind_w', 'D_sseldorf_grad_wind',
    'Essen_wind_w',      'Dortmund_wind_w',
    # Voisins temp (gradient de pression proxy)
    'Brussels_temp_w',   'Brussels_grad_temp',
    'London_temp_w',     'London_grad_temp',
    'Rotterdam_temp_w',  'Rotterdam_grad_temp',
]


# ═══════════════════════════════════════════════════════
# CHARGEMENT DES DONNEES (une seule fois)
# ═══════════════════════════════════════════════════════

t_global = time.time()
paris    = load_and_engineer()

FEATURES_RIDGE = [f for f in FEATURES_RIDGE_BASE if f in paris.columns]
FEATURES_GB    = [f for f in FEATURES_GB_BASE    if f in paris.columns]

print("=" * 60)
print("  ETAPE 4 — Split Train 2020-2024 / Test 2025-2026")
print("=" * 60)
print(f"  Features Ridge actives : {len(FEATURES_RIDGE)} / {len(FEATURES_RIDGE_BASE)}")
print(f"  Features GB/RF actives : {len(FEATURES_GB)} / {len(FEATURES_GB_BASE)}")

train = paris[paris['timestamp'].dt.year <= 2024].copy()
test  = paris[paris['timestamp'].dt.year >= 2025].copy()

print(f"\n  [TRAIN] {len(train):>7,} samples  "
      f"({train['timestamp'].dt.year.min()}-{train['timestamp'].dt.year.max()}) "
      f"<-- ENTRAINEMENT")
print(f"  [TEST]  {len(test):>7,} samples  "
      f"({test['timestamp'].dt.year.min()}-{test['timestamp'].dt.year.max()}) "
      f"<-- EVALUATION MAE")

for yr in [2025, 2026]:
    n      = (test['timestamp'].dt.year == yr).sum()
    status = "[OK]" if n > 0 else "[!] VIDE"
    print(f"         {status} {yr} : {n:,} lignes dans le test set")

print("\n  -> Preparation des matrices numpy X/y...")
X_train_ridge = train[FEATURES_RIDGE].values.astype(np.float32)
X_test_ridge  = test[FEATURES_RIDGE].values.astype(np.float32)
X_train_gb    = train[FEATURES_GB].values.astype(np.float32)
X_test_gb     = test[FEATURES_GB].values.astype(np.float32)

y_train_wind  = train['target_wind'].values.astype(np.float32)
y_test_wind   = test['target_wind'].values.astype(np.float32)

print(f"  [OK] X_train_ridge : {X_train_ridge.shape}  |  X_test_ridge : {X_test_ridge.shape}")
print(f"  [OK] X_train_gb    : {X_train_gb.shape}  |  X_test_gb    : {X_test_gb.shape}")
print(f"  [OK] y_test  plage : [{y_test_wind.min():.2f} m/s ... {y_test_wind.max():.2f} m/s]")
print(f"  [OK] y_test  moy.  : {y_test_wind.mean():.2f} m/s  std : {y_test_wind.std():.2f} m/s")


# ═══════════════════════════════════════════════════════
# BOUCLE DE TEST x 5
# ═══════════════════════════════════════════════════════

print("\n\n" + "#" * 60)
print("  BOUCLE DE TEST — 5 ITERATIONS — VENT T+6h")
print(f"  Entraine sur : 2020-2024  |  Teste sur : 2025-2026")
print("#" * 60)

mae_ridge_list = []
mae_gb_list    = []
mae_rf_list    = []

for loop in range(1, N_LOOPS + 1):

    t_loop = time.time()
    print(f"\n{'=' * 60}")
    print(f"  >>> ITERATION {loop} / {N_LOOPS}  —  VENT T+6h")
    print(f"{'=' * 60}")

    # ─────────────────────────────────────────────────────
    # RIDGE
    # ─────────────────────────────────────────────────────
    print(f"\n  +-- [RIDGE] -------------------------------------------+")

    if loop == 1:
        print(f"  |  Iteration 1 -> CREATION scaler + modele Ridge")
        print(f"  |  -> StandardScaler fit sur X_train "
              f"({X_train_ridge.shape[0]:,} samples x {X_train_ridge.shape[1]} features)...")
        t0     = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_ridge)
        X_test_scaled  = scaler.transform(X_test_ridge)
        print(f"  |  [OK] Scaling en {time.time()-t0:.2f}s  "
              f"(mean~{X_train_scaled.mean():.4f}  std~{X_train_scaled.std():.4f})")

        print(f"  |  -> Entrainement Ridge(alpha=1.0) sur vent T+6h...")
        t0           = time.time()
        model_ridge  = Ridge(alpha=1.0)
        model_ridge.fit(X_train_scaled, y_train_wind)
        print(f"  |  [OK] Ridge entraine en {time.time()-t0:.2f}s")

        joblib.dump(scaler,      PKL_SCALER)
        joblib.dump(model_ridge, PKL_RIDGE)
        print(f"  |  [SAVE] scaler_wind.pkl + model_wind_ridge.pkl crees")

    else:
        print(f"  |  Iteration {loop} -> CHARGEMENT des .pkl existants")
        scaler      = joblib.load(PKL_SCALER)
        model_ridge = joblib.load(PKL_RIDGE)
        print(f"  |  [OK] scaler_wind.pkl + model_wind_ridge.pkl charges")

        X_train_scaled = scaler.transform(X_train_ridge)
        X_test_scaled  = scaler.transform(X_test_ridge)

        print(f"  |  -> Re-entrainement Ridge (mise a jour des poids)...")
        t0 = time.time()
        model_ridge.fit(X_train_scaled, y_train_wind)
        print(f"  |  [OK] Re-entrainement en {time.time()-t0:.2f}s")

        joblib.dump(scaler,      PKL_SCALER)
        joblib.dump(model_ridge, PKL_RIDGE)
        print(f"  |  [SAVE] scaler_wind.pkl + model_wind_ridge.pkl mis a jour")

    print(f"  |  -> Prediction vent sur TEST 2025-2026 "
          f"({X_test_scaled.shape[0]:,} samples)...")
    pred_ridge = np.clip(model_ridge.predict(X_test_scaled), 0, None)
    mae_ridge  = float(np.mean(np.abs(pred_ridge - y_test_wind)))
    mae_ridge_list.append(mae_ridge)

    for yr in [2025, 2026]:
        mask_yr = test['timestamp'].dt.year.values == yr
        if mask_yr.sum() > 0:
            m = np.mean(np.abs(pred_ridge[mask_yr] - y_test_wind[mask_yr]))
            print(f"  |     {yr} -> MAE Ridge = {m:.4f} m/s  ({mask_yr.sum():,} heures)")

    print(f"  |")
    print(f"  |  [RESULTAT] MAE Ridge vent (2025+2026) = {mae_ridge:.4f} m/s  "
          f"| score = {-mae_ridge/STD_WIND:.4f}")
    print(f"  +------------------------------------------------------+")

    # ─────────────────────────────────────────────────────
    # GRADIENT BOOSTING
    # ─────────────────────────────────────────────────────
    print(f"\n  +-- [GRADIENT BOOSTING] --------------------------------+")

    if loop == 1:
        print(f"  |  Iteration 1 -> CREATION du modele GBR vent")
        print(f"  |  -> Params : n_estimators=500  lr=0.03  max_depth=4")
        print(f"  |              subsample=0.8  warm_start=True")
        print(f"  |  -> Entrainement sur {X_train_gb.shape[0]:,} samples "
              f"x {X_train_gb.shape[1]} features...")
        print(f"  |     (peut prendre 1-3 minutes...)")
        t0       = time.time()
        model_gb = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=42,
            warm_start=True,
        )
        model_gb.fit(X_train_gb, y_train_wind)
        elapsed = time.time() - t0
        print(f"  |  [OK] GBR entraine en {elapsed:.1f}s  "
              f"({model_gb.n_estimators_} arbres)")

        feat_imp = sorted(
            zip(FEATURES_GB, model_gb.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]
        print(f"  |  -> Top 5 features importantes :")
        for rank, (fname, fimp) in enumerate(feat_imp, 1):
            bar = "#" * int(fimp * 200)
            print(f"  |       {rank}. {fname:<35} {fimp:.4f}  {bar}")

        joblib.dump(model_gb, PKL_GB)
        print(f"  |  [SAVE] model_wind_gb.pkl cree et sauvegarde")

    else:
        print(f"  |  Iteration {loop} -> CHARGEMENT du GBR existant")
        model_gb = joblib.load(PKL_GB)
        n_before = model_gb.n_estimators_
        n_new    = n_before + 50
        print(f"  |  [OK] GBR charge — {n_before} arbres existants")
        print(f"  |  -> warm_start : ajout 50 arbres "
              f"({n_before} -> {n_new})...")
        model_gb.set_params(warm_start=True, n_estimators=n_new)
        t0 = time.time()
        model_gb.fit(X_train_gb, y_train_wind)
        print(f"  |  [OK] Incremental en {time.time()-t0:.1f}s  "
              f"({model_gb.n_estimators_} arbres au total)")

        joblib.dump(model_gb, PKL_GB)
        print(f"  |  [SAVE] model_wind_gb.pkl mis a jour")

    print(f"  |  -> Prediction vent sur TEST 2025-2026 "
          f"({X_test_gb.shape[0]:,} samples)...")
    pred_gb = np.clip(model_gb.predict(X_test_gb), 0, None)
    mae_gb  = float(np.mean(np.abs(pred_gb - y_test_wind)))
    mae_gb_list.append(mae_gb)

    for yr in [2025, 2026]:
        mask_yr = test['timestamp'].dt.year.values == yr
        if mask_yr.sum() > 0:
            m = np.mean(np.abs(pred_gb[mask_yr] - y_test_wind[mask_yr]))
            print(f"  |     {yr} -> MAE GBR = {m:.4f} m/s  ({mask_yr.sum():,} heures)")

    print(f"  |")
    print(f"  |  [RESULTAT] MAE GBR vent (2025+2026) = {mae_gb:.4f} m/s  "
          f"| score = {-mae_gb/STD_WIND:.4f}")
    print(f"  +------------------------------------------------------+")

    # ─────────────────────────────────────────────────────
    # RANDOM FOREST
    # ─────────────────────────────────────────────────────
    print(f"\n  +-- [RANDOM FOREST] ------------------------------------+")

    if loop == 1:
        print(f"  |  Iteration 1 -> CREATION du modele Random Forest vent")
        print(f"  |  -> Params : n_estimators=300  max_depth=12")
        print(f"  |              min_samples_leaf=8  max_features=0.5")
        print(f"  |              n_jobs=-1 (tous les coeurs CPU)")
        print(f"  |  -> Entrainement sur {X_train_gb.shape[0]:,} samples "
              f"x {X_train_gb.shape[1]} features...")
        print(f"  |     (peut prendre 30-90s...)")
        t0       = time.time()
        model_rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=8,
            max_features=0.5,
            n_jobs=-1,
            random_state=42,
            warm_start=True,
        )
        model_rf.fit(X_train_gb, y_train_wind)
        elapsed = time.time() - t0
        print(f"  |  [OK] RF entraine en {elapsed:.1f}s  "
              f"({model_rf.n_estimators} arbres)")

        feat_imp_rf = sorted(
            zip(FEATURES_GB, model_rf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]
        print(f"  |  -> Top 5 features importantes :")
        for rank, (fname, fimp) in enumerate(feat_imp_rf, 1):
            bar = "#" * int(fimp * 200)
            print(f"  |       {rank}. {fname:<35} {fimp:.4f}  {bar}")

        joblib.dump(model_rf, PKL_RF)
        print(f"  |  [SAVE] model_wind_rf.pkl cree et sauvegarde")

    else:
        print(f"  |  Iteration {loop} -> CHARGEMENT du RF existant")
        model_rf = joblib.load(PKL_RF)
        n_before = model_rf.n_estimators
        n_new    = n_before + 50
        print(f"  |  [OK] RF charge — {n_before} arbres existants")
        print(f"  |  -> warm_start : ajout 50 arbres "
              f"({n_before} -> {n_new})...")
        model_rf.set_params(warm_start=True, n_estimators=n_new)
        t0 = time.time()
        model_rf.fit(X_train_gb, y_train_wind)
        print(f"  |  [OK] Incremental en {time.time()-t0:.1f}s  "
              f"({model_rf.n_estimators} arbres au total)")

        joblib.dump(model_rf, PKL_RF)
        print(f"  |  [SAVE] model_wind_rf.pkl mis a jour")

    print(f"  |  -> Prediction vent sur TEST 2025-2026 "
          f"({X_test_gb.shape[0]:,} samples)...")
    pred_rf = np.clip(model_rf.predict(X_test_gb), 0, None)
    mae_rf  = float(np.mean(np.abs(pred_rf - y_test_wind)))
    mae_rf_list.append(mae_rf)

    for yr in [2025, 2026]:
        mask_yr = test['timestamp'].dt.year.values == yr
        if mask_yr.sum() > 0:
            m = np.mean(np.abs(pred_rf[mask_yr] - y_test_wind[mask_yr]))
            print(f"  |     {yr} -> MAE RF = {m:.4f} m/s  ({mask_yr.sum():,} heures)")

    print(f"  |")
    print(f"  |  [RESULTAT] MAE RF vent (2025+2026) = {mae_rf:.4f} m/s  "
          f"| score = {-mae_rf/STD_WIND:.4f}")
    print(f"  +------------------------------------------------------+")

    # ─────────────────────────────────────────────────────
    # RESUME ITERATION
    # ─────────────────────────────────────────────────────
    scores   = {'Ridge': mae_ridge, 'GBR': mae_gb, 'RF': mae_rf}
    winner   = min(scores, key=scores.get)
    t_iter   = time.time() - t_loop

    print(f"\n  >>> RESUME ITERATION {loop} — VENT T+6h :")
    print(f"      Ridge MAE = {mae_ridge:.4f} m/s  | score = {-mae_ridge/STD_WIND:.4f}")
    print(f"      GBR   MAE = {mae_gb:.4f} m/s  | score = {-mae_gb/STD_WIND:.4f}")
    print(f"      RF    MAE = {mae_rf:.4f} m/s  | score = {-mae_rf/STD_WIND:.4f}")
    print(f"      => Meilleur cette iteration : {winner}  ({scores[winner]:.4f} m/s)")
    if loop > 1:
        tr = mae_ridge_list[-1] - mae_ridge_list[-2]
        tg = mae_gb_list[-1]    - mae_gb_list[-2]
        tf = mae_rf_list[-1]    - mae_rf_list[-2]
        dir_r = "baisse" if tr < 0 else "hausse" if tr > 0 else "stable"
        dir_g = "baisse" if tg < 0 else "hausse" if tg > 0 else "stable"
        dir_f = "baisse" if tf < 0 else "hausse" if tf > 0 else "stable"
        print(f"      Evolution  Ridge : {dir_r} ({tr:+.4f})  "
              f"|  GBR : {dir_g} ({tg:+.4f})  "
              f"|  RF : {dir_f} ({tf:+.4f})")
    print(f"      Temps iteration : {t_iter:.1f}s")


# ═══════════════════════════════════════════════════════
# TABLEAU FINAL
# ═══════════════════════════════════════════════════════

print(f"\n\n{'#' * 70}")
print(f"  RECAPITULATIF — MAE VENT (m/s) | TEST : 2025-2026")
print(f"{'#' * 70}")
print(f"  {'Test':^5} | {'Ridge (m/s)':^12} | {'GBR (m/s)':^12} | "
      f"{'RF (m/s)':^12} | {'Meilleur':^10}")
print(f"  {'-' * 62}")
for i, (mr, mg, mf) in enumerate(zip(mae_ridge_list, mae_gb_list, mae_rf_list), start=1):
    scores = {'Ridge': mr, 'GBR': mg, 'RF': mf}
    best   = min(scores, key=scores.get)
    print(f"  {i:^5} | {mr:^12.4f} | {mg:^12.4f} | {mf:^12.4f} | {best:^10}")
print(f"  {'-' * 62}")
print(f"  {'Moy.':^5} | {np.mean(mae_ridge_list):^12.4f} | "
      f"{np.mean(mae_gb_list):^12.4f} | {np.mean(mae_rf_list):^12.4f} |")
print(f"  {'Min.':^5} | {np.min(mae_ridge_list):^12.4f} | "
      f"{np.min(mae_gb_list):^12.4f} | {np.min(mae_rf_list):^12.4f} |")
print(f"{'#' * 70}")
print(f"  Temps total : {time.time()-t_global:.1f}s\n")


# ═══════════════════════════════════════════════════════
# GRAPHE — 3 COURBES
# ═══════════════════════════════════════════════════════

print("-> Generation du graphe d'evolution MAE vent (3 modeles)...")

iterations   = list(range(1, N_LOOPS + 1))
COLOR_RIDGE  = '#00b4d8'   # cyan
COLOR_GB     = '#f77f00'   # orange
COLOR_RF     = '#7bc67e'   # vert clair

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#161b22')

# Remplissage sous les courbes
ax.fill_between(iterations, mae_ridge_list, alpha=0.08, color=COLOR_RIDGE)
ax.fill_between(iterations, mae_gb_list,    alpha=0.08, color=COLOR_GB)
ax.fill_between(iterations, mae_rf_list,    alpha=0.08, color=COLOR_RF)

# Courbes
ax.plot(iterations, mae_ridge_list,
        marker='o', markersize=10, linewidth=2.5, color=COLOR_RIDGE,
        label=f'Ridge              (moy. = {np.mean(mae_ridge_list):.3f} m/s)',
        zorder=5)

ax.plot(iterations, mae_gb_list,
        marker='s', markersize=10, linewidth=2.5, color=COLOR_GB,
        label=f'Gradient Boosting  (moy. = {np.mean(mae_gb_list):.3f} m/s)',
        zorder=5)

ax.plot(iterations, mae_rf_list,
        marker='^', markersize=10, linewidth=2.5, color=COLOR_RF,
        label=f'Random Forest      (moy. = {np.mean(mae_rf_list):.3f} m/s)',
        zorder=5)

# Annotations valeurs
for i, (mr, mg, mf) in enumerate(
        zip(mae_ridge_list, mae_gb_list, mae_rf_list), start=1):
    ax.annotate(f'{mr:.3f}',
                xy=(i, mr), xytext=(0, 13), textcoords='offset points',
                ha='center', fontsize=8.5, color=COLOR_RIDGE, fontweight='bold')
    ax.annotate(f'{mg:.3f}',
                xy=(i, mg), xytext=(0, -20), textcoords='offset points',
                ha='center', fontsize=8.5, color=COLOR_GB, fontweight='bold')
    ax.annotate(f'{mf:.3f}',
                xy=(i, mf), xytext=(14, 0), textcoords='offset points',
                ha='left', fontsize=8.5, color=COLOR_RF, fontweight='bold')

# Lignes de moyenne
ax.axhline(np.mean(mae_ridge_list), color=COLOR_RIDGE,
           linestyle='--', linewidth=1.1, alpha=0.40)
ax.axhline(np.mean(mae_gb_list), color=COLOR_GB,
           linestyle='--', linewidth=1.1, alpha=0.40)
ax.axhline(np.mean(mae_rf_list), color=COLOR_RF,
           linestyle='--', linewidth=1.1, alpha=0.40)

# Labels et titre
ax.set_xlabel('Numero du test (iteration)', fontsize=12,
              color='#e0e0e0', labelpad=10)
ax.set_ylabel('MAE Vent  |m/s|', fontsize=12,
              color='#e0e0e0', labelpad=10)
ax.set_title(
    'Evolution de la MAE Vent T+6h  —  Paris\n'
    'Entrainement : 2020-2024   |   Evaluation : 2025-2026',
    fontsize=14, fontweight='bold', color='#ffffff', pad=16
)

ax.set_xticks(iterations)
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.tick_params(colors='#aaaaaa', labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor('#2d333b')

ax.grid(True, which='major', linestyle='--',
        linewidth=0.6, color='#2d333b', alpha=0.8)
ax.legend(loc='upper right', fontsize=10,
          facecolor='#1e2430', edgecolor='#3d4556',
          labelcolor='#e0e0e0', framealpha=0.9)

all_mae = mae_ridge_list + mae_gb_list + mae_rf_list
ax.set_ylim(max(0, min(all_mae) - 0.3), max(all_mae) + 0.5)
ax.set_xlim(0.7, N_LOOPS + 0.3)

plt.tight_layout(pad=1.5)

output_plot = os.path.join(DATA_DIR, 'mae_evolution_vent.png')
plt.savefig(output_plot, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"[OK] Graphe sauvegarde -> {output_plot}\n")
plt.show()

print("SCRIPT TERMINE. Fichiers .pkl disponibles :")
print(f"  {DATA_DIR}")
print(f"  > scaler_wind.pkl      (StandardScaler Ridge vent)")
print(f"  > model_wind_ridge.pkl (Ridge — vent T+6h)")
print(f"  > model_wind_gb.pkl    (Gradient Boosting — vent T+6h)")
print(f"  > model_wind_rf.pkl    (Random Forest — vent T+6h)")
print(f"\n  Temps total : {time.time()-t_global:.1f}s")