import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────

STD_TEMP  = 7.49
STD_WIND  = 5.05
STD_RAIN  = 0.40
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


# ═══════════════════════════════════════════════════════
# CHARGEMENT + FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════

def load_and_engineer():

    print("\n" + "=" * 60)
    print("  ETAPE 1 — Chargement des fichiers CSV")
    print("=" * 60)

    dfs = []
    for year in range(2020, 2027):
        path = os.path.join(DATA_DIR, f"weather_{year}_clean.csv")
        try:
            t0      = time.time()
            df_year = pd.read_csv(path)
            df_year['timestamp'] = pd.to_datetime(df_year['timestamp'], utc=True)
            dfs.append(df_year)
            print(f"  [OK] {year} charge — {len(df_year):>7,} lignes | {time.time()-t0:.2f}s")
        except FileNotFoundError:
            print(f"  [!]  {year} — introuvable, ignore")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['city_name', 'timestamp']).reset_index(drop=True)
    print(f"\n  [OK] Total : {len(df):,} lignes | {df['city_name'].nunique()} villes\n")

    print("=" * 60)
    print("  ETAPE 2 — Feature Engineering (Paris) — 3 cibles T+6h")
    print("=" * 60)

    paris = df[df['city_name'] == 'Paris'].copy()
    paris = paris.sort_values('timestamp').reset_index(drop=True)
    print(f"  [OK] Paris : {len(paris):,} lignes")

    print("  -> Lags temperature, vent, pluie, humidite...")
    for h in [1, 3, 6, 12, 24]:
        paris[f'lag_temp_{h}h'] = paris['temperature'].shift(h)
        paris[f'lag_wind_{h}h'] = paris['wind_speed'].shift(h)
    for h in [1, 3, 6, 12]:
        paris[f'lag_rain_{h}h'] = paris['rain'].shift(h)
    for h in [6, 12]:
        paris[f'lag_hum_{h}h']  = paris['humidity'].shift(h)
    for h in [1, 6, 12]:
        paris[f'lag_dir_{h}h']  = paris['wind_direction'].shift(h)

    print("  -> Deltas temperature, vent, pluie...")
    for h in [1, 3, 6, 12]:
        paris[f'delta_temp_{h}h'] = paris['temperature'].diff(h)
        paris[f'delta_wind_{h}h'] = paris['wind_speed'].diff(h)
    for h in [1, 3, 6]:
        paris[f'delta_rain_{h}h'] = paris['rain'].diff(h)

    print("  -> Rolling stats (3h 6h 12h 24h) : mean/std/min/max...")
    for w in [3, 6, 12, 24]:
        paris[f'roll_mean_temp_{w}h'] = paris['temperature'].rolling(w).mean()
        paris[f'roll_std_temp_{w}h']  = paris['temperature'].rolling(w).std()
        paris[f'roll_min_temp_{w}h']  = paris['temperature'].rolling(w).min()
        paris[f'roll_max_temp_{w}h']  = paris['temperature'].rolling(w).max()
        paris[f'roll_mean_wind_{w}h'] = paris['wind_speed'].rolling(w).mean()
        paris[f'roll_std_wind_{w}h']  = paris['wind_speed'].rolling(w).std()
        paris[f'roll_min_wind_{w}h']  = paris['wind_speed'].rolling(w).min()
        paris[f'roll_max_wind_{w}h']  = paris['wind_speed'].rolling(w).max()
    for w in [3, 6, 12]:
        paris[f'roll_mean_rain_{w}h'] = paris['rain'].rolling(w).mean()
        paris[f'roll_sum_rain_{w}h']  = paris['rain'].rolling(w).sum()
    for w in [6, 12]:
        paris[f'roll_mean_hum_{w}h']   = paris['humidity'].rolling(w).mean()
        paris[f'roll_mean_cloud_{w}h'] = paris['clouds'].rolling(w).mean()

    print("  -> Encodage cyclique heure/mois (sin/cos)...")
    paris['sin_hour']  = np.sin(2 * np.pi * paris['hour']  / 24)
    paris['cos_hour']  = np.cos(2 * np.pi * paris['hour']  / 24)
    paris['sin_month'] = np.sin(2 * np.pi * paris['month'] / 12)
    paris['cos_month'] = np.cos(2 * np.pi * paris['month'] / 12)

    print("  -> Decomposition vent U/V...")
    paris['wind_u'] = paris['wind_speed'] * np.sin(np.deg2rad(paris['wind_direction']))
    paris['wind_v'] = paris['wind_speed'] * np.cos(np.deg2rad(paris['wind_direction']))
    for h in [1, 6, 12]:
        paris[f'lag_wind_u_{h}h'] = paris['wind_u'].shift(h)
        paris[f'lag_wind_v_{h}h'] = paris['wind_v'].shift(h)

    print("  -> Dew point depression...")
    T     = paris['temperature']
    RH    = paris['humidity'].clip(1, 100)
    a, b  = 17.27, 237.7
    gamma = (a * T) / (b + T) + np.log(RH / 100)
    paris['dew_depression'] = T - (b * gamma) / (a - gamma)

    print("  -> Persistance pluie et vent fort...")
    paris['rain_binary']     = (paris['rain'] > 0.1).astype(int)
    paris['rain_persistence'] = (
        paris['rain_binary']
        .groupby((paris['rain_binary'] == 0).cumsum())
        .cumsum()
    )
    paris['wind_strong']      = (paris['wind_speed'] > 5).astype(int)
    paris['wind_persistence'] = (
        paris['wind_strong']
        .groupby((paris['wind_strong'] == 0).cumsum())
        .cumsum()
    )

    print("\n" + "=" * 60)
    print("  ETAPE 3 — Features villes voisines")
    print("=" * 60)

    for city, dist in NEIGHBORS.items():
        w      = WEIGHTS[city]
        prefix = (city.replace(' ', '_')
                      .replace('ü', 'u')
                      .replace('ö', 'o'))
        print(f"  -> {city:<22} dist={dist}km  poids={w:.4f}...")
        city_df = df[df['city_name'] == city][
            ['timestamp', 'temperature', 'rain', 'wind_speed', 'humidity']
        ].copy().sort_values('timestamp')
        city_df = city_df.rename(columns={
            'temperature': f'{prefix}_temp',
            'rain':        f'{prefix}_rain',
            'wind_speed':  f'{prefix}_wind',
            'humidity':    f'{prefix}_hum',
        })
        paris = paris.merge(
            city_df[['timestamp', f'{prefix}_temp',
                     f'{prefix}_rain', f'{prefix}_wind', f'{prefix}_hum']],
            on='timestamp', how='left'
        )
        paris[f'{prefix}_temp_w']    = paris[f'{prefix}_temp'] * w
        paris[f'{prefix}_wind_w']    = paris[f'{prefix}_wind'] * w
        paris[f'{prefix}_rain_w']    = paris[f'{prefix}_rain'] * w
        paris[f'{prefix}_grad_temp'] = (paris['temperature'] - paris[f'{prefix}_temp']) * w
        paris[f'{prefix}_grad_wind'] = (paris['wind_speed']  - paris[f'{prefix}_wind']) * w
        paris[f'{prefix}_grad_rain'] = (paris['rain']        - paris[f'{prefix}_rain']) * w
        paris[f'{prefix}_temp_lag6'] = paris[f'{prefix}_temp'].shift(6) * w
        paris[f'{prefix}_wind_lag6'] = paris[f'{prefix}_wind'].shift(6) * w

    print("\n  -> Creation des 3 targets T+6h...")
    paris['target_temp'] = paris['temperature'].shift(-6)
    paris['target_wind'] = paris['wind_speed'].shift(-6)
    paris['target_rain'] = paris['rain'].shift(-6)

    n_before = len(paris)
    paris    = paris.dropna().reset_index(drop=True)
    print(f"  -> NaN supprimes : {n_before:,} -> {len(paris):,} lignes")
    years_ok = sorted(paris['timestamp'].dt.year.unique())
    print(f"  [OK] Dataset final : {len(paris):,} lignes | annees : {years_ok}\n")
    return paris


# ═══════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════

FEATURES_RIDGE = [
    'temperature', 'humidity', 'clouds', 'rain', 'snow',
    'wind_u', 'wind_v', 'dew_depression',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'lag_temp_1h', 'lag_temp_3h', 'lag_temp_6h', 'lag_temp_12h', 'lag_temp_24h',
    'lag_wind_1h', 'lag_wind_3h', 'lag_wind_6h', 'lag_wind_12h',
    'lag_rain_1h', 'lag_rain_3h', 'lag_rain_6h',
    'delta_temp_1h', 'delta_temp_3h', 'delta_temp_6h',
    'delta_wind_1h', 'delta_wind_3h', 'delta_wind_6h',
    'roll_mean_temp_6h', 'roll_std_temp_6h', 'roll_mean_temp_12h', 'roll_mean_temp_24h',
    'roll_mean_wind_6h', 'roll_std_wind_6h', 'roll_mean_wind_12h',
    'roll_mean_rain_6h', 'roll_sum_rain_6h',
    'roll_mean_hum_6h', 'roll_mean_cloud_6h',
    'Brussels_temp_w', 'Brussels_wind_w', 'Brussels_grad_temp', 'Brussels_temp_lag6',
    'London_temp_w',   'London_wind_w',   'London_grad_temp',   'London_temp_lag6',
    'Rotterdam_temp_w','Rotterdam_wind_w','Rotterdam_grad_temp',
    'Amsterdam_temp_w','Amsterdam_wind_w',
    'D_sseldorf_temp_w','D_sseldorf_wind_w',
]

FEATURES_GB = [
    'temperature', 'humidity', 'clouds', 'rain', 'snow',
    'wind_u', 'wind_v', 'wind_speed', 'wind_direction',
    'dew_depression', 'rain_persistence', 'wind_persistence',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'lag_temp_1h', 'lag_temp_3h', 'lag_temp_6h', 'lag_temp_12h', 'lag_temp_24h',
    'lag_wind_1h', 'lag_wind_3h', 'lag_wind_6h', 'lag_wind_12h', 'lag_wind_24h',
    'lag_dir_1h', 'lag_dir_6h', 'lag_dir_12h',
    'lag_wind_u_1h', 'lag_wind_v_1h', 'lag_wind_u_6h', 'lag_wind_v_6h',
    'lag_rain_1h', 'lag_rain_3h', 'lag_rain_6h', 'lag_rain_12h',
    'lag_hum_6h',  'lag_hum_12h',
    'delta_temp_1h', 'delta_temp_3h', 'delta_temp_6h', 'delta_temp_12h',
    'delta_wind_1h', 'delta_wind_3h', 'delta_wind_6h', 'delta_wind_12h',
    'delta_rain_1h', 'delta_rain_3h', 'delta_rain_6h',
    'roll_mean_temp_3h', 'roll_std_temp_3h',
    'roll_mean_temp_6h', 'roll_std_temp_6h', 'roll_min_temp_6h', 'roll_max_temp_6h',
    'roll_mean_temp_12h', 'roll_mean_temp_24h',
    'roll_mean_wind_3h', 'roll_std_wind_3h',
    'roll_mean_wind_6h', 'roll_std_wind_6h', 'roll_min_wind_6h', 'roll_max_wind_6h',
    'roll_mean_wind_12h', 'roll_mean_wind_24h',
    'roll_mean_rain_3h', 'roll_sum_rain_3h',
    'roll_mean_rain_6h', 'roll_sum_rain_6h', 'roll_mean_rain_12h',
    'roll_mean_hum_6h',   'roll_mean_hum_12h',
    'roll_mean_cloud_6h', 'roll_mean_cloud_12h',
    'Brussels_temp_w', 'Brussels_wind_w', 'Brussels_rain_w',
    'Brussels_grad_temp', 'Brussels_grad_wind', 'Brussels_temp_lag6', 'Brussels_wind_lag6',
    'London_temp_w',   'London_wind_w',   'London_rain_w',
    'London_grad_temp','London_grad_wind','London_temp_lag6', 'London_wind_lag6',
    'Rotterdam_temp_w','Rotterdam_wind_w','Rotterdam_grad_temp','Rotterdam_grad_wind',
    'Amsterdam_temp_w','Amsterdam_wind_w','Amsterdam_grad_temp',
    'D_sseldorf_temp_w','D_sseldorf_wind_w','D_sseldorf_grad_temp',
    'Essen_temp_w','Dortmund_temp_w',
]


# ═══════════════════════════════════════════════════════
# CHARGEMENT DONNEES
# ═══════════════════════════════════════════════════════

t_global = time.time()
paris    = load_and_engineer()

FEAT_RIDGE = [f for f in FEATURES_RIDGE if f in paris.columns]
FEAT_GB    = [f for f in FEATURES_GB    if f in paris.columns]

print("=" * 60)
print("  ETAPE 4 — Split Train 2020-2024 / Test 2025-2026")
print("=" * 60)
print(f"  Features Ridge : {len(FEAT_RIDGE)}")
print(f"  Features GB/RF : {len(FEAT_GB)}")

train = paris[paris['timestamp'].dt.year <= 2024].copy()
test  = paris[paris['timestamp'].dt.year >= 2025].copy()

print(f"\n  [TRAIN] {len(train):>7,} samples (2020-2024)")
print(f"  [TEST]  {len(test):>7,} samples (2025-2026)")
for yr in [2025, 2026]:
    n = (test['timestamp'].dt.year == yr).sum()
    print(f"         [OK] {yr} : {n:,} lignes")

X_train_ridge = train[FEAT_RIDGE].values.astype(np.float32)
X_test_ridge  = test[FEAT_RIDGE].values.astype(np.float32)
X_train_gb    = train[FEAT_GB].values.astype(np.float32)
X_test_gb     = test[FEAT_GB].values.astype(np.float32)

y_train_temp = train['target_temp'].values.astype(np.float32)
y_test_temp  = test['target_temp'].values.astype(np.float32)
y_train_wind = train['target_wind'].values.astype(np.float32)
y_test_wind  = test['target_wind'].values.astype(np.float32)
y_train_rain = train['target_rain'].values.astype(np.float32)
y_test_rain  = test['target_rain'].values.astype(np.float32)

print(f"\n  [OK] X_train_ridge : {X_train_ridge.shape}")
print(f"  [OK] X_train_gb    : {X_train_gb.shape}")


# ═══════════════════════════════════════════════════════
# SCALING
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  ETAPE 5 — StandardScaler (Ridge + KNN)")
print("=" * 60)

scaler_ridge      = StandardScaler()
X_train_ridge_sc  = scaler_ridge.fit_transform(X_train_ridge)
X_test_ridge_sc   = scaler_ridge.transform(X_test_ridge)

scaler_gb         = StandardScaler()
X_train_gb_sc     = scaler_gb.fit_transform(X_train_gb)
X_test_gb_sc      = scaler_gb.transform(X_test_gb)

print(f"  [OK] Scalers fits OK\n")


# ═══════════════════════════════════════════════════════
# CONFIG MODELES
# ═══════════════════════════════════════════════════════

# Chaque modele : (objet, X_train, X_test, use_warm, add_trees)
def make_models():
    return {
        'Ridge': {
            'model'     : Ridge(alpha=1.0),
            'X_train'   : X_train_ridge_sc,
            'X_test'    : X_test_ridge_sc,
            'warm'      : False,
        },
        'GBR': {
            'model'     : GradientBoostingRegressor(
                            n_estimators=500, learning_rate=0.03, max_depth=4,
                            subsample=0.8, min_samples_leaf=8, max_features=0.8,
                            random_state=42, warm_start=True),
            'X_train'   : X_train_gb,
            'X_test'    : X_test_gb,
            'warm'      : True,
            'add_trees' : 50,
        },
        'RF': {
            'model'     : RandomForestRegressor(
                            n_estimators=300, max_depth=12, min_samples_leaf=8,
                            max_features=0.5, n_jobs=-1, random_state=42, warm_start=True),
            'X_train'   : X_train_gb,
            'X_test'    : X_test_gb,
            'warm'      : True,
            'add_trees' : 50,
        },
        'KNN': {
            'model'     : KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1),
            'X_train'   : X_train_gb_sc,
            'X_test'    : X_test_gb_sc,
            'warm'      : False,
        },
    }

TARGETS_CFG = {
    'temp': {
        'label'  : 'Temperature T+6h (°C)',
        'unit'   : '°C',
        'std'    : STD_TEMP,
        'y_train': y_train_temp,
        'y_test' : y_test_temp,
        'clip_0' : False,
    },
    'wind': {
        'label'  : 'Vent T+6h (m/s)',
        'unit'   : 'm/s',
        'std'    : STD_WIND,
        'y_train': y_train_wind,
        'y_test' : y_test_wind,
        'clip_0' : True,
    },
    'rain': {
        'label'  : 'Pluie T+6h (mm)',
        'unit'   : 'mm',
        'std'    : STD_RAIN,
        'y_train': y_train_rain,
        'y_test' : y_test_rain,
        'clip_0' : True,
    },
}

# MAE storage
maes       = {nom: {t: [] for t in TARGETS_CFG} for nom in ['Ridge','GBR','RF','KNN']}
is_trained = {t: {nom: False for nom in ['Ridge','GBR','RF','KNN']} for t in TARGETS_CFG}
all_models = {t: make_models() for t in TARGETS_CFG}


# ═══════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════════

print("\n" + "#" * 60)
print("  BOUCLE DE TEST — 5 ITERATIONS")
print("  Entrainement : 2020-2024  |  Test : 2025-2026")
print("#" * 60)

for loop in range(1, N_LOOPS + 1):

    t_loop = time.time()
    print(f"\n{'=' * 60}")
    print(f"  >>> ITERATION {loop} / {N_LOOPS}")
    print(f"{'=' * 60}")

    for target, tcfg in TARGETS_CFG.items():

        print(f"\n  ── TARGET : {target.upper()} ──────────────────────────")

        for nom, mcfg in all_models[target].items():

            model  = mcfg['model']
            X_tr   = mcfg['X_train']
            X_te   = mcfg['X_test']
            y_tr   = tcfg['y_train']
            y_te   = tcfg['y_test']

            print(f"\n  +-- [{nom}] {'─' * (50 - len(nom))}+")

            if not is_trained[target][nom]:
                print(f"  |  Iteration 1 -> ENTRAINEMENT {nom} [{target}]...")
                t0 = time.time()
                model.fit(X_tr, y_tr)
                print(f"  |  [OK] {nom} [{target}] entraine en {time.time()-t0:.1f}s")
            else:
                if mcfg['warm'] and hasattr(model, 'n_estimators'):
                    n_old = model.n_estimators
                    n_new = n_old + mcfg.get('add_trees', 50)
                    model.set_params(n_estimators=n_new)
                    print(f"  |  Iteration {loop} -> warm_start {nom} [{target}] "
                          f"({n_old} -> {n_new} arbres)...")
                    t0 = time.time()
                    model.fit(X_tr, y_tr)
                    print(f"  |  [OK] {time.time()-t0:.1f}s")
                else:
                    print(f"  |  Iteration {loop} -> re-entrainement {nom} [{target}]...")
                    t0 = time.time()
                    model.fit(X_tr, y_tr)
                    print(f"  |  [OK] {time.time()-t0:.1f}s")

            # Prediction
            pred = model.predict(X_te)
            if tcfg['clip_0']:
                pred = np.clip(pred, 0, None)

            mae = float(np.mean(np.abs(pred - y_te)))
            maes[nom][target].append(mae)

            for yr in [2025, 2026]:
                mask = test['timestamp'].dt.year.values == yr
                if mask.sum() > 0:
                    m = np.mean(np.abs(pred[mask] - y_te[mask]))
                    print(f"  |     {yr} -> MAE {nom} [{target}] = {m:.4f} {tcfg['unit']} "
                          f"| score = {-m/tcfg['std']:.4f}")

            print(f"  |  [RESULTAT] MAE {nom} [{target}] = {mae:.4f} {tcfg['unit']} "
                  f"| score = {-mae/tcfg['std']:.4f}")
            print(f"  +{'─' * 54}+")

            is_trained[target][nom] = True

    # Resume
    t_iter = time.time() - t_loop
    print(f"\n  >>> RESUME ITERATION {loop} :")
    print(f"  {'Modele':<8} | {'Temp (°C)':^12} | {'Vent (m/s)':^12} | {'Pluie (mm)':^12}")
    print(f"  {'-' * 54}")
    for nom in ['Ridge', 'GBR', 'RF', 'KNN']:
        mt = maes[nom]['temp'][-1]
        mw = maes[nom]['wind'][-1]
        mr = maes[nom]['rain'][-1]
        print(f"  {nom:<8} | {mt:^12.4f} | {mw:^12.4f} | {mr:^12.4f}")
    print(f"  Temps iteration : {t_iter:.1f}s")


# ═══════════════════════════════════════════════════════
# TABLEAUX RECAPITULATIFS
# ═══════════════════════════════════════════════════════

print(f"\n\n{'#' * 70}")
print(f"  RECAPITULATIF FINAL — MAE sur TEST 2025-2026")
print(f"{'#' * 70}")

for target, tcfg in TARGETS_CFG.items():
    print(f"\n  [{target.upper()}] {tcfg['label']} — std={tcfg['std']}")
    header = f"  {'Test':^5} | " + " | ".join([f"{n:^10}" for n in ['Ridge','GBR','RF','KNN']]) + " | {'Meilleur':^10}"
    print(header)
    print(f"  {'-' * 60}")
    for i in range(N_LOOPS):
        vals   = {n: maes[n][target][i] for n in ['Ridge','GBR','RF','KNN']}
        winner = min(vals, key=vals.get)
        row    = " | ".join([f"{v:^10.4f}" for v in vals.values()])
        print(f"  {i+1:^5} | {row} | {winner:^10}")
    print(f"  {'-' * 60}")
    moys = {n: np.mean(maes[n][target]) for n in ['Ridge','GBR','RF','KNN']}
    row  = " | ".join([f"{v:^10.4f}" for v in moys.values()])
    print(f"  {'Moy.':^5} | {row} |")
    winner_global = min(moys, key=moys.get)
    print(f"  => Meilleur modele overall : {winner_global} (moy. MAE = {moys[winner_global]:.4f} {tcfg['unit']})")

print(f"\n  Temps total : {time.time()-t_global:.1f}s\n")


# ═══════════════════════════════════════════════════════
# 3 GRAPHES
# ═══════════════════════════════════════════════════════

print("-> Generation des 3 graphes MAE (temperature, vent, pluie)...")

COLORS = {
    'Ridge': '#00b4d8',
    'GBR':   '#f77f00',
    'RF':    '#7bc67e',
    'KNN':   '#e040fb',
}
MARKERS = {
    'Ridge': 'o',
    'GBR':   's',
    'RF':    '^',
    'KNN':   'D',
}
TITLES = {
    'temp': 'Evolution MAE Temperature T+6h — Paris\nEntrainement : 2020-2024   |   Evaluation : 2025-2026',
    'wind': 'Evolution MAE Vent T+6h — Paris\nEntrainement : 2020-2024   |   Evaluation : 2025-2026',
    'rain': 'Evolution MAE Pluie T+6h — Paris\nEntrainement : 2020-2024   |   Evaluation : 2025-2026',
}
YLABELS = {
    'temp': 'MAE Temperature |°C|',
    'wind': 'MAE Vent |m/s|',
    'rain': 'MAE Pluie |mm|',
}

iterations = list(range(1, N_LOOPS + 1))

for target, tcfg in TARGETS_CFG.items():

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#161b22')

    for nom in ['Ridge', 'GBR', 'RF', 'KNN']:
        vals  = maes[nom][target]
        color = COLORS[nom]
        moy   = np.mean(vals)

        ax.fill_between(iterations, vals, alpha=0.08, color=color)
        ax.plot(iterations, vals,
                marker=MARKERS[nom], markersize=10, linewidth=2.5,
                color=color, label=f'{nom:<6}  (moy. = {moy:.4f} {tcfg["unit"]})',
                zorder=5)
        ax.axhline(moy, color=color, linestyle='--', linewidth=1.1, alpha=0.40)

        offsets = {'Ridge': 12, 'GBR': -20, 'RF': 12, 'KNN': -20}
        for i, v in enumerate(vals, start=1):
            ax.annotate(f'{v:.3f}',
                        xy=(i, v), xytext=(0, offsets[nom]),
                        textcoords='offset points',
                        ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('Numero du test (iteration)', fontsize=12,
                  color='#e0e0e0', labelpad=10)
    ax.set_ylabel(YLABELS[target], fontsize=12, color='#e0e0e0', labelpad=10)
    ax.set_title(TITLES[target], fontsize=14, fontweight='bold',
                 color='#ffffff', pad=16)

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

    all_vals = [v for nom in ['Ridge','GBR','RF','KNN'] for v in maes[nom][target]]
    ax.set_ylim(max(0, min(all_vals) - 0.3), max(all_vals) + 0.5)
    ax.set_xlim(0.7, N_LOOPS + 0.3)

    plt.tight_layout(pad=1.5)

    output_plot = os.path.join(DATA_DIR, f'mae_evolution_{target}.png')
    plt.savefig(output_plot, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  [OK] Graphe sauvegarde -> {output_plot}")
    plt.show()

print(f"\nSCRIPT TERMINE.")
print(f"  > mae_evolution_temp.png")
print(f"  > mae_evolution_wind.png")
print(f"  > mae_evolution_rain.png")
print(f"\n  Temps total : {time.time()-t_global:.1f}s")
