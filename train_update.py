import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier
)
import joblib

# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────

CITIES = [
    "Amsterdam", "Barcelona", "Birmingham", "Brussels",
    "Copenhagen", "Dortmund", "Dublin", "Düsseldorf",
    "Essen", "Frankfurt am Main", "Köln", "London",
    "Manchester", "Marseille", "Milan", "Munich",
    "Paris", "Rotterdam", "Stuttgart", "Turin"
]
PARIS_IDX = 16

# Colonnes CSV → index dans X (20, 24, 8)
# visibility manquante → on met 0.0 à l'index 6
FEATURES_CSV = [
    'temperature',     # index 0
    'rain',            # index 1
    'wind_speed',      # index 2
    'wind_direction',  # index 3
    'humidity',        # index 4
    'clouds',          # index 5
    'snow'             # index 7 (visibility=0 à index 6)
]

STD_TEMP = 7.49
STD_WIND = 5.05
STD_RAIN = 0.40

PARIS_IDX = 16
DISTANCES_PARIS = [431, 830, 502, 264, 1027, 469, 781, 411, 440, 479,
                   403, 344, 570, 661, 640, 685, 0, 373, 500, 580]

_dist     = np.array(DISTANCES_PARIS, dtype=np.float32)
_sigma    = float(np.median(_dist[_dist > 0]))
POIDS_GEO = (1.0 / (1.0 + _dist / _sigma)).astype(np.float32)

VOISINS_TRIES = sorted(
    [i for i in range(20) if i != PARIS_IDX],
    key=lambda i: DISTANCES_PARIS[i]
)

_heures = np.arange(24, dtype=np.float32)
_CYCL   = np.stack([
    np.sin(2 * np.pi * _heures / 24),
    np.cos(2 * np.pi * _heures / 24)
], axis=1).flatten()


# ─────────────────────────────────────────
# FONCTIONS DE FEATURES
# ─────────────────────────────────────────

def _lags(serie):
    last = serie[:, -1]
    return [last - serie[:, max(0, 23 - d)]
            for d in [1, 2, 3, 6, 12, 18]]

def _stats_fenetres(serie, fenetres=(6, 12, 24)):
    blocs = []
    for f in fenetres:
        w = serie[:, -f:]
        blocs += [w.mean(axis=1), w.std(axis=1),
                  w.min(axis=1),  w.max(axis=1)]
    return blocs

def _pentes(serie, fenetres=(6, 12, 24)):
    blocs = []
    for f in fenetres:
        w     = serie[:, -f:]
        x     = np.arange(f, dtype=np.float32)
        x     = x - x.mean()
        denom = float((x ** 2).sum())
        blocs.append((w * x[np.newaxis, :]).sum(axis=1) / denom)
    return blocs

def features_ridge(tableau):
    # Features pour Ridge → température
    seul = (tableau.ndim == 3)
    if seul:
        tableau = tableau[np.newaxis]

    nb     = tableau.shape[0]
    paris  = tableau[:, PARIS_IDX]
    p_last = paris[:, -1]

    blocs = [
        paris[:, :, 0],                           # temp 24h
        paris[:, :, 2],                           # vent 24h
        paris[:, :, 4],                           # humidité 24h
        paris[:, :, 1],                           # pluie 24h
        paris[:, :, 5],                           # nuages 24h
        p_last,                                   # dernière heure
        paris.mean(axis=1)[:, [0, 1, 2, 4, 5]],  # moyennes
        paris.std(axis=1)[:, [0, 2]],             # volatilité
        np.tile(_CYCL, (nb, 1)),                  # cycle 24h
    ]

    blocs += _lags(paris[:, :, 0])                # lags température
    blocs += _stats_fenetres(paris[:, :, 0])      # stats temp
    blocs += _pentes(paris[:, :, 0])              # pentes temp

    # 5 villes les plus proches pondérées
    for idx_ville in VOISINS_TRIES[:5]:
        v     = tableau[:, idx_ville]
        poids = float(POIDS_GEO[idx_ville])
        blocs.append(v[:, :, 0] * poids)
        blocs.append(v[:, -1:, [0, 2]] * poids)
        blocs.append((p_last[:, [0]] - v[:, -1, [0]]) * poids)

    sortie = np.concatenate([b.reshape(nb, -1) for b in blocs],
                             axis=1).astype(np.float32)
    return sortie[0] if seul else sortie

def features_gb(tableau):
    # Features pour GradientBoosting → vent + pluie
    seul = (tableau.ndim == 3)
    if seul:
        tableau = tableau[np.newaxis]

    nb     = tableau.shape[0]
    paris  = tableau[:, PARIS_IDX]
    p_last = paris[:, -1]

    blocs = [paris.reshape(nb, -1)]               # toutes données Paris

    blocs += _lags(paris[:, :, 0])                # lags température
    blocs += _lags(paris[:, :, 2])                # lags vent

    # Stats pour chaque feature
    for feat_idx in range(8):
        blocs += _stats_fenetres(paris[:, :, feat_idx])

    blocs += _pentes(paris[:, :, 0])              # pentes température
    blocs += _pentes(paris[:, :, 2])              # pentes vent

    blocs += [
        paris.mean(axis=1),                       # moyennes
        paris.std(axis=1),                        # volatilité
        paris.min(axis=1),                        # minimums
        paris.max(axis=1),                        # maximums
        paris[:, -6:].mean(axis=1),               # moyenne 6h
        paris[:, -6:].std(axis=1),                # volatilité 6h
        np.tile(_CYCL, (nb, 1)),                  # cycle 24h
    ]

    # Toutes les villes voisines pondérées
    for idx_ville in VOISINS_TRIES:
        v      = tableau[:, idx_ville]
        v_last = v[:, -1]
        v_6h   = v[:, max(0, 17)]
        poids  = float(POIDS_GEO[idx_ville])
        blocs += [
            v_last * poids,
            (v_last[:, [0, 2, 1]] - v_6h[:, [0, 2, 1]]) * poids,
            v[:, :, 0] * poids,
            v[:, :, 2] * poids,
            (p_last[:, [0, 2, 1]] - v_last[:, [0, 2, 1]]) * poids,
        ]
        if idx_ville in VOISINS_TRIES[:6]:
            blocs += _stats_fenetres(v[:, :, 0], fenetres=(6, 12))
            blocs += _pentes(v[:, :, 0], fenetres=(6,))

    # Moyenne globale pondérée toutes villes
    derniere_h = tableau[:, :, -1, :]
    poids_col  = POIDS_GEO.reshape(1, 20, 1)
    blocs += [
        (derniere_h * poids_col).mean(axis=1),
        derniere_h.std(axis=1),
        (tableau[:, :, -1, :] - tableau[:, :, -7, :]).mean(axis=1),
    ]

    sortie = np.concatenate([b.reshape(nb, -1) for b in blocs],
                             axis=1).astype(np.float32)
    return sortie[0] if seul else sortie


# ─────────────────────────────────────────
# 1. CHARGER ET CONSTRUIRE X (N, 20, 24, 8)
# ─────────────────────────────────────────

print("⏳ Chargement des données clean...")

dfs = []
for year in range(2020, 2027):
    try:
        df_year = pd.read_csv(f"weather_{year}_clean.csv")
        dfs.append(df_year)
        print(f"   ✅ {year} chargé — {len(df_year)} lignes")
    except FileNotFoundError:
        print(f"   ⚠️ {year} non trouvé")

df = pd.concat(dfs, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.sort_values(['timestamp', 'city_name']).reset_index(drop=True)

print(f"✅ Total : {len(df)} lignes\n")

# ─────────────────────────────────────────
# 2. CONSTRUIRE X (N, 20, 24, 8) et y (N, 3)
# ─────────────────────────────────────────

print("⏳ Construction de X et y...")

timestamps = sorted(df['timestamp'].unique())
X_list, y_list, ts_list = [], [], []

for i in range(24, len(timestamps) - 6):

    window_ts = timestamps[i-24:i]   # 24h d'historique
    target_ts = timestamps[i+5]      # T+6h

    # Construire X (20, 24, 8)
    X = np.zeros((20, 24, 8), dtype=np.float32)

    ok = True
    for j, city in enumerate(CITIES):
        city_data = df[df['city_name'] == city]
        for k, ts in enumerate(window_ts):
            row = city_data[city_data['timestamp'] == ts]
            if len(row) == 0:
                ok = False
                break
            # Remplir les 8 features
            # visibility = 0.0 à index 6
            X[j, k, 0] = row['temperature'].values[0]
            X[j, k, 1] = row['rain'].values[0]
            X[j, k, 2] = row['wind_speed'].values[0]
            X[j, k, 3] = row['wind_direction'].values[0]
            X[j, k, 4] = row['humidity'].values[0]
            X[j, k, 5] = row['clouds'].values[0]
            X[j, k, 6] = 0.0   # visibility manquante
            X[j, k, 7] = row['snow'].values[0]
        if not ok:
            break

    if not ok:
        continue

    # Target → Paris T+6h
    paris_target = df[
        (df['city_name'] == 'Paris') &
        (df['timestamp'] == target_ts)
    ]
    if len(paris_target) == 0:
        continue

    y = np.array([
        paris_target['temperature'].values[0],
        paris_target['wind_speed'].values[0],
        paris_target['rain'].values[0]
    ], dtype=np.float32)

    X_list.append(X)
    y_list.append(y)
    ts_list.append(timestamps[i])

X_all = np.array(X_list)   # (N, 20, 24, 8)
y_all = np.array(y_list)   # (N, 3)
ts_all = np.array(ts_list)

print(f"✅ X shape : {X_all.shape}")
print(f"✅ y shape : {y_all.shape}\n")

# ─────────────────────────────────────────
# 3. SPLIT TRAIN / TEST
# ─────────────────────────────────────────

print("⏳ Split Train / Test...")

years_all = pd.DatetimeIndex(ts_all).year

train_mask = years_all <= 2024
test_mask  = years_all >= 2025

X_train = X_all[train_mask]
y_train = y_all[train_mask]
X_test  = X_all[test_mask]
y_test  = y_all[test_mask]

print(f"   ✅ Train : {len(X_train)} samples (2020-2024)")
print(f"   ✅ Test  : {len(X_test)} samples (2025-2026)\n")

# ─────────────────────────────────────────
# 4. EXTRAIRE LES FEATURES
# ─────────────────────────────────────────

print("⏳ Extraction des features...")

f_ridge_train = features_ridge(X_train)
f_gb_train    = features_gb(X_train)
f_ridge_test  = features_ridge(X_test)
f_gb_test     = features_gb(X_test)

print(f"   ✅ features_ridge shape : {f_ridge_train.shape}")
print(f"   ✅ features_gb shape    : {f_gb_train.shape}\n")

# ─────────────────────────────────────────
# 5. SCALING → Ridge seulement
# ─────────────────────────────────────────

print("⏳ Scaling...")

scaler          = StandardScaler()
f_ridge_scaled  = scaler.fit_transform(f_ridge_train)
f_ridge_test_sc = scaler.transform(f_ridge_test)

joblib.dump(scaler, 'scaler.pkl')
print("   ✅ scaler.pkl sauvegardé\n")

# ─────────────────────────────────────────
# 6. MODÈLE TEMPÉRATURE → Ridge
# ─────────────────────────────────────────

print("⏳ Entraînement Ridge (température)...")

model_temp = Ridge(alpha=10.0)
model_temp.fit(f_ridge_scaled, y_train[:, 0])

joblib.dump(model_temp, 'model_temp.pkl')
print("   ✅ model_temp.pkl sauvegardé\n")

# ─────────────────────────────────────────
# 7. MODÈLE VENT → GradientBoosting
# ─────────────────────────────────────────

print("⏳ Entraînement GradientBoosting (vent)...")

model_wind = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)
model_wind.fit(f_gb_train, y_train[:, 1])

joblib.dump(model_wind, 'model_wind.pkl')
print("   ✅ model_wind.pkl sauvegardé\n")

# ─────────────────────────────────────────
# 8. MODÈLE PLUIE → Classifieur + Régresseur
# ─────────────────────────────────────────

print("⏳ Entraînement GradientBoosting (pluie)...")

y_train_rain     = y_train[:, 2]
y_train_rain_clf = (y_train_rain > 0.1).astype(int)

# Classifieur → va-t-il pleuvoir ?
model_rain_clf = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
model_rain_clf.fit(f_gb_train, y_train_rain_clf)

joblib.dump(model_rain_clf, 'model_rain_clf.pkl')
print("   ✅ model_rain_clf.pkl sauvegardé")

# Régresseur → combien de mm ?
rain_mask      = y_train_rain > 0.1
model_rain_reg = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
model_rain_reg.fit(f_gb_train[rain_mask], y_train_rain[rain_mask])

joblib.dump(model_rain_reg, 'model_rain_reg.pkl')
print("   ✅ model_rain_reg.pkl sauvegardé\n")

# ─────────────────────────────────────────
# 9. ÉVALUATION SUR TEST
# ─────────────────────────────────────────

print("⏳ Évaluation sur test (2025-2026)...")

# Prédictions
pred_temp  = model_temp.predict(f_ridge_test_sc)
pred_wind  = model_wind.predict(f_gb_test)

rain_proba = model_rain_clf.predict_proba(f_gb_test)[:, 1]
pred_rain  = np.where(
    rain_proba > 0.5,
    model_rain_reg.predict(f_gb_test),
    0.0
)

# MAE
mae_temp = np.mean(np.abs(pred_temp - y_test[:, 0]))
mae_wind = np.mean(np.abs(pred_wind - y_test[:, 1]))
mae_rain = np.mean(np.abs(pred_rain - y_test[:, 2]))

# Score final
score = -np.mean([
    mae_temp / STD_TEMP,
    mae_wind / STD_WIND,
    mae_rain / STD_RAIN
])

print(f"\n{'='*40}")
print(f"RÉSULTATS TEST (2025-2026)")
print(f"{'='*40}")
print(f"MAE température : {mae_temp:.2f}°C")
print(f"MAE vent        : {mae_wind:.2f} m/s")
print(f"MAE pluie       : {mae_rain:.2f} mm")
print(f"\nScore final  : {score:.4f}")
print(f"Baseline     : -0.3000")
print(f"{'='*40}")

# Détail par année
print(f"\nDÉTAIL PAR ANNÉE :")
print(f"{'='*40}")
years_test = pd.DatetimeIndex(ts_all[test_mask]).year
for year in [2025, 2026]:
    mask  = years_test == year
    if mask.sum() == 0:
        continue
    mae_t = np.mean(np.abs(pred_temp[mask] - y_test[mask, 0]))
    mae_w = np.mean(np.abs(pred_wind[mask] - y_test[mask, 1]))
    mae_r = np.mean(np.abs(pred_rain[mask] - y_test[mask, 2]))
    s     = -np.mean([mae_t/STD_TEMP, mae_w/STD_WIND, mae_r/STD_RAIN])
    print(f"{year} → temp:{mae_t:.2f}°C | vent:{mae_w:.2f}m/s | pluie:{mae_r:.2f}mm | score:{s:.4f}")

print(f"\n🏆 Modèles sauvegardés → prêts pour agent.py !")
