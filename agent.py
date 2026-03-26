"""
Weather Prediction Agent
Predict temperature, wind_speed, and rain for Paris at T+6h.
Input:  X_test of shape (20, 24, 8) - 20 cities x 24 hours x 8 features
Output: predictions of shape (3,)   - [temperature (°C), wind_speed (m/s), rain (mm)]

Features (indices):
    temperature(0), rain(1), wind_speed(2), wind_direction(3),
    humidity(4), clouds(5), visibility(6), snow(7)

Cities sorted alphabetically:
    Amsterdam(0), Barcelona(1), Birmingham(2), Brussels(3),
    Copenhagen(4), Dortmund(5), Dublin(6), Düsseldorf(7),
    Essen(8), Frankfurt am Main(9), Köln(10), London(11),
    Manchester(12), Marseille(13), Milan(14), Munich(15),
    Paris(16), Rotterdam(17), Stuttgart(18), Turin(19)

Metric: negated normalized MAE
    each error divided by its std (temp:7.49, wind:5.05, rain:0.40)
    Higher score = better.
"""

import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from xgboost import XGBRegressor

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

PARIS_IDX = 16

# Coordonnées GPS des 20 villes (lat, lon)
CITY_COORDS = {
    0:  ("Amsterdam",          52.3676,  4.9041),
    1:  ("Barcelona",          41.3851,  2.1734),
    2:  ("Birmingham",         52.4862, -1.8904),
    3:  ("Brussels",           50.8503,  4.3517),
    4:  ("Copenhagen",         55.6761, 12.5683),
    5:  ("Dortmund",           51.5136,  7.4653),
    6:  ("Dublin",             53.3498, -6.2603),
    7:  ("Düsseldorf",         51.2217,  6.7762),
    8:  ("Essen",              51.4556,  7.0116),
    9:  ("Frankfurt am Main",  50.1109,  8.6821),
    10: ("Köln",               50.9333,  6.9500),
    11: ("London",             51.5074, -0.1278),
    12: ("Manchester",         53.4808, -2.2426),
    13: ("Marseille",          43.2965,  5.3698),
    14: ("Milan",              45.4654,  9.1859),
    15: ("Munich",             48.1351, 11.5820),
    16: ("Paris",              48.8566,  2.3522),
    17: ("Rotterdam",          51.9244,  4.4777),
    18: ("Stuttgart",          48.7758,  9.1829),
    19: ("Turin",              45.0703,  7.6869),
}

PARIS_LAT = 48.8566
PARIS_LON = 2.3522

# Normalization stds donnés par ML Arena
TARGET_STDS = np.array([7.49, 5.05, 0.40])  # temp, wind_speed, rain

# Features dans X_test
FEAT_TEMP       = 0
FEAT_RAIN       = 1
FEAT_WIND       = 2
FEAT_WIND_DIR   = 3
FEAT_HUMIDITY   = 4
FEAT_CLOUDS     = 5
FEAT_VISIBILITY = 6
FEAT_SNOW       = 7


# ══════════════════════════════════════════════════════════════════════════════
# PONDÉRATION GÉOGRAPHIQUE
# ══════════════════════════════════════════════════════════════════════════════

def compute_geo_weights():
    """
    Calcule les poids géographiques de chaque ville par rapport à Paris.

    Logique :
    - Plus une ville est au NORD-OUEST de Paris, plus elle est utile
      pour prédire la météo parisienne (flux atlantiques dominants).
    - La pondération varie de façon CONTINUE selon la distance et la direction.
    - Les villes à l'est n'ont ni bonus ni malus (poids = 1.0 baseline).

    Formule :
        delta_lat = lat_ville - lat_paris  (>0 = nord)
        delta_lon = lon_paris - lon_ville  (>0 = ouest)
        composante_NW = (delta_lat + delta_lon) / 2
        poids = 1 + tanh(composante_NW / sigma) * amplitude
    """
    weights = {}
    sigma     = 5.0   # échelle de distance (degrés) pour la saturation
    amplitude = 2.0   # bonus maximum pour la ville la plus au NW

    for idx, (name, lat, lon) in CITY_COORDS.items():
        delta_lat = lat  - PARIS_LAT   # positif = nord
        delta_lon = PARIS_LON - lon    # positif = ouest

        # Composante nord-ouest projetée (moyenne des deux axes)
        nw_component = (delta_lat + delta_lon) / 2.0

        # Fonction tanh : continue, bornée, centrée en 0
        weight = 1.0 + np.tanh(nw_component / sigma) * amplitude

        # Paris elle-même : poids maximal car données locales
        if idx == PARIS_IDX:
            weight = 1.0 + amplitude  # bonus maximum

        weights[idx] = float(np.clip(weight, 0.1, 1.0 + amplitude))

    return weights


# ══════════════════════════════════════════════════════════════════════════════
# PONDÉRATION TEMPORELLE
# ══════════════════════════════════════════════════════════════════════════════

def compute_temporal_weight(record_month: int, current_month: int) -> float:
    """
    Calcule le poids temporel d'une observation en fonction de l'écart
    en mois entre sa date et la date actuelle (toutes années confondues).

    Règle :
        écart 0 mois  → +7
        écart 1 mois  → +6
        écart 2 mois  → +5
        écart 3 mois  → +4
        écart 4 mois  → +3
        écart 5 mois  → +2
        écart 6 mois  → +1
        écart > 6 mois → +0  (poids de base = 1)
    """
    # Distance cyclique sur 12 mois
    diff = abs(record_month - current_month)
    diff = min(diff, 12 - diff)

    bonus = max(0, 7 - diff)
    return 1.0 + bonus


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT ET PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_training_data():
    """
    Charge les fichiers weather_{year}_clean.csv depuis le même dossier
    que agent.py (tel que déployé sur ML Arena).

    Structure attendue des CSV :
        timestamp, city_name, country_code, latitude, longitude,
        temperature, rain, wind_speed, wind_direction, humidity,
        clouds, visibility, snow, hour, month
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    years    = range(2020, 2027)
    frames   = []

    for year in years:
        path = os.path.join(base_dir, f"weather_{year}_clean.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])
        frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["city_name", "timestamp"]).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DES FEATURES D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def build_features_from_df(df: pd.DataFrame,
                            geo_weights: dict,
                            current_month: int):
    """
    Transforme le DataFrame historique en matrices (X_train, y_train, sample_weights).

    Pour chaque fenêtre de 24h consécutives par ville → on prédit la valeur T+6h Paris.
    """

    city_name_to_idx = {v[0]: k for k, v in CITY_COORDS.items()}

    # ── Dictionnaire : city_name → tableau trié par timestamp ──────────────
    city_data = {}
    for city, grp in df.groupby("city_name"):
        city_data[city] = grp.reset_index(drop=True)

    # ── Trouver la longueur minimale commune ───────────────────────────────
    paris_df  = city_data.get("Paris")
    if paris_df is None:
        return None, None, None

    n_paris   = len(paris_df)
    all_cities_sorted = sorted(city_data.keys())   # ordre alphabétique ML Arena

    # ── Construction des exemples ──────────────────────────────────────────
    X_list, y_list, w_list = [], [], []
    window  = 24
    horizon = 6

    # On itère sur les indices de Paris comme référence temporelle
    for t in range(window, n_paris - horizon):
        # ── Features : 20 villes × 24h × 8 features ──────────────────────
        x_window = np.zeros((20, window, 8), dtype=np.float32)

        for city_idx, city_name in enumerate(all_cities_sorted):
            if city_name not in city_data:
                continue
            cdf = city_data[city_name]
            if len(cdf) <= t:
                continue

            # Aligner sur la même plage temporelle que Paris
            slice_df = cdf.iloc[t - window: t]
            if len(slice_df) < window:
                continue

            x_window[city_idx, :, FEAT_TEMP]       = slice_df["temperature"].values
            x_window[city_idx, :, FEAT_RAIN]        = slice_df["rain"].values
            x_window[city_idx, :, FEAT_WIND]        = slice_df["wind_speed"].values
            x_window[city_idx, :, FEAT_WIND_DIR]    = slice_df["wind_direction"].values
            x_window[city_idx, :, FEAT_HUMIDITY]    = slice_df["humidity"].values
            x_window[city_idx, :, FEAT_CLOUDS]      = slice_df["clouds"].values
            x_window[city_idx, :, FEAT_VISIBILITY]  = slice_df["visibility"].values
            x_window[city_idx, :, FEAT_SNOW]        = slice_df["snow"].values

        # ── Cibles : température, vent, pluie de Paris à T+6h ─────────────
        target_row = paris_df.iloc[t + horizon]
        y = np.array([
            target_row["temperature"],
            target_row["wind_speed"],
            target_row["rain"],
        ], dtype=np.float32)

        # ── Poids temporel basé sur le mois de la fenêtre ─────────────────
        record_month = paris_df.iloc[t]["timestamp"].month
        w_temporal   = compute_temporal_weight(record_month, current_month)

        X_list.append(x_window)
        y_list.append(y)
        w_list.append(w_temporal)

    if not X_list:
        return None, None, None

    X_raw = np.array(X_list)   # (N, 20, 24, 8)
    y_raw = np.array(y_list)   # (N, 3)
    w_raw = np.array(w_list)   # (N,)

    return X_raw, y_raw, w_raw


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION DE FEATURES POUR XGBOOST (flatten + features avancées)
# ══════════════════════════════════════════════════════════════════════════════

def extract_xgb_features(X: np.ndarray, geo_weights: dict) -> np.ndarray:
    """
    Transforme X de shape (N, 20, 24, 8) en vecteur de features plat
    pour XGBoost.

    Features extraites :
    ┌─────────────────────────────────────────────────────────────────┐
    │ Pour chaque ville (pondérée géo) × chaque feature :            │
    │   - mean, std, min, max sur les 24h                            │
    │   - dernière valeur (h-1), avant-dernière (h-2)                │
    │   - tendance (pente OLS sur 24h)                               │
    │   - moyenne des 6 dernières heures                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ Features croisées Paris × voisins NW :                         │
    │   - corrélation temp Paris vs London, Brussels, Rotterdam       │
    │   - différence de pression (humidity proxy) Paris vs ouest      │
    └─────────────────────────────────────────────────────────────────┘
    """
    N = X.shape[0]
    feature_vectors = []

    hours = np.arange(24, dtype=np.float32)
    hours_centered = hours - hours.mean()

    for i in range(N):
        row_feats = []

        for city_idx in range(20):
            geo_w = geo_weights.get(city_idx, 1.0)

            for feat_idx in range(8):
                series = X[i, city_idx, :, feat_idx].astype(np.float32)
                w_series = series * geo_w

                mean_v  = np.mean(w_series)
                std_v   = np.std(w_series)
                min_v   = np.min(w_series)
                max_v   = np.max(w_series)
                last_v  = w_series[-1]
                prev_v  = w_series[-2]
                last6_v = np.mean(w_series[-6:])

                # Tendance linéaire (pente OLS)
                if std_v > 1e-8:
                    trend = np.polyfit(hours_centered, w_series, 1)[0]
                else:
                    trend = 0.0

                row_feats.extend([mean_v, std_v, min_v, max_v,
                                   last_v, prev_v, last6_v, trend])

        # ── Features croisées ────────────────────────────────────────────
        # Indices des villes NW influentes : London(11), Brussels(3), Rotterdam(17), Amsterdam(0)
        nw_cities = [0, 3, 11, 17]
        paris_temp = X[i, PARIS_IDX, :, FEAT_TEMP]

        for nw_idx in nw_cities:
            nw_temp = X[i, nw_idx, :, FEAT_TEMP]
            # Corrélation temp NW → Paris
            if np.std(paris_temp) > 1e-8 and np.std(nw_temp) > 1e-8:
                corr = np.corrcoef(paris_temp, nw_temp)[0, 1]
            else:
                corr = 0.0
            # Différence de température NW - Paris (gradient)
            diff = np.mean(nw_temp) - np.mean(paris_temp)
            row_feats.extend([corr, diff])

        # Gradient nord-sud (Copenhagen vs Marseille) pour détecter les masses d'air
        cph_temp  = X[i, 4,  :, FEAT_TEMP]   # Copenhagen
        mars_temp = X[i, 13, :, FEAT_TEMP]   # Marseille
        ns_gradient = np.mean(cph_temp) - np.mean(mars_temp)
        row_feats.append(ns_gradient)

        # Humidité relative Paris dernière heure
        row_feats.append(X[i, PARIS_IDX, -1, FEAT_HUMIDITY])

        # Vitesse vent Paris tendance 6 dernières heures
        paris_wind_trend = np.polyfit(
            np.arange(6, dtype=np.float32),
            X[i, PARIS_IDX, -6:, FEAT_WIND],
            1
        )[0]
        row_feats.append(paris_wind_trend)

        feature_vectors.append(row_feats)

    return np.array(feature_vectors, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION ET ENTRAÎNEMENT DES MODÈLES XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def build_and_train_models(X_feat: np.ndarray,
                            y: np.ndarray,
                            sample_weights: np.ndarray):
    """
    Entraîne 3 modèles XGBoost indépendants (un par cible).
    Hyperparamètres calibrés pour données météo horaires.
    """
    xgb_params = dict(
        n_estimators        = 1200,
        learning_rate       = 0.03,
        max_depth           = 7,
        subsample           = 0.8,
        colsample_bytree    = 0.7,
        colsample_bylevel   = 0.7,
        min_child_weight    = 5,
        reg_alpha           = 0.1,      # L1
        reg_lambda          = 1.5,      # L2
        gamma               = 0.05,
        tree_method         = "hist",   # rapide sur CPU
        random_state        = 42,
        n_jobs              = -1,
    )

    target_names = ["temperature", "wind_speed", "rain"]
    models = {}

    for t_idx, t_name in enumerate(target_names):
        print(f"  Entraînement XGBoost → {t_name} ...", flush=True)
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_feat,
            y[:, t_idx],
            sample_weight = sample_weights,
            verbose       = False,
        )
        models[t_name] = model
        print(f"    ✓ {t_name} — best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE AGENT (interface ML Arena)
# ══════════════════════════════════════════════════════════════════════════════

class Agent:

    def __init__(self):
        self.models      = None   # dict {target_name: XGBRegressor}
        self.geo_weights = compute_geo_weights()
        self._trained    = False

        print("Agent initialisé — calcul des poids géographiques :", flush=True)
        for idx, (name, lat, lon) in CITY_COORDS.items():
            print(f"  {name:<22} lat={lat:+.2f} lon={lon:+.2f}  geo_w={self.geo_weights[idx]:.4f}")

        # Entraînement automatique au premier chargement
        self._auto_train()

    # ──────────────────────────────────────────────────────────────────────────
    def _auto_train(self):
        """
        Charge les CSV, construit les features, entraîne les modèles.
        Appelé une seule fois à l'instanciation.
        """
        print("\n[AUTO-TRAIN] Chargement des données ...", flush=True)
        df = load_training_data()

        if df is None or len(df) == 0:
            print("[AUTO-TRAIN] ⚠️  Aucun fichier CSV trouvé — mode fallback activé.")
            self._trained = False
            return

        print(f"[AUTO-TRAIN] {len(df):,} lignes chargées.", flush=True)

        current_month = datetime.utcnow().month
        print(f"[AUTO-TRAIN] Mois courant : {current_month} — pondération temporelle appliquée.", flush=True)

        print("[AUTO-TRAIN] Construction des features ...", flush=True)
        X_raw, y_raw, w_raw = build_features_from_df(df, self.geo_weights, current_month)

        if X_raw is None:
            print("[AUTO-TRAIN] ⚠️  Impossible de construire les features.")
            self._trained = False
            return

        print(f"[AUTO-TRAIN] {X_raw.shape[0]:,} exemples — extraction XGBoost features ...", flush=True)
        X_feat = extract_xgb_features(X_raw, self.geo_weights)
        print(f"[AUTO-TRAIN] Shape features : {X_feat.shape}", flush=True)

        print("[AUTO-TRAIN] Entraînement des modèles ...", flush=True)
        self.models   = build_and_train_models(X_feat, y_raw, w_raw)
        self._trained = True
        print("[AUTO-TRAIN] ✅  Entraînement terminé.", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit température, vitesse du vent et pluie pour Paris à T+6h.

        Args:
            X_test : np.ndarray de shape (20, 24, 8)
                     20 villes × 24 heures × 8 features

        Returns:
            predictions : np.ndarray de shape (3,)
                          [temperature (°C), wind_speed (m/s), rain (mm)]
        """
        # ── Reshape pour être compatible avec extract_xgb_features ──────────
        X_input = X_test[np.newaxis, :, :, :]   # (1, 20, 24, 8)
        X_feat  = extract_xgb_features(X_input, self.geo_weights)  # (1, n_features)

        if self._trained and self.models is not None:
            # ── Prédiction XGBoost ─────────────────────────────────────────
            pred_temp  = float(self.models["temperature"].predict(X_feat)[0])
            pred_wind  = float(self.models["wind_speed"].predict(X_feat)[0])
            pred_rain  = float(self.models["rain"].predict(X_feat)[0])

            # Contraintes physiques
            pred_wind  = max(0.0, pred_wind)
            pred_rain  = max(0.0, pred_rain)

            predictions = np.array([pred_temp, pred_wind, pred_rain], dtype=np.float64)

        else:
            # ── Fallback : baseline statistique sur Paris ──────────────────
            print("[PREDICT] ⚠️  Modèle non entraîné — utilisation du fallback.", flush=True)
            paris_data  = X_test[PARIS_IDX]   # (24, 8)
            pred_temp   = float(np.mean(paris_data[-6:, FEAT_TEMP]))
            pred_wind   = float(np.mean(paris_data[-6:, FEAT_WIND]))
            pred_rain   = float(np.mean(paris_data[-6:, FEAT_RAIN]))
            pred_wind   = max(0.0, pred_wind)
            pred_rain   = max(0.0, pred_rain)
            predictions = np.array([pred_temp, pred_wind, pred_rain], dtype=np.float64)

        return predictions


# ══════════════════════════════════════════════════════════════════════════════
# TEST LOCAL (ne s'exécute pas sur ML Arena)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TEST LOCAL — agent.py")
    print("=" * 60)

    agent = Agent()

    # Simuler un X_test aléatoire réaliste
    np.random.seed(42)
    X_dummy = np.zeros((20, 24, 8), dtype=np.float32)
    for city_idx in range(20):
        X_dummy[city_idx, :, FEAT_TEMP]       = np.random.normal(12, 5, 24)
        X_dummy[city_idx, :, FEAT_RAIN]        = np.random.exponential(0.3, 24)
        X_dummy[city_idx, :, FEAT_WIND]        = np.abs(np.random.normal(5, 3, 24))
        X_dummy[city_idx, :, FEAT_WIND_DIR]    = np.random.uniform(0, 360, 24)
        X_dummy[city_idx, :, FEAT_HUMIDITY]    = np.random.uniform(50, 95, 24)
        X_dummy[city_idx, :, FEAT_CLOUDS]      = np.random.uniform(0, 100, 24)
        X_dummy[city_idx, :, FEAT_VISIBILITY]  = np.random.uniform(5000, 10000, 24)
        X_dummy[city_idx, :, FEAT_SNOW]        = np.random.exponential(0.01, 24)

    preds = agent.predict(X_dummy)
    print("\n📊 Prédictions Paris T+6h :")
    print(f"  Température  : {preds[0]:.2f} °C")
    print(f"  Vent         : {preds[1]:.2f} m/s")
    print(f"  Pluie        : {preds[2]:.4f} mm")
    print("\nShape sortie :", preds.shape)
    print("=" * 60)
