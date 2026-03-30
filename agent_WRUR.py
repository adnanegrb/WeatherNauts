"""
Weather Prediction Agent - WeatherNauts
Predict temperature, wind_speed, and rain for Paris at T+6h.
Input : X_test shape (20, 24, 8) - 20 cities x 24 hours x 8 features
Output: predictions shape (3,) - [temperature (°C), wind_speed (m/s), rain (mm)]

Features indices: temperature(0), rain(1), wind_speed(2), wind_direction(3),
                  humidity(4), clouds(5), visibility(6), snow(7)
"""
import numpy as np
import pandas as pd
import joblib
import os

PARIS_IDX = 16

# Ordre alphabétique des villes (comme ML Arena)
CITIES = [
    "Amsterdam", "Barcelona", "Birmingham", "Brussels", "Copenhagen",
    "Dortmund", "Dublin", "Düsseldorf", "Essen", "Frankfurt am Main",
    "Köln", "London", "Manchester", "Marseille", "Milan", "Munich",
    "Paris", "Rotterdam", "Stuttgart", "Turin"
]

# Voisins pondérés par distance
NEIGHBORS = {
    'Brussels': 264, 'London': 344, 'Rotterdam': 373,
    'Köln': 403, 'Düsseldorf': 411, 'Amsterdam': 431,
    'Essen': 440, 'Dortmund': 469, 'Frankfurt am Main': 479,
}
_sigma  = float(np.median(list(NEIGHBORS.values())))
WEIGHTS = {c: 1.0 / (1.0 + d / _sigma) for c, d in NEIGHBORS.items()}

# Index des voisins dans la liste CITIES
NEIGHBOR_IDX = {city: CITIES.index(city) for city in NEIGHBORS}


def _extract_features(X_test):
    """
    Extrait les features depuis X_test (20, 24, 8)
    Même logique que build_features() dans train.py
    X_test[:, :, :] → [city, hour, feature]
    Features: temp(0), rain(1), wind(2), dir(3), hum(4), clouds(5), vis(6), snow(7)
    """
    paris = X_test[PARIS_IDX]  # shape (24, 8)

    # Dernière heure disponible (t=0)
    last = paris[-1]

    feats = []

    # Valeurs actuelles Paris
    feats += [
        last[0],  # temperature
        last[2],  # wind_speed
        last[1],  # rain
        last[4],  # humidity
        last[5],  # clouds
        last[7],  # snow
    ]

    # Sin/Cos heure (dernière heure = heure 23 de la fenêtre)
    # On ne connaît pas l'heure exacte mais on utilise la position dans la fenêtre
    for h in range(24):
        pass  # calculé via la fenêtre
    hour_idx = 23  # dernière heure de la fenêtre
    feats += [
        np.sin(2 * np.pi * hour_idx / 24),
        np.cos(2 * np.pi * hour_idx / 24),
        np.sin(2 * np.pi * 1 / 12),   # mois inconnu → valeur neutre
        np.cos(2 * np.pi * 1 / 12),
    ]

    # Wind U,V decomposition
    wind_u = last[2] * np.sin(np.deg2rad(last[3]))
    wind_v = last[2] * np.cos(np.deg2rad(last[3]))
    feats += [wind_u, wind_v]

    # Dew point depression
    T  = float(last[0])
    RH = float(np.clip(last[4], 1, 100))
    gamma = (17.27 * T) / (237.7 + T) + np.log(RH / 100)
    dew_depression = T - (237.7 * gamma) / (17.27 - gamma)
    feats.append(dew_depression)

    # Lags température (1,3,6,12,24h)
    for h in [1, 3, 6, 12, 24]:
        feats.append(float(paris[max(0, 23 - h), 0]))

    # Lags vent (1,3,6,12,24h)
    for h in [1, 3, 6, 12, 24]:
        feats.append(float(paris[max(0, 23 - h), 2]))

    # Lags pluie (1,3,6,12h)
    for h in [1, 3, 6, 12]:
        feats.append(float(paris[max(0, 23 - h), 1]))

    # Lags humidité (6,12h)
    for h in [6, 12]:
        feats.append(float(paris[max(0, 23 - h), 4]))

    # Lags direction (1,6,12h) + U,V
    for h in [1, 6, 12]:
        d = float(paris[max(0, 23 - h), 3])
        w = float(paris[max(0, 23 - h), 2])
        feats.append(d)
        feats.append(w * np.sin(np.deg2rad(d)))  # lag_wind_u
        feats.append(w * np.cos(np.deg2rad(d)))  # lag_wind_v

    # Deltas température (1,3,6,12h)
    for h in [1, 3, 6, 12]:
        feats.append(float(last[0] - paris[max(0, 23 - h), 0]))

    # Deltas vent (1,3,6,12h)
    for h in [1, 3, 6, 12]:
        feats.append(float(last[2] - paris[max(0, 23 - h), 2]))

    # Deltas direction (1,3,6,12h)
    for h in [1, 3, 6, 12]:
        feats.append(float(last[3] - paris[max(0, 23 - h), 3]))

    # Deltas pluie (1,3,6h)
    for h in [1, 3, 6]:
        feats.append(float(last[1] - paris[max(0, 23 - h), 1]))

    # Stats fenêtres température (3,6,12,24h × mean,std,min,max)
    for w in [3, 6, 12, 24]:
        window = paris[-w:, 0]
        feats += [float(window.mean()), float(window.std()),
                  float(window.min()),  float(window.max())]

    # Stats fenêtres vent (3,6,12,24h × mean,std,min,max)
    for w in [3, 6, 12, 24]:
        window = paris[-w:, 2]
        feats += [float(window.mean()), float(window.std()),
                  float(window.min()),  float(window.max())]

    # Stats pluie mean+sum (3,6,12h)
    for w in [3, 6, 12]:
        window = paris[-w:, 1]
        feats += [float(window.mean()), float(window.sum())]

    # Stats humidité mean (6,12h)
    for w in [6, 12]:
        feats.append(float(paris[-w:, 4].mean()))

    # Stats nuages mean (6,12h)
    for w in [6, 12]:
        feats.append(float(paris[-w:, 5].mean()))

    # Pentes température (3,6,12h)
    for w in [3, 6, 12]:
        feats.append(float((last[0] - paris[max(0, 23 - w), 0]) / w))

    # Pentes vent (3,6,12h)
    for w in [3, 6, 12]:
        feats.append(float((last[2] - paris[max(0, 23 - w), 2]) / w))

    # Voisins pondérés (9 villes × 9 features)
    for city, dist in NEIGHBORS.items():
        w      = WEIGHTS[city]
        c_idx  = NEIGHBOR_IDX[city]
        c_last = X_test[c_idx, -1, :]  # dernière heure du voisin

        c_temp = float(c_last[0])
        c_wind = float(c_last[2])
        c_rain = float(c_last[1])
        c_dir  = float(c_last[3])
        c_lag6 = X_test[c_idx, max(0, 23 - 6), :]

        feats += [
            c_temp * w,                                          # temp_w
            c_wind * w,                                          # wind_w
            c_rain * w,                                          # rain_w
            (float(last[0]) - c_temp) * w,                      # grad_temp
            (float(last[2]) - c_wind) * w,                      # grad_wind
            float(c_lag6[0]) * w,                               # temp_lag6
            float(c_lag6[2]) * w,                               # wind_lag6
            c_wind * np.sin(np.deg2rad(c_dir)) * w,             # wind_u
            c_wind * np.cos(np.deg2rad(c_dir)) * w,             # wind_v
        ]

    return np.array(feats, dtype=np.float32)


class Agent:
    def __init__(self):
        # Charger les modèles depuis le même dossier que agent.py
        base = os.path.dirname(os.path.abspath(__file__))
        self.gbr_temp  = joblib.load(os.path.join(base, 'model_temp.pkl'))
        self.gbr_wind  = joblib.load(os.path.join(base, 'model_wind.pkl'))
        self.knn_rain  = joblib.load(os.path.join(base, 'model_rain.pkl'))
        self.scaler    = joblib.load(os.path.join(base, 'scaler.pkl'))
        self.SEUIL     = 1.0   # seuil optimal pluie

    def predict(self, X_test):
        """
        Predict Paris weather at T+6h.

        Args:
            X_test: shape (20, 24, 8)

        Returns:
            predictions: shape (3,) - [temperature, wind_speed, rain]
        """
        X_test = np.array(X_test, dtype=np.float32)

        # Extraire les features
        feats = _extract_features(X_test).reshape(1, -1)

        # Valeurs actuelles Paris (pour reconstruire depuis delta)
        temp_t0 = float(X_test[PARIS_IDX, -1, 0])
        wind_t0 = float(X_test[PARIS_IDX, -1, 2])

        # Température → delta + reconstruction
        delta_temp = float(self.gbr_temp.predict(feats)[0])
        pred_temp  = temp_t0 + delta_temp

        # Vent → delta + reconstruction
        delta_wind = float(self.gbr_wind.predict(feats)[0])
        pred_wind  = max(0.0, wind_t0 + delta_wind)

        # Pluie → KNN + seuil
        feats_sc  = self.scaler.transform(feats)
        pred_rain = float(self.knn_rain.predict(feats_sc)[0])
        pred_rain = max(0.0, pred_rain)
        pred_rain = 0.0 if pred_rain < self.SEUIL else pred_rain

        return np.array([pred_temp, pred_wind, pred_rain], dtype=np.float32)
