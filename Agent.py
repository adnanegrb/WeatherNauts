import pickle
import numpy as np

# Index de Paris dans la liste des villes (ordre alphabétique)
PARIS_IDX = 16

# Index des variables dans le tenseur X
F_TEMP, F_RAIN, F_WIND, F_WDIR, F_HUM, F_CLOUD, F_VIS, F_SNOW = range(8)

# Distance (km) de chaque ville à Paris
DIST = np.array([
    431, 830, 502, 264, 1027, 469, 781, 411, 440, 479,
    403, 344, 570, 661, 640, 685,   0, 373, 500, 580,
], dtype=np.float32)

# Villes à moins de 500km de Paris (pour advection / laplacien)
NEARBY = np.where((DIST > 0) & (DIST < 500))[0]

# Poids géographiques : plus une ville est proche, plus elle pèse
_w = 1.0 / (1.0 + DIST / float(np.median(DIST[DIST > 0])))
_w[PARIS_IDX] = 0.0
W_GEO = (_w / _w.sum()).astype(np.float32)
DIST2 = (DIST ** 2 + 1e-8).astype(np.float32)

# Coordonnées (lat, lon) des 20 villes
_coords = np.array([
    [52.37,  4.90], [41.39,  2.17], [52.49, -1.90], [50.85,  4.35],
    [55.68, 12.57], [51.51,  7.47], [53.35, -6.26], [51.23,  6.78],
    [51.46,  7.01], [50.11,  8.68], [50.94,  6.96], [51.51, -0.13],
    [53.48, -2.24], [43.30,  5.37], [45.46,  9.19], [48.14, 11.58],
    [48.85,  2.35], [51.92,  4.48], [48.78,  9.18], [45.07,  7.69],
], dtype=np.float32)

# Vecteurs unitaires ville → Paris (utilisés pour l'advection)
_dx = 2.35 - _coords[:, 1]
_dy = 48.85 - _coords[:, 0]
_n  = np.sqrt(_dx**2 + _dy**2) + 1e-8
DIR_X = (_dx / _n).astype(np.float32)
DIR_Y = (_dy / _n).astype(np.float32)


def _build_features(X):
    # X : (20, 24, 8) → on ajoute une dimension batch → (1, 20, 24, 8)
    X     = X[np.newaxis]
    paris = X[:, PARIS_IDX]   # (1, 24, 8) — historique Paris uniquement
    last  = paris[:, -1]      # (1, 8)     — valeurs actuelles Paris
    f     = []

    # Bloc 1 — état actuel de Paris (8 valeurs)
    f.append(last)

    # Bloc 2 — moyenne et écart-type sur 3h, 6h, 12h, 24h (40 valeurs)
    for fi in [F_TEMP, F_WIND, F_RAIN, F_HUM, F_CLOUD]:
        for w in [3, 6, 12, 24]:
            c = paris[:, -w:, fi]
            f += [c.mean(1, keepdims=True), c.std(1, keepdims=True)]

    # Bloc 3 — lags : différence entre maintenant et il y a X heures (18 valeurs)
    for fi in [F_TEMP, F_WIND, F_HUM]:
        for lag in [1, 2, 3, 6, 12, 23]:
            f.append((last[:, fi] - paris[:, -(1 + lag), fi]).reshape(-1, 1))

    # Bloc 4 — encodage cyclique de l'heure (2 valeurs)
    f += [
        np.full((1, 1), np.sin(2 * np.pi * 23 / 24), dtype=np.float32),
        np.full((1, 1), np.cos(2 * np.pi * 23 / 24), dtype=np.float32),
    ]

    # Bloc 5 — moyenne pondérée des voisins + écart Paris/voisins (16 valeurs)
    all_last = X[:, :, -1, :]
    for fi in [F_TEMP, F_WIND, F_RAIN, F_HUM]:
        wm = (all_last[:, :, fi] * W_GEO[None, :]).sum(1, keepdims=True)
        f += [wm, last[:, fi:fi+1] - wm]

    # Bloc 6 — composantes U (est-ouest) et V (nord-sud) du vent (4 valeurs)
    wr  = np.deg2rad(paris[:, :, F_WDIR])
    ws  = paris[:, :, F_WIND]
    u_p = ws * np.sin(wr)
    v_p = ws * np.cos(wr)
    f  += [u_p[:, -1:], v_p[:, -1:],
           u_p[:, -6:].mean(1, keepdims=True),
           v_p[:, -6:].mean(1, keepdims=True)]

    # Bloc 7 — advection : est-ce que le vent apporte de l'air chaud/froid vers Paris ? (3 valeurs)
    wr_all = np.deg2rad(X[:, :, -1, F_WDIR])
    ws_all = X[:, :, -1, F_WIND]
    u_all  = ws_all * np.sin(wr_all)
    v_all  = ws_all * np.cos(wr_all)
    adv    = u_all * DIR_X[None, :] + v_all * DIR_Y[None, :]
    adv_n  = adv[:, NEARBY]
    t_all  = X[:, :, -1, F_TEMP]
    w_all  = X[:, :, -1, F_WIND]
    r_all  = X[:, :, -1, F_RAIN]
    f.append((adv_n * (t_all[:, NEARBY] - t_all[:, PARIS_IDX:PARIS_IDX+1])).sum(1, keepdims=True))
    f.append((adv_n * (w_all[:, NEARBY] - w_all[:, PARIS_IDX:PARIS_IDX+1])).sum(1, keepdims=True))
    f.append((adv_n *  r_all[:, NEARBY]).sum(1, keepdims=True))

    # Bloc 8 — accélération du vent et de la température (dérivée seconde) (2 valeurs)
    f.append((paris[:, -1, F_WIND] - 2*paris[:, -2, F_WIND] + paris[:, -3, F_WIND]).reshape(-1, 1))
    f.append((paris[:, -1, F_TEMP] - 2*paris[:, -2, F_TEMP] + paris[:, -3, F_TEMP]).reshape(-1, 1))

    # Bloc 9 — laplacien spatial : Paris est-il plus chaud/froid que ses voisins ? (3 valeurs)
    for fi in [F_TEMP, F_HUM, F_WIND]:
        lap = ((all_last[:, NEARBY, fi] - all_last[:, PARIS_IDX:PARIS_IDX+1, fi])
               / DIST2[None, NEARBY]).mean(1, keepdims=True)
        f.append(lap)

    # Bloc 10 — convergence du vent : les vents voisins sont-ils cohérents ? (2 valeurs)
    f.append((adv_n / (DIST[NEARBY][None, :] + 1e-8)).sum(1, keepdims=True))
    wsin = np.sin(wr_all[:, NEARBY])
    wcos = np.cos(wr_all[:, NEARBY])
    Rbar = np.sqrt(wsin.mean(1)**2 + wcos.mean(1)**2)
    f.append((1.0 - Rbar).reshape(-1, 1))

    # Bloc 10b — détecteur de front météo (2 valeurs)
    for fi in [F_TEMP, F_HUM]:
        spat = all_last[:, NEARBY, fi].std(1)
        tend = np.abs(paris[:, -1, fi] - paris[:, -7, fi])
        f.append((spat * tend).reshape(-1, 1))

    # Bloc 10c — pentes linéaires sur 6h, 12h, 24h (9 valeurs)
    for fi in [F_TEMP, F_WIND, F_HUM]:
        for w in [6, 12, 24]:
            c   = paris[:, -w:, fi]
            t   = np.arange(w, dtype=np.float32)
            tm  = t.mean()
            num = ((c - c.mean(1, keepdims=True)) * (t - tm)[None, :]).sum(1)
            den = float(((t - tm)**2).sum()) + 1e-8
            f.append((num / den).reshape(-1, 1))

    # Bloc 11 — rain_proxy : humidité × nuages → signal de pluie imminente (1 valeur)
    rain_proxy = (np.maximum(0.0, last[:, F_HUM] - 80.0) * last[:, F_CLOUD] / 100.0).reshape(-1, 1)
    f.append(rain_proxy)

    # Bloc 12 — T_residual : Paris est-il au-dessus de sa moyenne 24h ? (1 valeur)
    t_residual = (last[:, F_TEMP] - paris[:, :, F_TEMP].mean(1)).reshape(-1, 1)
    f.append(t_residual)

    out = np.hstack(f)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class Agent:
    def __init__(self):
        # Chargement des 3 modèles entraînés
        with open("model_temp.pkl", "rb") as f: self.model_temp = pickle.load(f)
        with open("model_wind.pkl", "rb") as f: self.model_wind = pickle.load(f)
        with open("model_rain.pkl", "rb") as f: self.model_rain = pickle.load(f)

    def predict(self, X_test):
        last  = X_test[PARIS_IDX, -1, :]   # état actuel Paris
        feats = _build_features(X_test)     # 103 features

        # Température : valeur actuelle + delta prédit
        pred_temp = float(last[F_TEMP]) + float(self.model_temp.predict(feats)[0])

        # Vent : valeur actuelle + delta prédit, jamais négatif
        pred_wind = max(0.0, float(last[F_WIND]) + float(self.model_wind.predict(feats)[0]))

        # Pluie : prédiction directe du niveau, jamais négatif
        pred_rain = max(0.0, float(self.model_rain.predict(feats)[0]))

        return np.array([pred_temp, pred_wind, pred_rain], dtype=np.float32)
