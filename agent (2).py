
import numpy as np

# on ne connaît pas trop ML Arena, on a eu l'aide de l'IA pour essayer d'éviter les erreurs 🤞

PARIS_IDX = 16

DISTANCES_PARIS = [431, 830, 502, 264, 1027, 469, 781, 411, 440, 479,
                   403, 344, 570, 661, 640, 685, 0, 373, 500, 580]

_dist  = np.array(DISTANCES_PARIS, dtype=np.float32)
_sigma = float(np.median(_dist[_dist > 0]))
POIDS_GEO = (1.0 / (1.0 + _dist / _sigma)).astype(np.float32)

VOISINS_TRIES = sorted(
    [i for i in range(20) if i != PARIS_IDX],
    key=lambda i: DISTANCES_PARIS[i]
)

_heures = np.arange(24, dtype=np.float32)
_CYCL = np.stack([
    np.sin(2 * np.pi * _heures / 24),
    np.cos(2 * np.pi * _heures / 24)
], axis=1).flatten()


def features_ridge(tableau):
    seul = (tableau.ndim == 3)
    if seul:
        tableau = tableau[np.newaxis]

    nb     = tableau.shape[0]
    paris  = tableau[:, PARIS_IDX]
    p_last = paris[:, -1]

    blocs = [
        paris[:, :, 0],
        paris[:, :, 2],
        paris[:, :, 4],
        p_last,
        paris[:, -6:, 0],
        paris.mean(axis=1)[:, [0, 2, 4]],
        paris.std(axis=1)[:, [0, 2]],
        np.tile(_CYCL, (nb, 1)),
    ]

    for idx_ville in VOISINS_TRIES[:5]:
        v     = tableau[:, idx_ville]
        poids = float(POIDS_GEO[idx_ville])
        blocs.append(v[:, :, 0] * poids)
        blocs.append(v[:, -1:, [0, 2]] * poids)

    sortie = np.concatenate([b.reshape(nb, -1) for b in blocs], axis=1).astype(np.float32)
    return sortie[0] if seul else sortie


def features_xgb(tableau):
    seul = (tableau.ndim == 3)
    if seul:
        tableau = tableau[np.newaxis]

    nb     = tableau.shape[0]
    paris  = tableau[:, PARIS_IDX]
    p_last = paris[:, -1]

    blocs = [paris.reshape(nb, -1)]

    for decalage in [1, 2, 3, 6, 12, 18]:
        blocs.append(p_last - paris[:, max(0, 23 - decalage)])

    blocs += [
        paris.mean(axis=1),
        paris.std(axis=1),
        paris.min(axis=1),
        paris.max(axis=1),
        paris[:, -6:].mean(axis=1),
        paris[:, -6:].std(axis=1),
        np.tile(_CYCL, (nb, 1)),
    ]

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

    derniere_h = tableau[:, :, -1, :]
    poids_col  = POIDS_GEO.reshape(1, 20, 1)
    blocs += [
        (derniere_h * poids_col).mean(axis=1),
        derniere_h.std(axis=1),
        (tableau[:, :, -1, :] - tableau[:, :, -7, :]).mean(axis=1),
    ]

    sortie = np.concatenate([b.reshape(nb, -1) for b in blocs], axis=1).astype(np.float32)
    return sortie[0] if seul else sortie


class Agent:
    def __init__(self):
        self.model = None
        self.entraine = False

    def train(self, X_train, y_train):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from xgboost import XGBRegressor

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        f_ridge = features_ridge(X_train)
        f_xgb   = features_xgb(X_train)

        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=0.1))
        ])
        ridge.fit(f_ridge, y_train[:, 0])

        modeles = {"temperature": ridge}

        for i, nom in enumerate(["wind_speed", "rain"]):
            xgb = XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=2.0,
                min_child_weight=5,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            )
            xgb.fit(f_xgb, y_train[:, i + 1])
            modeles[nom] = xgb

        self.model = modeles
        self.entraine = True

    def predict(self, X_test):
        """
        Predict Paris weather at T+6h.

        Args:
            X_test: shape (20, 24, 8) - 20 cities, 24 hours, 8 features

        Returns:
            predictions: shape (3,) - [temperature, wind_speed, rain]
        """
        X_test = np.array(X_test, dtype=np.float32)

        if not self.entraine or self.model is None:
            paris = X_test[PARIS_IDX]
            poids = np.linspace(0.3, 1.0, 24)
            poids = poids / poids.sum()
            tend_t = float(paris[-1, 0] - paris[-7, 0])
            tend_w = float(paris[-1, 2] - paris[-7, 2])
            corr_t, corr_w = 0.0, 0.0
            for k, idx_v in enumerate([3, 11, 17, 10, 0]):
                pw = [0.35, 0.25, 0.15, 0.15, 0.10][k]
                v  = X_test[idx_v]
                corr_t += pw * float(v[-1, 0] - v[-7, 0]) * 0.25
                corr_w += pw * float(v[-1, 2] - v[-7, 2]) * 0.25
            return np.array([
                float(np.sum(poids * paris[:, 0])) + 0.35 * tend_t + corr_t,
                max(0.0, float(np.sum(poids * paris[:, 2])) + 0.35 * tend_w + corr_w),
                max(0.0, float(np.sum(poids * paris[:, 1])))
            ], dtype=np.float32)

        f_ridge = features_ridge(X_test).reshape(1, -1)
        f_xgb   = features_xgb(X_test).reshape(1, -1)

        return np.array([
            float(self.model["temperature"].predict(f_ridge)[0]),
            max(0.0, float(self.model["wind_speed"].predict(f_xgb)[0])),
            max(0.0, float(self.model["rain"].predict(f_xgb)[0]))
        ], dtype=np.float32)
