import numpy as np

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


def _features(X):
    paris = X[PARIS_IDX]          # (24, N_feat)
    last  = paris[-1]             # état actuel

    # Deltas temporels
    diff1  = paris[-1] - paris[-2]
    diff6  = paris[-1] - paris[-7]
    diff12 = paris[-1] - paris[-13]

    # Stats globales Paris
    mean24 = paris.mean(0)
    std24  = paris.std(0)

    # Voisins géo-pondérés (top 5)
    voisins = []
    for idx in VOISINS_TRIES[:5]:
        poids = float(POIDS_GEO[idx])
        v = X[idx]
        voisins.append(v[-1] * poids)
        voisins.append((last - v[-1]) * poids)

    return np.concatenate([
        last, diff1, diff6, diff12,
        mean24, std24,
        _CYCL,
        *voisins,
    ]).astype(np.float32)


class Agent:
    def __init__(self):
        self.models   = {}
        self.entraine = False

    def train(self, X_train, y_train):
        try:
            import lightgbm as lgb
            USE_LGB = True
        except (ImportError, OSError):
            from sklearn.ensemble import GradientBoostingRegressor
            USE_LGB = False

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        F          = np.array([_features(x) for x in X_train])
        last_paris = X_train[:, PARIS_IDX, -1, :]

        targets = {
            "temp" : y_train[:, 0] - last_paris[:, 0],
            "wind" : y_train[:, 1] - last_paris[:, 2],
            "rain" : y_train[:, 2] - last_paris[:, 1],
        }

        if USE_LGB:
            params = {
                "temp": dict(
                    boosting_type="dart",
                    objective="regression_l1",
                    num_leaves=63,
                    learning_rate=0.05,
                    n_estimators=800,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_samples=10,
                    drop_rate=0.1,
                    max_drop=50,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=-1,
                ),
                "wind": dict(
                    boosting_type="dart",
                    objective="huber",
                    alpha=0.85,
                    num_leaves=255,
                    learning_rate=0.01,
                    n_estimators=3000,
                    drop_rate=0.08,
                    max_drop=60,
                    skip_drop=0.45,
                    subsample=0.75,
                    subsample_freq=1,
                    colsample_bytree=0.5,
                    colsample_bylevel=0.6,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    min_child_samples=5,
                    min_child_weight=1e-4,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=-1,
                ),
                "rain": dict(
                    boosting_type="goss",
                    objective="tweedie",
                    tweedie_variance_power=1.8,
                    num_leaves=63,
                    learning_rate=0.015,
                    n_estimators=2500,
                    top_rate=0.2,
                    other_rate=0.1,
                    colsample_bytree=0.5,
                    colsample_bylevel=0.7,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    min_child_samples=12,
                    min_child_weight=1e-3,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=-1,
                ),
            }
            for name, delta in targets.items():
                m = lgb.LGBMRegressor(**params[name])
                m.fit(F, delta)
                self.models[name] = m

        else:
            params = {
                "temp": dict(loss="absolute_error", n_estimators=300, max_depth=5, learning_rate=0.05),
                "wind": dict(loss="huber",          n_estimators=300, max_depth=6, learning_rate=0.05, alpha=0.85),
                "rain": dict(loss="huber",          n_estimators=200, max_depth=4, learning_rate=0.05, alpha=0.9),
            }
            for name, delta in targets.items():
                m = GradientBoostingRegressor(**params[name], random_state=42)
                m.fit(F, delta)
                self.models[name] = m

        self.entraine = True

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float32)
        last   = X_test[PARIS_IDX, -1]

        if not self.entraine:
            # Fallback naif si pas encore entraîné
            paris = X_test[PARIS_IDX]
            return np.array([
                float(paris[-1, 0]),
                max(0., float(paris[-1, 2])),
                max(0., float(paris[-1, 1])),
            ], dtype=np.float32)

        f = _features(X_test).reshape(1, -1)

        temp = last[0] + float(self.models["temp"].predict(f))
        wind = last[2] + float(self.models["wind"].predict(f))
        rain = last[1] + float(self.models["rain"].predict(f))

        return np.array([temp, max(0., wind), max(0., rain)], dtype=np.float32)
