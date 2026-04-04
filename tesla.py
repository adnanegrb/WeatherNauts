import numpy as np

class _LogReg:
    """Régression logistique minimale — zéro dépendance externe."""
    def __init__(self, lr=0.01, n_iter=300):
        self.lr, self.n_iter, self.w, self.b = lr, n_iter, None, 0.0

    def fit(self, X, y):
        X = X.astype(np.float32)
        self.w = np.zeros(X.shape[1], dtype=np.float32)
        for _ in range(self.n_iter):
            p   = 1.0 / (1.0 + np.exp(-np.clip(X @ self.w + self.b, -30, 30)))
            err = p - y
            self.w -= self.lr * (X.T @ err) / len(y)
            self.b -= self.lr * err.mean()

    def predict_proba(self, X):
        X = X.astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ self.w + self.b, -30, 30)))
        return np.stack([1 - p, p], axis=1)




PARIS_IDX = 16

DISTANCES_PARIS = [431, 830, 502, 264, 1027, 469, 781, 411, 440, 479,
                   403, 344, 570, 661, 640, 685, 0, 373, 500, 580]

_dist  = np.array(DISTANCES_PARIS, dtype=np.float32)
_sigma = float(np.median(_dist[_dist > 0]))
POIDS_GEO = (1.0 / (1.0 + _dist / _sigma)).astype(np.float32)
DIST2     = (_dist ** 2 + 1e-8).astype(np.float32)

VOISINS_TRIES = sorted(
    [i for i in range(20) if i != PARIS_IDX],
    key=lambda i: DISTANCES_PARIS[i]
)
NEARBY = np.array([i for i in range(20) if i != PARIS_IDX and DISTANCES_PARIS[i] < 500])

_heures = np.arange(24, dtype=np.float32)
_CYCL   = np.stack([
    np.sin(2 * np.pi * _heures / 24),
    np.cos(2 * np.pi * _heures / 24)
], axis=1).flatten()

_coords = np.array([
    [52.37, 4.90], [41.39, 2.17], [52.49, -1.90], [50.85, 4.35],
    [55.68, 12.57],[51.51, 7.47], [53.35, -6.26], [51.23, 6.78],
    [51.46, 7.01], [50.11, 8.68], [50.94, 6.96],  [51.51, -0.13],
    [53.48, -2.24],[43.30, 5.37], [45.46, 9.19],  [48.14, 11.58],
    [48.85, 2.35], [51.92, 4.48], [48.78, 9.18],  [45.07, 7.69]
], dtype=np.float32)
_dx   = 2.35 - _coords[:, 1]
_dy   = 48.85 - _coords[:, 0]
_n    = np.sqrt(_dx**2 + _dy**2) + 1e-8
DIR_X = (_dx / _n).astype(np.float32)
DIR_Y = (_dy / _n).astype(np.float32)

F_TEMP, F_RAIN, F_WIND, F_WDIR = 0, 1, 2, 3


def _features(X):
    paris    = X[PARIS_IDX]
    last     = paris[-1]
    all_last = X[:, -1, :]

    diff1  = paris[-1] - paris[-2]
    diff6  = paris[-1] - paris[-7]
    diff12 = paris[-1] - paris[-13]
    mean24 = paris.mean(0)
    std24  = paris.std(0)

    voisins = []
    for idx in VOISINS_TRIES[:5]:
        poids = float(POIDS_GEO[idx])
        v = X[idx]
        voisins.append(v[-1] * poids)
        voisins.append((last - v[-1]) * poids)

    # ── Physique #1 — Advection ───────────────────────────────────────────────
    if X.shape[2] > F_WDIR:
        wdir = np.deg2rad(all_last[:, F_WDIR])
        wspd = all_last[:, F_WIND]
        u    = wspd * np.sin(wdir)
        v    = wspd * np.cos(wdir)
        adv  = u * DIR_X + v * DIR_Y
        adv_n    = adv[NEARBY]
        adv_feats = np.array([
            (adv_n * (all_last[NEARBY, F_TEMP] - last[F_TEMP])).sum(),
            (adv_n * (all_last[NEARBY, F_WIND] - last[F_WIND])).sum(),
            (adv_n *  all_last[NEARBY, F_RAIN]).sum(),
        ], dtype=np.float32)
    else:
        adv_feats = np.zeros(3, dtype=np.float32)

    # ── Physique #2 — Laplacien spatial ──────────────────────────────────────
    lap_feats = np.array([
        ((all_last[NEARBY, F_TEMP] - last[F_TEMP]) / DIST2[NEARBY]).mean(),
        ((all_last[NEARBY, F_WIND] - last[F_WIND]) / DIST2[NEARBY]).mean(),
        ((all_last[NEARBY, F_RAIN] - last[F_RAIN]) / DIST2[NEARBY]).mean(),
    ], dtype=np.float32)

    # ── Physique #3 — Convergence du vent ────────────────────────────────────
    if X.shape[2] > F_WDIR:
        wdir_n = np.deg2rad(all_last[NEARBY, F_WDIR])
        R_bar  = float(np.sqrt(np.sin(wdir_n).mean()**2 + np.cos(wdir_n).mean()**2))
        conv   = np.array([1.0 - R_bar], dtype=np.float32)
    else:
        conv = np.zeros(1, dtype=np.float32)

    return np.concatenate([
        last, diff1, diff6, diff12,
        mean24, std24, _CYCL,
        *voisins,
        adv_feats, lap_feats, conv,
    ]).astype(np.float32)


class Agent:
    def __init__(self):
        self.models   = {}
        self.rain_clf = None
        self.biais    = {"temp": 0.0, "wind": 0.0, "rain": 0.0}
        self.entraine = False
        self._buf_X   = []
        self._buf_y   = []

    def train(self, X_train, y_train, sample_weight=None):
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
        rain       = y_train[:, 2]
        sw         = sample_weight

        targets = {
            "temp": y_train[:, 0] - last_paris[:, 0],
            "wind": y_train[:, 1] - last_paris[:, 2],
            "rain": rain,
        }

        # Two-stage pluie — classifier
        self.rain_clf = _LogReg()
        self.rain_clf.fit(F, (rain > 0.05).astype(int))

        if USE_LGB:
            params = {
                "temp": dict(
                    boosting_type="dart", objective="regression_l1",
                    num_leaves=63, learning_rate=0.05, n_estimators=800,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
                    drop_rate=0.1, max_drop=50,
                    n_jobs=-1, random_state=42, verbosity=-1,
                ),
                "wind": dict(
                    boosting_type="dart", objective="huber", alpha=0.85,
                    num_leaves=255, learning_rate=0.01, n_estimators=3000,
                    drop_rate=0.08, max_drop=60, skip_drop=0.45,
                    subsample=0.75, subsample_freq=1,
                    colsample_bytree=0.5, colsample_bylevel=0.6,
                    reg_alpha=0.1, reg_lambda=2.0,
                    min_child_samples=5, min_child_weight=1e-4,
                    n_jobs=-1, random_state=42, verbosity=-1,
                ),
                "rain": dict(
                    boosting_type="goss", objective="tweedie",
                    tweedie_variance_power=1.8,
                    num_leaves=63, learning_rate=0.015, n_estimators=2500,
                    top_rate=0.2, other_rate=0.1,
                    colsample_bytree=0.5, colsample_bylevel=0.7,
                    reg_alpha=0.5, reg_lambda=2.0,
                    min_child_samples=12, min_child_weight=1e-3,
                    n_jobs=-1, random_state=42, verbosity=-1,
                ),
            }
            for name, target in targets.items():
                mask  = (rain > 0.05) if name == "rain" else np.ones(len(F), dtype=bool)
                F_fit = F[mask] if mask.sum() > 20 else F
                t_fit = target[mask] if mask.sum() > 20 else target
                sw_fit = sw[mask] if (sw is not None and mask.sum() > 20) else sw
                m = lgb.LGBMRegressor(**params[name])
                m.fit(F_fit, t_fit)
                self.models[name] = m

        else:
            from sklearn.ensemble import GradientBoostingRegressor
            params = {
                "temp": dict(loss="absolute_error", n_estimators=300, max_depth=5, learning_rate=0.05),
                "wind": dict(loss="huber", n_estimators=300, max_depth=6, learning_rate=0.05, alpha=0.85),
                "rain": dict(loss="huber", n_estimators=200, max_depth=4, learning_rate=0.05, alpha=0.9),
            }
            for name, target in targets.items():
                mask  = (rain > 0.05) if name == "rain" else np.ones(len(F), dtype=bool)
                F_fit = F[mask] if mask.sum() > 20 else F
                t_fit = target[mask] if mask.sum() > 20 else target
                m = GradientBoostingRegressor(**params[name], random_state=42)
                m.fit(F_fit, t_fit)
                self.models[name] = m

        self.biais    = {"temp": 0.0, "wind": 0.0, "rain": 0.0}
        self.entraine = True

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float32)
        last   = X_test[PARIS_IDX, -1]

        if not self.entraine:
            paris = X_test[PARIS_IDX]
            return np.array([
                float(paris[-1, 0]),
                max(0., float(paris[-1, 2])),
                max(0., float(paris[-1, 1])),
            ], dtype=np.float32)

        f = _features(X_test).reshape(1, -1)

        temp = last[0] + float(self.models["temp"].predict(f)) + self.biais["temp"]
        wind = max(0., last[2] + float(self.models["wind"].predict(f)) + self.biais["wind"])

        p_rain    = float(self.rain_clf.predict_proba(f)[0, 1])
        rain_cond = max(0., float(self.models["rain"].predict(f)))
        rain      = max(0., p_rain * rain_cond + self.biais["rain"])

        return np.array([temp, wind, rain], dtype=np.float32)

    def update(self, X, y_true):
        """Appeler après chaque step — bias EMA + stockage buffer."""
        y_true = np.array(y_true, dtype=np.float32)
        y_pred = self.predict(X)
        for i, k in enumerate(["temp", "wind", "rain"]):
            self.biais[k] = 0.9 * self.biais[k] + 0.1 * float(y_true[i] - y_pred[i])
        self._buf_X.append(np.array(X, dtype=np.float32))
        self._buf_y.append(y_true)

    def refit(self, window=200, half_life=72):
        """Appeler toutes les ~50 obs — sliding window + exponential decay."""
        if len(self._buf_X) < 30:
            return
        X_w = np.array(self._buf_X[-window:], dtype=np.float32)
        y_w = np.array(self._buf_y[-window:], dtype=np.float32)
        n   = len(X_w)
        age = np.arange(n - 1, -1, -1, dtype=np.float32)
        sw  = np.exp(-np.log(2) * age / half_life)
        self.train(X_w, y_w, sample_weight=sw / sw.sum())
