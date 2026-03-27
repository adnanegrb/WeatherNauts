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


def _lags(serie):
    last = serie[:, -1]
    return [last - serie[:, max(0, 23 - d)] for d in [1, 2, 3, 4, 6, 8, 12, 18, 23]]


def _stats_fenetres(serie, fenetres=(6, 12, 24)):
    blocs = []
    for f in fenetres:
        w = serie[:, -f:]
        blocs += [w.mean(axis=1), w.std(axis=1), w.min(axis=1), w.max(axis=1)]
    return blocs


def _pentes(serie, fenetres=(6, 12, 24)):
    blocs = []
    for f in fenetres:
        w = serie[:, -f:]
        x = np.arange(f, dtype=np.float32)
        x = x - x.mean()
        denom = float((x ** 2).sum())
        blocs.append((w * x[np.newaxis, :]).sum(axis=1) / denom)
    return blocs


def _pentes_multi(tableau, fenetres=(6, 12, 24)):
    blocs = []
    for f in fenetres:
        w = tableau[:, -f:, :]
        x = np.arange(f, dtype=np.float32)
        x = x - x.mean()
        denom = float((x ** 2).sum())
        blocs.append((w * x[np.newaxis, :, np.newaxis]).sum(axis=1) / denom)
    return blocs


def features_ridge(tableau):
    seul = (tableau.ndim == 3)
    if seul:
        tableau = tableau[np.newaxis]

    nb    = tableau.shape[0]
    paris = tableau[:, PARIS_IDX]
    p_last = paris[:, -1]

    blocs = [
        paris[:, :, 0],
        paris[:, :, 2],
        paris[:, :, 4],
        paris[:, :, 5],
        paris[:, :, 1],
        p_last,
        paris[:, -6:, 0],
        paris[:, -12:, 0],
        paris.mean(axis=1)[:, [0, 1, 2, 4, 5]],
        paris.std(axis=1)[:, [0, 2]],
        np.tile(_CYCL, (nb, 1)),
    ]

    blocs += _lags(paris[:, :, 0])
    blocs += _stats_fenetres(paris[:, :, 0])
    blocs += _pentes(paris[:, :, 0])
    blocs += _pentes_multi(paris)

    for idx_ville in VOISINS_TRIES[:5]:
        v     = tableau[:, idx_ville]
        poids = float(POIDS_GEO[idx_ville])
        blocs.append(v[:, :, 0] * poids)
        blocs.append(v[:, -1:, [0, 2]] * poids)
        blocs.append((p_last[:, [0]] - v[:, -1, [0]]) * poids)

    sortie = np.concatenate([b.reshape(nb, -1) for b in blocs], axis=1).astype(np.float32)
    return sortie[0] if seul else sortie


def features_xgb(tableau):
    seul = (tableau.ndim == 3)
    if seul:
        tableau = tableau[np.newaxis]

    nb    = tableau.shape[0]
    paris = tableau[:, PARIS_IDX]
    p_last = paris[:, -1]

    blocs = [paris.reshape(nb, -1)]

    blocs += _lags(paris[:, :, 0])
    blocs += _lags(paris[:, :, 2])

    for feat_idx in range(tableau.shape[3]):
        blocs += _stats_fenetres(paris[:, :, feat_idx])

    blocs += _pentes_multi(paris)

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
        if idx_ville in VOISINS_TRIES[:6]:
            blocs += _stats_fenetres(v[:, :, 0], fenetres=(6, 12))
            blocs += _pentes(v[:, :, 0], fenetres=(6,))

    derniere_h = tableau[:, :, -1, :]
    poids_col  = POIDS_GEO.reshape(1, 20, 1)
    blocs += [
        (derniere_h * poids_col).mean(axis=1),
        derniere_h.std(axis=1),
        (tableau[:, :, -1, :] - tableau[:, :, -7, :]).mean(axis=1),
        tableau[:, :, -1, :].max(axis=1) - tableau[:, :, -1, :].min(axis=1),
    ]

    sortie = np.concatenate([b.reshape(nb, -1) for b in blocs], axis=1).astype(np.float32)
    return sortie[0] if seul else sortie


class Agent:
    def __init__(self):
        self.model    = None
        self.biais    = {}
        self.entraine = False

    def train(self, X_train, y_train):
        from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            ExtraTreesRegressor,
            RandomForestRegressor,
            StackingRegressor,
        )
        from sklearn.neural_network import MLPRegressor
        from sklearn.feature_selection import SelectPercentile, f_regression
        from xgboost import XGBRegressor

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        f_ridge = features_ridge(X_train)
        f_xgb   = features_xgb(X_train)

        ridge = Pipeline([
            ("scaler", RobustScaler()),
            ("model",  Ridge(alpha=0.05))
        ])

        huber = Pipeline([
            ("scaler", RobustScaler()),
            ("model",  HuberRegressor(epsilon=1.15, alpha=0.01, max_iter=400))
        ])

        elastic = Pipeline([
            ("scaler", RobustScaler()),
            ("model",  ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=1000))
        ])

        mlp = Pipeline([
            ("scaler",    StandardScaler()),
            ("selection", SelectPercentile(f_regression, percentile=70)),
            ("model",     MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                learning_rate_init=0.001,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                alpha=0.001,
            ))
        ])

        gbr = Pipeline([
            ("scaler",    RobustScaler()),
            ("selection", SelectPercentile(f_regression, percentile=60)),
            ("model",     GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            ))
        ])

        temp_stack = StackingRegressor(
            estimators=[
                ("ridge",   ridge),
                ("huber",   huber),
                ("elastic", elastic),
                ("mlp",     mlp),
                ("gbr",     gbr),
            ],
            final_estimator=Ridge(alpha=0.1),
            cv=5,
            passthrough=False,
            n_jobs=-1,
        )
        temp_stack.fit(f_ridge, y_train[:, 0])

        modeles = {"temperature": temp_stack}

        xgb_params = dict(
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.6,
            colsample_bylevel=0.8,
            reg_alpha=0.3,
            reg_lambda=1.5,
            min_child_weight=4,
            gamma=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )

        for i, nom in enumerate(["wind_speed", "rain"]):
            xgb = XGBRegressor(**xgb_params)
            xgb.fit(f_xgb, y_train[:, i + 1], verbose=False)

            et = Pipeline([
                ("scaler", RobustScaler()),
                ("model",  ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=42,
                ))
            ])
            et.fit(f_xgb, y_train[:, i + 1])

            rf = Pipeline([
                ("scaler", RobustScaler()),
                ("model",  RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=4,
                    n_jobs=-1,
                    random_state=42,
                ))
            ])
            rf.fit(f_xgb, y_train[:, i + 1])

            modeles[nom] = {"xgb": xgb, "et": et, "rf": rf}

        pred_temp = temp_stack.predict(f_ridge)
        self.biais["temperature"] = float(np.mean(y_train[:, 0] - pred_temp))

        for i, nom in enumerate(["wind_speed", "rain"]):
            p = (0.60 * modeles[nom]["xgb"].predict(f_xgb)
               + 0.25 * modeles[nom]["et"].predict(f_xgb)
               + 0.15 * modeles[nom]["rf"].predict(f_xgb))
            self.biais[nom] = float(np.mean(y_train[:, i + 1] - p))

        self.model    = modeles
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

        pred_temp = float(self.model["temperature"].predict(f_ridge)[0])
        pred_temp += self.biais.get("temperature", 0.0)

        resultats = [pred_temp]
        for nom in ["wind_speed", "rain"]:
            p = (0.60 * float(self.model[nom]["xgb"].predict(f_xgb)[0])
               + 0.25 * float(self.model[nom]["et"].predict(f_xgb)[0])
               + 0.15 * float(self.model[nom]["rf"].predict(f_xgb)[0]))
            p += self.biais.get(nom, 0.0)
            resultats.append(max(0.0, p))

        return np.array(resultats, dtype=np.float32)