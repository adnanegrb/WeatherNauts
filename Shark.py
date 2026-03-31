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
        import lightgbm as lgb
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import RobustScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import KFold

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        f_ridge = features_ridge(X_train)
        f_xgb   = features_xgb(X_train)
        lgb_params_temp_base = dict(
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
        )

        lgb_params_temp_alt = dict(
            boosting_type="gbdt",
            objective="huber",
            alpha=0.9,                       
            num_leaves=127,
            learning_rate=0.03,
            n_estimators=1200,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.05,
            reg_lambda=0.5,
            min_child_samples=8,
            n_jobs=-1,
            random_state=123,
            verbosity=-1,
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_base = np.zeros(len(f_ridge))
        oof_alt  = np.zeros(len(f_ridge))

        models_base, models_alt = [], []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(f_ridge)):
            X_tr, X_val = f_ridge[tr_idx], f_ridge[val_idx]
            y_tr, y_val = y_train[tr_idx, 0], y_train[val_idx, 0]

            m_base = lgb.LGBMRegressor(**lgb_params_temp_base)
            m_base.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            oof_base[val_idx] = m_base.predict(X_val)
            models_base.append(m_base)

            m_alt = lgb.LGBMRegressor(**lgb_params_temp_alt)
            m_alt.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            oof_alt[val_idx] = m_alt.predict(X_val)
            models_alt.append(m_alt)
        meta_X = np.stack([oof_base, oof_alt], axis=1)
        meta_ridge = Pipeline([
            ("scaler", RobustScaler()),
            ("model",  Ridge(alpha=0.1)),
        ])
        meta_ridge.fit(meta_X, y_train[:, 0])

        temp_ensemble = {
            "models_base": models_base,
            "models_alt":  models_alt,
            "meta":        meta_ridge,
        }
        lgb_params_wind = dict(
            boosting_type="goss",            
            objective="tweedie",
            tweedie_variance_power=1.5,
            num_leaves=127,
            learning_rate=0.02,
            n_estimators=2000,
            top_rate=0.2,
            other_rate=0.1,
            colsample_bytree=0.6,
            colsample_bylevel=0.8,
            reg_alpha=0.3,
            reg_lambda=1.5,
            min_child_samples=8,
            min_child_weight=1e-3,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        )

        lgb_params_rain = dict(
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
        )
        lgb_params_sec = dict(
            boosting_type="gbdt",
            objective="regression_l1",
            num_leaves=63,
            learning_rate=0.04,
            n_estimators=800,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=1.0,
            min_child_samples=10,
            n_jobs=-1,
            random_state=99,
            verbosity=-1,
        )

        modeles = {"temperature": temp_ensemble}

        kf2 = KFold(n_splits=5, shuffle=True, random_state=7)

        for i, nom in enumerate(["wind_speed", "rain"]):
            y_col  = y_train[:, i + 1]
            params = lgb_params_wind if nom == "wind_speed" else lgb_params_rain
            oof_main = np.zeros(len(f_xgb))
            oof_sec  = np.zeros(len(f_xgb))
            models_main, models_sec_list = [], []

            for tr_idx, val_idx in kf2.split(f_xgb):
                X_tr, X_val = f_xgb[tr_idx], f_xgb[val_idx]
                y_tr, y_val = y_col[tr_idx], y_col[val_idx]

                m_main = lgb.LGBMRegressor(**params)
                m_main.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
                )
                oof_main[val_idx] = np.maximum(0.0, m_main.predict(X_val))
                models_main.append(m_main)

                m_sec = lgb.LGBMRegressor(**lgb_params_sec)
                m_sec.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                )
                oof_sec[val_idx] = np.maximum(0.0, m_sec.predict(X_val))
                models_sec_list.append(m_sec)
            meta_wr = Pipeline([
                ("scaler", RobustScaler()),
                ("model",  Ridge(alpha=0.1, positive=True)),
            ])
            meta_wr.fit(np.stack([oof_main, oof_sec], axis=1), y_col)

            modeles[nom] = {
                "models_main": models_main,
                "models_sec":  models_sec_list,
                "meta":        meta_wr,
            }
        def _pred_temp(f):
            base = np.mean([m.predict(f) for m in temp_ensemble["models_base"]], axis=0)
            alt  = np.mean([m.predict(f) for m in temp_ensemble["models_alt"]],  axis=0)
            return temp_ensemble["meta"].predict(np.stack([base, alt], axis=1))

        self.biais["temperature"] = float(
            np.mean(y_train[:, 0] - _pred_temp(f_ridge))
        )

        for i, nom in enumerate(["wind_speed", "rain"]):
            ens = modeles[nom]
            p_main = np.mean([m.predict(f_xgb) for m in ens["models_main"]], axis=0)
            p_sec  = np.mean([m.predict(f_xgb) for m in ens["models_sec"]],  axis=0)
            p = np.maximum(0.0, ens["meta"].predict(np.stack([p_main, p_sec], axis=1)))
            self.biais[nom] = float(np.mean(y_train[:, i + 1] - p))

        self.model    = modeles
        self.entraine = True

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float32)

        if not self.entraine or self.model is None:
            paris  = X_test[PARIS_IDX]
            poids  = np.linspace(0.3, 1.0, 24)
            poids  = poids / poids.sum()
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
        ens     = self.model
        base_pred = np.mean([m.predict(f_ridge) for m in ens["temperature"]["models_base"]], axis=0)
        alt_pred  = np.mean([m.predict(f_ridge) for m in ens["temperature"]["models_alt"]],  axis=0)
        pred_temp = float(
            ens["temperature"]["meta"].predict(np.stack([base_pred, alt_pred], axis=1))[0]
        )
        pred_temp += self.biais.get("temperature", 0.0)

        resultats = [pred_temp]
        for nom in ["wind_speed", "rain"]:
            p_main = np.mean([m.predict(f_xgb) for m in ens[nom]["models_main"]], axis=0)
            p_sec  = np.mean([m.predict(f_xgb) for m in ens[nom]["models_sec"]],  axis=0)
            p = float(
                ens[nom]["meta"].predict(np.stack([p_main, p_sec], axis=1))[0]
            )
            p += self.biais.get(nom, 0.0)
            resultats.append(max(0.0, p))

        return np.array(resultats, dtype=np.float32)