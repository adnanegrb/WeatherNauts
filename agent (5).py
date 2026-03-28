import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from scipy.stats import norm, gamma as sp_gamma

PARIS_IDX = 16

CITY_COORDS = np.array([
    [52.374,  4.890],
    [41.389,  2.159],
    [52.481, -1.900],
    [50.850,  4.349],
    [55.676, 12.566],
    [51.515,  7.466],
    [53.333, -6.249],
    [51.222,  6.776],
    [51.457,  7.012],
    [50.116,  8.684],
    [50.933,  6.950],
    [51.509, -0.126],
    [53.481, -2.237],
    [43.297,  5.381],
    [45.464,  9.190],
    [48.137, 11.575],
    [48.853,  2.349],
    [51.923,  4.479],
    [48.782,  9.177],
    [45.070,  7.687],
], dtype=np.float32)

PARIS_COORD = CITY_COORDS[PARIS_IDX]

DISTANCES_PARIS = np.array([
    431, 830, 502, 264, 1027, 469, 781, 411, 440, 479,
    403, 344, 570, 661, 640, 685, 0, 373, 500, 580
], dtype=np.float32)

_sigma = float(np.median(DISTANCES_PARIS[DISTANCES_PARIS > 0]))
POIDS_GEO = (1.0 / (1.0 + DISTANCES_PARIS / _sigma)).astype(np.float32)

_DELTA_LON = CITY_COORDS[:, 1] - PARIS_COORD[1]
_DELTA_LAT = CITY_COORDS[:, 0] - PARIS_COORD[0]
_DIST_NORM = np.sqrt(_DELTA_LON**2 + _DELTA_LAT**2 + 1e-6)
_UNIT_LON  = _DELTA_LON / _DIST_NORM
_UNIT_LAT  = _DELTA_LAT / _DIST_NORM

VOISINS_TRIES = sorted(
    [i for i in range(20) if i != PARIS_IDX],
    key=lambda i: DISTANCES_PARIS[i]
)

_HEURES = np.arange(24, dtype=np.float32)
_CYCL_H = np.stack([
    np.sin(2 * np.pi * _HEURES / 24),
    np.cos(2 * np.pi * _HEURES / 24)
], axis=1)


def _poids_vent_dynamique(wind_dir_deg_mean):
    wd_rad   = np.deg2rad(float(wind_dir_deg_mean))
    flux_lon = np.sin(wd_rad)
    flux_lat = np.cos(wd_rad)
    alignement = flux_lon * (-_UNIT_LON) + flux_lat * (-_UNIT_LAT)
    alignement = np.clip(alignement, 0.0, None)
    poids = POIDS_GEO * (1.0 + 2.0 * alignement)
    poids[PARIS_IDX] = 0.0
    return (poids / (poids.sum() + 1e-8)).astype(np.float32)


def _enrich(tableau):
    seul = (tableau.ndim == 3)
    if seul:
        tableau = tableau[np.newaxis]
    tableau = np.array(tableau, dtype=np.float32)
    N, V, H, F = tableau.shape
    wd = tableau[:, :, :, 3]
    rad = np.deg2rad(wd)
    wd_sin = np.sin(rad)
    wd_cos = np.cos(rad)
    h = np.arange(H, dtype=np.float32)
    h_sin = np.broadcast_to(np.sin(2*np.pi*h/24)[None,None,:], (N,V,H)).copy()
    h_cos = np.broadcast_to(np.cos(2*np.pi*h/24)[None,None,:], (N,V,H)).copy()
    extra = np.stack([wd_sin, wd_cos, h_sin, h_cos], axis=-1)
    out = np.concatenate([tableau, extra], axis=-1)
    return out[0] if seul else out


def _lags(s, lags=(1,2,3,4,6,8,12,18,23)):
    last = s[:, -1]
    return [last - s[:, max(0, 23-d)] for d in lags]


def _stats(s, wins=(6,12,24)):
    blocs = []
    for f in wins:
        w = s[:, -f:]
        blocs += [w.mean(1), w.std(1), w.min(1), w.max(1)]
    return blocs


def _slope(s, wins=(6,12,24)):
    blocs = []
    for f in wins:
        w = s[:, -f:]
        x = np.arange(f, dtype=np.float32) - (f-1)/2.0
        d = float((x**2).sum()) + 1e-8
        blocs.append((w * x[None,:]).sum(1) / d)
    return blocs


def _slope_multi(t, wins=(6,12,24)):
    blocs = []
    for f in wins:
        w = t[:, -f:, :]
        x = np.arange(f, dtype=np.float32) - (f-1)/2.0
        d = float((x**2).sum()) + 1e-8
        blocs.append((w * x[None,:,None]).sum(1) / d)
    return blocs


def _feat_temperature(tableau_raw):
    seul = (tableau_raw.ndim == 3)
    tab  = _enrich(tableau_raw)
    if seul:
        tab = tab[np.newaxis]
    N     = tab.shape[0]
    paris = tab[:, PARIS_IDX]
    p_last = paris[:, -1]
    temp   = paris[:, :, 0]
    rain   = paris[:, :, 1]
    wspeed = paris[:, :, 2]
    hum    = paris[:, :, 4]
    clouds = paris[:, :, 5]
    wd_sin = paris[:, :, 8]
    wd_cos = paris[:, :, 9]

    wd_mean = float(tab[0, PARIS_IDX, -6:, 3].mean()) if seul else tab[:, PARIS_IDX, -6:, 3].mean()
    if seul:
        pw = _poids_vent_dynamique(wd_mean)
    else:
        pw = np.stack([
            _poids_vent_dynamique(float(tab[n, PARIS_IDX, -6:, 3].mean()))
            for n in range(N)
        ], axis=0)

    blocs = [
        temp, wspeed, hum, clouds, rain,
        p_last,
        paris[:, -6:, 0],
        paris[:, -12:, 0],
        paris.mean(1)[:, [0,1,2,4,5]],
        paris.std(1)[:, [0,2]],
        np.tile(_CYCL_H.flatten(), (N,1)),
        paris[:, -1, 8:10],
        paris[:, -6:, 8],
        paris[:, -6:, 9],
        wspeed * wd_sin,
        wspeed * wd_cos,
        hum * temp / 100.0,
        clouds * hum / 100.0,
    ]
    blocs += _lags(temp)
    blocs += _stats(temp)
    blocs += _slope(temp)
    blocs += _slope_multi(paris[:, :, :6])

    for idx in VOISINS_TRIES[:8]:
        v = tab[:, idx]
        if seul:
            w = float(pw[idx])
        else:
            w = pw[:, idx]
        blocs.append(v[:, :, 0] * w)
        blocs.append(v[:, -1:, [0,2]] * w)
        blocs.append((p_last[:, [0]] - v[:, -1, [0]]) * w)
        blocs.append((temp - v[:, :, 0]) * w)
        if idx in VOISINS_TRIES[:5]:
            blocs += _stats(v[:, :, 0], (6,12))
            blocs += _slope(v[:, :, 0], (6,12))

    out = np.concatenate([b.reshape(N,-1) for b in blocs], axis=1).astype(np.float32)
    return out[0] if seul else out


def _feat_vent_pluie(tableau_raw):
    seul = (tableau_raw.ndim == 3)
    tab  = _enrich(tableau_raw)
    if seul:
        tab = tab[np.newaxis]
    N     = tab.shape[0]
    paris = tab[:, PARIS_IDX]
    p_last = paris[:, -1]
    temp   = paris[:, :, 0]
    rain   = paris[:, :, 1]
    wspeed = paris[:, :, 2]
    hum    = paris[:, :, 4]
    clouds = paris[:, :, 5]
    wd_sin = paris[:, :, 8]
    wd_cos = paris[:, :, 9]

    if seul:
        wd_mean = float(tab[0, PARIS_IDX, -6:, 3].mean())
        pw = _poids_vent_dynamique(wd_mean)
    else:
        pw = np.stack([
            _poids_vent_dynamique(float(tab[n, PARIS_IDX, -6:, 3].mean()))
            for n in range(N)
        ], axis=0)

    blocs = [
        paris[:, :, :8].reshape(N,-1),
        np.tile(_CYCL_H.flatten(), (N,1)),
        paris[:, -1, 8:10],
        wspeed * wd_sin,
        wspeed * wd_cos,
        clouds * hum / 100.0,
        paris.mean(1)[:, :8],
        paris.std(1)[:, :8],
        paris.min(1)[:, :8],
        paris.max(1)[:, :8],
        paris[:, -6:].mean(1)[:, :8],
        paris[:, -6:].std(1)[:, :8],
    ]
    blocs += _lags(temp)
    blocs += _lags(rain)
    blocs += _lags(wspeed)
    blocs += _stats(temp)
    blocs += _stats(rain, (6,12))
    blocs += _stats(wspeed, (6,12))
    blocs += _slope(temp)
    blocs += _slope(wspeed)

    for idx in VOISINS_TRIES:
        v      = tab[:, idx]
        v_last = v[:, -1]
        v_6h   = v[:, max(0,17)]
        if seul:
            w = float(pw[idx])
        else:
            w = pw[:, idx]
        blocs += [
            v_last[:, :6] * w,
            (v_last[:, [0,2,1]] - v_6h[:, [0,2,1]]) * w,
            v[:, :, 0] * w,
            v[:, :, 2] * w,
            (p_last[:, [0,2,1]] - v_last[:, [0,2,1]]) * w,
        ]
        if idx in VOISINS_TRIES[:6]:
            blocs += _stats(v[:, :, 0], (6,12))
            blocs += _slope(v[:, :, 0], (6,))

    dh = tab[:, :, -1, :8]
    pw_col = POIDS_GEO.reshape(1,20,1)
    blocs += [
        (dh * pw_col).mean(1),
        dh.std(1),
        (tab[:, :, -1, :8] - tab[:, :, -7, :8]).mean(1),
        tab[:, :, -1, :8].max(1) - tab[:, :, -1, :8].min(1),
    ]

    out = np.concatenate([b.reshape(N,-1) for b in blocs], axis=1).astype(np.float32)
    return out[0] if seul else out


def _feat_knn(tableau_raw):
    seul = (tableau_raw.ndim == 3)
    tab  = _enrich(tableau_raw)
    if seul:
        tab = tab[np.newaxis]
    N     = tab.shape[0]
    paris = tab[:, PARIS_IDX]
    temp   = paris[:, :, 0]
    wspeed = paris[:, :, 2]
    hum    = paris[:, :, 4]
    clouds = paris[:, :, 5]
    wd_sin = paris[:, :, 8]
    wd_cos = paris[:, :, 9]

    if seul:
        wd_mean = float(tab[0, PARIS_IDX, -6:, 3].mean())
        pw = _poids_vent_dynamique(wd_mean)
    else:
        pw = np.stack([
            _poids_vent_dynamique(float(tab[n, PARIS_IDX, -6:, 3].mean()))
            for n in range(N)
        ], axis=0)

    blocs = [
        temp, wspeed, hum, clouds,
        wd_sin, wd_cos,
        paris[:, :, 10], paris[:, :, 11],
        wspeed * wd_sin,
        wspeed * wd_cos,
    ]
    for f in [6, 24]:
        w = temp[:, -f:]
        blocs += [
            w.mean(1, keepdims=True),
            w.std(1, keepdims=True),
            (w.max(1) - w.min(1)).reshape(N,1),
        ]

    for idx in VOISINS_TRIES[:6]:
        v = tab[:, idx]
        if seul:
            w = float(pw[idx])
        else:
            w = pw[:, idx]
        blocs.append((temp - v[:, :, 0]) * w)
        blocs.append(v[:, -1:, 0] * w)

    out = np.concatenate([b.reshape(N,-1) for b in blocs], axis=1).astype(np.float32)
    return out[0] if seul else out


class _ProbCorrector:
    def __init__(self):
        self.p = {}

    def fit(self, y_true, y_pred, nom):
        res = np.asarray(y_true, np.float64) - np.asarray(y_pred, np.float64)
        if nom == "temperature":
            mu, sigma = norm.fit(res)
            self.p[nom] = {"t": "norm", "mu": mu, "sigma": sigma}
        else:
            shift = res.min() - 1e-6
            try:
                a, _, scale = sp_gamma.fit(res - shift, floc=0)
                self.p[nom] = {"t": "gamma", "a": a, "scale": scale, "shift": shift}
            except Exception:
                self.p[nom] = {"t": "med", "v": float(np.median(res))}

    def correct(self, pred, nom):
        if nom not in self.p:
            return pred
        d = self.p[nom]
        if d["t"] == "norm":
            return float(pred) + d["mu"]
        elif d["t"] == "gamma":
            return float(pred) + d["a"] * d["scale"] + d["shift"]
        else:
            return float(pred) + d["v"]


class Agent:
    def __init__(self):
        self.modeles    = {}
        self.scalers    = {}
        self.knns       = {}
        self.correcteur = _ProbCorrector()
        self.entraine   = False

    def train(self, X_train, y_train):
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        f_t   = _feat_temperature(X_train)
        f_vp  = _feat_vent_pluie(X_train)
        f_knn = _feat_knn(X_train)

        def make_stack_t():
            return StackingRegressor(
                estimators=[
                    ("ridge", Pipeline([
                        ("sc", RobustScaler()),
                        ("sel", SelectPercentile(f_regression, percentile=75)),
                        ("m", Ridge(alpha=0.05))
                    ])),
                    ("huber", Pipeline([
                        ("sc", RobustScaler()),
                        ("sel", SelectPercentile(f_regression, percentile=70)),
                        ("m", HuberRegressor(epsilon=1.15, alpha=0.01, max_iter=500))
                    ])),
                    ("elastic", Pipeline([
                        ("sc", RobustScaler()),
                        ("sel", SelectPercentile(f_regression, percentile=65)),
                        ("m", ElasticNet(alpha=0.008, l1_ratio=0.25, max_iter=2000))
                    ])),
                ],
                final_estimator=Ridge(alpha=0.1),
                cv=5, passthrough=False, n_jobs=-1,
            )

        def make_stack_vp():
            return StackingRegressor(
                estimators=[
                    ("ridge", Pipeline([
                        ("sc", RobustScaler()),
                        ("sel", SelectPercentile(f_regression, percentile=70)),
                        ("m", Ridge(alpha=0.1))
                    ])),
                    ("huber", Pipeline([
                        ("sc", RobustScaler()),
                        ("sel", SelectPercentile(f_regression, percentile=65)),
                        ("m", HuberRegressor(epsilon=1.2, alpha=0.02, max_iter=400))
                    ])),
                    ("elastic", Pipeline([
                        ("sc", RobustScaler()),
                        ("sel", SelectPercentile(f_regression, percentile=60)),
                        ("m", ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=2000))
                    ])),
                ],
                final_estimator=Ridge(alpha=0.1),
                cv=5, passthrough=False, n_jobs=-1,
            )

        st = make_stack_t()
        st.fit(f_t, y_train[:, 0])
        self.modeles["temperature"] = st

        sc_knn_t = RobustScaler()
        fk_t = sc_knn_t.fit_transform(f_knn)
        knn_t = KNeighborsRegressor(n_neighbors=15, weights="distance", metric="euclidean", n_jobs=-1)
        knn_t.fit(fk_t, y_train[:, 0])
        self.knns["temperature"]    = knn_t
        self.scalers["knn_t"]       = sc_knn_t

        sc_knn_vp = RobustScaler()
        fk_vp = sc_knn_vp.fit_transform(f_knn)
        self.scalers["knn_vp"] = sc_knn_vp

        for i, nom in enumerate(["wind_speed", "rain"]):
            sv = make_stack_vp()
            sv.fit(f_vp, y_train[:, i+1])
            self.modeles[nom] = sv

            knn_v = KNeighborsRegressor(n_neighbors=15, weights="distance", metric="euclidean", n_jobs=-1)
            knn_v.fit(fk_vp, y_train[:, i+1])
            self.knns[nom] = knn_v

        pred_t = (0.65 * self.modeles["temperature"].predict(f_t)
                + 0.35 * self.knns["temperature"].predict(fk_t))
        self.correcteur.fit(y_train[:, 0], pred_t, "temperature")

        for i, nom in enumerate(["wind_speed", "rain"]):
            pr = self.modeles[nom].predict(f_vp)
            pk = self.knns[nom].predict(fk_vp)
            self.correcteur.fit(y_train[:, i+1], 0.65*pr + 0.35*pk, nom)

        self.entraine = True

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float32)

        if not self.entraine or not self.modeles:
            return self._fallback(X_test)

        f_t   = _feat_temperature(X_test).reshape(1, -1)
        f_vp  = _feat_vent_pluie(X_test).reshape(1, -1)
        f_knn = _feat_knn(X_test).reshape(1, -1)

        fk_t  = self.scalers["knn_t"].transform(f_knn)
        fk_vp = self.scalers["knn_vp"].transform(f_knn)

        pred_t = (0.65 * float(self.modeles["temperature"].predict(f_t)[0])
                + 0.35 * float(self.knns["temperature"].predict(fk_t)[0]))
        pred_t = self.correcteur.correct(pred_t, "temperature")

        res = [pred_t]
        for nom in ["wind_speed", "rain"]:
            p = (0.65 * float(self.modeles[nom].predict(f_vp)[0])
               + 0.35 * float(self.knns[nom].predict(fk_vp)[0]))
            p = self.correcteur.correct(p, nom)
            res.append(max(0.0, p))

        return np.array(res, dtype=np.float32)

    def _fallback(self, X_test):
        tab   = _enrich(X_test)
        paris = tab[PARIS_IDX]
        wd_mean = float(paris[-6:, 3].mean())
        pw    = _poids_vent_dynamique(wd_mean)
        poids = np.linspace(0.3, 1.0, 24)
        poids = poids / poids.sum()
        tend_t = float(paris[-1, 0] - paris[-7, 0])
        tend_w = float(paris[-1, 2] - paris[-7, 2])
        corr_t, corr_w = 0.0, 0.0
        top5 = sorted(VOISINS_TRIES[:5], key=lambda i: pw[i], reverse=True)
        weights_top5 = [0.35, 0.25, 0.15, 0.15, 0.10]
        for k, idx_v in enumerate(top5):
            v = tab[idx_v]
            corr_t += weights_top5[k] * float(v[-1, 0] - v[-7, 0]) * 0.25
            corr_w += weights_top5[k] * float(v[-1, 2] - v[-7, 2]) * 0.25
        return np.array([
            float(np.sum(poids * paris[:, 0])) + 0.35 * tend_t + corr_t,
            max(0.0, float(np.sum(poids * paris[:, 2])) + 0.35 * tend_w + corr_w),
            max(0.0, float(np.sum(poids * paris[:, 1])))
        ], dtype=np.float32)