import pickle
import numpy as np

PARIS_IDX = 16
F_TEMP, F_RAIN, F_WIND, F_WDIR, F_HUM, F_CLOUD, F_VIS, F_SNOW = range(8)

DIST = np.array([
    431, 830, 502, 264, 1027, 469, 781, 411, 440, 479,
    403, 344, 570, 661, 640, 685,   0, 373, 500, 580,
], dtype=np.float32)

NEARBY = np.where((DIST > 0) & (DIST < 500))[0]

_w = 1.0 / (1.0 + DIST / float(np.median(DIST[DIST > 0])))
_w[PARIS_IDX] = 0.0
W_GEO = (_w / _w.sum()).astype(np.float32)

DIST2 = (DIST ** 2 + 1e-8).astype(np.float32)

_coords = np.array([
    [52.37,  4.90], [41.39,  2.17], [52.49, -1.90], [50.85,  4.35],
    [55.68, 12.57], [51.51,  7.47], [53.35, -6.26], [51.23,  6.78],
    [51.46,  7.01], [50.11,  8.68], [50.94,  6.96], [51.51, -0.13],
    [53.48, -2.24], [43.30,  5.37], [45.46,  9.19], [48.14, 11.58],
    [48.85,  2.35], [51.92,  4.48], [48.78,  9.18], [45.07,  7.69],
], dtype=np.float32)

_dx = 2.35 - _coords[:, 1]
_dy = 48.85 - _coords[:, 0]
_n  = np.sqrt(_dx**2 + _dy**2) + 1e-8
DIR_X = (_dx / _n).astype(np.float32)
DIR_Y = (_dy / _n).astype(np.float32)


def _build_features(X):
    X = X[np.newaxis]
    paris = X[:, PARIS_IDX]
    last = paris[:, -1]
    f = []

    f.append(last)

    for fi in [F_TEMP, F_WIND, F_RAIN, F_HUM, F_CLOUD]:
        for w in [3, 6, 12, 24]:
            c = paris[:, -w:, fi]
            f += [c.mean(1, keepdims=True), c.std(1, keepdims=True)]

    for fi in [F_TEMP, F_WIND, F_HUM]:
        for lag in [1, 2, 3, 6, 12, 23]:
            f.append((last[:, fi] - paris[:, -(1 + lag), fi]).reshape(-1, 1))

    f += [
        np.full((1, 1), np.sin(2 * np.pi * 23 / 24), dtype=np.float32),
        np.full((1, 1), np.cos(2 * np.pi * 23 / 24), dtype=np.float32),
    ]

    all_last = X[:, :, -1, :]

    for fi in [F_TEMP, F_WIND, F_RAIN, F_HUM]:
        wm = (all_last[:, :, fi] * W_GEO[None, :]).sum(1, keepdims=True)
        f += [wm, last[:, fi:fi+1] - wm]

    wr = np.deg2rad(paris[:, :, F_WDIR])
    ws = paris[:, :, F_WIND]
    u_p = ws * np.sin(wr)
    v_p = ws * np.cos(wr)

    f += [
        u_p[:, -1:], v_p[:, -1:],
        u_p[:, -6:].mean(1, keepdims=True),
        v_p[:, -6:].mean(1, keepdims=True)
    ]

    wr_all = np.deg2rad(X[:, :, -1, F_WDIR])
    ws_all = X[:, :, -1, F_WIND]
    u_all = ws_all * np.sin(wr_all)
    v_all = ws_all * np.cos(wr_all)

    adv = u_all * DIR_X[None, :] + v_all * DIR_Y[None, :]
    adv_n = adv[:, NEARBY]

    t_all = X[:, :, -1, F_TEMP]
    w_all = X[:, :, -1, F_WIND]
    r_all = X[:, :, -1, F_RAIN]

    f.append((adv_n * (t_all[:, NEARBY] - t_all[:, PARIS_IDX:PARIS_IDX+1])).sum(1, keepdims=True))
    f.append((adv_n * (w_all[:, NEARBY] - w_all[:, PARIS_IDX:PARIS_IDX+1])).sum(1, keepdims=True))
    f.append((adv_n * r_all[:, NEARBY]).sum(1, keepdims=True))

    f.append((paris[:, -1, F_WIND] - 2*paris[:, -2, F_WIND] + paris[:, -3, F_WIND]).reshape(-1, 1))
    f.append((paris[:, -1, F_TEMP] - 2*paris[:, -2, F_TEMP] + paris[:, -3, F_TEMP]).reshape(-1, 1))

    for fi in [F_TEMP, F_HUM, F_WIND]:
        lap = ((all_last[:, NEARBY, fi] - all_last[:, PARIS_IDX:PARIS_IDX+1, fi])
               / DIST2[None, NEARBY]).mean(1, keepdims=True)
        f.append(lap)

    f.append((adv_n / (DIST[NEARBY][None, :] + 1e-8)).sum(1, keepdims=True))

    wsin = np.sin(wr_all[:, NEARBY])
    wcos = np.cos(wr_all[:, NEARBY])
    Rbar = np.sqrt(wsin.mean(1)**2 + wcos.mean(1)**2)
    f.append((1.0 - Rbar).reshape(-1, 1))

    for fi in [F_TEMP, F_HUM]:
        spat = all_last[:, NEARBY, fi].std(1)
        tend = np.abs(paris[:, -1, fi] - paris[:, -7, fi])
        f.append((spat * tend).reshape(-1, 1))

    for fi in [F_TEMP, F_WIND, F_HUM]:
        for w in [6, 12, 24]:
            c = paris[:, -w:, fi]
            t = np.arange(w, dtype=np.float32)
            tm = t.mean()
            num = ((c - c.mean(1, keepdims=True)) * (t - tm)[None, :]).sum(1)
            den = float(((t - tm)**2).sum()) + 1e-8
            f.append((num / den).reshape(-1, 1))

    rain_proxy = (np.maximum(0.0, last[:, F_HUM] - 80.0) * last[:, F_CLOUD] / 100.0).reshape(-1, 1)
    f.append(rain_proxy)

    t_residual = (last[:, F_TEMP] - paris[:, :, F_TEMP].mean(1)).reshape(-1, 1)
    f.append(t_residual)

    out = np.hstack(f)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class Agent:
    def __init__(self):
        self.model_temp = None
        self.model_wind = None
        self.model_rain = None

    def _load_models(self):
        if self.model_temp is None:
            with open("model_temp.pkl", "rb") as f:
                self.model_temp = pickle.load(f)
            with open("model_wind.pkl", "rb") as f:
                self.model_wind = pickle.load(f)
            with open("model_rain.pkl", "rb") as f:
                self.model_rain = pickle.load(f)

    def predict(self, X_test):
        self._load_models()

        last = X_test[PARIS_IDX, -1, :]
        feats = _build_features(X_test)

        paris_temp = X_test[PARIS_IDX, :, F_TEMP]
        t = np.arange(6, dtype=np.float32)
        tm = t.mean()
        c = paris_temp[-6:]

        slope_6h = float(((c - c.mean()) * (t - tm)).sum() / (((t - tm)**2).sum() + 1e-8))
        inertia_factor = 1.0 + np.clip(0.09 * slope_6h, -0.09, 0.09)

        pred_temp = (float(last[F_TEMP]) + float(self.model_temp.predict(feats)[0])) * inertia_factor
        pred_wind = max(0.0, float(last[F_WIND]) + float(self.model_wind.predict(feats)[0]))
        pred_rain = max(0.0, float(self.model_rain.predict(feats)[0]))

        return np.array([pred_temp, pred_wind, pred_rain], dtype=np.float32)