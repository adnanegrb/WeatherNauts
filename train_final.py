import os
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
DATA_DIR   = r"C:\Users\Massy\Documents\Optuna"
OUTPUT_DIR = r"C:\Users\Massy\Documents\Optuna\models_final"
HORIZON    = 6
N_TRIALS   = 50
PARIS_IDX  = 16
F_TEMP, F_RAIN, F_WIND, F_WDIR, F_HUM, F_CLOUD, F_VIS, F_SNOW = range(8)
CITIES = [
    "Amsterdam", "Barcelona", "Birmingham", "Brussels", "Copenhagen",
    "Dortmund", "Dublin", "Düsseldorf", "Essen", "Frankfurt am Main",
    "Köln", "London", "Manchester", "Marseille", "Milan", "Munich",
    "Paris", "Rotterdam", "Stuttgart", "Turin",
]
FEAT_COLS = [
    "temperature", "rain", "wind_speed", "wind_direction",
    "humidity", "clouds", "visibility", "snow",
]
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
_dx   =  2.35 - _coords[:, 1]
_dy   = 48.85 - _coords[:, 0]
_n    = np.sqrt(_dx**2 + _dy**2) + 1e-8
DIR_X = (_dx / _n).astype(np.float32)
DIR_Y = (_dy / _n).astype(np.float32)
SIGMA_T = 7.49
SIGMA_W = 5.05
SIGMA_R = 0.40
def load_years(years) -> pd.DataFrame:
    dfs = []
    for y in years:
        path = os.path.join(DATA_DIR, f"weather_{y}.csv")
        print(f"Chargement {path} ...")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df[df["city_name"].isin(CITIES)]
        dfs.append(df)
        print(f"{len(df)} lignes, {df['city_name'].nunique()} villes")
    return pd.concat(dfs, ignore_index=True)
def build_dataset(df: pd.DataFrame, label: str = "") -> tuple:
    pivot = (
        df.pivot_table(index="timestamp", columns="city_name", values=FEAT_COLS)
        .sort_index()
    )
    pivot = pivot.ffill().bfill()
    timestamps = pivot.index
    T = len(timestamps)
    print(f"{label} : {T} timesteps, {timestamps[0]} a {timestamps[-1]}")
    X_list, y_list = [], []
    skipped  = 0
    total    = T - 23 - HORIZON
    log_step = max(1, total // 10)
    for t in range(23, T - HORIZON):
        try:
            X = np.zeros((20, 24, 8), dtype=np.float32)
            for ci, city in enumerate(CITIES):
                for fi, feat in enumerate(FEAT_COLS):
                    try:
                        vals = pivot[(feat, city)].iloc[t - 23 : t + 1].values
                        if len(vals) == 24:
                            X[ci, :, fi] = vals
                    except KeyError:
                        pass
            y = np.array([
                float(pivot[("temperature", "Paris")].iloc[t + HORIZON]),
                float(pivot[("wind_speed",  "Paris")].iloc[t + HORIZON]),
                float(pivot[("rain",        "Paris")].iloc[t + HORIZON]),
            ], dtype=np.float32)
            if np.isnan(X).mean() > 0.05 or np.isnan(y).any():
                skipped += 1
                continue
            X_list.append(X)
            y_list.append(y)
        except Exception:
            skipped += 1
        done = t - 23
        if done % log_step == 0:
            pct = int(100 * done / total)
            print(f"{pct}%  {len(X_list)} ok, {skipped} ignores", end="\r")
    print()
    y_arr = np.array(y_list, dtype=np.float32)
    print(f"{len(X_list)} samples valides, {skipped} ignores")
    print(f"y_temp : mean={y_arr[:,0].mean():.2f}  std={y_arr[:,0].std():.2f}")
    print(f"y_wind : mean={y_arr[:,1].mean():.2f}  std={y_arr[:,1].std():.2f}")
    print(f"y_rain : mean={y_arr[:,2].mean():.4f}  zeros={100*(y_arr[:,2]==0).mean():.1f}%")
    return np.array(X_list, dtype=np.float32), y_arr
def build_features(X: np.ndarray) -> np.ndarray:
    N     = X.shape[0]
    paris = X[:, PARIS_IDX]
    last  = paris[:, -1]
    f     = []
    print("Bloc 1 : valeurs courantes")
    f.append(last)
    print("Bloc 2 : stats fenetrees")
    for fi in [F_TEMP, F_WIND, F_RAIN, F_HUM, F_CLOUD]:
        for w in [3, 6, 12, 24]:
            c = paris[:, -w:, fi]
            f += [c.mean(1, keepdims=True), c.std(1, keepdims=True)]
    print("Bloc 3 : lags temporels")
    for fi in [F_TEMP, F_WIND, F_HUM]:
        for lag in [1, 2, 3, 6, 12, 23]:
            f.append((last[:, fi] - paris[:, -(1 + lag), fi]).reshape(-1, 1))
    print("Bloc 4 : encodage cyclique heure")
    f += [
        np.full((N, 1), np.sin(2 * np.pi * 23 / 24), dtype=np.float32),
        np.full((N, 1), np.cos(2 * np.pi * 23 / 24), dtype=np.float32),
    ]
    print("Bloc 5 : voisins geografiques ponderes")
    all_last = X[:, :, -1, :]
    for fi in [F_TEMP, F_WIND, F_RAIN, F_HUM]:
        wm = (all_last[:, :, fi] * W_GEO[None, :]).sum(1, keepdims=True)
        f += [wm, last[:, fi : fi + 1] - wm]
    print("Bloc 6 : composantes vectorielles vent")
    wr  = np.deg2rad(paris[:, :, F_WDIR])
    ws  = paris[:, :, F_WIND]
    u_p = ws * np.sin(wr)
    v_p = ws * np.cos(wr)
    f += [
        u_p[:, -1:],
        v_p[:, -1:],
        u_p[:, -6:].mean(1, keepdims=True),
        v_p[:, -6:].mean(1, keepdims=True),
    ]
    print("Bloc 7 : advection thermique")
    wr_all = np.deg2rad(X[:, :, -1, F_WDIR])
    ws_all = X[:, :, -1, F_WIND]
    u_all  = ws_all * np.sin(wr_all)
    v_all  = ws_all * np.cos(wr_all)
    adv    = u_all * DIR_X[None, :] + v_all * DIR_Y[None, :]
    adv_n  = adv[:, NEARBY]
    t_all = X[:, :, -1, F_TEMP]
    w_all = X[:, :, -1, F_WIND]
    r_all = X[:, :, -1, F_RAIN]
    f.append((adv_n * (t_all[:, NEARBY] - t_all[:, PARIS_IDX : PARIS_IDX + 1])).sum(1, keepdims=True))
    f.append((adv_n * (w_all[:, NEARBY] - w_all[:, PARIS_IDX : PARIS_IDX + 1])).sum(1, keepdims=True))
    f.append((adv_n *  r_all[:, NEARBY]).sum(1, keepdims=True))
    print("Bloc 8 : acceleration vent et temperature")
    f.append((paris[:, -1, F_WIND] - 2*paris[:, -2, F_WIND] + paris[:, -3, F_WIND]).reshape(-1, 1))
    f.append((paris[:, -1, F_TEMP] - 2*paris[:, -2, F_TEMP] + paris[:, -3, F_TEMP]).reshape(-1, 1))
    print("Bloc 9 : laplacien spatial")
    for fi in [F_TEMP, F_HUM, F_WIND]:
        lap = (
            (all_last[:, NEARBY, fi] - all_last[:, PARIS_IDX : PARIS_IDX + 1, fi])
            / DIST2[None, NEARBY]
        ).mean(1, keepdims=True)
        f.append(lap)
    print("Bloc 10 : convergence et detecteur frontal")
    f.append((adv_n / (DIST[NEARBY][None, :] + 1e-8)).sum(1, keepdims=True))
    wsin = np.sin(wr_all[:, NEARBY])
    wcos = np.cos(wr_all[:, NEARBY])
    Rbar = np.sqrt(wsin.mean(1)**2 + wcos.mean(1)**2)
    f.append((1.0 - Rbar).reshape(-1, 1))
    for fi in [F_TEMP, F_HUM]:
        spat = all_last[:, NEARBY, fi].std(1)
        tend = np.abs(paris[:, -1, fi] - paris[:, -7, fi])
        f.append((spat * tend).reshape(-1, 1))
    print("Bloc 11 : pentes lineaires")
    for fi in [F_TEMP, F_WIND, F_HUM]:
        for w in [6, 12, 24]:
            c   = paris[:, -w:, fi]
            t   = np.arange(w, dtype=np.float32)
            tm  = t.mean()
            num = ((c - c.mean(1, keepdims=True)) * (t - tm)[None, :]).sum(1)
            den = float(((t - tm)**2).sum()) + 1e-8
            f.append((num / den).reshape(-1, 1))
    print("Bloc 12 : rain_proxy et T_residual")
    hum_last   = last[:, F_HUM]
    cloud_last = last[:, F_CLOUD]
    rain_proxy = (np.maximum(0.0, hum_last - 80.0) * cloud_last / 100.0).reshape(-1, 1)
    f.append(rain_proxy)
    t_mean_24h = paris[:, :, F_TEMP].mean(1)
    t_residual = (last[:, F_TEMP] - t_mean_24h).reshape(-1, 1)
    f.append(t_residual)
    out = np.hstack(f)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"Features OK : shape {out.shape}")
    return out
def optimize(name: str, Ft, yt, Fv, yv, n_trials: int):
    import time
    history  = []
    best_mae = float("inf")
    best_mdl = [None]
    t_start  = time.time()
    loss_choices = {
        "temp": ["squared_error", "absolute_error", "quantile"],
        "wind": ["squared_error", "absolute_error", "quantile"],
        "rain": ["squared_error", "absolute_error", "poisson"],
    }
    def objective(trial):
        nonlocal best_mae
        params = dict(
            loss              = trial.suggest_categorical("loss", loss_choices[name]),
            learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            max_iter          = trial.suggest_int("max_iter", 200, 2000),
            max_leaf_nodes    = trial.suggest_int("max_leaf_nodes", 15, 255),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 5, 50),
            l2_regularization = trial.suggest_float("l2_regularization", 1e-4, 5.0, log=True),
            max_bins          = trial.suggest_int("max_bins", 64, 255),
            early_stopping    = True,
            n_iter_no_change  = 20,
            validation_fraction = 0.1,
            random_state      = 42,
        )
        if params["loss"] == "quantile":
            params["quantile"] = trial.suggest_float("quantile", 0.4, 0.6)
        model = HistGradientBoostingRegressor(**params)
        model.fit(Ft, yt)
        pred = model.predict(Fv)
        if name in ["wind", "rain"]:
            pred = np.maximum(0.0, pred)
        mae = float(mean_absolute_error(yv, pred))
        history.append(mae)
        if len(history) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"{name} : {len(history)}/{n_trials} essais, best MAE = {best_mae:.4f}, {elapsed:.0f}s")
        if mae < best_mae:
            best_mae    = mae
            best_mdl[0] = model
            elapsed = time.time() - t_start
            print(f"{name} : essai {len(history)}/{n_trials}, nouvelle meilleure MAE = {mae:.4f} ({elapsed:.0f}s)")
        return mae
    print(f"Optuna {name} : {n_trials} essais")
    print(f"Train : {len(Ft)} samples, Val : {len(Fv)} samples")
    print(f"Target : mean={yt.mean():.4f}  std={yt.std():.4f}  min={yt.min():.4f}  max={yt.max():.4f}")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"Meilleure MAE {name} : {best_mae:.4f}")
    print(f"Meilleurs params {name} : {study.best_params}")
    return study.best_params, history, best_mdl[0]
def plot_optuna_histories(histories: dict, output_path: str):
    palette = {"temp": "#FF6B6B", "wind": "#4ECDC4", "rain": "#A8E6CF"}
    labels  = {"temp": "Temperature (C)", "wind": "Vent (km/h)", "rain": "Pluie (mm)"}
    bg_dark = "#0D1117"
    bg_card = "#161B22"
    text_c  = "#E6EDF3"
    grid_c  = "#21262D"
    sns.set_theme(style="dark")
    plt.rcParams.update({
        "figure.facecolor": bg_dark,
        "axes.facecolor":   bg_card,
        "axes.edgecolor":   grid_c,
        "axes.labelcolor":  text_c,
        "xtick.color":      text_c,
        "ytick.color":      text_c,
        "text.color":       text_c,
        "grid.color":       grid_c,
        "grid.linewidth":   0.6,
        "font.family":      "monospace",
    })
    fig = plt.figure(figsize=(20, 7), facecolor=bg_dark)
    fig.suptitle(
        "WeatherNauts - Optimisation Optuna - MAE par essai",
        fontsize=16, fontweight="bold", color=text_c, y=1.01,
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    for idx, (name, hist) in enumerate(histories.items()):
        ax       = fig.add_subplot(gs[idx])
        color    = palette[name]
        x        = np.arange(1, len(hist) + 1)
        best     = np.minimum.accumulate(hist)
        best_idx = int(np.argmin(hist))
        ax.fill_between(x, hist, best, alpha=0.12, color=color)
        ax.plot(x, hist, color=color, alpha=0.35, linewidth=1.0,
                linestyle="--", label="MAE essai")
        ax.plot(x, best, color=color, linewidth=2.5, label="Best cumulatif")
        ax.scatter(
            best_idx + 1, hist[best_idx],
            color=color, s=160, zorder=10,
            edgecolors="white", linewidths=1.5,
            label=f"Best = {hist[best_idx]:.4f}",
        )
        ax.axhline(hist[best_idx], color=color, linestyle=":", alpha=0.5, linewidth=1.2)
        ax.annotate(
            f" {hist[best_idx]:.4f}",
            xy=(best_idx + 1, hist[best_idx]),
            xytext=(best_idx + 3, hist[best_idx] * 1.015),
            color=color, fontsize=9,
        )
        ax.set_title(labels[name], fontsize=13, fontweight="bold", color=text_c, pad=12)
        ax.set_xlabel("Essai Optuna", fontsize=10, color=text_c)
        ax.set_ylabel("MAE", fontsize=10, color=text_c)
        ax.grid(True, alpha=0.4)
        ax.tick_params(colors=text_c)
        ax.legend(fontsize=8.5, facecolor=bg_dark, edgecolor=grid_c, labelcolor=text_c)
        ax.text(
            0.97, 0.97, f"n={len(hist)}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=color,
            bbox=dict(facecolor=bg_dark, edgecolor=color,
                      boxstyle="round,pad=0.3", alpha=0.8),
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=bg_dark)
    plt.show()
    print(f"Graphe sauvegarde : {output_path}")
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("ETAPE 1 - Chargement des donnees")
    print("Train 2020 a 2025 ...")
    df_train = load_years(range(2020, 2026))
    print("Val 2026 ...")
    val_path = os.path.join(DATA_DIR, "weather_2026.csv")
    if os.path.exists(val_path):
        df_val = pd.read_csv(val_path, parse_dates=["timestamp"])
        df_val = df_val[df_val["city_name"].isin(CITIES)]
        print(f"2026 charge : {len(df_val)} lignes")
    else:
        print("weather_2026.csv introuvable, utilisation des 20% derniers de 2025")
        cut      = int(len(df_train) * 0.8)
        df_val   = df_train.iloc[cut:].copy()
        df_train = df_train.iloc[:cut].copy()
    print("ETAPE 2 - Construction des datasets")
    print("Train 2020-2025 ...")
    X_train, y_train = build_dataset(df_train, "TRAIN")
    print("Val 2026 ...")
    X_val, y_val = build_dataset(df_val, "VAL")
    print("ETAPE 3 - Feature Engineering")
    print("Features train ...")
    F_train = build_features(X_train)
    print(f"{F_train.shape[1]} features, {F_train.shape[0]} samples")
    print("Features val ...")
    F_val = build_features(X_val)
    print(f"{F_val.shape[1]} features, {F_val.shape[0]} samples")
    last_tr = X_train[:, PARIS_IDX, -1, :]
    last_vl = X_val[:,   PARIS_IDX, -1, :]
    dt_temp = y_train[:, 0] - last_tr[:, F_TEMP]
    dt_wind = y_train[:, 1] - last_tr[:, F_WIND]
    rain_tr = y_train[:, 2]
    dv_temp = y_val[:, 0] - last_vl[:, F_TEMP]
    dv_wind = y_val[:, 1] - last_vl[:, F_WIND]
    rain_vl = y_val[:, 2]
    print(f"Delta temp : mean={dt_temp.mean():.4f}  std={dt_temp.std():.4f}")
    print(f"Delta wind : mean={dt_wind.mean():.4f}  std={dt_wind.std():.4f}")
    print(f"Rain : mean={rain_tr.mean():.4f}  zeros={100*(rain_tr==0).mean():.1f}%")
    print("ETAPE 4 - Optimisation Optuna")
    histories = {}
    bp_temp, hist_temp, _ = optimize("temp", F_train, dt_temp, F_val, dv_temp, N_TRIALS)
    histories["temp"] = hist_temp
    bp_wind, hist_wind, _ = optimize("wind", F_train, dt_wind, F_val, dv_wind, N_TRIALS)
    histories["wind"] = hist_wind
    bp_rain, hist_rain, _ = optimize("rain", F_train, rain_tr, F_val, rain_vl, N_TRIALS)
    histories["rain"] = hist_rain
    print("ETAPE 5 - Graphes Seaborn")
    plot_path = os.path.join(OUTPUT_DIR, "optuna_history_final.png")
    plot_optuna_histories(histories, plot_path)
    print("ETAPE 6 - Enrichissement : ajout 2026 au train")
    print("Concatenation train + 2026 ...")
    df_full = pd.concat([df_train, df_val], ignore_index=True)
    print(f"Total lignes : {len(df_full)}")
    print("Construction dataset enrichi ...")
    X_full, y_full = build_dataset(df_full, "FULL 2020-2026")
    print("Features enrichies ...")
    F_full  = build_features(X_full)
    last_fl = X_full[:, PARIS_IDX, -1, :]
    dt_temp_full = y_full[:, 0] - last_fl[:, F_TEMP]
    dt_wind_full = y_full[:, 1] - last_fl[:, F_WIND]
    rain_full    = y_full[:, 2]
    print(f"{F_full.shape[1]} features, {F_full.shape[0]} samples")
    print("ETAPE 7 - Entrainement final et sauvegarde")
    print(f"Temperature params : {bp_temp}")
    print(f"Entrainement sur {len(F_full)} samples ...")
    m_temp = HistGradientBoostingRegressor(**bp_temp)
    m_temp.fit(F_full, dt_temp_full)
    path = os.path.join(OUTPUT_DIR, "model_temp.pkl")
    pickle.dump(m_temp, open(path, "wb"))
    print(f"model_temp.pkl sauvegarde ({os.path.getsize(path)/1024:.0f} KB)")
    print(f"Vent params : {bp_wind}")
    print(f"Entrainement sur {len(F_full)} samples ...")
    m_wind = HistGradientBoostingRegressor(**bp_wind)
    m_wind.fit(F_full, dt_wind_full)
    path = os.path.join(OUTPUT_DIR, "model_wind.pkl")
    pickle.dump(m_wind, open(path, "wb"))
    print(f"model_wind.pkl sauvegarde ({os.path.getsize(path)/1024:.0f} KB)")
    print(f"Pluie params : {bp_rain}")
    print(f"Entrainement sur {len(F_full)} samples ...")
    m_rain = HistGradientBoostingRegressor(**bp_rain)
    m_rain.fit(F_full, rain_full)
    path = os.path.join(OUTPUT_DIR, "model_rain.pkl")
    pickle.dump(m_rain, open(path, "wb"))
    print(f"model_rain.pkl sauvegarde ({os.path.getsize(path)/1024:.0f} KB)")
    print("ETAPE 8 - Evaluation finale sur 2026")
    pred_temp = last_vl[:, F_TEMP] + m_temp.predict(F_val)
    pred_wind = np.maximum(0.0, last_vl[:, F_WIND] + m_wind.predict(F_val))
    pred_rain = np.maximum(0.0, m_rain.predict(F_val))
    mae_t = mean_absolute_error(y_val[:, 0], pred_temp)
    mae_w = mean_absolute_error(y_val[:, 1], pred_wind)
    mae_r = mean_absolute_error(y_val[:, 2], pred_rain)
    score = -(mae_t / SIGMA_T + mae_w / SIGMA_W + mae_r / SIGMA_R) / 3.0
    print(f"MAE Temperature : {mae_t:.4f} C")
    print(f"MAE Vent        : {mae_w:.4f} km/h")
    print(f"MAE Pluie       : {mae_r:.4f} mm")
    print(f"Score ML Arena  : {score:+.4f}")
    print(f"Fichiers sauvegardes dans : {OUTPUT_DIR}")
    print("train_final.py termine")
if __name__ == "__main__":
    main()