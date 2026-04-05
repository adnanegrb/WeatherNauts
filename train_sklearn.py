"""
train_sklearn.py
─────────────────────────────────────────────────────────────────
Script d'entraînement offline complet — 100% sklearn, zéro LightGBM :
  1. Charge weather_2020.csv → weather_2025.csv (train) + weather_2026.csv (val)
  2. Construit X_train, y_train, X_val, y_val
  3. Optuna x 100 essais par variable (temp, wind, rain)
  4. Réentraîne les modèles finaux avec les meilleurs params
  5. Sauvegarde model_temp.pkl, model_wind.pkl, model_rain.pkl, rain_clf.pkl
  6. Génère 3 graphes Seaborn

Usage :
  python train_sklearn.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Massy\Documents\Optuna"
OUTPUT_DIR = r"C:\Users\Massy\Documents\Optuna\models_sklearn"
HORIZON    = 6
N_TRIALS   = 100

# ── Constants ──────────────────────────────────────────────────────────────────
PARIS = 16
F_TEMP, F_RAIN, F_WIND, F_WDIR, F_HUM, F_CLOUD, F_VIS, F_SNOW = range(8)

CITIES = [
    "Amsterdam", "Barcelona", "Birmingham", "Brussels", "Copenhagen",
    "Dortmund", "Dublin", "Düsseldorf", "Essen", "Frankfurt am Main",
    "Köln", "London", "Manchester", "Marseille", "Milan", "Munich",
    "Paris", "Rotterdam", "Stuttgart", "Turin"
]

FEAT_COLS = ["temperature", "rain", "wind_speed", "wind_direction",
             "humidity", "clouds", "visibility", "snow"]

DIST = np.array([431, 830, 502, 264, 1027, 469, 781, 411, 440, 479,
                 403, 344, 570, 661, 640, 685, 0,   373, 500, 580], dtype=np.float32)
NEARBY = np.where((DIST > 0) & (DIST < 500))[0]
_w     = 1.0 / (1.0 + DIST / float(np.median(DIST[DIST > 0])))
_w[PARIS] = 0.0
W_GEO  = (_w / _w.sum()).astype(np.float32)
DIST2  = (DIST ** 2 + 1e-8).astype(np.float32)

_coords = np.array([
    [52.37, 4.90], [41.39, 2.17], [52.49, -1.90], [50.85, 4.35],
    [55.68,12.57], [51.51, 7.47], [53.35, -6.26], [51.23, 6.78],
    [51.46, 7.01], [50.11, 8.68], [50.94, 6.96],  [51.51,-0.13],
    [53.48,-2.24], [43.30, 5.37], [45.46, 9.19],  [48.14,11.58],
    [48.85, 2.35], [51.92, 4.48], [48.78, 9.18],  [45.07, 7.69]
], dtype=np.float32)
_dx   = 2.35 - _coords[:, 1]
_dy   = 48.85 - _coords[:, 0]
_n    = np.sqrt(_dx**2 + _dy**2) + 1e-8
DIR_X = (_dx / _n).astype(np.float32)
DIR_Y = (_dy / _n).astype(np.float32)


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_years(years) -> pd.DataFrame:
    dfs = []
    for y in years:
        path = os.path.join(DATA_DIR, f"weather_{y}.csv")
        print(f"  Chargement {path}...")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df[df["city_name"].isin(CITIES)]
        dfs.append(df)
        print(f"    OK {y} — {len(df):,} lignes | {df['city_name'].nunique()} villes")
    return pd.concat(dfs, ignore_index=True)


def build_dataset(df: pd.DataFrame) -> tuple:
    pivot = df.pivot_table(
        index="timestamp", columns="city_name", values=FEAT_COLS
    ).sort_index()
    pivot = pivot.ffill().bfill()

    timestamps = pivot.index
    T = len(timestamps)
    print(f"  {T:,} timesteps | {timestamps[0]} -> {timestamps[-1]}")
    print(f"  Villes : {list(pivot['temperature'].columns)}")
    print(f"  Construction fenêtres 24h -> T+{HORIZON}h...")

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
                        vals = pivot[(feat, city)].iloc[t-23:t+1].values
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
            print(f"    {pct:3d}% — {len(X_list):,} samples | {skipped} ignores", end="\r")

    print()
    y_arr = np.array(y_list, dtype=np.float32)
    print(f"  OK {len(X_list):,} samples valides | {skipped} ignores")
    print(f"  Shape X : ({len(X_list)}, 20, 24, 8)")
    print(f"  y_temp  : mean={y_arr[:,0].mean():.2f}C  std={y_arr[:,0].std():.2f}")
    print(f"  y_wind  : mean={y_arr[:,1].mean():.2f}km/h  std={y_arr[:,1].std():.2f}")
    print(f"  y_rain  : mean={y_arr[:,2].mean():.4f}mm  zeros={100*(y_arr[:,2]==0).mean():.1f}%")
    return np.array(X_list, dtype=np.float32), y_arr


# ── Feature Engineering ────────────────────────────────────────────────────────
def build_features(X: np.ndarray) -> np.ndarray:
    N     = X.shape[0]
    paris = X[:, PARIS]
    last  = paris[:, -1]
    f     = []

    print(f"    calcul features sur {N:,} samples...")

    f.append(last)

    for fi in [F_TEMP, F_WIND, F_RAIN, F_HUM, F_CLOUD]:
        for w in [3, 6, 12, 24]:
            c = paris[:, -w:, fi]
            f += [c.mean(1, keepdims=True), c.std(1, keepdims=True)]

    for fi in [F_TEMP, F_WIND, F_HUM]:
        for lag in [1, 2, 3, 6, 12, 23]:
            f.append((last[:, fi] - paris[:, -(1+lag), fi]).reshape(-1, 1))

    f += [np.full((N, 1), np.sin(2*np.pi*23/24), dtype=np.float32),
          np.full((N, 1), np.cos(2*np.pi*23/24), dtype=np.float32)]

    all_last = X[:, :, -1, :]
    for fi in [F_TEMP, F_WIND, F_RAIN, F_HUM]:
        wm = (all_last[:, :, fi] * W_GEO[None, :]).sum(1, keepdims=True)
        f += [wm, last[:, fi:fi+1] - wm]

    wr  = np.deg2rad(paris[:, :, F_WDIR])
    ws  = paris[:, :, F_WIND]
    u_p = ws * np.sin(wr)
    v_p = ws * np.cos(wr)
    f  += [u_p[:, -1:], v_p[:, -1:],
           u_p[:, -6:].mean(1, keepdims=True),
           v_p[:, -6:].mean(1, keepdims=True)]

    wr_all = np.deg2rad(X[:, :, -1, F_WDIR])
    ws_all = X[:, :, -1, F_WIND]
    u_all  = ws_all * np.sin(wr_all)
    v_all  = ws_all * np.cos(wr_all)
    adv    = u_all * DIR_X[None, :] + v_all * DIR_Y[None, :]
    adv_n  = adv[:, NEARBY]
    t_all  = X[:, :, -1, F_TEMP]
    w_all  = X[:, :, -1, F_WIND]
    r_all  = X[:, :, -1, F_RAIN]
    f.append((adv_n * (t_all[:, NEARBY] - t_all[:, PARIS:PARIS+1])).sum(1, keepdims=True))
    f.append((adv_n * (w_all[:, NEARBY] - w_all[:, PARIS:PARIS+1])).sum(1, keepdims=True))
    f.append((adv_n * r_all[:, NEARBY]).sum(1, keepdims=True))

    f.append((paris[:, -1, F_WIND] - 2*paris[:, -2, F_WIND] + paris[:, -3, F_WIND]).reshape(-1, 1))
    f.append((paris[:, -1, F_TEMP] - 2*paris[:, -2, F_TEMP] + paris[:, -3, F_TEMP]).reshape(-1, 1))

    for fi in [F_TEMP, F_HUM, F_WIND]:
        lap = ((all_last[:, NEARBY, fi] - all_last[:, PARIS:PARIS+1, fi]) / DIST2[None, NEARBY]).mean(1, keepdims=True)
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
            c   = paris[:, -w:, fi]
            t   = np.arange(w, dtype=np.float32)
            tm  = t.mean()
            num = ((c - c.mean(1, keepdims=True)) * (t - tm)[None, :]).sum(1)
            den = float(((t - tm)**2).sum()) + 1e-8
            f.append((num / den).reshape(-1, 1))

    out = np.hstack(f)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"    -> {out.shape[1]} features | NaN restants : {np.isnan(out).sum()}")
    return out


# ── Optuna ─────────────────────────────────────────────────────────────────────
def optimize(name: str, Ft, yt, Fv, yv, n_trials: int) -> tuple:
    history  = []
    best_mae = float("inf")
    best_mdl = [None]

    def objective(trial):
        nonlocal best_mae

        loss_choices = {
            "temp": ["squared_error", "absolute_error", "huber"],
            "wind": ["squared_error", "absolute_error", "huber"],
            "rain": ["squared_error", "absolute_error", "poisson"],
        }

        params = dict(
            loss             = trial.suggest_categorical("loss", loss_choices[name]),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            max_iter         = trial.suggest_int("max_iter", 200, 2000),
            max_leaf_nodes   = trial.suggest_int("max_leaf_nodes", 15, 255),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50),
            l2_regularization= trial.suggest_float("l2_regularization", 1e-4, 5.0, log=True),
            max_bins         = trial.suggest_int("max_bins", 64, 255),
            early_stopping   = True,
            n_iter_no_change = 20,
            validation_fraction = 0.1,
            random_state     = 42,
        )

        if params["loss"] == "huber":
            params["quantile"] = trial.suggest_float("quantile", 0.6, 0.99)

        model = HistGradientBoostingRegressor(**params)
        model.fit(Ft, yt)
        pred = np.maximum(0, model.predict(Fv)) if name in ["wind", "rain"] else model.predict(Fv)
        mae  = float(mean_absolute_error(yv, pred))
        history.append(mae)

        if mae < best_mae:
            best_mae    = mae
            best_mdl[0] = model
            print(f"    [{name}] #{len(history):3d} — MAE {mae:.4f} ✅")

        return mae

    print(f"\n[Optuna] {name} — {n_trials} essais...")
    print(f"  Train : {len(Ft):,} samples | Val : {len(Fv):,} samples")
    print(f"  Target : mean={yt.mean():.4f}  std={yt.std():.4f}  min={yt.min():.4f}  max={yt.max():.4f}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"\n  Best MAE    : {best_mae:.4f}")
    print(f"  Best params : {study.best_params}")
    return study.best_params, history, best_mdl[0]


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_histories(histories: dict):
    sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cfg = {
        "temp": ("Température", "#E74C3C", "°C"),
        "wind": ("Vent",        "#3498DB", "km/h"),
        "rain": ("Pluie",       "#2ECC71", "mm"),
    }

    for ax, (name, hist) in zip(axes, histories.items()):
        title, color, unit = cfg[name]
        x        = np.arange(1, len(hist)+1)
        best     = np.minimum.accumulate(hist)
        best_idx = int(np.argmin(hist))

        sns.lineplot(x=x, y=hist, ax=ax, color=color,
                     alpha=0.35, linewidth=1.2, label="MAE essai")
        sns.lineplot(x=x, y=best, ax=ax, color=color,
                     linewidth=2.5, label="Meilleur cumulatif")
        ax.scatter(best_idx+1, hist[best_idx], color=color,
                   s=130, zorder=5, label=f"Best={hist[best_idx]:.4f}")
        ax.axhline(hist[best_idx], color=color, linestyle="--", alpha=0.4)

        ax.set_title(f"{title} ({unit})", fontweight="bold", pad=10)
        ax.set_xlabel("Essai Optuna")
        ax.set_ylabel(f"MAE ({unit})")
        ax.legend(fontsize=9)

    plt.suptitle("Optimisation Optuna (sklearn) — Évolution MAE par variable",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "optuna_history_sklearn.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[Plot] -> {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Chargement ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ÉTAPE 1 — Chargement des données")
    print("="*55)
    print("\n[Train] 2020 -> 2025...")
    df_train = load_years(range(2020, 2026))

    print("\n[Val] 2026...")
    val_path = os.path.join(DATA_DIR, "weather_2026.csv")
    if os.path.exists(val_path):
        df_val = pd.read_csv(val_path, parse_dates=["timestamp"])
        df_val = df_val[df_val["city_name"].isin(CITIES)]
        print(f"  OK 2026 — {len(df_val):,} lignes")
    else:
        print("  weather_2026.csv introuvable — utilise 20% derniers de 2025")
        cut      = int(len(df_train) * 0.8)
        df_val   = df_train.iloc[cut:].copy()
        df_train = df_train.iloc[:cut].copy()

    # ── 2. Datasets ────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ÉTAPE 2 — Construction des datasets")
    print("="*55)
    print("\n[Train]")
    X_train, y_train = build_dataset(df_train)
    print("\n[Val]")
    X_val,   y_val   = build_dataset(df_val)

    # ── 3. Features ────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ÉTAPE 3 — Feature Engineering")
    print("="*55)
    print("  Train...")
    F_train = build_features(X_train)
    print("  Val...")
    F_val   = build_features(X_val)

    last_tr = X_train[:, PARIS, -1, :]
    last_vl = X_val[:,   PARIS, -1, :]

    dt_temp = y_train[:, 0] - last_tr[:, F_TEMP]
    dt_wind = y_train[:, 1] - last_tr[:, F_WIND]
    rain_tr = y_train[:, 2]

    dv_temp = y_val[:, 0] - last_vl[:, F_TEMP]
    dv_wind = y_val[:, 1] - last_vl[:, F_WIND]
    rain_vl = y_val[:, 2]

    wet_tr = rain_tr > 0.05
    wet_vl = rain_vl > 0.05

    print(f"\n  Delta temp  : mean={dt_temp.mean():.4f}  std={dt_temp.std():.4f}")
    print(f"  Delta wind  : mean={dt_wind.mean():.4f}  std={dt_wind.std():.4f}")
    print(f"  Rain positif: {wet_tr.sum():,}/{len(wet_tr):,} ({100*wet_tr.mean():.1f}%)")

    # ── 4. Optuna ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ÉTAPE 4 — Optimisation Optuna (sklearn)")
    print("="*55)

    histories = {}

    bp_temp, hist_temp, _ = optimize("temp", F_train, dt_temp, F_val, dv_temp, N_TRIALS)
    histories["temp"] = hist_temp

    bp_wind, hist_wind, _ = optimize("wind", F_train, dt_wind, F_val, dv_wind, N_TRIALS)
    histories["wind"] = hist_wind

    bp_rain, hist_rain, _ = optimize(
        "rain",
        F_train[wet_tr], rain_tr[wet_tr],
        F_val[wet_vl],   rain_vl[wet_vl],
        N_TRIALS,
    )
    histories["rain"] = hist_rain

    # ── 5. Graphes ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ÉTAPE 5 — Graphes Optuna")
    print("="*55)
    plot_histories(histories)

    # ── 6. Entraînement final ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ÉTAPE 6 — Entraînement final + sauvegarde")
    print("="*55)

    print("  -> Température (HistGBR sur delta T)...")
    print(f"     params : {bp_temp}")
    m_temp = HistGradientBoostingRegressor(**bp_temp)
    m_temp.fit(F_train, dt_temp)
    path = os.path.join(OUTPUT_DIR, "model_temp.pkl")
    pickle.dump(m_temp, open(path, "wb"))
    print(f"     OK model_temp.pkl ({os.path.getsize(path)/1024:.0f} KB)")

    print("  -> Vent (HistGBR sur delta W)...")
    print(f"     params : {bp_wind}")
    m_wind = HistGradientBoostingRegressor(**bp_wind)
    m_wind.fit(F_train, dt_wind)
    path = os.path.join(OUTPUT_DIR, "model_wind.pkl")
    pickle.dump(m_wind, open(path, "wb"))
    print(f"     OK model_wind.pkl ({os.path.getsize(path)/1024:.0f} KB)")

    print("  -> Pluie classifier (LogisticRegression)...")
    print(f"     {wet_tr.sum():,} samples positifs / {len(wet_tr):,} total")
    clf = LogisticRegression(C=1.0, max_iter=500)
    clf.fit(F_train, wet_tr.astype(int))
    path = os.path.join(OUTPUT_DIR, "rain_clf.pkl")
    pickle.dump(clf, open(path, "wb"))
    print(f"     OK rain_clf.pkl ({os.path.getsize(path)/1024:.0f} KB)")

    print("  -> Pluie regressor (HistGBR sur samples pluvieux)...")
    print(f"     params : {bp_rain}")
    m_rain = HistGradientBoostingRegressor(**bp_rain)
    m_rain.fit(F_train[wet_tr], rain_tr[wet_tr])
    path = os.path.join(OUTPUT_DIR, "model_rain.pkl")
    pickle.dump(m_rain, open(path, "wb"))
    print(f"     OK model_rain.pkl ({os.path.getsize(path)/1024:.0f} KB)")

    # ── 7. Évaluation ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ÉTAPE 7 — Évaluation sur 2026")
    print("="*55)

    pred_temp = last_vl[:, F_TEMP] + m_temp.predict(F_val)
    pred_wind = np.maximum(0, last_vl[:, F_WIND] + m_wind.predict(F_val))
    p_rain    = clf.predict_proba(F_val)[:, 1]
    pred_rain = np.maximum(0, p_rain * m_rain.predict(F_val))

    mae_t = mean_absolute_error(y_val[:, 0], pred_temp)
    mae_w = mean_absolute_error(y_val[:, 1], pred_wind)
    mae_r = mean_absolute_error(y_val[:, 2], pred_rain)
    score = -(mae_t/7.49 + mae_w/5.05 + mae_r/0.40) / 3

    print(f"\n  MAE Température : {mae_t:.4f} °C")
    print(f"  MAE Vent        : {mae_w:.4f} km/h")
    print(f"  MAE Pluie       : {mae_r:.4f} mm")
    print(f"\n  Score ML Arena  : {score:.4f}")
    print(f"\n  Fichiers dans   : {OUTPUT_DIR}")
    print("\n OK Termine !")


if __name__ == "__main__":
    main()
