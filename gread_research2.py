import os
import sys
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

DATA_DIR   = r"C:\Users\Massy\Documents\ProjetAi\Données\clean data"
AGENT_DIR  = r"C:\Users\Massy\Documents\ProjetAi\Données\clean data"
OUTPUT_TXT = r"C:\Users\Massy\Documents\ProjetAi\best_params_all.txt"

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

CITIES_ORDER = [
    "Amsterdam", "Athens", "Barcelona", "Belgrade", "Berlin",
    "Brussels", "Bucharest", "Budapest", "Copenhagen", "Dublin",
    "Hamburg", "Helsinki", "Kiev", "Lisbon", "London",
    "Madrid", "Paris", "Prague", "Rome", "Vienna",
]
PARIS_IDX  = 16
N_FEATURES = 6

feat_map = {
    "temperature": 0, "rain": 1, "wind_speed": 2,
    "wind_direction": 3, "humidity": 4, "clouds": 5,
}

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

GRID = {
    "boosting_type"    : ["dart", "gbdt", "goss"],
    "objective"        : ["huber", "regression_l1"],
    "num_leaves"       : [31, 63, 127],
    "learning_rate"    : [0.03, 0.05],
    "n_estimators"     : [400],
    "colsample_bytree" : [0.6, 0.8],
    "reg_alpha"        : [0.1, 0.5],
    "reg_lambda"       : [1.0, 2.0],
    "min_child_samples": [8, 15],
}

FIXED_PARAMS = {
    "min_child_weight": 1e-4,
    "n_jobs"          : -1,
    "random_state"    : 42,
    "verbosity"       : -1,
}

CONDITIONAL = {
    "dart": {"drop_rate": [0.08, 0.12], "skip_drop": [0.45], "max_drop": [50], "subsample": [0.75], "subsample_freq": [1]},
    "gbdt": {"subsample": [0.8], "subsample_freq": [1]},
    "goss": {"top_rate": [0.2], "other_rate": [0.1]},
}

OBJECTIVE_EXTRA = {
    "huber"  : {"alpha": [0.85, 0.9]},
    "tweedie": {"tweedie_variance_power": [1.8]},
}

TARGET_GRID = {
    "temp": {"objective": ["regression_l1", "huber"],  "boosting_type": ["dart", "gbdt"]},
    "wind": {"objective": ["huber"],                   "boosting_type": ["dart", "gbdt", "goss"]},
    "rain": {"objective": ["huber", "regression_l1"],  "boosting_type": ["gbdt", "goss"]},
}

N_SPLITS   = 3
EARLY_STOP = 30
MAX_COMBOS = 30


def _features(X):
    paris = X[PARIS_IDX]
    last  = paris[-1]
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
    return np.concatenate([last, diff1, diff6, diff12, mean24, std24, _CYCL, *voisins]).astype(np.float32)


def load_data():
    print("Chargement des données...")
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "weather" in f.lower()])
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV dans {DATA_DIR}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(DATA_DIR, f), parse_dates=["timestamp"])
        frames.append(df)
        print(f"  chargé {f} ({len(df):,} lignes)")

    data = pd.concat(frames, ignore_index=True)
    data.sort_values("timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)
    data["city_name"] = data["city_name"].str.strip()
    data = data[data["city_name"].isin(CITIES_ORDER)].copy()

    for col in feat_map:
        if col in data.columns:
            data[col] = data[col].fillna(0.0)

    timestamps   = np.sort(data["timestamp"].unique())
    ts_idx_map   = {t: i for i, t in enumerate(timestamps)}
    city_idx_map = {c: i for i, c in enumerate(CITIES_ORDER)}
    data["_ts_idx"]   = data["timestamp"].map(ts_idx_map)
    data["_city_idx"] = data["city_name"].map(city_idx_map)

    n_ts   = len(timestamps)
    n_city = len(CITIES_ORDER)
    cube   = np.zeros((n_ts, n_city, N_FEATURES), dtype=np.float32)

    for _, row in data.iterrows():
        ci = row.get("_city_idx")
        ti = row.get("_ts_idx")
        if pd.isna(ci) or pd.isna(ti):
            continue
        ci, ti = int(ci), int(ti)
        for fname, fidx in feat_map.items():
            if fname in data.columns:
                cube[ti, ci, fidx] = row[fname]

    WINDOW  = 24
    end_idx = n_ts - 6
    X_list, y_list = [], []

    print("Construction des fenêtres...")
    for i in range(WINDOW, end_idx):
        window = cube[i - WINDOW:i].transpose(1, 0, 2)
        target = cube[i, PARIS_IDX, :]
        X_list.append(window)
        y_list.append([target[0], target[2], target[1]])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Dataset : {len(X):,} samples  shape={X.shape}")
    return X, y


def build_features(X):
    print("Calcul des features delta...")
    F = np.array([_features(x) for x in X])
    print(f"Features shape : {F.shape}")
    return F


def build_deltas(X, y):
    last_paris = X[:, PARIS_IDX, -1, :]
    return {
        "temp": y[:, 0] - last_paris[:, 0],
        "wind": y[:, 1] - last_paris[:, 2],
        "rain": y[:, 2] - last_paris[:, 1],
    }


def build_combos(target_name):
    g = {k: v for k, v in GRID.items()}
    tg = TARGET_GRID[target_name]
    g["boosting_type"] = tg["boosting_type"]
    g["objective"]     = tg["objective"]

    base_keys   = list(g.keys())
    base_combos = list(itertools.product(*g.values()))
    all_combos  = []

    for base in base_combos:
        bd  = dict(zip(base_keys, base))
        bt  = bd["boosting_type"]
        obj = bd["objective"]

        cb = CONDITIONAL.get(bt, {})
        cb_combos = [dict(zip(cb.keys(), v)) for v in itertools.product(*cb.values())] if cb else [{}]
        co = OBJECTIVE_EXTRA.get(obj, {})
        co_combos = [dict(zip(co.keys(), v)) for v in itertools.product(*co.values())] if co else [{}]

        for c1 in cb_combos:
            for c2 in co_combos:
                all_combos.append({**bd, **c1, **c2, **FIXED_PARAMS})

    rng = np.random.default_rng(42)
    if len(all_combos) > MAX_COMBOS:
        idx = rng.choice(len(all_combos), MAX_COMBOS, replace=False)
        all_combos = [all_combos[i] for i in sorted(idx)]

    return all_combos


def evaluate_combo(params, F, delta):
    import lightgbm as lgb
    kf   = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    maes = []
    for fold, (tr, val) in enumerate(kf.split(F), 1):
        try:
            m = lgb.LGBMRegressor(**params)
            m.fit(
                F[tr], delta[tr],
                eval_set=[(F[val], delta[val])],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = m.predict(F[val])
            mae   = mean_absolute_error(delta[val], preds)
            maes.append(mae)
            print(f"    fold {fold}/{N_SPLITS}  MAE={mae:.4f}")
        except Exception as e:
            print(f"    fold {fold} erreur : {e}")
            return float("inf")
    return float(np.mean(maes))


def run_grid(target_name, F, delta):
    import lightgbm as lgb
    print(f"\n{'='*50}")
    print(f"  GRID SEARCH : {target_name.upper()}")
    print(f"{'='*50}")

    combos = build_combos(target_name)
    print(f"  {len(combos)} combinaisons à tester")

    results = []
    best_mae, best_params = float("inf"), None

    for i, params in enumerate(combos, 1):
        t0 = time.time()
        print(f"\ncombo {i}/{len(combos)}  bt={params['boosting_type']}  obj={params['objective']}  leaves={params['num_leaves']}  lr={params['learning_rate']}")
        mae = evaluate_combo(params, F, delta)
        elapsed = time.time() - t0
        print(f"  => MAE={mae:.4f}  ({elapsed:.1f}s)")

        results.append((f"test{i}", mae, params))

        if mae < best_mae:
            best_mae    = mae
            best_params = params.copy()
            print(f"  NOUVEAU MEILLEUR {best_mae:.4f}")

    return results, best_params, best_mae


def plot_results(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    targets   = ["temp", "wind", "rain"]
    titles    = ["Température", "Vent", "Pluie"]
    palette   = ["#2196F3", "#4CAF50", "#F44336"]

    for ax, target, title, color in zip(axes, targets, titles, palette):
        res   = all_results[target]
        names = [r[0] for r in res]
        maes  = [abs(r[1]) for r in res]
        best  = min(maes)

        colors = [color if m == best else "#BDBDBD" for m in maes]

        sns.barplot(x=names, y=maes, palette=colors, ax=ax, edgecolor="white", linewidth=0.5)

        ax.axhline(best, color=color, linestyle="--", linewidth=1.2, alpha=0.7, label=f"Best={best:.4f}")
        ax.set_title(f"{title}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Combinaison", fontsize=10)
        ax.set_ylabel("MAE absolue", fontsize=10)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#FAFAFA")

    plt.suptitle("Grid Search — MAE par combinaison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_fig = r"C:\Users\Massy\Documents\ProjetAi\grid_search_results.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nGraphe sauvegardé -> {out_fig}")
    plt.show()


def save_best(all_best, output_path):
    lines = ["=" * 60, "MEILLEURS PARAMS PAR TARGET", "=" * 60]
    for target, (params, mae) in all_best.items():
        lines.append(f"\n{target.upper()}  MAE={mae:.4f}")
        for k, v in params.items():
            if k not in {"n_jobs", "random_state", "verbosity", "min_child_weight"}:
                val_str = f'"{v}"' if isinstance(v, str) else str(v)
                lines.append(f"  {k} = {val_str}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSauvegardé -> {output_path}")


def main():
    try:
        import lightgbm as lgb
    except (ImportError, OSError):
        print("LightGBM non disponible, arrêt.")
        return

    t0 = time.time()

    X, y    = load_data()
    F       = build_features(X)
    deltas  = build_deltas(X, y)

    all_results = {}
    all_best    = {}

    for target in ["temp", "wind", "rain"]:
        results, best_params, best_mae = run_grid(target, F, deltas[target])
        all_results[target] = results
        all_best[target]    = (best_params, best_mae)
        print(f"\n{target.upper()} terminé — best MAE={best_mae:.4f}")

    print(f"\nDurée totale : {(time.time()-t0)/60:.1f} min")

    save_best(all_best, OUTPUT_TXT)
    plot_results(all_results)


if __name__ == "__main__":
    main()
