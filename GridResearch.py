"""
=============================================================
  GRID SEARCH — Hyperparamètres LightGBM pour le VENT
  VERSION TEST : 20 combinaisons max + affichage détaillé
=============================================================
"""

import os
import sys
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ─── CHEMINS ────────────────────────────────────────────────────────────────

DATA_DIR   = r"C:\Users\Massy\Documents\ProjetAi\Données\clean data"
AGENT_DIR  = r"C:\Users\Massy\Documents\ProjetAi\Données\clean data"
OUTPUT_TXT = r"C:\Users\Massy\Documents\ProjetAi\best_wind_params.txt"

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

# ─── PARAMÈTRES DU GRID ─────────────────────────────────────────────────────
# VERSION TEST : grids réduits → ~20 combinaisons rapides

GRID = {
    "boosting_type"    : ["dart", "gbdt"],
    "objective"        : ["huber", "regression_l1"],
    "num_leaves"       : [63, 127],
    "learning_rate"    : [0.05],
    "n_estimators"     : [300],        # réduit pour aller vite en test
    "colsample_bytree" : [0.7],
    "reg_alpha"        : [0.1],
    "reg_lambda"       : [1.0],
    "min_child_samples": [10],
}

FIXED_PARAMS = {
    "min_child_weight" : 1e-4,
    "n_jobs"           : -1,
    "random_state"     : 42,
    "verbosity"        : -1,
}

CONDITIONAL = {
    "dart": {
        "drop_rate"      : [0.10],
        "skip_drop"      : [0.45],
        "max_drop"       : [50],
        "subsample"      : [0.75],
        "subsample_freq" : [1],
    },
    "gbdt": {
        "subsample"      : [0.8],
        "subsample_freq" : [1],
    },
    "goss": {
        "top_rate"       : [0.2],
        "other_rate"     : [0.1],
    },
}

OBJECTIVE_EXTRA = {
    "tweedie" : {"tweedie_variance_power": [1.5]},
    "huber"   : {"alpha"                 : [0.85]},
}

N_SPLITS   = 3      # 3 folds en test (plus rapide que 5)
EARLY_STOP = 30     # early stopping réduit
MAX_COMBOS = 20     # ← limite test

# ─── CONFIG ──────────────────────────────────────────────────────────────────

CITIES_ORDER = [
    "Amsterdam", "Athens", "Barcelona", "Belgrade", "Berlin",
    "Brussels", "Bucharest", "Budapest", "Copenhagen", "Dublin",
    "Hamburg", "Helsinki", "Kiev", "Lisbon", "London",
    "Madrid", "Paris", "Prague", "Rome", "Vienna",
]
PARIS_IDX  = 16
N_FEATURES = 6

feat_map = {
    "temperature"   : 0,
    "rain"          : 1,
    "wind_speed"    : 2,
    "wind_direction": 3,
    "humidity"      : 4,
    "clouds"        : 5,
}

# ─── CHARGEMENT ──────────────────────────────────────────────────────────────

def load_data():
    print("\n" + "=" * 60)
    print("  ÉTAPE 1/4 — Chargement des données")
    print("=" * 60)

    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and "weather" in f.lower()
    ])

    if not csv_files:
        raise FileNotFoundError(
            f"Aucun CSV trouvé dans {DATA_DIR}\n"
            "Vérifie que les fichiers s'appellent weather_2020_clean.csv etc."
        )

    print(f"  Fichiers trouvés ({len(csv_files)}) :")
    for f in csv_files:
        print(f"    - {f}")

    frames = []
    for f in csv_files:
        path = os.path.join(DATA_DIR, f)
        df   = pd.read_csv(path, parse_dates=["timestamp"])
        print(f"  ✓ Chargé : {f}  ({len(df):,} lignes)")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data.sort_values("timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)
    data["city_name"] = data["city_name"].str.strip()
    data = data[data["city_name"].isin(CITIES_ORDER)].copy()

    for col in ["temperature", "rain", "wind_speed", "wind_direction", "humidity", "clouds"]:
        if col in data.columns:
            data[col] = data[col].fillna(0.0)

    print(f"\n  Total lignes après filtrage : {len(data):,}")
    print(f"  Période : {data['timestamp'].min()} → {data['timestamp'].max()}")

    timestamps    = np.sort(data["timestamp"].unique())
    ts_idx_map    = {t: i for i, t in enumerate(timestamps)}
    city_idx_map  = {c: i for i, c in enumerate(CITIES_ORDER)}
    data["_ts_idx"]   = data["timestamp"].map(ts_idx_map)
    data["_city_idx"] = data["city_name"].map(city_idx_map)

    n_ts   = len(timestamps)
    n_city = len(CITIES_ORDER)
    cube   = np.zeros((n_ts, n_city, N_FEATURES), dtype=np.float32)

    print(f"\n  Construction du cube 3D ({n_ts} timestamps × {n_city} villes × {N_FEATURES} features)...")
    for _, row in data.iterrows():
        ci = row.get("_city_idx")
        ti = row.get("_ts_idx")
        if pd.isna(ci) or pd.isna(ti):
            continue
        ci, ti = int(ci), int(ti)
        for fname, fidx in feat_map.items():
            if fname in data.columns:
                cube[ti, ci, fidx] = row[fname]
    print("  ✓ Cube 3D construit")

    WINDOW  = 24
    end_idx = n_ts - 6
    X_list, y_list = [], []

    print(f"\n  Construction des fenêtres glissantes (24h, pas=1h)...")
    total_windows = end_idx - WINDOW
    for i in range(WINDOW, end_idx):
        window = cube[i - WINDOW:i]
        target = cube[i, PARIS_IDX, 2]
        X_list.append(window.transpose(1, 0, 2))
        y_list.append(target)
        if (i - WINDOW) % 5000 == 0:
            pct = (i - WINDOW) / total_windows * 100
            print(f"    ... fenêtre {i - WINDOW:,}/{total_windows:,}  ({pct:.0f}%)")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  ✓ Échantillons construits : {len(X):,}  |  shape X = {X.shape}")
    return X, y


# ─── FEATURES ────────────────────────────────────────────────────────────────

def build_features(X):
    print("\n" + "=" * 60)
    print("  ÉTAPE 2/4 — Calcul des features")
    print("=" * 60)
    try:
        from agent import features_xgb
        print("  ✓ features_xgb importé depuis agent.py")
        feats = features_xgb(X)
        print(f"  ✓ Features calculées  |  shape = {feats.shape}")
        return feats
    except ImportError:
        print("  ⚠ agent.py introuvable — features simples utilisées")
        feats = _simple_features(X)
        print(f"  ✓ Features simples    |  shape = {feats.shape}")
        return feats


def _simple_features(X):
    paris = X[:, PARIS_IDX]
    wind  = paris[:, :, 2]
    return np.concatenate([
        wind,
        wind[:, -6:].mean(axis=1, keepdims=True),
        wind[:, -12:].mean(axis=1, keepdims=True),
        wind.std(axis=1, keepdims=True),
        wind[:, -1:] - wind[:, -7:-6],
    ], axis=1).astype(np.float32)


# ─── GRID ────────────────────────────────────────────────────────────────────

def build_all_combinations():
    base_keys   = list(GRID.keys())
    base_vals   = list(GRID.values())
    base_combos = list(itertools.product(*base_vals))
    all_combos  = []

    for base in base_combos:
        base_dict = dict(zip(base_keys, base))
        bt  = base_dict["boosting_type"]
        obj = base_dict["objective"]

        cond_b = CONDITIONAL.get(bt, {})
        cond_b_combos = [dict(zip(cond_b.keys(), v))
                         for v in itertools.product(*cond_b.values())] if cond_b else [{}]

        cond_o = OBJECTIVE_EXTRA.get(obj, {})
        cond_o_combos = [dict(zip(cond_o.keys(), v))
                         for v in itertools.product(*cond_o.values())] if cond_o else [{}]

        for cb in cond_b_combos:
            for co in cond_o_combos:
                all_combos.append({**base_dict, **cb, **co, **FIXED_PARAMS})

    return all_combos


# ─── ÉVALUATION ──────────────────────────────────────────────────────────────

def evaluate(params, f_xgb, y_wind, combo_idx, total):
    kf   = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    maes = []

    for fold_num, (tr_idx, val_idx) in enumerate(kf.split(f_xgb), start=1):
        X_tr, X_val = f_xgb[tr_idx], f_xgb[val_idx]
        y_tr, y_val = y_wind[tr_idx], y_wind[val_idx]

        model = lgb.LGBMRegressor(**params)
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(EARLY_STOP, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            preds = np.maximum(0.0, model.predict(X_val))
            fold_mae = mean_absolute_error(y_val, preds)
            maes.append(fold_mae)
            print(f"      fold {fold_num}/{N_SPLITS} → MAE = {fold_mae:.4f}")
        except Exception as e:
            print(f"      fold {fold_num}/{N_SPLITS} → ERREUR : {e}")
            return float("inf"), str(e)

    mean_mae = float(np.mean(maes))
    std_mae  = float(np.std(maes))
    print(f"      → Moyenne : {mean_mae:.4f}  (±{std_mae:.4f})")
    return mean_mae, None


# ─── RÉSULTATS ───────────────────────────────────────────────────────────────

def print_leaderboard(results):
    """Affiche le classement courant des combinaisons testées."""
    sorted_r = sorted(results, key=lambda x: x[0])
    top      = sorted_r[:min(5, len(sorted_r))]
    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print(  "  │  CLASSEMENT ACTUEL (top 5)                              │")
    print(  "  ├──────┬─────────┬──────────┬────────────────┬────────────┤")
    print(  "  │ Rang │   MAE   │ boosting │   objective    │   leaves   │")
    print(  "  ├──────┼─────────┼──────────┼────────────────┼────────────┤")
    for rank, (mae, p) in enumerate(top, start=1):
        star = " ★" if rank == 1 else "  "
        print(f"  │{star}{rank:>3}  │ {mae:.4f}  │ {p['boosting_type']:<8} │ {p['objective']:<14} │ {p['num_leaves']:<10} │")
    print(  "  └──────┴─────────┴──────────┴────────────────┴────────────┘")


def save_results(best_params, best_mae, results, output_path):
    lines = [
        "=" * 60,
        "  RÉSULTATS GRID SEARCH — VENT (wind_speed)",
        "=" * 60,
        f"  Meilleur MAE (CV-{N_SPLITS}) : {best_mae:.4f} m/s",
        "",
        "  ── Meilleurs paramètres ──",
    ]
    for k, v in best_params.items():
        lines.append(f"    {k:<30} = {v}")

    lines += ["", "  ── Top 10 combinaisons ──",
              f"  {'Rang':<5} {'MAE':>8}  {'boosting':<8} {'objective':<16} {'leaves':<8} {'lr':<8} {'n_est':<7}",
              "  " + "-" * 65]
    for rank, (mae, p) in enumerate(sorted(results, key=lambda x: x[0])[:10], start=1):
        lines.append(
            f"  {rank:<5} {mae:>8.4f}  "
            f"{p['boosting_type']:<8} {p['objective']:<16} "
            f"{p['num_leaves']:<8} {p['learning_rate']:<8} {p['n_estimators']:<7}"
        )

    lines += ["", "  ── Code à copier dans agent.py ──", "  lgb_params_wind = dict("]
    skip = {"n_jobs", "random_state", "verbosity"}
    for k, v in best_params.items():
        if k in skip:
            continue
        val_str = f'"{v}"' if isinstance(v, str) else str(v)
        lines.append(f"      {k:<30} = {val_str},")
    lines += ["      n_jobs                         = -1,",
              "      random_state                   = 42,",
              "      verbosity                      = -1,",
              "  )", "=" * 60]

    text = "\n".join(lines)
    print("\n" + text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  ✓ Résultats sauvegardés → {output_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("\n" + "=" * 60)
    print("   GRID SEARCH — LightGBM wind_speed  [MODE TEST 20 combos]")
    print("=" * 60)

    # 1) Données
    X, y_wind = load_data()

    # 2) Features
    f_xgb = build_features(X)

    # 3) Grid
    print("\n" + "=" * 60)
    print("  ÉTAPE 3/4 — Construction du grid")
    print("=" * 60)
    all_combos = build_all_combinations()
    print(f"  Combinaisons totales générées : {len(all_combos)}")

    if len(all_combos) > MAX_COMBOS:
        rng    = np.random.default_rng(42)
        idx    = rng.choice(len(all_combos), MAX_COMBOS, replace=False)
        combos = [all_combos[i] for i in sorted(idx)]
        print(f"  → Tirage aléatoire reproductible : {MAX_COMBOS} combinaisons retenues")
    else:
        combos = all_combos

    total = len(combos)
    print(f"  Combinaisons à tester : {total}")
    print(f"  Folds CV              : {N_SPLITS}")
    print(f"  Early stopping        : {EARLY_STOP} rounds")

    # Affiche toutes les combos avant de commencer
    print(f"\n  Liste des {total} combinaisons planifiées :")
    print(f"  {'#':<4} {'boosting':<8} {'objective':<16} {'leaves':<8} {'lr':<6} {'n_est'}")
    print("  " + "-" * 55)
    for i, p in enumerate(combos, start=1):
        print(f"  {i:<4} {p['boosting_type']:<8} {p['objective']:<16} "
              f"{p['num_leaves']:<8} {p['learning_rate']:<6} {p['n_estimators']}")

    # 4) Grid search
    print("\n" + "=" * 60)
    print("  ÉTAPE 4/4 — Évaluation des combinaisons")
    print("=" * 60)

    best_mae    = float("inf")
    best_params = None
    results     = []

    for i, params in enumerate(combos):
        elapsed = time.time() - t_start
        eta_s   = (elapsed / max(i, 1)) * (total - i) if i > 0 else 0
        print(f"\n  ╔══ Combo {i+1}/{total}  ({(i+1)/total*100:.0f}%)  "
              f"│  Écoulé : {elapsed/60:.1f}min  │  ETA : {eta_s/60:.1f}min ══╗")
        print(f"  ║  boosting={params['boosting_type']}  "
              f"objective={params['objective']}  "
              f"leaves={params['num_leaves']}  "
              f"lr={params['learning_rate']}  "
              f"n_est={params['n_estimators']}")
        print(f"  ╚{'═'*60}╝")

        mae, err = evaluate(params, f_xgb, y_wind, i, total)

        if err:
            print(f"  ✗ Combinaison ignorée (erreur)")
            continue

        results.append((mae, params))

        if mae < best_mae:
            best_mae    = mae
            best_params = params.copy()
            print(f"  ★ NOUVEAU MEILLEUR  MAE = {mae:.4f}")
        else:
            diff = mae - best_mae
            print(f"  → MAE = {mae:.4f}  (+{diff:.4f} vs meilleur)")

        # Classement mis à jour après chaque combo
        print_leaderboard(results)

    # 5) Résultats finaux
    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  Durée totale : {total_min:.1f} min")
    print(f"  Combinaisons testées avec succès : {len(results)}/{total}")
    print(f"{'='*60}")

    if best_params is not None:
        save_results(best_params, best_mae, results, OUTPUT_TXT)
    else:
        print("  ✗ Aucune combinaison n'a réussi. Vérifie tes données.")


if __name__ == "__main__":
    main()