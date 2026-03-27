import pandas as pd
import numpy as np

# Limites physiques basées sur les vrais records météo européens
# Sources : WMO, Météo-France, Met Office, AEMET, Copernicus
PHYSICAL_LIMITS = {
    'temperature':    (-45,  50),    # Record : -41°C (Mouthe) / +48.8°C (Sicile 2021)
    'wind_speed':     (0,    75),    # Record lowland : 63.3 m/s (Fraserburgh 1989)
    'rain':           (0,    200),   # Record : 184.6 mm/h (Valencia 2024)
    'humidity':       (0,    100),   # Limite physique pure
    'clouds':         (0,    100),   # Limite physique pure
    'snow':           (0,    40),    # Record : ~25 mm/h (Capracotta 2015)
    'wind_direction': (0,    360),   # Limite géométrique pure
}

# Les 20 villes européennes imposées par le concours
cities_20 = [
    "Amsterdam", "Barcelona", "Birmingham", "Brussels",
    "Copenhagen", "Dortmund", "Dublin", "Düsseldorf",
    "Essen", "Frankfurt am Main", "Köln", "London",
    "Manchester", "Marseille", "Milan", "Munich",
    "Paris", "Rotterdam", "Stuttgart", "Turin"
]

# Colonnes numériques à nettoyer
# visibility exclue car 100% NaN dans le dataset
numeric_cols = [
    'temperature', 'rain', 'wind_speed', 'wind_direction',
    'humidity', 'clouds', 'snow'
]

def clean_year(year):

    # Chargement du fichier brut
    df = pd.read_csv(f"weather_{year}.csv")

    # Garder uniquement les 20 villes du concours
    df = df[df['city_name'].isin(cities_20)]

    # Supprimer visibility → 100% NaN, inutile pour le modèle
    if 'visibility' in df.columns:
        df = df.drop(columns=['visibility'])

    # Parser le timestamp et extraire heure + mois
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(['city_name', 'timestamp'])
    df['hour']  = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    # Supprimer les éventuels doublons
    df = df.drop_duplicates(subset=['city_name', 'timestamp'])

    # Alerte si une ville manque dans le CSV
    missing = set(cities_20) - set(df['city_name'].unique())
    if missing:
        print(f"{year} — villes manquantes : {missing}")

    # Clip physique — corrige les valeurs impossibles
    for col, (lower, upper) in PHYSICAL_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower, upper)

    # Remplir les NaN par ville
    # interpolate → trous du milieu
    # ffill → trous de la fin
    # bfill → trous du début
    df[numeric_cols] = (
        df.groupby('city_name')[numeric_cols]
        .transform(lambda x: x.interpolate(method='linear')
                               .ffill()
                               .bfill())
    )

    # Dernier recours → médiane globale si NaN restants
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Vérifications finales — filet de sécurité
    assert df['city_name'].nunique() == 20, "Il manque des villes !"
    assert df.isnull().sum().sum() == 0,    "Il reste des NaN !"

    # Sauvegarde du fichier nettoyé
    df.to_csv(f"weather_{year}_clean.csv", index=False)
    print(f"{year} — shape : {df.shape}")

# Lancer le nettoyage sur toutes les années
for year in range(2020, 2026):
    clean_year(year)
