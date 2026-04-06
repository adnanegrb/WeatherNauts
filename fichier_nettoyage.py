import pandas as pd
import numpy as np

PHYSICAL_LIMITS = {
    'temperature':    (-45,  50),    # Record : -41°C (Mouthe) / +48.8°C (Sicile 2021)
    'wind_speed':     (0,    75),    # Record lowland : 63.3 m/s (Fraserburgh 1989)
    'rain':           (0,   200),    # Record : 184.6 mm/h (Valencia 2024)
    'humidity':       (0,   100),    # Limite physique pure
    'clouds':         (0,   100),    # Limite physique pure
    'snow':           (0,    40),    # Record : ~25 mm/h (Capracotta 2015)
    'wind_direction': (0,   360),    # Limite géométrique pure
}

cities_20 = [
    "Amsterdam", "Barcelona", "Birmingham", "Brussels",
    "Copenhagen", "Dortmund", "Dublin", "Düsseldorf",
    "Essen", "Frankfurt am Main", "Köln", "London",
    "Manchester", "Marseille", "Milan", "Munich",
    "Paris", "Rotterdam", "Stuttgart", "Turin"
]

numeric_cols = [
    'temperature', 'rain', 'wind_speed', 'wind_direction',
    'humidity', 'clouds', 'snow'
]

def clean_year(year):
    df = pd.read_csv(f"weather_{year}.csv")
    df = df[df['city_name'].isin(cities_20)]

    # visibility → inutile, mise à 0
    if 'visibility' in df.columns:
        df['visibility'] = 0

    # Parser le timestamp et extraire heure + mois
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(['city_name', 'timestamp']).drop_duplicates(subset=['city_name', 'timestamp'])
    df['hour']  = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    # Clip physique — corrige les valeurs impossibles
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Remplir les NaN par ville
    # interpolate → trous du milieu
    # ffill → trous de la fin
    # bfill → trous du début
    df[numeric_cols] = (
        df.groupby('city_name')[numeric_cols]
        .transform(lambda x: x.interpolate(method='linear').ffill().bfill())
    )

    # Dernier recours → médiane globale si NaN restants
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df.to_csv(f"weather_{year}_clean.csv", index=False)
    print(f"{year} — shape : {df.shape}")

for year in range(2020, 2027):
    clean_year(year)
