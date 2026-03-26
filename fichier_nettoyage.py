import pandas as pd
import numpy as np

# Records météo européens — limites physiques réelles
PHYSICAL_LIMITS = {
    'temperature':    (-45,  50),    # Record Europe : -41°C (Mouthe) / +48.8°C (Sicile 2021)
    'wind_speed':     (0,    75),    # Record lowland : 63.3 m/s (Fraserburgh 1989)
    'rain':           (0,    200),   # Record Europe : 184.6 mm/h (Valencia 2024)
    'humidity':       (0,    100),   # Limite physique pure
    'clouds':         (0,    100),   # Limite physique pure
    'visibility':     (0,    10000), # Convention METAR (cap à 9999m)
    'snow':           (0,    40),    # Record Europe : ~25 mm/h (Capracotta 2015)
    'wind_direction': (0,    360),   # Limite géométrique pure
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
    'humidity', 'clouds', 'visibility', 'snow'
]

def clean_year(year):

    df = pd.read_csv(f"weather_{year}.csv")
    df = df[df['city_name'].isin(cities_20)]
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(['city_name', 'timestamp'])
    df['hour']  = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    
    df = df.drop_duplicates(subset=['city_name', 'timestamp'])
    
    missing = set(cities_20) - set(df['city_name'].unique())
    if missing:
        print(f"{year} — villes manquantes : {missing}")
    
    for col, (lower, upper) in PHYSICAL_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower, upper)
    
    # Étape 1 : interpolation + ffill + bfill par ville
    df[numeric_cols] = (
        df.groupby('city_name')[numeric_cols]
        .transform(lambda x: x.interpolate(method='linear')
                               .ffill()
                               .bfill())
    )
    
    # Étape 2 : si NaN restants → médiane globale par colonne
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # Vérifications finales
    assert df['city_name'].nunique() == 20, "Il manque des villes !"
    assert df.isnull().sum().sum() == 0,    "Il reste des NaN !"
    
    df.to_csv(f"weather_{year}_clean.csv", index=False)

for year in range(2020, 2026):
    clean_year(year)
