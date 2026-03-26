import pandas as pd
import numpy as np

# Les 20 villes européennes du concours
cities_20 = [
    "Amsterdam", "Barcelona", "Birmingham", "Brussels",
    "Copenhagen", "Dortmund", "Dublin", "Düsseldorf",
    "Essen", "Frankfurt am Main", "Köln", "London",
    "Manchester", "Marseille", "Milan", "Munich",
    "Paris", "Rotterdam", "Stuttgart", "Turin"
]

# Supprime les outliers en clippant avec la méthode IQR 
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
    return df

for year in range(2020, 2026):
    
    # Chargement du fichier brut
    df = pd.read_csv(f"weather_{year}.csv")
    
    # Garder uniquement les 20 villes du concours
    df = df[df['city_name'].isin(cities_20)]
    
    # Parser le timestamp et extraire heure + mois
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(['city_name', 'timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    
    # Remplir les valeurs manquantes de visibility (ffill puis bfill)
    df['visibility'] = df.groupby('city_name')['visibility'].ffill()
    df['visibility'] = df.groupby('city_name')['visibility'].bfill()
    
    # Corriger les outliers sur les colonnes météo importantes
    df = remove_outliers_iqr(df, 'wind_speed')
    df = remove_outliers_iqr(df, 'temperature')
    df = remove_outliers_iqr(df, 'rain')
    
    # Sauvegarder le fichier nettoyé
    df.to_csv(f"weather_{year}_clean.csv", index=False)
