import pandas as pd
import numpy as np

PARIS = "Paris"

def add_cyclical(df):
    # Encodage cyclique heure et mois → 23h et 0h sont voisins
    df['sin_hour']  = np.sin(2 * np.pi * df['hour']  / 24)
    df['cos_hour']  = np.cos(2 * np.pi * df['hour']  / 24)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def add_uv_wind(df):
    # Décomposition vent → composantes cartésiennes
    rad = np.deg2rad(df['wind_direction'])
    df['wind_u'] = df['wind_speed'] * np.sin(rad)
    df['wind_v'] = df['wind_speed'] * np.cos(rad)
    return df

def add_dew_point(df):
    # Dew point depression → proxy probabilité de pluie
    # Plus la valeur est petite → plus la pluie est probable
    T      = df['temperature']
    RH     = df['humidity'].clip(1, 100)
    a, b   = 17.27, 237.7
    gamma  = (a * T) / (b + T) + np.log(RH / 100)
    df['dew_depression'] = T - (b * gamma) / (a - gamma)
    return df

def add_lags(paris_df):
    # Lags température, pluie, vent → "qu'est ce qui s'est passé avant ?"
    for col in ['temperature', 'rain', 'wind_speed']:
        for lag in [1, 3, 6, 12]:
            paris_df[f'paris_{col}_lag{lag}'] = paris_df[col].shift(lag)
    return paris_df

def add_deltas(paris_df):
    # Deltas → "ça monte ou ça descend ?"
    for col in ['temperature', 'rain', 'wind_speed', 'humidity']:
        paris_df[f'delta_{col}_1h'] = paris_df[col].diff(1)
        paris_df[f'delta_{col}_6h'] = paris_df[col].diff(6)
    return paris_df

def add_upwind(df, COORDS):
    # Ville upwind = ville dont la météo arrive à Paris dans 6h
    # On détecte dynamiquement selon la direction du vent
    paris_lat, paris_lon = COORDS[PARIS]

    upwind_temp, upwind_rain, upwind_wind = [], [], []

    paris_rows = df[df['city_name'] == PARIS]

    for _, paris_row in paris_rows.iterrows():

        ts           = paris_row['timestamp']
        wind_dir_rad = np.deg2rad(paris_row['wind_direction'])

        # Vecteur d'où vient le vent
        wind_from_u = -np.sin(wind_dir_rad)
        wind_from_v = -np.cos(wind_dir_rad)

        # Snapshot de toutes les villes à ce timestamp
        snapshot = df[df['timestamp'] == ts]

        best_score = -np.inf
        best_temp  = paris_row['temperature']
        best_rain  = paris_row['rain']
        best_wind  = paris_row['wind_speed']

        for _, city_row in snapshot.iterrows():
            city = city_row['city_name']
            if city == PARIS:
                continue

            city_lat, city_lon = COORDS[city]

            # Score upwind : projection ville→Paris sur direction vent
            dlat  = paris_lat - city_lat
            dlon  = paris_lon - city_lon
            score = dlat * wind_from_v + dlon * wind_from_u

            if score > best_score:
                best_score = score
                best_temp  = city_row['temperature']
                best_rain  = city_row['rain']
                best_wind  = city_row['wind_speed']

        upwind_temp.append(best_temp)
        upwind_rain.append(best_rain)
        upwind_wind.append(best_wind)

    # Ajouter les colonnes upwind aux lignes Paris
    paris_idx = df[df['city_name'] == PARIS].index
    df.loc[paris_idx, 'upwind_temp'] = upwind_temp
    df.loc[paris_idx, 'upwind_rain'] = upwind_rain
    df.loc[paris_idx, 'upwind_wind'] = upwind_wind

    return df

def build_features(year):

    # Charger le fichier nettoyé
    df = pd.read_csv(f"weather_{year}_clean.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(['city_name', 'timestamp']).reset_index(drop=True)

    # Extraire les coords GPS directement du CSV
    coords_df = df[['city_name', 'latitude', 'longitude']].drop_duplicates()
    COORDS    = dict(zip(coords_df['city_name'],
                         zip(coords_df['latitude'], coords_df['longitude'])))

    # Appliquer les features sur tout le dataframe
    df = add_cyclical(df)
    df = add_uv_wind(df)
    df = add_dew_point(df)
    df = add_upwind(df, COORDS)

    # Lags et deltas uniquement sur Paris
    paris_df    = df[df['city_name'] == PARIS].copy()
    paris_df    = add_lags(paris_df)
    paris_df    = add_deltas(paris_df)

    # Colonnes ajoutées
    lag_cols   = [c for c in paris_df.columns if 'lag'   in c]
    delta_cols = [c for c in paris_df.columns if 'delta' in c]

    # Merger les features Paris dans le dataframe principal
    paris_extra = paris_df[['timestamp'] + lag_cols + delta_cols]
    df = df.merge(paris_extra, on='timestamp', how='left')

    # Supprimer les premières lignes NaN dues aux lags
    df = df.dropna(subset=lag_cols + delta_cols)

    # Sauvegarder
    df.to_csv(f"weather_{year}_features.csv", index=False)
    print(f"{year} — shape : {df.shape}")

# Lancer sur toutes les années
for year in range(2020, 2026):
    build_features(year)
