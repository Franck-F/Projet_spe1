#!/usr/bin/env python
# coding: utf-8

# ## Feature Engineering - Prix Électricité France (2020-2025)

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# # 4. Chargement du Dataset Nettoyé

# In[2]:


# Charger le dataset nettoyé du notebook EDA
df = pd.read_csv('../../data/processed/df_features_france_2020_2025.csv', 
                parse_dates=['utc_timestamp'],
                index_col='utc_timestamp')
print(f"Dataset chargé: {df.shape}")
print(f"Colonnes: {list(df.columns)}")

# ## Feature Engineering

# #### Features Temporelles

# In[3]:


print(" Création des features temporelles...")
df_features = df.copy()

df_features['hour'] = df_features.index.hour
df_features['day_of_week'] = df_features.index.dayofweek
df_features['day_of_year'] = df_features.index.dayofyear
df_features['month'] = df_features.index.month
df_features['year'] = df_features.index.year
df_features['quarter'] = df_features.index.quarter
df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)

season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall'}
df_features['season'] = df_features['month'].map(season_map)

french_holidays = [(1,1), (5,1), (5,8), (7,14), (8,15), (11,1), (11,11), (12,25)]
df_features['is_holiday'] = df_features.index.to_series().apply(
    lambda x: 1 if (x.month, x.day) in french_holidays else 0
)

print(f" features temporelles créées")

# #### Lag Features

# In[4]:


print("\n Création des lag features...")
target = 'price_day_ahead'

# Lags prix
for lag in [1, 3, 6, 12, 24, 168]:
    df_features[f'price_lag_{lag}h'] = df_features[target].shift(lag)

# Lags charge
for lag in [1, 3, 6, 12, 24]:
    df_features[f'load_lag_{lag}h'] = df_features['load'].shift(lag)

print(f"lag features créées")

# #### Rolling Windows

# In[5]:


print("\n Création des rolling windows...")

for window in [6, 24, 168]:
    df_features[f'price_rolling_mean_{window}h'] = df_features[target].shift(1).rolling(window=window).mean()
    df_features[f'price_rolling_std_{window}h'] = df_features[target].shift(1).rolling(window=window).std()
    df_features[f'price_rolling_min_{window}h'] = df_features[target].shift(1).rolling(window=window).min()
    df_features[f'price_rolling_max_{window}h'] = df_features[target].shift(1).rolling(window=window).max()

for window in [6, 24]:
    df_features[f'load_rolling_mean_{window}h'] = df_features['load'].rolling(window=window).mean()
    df_features[f'load_rolling_std_{window}h'] = df_features['load'].rolling(window=window).std()

print(f"  rolling window features créées")

# #### Features Dérivées

# In[6]:


print("\nCréation des features dérivées...")

if 'solar' in df_features.columns and 'wind' in df_features.columns:
    df_features['renewable_generation'] = df_features['solar'] + df_features['wind']

if 'renewable_generation' in df_features.columns and 'nuclear' in df_features.columns:
    df_features['total_generation'] = df_features['renewable_generation'] + df_features['nuclear']

if 'load' in df_features.columns and 'total_generation' in df_features.columns:
    df_features['residual_load'] = df_features['load'] - df_features['total_generation']

df_features['price_delta'] = df_features[target].diff()
df_features['price_delta_pct'] = df_features[target].pct_change() * 100

if 'renewable_generation' in df_features.columns and 'load' in df_features.columns:
    df_features['renewable_ratio'] = df_features['renewable_generation'] / (df_features['load'] + 1)

print(f" features dérivées créées")

# #### Features Interactives

# In[7]:


print("\nCréation des features interactives...")

df_features['load_x_hour'] = df_features['load'] * df_features['hour'] / 100

if 'temperature' in df_features.columns and 'cloud_cover' in df_features.columns:
    df_features['temp_x_cloud'] = df_features['temperature'] * df_features['cloud_cover']

if 'temperature' in df_features.columns and 'load' in df_features.columns:
    df_features['temp_x_load'] = df_features['temperature'] * df_features['load'] / 1000

if 'wind' in df_features.columns and 'wind_speed' in df_features.columns:
    df_features['wind_x_speed'] = df_features['wind'] * df_features['wind_speed']

print(f"  features interactives créées")

# ## Résumé et Nettoyage

# In[8]:


print("\n" + "="*60)
print(f"Features totales: {df_features.shape[1]}")
print(f"Observations avant nettoyage: {df_features.shape[0]}")


# In[9]:



df_features = df_features.dropna()
print(f"Observations après nettoyage: {df_features.shape[0]}")
print(f"Lignes supprimées (NaN): {df.shape[0] - df_features.shape[0]}")


# In[10]:



df_ml = df_features.copy()
print(f"\nDataset ML prêt: {df_ml.shape}")


# In[11]:



new_features = [c for c in df_features.columns if c not in df.columns]
print(f"\nNouvelles features ({len(new_features)}):")
for i, feat in enumerate(new_features, 1):
    print(f"  {i:2d}. {feat}")

# ## Sauvegarde

# In[12]:


# Sauvegarder pour le notebook de modélisation
df_ml.to_csv('../../data/processed/df_ml_france_2020_2025.csv')
print("\n Dataset sauvegardé: ../data/processed/df_ml_france_2020_2025.csv")
