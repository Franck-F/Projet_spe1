#!/usr/bin/env python
# coding: utf-8

# # CHARGEMENT ET EXPLORATION DES DONNÉES

# ### Chargement et Nettoyage Initial

# In[1]:


import urllib.request
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from statsmodels.tsa.seasonal import STL
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import optuna
import shap
import seaborn as sns
import skimpy as sk
import summarytools as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
import plotly.io as pio
import calendar
pio.templates.default = "plotly_white"

print("Environnement configuré avec succès!")
print(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# In[2]:


# dataset
df = pd.read_csv('../data/raw/time_series_60min_fr_dk_2015_2020.csv', 
    parse_dates=['utc_timestamp', 'cet_cest_timestamp'],
    low_memory=False)
df = df.set_index('utc_timestamp')
df.head()


# **Selection France**

# In[3]:


# Sélectionner les colonnes France pertinentes
france_cols = ['FR_load_actual_entsoe_transparency', 'FR_load_forecast_entsoe_transparency',
               'FR_solar_generation_actual', 'FR_wind_onshore_generation_actual',
               'IT_NORD_FR_price_day_ahead', 'temperature_france',
               'cloud_cover_france', 'FR_nuclear_generation_actual', 'wind_speed_france']

df_france = df[france_cols].copy()
df_france.head()


# In[4]:



# Renommer les colonnes pour faciliter l'utilisation
df_france.columns = ['load', 'load_forecast', 'solar', 'wind',
                     'price_day_ahead', 'temperature', 'cloud_cover', 'nuclear', 'wind_speed']
print(f"Shape initial : {df_france.shape}")
#print(f"Périodе : {df_france['utc_timestamp'].min()} à {df_france['utc_timestamp'].max()}")
df_france.head()


# **Valeurs manquantes et doublons**

# * Doublons

# In[5]:


# Vérification des doublons sur l'index utc_timestamp
total = len(df_france)
unique = df_france.index.nunique()
dup = total - unique

print(f"Total lignes: {total}")
print(f"Lignes uniques par utc_timestamp: {unique}")
print(f"Doublons détectés: {dup}")

if dup:
    dup_timestamps = df.index[df.index.duplicated(keep=False)].unique()
    print(f"Nombre de timestamps dupliqués uniques: {len(dup_timestamps)}")
    display(pd.DataFrame({"duplicated_timestamp": dup_timestamps}).head(20))
    # Afficher un échantillon des lignes dupliquées pour inspection
    sample_ts = dup_timestamps[:5]
    for ts in sample_ts:
        print(f"\nExemple pour timestamp dupliqué: {ts}")
        display(df_france.loc[ts])
else:
    print("Aucun doublon trouvé sur utc_timestamp.")

# * Valeurs manquantes

# In[6]:


# Quantification et visualisation des valeurs manquantes (df_france)

missing_count = df_france.isna().sum()
missing_pct = (missing_count / len(df_france)) * 100
missing_df_all = (
    pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    .sort_values("missing_pct", ascending=False)
)
display(missing_df_all)

# Bar plot des pourcentages de valeurs manquantes
fig_missing_bar = px.bar(
    missing_df_all.reset_index().rename(columns={"index": "column"}),
    x="missing_pct",
    y="column",
    orientation="h",
    text="missing_pct",
    title="Pourcentage de valeurs manquantes par colonne (df_france)",
    labels={"missing_pct": "% NaN", "column": "Colonne"},
)
fig_missing_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

fig_missing_bar.show()

# In[7]:


# Périodes avec valeurs manquantes pour IT_NORD_FR_price_day_ahead
col = "price_day_ahead"
mask = df_france[col].isna()

if not mask.any():
    print(f"Aucune valeur manquante pour {col}.")
else:
    # numéroter les runs (changes de state)
    run_id = (mask != mask.shift(1)).cumsum()
    runs = (
        df_france[mask]
        .groupby(run_id[mask])
        .apply(lambda x: pd.Series({
            "start": x.index.min(),
            "end": x.index.max(),
            "n_points": len(x)
        }))
        .reset_index(drop=True)
    )
    runs["duration_hours"] = (runs["end"] - runs["start"]) / np.timedelta64(1, "h") + 1
    runs = runs.sort_values("start").reset_index(drop=True)

    print(f"Nombre de périodes disjointes avec des NaN pour {col} : {len(runs)}")
    display(runs)

    overall = pd.Series({
        "first_nan": runs["start"].min(),
        "last_nan": runs["end"].max(),
        "total_nan_points": int(mask.sum()),
        "total_points": len(df_france),
        "nan_pct": mask.mean() * 100
    })
    display(overall)

# ***Troncature des données --- Limite du DataFrame à la plage de dates pour laquelle les données de prix sont disponibles***

# In[8]:


start_date = '2015-01-05'
end_date = '2017-12-05'
df_france = df_france.loc[start_date:end_date]

# In[9]:


missing_count = df_france.isna().sum()
missing_pct = (missing_count / len(df_france)) * 100
missing_df_all = (
    pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    .sort_values("missing_pct", ascending=False)
)
display(missing_df_all)

# ***Imputation des valeurs manquantes***

# In[10]:


# Utiliser l'interpolation linéaire pour les quelques NaN restants
df_france.interpolate(method='linear', inplace=True)

# In[11]:


missing_count = df_france.isna().sum()
missing_pct = (missing_count / len(df_france)) * 100
missing_df_all = (
    pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    .sort_values("missing_pct", ascending=False)
)
display(missing_df_all)

# ### Statistiques descriptive

# In[12]:


df_france.describe()

# In[13]:


sk.skim(df_france)

# In[14]:


st.dfSummary(df_france)

# ## EDA France

# * **Analyse du "Price_day_ahead"**

# In[15]:



# Distribution du prix
fig = go.Figure()
fig.add_trace(go.Histogram(x=df_france['price_day_ahead'], nbinsx=50, name='Historique'))
fig.update_layout(
    title="Distribution des Prix (price_day_ahead)",
    xaxis_title="Prix (€/MWh)",
    yaxis_title="Fréquence",
    template="plotly_white",
    hovermode="x unified",
    height=400
)
fig.show()


# In[16]:



# Série temporelle du prix
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_france.index, y=df_france['price_day_ahead'],
                         mode='lines', name='Prix', line=dict(color='blue')))
fig.update_layout(
    title="Évolution Temporelle du Prix (2015-2017)",
    xaxis_title="Date",
    yaxis_title="Prix (€/MWh)",
    template="plotly_white",
    hovermode="x unified",
    height=400
)
fig.show()


# In[17]:


# Prix vs Charge
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_france['load'], y=df_france['price_day_ahead'],
                         mode='markers', marker=dict(size=4, opacity=0.5, 
                         color=df_france['temperature'], colorscale='Viridis'),
                         text=df_france.index.strftime('%Y-%m-%d'),
                         hovertemplate='<b>Date:</b> %{text}<br><b>Charge:</b> %{x:.0f} MW<br><b>Prix:</b> %{y:.2f} €/MWh<extra></extra>'))
fig.update_layout(
    title="Prix vs Charge Électrique",
    xaxis_title="Charge (MW)",
    yaxis_title="Prix (€/MWh)",
    template="plotly_white",
    height=500
)
fig.show()


# In[18]:



# Prix vs Température
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_france['temperature'], y=df_france['price_day_ahead'],
                         mode='markers', marker=dict(size=4, opacity=0.5, color='red')))
fig.update_layout(
    title="Prix vs Température",
    xaxis_title="Température (°C)",
    yaxis_title="Prix (€/MWh)",
    template="plotly_white"
)
fig.show()


# * **Saisonnalité**

# In[19]:


#Analyse de la saisonnalité
df_seasonal = df_france.copy()
df_seasonal['month'] = df_seasonal.index.month_name()
df_seasonal['day_of_week'] = df_seasonal.index.day_name()
df_seasonal['hour'] = df_seasonal.index.hour

# In[20]:


# Saisonnalité annuelle : distribution mensuelle des prix

fig = px.box(
    df_seasonal,
    x="month",
    y="price_day_ahead",
    points="outliers", 
    title="Saisonnalité annuelle : distribution mensuelle des prix",
    labels={"month": "Mois", "price": "Prix (€/MWh)"},
    template="plotly_white",
)
fig.update_layout(xaxis=dict(dtick=1))

fig.show()

# In[21]:


# Saisonnalité quotidienne : distribution des prix par heure

fig = px.box(
    df_seasonal,
    x="hour",
    y="price_day_ahead",
    points="outliers",
    title="Saisonnalité quotidienne : distribution des prix par heure",
    labels={"hour": "Heure de la journée", "price": "Prix (€/MWh)"},
    template="plotly_white",
)
fig.update_xaxes(dtick=1)

fig.show()

# In[22]:


# Semaine vs week-end
df_seasonal["week_period"] = np.where(
    df_seasonal["day_of_week"].isin(["Saturday", "Sunday"]),
    "Weekend",
    "Weekday",
)

fig_week = px.box(
    df_seasonal,
    x="week_period",
    y="price_day_ahead",
    color="week_period",
    points="outliers",
    title="Distribution des prix : Semaine vs Week-end",
    labels={"week_period": "", "price": "Prix (€/MWh)"},
    template="plotly_white",
)
fig_week.update_layout(showlegend=False)
fig_week.show()

# In[23]:


# Saisons (été/hiver/printemps/automne)
SEASON_LABELS = {
    "December": "Winter", "January": "Winter", "February": "Winter",
    "March": "Spring", "April": "Spring", "May": "Spring",
    "June": "Summer", "July": "Summer", "August": "Summer",
    "September": "Autumn", "October": "Autumn", "November": "Autumn",
}
df_seasonal["season"] = df_seasonal["month"].map(SEASON_LABELS)
season_order = ["Winter", "Spring", "Summer", "Autumn"]

fig_season = px.box(
    df_seasonal,
    x="season",
    y="price_day_ahead",
    category_orders={"season": season_order},
    points="outliers",
    title="Distribution des prix : Saisons",
    labels={"season": "Saison", "price": "Prix (€/MWh)"},
    template="plotly_white",
)
fig_season.show()

# In[24]:


# Saisonnalité hebdomadaire : distribution des prix par jour de la semaine

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig = px.box(
    df_seasonal,
    x="day_of_week",
    y="price_day_ahead",
    category_orders={"day_of_week": day_order},
    points="outliers",
    title="Saisonnalité hebdomadaire : distribution des prix par jour de la semaine",
    labels={"day_of_week": "Jour de la semaine", "price": "Prix (€/MWh)"},
    template="plotly_white",
)
fig.update_xaxes(tickangle=-45)

fig.show()

# * **Analyse des correlations**

# In[25]:


# Matrice de correlation 
corr_matrix = df_france.corr(method='pearson')

# In[26]:


# Heatmap de la matrice de corrélation

fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Correlation Matrix of Variables",
    labels=dict(x="Variables", y="Variables", color="Corr"),
)
fig.update_layout(height=800, width=900, template="plotly_white")

fig.show() 

# In[27]:



# Corrélations avec le prix
print("\n" + "=" * 80)
print("CORRÉLATIONS AVEC LE PRIX")
print("=" * 80)
correlations = df_france.corr()['price_day_ahead'].drop('price_day_ahead').sort_values(ascending=False)
print(correlations.round(4))

# Visualisation corrélations
fig = go.Figure()
fig.add_trace(go.Bar(x=correlations.values, y=correlations.index, 
                      orientation='h', marker=dict(color=correlations.values, 
                      colorscale='RdBu', cmid=0)))
fig.update_layout(
    title="Corrélations avec le Prix",
    xaxis_title="Corrélation",
    template="plotly_white",
    height=400
)
fig.show()


# ## FEATURE ENGINEERING

# * **Création des Features**

# In[28]:


# Copier le dataframe
df_features = df_france.copy()


# In[29]:


# === FEATURES TEMPORELLES ===
df_features['hour'] = df_features.index.hour
df_features['day_of_week'] = df_features.index.dayofweek # Monday=0, Sunday=6
df_features['day_of_year'] = df_features.index.dayofyear
df_features['month'] = df_features.index.month
df_features['year'] = df_features.index.year
df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)


# In[30]:


# Saison
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall'}
df_features['season'] = df_features['month'].map(season_map)


# In[31]:


# === LAG FEATURES (Prix et Charge) ===
for lag in [1, 3, 6, 12, 24]:
    df_features[f'price_lag_{lag}h'] = df_features['price_day_ahead'].shift(lag)
    df_features[f'load_lag_{lag}h'] = df_features['load'].shift(lag)


# In[32]:



# === ROLLING WINDOWS ===
for window in [6, 24]:
    df_features[f'price_rolling_mean_{window}h'] = df_features['price_day_ahead'].shift(1).rolling(window=window).mean()
    df_features[f'price_rolling_std_{window}h'] = df_features['price_day_ahead'].shift(1).rolling(window=window).std()
    df_features[f'load_rolling_mean_{window}h'] = df_features['load'].rolling(window=window).mean()


# In[33]:


# === FEATURES DÉRIVÉES ===
df_features['renewable_generation'] = df_features['solar'] + df_features['wind']
df_features['total_generation'] = df_features['renewable_generation'] + df_features['nuclear']
df_features['price_delta'] = df_features['price_day_ahead'].diff()


# In[34]:


# === INTERACTIVES ===
df_features['load_x_hour'] = df_features['load'] * df_features['hour'] / 100
df_features['temp_x_cloud'] = df_features['temperature'] * df_features['cloud_cover']


# In[35]:



print(f"Features créées : {df_features.shape[1]}")
print(f"Observations après lag/rolling : {df_features.shape[0]}")


# In[36]:



# Nettoyer les NaN générés par les lags et rolling windows
df_features = df_features.dropna()
print(f"Observations finales : {df_features.shape[0]}")
print(f"\nNom des features : {list(df_features.columns)}")


# In[37]:



# Sauvegarder temporairement
df_features.to_csv('df_features_france_2015_2017.csv')

