#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA France - Analyse et Prédiction des Prix de l'Électricité
=============================================================

Ce script effectue une analyse exploratoire complète des données de prix de l'électricité
en France et construit un modèle de prédiction utilisant LightGBM avec optimisation Optuna.

Date d'extraction: Novembre 2024
"""

# =============================================================================
# IMPORTS
# =============================================================================

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
import calendar
import plotly.io as pio

# Configuration des warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# Configuration plotly
pio.templates.default = "plotly_white"

print("Environnement configuré avec succès!")
print(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# =============================================================================
# 1. TÉLÉCHARGEMENT ET CHARGEMENT DES DONNÉES
# =============================================================================

# Créer dossier data si inexistant
os.makedirs('../data/raw', exist_ok=True)

# URL dataset 60min
url = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
destination = "../data/raw/time_series_60min.csv"

# Télécharger si pas déjà présent
if not os.path.exists(destination):
    print(" Téléchargement du dataset (124 MB)... Patience!")
    urllib.request.urlretrieve(url, destination)
    print(" Dataset téléchargé!")
else:
    print(" Dataset déjà présent localement")

# Charger dataset
df = pd.read_csv('../data/raw/time_series_60min.csv',
    parse_dates=['utc_timestamp', 'cet_cest_timestamp'],
    low_memory=False
)

# Définir timestamp comme index
df = df.set_index('utc_timestamp')

print(f" Shape du dataset: {df.shape}")
print(f" Période: {df.index.min()} → {df.index.max()}")
print(f"\n Premières lignes:")
print(df.head(5))

# =============================================================================
# 2. EXTRACTION DES DONNÉES FRANCE
# =============================================================================

# Sélection des colonnes françaises
france_cols = [col for col in df.columns if 'FR' in col]
df_france = df[france_cols].copy()

# Résumé du dataset France
print("\n" + "="*60)
print("Résumé du dataset France:")
print("="*60)
print(f"Dimensions: {df_france.shape[0]} lignes × {df_france.shape[1]} colonnes")
print(f"Période: {df_france.index.min()} à {df_france.index.max()}")

# Aperçu des premières lignes
print("\n Aperçu des données:")
print(df_france.head())

print("\n Statistiques descriptives:")
print(df_france.describe())

# Analyse avec skimpy et summarytools
sk.skim(df_france)
st.dfSummary(df_france)

# =============================================================================
# 3. VÉRIFICATION DES DOUBLONS
# =============================================================================

print("\n" + "="*60)
print("Vérification des doublons")
print("="*60)

total = len(df_france)
unique = df_france.index.nunique()
dup = total - unique

print(f"Total lignes: {total}")
print(f"Lignes uniques par utc_timestamp: {unique}")
print(f"Doublons détectés: {dup}")

if dup:
    dup_timestamps = df.index[df.index.duplicated(keep=False)].unique()
    print(f"Nombre de timestamps dupliqués uniques: {len(dup_timestamps)}")
    print(pd.DataFrame({"duplicated_timestamp": dup_timestamps}).head(20))
    
    # Afficher un échantillon des lignes dupliquées pour inspection
    sample_ts = dup_timestamps[:5]
    for ts in sample_ts:
        print(f"\nExemple pour timestamp dupliqué: {ts}")
        print(df_france.loc[ts])
else:
    print("Aucun doublon trouvé sur utc_timestamp.")

# =============================================================================
# 4. ANALYSE DES VALEURS MANQUANTES
# =============================================================================

print("\n" + "="*60)
print("Analyse des valeurs manquantes")
print("="*60)

# Quantification et visualisation des valeurs manquantes
missing_count = df_france.isna().sum()
missing_pct = (missing_count / len(df_france)) * 100
missing_df_all = (
    pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    .sort_values("missing_pct", ascending=False)
)
print(missing_df_all)

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

# Périodes avec valeurs manquantes pour IT_NORD_FR_price_day_ahead
col = "IT_NORD_FR_price_day_ahead"
mask = df_france[col].isna()

if not mask.any():
    print(f"Aucune valeur manquante pour {col}.")
else:
    # Numéroter les runs (changes de state)
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

    print(f"Nombre de périodes disjointes avec des NaN pour {col}: {len(runs)}")
    print(runs)

    overall = pd.Series({
        "first_nan": runs["start"].min(),
        "last_nan": runs["end"].max(),
        "total_nan_points": int(mask.sum()),
        "total_points": len(df_france),
        "nan_pct": mask.mean() * 100
    })
    print(overall)

# =============================================================================
# 5. PRÉTRAITEMENT DES DONNÉES
# =============================================================================

print("\n" + "="*60)
print("Prétraitement des données")
print("="*60)

# Renommer les colonnes pour plus de clarté
df_france.rename(columns={
    'IT_NORD_FR_price_day_ahead': 'price',
    'FR_load_actual_entsoe_transparency': 'load_actual',
    'FR_load_forecast_entsoe_transparency': 'load_forecast',
    'FR_solar_generation_actual': 'solar_generation',
    'FR_wind_onshore_generation_actual': 'wind_generation'
}, inplace=True)

# Filtrer sur la période d'intérêt
start_date = '2015-01-05'
end_date = '2017-12-05'
df_france = df_france.loc[start_date:end_date]

print(f"Période filtrée: {start_date} → {end_date}")
print(f"Nouvelles dimensions: {df_france.shape}")

# Vérifier les valeurs manquantes après filtrage
missing_count = df_france.isna().sum()
missing_pct = (missing_count / len(df_france)) * 100
missing_df_all = (
    pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    .sort_values("missing_pct", ascending=False)
)
print("\nValeurs manquantes après filtrage:")
print(missing_df_all)

# Utiliser l'interpolation linéaire pour les quelques NaN restants
df_france.interpolate(method='linear', inplace=True)

# Vérification finale
missing_count = df_france.isna().sum()
missing_pct = (missing_count / len(df_france)) * 100
missing_df_all = (
    pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    .sort_values("missing_pct", ascending=False)
)
print("\nValeurs manquantes après interpolation:")
print(missing_df_all)

print("\n Prétraitement terminé")
print(df_france.head())

# =============================================================================
# 6. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# =============================================================================

print("\n" + "="*60)
print("Analyse Exploratoire des Données")
print("="*60)

# 6.1 Distribution du prix day-ahead
fig = px.histogram(
    df_france,
    x="price",
    nbins=25,
    marginal="rug",           
    opacity=0.75,
    title="Distribution du Day-Ahead Price (€/MWh)",
    labels={"price": "Price (€/MWh)", "count": "Frequency"},
)
fig.update_layout(bargap=0.02, template="plotly_white")
fig.show()

# 6.2 Evolution temporelle du prix
fig = px.line(
    df_france.reset_index(),
    x=df_france.index.name or "index",
    y="price",
    title="Evolution du Day-Ahead Price (2015-2017)",
    labels={
        df_france.index.name or "index": "Date",
        "price": "Price (€/MWh)",
    },
)
fig.update_layout(template="plotly_white")
fig.show()

# 6.3 Analyse de la saisonnalité
df_seasonal = df_france.copy()
df_seasonal['month'] = df_seasonal.index.month_name()
df_seasonal['day_of_week'] = df_seasonal.index.day_name()
df_seasonal['hour'] = df_seasonal.index.hour

# Saisonnalité annuelle : distribution mensuelle des prix
fig = px.box(
    df_seasonal,
    x="month",
    y="price",
    points="outliers", 
    title="Saisonnalité annuelle : distribution mensuelle des prix",
    labels={"month": "Mois", "price": "Prix (€/MWh)"},
    template="plotly_white",
)
fig.update_layout(xaxis=dict(dtick=1))
fig.show()

# Saisonnalité hebdomadaire : distribution des prix par jour de la semaine
order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fig = px.box(
    df_seasonal,
    x="day_of_week",
    y="price",
    category_orders={"day_of_week": order_days},
    title="Saisonnalité hebdomadaire : distribution des prix par jour de la semaine",
    labels={"day_of_week": "Jour de la semaine", "price": "Prix (€/MWh)"},
    template="plotly_white",
)
fig.show()

# Saisonnalité journalière : prix moyen par heure
hourly_mean = df_seasonal.groupby('hour')['price'].mean().reset_index()
fig = px.line(
    hourly_mean,
    x="hour",
    y="price",
    markers=True,
    title="Saisonnalité journalière : prix moyen par heure",
    labels={"hour": "Heure", "price": "Prix moyen (€/MWh)"},
    template="plotly_white",
)
fig.update_layout(xaxis=dict(dtick=1))
fig.show()

# Heatmap : prix moyen par jour de la semaine et heure
pivot_day_hour = df_seasonal.pivot_table(
    values='price',
    index='day_of_week',
    columns='hour',
    aggfunc='mean'
)
pivot_day_hour = pivot_day_hour.reindex(order_days)

fig = go.Figure(data=go.Heatmap(
    z=pivot_day_hour.values,
    x=pivot_day_hour.columns,
    y=pivot_day_hour.index,
    colorscale='RdYlBu_r',
    colorbar=dict(title="Prix (€/MWh)"),
))
fig.update_layout(
    title="Prix moyen par jour de la semaine et heure",
    xaxis_title="Heure",
    yaxis_title="Jour de la semaine",
    template="plotly_white",
)
fig.show()

# 6.4 Corrélation entre les variables
correlation_matrix = df_france.corr()

fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorscale='RdBu_r',
    zmid=0,
    text=correlation_matrix.values,
    texttemplate='%{text:.2f}',
    colorbar=dict(title="Corrélation"),
))
fig.update_layout(
    title="Matrice de corrélation",
    template="plotly_white",
    width=800,
    height=700,
)
fig.show()

# 6.5 Décomposition saisonnière (STL)
print("Décomposition saisonnière en cours...")

stl = STL(df_france['price'], seasonal=13, period=24*7)
result = stl.fit()

fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=('Prix observé', 'Tendance', 'Saisonnalité', 'Résidu'),
    vertical_spacing=0.05,
)

fig.add_trace(
    go.Scatter(x=df_france.index, y=df_france['price'], mode='lines', name='Observé'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df_france.index, y=result.trend, mode='lines', name='Tendance'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=df_france.index, y=result.seasonal, mode='lines', name='Saisonnalité'),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=df_france.index, y=result.resid, mode='lines', name='Résidu'),
    row=4, col=1
)

fig.update_layout(
    height=900,
    title_text="Décomposition STL du prix de l'électricité",
    showlegend=False,
    template="plotly_white",
)
fig.show()

# =============================================================================
# 7. FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*60)
print("Feature Engineering")
print("="*60)

df_featured = df_france.copy()

# 7.1 Features temporelles
df_featured['hour'] = df_featured.index.hour
df_featured['day_of_week'] = df_featured.index.dayofweek
df_featured['day_of_month'] = df_featured.index.day
df_featured['month'] = df_featured.index.month
df_featured['year'] = df_featured.index.year
df_featured['quarter'] = df_featured.index.quarter
df_featured['week_of_year'] = df_featured.index.isocalendar().week.astype(int)
df_featured['is_weekend'] = (df_featured['day_of_week'] >= 5).astype(int)

# 7.2 Features cycliques
df_featured['hour_sin'] = np.sin(2 * np.pi * df_featured['hour'] / 24)
df_featured['hour_cos'] = np.cos(2 * np.pi * df_featured['hour'] / 24)
df_featured['day_sin'] = np.sin(2 * np.pi * df_featured['day_of_week'] / 7)
df_featured['day_cos'] = np.cos(2 * np.pi * df_featured['day_of_week'] / 7)
df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)

# 7.3 Features de lag (décalage)
for lag in [1, 2, 3, 24, 48, 168]:
    df_featured[f'price_lag_{lag}'] = df_featured['price'].shift(lag)
    df_featured[f'load_actual_lag_{lag}'] = df_featured['load_actual'].shift(lag)

# 7.4 Moyennes mobiles
for window in [24, 168]:
    df_featured[f'price_rolling_mean_{window}'] = df_featured['price'].rolling(window=window).mean()
    df_featured[f'price_rolling_std_{window}'] = df_featured['price'].rolling(window=window).std()
    df_featured[f'load_actual_rolling_mean_{window}'] = df_featured['load_actual'].rolling(window=window).mean()

# 7.5 Différences
df_featured['price_diff_1'] = df_featured['price'].diff(1)
df_featured['price_diff_24'] = df_featured['price'].diff(24)

# 7.6 Ratios et interactions
df_featured['load_forecast_error'] = df_featured['load_forecast'] - df_featured['load_actual']
df_featured['renewable_generation'] = df_featured['solar_generation'] + df_featured['wind_generation']
df_featured['renewable_ratio'] = df_featured['renewable_generation'] / (df_featured['load_actual'] + 1)

# Supprimer les lignes avec NaN créées par les lags et moyennes mobiles
df_featured = df_featured.dropna()

print(f"Features créées")
print(f"Nouvelles dimensions: {df_featured.shape}")
print(f"\nColonnes disponibles:")
print(df_featured.columns.tolist())

# =============================================================================
# 8. PRÉPARATION DES DONNÉES POUR LE MODÈLE
# =============================================================================

print("\n" + "="*60)
print("Préparation des données pour le modèle")
print("="*60)

# Séparer features (X) et target (y)
X = df_featured.drop(columns=['price'])
y = df_featured['price']

# Split temporel : 80% train, 20% test
split_date = df_featured.index[int(len(df_featured) * 0.8)]
X_train = X.loc[X.index < split_date]
X_test = X.loc[X.index >= split_date]
y_train = y.loc[y.index < split_date]
y_test = y.loc[y.index >= split_date]

print(f" Ensemble d'entraînement: {X_train.shape}")
print(f" Ensemble de test: {X_test.shape}")
print(f" Date de split: {split_date}")

# =============================================================================
# 9. MODÈLE DE BASE (LIGHTGBM SANS OPTIMISATION)
# =============================================================================

print("\n" + "="*60)
print("Entraînement du modèle LightGBM de base")
print("="*60)

# Entraînement du modèle
model = lgb.LGBMRegressor(random_state=42)
print(" Entraînement en cours...")
model.fit(X_train, y_train)
print(" Entraînement terminé")

# Prédictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("\n" + "="*60)
print("Performance du modèle de base")
print("="*60)
print(f"Mean Absolute Error (MAE): {mae:.2f} €/MWh")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} €/MWh")

# Visualisation
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=y_test.index,
        y=y_test,
        mode="lines",
        name="Actual Price",
        line=dict(color="royalblue", width=2),
        opacity=0.8,
    )
)
fig.add_trace(
    go.Scatter(
        x=y_test.index,
        y=y_pred,
        mode="lines",
        name="Predicted Price",
        line=dict(color="firebrick", width=2, dash="dash"),
    )
)
fig.update_layout(
    title="Actual vs. Predicted Prices (Test Set) - Modèle de base",
    xaxis_title="Date",
    yaxis_title="Price (€/MWh)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.show()

# =============================================================================
# 10. OPTIMISATION DES HYPERPARAMÈTRES AVEC OPTUNA
# =============================================================================

print("\n" + "="*60)
print("Optimisation des hyperparamètres avec Optuna")
print("="*60)

def objective(trial, X_train, y_train, X_val, y_val):
    """Fonction objectif pour Optuna"""
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }
    
    # Entraînement
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    
    return mae

# Split Train / Validation / Test
# Test sur les 3 derniers mois
test_start_date = df_featured.index.max() - pd.DateOffset(months=3)
# Validation sur les 3 mois avant le test
val_start_date = test_start_date - pd.DateOffset(months=3)

X_train_full = X.loc[X.index < val_start_date]
y_train_full = y.loc[y.index < val_start_date]
X_val = X.loc[(X.index >= val_start_date) & (X.index < test_start_date)]
y_val = y.loc[(y.index >= val_start_date) & (y.index < test_start_date)]
X_test = X.loc[X.index >= test_start_date]
y_test = y.loc[y.index >= test_start_date]

print(f"Ensemble d'entraînement: {X_train_full.shape}")
print(f"Ensemble de validation: {X_val.shape}")
print(f"Ensemble de test: {X_test.shape}\n")

# Lancement de l'optimisation
print("Début du réglage d'hyperparamètres avec Optuna...")
print("(Cela peut prendre plusieurs minutes)")

study = optuna.create_study(direction='minimize')
study.optimize(
    lambda trial: objective(trial, X_train_full, y_train_full, X_val, y_val), 
    n_trials=50
)

print("Réglage terminé!")
print(f"Meilleur MAE: {study.best_value:.4f} €/MWh")
print("\nMeilleurs hyperparamètres:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# =============================================================================
# 11. ENTRAÎNEMENT DU MODÈLE FINAL
# =============================================================================

print("\n" + "="*60)
print("Entraînement du modèle final")
print("="*60)

# Réentraînement sur l'ensemble complet (train + validation)
X_train_final = pd.concat([X_train_full, X_val])
y_train_final = pd.concat([y_train_full, y_val])

best_params = study.best_params
final_model = lgb.LGBMRegressor(**best_params, random_state=42)

print("Entraînement en cours...")
final_model.fit(X_train_final, y_train_final)
print("Entraînement terminé")

# Evaluation finale
final_preds = final_model.predict(X_test)
final_mae = mean_absolute_error(y_test, final_preds)
final_rmse = root_mean_squared_error(y_test, final_preds)

print("\n" + "="*60)
print("Performance finale du modèle sur l'ensemble de test")
print("="*60)
print(f"Mean Absolute Error (MAE): {final_mae:.2f} €/MWh")
print(f"Root Mean Squared Error (RMSE): {final_rmse:.2f} €/MWh")

# Visualisation finale
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=y_test.index,
        y=y_test,
        mode="lines",
        name="Actual Price",
        line=dict(color="royalblue", width=2),
        opacity=0.8,
    )
)
fig.add_trace(
    go.Scatter(
        x=y_test.index,
        y=final_preds,
        mode="lines",
        name="Predicted Price",
        line=dict(color="firebrick", width=2, dash="dash"),
    )
)
fig.update_layout(
    title="Actual vs Predicted Prices (Test Set) - Modèle Final Optimisé",
    xaxis_title="Date",
    yaxis_title="Price (€/MWh)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.show()

# =============================================================================
# 12. ANALYSE SHAP (EXPLICABILITÉ)
# =============================================================================

print("\n" + "="*60)
print("Analyse SHAP - Explicabilité du modèle")
print("="*60)

# Réentraînement sur l'ensemble complet pour SHAP
# (Utilisation des meilleurs paramètres trouvés par Optuna)
best_params_shap = {
    'n_estimators': 965,
    'learning_rate': 0.023305698375173753,
    'num_leaves': 62, 
    'max_depth': 6, 
    'min_child_samples': 73, 
    'feature_fraction': 0.9107029697230191, 
    'bagging_fraction': 0.684479044902335, 
    'bagging_freq': 3, 
    'lambda_l1': 0.029500110573817455, 
    'lambda_l2': 0.16428913564785613,
    'random_state': 42
}

print(" Entraînement du modèle complet pour SHAP...")
shap_model = lgb.LGBMRegressor(**best_params_shap)
shap_model.fit(X, y)
print("Entraînement terminé")

# Calcul des valeurs SHAP
test_start_date = df_featured.index.max() - pd.DateOffset(months=3)
X_test_shap = X.loc[X.index >= test_start_date]

print("Calcul des valeurs SHAP...")
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_test_shap)
print("Valeurs SHAP calculées")

# Visualisation 1: Importance globale des features
if isinstance(shap_values, list):
    shap_array = shap_values[0]
else:
    shap_array = shap_values

shap_importance = (
    pd.DataFrame({
        "feature": X_test_shap.columns,
        "mean_abs_shap": np.abs(shap_array).mean(axis=0),
    })
    .sort_values("mean_abs_shap", ascending=True)
    .reset_index(drop=True)
)

fig = px.bar(
    shap_importance,
    x="mean_abs_shap",
    y="feature",
    orientation="h",
    title="Importance globale des caractéristiques (|SHAP| moyen)",
    labels={"mean_abs_shap": "|SHAP| moyen", "feature": "Caractéristique"},
    template="plotly_white",
)
fig.update_layout(margin=dict(l=120, r=40, t=60, b=40))
fig.show()

# Visualisation 2: SHAP Summary Plot (Beeswarm)
index_name = X_test_shap.index.name or "index"
shap_frame = pd.DataFrame(shap_array, columns=X_test_shap.columns)
shap_frame[index_name] = X_test_shap.index
shap_long = shap_frame.melt(id_vars=index_name, var_name="feature", value_name="shap_value")

feature_long = (
    X_test_shap.reset_index()
    .melt(id_vars=index_name, var_name="feature", value_name="feature_value")
)

merged = shap_long.merge(feature_long, on=[index_name, "feature"])

fig = px.strip(
    merged,
    x="shap_value",
    y="feature",
    color="feature_value",
    title="SHAP Summary Plot (Beeswarm) - Distribution et impact des caractéristiques",
    labels={
        "shap_value": "Valeur SHAP", 
        "feature": "Caractéristique", 
        "feature_value": "Valeur de la caractéristique"
    },
    template="plotly_white",
    orientation="h",
    width=900,
    height=600,
)
fig.update_traces(opacity=0.6, jitter=0.35, marker=dict(symbol="circle", size=6))
fig.update_layout(margin=dict(l=140, r=40, t=60, b=40))
fig.show()

# =============================================================================
# FIN DU SCRIPT
# =============================================================================

print("\n" + "="*60)
print("Analyse terminée avec succès!")
print("="*60)
print(f"\nRésumé des résultats:")
print(f"  - MAE final: {final_mae:.2f} €/MWh")
print(f"  - RMSE final: {final_rmse:.2f} €/MWh")
print(f"  - Nombre de features: {X.shape[1]}")
print(f"  - Période de test: {X_test.index.min()} → {X_test.index.max()}")
print(f"Graphiques générés et modèle entraîné avec succès!")
