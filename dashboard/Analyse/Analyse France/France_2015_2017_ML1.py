#!/usr/bin/env python
# coding: utf-8

# # ML pour la France

# In[12]:


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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error,mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
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

# ## Analyse des Features Créées

# In[13]:


df_featured = pd.read_csv('../../data/processed/df_features_france_2015_2017.csv',
                        parse_dates=['utc_timestamp'], index_col='utc_timestamp',
                        low_memory=False)
df_featured.head()

# In[14]:


# Calculer les corrélations
numeric_cols = df_featured.select_dtypes(include=[np.number]).columns.tolist()
feature_corr = df_featured[numeric_cols].corr()['price_day_ahead'].drop('price_day_ahead').sort_values(ascending=False)


# In[15]:


# HEATMAP CORRÉLATIONS

corr_matrix = df_featured[numeric_cols].corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=np.round(corr_matrix.values, 3),
    texttemplate='%{text:.2f}',
    textfont={"size": 9},
    colorbar=dict(title="Corrélation")
))

fig.update_layout(
    title="<b>Matrice de Corrélation - Top 15 Features + Target</b>",
    height=1500,
    width=1500
)
fig.show()


# In[16]:


# Visualisation de TOUTES les corrélations
fig = go.Figure()

fig.add_trace(go.Bar(
    x=feature_corr.values,
    y=feature_corr.index,
    orientation='h',
    marker=dict(
        color=['green' if x > 0 else 'red' for x in feature_corr.values]
    ),
    text=[f"{x:.3f}" for x in feature_corr.values],
    textposition='auto',
    hovertemplate='<b>%{y}</b><br>Corrélation: %{x:.4f}<extra></extra>'
))

fig.add_vline(x=0, line_dash="dash", line_color="black")

fig.update_layout(
    title="<b>Toutes les Corrélations - Features vs Prix</b>",
    xaxis_title="Corrélation",
    template="plotly_white",
    height=800,
    margin=dict(l=300)
)
fig.show()


# ## PRÉPARATION DES DONNÉES POUR ML

# In[17]:


# Séparer features et target
X = df_featured.drop('price_day_ahead', axis=1)
y = df_featured['price_day_ahead']


# * Encodage des données

# In[18]:


# Encodage de la saison
X_encoded = X.copy()
season_encoding = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
X_encoded['season'] = X_encoded['season'].map(season_encoding)

# Encodage de la semaine 
X_encoded['week'] = X_encoded.index.isocalendar().week

# Encodage du mois
X_encoded['month'] = X_encoded.index.month

# Encodage du jour de la semaine
X_encoded['dayofweek'] = X_encoded.index.dayofweek

# Encodage de l'heure
X_encoded['hour'] = X_encoded.index.hour

# Encodage de la date
X_encoded['date'] = X_encoded.index.date

# * Split train/test

# In[19]:



# Split temporel (80/20)
split_idx = int(len(X_encoded) * 0.8)
X_train = X_encoded[:split_idx]
X_test = X_encoded[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]


# In[20]:


df_featured.index = pd.to_datetime(df_featured.index)

print(f"Train set : {X_train.shape} observations ({len(X_train)/len(X_encoded)*100:.1f}%)")
print(f"Test set : {X_test.shape} observations ({len(X_test)/len(X_encoded)*100:.1f}%)")
print(f"Features : {X_train.shape[1]}")
print(f"Période train : {df_featured.index[0]} à {df_featured.index[split_idx-1]} ")
print(f"Période test : {df_featured.index[split_idx]} à {df_featured.index[-1]} ")

# * Normalisation des données

# In[ ]:


# Définition des colonnes à exclure
columns_to_drop = ['day_name', 'season_lbl', 'season', 'date', 'utc_timestamp']
columns_to_drop = [c for c in columns_to_drop if c in X_train.columns]

X_train_numeric = X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns])
X_test_numeric = X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns])

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)


# ## Modélisation

# * **LightGBM**

# In[40]:


# Entraînement du modèle LightGBM
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# In[41]:


# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# In[42]:


# Evaluation de la performance
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)

print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f} | MAPE: {MAPE:.2f}%")


# In[44]:


# Visualisation des prédictions
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Réel', line=dict(color='#2E7D32', width=3)))
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, name='Prédiction', line=dict(color='#FF6F00', width=2)))
fig.update_layout(title=f'Prédiction vs Réel', height=500)
fig.show()


# In[45]:



# Erreur d'approximation
errors = y_test.values - y_pred
fig = go.Figure()
fig.add_trace(go.Histogram(x=errors, nbinsx=50, name='Erreur'))
fig.update_layout(
    title="Distribution des Erreurs de Prédiction",
    xaxis_title="Erreur (€/MWh)",
    yaxis_title="Fréquence",
    template="plotly_white"
)
fig.show()


# In[ ]:


# Sauvegarder
# import joblib
# joblib.dump(model, 'model_lgbm_france.pkl')
# joblib.dump(scaler, 'scaler_france.pkl')
# print("\n✓ Modèles sauvegardés")

# ## Analyse SHAP

# In[46]:


# Créer l'explainer
explainer = shap.TreeExplainer(model)


# In[47]:


# Calculer les valeurs SHAP 
shap_values = explainer.shap_values(X_test_scaled)
print(f"SHAP values shape : {shap_values.shape}")
print(f"X_test shape : {X_test_scaled.shape}")


# In[48]:


# --- Visualisation SHAP (barres) ---
if isinstance(shap_values, list):
    shap_array = shap_values[0]
else:
    shap_array = shap_values

# Récupérer les noms des features (colonnes de X_test_numeric)
feature_names = X_test_numeric.columns.tolist()

shap_importance = (
    pd.DataFrame({
        "feature": feature_names,
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

# In[49]:


# Préparer les données
feature_names = X_test_numeric.columns.tolist()
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# Calculer l'importance moyenne pour trier les features
mean_abs_shap = np.abs(shap_array).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=True)

# toutes les features
all_features = feature_importance['feature'].tolist()

# DataFrame pour Plotly
plot_data = []
for i, feat in enumerate(all_features):
    feat_idx = feature_names.index(feat)
    # Ajouter un jitter aléatoire sur l'axe y pour l'effet beeswarm
    n_points = len(shap_array)
    y_jitter = np.random.uniform(-0.3, 0.3, n_points)
    
    plot_data.append(pd.DataFrame({
        'feature': feat,
        'feature_num': i,
        'y_position': i + y_jitter,
        'shap_value': shap_array[:, feat_idx],
        'feature_value': X_test_scaled_df.iloc[:, feat_idx]
    }))

df_plot = pd.concat(plot_data, ignore_index=True)

# graphique
fig = px.scatter(
    df_plot,
    x='shap_value',
    y='y_position',
    color='feature_value',
    title='resume plot beeswarm ',
    labels={
        'shap_value': 'Valeur SHAP (impact sur la prédiction)',
        'feature_value': 'Valeur de la caractéristique'
    },
    color_continuous_scale='RdBu_r',
    height=600,
    width=1000
)

fig.update_traces(
    marker=dict(size=5, opacity=0.6, line=dict(width=0))
)

fig.update_layout(
    template='plotly_white',
    margin=dict(l=150, r=100, t=80, b=60),
    font=dict(size=11),
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(len(all_features))),
        ticktext=all_features,
        title='Caractéristique'
    ),
    coloraxis_colorbar=dict(
        title="Valeur<br>normalisée",
        thickness=15,
        len=0.7
    ),
    showlegend=False
)

fig.show()

# In[50]:


# Calculer les indices des top features
mean_abs_shap = np.abs(shap_array).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[-3:][::-1]  # Top 3 features (ordre décroissant)

sample_size = len(X_test_scaled)  # ou un nombre plus petit si nécessaire

# Scatter plot - Feature vs SHAP value
for idx in top_indices:
    feature = X_test_numeric.columns[idx]
    
    # Préparer les dates pour le hover (si disponibles)
    if hasattr(test_dates, 'strftime'):
        # test_dates est un DatetimeIndex
        hover_text = test_dates[:sample_size].strftime('%Y-%m-%d %H:%M')
    elif isinstance(test_dates, pd.Series) and pd.api.types.is_datetime64_any_dtype(test_dates):
        # test_dates est une Series datetime
        hover_text = test_dates.iloc[:sample_size].dt.strftime('%Y-%m-%d %H:%M')
    else:
        # Pas de dates disponibles, utiliser les indices
        hover_text = [f"Index {i}" for i in range(sample_size)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_test_numeric.iloc[:sample_size, idx],
        y=shap_array[:sample_size, idx],
        mode='markers',
        marker=dict(
            size=6, 
            color=X_test_numeric.iloc[:sample_size, idx], 
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(title="Valeur<br>feature")
        ),
        text=hover_text,
        hovertemplate='<b>Date:</b> %{text}<br><b>Valeur:</b> %{x:.2f}<br><b>SHAP:</b> %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=f'SHAP Dependence Plot - {feature}',
        xaxis_title=f'{feature}',
        yaxis_title='Valeur SHAP (impact sur la prédiction)',
        template="plotly_white",
        height=500,
        width=800
    )
    fig.show()
