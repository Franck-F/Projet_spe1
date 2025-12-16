#!/usr/bin/env python
# coding: utf-8

# 1.  **Approche Statistique (SARIMAX)** : Comprendre les dynamiques temporelles et l'impact des fondamentaux (Charge, Vent, Solaire).
# 2.  **Approche Machine Learning (LightGBM)** : Capturer les non-linéarités complexes du marché.
# 3.  **Analyse de la Volatilité** : Expliquer les pics de prix par la théorie économique du "Merit Order".
# 
# ---

# In[13]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

# ## Chargement de données features 

# In[14]:


df = pd.read_csv('../../data/processed/df_features_france_2015_2017.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')
target_col = 'price_day_ahead'
print(f"Dataset chargé : {df.shape} lignes.")
df.head()

# ## Modélisation Statistique : SARIMAX
# 
# ### Pourquoi SARIMAX ?
# Le modèle **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) permet de modéliser une série temporelle en se basant sur son passé (AutoRegressive) et ses erreurs passées (Moving Average), tout en gérant la saisonnalité (Seasonal).
# 
# Cependant, le prix de l'électricité n'est pas juste une suite de nombres abstraits : il est piloté par des réalités physiques. C'est pourquoi nous utilisons **SARIMAX** (avec **X** pour eXogenous variables).
# 
# ### Les Variables Explicatives (Exogènes)
# Nous intégrons ici les facteurs fondamentaux du marché :
# -   **La Consommation (`load`)** : C'est le facteur #1. Plus la demande est forte, plus on doit activer des centrales coûteuses (gaz/charbon).
# -   **La Production Renouvelable & Météo** : `wind` et `solar` sont les productions, mais `wind_speed` et `cloud_cover` (nébulosité) sont les causes racines météorologiques. Les ajouter augmente la précision.
# -   **Le Nucléaire (`nuclear`)** : Base de la production française, offre stable et peu coûteuse.
# -   **La Température (`temperature`)** : Driver indirect de la demande (chauffage/clim).
# 

# In[15]:


# Sélection des variables exogènes disponibles
exog_vars = ['load', 'solar', 'wind', 'nuclear', 'temperature', 'cloud_cover', 'wind_speed']
available_exog = [c for c in exog_vars if c in df.columns]
print(f"Variables Exogènes utilisées : {available_exog}")

# Agrégation journalière (Moyenne)
cols_to_use = [target_col] + available_exog
df_daily = df[cols_to_use].resample('D').mean().dropna()

# Split Chronologique (On garde la fin pour le test)
train_size = int(len(df_daily) * 0.95)
train, test = df_daily.iloc[:train_size], df_daily.iloc[train_size:]

y_train = train[target_col]
X_train = train[available_exog]
y_test = test[target_col]
X_test = test[available_exog]

print(f"Entraînement sur {len(train)} jours. Test sur {len(test)} jours.")

# Modèle SARIMAX (1,1,1) x (0,1,1,7)
# Saisonnalité hebdomadaire (7) car on observe souvent des cycles semaine/week-end
model_sarima = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
results_sarima = model_sarima.fit(disp=False)


# In[22]:



# Prédiction
forecast = results_sarima.get_forecast(steps=len(test), exog=X_test)
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()

# Scores
mae = mean_absolute_error(y_test, mean_forecast)
rmse = root_mean_squared_error(y_test, mean_forecast)
r2 = r2_score(y_test, mean_forecast)
mape = mean_absolute_percentage_error(y_test, mean_forecast)
print(f"MAE: {mae:.2f}€/MWh | RMSE: {rmse:.2f} | R2: {r2:.2f} | MAPE: {mape:.2f}")


# In[17]:


# Visualisation
fig = go.Figure()
train_visu = y_train.iloc[-180:] # Zoom sur les derniers mois du train
fig.add_trace(go.Scatter(x=train_visu.index, y=train_visu, name='Historique (Train)', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Réalité (Test)', line=dict(color='green')))
fig.add_trace(go.Scatter(x=y_test.index, y=mean_forecast, name='Prédiction SARIMAX', line=dict(color='red', dash='dash')))
fig.add_trace(go.Scatter(
    x=y_test.index.tolist() + y_test.index.tolist()[::-1],
    y=conf_int.iloc[:, 1].tolist() + conf_int.iloc[:, 0].tolist()[::-1],
    fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", showlegend=True, name='Intervalle de Confiance 95%'
))
fig.update_layout(title=f'SARIMAX (MAE: {mae:.2f} €)', template='plotly_white')
fig.show()

# ## Modélisation Avancée : LightGBM
# 
# Contrairement à SARIMAX qui est un modèle linéaire (une ligne droite + des cycles), LightGBM est un modèle à base d'arbres de décision (Gradient Boosting). Voici pourquoi c'est crucial pour l'électricité :
# 
# 1.  **Gestion des Effets de Seuil (Non-linéarité)** :
#     *   *Logique linéaire (SARIMAX)* : "Si la demande augmente de 1GW, le prix augmente de 10€, quel que soit le niveau actuel."
#     *   *Réalité physique* : Si la demande est faible, +1GW ne change rien (on utilise du nucléaire pas cher). Mais si le réseau est saturé, +1GW oblige à démarrer une centrale à gaz ou charbon très chère, faisant exploser le prix. LightGBM capture ces seuils ("SI demande > 60GW ALORS prix +++").
# 
# 2.  **Interactions Complexes** :
#     LightGBM détecte automatiquement des combinaisons comme : *"S'il fait froid (demande chauffage) ET qu'il n'y a pas de vent (pas d'éolien), alors le prix s'envole"*. Un modèle classique doit être programmé manuellement pour voir cette interaction.
# 
# ### Optimisation des Hyperparamètres (Grid Search)
# Nous cherchons la meilleure configuration de l'algorithme pour minimiser l'erreur.

# In[18]:


# Préparation Données Horaires
df_lgb = df.dropna().copy()

# Encodage des catégories (Saison)
for col in df_lgb.select_dtypes(include=['object']).columns:
    df_lgb[col] = df_lgb[col].astype('category').cat.codes

features = [c for c in df_lgb.columns if c not in [target_col, 'price_rolling_mean_24h', 'price_rolling_std_24h']]
X = df_lgb[features]
y = df_lgb[target_col]

# Échantillon récent pour l'optimisation 
X_sample = X.iloc[-20000:]
y_sample = y.iloc[-20000:]

param_grid = {
    'num_leaves': [31, 50, 70],         # Complexité des arbres
    'learning_rate': [0.01, 0.05, 0.1], # Vitesse d'apprentissage
    'n_estimators': [100, 200, 500],     # Nombre d'arbres
    'n_depth': [-1, 5, 10] # Profondeur 
}

gbm = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
tscv = TimeSeriesSplit(n_splits=3) # Validation croisée temporelle stricte

grid = GridSearchCV(gbm, param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
grid.fit(X_sample, y_sample)

print(f"Meilleure config : {grid.best_params_}")


# In[26]:



# Score MAE sur l'échantillon d'optimisation
best_preds = grid.predict(X_sample)
mae_lgb = mean_absolute_error(y_sample, best_preds)
rmse_lgb = root_mean_squared_error(y_sample, best_preds)
r2_lgb = r2_score(y_sample, best_preds)
mape_lgb = mean_absolute_percentage_error(y_sample, best_preds)
print(f"MAE LightGBM (Optimisé) : {mae_lgb:.2f} €/MWh | RMSE : {rmse_lgb:.2f} €/MWh | R2 : {r2_lgb:.2f} | MAPE : {mape_lgb:.2f}%")

# ## Analyse de la Volatilité : La courbe en "Crosse de Hockey"
# 
# Les prix de l'électricité sont connus pour leur volatilité extrême. Ce phénomène s'explique par la courbe d'offre du marché, souvent appelée **"Merit Order"**.
# 
# ### Explication Économique
# -   **La Base (Plat)** : Les premiers MW sont fournis par les EnR (Vent, Solaire) et le Nucléaire. Leur coût marginal est faible. Tant que la demande reste dans cette zone, le prix est bas et stable.
# -   **La Pointe (Verticale)** : Quand la demande dépasse les capacités de base, on appelle les centrales à gaz ou charbon. Elles sont chères (coût du combustible + CO2). Dès qu'elles fixent le prix marginal, le prix du marché saute brutalement.
# 
# Le graphique ci-dessous visualise cette relation non-linéaire entre la **Charge (Load)** et le **Prix**.

# In[20]:


df['volatility'] = df[target_col].rolling('24h').std()

# Définition des pics (> 95e percentile)
threshold = df[target_col].quantile(0.95)
df['status'] = df[target_col].apply(lambda x: 'Pic de Prix' if x > threshold else 'Normal')

fig = px.scatter(
    df.iloc[::10], # Échantillonnage pour lisibilité
    x='load',
    y=target_col,
    color='status',
    color_discrete_map={'Normal': 'lightgray', 'Pic de Prix': 'red'},
    title='Relation Charge vs Prix : La "Crosse de Hockey"',
    labels={'load': 'Consommation (MW)', 'price_day_ahead': 'Prix (€/MWh)'},
    opacity=0.6
)
fig.update_layout(template='plotly_white')
fig.show()

# **Observation** : On voit clairement que pour une consommation faible (< 50GW), les points gris sont plats. Passé un certain seuil, les points rouges s'envolent verticalement. C'est la signature visuelle de la tension sur le réseau.
