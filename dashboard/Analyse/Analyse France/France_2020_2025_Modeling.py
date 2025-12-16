#!/usr/bin/env python
# coding: utf-8

# # Modélisation - Prix Électricité France (2020-2025)

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ### Préparation Modélisation

# In[12]:


# Charger le dataset avec features
df_ml = pd.read_csv('../../data/processed/df_ml_france_2020_2025.csv', 
                    parse_dates=['utc_timestamp'], 
                    index_col='utc_timestamp')
print(f"Dataset chargé: {df_ml.shape}")
df_ml.head()

# In[13]:


# Split temporel (Test = 9 derniers mois 2025)
test_start_date = '2025-04-01'
train = df_ml[df_ml.index < test_start_date]
test = df_ml[df_ml.index >= test_start_date]

print(f"Train: {train.index.min()} → {train.index.max()} ({len(train):,} samples)")
print(f"Test: {test.index.min()} → {test.index.max()} ({len(test):,} samples)")


# In[14]:


# Traitement outliers sur train uniquement
target = 'price_day_ahead'
Q1, Q3 = train[target].quantile([0.25, 0.75])
IQR = Q3 - Q1
train = train.copy()
train[target] = train[target].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# In[15]:


# Préparation X, y
target = 'price_day_ahead'

# Colonnes techniques à supprimer (y compris load_bin qui plante LightGBM)
drop_cols_technical = ['day_name', 'season_lbl', 'season', 'price_raw', 'load_bin', 'utc_timestamp']

# Colonnes cibles ou fuites potentielles (Leakage)
# On supprime TOUT ce qui contient 'price_day_ahead' SAUF si c'est un lag ou un rolling
drop_cols_leakage = [c for c in df_ml.columns if target in c and 'lag' not in c and 'rolling' not in c]

# Fusion des listes
drop_cols = list(set(drop_cols_technical + drop_cols_leakage))
# Filtrer pour ne garder que ce qui existe vraiment
drop_cols = [c for c in drop_cols if c in train.columns]

print(f"Colonnes supprimées : {len(drop_cols)}")
print(drop_cols)


# In[16]:



X_train, y_train = train.drop(columns=drop_cols), train[target]
X_test, y_test = test.drop(columns=drop_cols), test[target]

print(f"\nX_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# ### Modélisation LightGBM

# In[17]:


def safe_mape(y_true, y_pred):
    mask = y_true > 1.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

# Baseline
print("\n LightGBM Baseline ")
model_base = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
model_base.fit(X_train, y_train)
preds_base = model_base.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds_base))
mae_base = mean_absolute_error(y_test, preds_base)
r2_base = r2_score(y_test, preds_base)
mape_base = safe_mape(y_test.values, preds_base)
print(f" RMSE_base: {rmse:.2f} | MAE_base: {mae_base:.2f} | R²_base: {r2_base:.3f} | MAPE_base: {mape_base:.2f}%")

# In[18]:


# Optimisé
print("\n  LightGBM Optimisé ")
param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
                'max_depth': [-1, 5, 10],
                'n_estimators': [100, 200, 500]
            }
gbm = lgb.LGBMRegressor(random_state=42, 
                        verbose=-1, 
                        n_jobs=-1)
                        
tscv = TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(gbm, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
grid.fit(X_train.iloc[-len(X_train)//2:], y_train.iloc[-len(y_train)//2:])

print(f"Hyperparamètres optimaux: {grid.best_params_}")


# In[19]:



best_gbm = grid.best_estimator_
best_gbm.fit(X_train, y_train)
preds_optim = best_gbm.predict(X_test)

rmse_optim = root_mean_squared_error(y_test, preds_optim)
mae_optim = mean_absolute_error(y_test, preds_optim)
r2_optim = r2_score(y_test, preds_optim)
mape_optim = safe_mape(y_test.values, preds_optim)
print(f"RMSE_optim: {rmse_optim:.2f} | MAE_optim: {mae_optim:.2f} | R²_optim: {r2_optim:.3f} | MAPE_optim: {mape_optim:.2f}%")
print(f"Amélioration: {mae_base - mae_optim:.2f} €/MWh")

# ## Visualisations

# In[20]:


layout_config = dict(
    font=dict(size=16, family='Arial'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='black'),
    yaxis=dict(showgrid=True, gridcolor='lightgray', linecolor='black')
)

# Baseline
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Réel', line=dict(color='#2E7D32', width=3)))
fig.add_trace(go.Scatter(x=y_test.index, y=preds_base, name='prédit', line=dict(color='red', width=2, dash='dot')))
fig.update_layout(
    title=dict(text=f'<b>LightGBM Baseline</b><br>MAE: {mae_base:.2f} €/MWh | R²: {r2_base:.3f}', font=dict(size=20)),
    xaxis_title='Date', yaxis_title='Prix (€/MWh)', height=600, width=1200, **layout_config
)
fig.show()

# In[ ]:


# Optimisé
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Réel', line=dict(color='#2E7D32', width=3)))
fig.add_trace(go.Scatter(x=y_test.index, y=preds_optim, name='prédit', line=dict(color='#D32F2F', width=2)))
fig.update_layout(
    title=dict(text=f'<b>LightGBM Optimisé</b><br>MAE: {mae_optim:.2f} €/MWh | R²: {r2_optim:.3f}', font=dict(size=20)),
    xaxis_title='Date', yaxis_title='Prix (€/MWh)', height=600, width=1200, **layout_config
)
fig.show()

# In[ ]:


# Comparaison
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Réel', line=dict(color='#2E7D32', width=3)))
fig.add_trace(go.Scatter(x=y_test.index, y=preds_base, name=f'Baseline (MAE={mae_base:.1f})', 
                         line=dict(color='#1976D2', width=1.5, dash='dot'), opacity=0.7))
fig.add_trace(go.Scatter(x=y_test.index, y=preds_optim, name=f'Optimisé (MAE={mae_optim:.1f})', 
                         line=dict(color='#D32F2F', width=2)))
fig.update_layout(
    title=dict(text='<b>Comparaison LightGBM</b>', font=dict(size=22)),
    xaxis_title='Date', yaxis_title='Prix (€/MWh)', height=700, width=1200,
    legend=dict(font=dict(size=14), x=0.02, y=0.98), **layout_config
)
fig.show()

# In[ ]:


# Résidus
residuals_base = y_test - preds_base
residuals_optim = y_test - preds_optim

fig = make_subplots(rows=1, cols=2, subplot_titles=('Résidus Baseline', 'Résidus Optimisé'))
fig.add_trace(go.Scatter(x=y_test.index, y=residuals_base, mode='markers',
                         marker=dict(size=4, color='#1976D2', opacity=0.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=y_test.index, y=residuals_optim, mode='markers',
                         marker=dict(size=4, color='#D32F2F', opacity=0.5)), row=1, col=2)
fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
fig.update_layout(title="<b>Analyse des Résidus</b>", height=500, showlegend=False, font=dict(size=14))
fig.show()

# ## ANALYSE SHAP

# In[ ]:


# coefficients SHAP
X_shap = X_test.iloc[:1000]
explainer = shap.TreeExplainer(best_gbm)
shap_values = explainer.shap_values(X_shap)


# In[ ]:


# visualisation SHAP
shap_importance = np.abs(shap_values).mean(axis=0)

df_shap = pd.DataFrame({
    'Feature': X_shap.columns,
    'Importance': shap_importance
})

# Tri par ordre d'importance
df_shap = df_shap.sort_values(by='Importance', ascending=True)

# Visualisation 
fig = px.bar(df_shap, 
             x='Importance', 
             y='Feature', 
             orientation='h',
             title='<b>Importance des Features (SHAP Global)</b>',
             text_auto='.2f',  
             color='Importance',
             color_continuous_scale='Viridis')

fig.update_layout(
    height=800, 
    xaxis_title="Impact Moyen absolu sur le prix (€/MWh)",
    yaxis_title="",
    font=dict(size=12)
)

fig.show()

# ### SARIMAX

# In[ ]:


# selection des features
features = ['gas', 'coal', 'nuclear', 'solar', 'wind', 'biomass', 'waste', 'load', 'temperature', 'cloud_cover', 'wind_speed']


# In[ ]:


# Split
train_start, train_end = '2020-01-08', '2024-12-31'
test_start = df_ml.index.max() - pd.DateOffset(months=12)

train_mask = (df_ml.index >= train_start) & (df_ml.index <= train_end)
test_mask = df_ml.index >= test_start

print(f'Features: {features}')
print(f'Train: {train_start} -> {train_end}')
print(f'Test: {test_start} -> {df_ml.index.max()}')

X_train = df_ml.loc[train_mask, features].fillna(0)
y_train = df_ml.loc[train_mask, 'price_day_ahead']
X_test = df_ml.loc[test_mask, features].fillna(0)
y_test = df_ml.loc[test_mask, 'price_day_ahead']

# In[ ]:


# Entraînement
model = SARIMAX(y_train, exog=X_train, order=(2, 2, 2), seasonal_order=(0, 0, 0, 6),
                enforce_stationarity=False, enforce_invertibility=False)
fitted = model.fit(disp=False, maxiter=100)


# In[ ]:



# Prédiction
preds = fitted.forecast(steps=len(y_test), exog=X_test)
preds.index = y_test.index

# In[ ]:


# Métriques
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test[mask_mape] - y_pred[mask_mape]) / y_test[mask_mape])) * 100

print(f'MAE  : {mae:.2f} €/MWh |RMSE : {rmse:.2f} €/MWh |R²   : {r2_sarimax:.3f} |MAPE : {mape_sarimax:.2f} %')

# In[ ]:


# Visualisation
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_true, name='Réel', line=dict(color='black', width=1)))
fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, name='prédiction', line=dict(color='#AB63FA', width=2)))
fig.update_layout(title=f'prediction VS réel (MAE: {mae:.2f}€)', xaxis_title='Date', yaxis_title='Prix (€/MWh)', height=600, template='plotly_white')
fig.show()

# #### Comparaison Finale

# In[ ]:


print("\n" + "="*70)
print("RÉSUMÉ COMPARATIF")
print("="*70)
print(f"{'Modèle':<25} {'MAE':<15} {'R²':<10} {'MAPE (%)':<10}")
print("-"*70)
print(f"{'LightGBM Baseline':<25} {mae_base:<15.2f} {rmse_base:<15.2f} {r2_base:<10.3f} {mape_base:<10.2f}")
print(f"{'LightGBM Optimisé':<25} {mae_optim:<15.2f} {rmse_optim:<15.2f} {r2_optim:<10.3f} {mape_optim:<10.2f}")
print(f"{'SARIMAX':<25} {mae_sarimax:<15.2f} {rmse_sarimax:<15.2f} {r2_sarimax:<10.3f} {mape_sarimax:<10.2f}")
print("="*70)

best = min([('LightGBM Baseline', mae_base), ('LightGBM Optimisé', mae_optim), ('SARIMAX', mae_sarimax)], key=lambda x: x[1])
print(f"\n Meilleur: {best[0]} (MAE = {best[1]:.2f} €/MWh)")
