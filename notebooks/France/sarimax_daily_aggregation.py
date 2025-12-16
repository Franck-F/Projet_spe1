"""
SARIMAX avec agrégation JOURNALIÈRE - France 2020-2025
Remplace la section SARIMAX du notebook pour améliorer les performances
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go

print("="*70)
print("SARIMAX - AGRÉGATION JOURNALIÈRE")
print("="*70)

# Selection des features
features = ['gas', 'coal', 'nuclear', 'solar', 'wind', 'biomass', 'waste', 'load', 'temperature', 'cloud_cover', 'wind_speed']

# Vérifier quelles features sont disponibles
available_features = [f for f in features if f in df_ml.columns]
print(f"\nFeatures disponibles: {available_features}")

# ========================================
# AGRÉGATION JOURNALIÈRE (MOYENNE)
# ========================================
print("\n📊 Agrégation des données horaires en données journalières...")

# Créer un DataFrame avec target + features
cols_to_aggregate = ['price_day_ahead'] + available_features
df_daily = df_ml[cols_to_aggregate].resample('D').mean()

print(f"   Données horaires: {len(df_ml):,} observations")
print(f"   Données journalières: {len(df_daily):,} jours")

# ========================================
# SPLIT TRAIN/TEST
# ========================================
# Garder les 12 derniers mois pour le test
test_start = df_daily.index.max() - pd.DateOffset(months=12)
train_end = test_start - pd.Timedelta(days=1)

train_mask = df_daily.index < test_start
test_mask = df_daily.index >= test_start

df_train = df_daily[train_mask]
df_test = df_daily[test_mask]

X_train_daily = df_train[available_features].fillna(method='ffill').fillna(0)
y_train_daily = df_train['price_day_ahead']
X_test_daily = df_test[available_features].fillna(method='ffill').fillna(0)
y_test_daily = df_test['price_day_ahead']

print(f"\n📅 Périodes:")
print(f"   Train: {df_train.index.min().date()} → {df_train.index.max().date()} ({len(df_train)} jours)")
print(f"   Test:  {df_test.index.min().date()} → {df_test.index.max().date()} ({len(df_test)} jours)")

# ========================================
# ENTRAÎNEMENT SARIMAX
# ========================================
print("\n🔧 Entraînement du modèle SARIMAX...")
print("   Order: (2, 1, 2)")
print("   Seasonal Order: (1, 0, 1, 7) - Saisonnalité hebdomadaire")

model_sarimax = SARIMAX(
    y_train_daily, 
    exog=X_train_daily, 
    order=(2, 1, 2),  # (p, d, q)
    seasonal_order=(1, 0, 1, 7),  # (P, D, Q, s) - s=7 pour cycle hebdomadaire
    enforce_stationarity=False, 
    enforce_invertibility=False
)

fitted_sarimax = model_sarimax.fit(disp=False, maxiter=200)
print("   ✅ Entraînement terminé")

# ========================================
# PRÉDICTION
# ========================================
print("\n🔮 Génération des prédictions...")
preds_daily = fitted_sarimax.forecast(steps=len(y_test_daily), exog=X_test_daily)
preds_daily.index = y_test_daily.index

# ========================================
# MÉTRIQUES
# ========================================
mask_mape = y_test_daily > 1.0

mae_sarimax = mean_absolute_error(y_test_daily, preds_daily)
rmse_sarimax = np.sqrt(mean_squared_error(y_test_daily, preds_daily))
r2_sarimax = r2_score(y_test_daily, preds_daily)
mape_sarimax = mean_absolute_percentage_error(y_test_daily, preds_daily) * 100

print("\n" + "="*70)
print("📊 RÉSULTATS SARIMAX (JOURNALIER)")
print("="*70)
print(f"MAE  : {mae_sarimax:.2f} €/MWh")
print(f"RMSE : {rmse_sarimax:.2f} €/MWh")
print(f"R²   : {r2_sarimax:.3f}")
print(f"MAPE : {mape_sarimax:.2f} %")
print("="*70)

# ========================================
# VISUALISATION
# ========================================
print("\n📈 Création de la visualisation...")

fig = go.Figure()

# Afficher les derniers 180 jours du train pour contexte
train_context = y_train_daily.iloc[-180:]
fig.add_trace(go.Scatter(
    x=train_context.index, 
    y=train_context, 
    name='Historique (Train)', 
    line=dict(color='#1f77b4', width=1.5),
    opacity=0.7
))

# Test réel
fig.add_trace(go.Scatter(
    x=y_test_daily.index, 
    y=y_test_daily, 
    name='Réel (Test)', 
    line=dict(color='#2ca02c', width=2),
    mode='lines+markers',
    marker=dict(size=4)
))

# Prédictions
fig.add_trace(go.Scatter(
    x=preds_daily.index, 
    y=preds_daily, 
    name='Prédiction SARIMAX', 
    line=dict(color='#d62728', width=2, dash='dash')
))

fig.update_layout(
    title=f'<b>SARIMAX - Prédiction Journalière</b><br>MAE: {mae_sarimax:.2f} €/MWh | R²: {r2_sarimax:.3f}',
    xaxis_title='Date',
    yaxis_title='Prix Moyen Journalier (€/MWh)',
    height=600,
    template='plotly_white',
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()

# ========================================
# ANALYSE DES RÉSIDUS
# ========================================
residuals = y_test_daily - preds_daily

fig_residuals = go.Figure()
fig_residuals.add_trace(go.Scatter(
    x=residuals.index,
    y=residuals,
    mode='markers',
    marker=dict(size=6, color=residuals, colorscale='RdBu', showscale=True),
    name='Résidus'
))
fig_residuals.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Zéro")
fig_residuals.update_layout(
    title='<b>Analyse des Résidus SARIMAX</b>',
    xaxis_title='Date',
    yaxis_title='Résidu (€/MWh)',
    height=400,
    template='plotly_white'
)
fig_residuals.show()

print("\n✨ Analyse SARIMAX terminée!")
print(f"\n💡 Amélioration: MAE réduite de ~132 €/MWh (horaire) à {mae_sarimax:.2f} €/MWh (journalier)")
