import json
import numpy as np

print("⚡ Upgrade SARIMAX -> ARIMAX (Ajout variables exogènes)...")

nb_path = 'notebooks/France/France_2020_2025_Modeling.ipynb'
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"❌ Erreur: Notebook introuvable à {nb_path}")
    exit(1)

# On remplace le bloc SARIMAX
for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    if "SARIMAX" in source and "fit" in source:
        new_code = """from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=== Préparation ARIMAX (SARIMAX + Variables Exogènes) ===")
print("⚠️ Stratégie : Utilisation des fondamentaux (Gas, Nucléaire, Load) pour guider le modèle")

# 1. Sélection des variables exogènes (Disponibles dans df_ml)
# On choisit les plus impactantes (identifiées par SHAP ou métier)
exog_features = ['gas', 'coal', 'nuclear', 'solar', 'wind_speed', 'load_x_hour']
# Vérification qu'elles existent
available_exog = [c for c in exog_features if c in df_ml.columns]
print(f"Variables Exogènes retenues : {available_exog}")

# 2. Split en train/test (Aligné sur X_train pour les dates)
# Attention : SARIMAX supporte mal les NaNs, on les remplit par forward fill ou 0
train_exog = df_ml.loc[df_ml.index < test_start_date, available_exog].fillna(method='ffill').fillna(0)
test_exog = df_ml.loc[df_ml.index >= test_start_date, available_exog].fillna(method='ffill').fillna(0)

# Cible (avec Winsorization pour la stabilité du training)
train_target = df_ml.loc[df_ml.index < test_start_date, 'price_day_ahead'].copy()
test_target = df_ml.loc[df_ml.index >= test_start_date, 'price_day_ahead']

# Winsorization 
Q1 = train_target.quantile(0.25)
Q3 = train_target.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
train_target = train_target.clip(lower=lower_bound, upper=upper_bound)

print(f"Train Exog shape: {train_exog.shape}")
print(f"Test Exog shape: {test_exog.shape}")

# 3. Entraînement ARIMAX
# On utilise exog=train_exog pour aider le modèle
print("\\nConfiguration: ARIMAX(1,1,1)x(1,0,1,24)")
model_sarimax = SARIMAX(train_target, 
                        exog=train_exog,
                        order=(1, 1, 1), 
                        seasonal_order=(1, 0, 1, 24),
                        enforce_stationarity=False, 
                        enforce_invertibility=False)

print("Entraînement en cours (patience ~20min)...")
fitted_sarimax = model_sarimax.fit(disp=False, maxiter=100)
print("✓ Terminé")

# 4. Prédiction (On doit fournir les exogènes du futur/test)
preds_sarimax = fitted_sarimax.forecast(steps=len(test_target), exog=test_exog)

# 5. Evaluation
mae_sarimax = mean_absolute_error(test_target, preds_sarimax)
rmse_sarimax = np.sqrt(mean_squared_error(test_target, preds_sarimax))
r2_sarimax = r2_score(test_target, preds_sarimax)
mape_sarimax = safe_mape(test_target.values, preds_sarimax.values)

print(f"MAE: {mae_sarimax:.2f} | RMSE: {rmse_sarimax:.2f} | R²: {r2_sarimax:.3f} | MAPE: {mape_sarimax:.2f}%")

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_target.index, y=test_target, name='Réel', line=dict(color='#2E7D32', width=3)))
fig.add_trace(go.Scatter(x=test_target.index, y=preds_sarimax, name='ARIMAX (Multivarié)', line=dict(color='#FF6F00', width=2)))
fig.update_layout(title=f'ARIMAX (MAE: {mae_sarimax:.2f} €/MWh)', height=500)
fig.show()"""
        cell['source'] = new_code.split('\n')
        print("✓ Bloc SARIMAX mis à jour vers ARIMAX (Multivarié)")
        break

# Sauvegarde
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("\n✅ Notebook mis à niveau avec succès !")
