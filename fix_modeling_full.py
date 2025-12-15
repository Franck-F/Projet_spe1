import json
import numpy as np

print("🔧 Réparation intégrale du notebook Modeling...")

nb_path = 'notebooks/France/France_2020_2025_Modeling.ipynb'
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except FileNotFoundError:
    print(f"❌ Erreur: Notebook introuvable à {nb_path}")
    exit(1)

# --- 1. Fix Path Data ---
for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    if "pd.read_csv" in source and "df_ml_france" in source:
        # On s'assure que le chemin est relatif correct (remonter de 2 niveaux)
        new_source = [l.replace("'../data", "'../../data").replace("'data/", "'../../data/") for l in cell['source']]
        cell['source'] = new_source
        print("✓ Chemin des données corrigé")

# --- 2. Fix LightGBM (Leakage & Load_bin) ---
for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    if "drop_cols =" in source and "X_train" in source:
        # On remplace la logique de drop par une version blindée
        new_code = """# Préparation X, y
target = 'price_day_ahead'

# 1. Colonnes techniques à supprimer (y compris load_bin qui plante LightGBM)
drop_cols_technical = ['day_name', 'season_lbl', 'season', 'price_raw', 'load_bin', 'utc_timestamp']

# 2. Colonnes cibles ou fuites potentielles (Leakage)
# On supprime TOUT ce qui contient 'price_day_ahead' SAUF si c'est un lag ou un rolling
drop_cols_leakage = [c for c in df_ml.columns if target in c and 'lag' not in c and 'rolling' not in c]

# Fusion des listes
drop_cols = list(set(drop_cols_technical + drop_cols_leakage))
# Filtrer pour ne garder que ce qui existe vraiment
drop_cols = [c for c in drop_cols if c in train.columns]

print(f"Colonnes supprimées : {len(drop_cols)}")
# print(drop_cols)

X_train, y_train = train.drop(columns=drop_cols), train[target]
X_test, y_test = test.drop(columns=drop_cols), test[target]

print(f"\\nX_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")"""
        
        cell['source'] = new_code.split('\n')
        print("✓ Sécurité LightGBM (Anti-Leakage + Anti-Crash) appliquée")

# --- 3. Fix RMSE Calculation ---
for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    if "root_mean_squared_error" in source and "rmse =" in source:
         # On remplace par np.sqrt(mean_squared_error) qui est universel, 
         # car root_mean_squared_error est récent dans sklearn et peut manquer
         new_source = []
         for line in cell['source']:
             if "root_mean_squared_error" in line:
                 line = line.replace("root_mean_squared_error(y_test, preds_base)", "np.sqrt(mean_squared_error(y_test, preds_base))")
             new_source.append(line)
         cell['source'] = new_source

# --- 4. Fix SARIMAX (Winsorization) ---
sarimax_fixed = False
for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    if "SARIMAX" in source and "fit" in source:
        # On remplace tout le bloc SARIMAX
        new_code = """from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=== Préparation SARIMAX (avec Winsorization) ===")
print("⚠️ Stratégie : Utilisation de TOUT l'historique avec plafonnement des prix extrêmes")

# 1. Split en train/test
train_sarimax = df_ml[df_ml.index < test_start_date]['price_day_ahead'].copy()
test_sarimax = df_ml[df_ml.index >= test_start_date]['price_day_ahead']

# 2. Winsorization (Plafonnement intelligent)
Q1 = train_sarimax.quantile(0.25)
Q3 = train_sarimax.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Seuils Winsorization : [{lower_bound:.2f}, {upper_bound:.2f}] €/MWh")
train_sarimax = train_sarimax.clip(lower=lower_bound, upper=upper_bound)

# 3. Entraînement SARIMAX
print("\\nConfiguration: SARIMAX(1,1,1)x(1,0,1,24)")
model_sarimax = SARIMAX(train_sarimax, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24),
                        enforce_stationarity=False, enforce_invertibility=False)

print("Entraînement en cours (patience ~15min)...")
fitted_sarimax = model_sarimax.fit(disp=False, maxiter=100)

# 4. Evaluation
preds_sarimax = fitted_sarimax.forecast(steps=len(test_sarimax))
mae_sarimax = mean_absolute_error(test_sarimax, preds_sarimax)
rmse_sarimax = np.sqrt(mean_squared_error(test_sarimax, preds_sarimax))
r2_sarimax = r2_score(test_sarimax, preds_sarimax)
mape_sarimax = safe_mape(test_sarimax.values, preds_sarimax.values)

print(f"MAE: {mae_sarimax:.2f} | RMSE: {rmse_sarimax:.2f} | R²: {r2_sarimax:.3f} | MAPE: {mape_sarimax:.2f}%")

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_sarimax.index, y=test_sarimax, name='Réel', line=dict(color='#2E7D32', width=3)))
fig.add_trace(go.Scatter(x=test_sarimax.index, y=preds_sarimax, name='SARIMAX', line=dict(color='#FF6F00', width=2)))
fig.update_layout(title=f'SARIMAX (Winsorized)', height=500)
fig.show()"""
        cell['source'] = new_code.split('\n')
        sarimax_fixed = True
        print("✓ Bloc SARIMAX entièrement refondu")
        break

# Sauvegarde
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("\n✅ Notebook entièrement réparé et cohérent !")
