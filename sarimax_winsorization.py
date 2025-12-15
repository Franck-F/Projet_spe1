import json

print("Configuration SARIMAX : Winsorization sur TOUT l'historique...")

nb_path = 'notebooks/France/France_2020_2025_Modeling.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# On modifie la cellule SARIMAX pour utiliser la Winsorization
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if "SARIMAX" in source and "df_ml[" in source:
            # Code avec Winsorization explicite pour SARIMAX
            new_code = """from statsmodels.tsa.statespace.sarimax import SARIMAX

print("=== Préparation SARIMAX (avec Winsorization) ===")
print("⚠️ Stratégie : Utilisation de TOUT l'historique mais avec plafonnement des prix extrêmes")

# 1. Séparation Train/Test sur tout l'historique
train_sarimax = df_ml[df_ml.index < test_start_date]['price_day_ahead'].copy()
test_sarimax = df_ml[df_ml.index >= test_start_date]['price_day_ahead']

# 2. Winsorization (Plafonnement)
# On utilise les quantiles calculés précédemment sur le train set global
# (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
Q1 = train_sarimax.quantile(0.25)
Q3 = train_sarimax.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Seuils Winsorization : [{lower_bound:.2f}, {upper_bound:.2f}] €/MWh")
print(f"Max avant: {train_sarimax.max():.2f} €/MWh")

# Appliquer le plafonnement
train_sarimax = train_sarimax.clip(lower=lower_bound, upper=upper_bound)

print(f"Max après: {train_sarimax.max():.2f} €/MWh")
print(f"Train: {train_sarimax.index.min()} → {train_sarimax.index.max()} ({len(train_sarimax):,} obs)")
"""
            
            cell['source'] = new_code.split('\n')
            print("✓ SARIMAX configuré avec Winsorization")

# Sauvegarde
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("\n✅ Notebook prêt pour test Winsorization + SARIMAX")
