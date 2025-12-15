import json

print("Amélioration du notebook Modeling (Correctif LightGBM + Optimisation SARIMAX)...")

nb_path = 'notebooks/France_2020_2025_Modeling.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Correctif LightGBM : Exclure 'load_bin'
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if "drop_cols =" in source and "target," in source:
            # On cherche la ligne de définition des colonnes à supprimer
            new_source = []
            for line in cell['source']:
                if "drop_cols = [" in line or "drop_cols =" in line:
                    # On remplace la ligne pour inclure load_bin explicitement
                    # On garde la structure logique mais on ajoute 'load_bin'
                    if "'load_bin'" not in line:
                         # Version robuste : on réécrit la ligne
                         line = "drop_cols = [target, 'day_name', 'season_lbl', 'season', 'price_raw', 'load_bin'] if 'price_raw' in train.columns else [target, 'day_name', 'season_lbl', 'season', 'load_bin']"
                new_source.append(line)
            cell['source'] = new_source
            print("✓ Correctif LightGBM ('load_bin') appliqué")

# 2. Amélioration SARIMAX : Filtrer 2023+
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if "train_sarimax =" in source and "df_ml[" in source:
            # On cherche la cellule de prépa SARIMAX
            # On va injecter le filtrage
            new_code = """from statsmodels.tsa.statespace.sarimax import SARIMAX

print("=== Préparation SARIMAX ===")
print("⚠️ Stratégie : Utilisation des données post-crise (2023-2025) pour la stabilité")

# Sélection des données : On coupe avant 2023 pour éviter la volatilité extrême de 2022
# qui fausse les coefficients du modèle linéaire SARIMAX
start_date_sarimax = '2023-01-01'

train_sarimax = df_ml[(df_ml.index >= start_date_sarimax) & (df_ml.index < test_start_date)]['price_day_ahead']
test_sarimax = df_ml[df_ml.index >= test_start_date]['price_day_ahead']

print(f"Train: {train_sarimax.index.min()} → {train_sarimax.index.max()}")
print(f"  {len(train_sarimax):,} observations")
print(f"Test: {test_sarimax.index.min()} → {test_sarimax.index.max()}")
print(f"  {len(test_sarimax):,} observations")"""
            
            cell['source'] = new_code.split('\n')
            print("✓ Optimisation SARIMAX (Filtre 2023+) appliquée")

# Sauvegarde
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("\n✅ Notebook mis à jour avec succès !")
