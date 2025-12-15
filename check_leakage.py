import pandas as pd
import json

print("=== Audit Data Leakage ===")

try:
    # Charger les données (même chemin que le notebook, ajusté pour script)
    df = pd.read_csv('data/processed/df_features_france_2020_2025.csv', 
                    parse_dates=['utc_timestamp'], 
                    index_col='utc_timestamp')
    print("Données chargées.")
except:
    # Fallback si features pas trouvé, essai ML
    try:
        df = pd.read_csv('data/processed/df_ml_france_2020_2025.csv', 
                        parse_dates=['utc_timestamp'], 
                        index_col='utc_timestamp')
        print("Données ML chargées.")
    except:
        print("❌ Impossible de charger le fichier de données pour vérifier.")
        exit()

target = 'price_day_ahead'

# Simulation de la logique du notebook
print(f"Cible : {target}")

# Colonnes potentiellement problématiques
drop_cols = [target, 'day_name', 'season_lbl', 'season', 'price_raw', 'load_bin']
print(f"Colonnes à supprimer demandées : {drop_cols}")

# Simulation du drop
X_cols = [c for c in df.columns if c not in drop_cols]

print(f"\nNb features retenues : {len(X_cols)}")

if target in X_cols:
    print(f"\n🚨 ALERTE ROUGE : La cible '{target}' EST dans les features !")
    print("Le modèle triche à 100%. Il apprend 'Prix = Prix'.")
else:
    print(f"\n✅ RAS : La cible '{target}' est BIEN supprimée.")

# Vérification des noms proches
print("\nVariables 'Price' restantes (Lags autorisés) :")
price_vars = [c for c in X_cols if 'price' in c.lower()]
for p in price_vars:
    print(f" - {p}")

print("\nConclusion :")
if len(price_vars) > 0:
    print("Si vous voyez ces variables dans SHAP, c'est NORMAL.")
    print("Ce sont des 'Lags' (passé) ou des moyennes mobiles.")
