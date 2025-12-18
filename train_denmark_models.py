"""
Script d'entraînement des modèles Danemark (DK1 et DK2)
Sauvegarde les modèles et leurs métadonnées dans models/Danemark_models/
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

print("=" * 80)
print("ENTRAÎNEMENT DES MODÈLES DANEMARK (2020-2025)")
print("=" * 80)

# Créer le dossier de destination
output_dir = Path("models/Danemark_models")
output_dir.mkdir(parents=True, exist_ok=True)

# Charger les données
print("\n1. Chargement des données...")
df = pd.read_csv("data/raw/time_series_60min_fr_dk_20-25_ENRICHIE_FULL.csv", parse_dates=['utc_timestamp'])
df = df.set_index('utc_timestamp')
print(f"   Dataset chargé : {df.shape}")

# Fonction d'entraînement générique
def train_models_for_zone(zone_name, target_col, load_col):
    """
    Entraîne les modèles baseline et optimisé pour une zone (DK1 ou DK2)
    """
    print(f"\n{'=' * 80}")
    print(f"ZONE {zone_name}")
    print(f"{'=' * 80}")
    
    # Préparation des features
    print(f"\n2. Préparation des features pour {zone_name}...")
    data = df.copy().sort_index()
    
    # Variables temporelles
    data["hour"] = data.index.hour
    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    
    # Lags de prix
    data[f"{target_col}_lag1"] = data[target_col].shift(1)
    data[f"{target_col}_lag24"] = data[target_col].shift(24)
    
    # Moyenne mobile 24h
    data[f"{target_col}_roll24"] = data[target_col].rolling(24).mean()
    
    features = [
        load_col,
        "wind_speed_denmark",
        "temperature_denmark",
        "hour", "dayofweek", "month",
        f"{target_col}_lag1",
        f"{target_col}_lag24",
        f"{target_col}_roll24"
    ]
    
    # Nettoyage
    data = data.dropna(subset=[target_col] + features)
    
    X = data[features]
    y = data[target_col]
    
    # Split 80/20
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # ===== MODÈLE BASELINE =====
    print(f"\n3. Entraînement LightGBM Baseline {zone_name}...")
    
    lgb_baseline = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_baseline.fit(X_train, y_train)
    y_pred_base = lgb_baseline.predict(X_test)
    
    # Métriques baseline
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
    mae_base = mean_absolute_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)
    
    print(f"   ✓ Baseline -> RMSE={rmse_base:.2f} | MAE={mae_base:.2f} | R²={r2_base:.3f}")
    
    # Sauvegarder baseline
    baseline_path = output_dir / f"model_{zone_name}_LightGBM_baseline.pkl"
    joblib.dump(lgb_baseline, baseline_path)
    print(f"   ✓ Modèle sauvegardé : {baseline_path}")
    
    # ===== MODÈLE OPTIMISÉ =====
    print(f"\n4. Entraînement LightGBM Optimisé {zone_name} (GridSearch)...")
    
    param_grid = {
        "num_leaves": [31, 50, 70],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 10, 20],
        "n_estimators": [300, 600]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    lgb_opt = lgb.LGBMRegressor(random_state=42, verbose=-1)
    
    grid_search = GridSearchCV(
        estimator=lgb_opt,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_lgb = grid_search.best_estimator_
    
    print(f"   ✓ Meilleurs hyperparamètres : {grid_search.best_params_}")
    
    y_pred_opt = best_lgb.predict(X_test)
    
    # Métriques optimisé
    rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
    mae_opt = mean_absolute_error(y_test, y_pred_opt)
    r2_opt = r2_score(y_test, y_pred_opt)
    
    print(f"   ✓ Optimisé -> RMSE={rmse_opt:.2f} | MAE={mae_opt:.2f} | R²={r2_opt:.3f}")
    
    # Sauvegarder optimisé
    optimized_path = output_dir / f"model_{zone_name}_LightGBM_optimise.pkl"
    joblib.dump(best_lgb, optimized_path)
    print(f"   ✓ Modèle sauvegardé : {optimized_path}")
    
    # ===== SAUVEGARDER MÉTADONNÉES =====
    print(f"\n5. Sauvegarde des métadonnées {zone_name}...")
    
    metadata = {
        "zone": zone_name,
        "target": target_col,
        "features": features,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_period": {
            "start": str(X_train.index.min()),
            "end": str(X_train.index.max())
        },
        "test_period": {
            "start": str(X_test.index.min()),
            "end": str(X_test.index.max())
        },
        "baseline": {
            "model_type": "LightGBM",
            "hyperparameters": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31
            },
            "metrics": {
                "MAE": float(mae_base),
                "RMSE": float(rmse_base),
                "R2": float(r2_base)
            }
        },
        "optimized": {
            "model_type": "LightGBM",
            "hyperparameters": grid_search.best_params_,
            "metrics": {
                "MAE": float(mae_opt),
                "RMSE": float(rmse_opt),
                "R2": float(r2_opt)
            }
        }
    }
    
    metadata_path = output_dir / f"metadata_{zone_name}.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"   ✓ Métadonnées sauvegardées : {metadata_path}")
    
    return metadata

# ===== ENTRAÎNEMENT DK1 =====
metadata_dk1 = train_models_for_zone(
    zone_name="DK1",
    target_col="DK_1_price_day_ahead",
    load_col="DK_1_load_actual_entsoe_transparency"
)

# ===== ENTRAÎNEMENT DK2 =====
metadata_dk2 = train_models_for_zone(
    zone_name="DK2",
    target_col="DK_2_price_day_ahead",
    load_col="DK_2_load_actual_entsoe_transparency"
)

# ===== RÉSUMÉ FINAL =====
print("\n" + "=" * 80)
print("RÉSUMÉ FINAL")
print("=" * 80)

print("\n📊 DK1 (Ouest)")
print(f"   Baseline  -> MAE: {metadata_dk1['baseline']['metrics']['MAE']:.2f} €/MWh | R²: {metadata_dk1['baseline']['metrics']['R2']:.3f}")
print(f"   Optimisé  -> MAE: {metadata_dk1['optimized']['metrics']['MAE']:.2f} €/MWh | R²: {metadata_dk1['optimized']['metrics']['R2']:.3f}")

print("\n📊 DK2 (Est)")
print(f"   Baseline  -> MAE: {metadata_dk2['baseline']['metrics']['MAE']:.2f} €/MWh | R²: {metadata_dk2['baseline']['metrics']['R2']:.3f}")
print(f"   Optimisé  -> MAE: {metadata_dk2['optimized']['metrics']['MAE']:.2f} €/MWh | R²: {metadata_dk2['optimized']['metrics']['R2']:.3f}")

print("\n✅ Tous les modèles et métadonnées ont été sauvegardés dans :")
print(f"   {output_dir.absolute()}")

print("\n📁 Fichiers créés :")
for file in sorted(output_dir.glob("*")):
    print(f"   - {file.name}")

print("\n" + "=" * 80)
print("ENTRAÎNEMENT TERMINÉ")
print("=" * 80)
