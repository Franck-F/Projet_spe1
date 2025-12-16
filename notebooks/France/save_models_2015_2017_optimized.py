"""
Code pour sauvegarder les modèles - France 2015-2017 ML Optimisé
À ajouter à la fin de votre notebook France_2015_2017_ML_Optimisé.ipynb
"""

import os
import joblib
from datetime import datetime

# Créer le dossier models/France_models s'il n'existe pas
models_dir = '../../models/France_models'
os.makedirs(models_dir, exist_ok=True)

print("="*60)
print("SAUVEGARDE DES MODÈLES FRANCE 2015-2017 (OPTIMISÉ)")
print("="*60)

# 1. Sauvegarder le modèle SARIMAX
if 'results_sarima' in locals() or 'results_sarima' in globals():
    sarimax_path = os.path.join(models_dir, 'sarimax_france_2015_2017.pkl')
    joblib.dump(results_sarima, sarimax_path)
    print(f"✅ Modèle SARIMAX sauvegardé: {sarimax_path}")
else:
    print("⚠️  Variable 'results_sarima' introuvable")

# 2. Sauvegarder le modèle LightGBM optimisé (GridSearchCV)
if 'grid' in locals() or 'grid' in globals():
    lgbm_optimized_path = os.path.join(models_dir, 'lightgbm_france_2015_2017_gridsearch.pkl')
    joblib.dump(grid, lgbm_optimized_path)
    print(f"✅ LightGBM GridSearch sauvegardé: {lgbm_optimized_path}")
    
    # Sauvegarder aussi le meilleur estimateur directement
    best_model_path = os.path.join(models_dir, 'lightgbm_france_2015_2017_best_estimator.pkl')
    joblib.dump(grid.best_estimator_, best_model_path)
    print(f"✅ Meilleur estimateur LightGBM sauvegardé: {best_model_path}")
else:
    print("⚠️  Variable 'grid' introuvable")

# 3. Sauvegarder les prédictions SARIMAX
if 'mean_forecast' in locals() or 'mean_forecast' in globals():
    forecast_path = os.path.join(models_dir, 'sarimax_france_2015_2017_forecast.pkl')
    forecast_data = {
        'predictions': mean_forecast,
        'confidence_interval': conf_int if 'conf_int' in locals() or 'conf_int' in globals() else None,
        'y_test': y_test if 'y_test' in locals() or 'y_test' in globals() else None
    }
    joblib.dump(forecast_data, forecast_path)
    print(f"✅ Prédictions SARIMAX sauvegardées: {forecast_path}")

# 4. Sauvegarder les métadonnées
metadata = {
    'model_name': 'SARIMAX + LightGBM Optimisé France 2015-2017',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'df_features_france_2015_2017.csv',
    'target': 'price_day_ahead',
    
    # SARIMAX
    'sarimax': {
        'order': (1, 1, 1),
        'seasonal_order': (0, 1, 1, 7),
        'exog_vars': available_exog if 'available_exog' in locals() or 'available_exog' in globals() else [],
        'aggregation': 'Daily (mean)',
        'train_size': train_size if 'train_size' in locals() or 'train_size' in globals() else None,
        'train_days': len(train) if 'train' in locals() or 'train' in globals() else None,
        'test_days': len(test) if 'test' in locals() or 'test' in globals() else None,
        'metrics': {
            'MAE': float(mae) if 'mae' in locals() or 'mae' in globals() else None,
            'RMSE': float(rmse) if 'rmse' in locals() or 'rmse' in globals() else None,
            'R2': float(r2) if 'r2' in locals() or 'r2' in globals() else None,
            'MAPE': float(mape) if 'mape' in locals() or 'mape' in globals() else None
        }
    },
    
    # LightGBM
    'lightgbm': {
        'optimization_method': 'GridSearchCV',
        'cv_splits': 3,
        'cv_method': 'TimeSeriesSplit',
        'sample_size': len(X_sample) if 'X_sample' in locals() or 'X_sample' in globals() else None,
        'features': features if 'features' in locals() or 'features' in globals() else [],
        'n_features': len(features) if 'features' in locals() or 'features' in globals() else None,
        'best_params': grid.best_params_ if 'grid' in locals() or 'grid' in globals() else {},
        'param_grid': param_grid if 'param_grid' in locals() or 'param_grid' in globals() else {},
        'metrics': {
            'MAE': float(mae_lgb) if 'mae_lgb' in locals() or 'mae_lgb' in globals() else None,
            'RMSE': float(rmse_lgb) if 'rmse_lgb' in locals() or 'rmse_lgb' in globals() else None,
            'R2': float(r2_lgb) if 'r2_lgb' in locals() or 'r2_lgb' in globals() else None,
            'MAPE': float(mape_lgb) if 'mape_lgb' in locals() or 'mape_lgb' in globals() else None
        }
    },
    
    # Analyse de volatilité
    'volatility_analysis': {
        'threshold_percentile': 95,
        'threshold_value': float(threshold) if 'threshold' in locals() or 'threshold' in globals() else None
    }
}

metadata_path = os.path.join(models_dir, 'france_2015_2017_optimized_metadata.pkl')
joblib.dump(metadata, metadata_path)
print(f"✅ Métadonnées sauvegardées: {metadata_path}")

print("\n" + "="*60)
print(f"📁 Tous les fichiers sauvegardés dans: {os.path.abspath(models_dir)}")
print("="*60)

# Afficher un résumé
print("\n📊 RÉSUMÉ DES MÉTRIQUES:")
print("\n🔹 SARIMAX (Journalier):")
if 'mae' in locals() or 'mae' in globals():
    print(f"   MAE:  {mae:.2f} €/MWh")
if 'rmse' in locals() or 'rmse' in globals():
    print(f"   RMSE: {rmse:.2f} €/MWh")
if 'r2' in locals() or 'r2' in globals():
    print(f"   R²:   {r2:.4f}")
if 'mape' in locals() or 'mape' in globals():
    print(f"   MAPE: {mape:.2f}%")

print("\n🔹 LightGBM Optimisé (Horaire):")
if 'mae_lgb' in locals() or 'mae_lgb' in globals():
    print(f"   MAE:  {mae_lgb:.2f} €/MWh")
if 'rmse_lgb' in locals() or 'rmse_lgb' in globals():
    print(f"   RMSE: {rmse_lgb:.2f} €/MWh")
if 'r2_lgb' in locals() or 'r2_lgb' in globals():
    print(f"   R²:   {r2_lgb:.4f}")
if 'mape_lgb' in locals() or 'mape_lgb' in globals():
    print(f"   MAPE: {mape_lgb:.2f}%")

if 'grid' in locals() or 'grid' in globals():
    print(f"\n⚙️  Meilleurs hyperparamètres LightGBM:")
    for param, value in grid.best_params_.items():
        print(f"   {param}: {value}")

print("\n✨ Sauvegarde terminée avec succès!")
