"""
Code pour sauvegarder les modèles de la période 2020-2025
À ajouter à la fin de votre notebook France_2020_2025_Modeling.ipynb
"""

import os
import joblib
from datetime import datetime

# Créer le dossier models s'il n'existe pas
models_dir = '../../models'
os.makedirs(models_dir, exist_ok=True)

# Sauvegarder le modèle LightGBM de base
if 'model_base' in locals():
    model_base_path = os.path.join(models_dir, 'lightgbm_france_2020_2025_base.pkl')
    joblib.dump(model_base, model_base_path)
    print(f"✅ Modèle de base sauvegardé: {model_base_path}")

# Sauvegarder le modèle LightGBM optimisé
if 'best_gbm' in locals():
    model_optim_path = os.path.join(models_dir, 'lightgbm_france_2020_2025_optimized.pkl')
    joblib.dump(best_gbm, model_optim_path)
    print(f"✅ Modèle optimisé sauvegardé: {model_optim_path}")

# Sauvegarder le modèle SARIMAX
if 'fitted' in locals():
    sarimax_path = os.path.join(models_dir, 'sarimax_france_2020_2025.pkl')
    joblib.dump(fitted, sarimax_path)
    print(f"✅ Modèle SARIMAX sauvegardé: {sarimax_path}")

# Sauvegarder l'explainer SHAP
if 'explainer' in locals():
    shap_explainer_path = os.path.join(models_dir, 'shap_explainer_france_2020_2025.pkl')
    joblib.dump(explainer, shap_explainer_path)
    print(f"✅ SHAP Explainer sauvegardé: {shap_explainer_path}")

# Sauvegarder les valeurs SHAP
if 'shap_values' in locals():
    shap_values_path = os.path.join(models_dir, 'shap_values_france_2020_2025.pkl')
    joblib.dump(shap_values, shap_values_path)
    print(f"✅ SHAP Values sauvegardées: {shap_values_path}")

# Sauvegarder les métadonnées du modèle
metadata = {
    'model_name': 'LightGBM + SARIMAX France 2020-2025',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'df_ml_france_2020_2025.csv',
    'target': 'price_day_ahead',
    'features': list(X_train.columns) if 'X_train' in locals() else [],
    'test_period': {
        'start': test_start_date if 'test_start_date' in locals() else None,
        'samples': len(test) if 'test' in locals() else None
    },
    'metrics_lightgbm_base': {
        'MAE': mae_base if 'mae_base' in locals() else None,
        'RMSE': rmse if 'rmse' in locals() else None,
        'R2': r2_base if 'r2_base' in locals() else None,
        'MAPE': mape_base if 'mape_base' in locals() else None
    },
    'metrics_lightgbm_optimized': {
        'MAE': mae_optim if 'mae_optim' in locals() else None,
        'RMSE': rmse_optim if 'rmse_optim' in locals() else None,
        'R2': r2_optim if 'r2_optim' in locals() else None,
        'MAPE': mape_optim if 'mape_optim' in locals() else None
    },
    'metrics_sarimax': {
        'MAE': mae_sarimax if 'mae_sarimax' in locals() else None,
        'RMSE': rmse_sarimax if 'rmse_sarimax' in locals() else None,
        'R2': r2_sarimax if 'r2_sarimax' in locals() else None,
        'MAPE': mape_sarimax if 'mape_sarimax' in locals() else None
    },
    'hyperparameters_lightgbm': grid.best_params_ if 'grid' in locals() else {},
    'sarimax_order': (2, 2, 2),
    'sarimax_seasonal_order': (0, 0, 0, 6)
}

metadata_path = os.path.join(models_dir, 'models_france_2020_2025_metadata.pkl')
joblib.dump(metadata, metadata_path)
print(f"✅ Métadonnées sauvegardées: {metadata_path}")

print("\n📁 Tous les modèles ont été sauvegardés dans:", os.path.abspath(models_dir))
