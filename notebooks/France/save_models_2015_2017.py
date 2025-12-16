"""
Code pour sauvegarder les modèles LightGBM entraînés - France 2015-2017
À ajouter à la fin de votre notebook France_2015_2017_ML1.ipynb
"""

import os
import joblib
from datetime import datetime

# Créer le dossier models/France_models s'il n'existe pas
models_dir = '../../models/France_models'
os.makedirs(models_dir, exist_ok=True)

print("="*60)
print("SAUVEGARDE DES MODÈLES FRANCE 2015-2017")
print("="*60)

# 1. Sauvegarder le modèle LightGBM
if 'model' in locals() or 'model' in globals():
    model_path = os.path.join(models_dir, 'lightgbm_france_2015_2017.pkl')
    joblib.dump(model, model_path)
    print(f"✅ Modèle LightGBM sauvegardé: {model_path}")
else:
    print("⚠️  Variable 'model' introuvable")

# 2. Sauvegarder le scaler
if 'scaler' in locals() or 'scaler' in globals():
    scaler_path = os.path.join(models_dir, 'scaler_france_2015_2017.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler sauvegardé: {scaler_path}")
else:
    print("⚠️  Variable 'scaler' introuvable")

# 3. Sauvegarder l'explainer SHAP
if 'explainer' in locals() or 'explainer' in globals():
    explainer_path = os.path.join(models_dir, 'shap_explainer_france_2015_2017.pkl')
    joblib.dump(explainer, explainer_path)
    print(f"✅ SHAP Explainer sauvegardé: {explainer_path}")
else:
    print("⚠️  Variable 'explainer' introuvable")

# 4. Sauvegarder les valeurs SHAP
if 'shap_values' in locals() or 'shap_values' in globals():
    shap_values_path = os.path.join(models_dir, 'shap_values_france_2015_2017.pkl')
    joblib.dump(shap_values, shap_values_path)
    print(f"✅ SHAP Values sauvegardées: {shap_values_path}")
else:
    print("⚠️  Variable 'shap_values' introuvable")

# 5. Sauvegarder les métadonnées du modèle
metadata = {
    'model_name': 'LightGBM France 2015-2017',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'df_features_france_2015_2017.csv',
    'target': 'price_day_ahead',
    'train_test_split': '80/20',
    'features': list(X_train_numeric.columns) if 'X_train_numeric' in locals() or 'X_train_numeric' in globals() else [],
    'n_features': X_train_numeric.shape[1] if 'X_train_numeric' in locals() or 'X_train_numeric' in globals() else None,
    'train_samples': len(X_train) if 'X_train' in locals() or 'X_train' in globals() else None,
    'test_samples': len(X_test) if 'X_test' in locals() or 'X_test' in globals() else None,
    'train_period': {
        'start': str(df_featured.index[0]) if 'df_featured' in locals() or 'df_featured' in globals() else None,
        'end': str(df_featured.index[split_idx-1]) if 'df_featured' in locals() and 'split_idx' in locals() else None
    },
    'test_period': {
        'start': str(df_featured.index[split_idx]) if 'df_featured' in locals() and 'split_idx' in locals() else None,
        'end': str(df_featured.index[-1]) if 'df_featured' in locals() or 'df_featured' in globals() else None
    },
    'metrics': {
        'MAE': float(mae) if 'mae' in locals() or 'mae' in globals() else None,
        'RMSE': float(rmse) if 'rmse' in locals() or 'rmse' in globals() else None,
        'R2': float(r2) if 'r2' in locals() or 'r2' in globals() else None,
        'MAPE': float(MAPE) if 'MAPE' in locals() or 'MAPE' in globals() else None
    },
    'model_params': model.get_params() if 'model' in locals() or 'model' in globals() else {},
    'preprocessing': {
        'scaler': 'StandardScaler',
        'season_encoding': {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
    }
}

metadata_path = os.path.join(models_dir, 'lightgbm_france_2015_2017_metadata.pkl')
joblib.dump(metadata, metadata_path)
print(f"✅ Métadonnées sauvegardées: {metadata_path}")

print("\n" + "="*60)
print(f"📁 Tous les fichiers sauvegardés dans: {os.path.abspath(models_dir)}")
print("="*60)

# Afficher un résumé
print("\n📊 RÉSUMÉ DES MÉTRIQUES:")
if 'mae' in locals() or 'mae' in globals():
    print(f"   MAE:  {mae:.2f} €/MWh")
if 'rmse' in locals() or 'rmse' in globals():
    print(f"   RMSE: {rmse:.2f} €/MWh")
if 'r2' in locals() or 'r2' in globals():
    print(f"   R²:   {r2:.4f}")
if 'MAPE' in locals() or 'MAPE' in globals():
    print(f"   MAPE: {MAPE:.2f}%")

print("\n✨ Sauvegarde terminée avec succès!")
