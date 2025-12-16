"""
Model Loader Utility
Charge les modèles et métadonnées sauvegardés pour le dashboard
"""

import joblib
import os
from pathlib import Path
import streamlit as st

@st.cache_data
def load_model_metadata(model_name):
    """
    Charge les métadonnées d'un modèle
    
    Args:
        model_name: Nom du fichier de métadonnées (sans extension)
    
    Returns:
        dict: Métadonnées du modèle ou None si erreur
    """
    possible_paths = [
        Path("../models/France_models"),
        Path("models/France_models"),
        Path("../../models/France_models"),
        Path("../models"),
        Path("models"),
        Path("../../models")
    ]
    
    for base_path in possible_paths:
        metadata_path = base_path / f"{model_name}.pkl"
        if metadata_path.exists():
            try:
                metadata = joblib.load(metadata_path)
                return metadata
            except Exception as e:
                st.warning(f"Erreur lors du chargement de {metadata_path}: {e}")
                return None
    
    return None



@st.cache_resource
def load_model(model_name):
    """
    Charge un modèle sauvegardé
    
    Args:
        model_name: Nom du fichier modèle (sans extension)
    
    Returns:
        Modèle chargé ou None si erreur
    """
    possible_paths = [
        Path("../models/France_models"),
        Path("models/France_models"),
        Path("../../models/France_models"),
        Path("../models"),
        Path("models"),
        Path("../../models")
    ]
    
    for base_path in possible_paths:
        model_path = base_path / f"{model_name}.pkl"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                return model
            except Exception as e:
                st.warning(f"Erreur lors du chargement de {model_path}: {e}")
                return None
    
    return None



def get_france_models_info():
    """
    Récupère les informations de tous les modèles France disponibles
    
    Returns:
        dict: Informations sur les modèles disponibles
    """
    models_info = {
        '2015_2017': {
            'base': None,
            'optimized': None,
            'sarimax': None
        },
        '2020_2025': {
            'base': None,
            'optimized': None,
            'sarimax': None
        }
    }
    
    # Charger métadonnées 2015-2017
    metadata_2015_base = load_model_metadata('lightgbm_france_2015_2017_metadata')
    metadata_2015_opt = load_model_metadata('france_2015_2017_optimized_metadata')
    
    if metadata_2015_base:
        models_info['2015_2017']['base'] = metadata_2015_base
    
    if metadata_2015_opt:
        models_info['2015_2017']['optimized'] = metadata_2015_opt.get('lightgbm', {})
        models_info['2015_2017']['sarimax'] = metadata_2015_opt.get('sarimax', {})
    
    # Charger métadonnées 2020-2025
    metadata_2020 = load_model_metadata('models_france_2020_2025_metadata')
    
    if metadata_2020:
        models_info['2020_2025']['base'] = metadata_2020.get('metrics_lightgbm_base', {})
        models_info['2020_2025']['optimized'] = metadata_2020.get('metrics_lightgbm_optimized', {})
        models_info['2020_2025']['sarimax'] = metadata_2020.get('metrics_sarimax', {})
    
    return models_info


def format_metric(value, decimals=2):
    """Formate une métrique pour l'affichage"""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"
