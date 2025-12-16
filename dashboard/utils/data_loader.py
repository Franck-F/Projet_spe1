import pandas as pd
import streamlit as st
import os
from pathlib import Path

@st.cache_data
def load_data():
    """
    Loads the original electricity data (time_series_60min.csv) for backward compatibility.
    Returns a pandas DataFrame.
    """
    possible_paths = [
        "../data/raw/time_series_60min_fr_dk_2015_2020.csv",
        "data/raw/time_series_60min_fr_dk_2015_2020.csv",
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path is None:
        st.error("Fichier de données introuvable. Veuillez vérifier le chemin.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, parse_dates=['utc_timestamp'], low_memory=False)
        df.set_index('utc_timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()


@st.cache_data
def load_france_data():
    """
    Loads both French processed datasets for the enhanced France page.
    Returns a dict: {'2015_2017': df1, '2020_2025': df2}
    """
    base_paths = [
        Path("../data/processed"),
        Path("data/processed"),
    ]
    
    result = {}
    
    # Find the correct base path
    base = None
    for bp in base_paths:
        if bp.exists():
            base = bp
            break
    
    if base is None:
        st.error("Dossier data/processed introuvable.")
        return result
    
    # Load 2015-2017
    path_2015 = base / "df_features_france_2015_2017.csv"
    if path_2015.exists():
        try:
            df1 = pd.read_csv(path_2015, parse_dates=['utc_timestamp'], index_col='utc_timestamp', low_memory=False)
            result['2015_2017'] = df1
        except Exception as e:
            st.warning(f"Erreur chargement 2015-2017: {e}")
    else:
        st.warning(f"Fichier {path_2015} introuvable.")
    
    # Load 2020-2025
    path_2020 = base / "df_features_france_2020_2025.csv"
    if path_2020.exists():
        try:
            df2 = pd.read_csv(path_2020, parse_dates=['utc_timestamp'], index_col='utc_timestamp', low_memory=False)
            result['2020_2025'] = df2
        except Exception as e:
            st.warning(f"Erreur chargement 2020-2025: {e}")
    else:
        st.warning(f"Fichier {path_2020} introuvable.")
    
    return result
