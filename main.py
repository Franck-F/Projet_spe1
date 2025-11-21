"""
Script de téléchargement et chargement des données Open Power System Data (OPSD)
Projet : Prédiction du prix de l'électricité en Europe
Auteur : Franck F. Charlotte M. Djourah O. Koffi A. Youssef S.
Date : 21 Novembre 2025
"""

import pandas as pd
import requests
import os
from pathlib import Path
import zipfile
import io
import warnings
warnings.filterwarnings('ignore')

# Configuration des chemins et URLs
# Nous créons une structure de dossiers organisée pour notre projet
DATA_DIR = Path("data")  # Dossier principal pour toutes les données
RAW_DATA_DIR = DATA_DIR / "raw"  # Sous-dossier pour les données brutes
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Sous-dossier pour les données traitées

# URL de base pour télécharger les données OPSD
# Ces données proviennent du projet Open Power System Data
OPSD_BASE_URL = "https://data.open-power-system-data.org/time_series/2020-10-06"

# Nous allons télécharger plusieurs versions des données :
# - Version horaire complète (time_series_60min_singleindex.csv)
# - Version journalière agrégée si disponible
DATA_FILES = {
    "hourly": "time_series_60min_singleindex.csv",
    # Notez : il existe aussi des versions 15min et 30min, mais 60min est suffisant
}


def create_directories():
    """
    Crée la structure de dossiers nécessaire pour le projet.
    
    Cette fonction s'assure que tous les dossiers existent avant d'essayer
    d'y sauvegarder des fichiers. C'est comme préparer des tiroirs avant
    de ranger des documents.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("✓ Structure de dossiers créée avec succès")
    print(f"  - Dossier données brutes : {RAW_DATA_DIR}")
    print(f"  - Dossier données traitées : {PROCESSED_DATA_DIR}")


def download_data(file_type="hourly", force_download=False):
    """
    Télécharge les données depuis le serveur OPSD.
    
    Paramètres:
    -----------
    file_type : str
        Type de fichier à télécharger ('hourly' par défaut)
    force_download : bool
        Si True, télécharge même si le fichier existe déjà localement
        
    Cette fonction vérifie d'abord si le fichier existe déjà pour éviter
    des téléchargements inutiles. C'est comme vérifier dans votre frigo
    avant d'aller faire les courses.
    """
    filename = DATA_FILES[file_type]
    local_path = RAW_DATA_DIR / filename
    
    # Vérification : le fichier existe-t-il déjà ?
    if local_path.exists() and not force_download:
        print(f"✓ Le fichier {filename} existe déjà localement")
        print(f"  Taille : {local_path.stat().st_size / (1024**2):.2f} MB")
        return local_path
    
    # Construction de l'URL complète
    url = f"{OPSD_BASE_URL}/{filename}"
    
    print(f"📥 Téléchargement de {filename} en cours...")
    print(f"   URL : {url}")
    print("   (Cela peut prendre quelques minutes selon votre connexion)")
    
    try:
        # Envoi de la requête HTTP
        # stream=True permet de télécharger par morceaux (chunks)
        # pour ne pas saturer la mémoire avec de gros fichiers
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()  # Lève une exception si erreur HTTP
        
        # Calcul de la taille totale du fichier
        total_size = int(response.headers.get('content-length', 0))
        
        # Téléchargement et sauvegarde par morceaux de 8 KB
        with open(local_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filtre les keep-alive
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Affichage de la progression (tous les 10 MB)
                    if downloaded % (10 * 1024 * 1024) == 0:
                        progress = (downloaded / total_size) * 100 if total_size else 0
                        print(f"   Progression : {downloaded / (1024**2):.1f} MB "
                              f"({progress:.1f}%)")
        
        file_size_mb = local_path.stat().st_size / (1024**2)
        print(f"✓ Téléchargement terminé : {file_size_mb:.2f} MB")
        return local_path
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        print("\nSolutions possibles :")
        print("1. Vérifiez votre connexion internet")
        print("2. Téléchargez manuellement depuis : https://data.open-power-system-data.org/time_series/")
        print(f"3. Placez le fichier dans : {RAW_DATA_DIR}")
        return None


def load_data(file_path, nrows=None):
    """
    Charge les données CSV dans un DataFrame Pandas.
    
    Paramètres:
    -----------
    file_path : Path
        Chemin vers le fichier CSV à charger
    nrows : int, optional
        Nombre de lignes à charger (utile pour tests rapides)
        
    Cette fonction gère intelligemment les types de données et les dates.
    Pandas va automatiquement détecter et convertir les colonnes en
    types appropriés (nombres, dates, etc.).
    """
    print(f"\n📊 Chargement des données depuis {file_path.name}...")
    
    if nrows:
        print(f"   Mode test : chargement de {nrows} lignes seulement")
    
    try:
        # parse_dates=['utc_timestamp'] : convertit automatiquement cette colonne en datetime
        # index_col='utc_timestamp' : utilise cette colonne comme index (pratique pour séries temporelles)
        # low_memory=False : lit tout le fichier en une fois pour mieux détecter les types
        df = pd.read_csv(
            file_path,
            parse_dates=['utc_timestamp'],
            index_col='utc_timestamp',
            low_memory=False,
            nrows=nrows
        )
        
        print(f"✓ Données chargées avec succès !")
        print(f"  - Nombre de lignes : {len(df):,}")
        print(f"  - Nombre de colonnes : {len(df.columns):,}")
        print(f"  - Période couverte : {df.index.min()} à {df.index.max()}")
        print(f"  - Mémoire utilisée : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None


def get_available_prices(df):
    """
    Identifie quels pays ont des prix day-ahead disponibles.
    
    Cette fonction parcourt toutes les colonnes du DataFrame et
    identifie celles qui contiennent 'price_day_ahead' dans leur nom.
    C'est comme faire l'inventaire de ce qu'on a dans notre dataset.
    """
    # Filtrage des colonnes contenant 'price_day_ahead'
    price_columns = [col for col in df.columns if 'price_day_ahead' in col.lower()]
    
    print("\n💰 Prix day-ahead disponibles :")
    print(f"   Nombre de zones avec prix : {len(price_columns)}")
    
    # Analyse de la disponibilité des données pour chaque zone
    price_info = []
    for col in price_columns:
        non_null = df[col].notna().sum()
        total = len(df)
        percentage = (non_null / total) * 100
        
        price_info.append({
            'Zone': col.replace('_price_day_ahead', ''),
            'Valeurs non-nulles': non_null,
            'Pourcentage': percentage,
            'Première date': df[df[col].notna()].index.min(),
            'Dernière date': df[df[col].notna()].index.max()
        })
    
    # Création d'un DataFrame récapitulatif
    price_df = pd.DataFrame(price_info)
    price_df = price_df.sort_values('Pourcentage', ascending=False)
    
    return price_df, price_columns


def display_data_summary(df):
    """
    Affiche un résumé détaillé des données chargées.
    
    Cette fonction vous donne une vue d'ensemble de vos données,
    comme lire la table des matières d'un livre avant de le lire.
    """
    print("\n" + "="*70)
    print("📋 RÉSUMÉ DES DONNÉES CHARGÉES")
    print("="*70)
    
    print(f"\n🗓️  Informations temporelles :")
    print(f"   - Date de début : {df.index.min()}")
    print(f"   - Date de fin : {df.index.max()}")
    print(f"   - Durée totale : {(df.index.max() - df.index.min()).days} jours")
    
    print(f"\n📊 Structure des données :")
    print(f"   - Dimensions : {df.shape[0]:,} lignes × {df.shape[1]:,} colonnes")
    print(f"   - Mémoire : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    print(f"\n🔍 Aperçu des types de colonnes :")
    # Grouper les colonnes par type de données
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"   - {dtype} : {count} colonnes")
    
    print(f"\n❓ Valeurs manquantes globales :")
    missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"   - Pourcentage total : {missing_percentage:.2f}%")
    
    print("\n" + "="*70)


def main():
    """
    Fonction principale qui orchestre tout le processus.
    
    C'est le chef d'orchestre qui va appeler toutes les autres fonctions
    dans le bon ordre pour accomplir notre tâche.
    """
    print("="*70)
    print("🔌 PROJET : PRÉDICTION DU PRIX DE L'ÉLECTRICITÉ EN EUROPE")
    print("   Étape 1 : Téléchargement et chargement des données")
    print("="*70 + "\n")
    
    # Étape 1 : Créer la structure de dossiers
    create_directories()
    
    # Étape 2 : Télécharger les données
    file_path = download_data(file_type="hourly", force_download=False)
    
    if file_path is None:
        print("\n⚠️  Impossible de continuer sans les données")
        return None
    
    # Étape 3 : Charger les données
    # Pour un premier test, vous pouvez limiter avec nrows=10000
    df = load_data(file_path, nrows=None)  # Changez en nrows=10000 pour test rapide
    
    if df is None:
        print("\n⚠️  Échec du chargement des données")
        return None
    
    # Étape 4 : Analyser les prix disponibles
    price_summary, price_columns = get_available_prices(df)
    print("\n" + price_summary.to_string(index=False))
    
    # Étape 5 : Afficher le résumé global
    display_data_summary(df)
    
    # Étape 6 : Afficher un aperçu des premières lignes
    print("\n👀 Aperçu des 5 premières lignes (colonnes de prix uniquement) :")
    print(df[price_columns].head())
    
    print("\n✅ Chargement terminé avec succès !")
    print(f"\n💡 Le DataFrame est maintenant disponible dans la variable 'df'")
    print(f"   Utilisez df.head() pour voir les premières lignes")
    print(f"   Utilisez df.info() pour plus de détails")
    
    return df, price_summary, price_columns


# Point d'entrée du script
if __name__ == "__main__":
    # Exécution de la fonction principale
    df, price_summary, price_columns = main()