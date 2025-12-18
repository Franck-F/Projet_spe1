import pandas as pd
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration des chemins relatifs à la racine du projet
# Path(__file__).parent renvoie le dossier 'src'
# .parent remonte au dossier racine 'electricite-prediction-europe'
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# URL de base pour les données OPSD
OPSD_BASE_URL = "https://data.open-power-system-data.org/time_series/2020-10-06"

# Fichiers disponibles au téléchargement
DATA_FILES = {
    "hourly": "time_series_60min_singleindex.csv",
    "15min": "time_series_15min_singleindex.csv",
    "30min": "time_series_30min_singleindex.csv"
}


def ensure_directories():
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Création du dossier external
    external_dir = DATA_DIR / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'raw': RAW_DATA_DIR,
        'processed': PROCESSED_DATA_DIR,
        'external': external_dir
    }


def download_opsd_data(file_type="hourly", force_download=False, verbose=True):
   
    # S'assurer que les dossiers existent
    ensure_directories()
    
    # Validation du paramètre file_type
    if file_type not in DATA_FILES:
        raise ValueError(f"file_type doit être parmi {list(DATA_FILES.keys())}")
    
    filename = DATA_FILES[file_type]
    local_path = RAW_DATA_DIR / filename
    
    # Vérification de l'existence du fichier
    if local_path.exists() and not force_download:
        if verbose:
            file_size_mb = local_path.stat().st_size / (1024**2)
            print(f"✓ Fichier {filename} déjà présent ({file_size_mb:.2f} MB)")
            print(f"  Emplacement : {local_path}")
        return local_path
    
    # Construction de l'URL complète
    url = f"{OPSD_BASE_URL}/{filename}"
    
    if verbose:
        print(f"Téléchargement de {filename}...")
        print(f"   Source : {url}")
        print("   (Cela peut prendre quelques minutes)")
    
    try:
        # Requête HTTP avec streaming pour gérer les gros fichiers
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        # Récupération de la taille totale si disponible
        total_size = int(response.headers.get('content-length', 0))
        
        # Téléchargement par chunks pour éviter de saturer la mémoire
        with open(local_path, 'wb') as f:
            downloaded = 0
            chunk_size = 8192  # 8 KB par chunk
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Affichage périodique de la progression
                    if verbose and downloaded % (10 * 1024 * 1024) == 0:
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"   {downloaded/(1024**2):.1f} MB / "
                                  f"{total_size/(1024**2):.1f} MB ({progress:.1f}%)")
                        else:
                            print(f"   {downloaded/(1024**2):.1f} MB téléchargés")
        
        if verbose:
            file_size_mb = local_path.stat().st_size / (1024**2)
            print(f"✓ Téléchargement terminé : {file_size_mb:.2f} MB")
            print(f"  Sauvegardé dans : {local_path}")
        
        return local_path
        
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"Erreur lors du téléchargement : {e}")
            print("\nSolutions possibles :")
            print("   1. Vérifiez votre connexion internet")
            print("   2. Téléchargez manuellement depuis :")
            print("      https://data.open-power-system-data.org/time_series/")
            print(f"   3. Placez le fichier dans : {RAW_DATA_DIR}")
        return None


def load_opsd_data(file_type="hourly", nrows=None, verbose=True):
    """
    Charge les données OPSD dans un DataFrame Pandas.
    """
    # Téléchargement ou vérification du fichier
    file_path = download_opsd_data(file_type=file_type, verbose=verbose)
    
    if file_path is None:
        raise FileNotFoundError(
            f"Impossible de charger les données. "
            f"Le fichier {DATA_FILES[file_type]} n'a pas pu être téléchargé."
        )
    
    if verbose:
        print(f"\nChargement des données depuis {file_path.name}...")
        if nrows:
            print(f"   Mode test : {nrows:,} lignes seulement")
    
    try:
        # Lecture optimisée du CSV avec Pandas
        # parse_dates : conversion automatique des dates
        # index_col : utilisation de la colonne temporelle comme index
        # low_memory : False pour une meilleure détection des types
        df = pd.read_csv(
            file_path,
            parse_dates=['utc_timestamp'],
            index_col='utc_timestamp',
            low_memory=False,
            nrows=nrows
        )
        
        if verbose:
            print(f"Données chargées avec succès")
            print(f"  Dimensions : {df.shape[0]:,} lignes × {df.shape[1]:,} colonnes")
            print(f"  Période : {df.index.min().date()} à {df.index.max().date()}")
            print(f"  Mémoire : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        raise


def get_available_countries(df, verbose=True):
    """
    Identifie tous les pays disponibles dans le dataset.
    """
    # Extraction des codes pays depuis les noms de colonnes
    # Les colonnes suivent le format : PAYS_type_de_donnée
    countries = set()
    
    for col in df.columns:
        # Séparation par underscore
        parts = col.split('_')
        if len(parts) > 0:
            # Le code pays est généralement le premier élément
            country_code = parts[0]
            # Filtrage des codes valides (2-3 lettres majuscules)
            if country_code.isupper() and 2 <= len(country_code) <= 3:
                countries.add(country_code)
    
    countries = sorted(list(countries))
    
    if verbose:
        print(f"\nPays disponibles dans le dataset : {len(countries)}")
        # Affichage en colonnes pour meilleure lisibilité
        for i in range(0, len(countries), 10):
            print(f"   {', '.join(countries[i:i+10])}")
    
    return countries


def get_price_columns(df, verbose=True):
    """
    Identifie toutes les colonnes contenant des prix day-ahead.
    """
    # Filtrage des colonnes de prix
    price_cols = [col for col in df.columns if 'price_day_ahead' in col.lower()]
    
    if len(price_cols) == 0:
        if verbose:
            print("Aucune colonne de prix day-ahead trouvée")
        return pd.DataFrame()
    
    # Analyse de chaque colonne de prix
    price_info = []
    for col in price_cols:
        non_null = df[col].notna().sum()
        total = len(df)
        completeness = (non_null / total) * 100
        
        # Extraction de la zone depuis le nom de colonne
        zone = col.replace('_price_day_ahead', '')
        
        # Calcul des dates de début et fin des données disponibles
        valid_data = df[df[col].notna()]
        if len(valid_data) > 0:
            first_date = valid_data.index.min()
            last_date = valid_data.index.max()
        else:
            first_date = None
            last_date = None
        
        price_info.append({
            'Zone': zone,
            'Column_name': col,
            'Non_null_values': non_null,
            'Total_values': total,
            'Completeness_%': completeness,
            'First_date': first_date,
            'Last_date': last_date,
            'Coverage_days': (last_date - first_date).days if first_date else 0
        })
    
    # Création du DataFrame récapitulatif
    price_df = pd.DataFrame(price_info)
    price_df = price_df.sort_values('Completeness_%', ascending=False)
    
    if verbose:
        print(f"\nAnalyse des prix day-ahead disponibles")
        print(f"   Nombre de zones avec prix : {len(price_cols)}")
        print(f"\n   Top 10 des zones les mieux couvertes :")
        print(price_df[['Zone', 'Completeness_%', 'Coverage_days']].head(10).to_string(index=False))
    
    return price_df


def save_processed_data(df, filename, verbose=True):
    """
    Sauvegarde un DataFrame traité dans le dossier processed.
    """
    # S'assurer que les dossiers existent
    ensure_directories()
    
    # Ajout de l'extension .csv si nécessaire
    if not filename.endswith('.csv'):
        filename = f"{filename}.csv"
    
    # Chemin complet
    filepath = PROCESSED_DATA_DIR / filename
    
    # Sauvegarde
    df.to_csv(filepath)
    
    if verbose:
        file_size_mb = filepath.stat().st_size / (1024**2)
        print(f"Données sauvegardées : {filepath}")
        print(f"  Dimensions : {df.shape[0]:,} × {df.shape[1]:,}")
        print(f"  Taille : {file_size_mb:.2f} MB")
    
    return filepath


def load_processed_data(filename, verbose=True):
    """
    Charge un DataFrame depuis le dossier processed.
    """
    # Ajout de l'extension .csv si nécessaire
    if not filename.endswith('.csv'):
        filename = f"{filename}.csv"
    
    filepath = PROCESSED_DATA_DIR / filename
    
    # Vérification de l'existence
    if not filepath.exists():
        raise FileNotFoundError(
            f"Le fichier {filename} n'existe pas dans {PROCESSED_DATA_DIR}"
        )
    
    # Chargement avec parsing de l'index temporel si présent
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    if verbose:
        print(f"✓ Données chargées : {filepath.name}")
        print(f"  Dimensions : {df.shape[0]:,} × {df.shape[1]:,}")
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"  Période : {df.index.min().date()} à {df.index.max().date()}")
    
    return df