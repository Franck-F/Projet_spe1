# Projet : Prédiction du prix de l'électricité en Europe

Résumé
------

le projet **Projet_spe1** est une chaîne de traitement et de modélisation
conçue pour prédire le prix day-ahead de l'électricité pour plusieurs zones européennes
à partir des données publiques Open Power System Data (OPSD).


Principaux résultats
-------------------

- Pipelines d'ingestion et de prétraitement des séries temporelles horaires.
- Notebooks d'exploration et de feature engineering.
- Modèles XGBoost pré-entraînés pour la France et le Danemark disponibles dans `models/`.

Organisation du dépôt
---------------------

- `data/` : emplacement local attendu des jeux de données (les gros fichiers ne sont pas versionnés).
  - `data/raw/` : données brutes (à placer localement, exclues du dépôt via `.gitignore`).
- `notebooks/` : notebooks Jupyter pour exploration, feature engineering et modélisation.
- `src/` : modules Python pour charger, nettoyer, transformer et évaluer les données.
- `models/` : modèles entraînés (.pkl) prêts à l'emploi.
- `rapports/` : documents et présentations.
- `main.py` : script principal d'orchestration (création de dossiers, téléchargement, chargement).

Prérequis
---------

- Python 3.11+ (le projet a été testé avec Python 3.13.1)
- Virtualenv recommandé
- Outils : `git`, `uv`, `pip`  (ou `pipenv`/`poetry` selon votre flux)

Installation rapide
------------------

1. Créez et activez un virtualenv :

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

2. Mettre uv et pip à jour et installer les dépendances :

```powershell
& ".\\.venv\\Scripts\\python.exe" -m uv pip install -U pip
& ".\\.venv\\Scripts\\python.exe" -m uv pip install -r requirements.txt
```

Si `requirements.txt` n'existe pas encore, installez les paquets essentiels :

```powershell
pip install pandas requests numpy scikit-learn ipykernel
```

Données — note importante
-------------------------

Le fichier `time_series_60min_singleindex.csv` fourni par OPSD est volumineux (>100 MB). 

1. Exécuter `main.py` qui  télécharge le fichier depuis OPSD et le place dans `data/raw/`.
2. Télécharger manuellement depuis : https://data.open-power-system-data.org/time_series/ et
   placer le fichier dans `data/raw/`.
3. Configurer Git LFS si vous souhaitez versionner ce fichier.

Utilisation rapide
------------------

- Lancer le pipeline minimal (création des dossiers + téléchargement) :

```powershell
& ".\\.venv\\Scripts\\python.exe" main.py
```

- Lancer Jupyter Lab pour ouvrir les notebooks :

```powershell
jupyter lab
```

Reproductibilité
----------------

- Les notebooks dans `notebooks/` contiennent les étapes d'analyse et les paramètres
  d'entraînement utilisés. Pour reproduire les entraînements, exécutez les cellules des
  notebooks correspondants ou exécutez les scripts dédiés dans `src/`.


Licence
-------
*MIT*

