# Projet : Prédiction du prix de l'électricité en Europe

### Résumé

Le projet **Projet_spe1** contient une chaîne complète pour prédire le prix day-ahead de l'électricité
en Europe à partir des données Open Power System Data (OPSD). Il inclut l'ingestion, l'exploration,
le feature engineering, l'entraînement de modèles et les artefacts utilisés pour reproduire les résultats.

###  Structure du projet

```
electricite-prediction-europe/
│
├── data/                           # Données brutes et traitées
│   ├── raw/                        # CSV téléchargés 
│   ├── processed/                  # Données après preprocessing
│   └── external/                   # Données externes 
│
├── notebooks/                      # Notebooks Jupyter 
│   ├── 01_exploration_france.ipynb
│   ├── 02_exploration_danemark.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelisation_france.ipynb
│   ├── 05_modelisation_danemark.ipynb
│   ├── 06_comparaison_fr_vs_dk.ipynb
│   └── 07_analyse_shap.ipynb
│
├── src/                            # Code Python réutilisable
│   ├── __init__.py
│   ├── data_loader.py              # Fonctions chargement données
│   ├── preprocessing.py            # Feature engineering
│   ├── models.py                   # Modèles ML
│   ├── evaluation.py               # Métriques et visualisations
│   └── utils.py                    # Utilitaires divers
│
├── dashboard/                      # Dashboard interactif
│   ├── app.py                      # Application Streamlit
│   ├── components/                 # Composants réutilisables
│   └── assets/                     # CSS, images
│
├── models/                         # Modèles entraînés sauvegardés
│   ├── france_xgboost.pkl
│   └── danemark_xgboost.pkl
│
├── reports/                        # Rapport et visualisations
│   ├── figures/                    # Graphiques exportés
│   ├── rapport_projet.md           # Rapport en Markdown
│   └── presentation.pptx           # Slides soutenance
│
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation projet
└── .gitignore                      # Fichiers à ignorer (data/raw/, models/)

```

### Table des matières

- Installation rapide
- Données
- Utilisation
- Reproductibilité
- Bonnes pratiques
- Annexes (Git LFS, requirements)

### Installation rapide

1. Créer et activer un virtualenv :

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

2. Mettre `pip` à jour et installer les dépendances :

```powershell
& ".\\.venv\\Scripts\\python.exe" -m pip install -U pip
& ".\\.venv\\Scripts\\python.exe" -m pip install -r requirements.txt
```

Si `requirements.txt` n'existe pas :

```powershell
pip install pandas requests numpy scikit-learn ipykernel
```

### Données

Le fichier OPSD `time_series_60min_singleindex.csv` est volumineux (>100 MB) et **n'est pas**
commité dans ce dépôt pour éviter les erreurs de push vers GitHub. Options :

1. Lancer `main.py` pour télécharger automatiquement le fichier dans `data/raw/`.
2. Télécharger manuellement depuis : https://data.open-power-system-data.org/time_series/ et
  placer le fichier dans `data/raw/`.
3. Configurer Git LFS pour versionner les CSV volumineux (voir Annexes).

### Utilisation rapide

- Exécuter le runner principal :

```powershell
& ".\\.venv\\Scripts\\python.exe" main.py
```

- Lancer Jupyter Lab pour explorer les notebooks :

```powershell
jupyter lab
```

### Reproductibilité

Les notebooks dans `notebooks/` décrivent les étapes et hyperparamètres utilisés. Pour ré-entraîner
les modèles, exécutez les notebooks ou les scripts dans `src/`.

### Bonnes pratiques

- Ne pas committer les données brutes volumineuses dans Git — `data/raw/` est listé dans `.gitignore`.
- Pour stocker des artefacts volumineux (modèles / données), utilisez Git LFS.
- Documentez les expériences et conservez les scripts de reproduction.

### Annexes — commandes utiles

- Générer `requirements.txt` depuis l'environnement actif :

```powershell
& ".\\.venv\\Scripts\\python.exe" -m pip freeze > requirements.txt
git add requirements.txt && git commit -m "chore: add requirements.txt" && git push
```

- Activer Git LFS et tracker les CSV volumineux :

```powershell
git lfs install
git lfs track "data/raw/*.csv"
git add .gitattributes
git commit -m "chore: track raw CSV with Git LFS"
git push
```

### Licence

Ce projet est publié sous licence **MIT**.



