# Projet : Prédiction du prix de l'électricité en Europe

### Résumé

Ce projet vise à prédire le prix day-ahead de l'électricité en Europe (focus France et Danemark) en utilisant des techniques de Machine Learning avancées et des séries temporelles. Il analyse les données de 2020 à 2025 (incluant la crise énergétique et la stabilisation).

Principales réalisations :

- **Nettoyage avancé** : Gestion des doublons, imputation intelligente, traitement des outliers (Winsorization).
- **Feature Engineering** : Création de ~50 variables (Lags, Rolling stats, Saisonalités, Mix énergétique).
- **Modélisation Hybride** : Comparaison rigoureuse entre **LightGBM** (Machine Learning) et **SARIMAX** (Séries Temporelles).
- **Interprétabilité** : Analyse SHAP pour comprendre les facteurs d'influence.

### Structure du projet

```
electricite-prediction-europe/
│
├── data/                           # Données
│   ├── raw/                        # time_series_60min_fr_dk_2020_2025.csv (Source)
│   ├── processed/                  # df_eda_cleaned.csv, df_features.csv (Intermédiaires)
│
├── notebooks/                      # Notebooks Modulaires (2020-2025)
│   ├── France_2020_2025_EDA.ipynb        # 1. Analyse Exploratoire (EDA) & Nettoyage
│   ├── France_2020_2025_Features.ipynb   # 2. Feature Engineering
│   ├── France_2020_2025_Modeling.ipynb   # 3. Modélisation (LightGBM vs SARIMAX)
│   └── Anciens/                          # (Archives 2015-2017)
│
├── src/                            # Code réutilisable (WIP)
├── models/                         # Modèles sauvegardés (.pkl)
├── requirements.txt                # Dépendances Python
└── README.md                       # Documentation
```

### Détail des Notebooks (Workflow 2020-2025)

**1. `France_2020_2025_EDA.ipynb` (Exploration & Nettoyage)**

- **Objectif** : Comprendre la donnée et la nettoyer.
- **Actions** :
  - Chargement et typage des colonnes (France uniquement).
  - Gestion des valeurs manquantes : Suppression si >50% manquants, Imputation temporelle sinon.
  - **Analyses Visuelles** : Distribution des prix, corrélations (Heatmaps), évolution temporelle.
  - **Mix Énergétique** : Analyse de la part Nucléaire/Renouvelable/Fossile.
  - **Outliers** : Détection (IQR) et traitement par **Winsorization** (clipping) pour stabiliser l'apprentissage sans perdre l'information des pics.

**2. `France_2020_2025_Features.ipynb` (Ingénierie des Variables)**

- **Objectif** : Créer des variables prédictives pour le ML.
- **Features créées** (~48 variables) :
  - *Temporelles* : Heure, Jour, Mois, Saisons, Jours fériés, Weekend.
  - *Lags* : Prix décalés (1h, 24h, 1 semaine) pour capturer l'autocorrélation.
  - *Rolling Stats* : Moyennes/Ecarts-types mobiles sur 6h, 24h.
  - *Métier* : Charge résiduelle, Ratio renouvelable, Capacité disponible.

**3. `France_2020_2025_Modeling.ipynb` (Modélisation & Évaluation)**

- **Objectif** : Prédire le prix sur les 9 derniers mois de 2025.
- **Modèles** :
  - **LightGBM** : Baseline vs Optimisé (GridSearch sur `num_leaves`, `learning_rate`, `n_estimators`).
  - **SARIMAX** : Modèle statistique avec saisonnalité journalière (24h) entraîné sur l'ensemble du dataset.
- **Évaluation** : Comparaison via MAE, RMSE, et R².
- **Interprétation** : Analyse SHAP pour identifier les features clés (ex: Prix de la veille, Prix du gaz, Charge).

### Installation et Utilisation

1. **Prérequis** : Python 3.10+
2. **Installation** :

```powershell
# Créer l'environnement
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

3. **Lancer l'analyse** :
    - Ouvrir Jupyter Lab : `jupyter lab`
    - Exécuter les notebooks dans l'ordre : `EDA` -> `Features` -> `Modeling`.

### Résultats Clés

- Le modèle **LightGBM** surpasse généralement SARIMAX grâce à sa capacité à gérer les non-linéarités et les interactions complexes (ex: effet du mix énergétique sur le prix).
- L'analyse SHAP révèle que les **prix passés (Lags)** et le **prix du gaz/CO2** sont souvent les prédicteurs les plus influents.
- La **Winsorization** des prix a permis de réduire l'erreur (RMSE) en évitant que le modèle ne sur-réagisse aux pics extrêmes de 2022.

### Auteurs
