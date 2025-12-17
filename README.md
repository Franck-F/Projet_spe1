# ⚡ Prédiction & Analyse du Prix de l'Électricité en Europe

![alt text](image.png)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-%233F4F75.svg?style=for-the-badge&logo=LightGBM&logoColor=white)

---

## 📖 À propos du projet

Ce projet vise à **prédire et analyser les prix "day-ahead" de l'électricité en Europe** avec un focus sur **la France et le Danemark** sur la période critique **2020-2025**. Cette période inclut des dynamiques de marché complexes : stabilité initiale, choc de la crise COVID-19, et crise énergétique majeure de 2022.

L'objectif est triple :
1. **Modéliser** les prix futurs grâce à des algorithmes de Machine Learning (LightGBM) et de séries temporelles (SARIMAX).
2. **Visualiser et Expliquer** les dynamiques de marché via un Dashboard interactif complet.
3. **Comparer** deux modèles énergétiques radicalement différents : **France (Nucléaire)** vs **Danemark (Éolien)**.

---

## 🚀 Fonctionnalités Clés

### 📊 Dashboard Interactif (Streamlit)

Une application web complète pour explorer les données et les modèles, avec **3 sections principales** :

#### 🇫🇷 **Dashboard France**
- **Vue d'Ensemble** : Métriques clés (Prix moyen, Volatilité, Production nucléaire)
- **Analyse EDA** : Distribution des prix, saisonnalités, détection d'outliers
- **Mix Énergétique** : Répartition nucléaire/renouvelable, impact sur les prix
- **Corrélations** : Heatmaps des relations prix/production/consommation
- **Performance Modèles** : Comparaison visuelle (Réel vs Prédictions) et métriques (MAE/RMSE)
- **Analyse de Volatilité** : Graphiques SHAP et lexique des features

#### 🇩🇰 **Dashboard Danemark**
- **Vue d'Ensemble** : Comparaison DK1 (Ouest) vs DK2 (Est)
- **Analyse EDA** : Distributions, évolution temporelle, saisonnalité, outliers
- **Mix Énergétique** : Camemberts DK1/DK2, prix annuel, impact du vent
- **Corrélations** : Matrices prix/conso/production + facteurs d'influence par zone
- **Performance Modèles** : Placeholder (modèles en cours d'entraînement)
- **Analyse Volatilité** : Placeholder SHAP + lexique features

#### ⚖️ **Comparaison France-Danemark**
- **Métriques Comparatives** : Prix moyens, volatilité, écarts
- **Évolution des Prix** : Graphiques temporels superposés
- **Mix Énergétique** : Camemberts côte-à-côte (Nucléaire 70% vs Éolien 55%)
- **Distribution des Prix** : Histogrammes comparatifs
- **Analyse de Volatilité** : Comparaison mensuelle
- **Tableau Détaillé** : 9 caractéristiques clés
- **Insights Stratégiques** : Avantages/inconvénients de chaque modèle

### 🧠 Pipeline Machine Learning

**Feature Engineering Avancé** :
- Variables temporelles (Saisons, Heures, Jours fériés)
- Lag Features (Prix passés à 1h, 3h, 6h, 12h, 24h, 168h)
- Rolling Statistics (Moyennes mobiles, volatilité glissante)
- Features énergétiques (Production nucléaire, éolienne, solaire, charge résiduelle)

**Modèles Comparés** :
- **LightGBM** (Gradient Boosting) : Excellent pour capturer les non-linéarités complexes
- **SARIMAX** : Référence statistique pour les séries temporelles

---

## 📂 Structure du Projet

```text
Projet_spe1/
│
├── 📊 dashboard/                 # Application Streamlit
│   ├── app.py                    # Point d'entrée principal
│   ├── views/                    # Pages du dashboard
│   │   ├── france.py            # Dashboard France (~1400 lignes)
│   │   ├── denmark.py           # Dashboard Danemark (~630 lignes)
│   │   └── comparison.py        # Comparaison FR-DK (~350 lignes)
│   ├── utils/                    # Utilitaires
│   │   └── data_loader.py       # Chargement des données
│   ├── Analyse/                  # Modules d'analyse métier
│   │   ├── Analyse France/      # Scripts d'analyse France
│   │   └── Analyse Danemark/    # Scripts d'analyse Danemark
│   └── asset/                    # Ressources statiques (Images, Drapeaux)
│
├── 📓 notebooks/                 # Labo de Data Science
│   ├── France/                   # Modélisation Focus France
│   │   ├── EDA_France.ipynb                    # Exploration & Nettoyage
│   │   ├── France_2020_2025_Features.ipynb     # Feature Engineering
│   │   ├── France_2020_2025_Modeling.ipynb     # Entraînement & Validation
│   │   └── save_models_2015_2017.py            # Sauvegarde modèles période stable
│   └── Danemark/                 # Notebooks Danemark
│       ├── EDA_Danemark_DK1 2017-2019.ipynb
│       └── EDA_Danemark_DK2 2017-2019.ipynb
│
├── 🛠 src/                       # Scripts utilitaires
│   └── data_downloader.py        # Script de téléchargement des données OPSD
│
├── 💾 data/
│   ├── raw/                      # Données brutes (ENTSO-E/OPSD)
│   │   └── time_series_60min_fr_dk_20-25_ENRICHIE_FULL.csv  # Dataset principal
│   └── processed/                # Données nettoyées (Parquet/CSV)
│       ├── df_features_france_2015_2017.csv
│       └── df_features_france_2020_2025.csv
│
├── 📦 models/                    # Modèles ML sérialisés (.pkl)
│   ├── lgbm_base_2015_2017.pkl
│   ├── lgbm_optimized_2015_2017.pkl
│   ├── lgbm_base_2020_2025.pkl
│   └── lgbm_optimized_2020_2025.pkl
│
├── 📄 rapports/                  # Documentation & Slides
│   └── Projet1-DataBI.pdf        # Présentation du projet
│
└── requirements.txt              # Dépendances du projet
```

---

## 🛠 Installation et Utilisation

### 1. Cloner et Installer

Assurez-vous d'avoir **Python 3.10+**.

```bash
# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows)
.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Lancer le Dashboard

Pour explorer les résultats via l'interface interactive :

```bash
cd dashboard
streamlit run app.py
```

Le dashboard sera accessible à l'adresse : `http://localhost:8501`

### 3. Navigation

- **Page d'accueil** : Sélection France, Danemark ou Comparaison
- **France** : 6 onglets d'analyse complète
- **Danemark** : 6 onglets avec focus DK1 vs DK2
- **Comparaison** : 7 sections de comparaison stratégique

### 4. Ré-entraîner les modèles (Optionnel)

Si vous souhaitez régénérer les modèles :

**Pour la France** :
1. Lancer `notebooks/France/EDA_France.ipynb`
2. Lancer `notebooks/France/France_2020_2025_Features.ipynb`
3. Lancer `notebooks/France/France_2020_2025_Modeling.ipynb`

**Pour le Danemark** :
1. Lancer `dashboard/Analyse/Analyse Danemark/Analyse DK1 DK2 2020-2025.py`

---

## 📈 Résultats et Insights

### 🇫🇷 France - Modèle Nucléaire

**Performance Modèles** :
- **LightGBM Optimisé** : MAE ~0.85 €/MWh (période stable 2015-2017)
- **SARIMAX** : Excellent pour capturer la saisonnalité
- **R²** : >0.95 sur période stable

**Facteurs Clés** (SHAP) :
- Prix passés (lags 1h, 24h, 168h)
- Production nucléaire (corrélation inverse avec prix)
- Charge résiduelle
- Prix du gaz et CO2

**Avantages** :
- ✅ Stabilité des prix (nucléaire pilotable)
- ✅ Mix décarboné (~90%)
- ✅ Indépendance énergétique

**Défis** :
- ⚠️ Rigidité face aux pics de demande
- ⚠️ Dépendance à la disponibilité du parc nucléaire

### 🇩🇰 Danemark - Modèle Éolien

**Caractéristiques** :
- **Champion mondial de l'éolien** : ~55% du mix
- **Volatilité élevée** : Prix très dépendants de la météo
- **Prix négatifs fréquents** : Surproduction éolienne

**Facteur Clé** (Roi Vent) :
- **Vitesse du vent** : Corrélation inverse très forte avec les prix
- Vent fort → Production abondante → Prix bas (parfois négatifs)
- Vent faible → Imports + thermique → Prix élevés

**Avantages** :
- ✅ Leader en technologies vertes
- ✅ Forte interconnexion (flexibilité)
- ✅ ~60% d'énergies renouvelables

**Défis** :
- ⚠️ Volatilité importante
- ⚠️ Intermittence (besoin d'imports)

### ⚖️ Comparaison Stratégique

| Caractéristique | 🇫🇷 France | 🇩🇰 Danemark |
|----------------|-----------|-------------|
| **Source Dominante** | Nucléaire (~70%) | Éolien (~55%) |
| **Prix Moyen** | ~95 €/MWh | ~94 €/MWh |
| **Volatilité** | Modérée | Élevée |
| **Prix Négatifs** | Rares | Fréquents |
| **Facteur Clé** | Production nucléaire | Vitesse du vent |
| **Stratégie** | Stabilité | Agilité |

**Conclusion** : Les deux pays illustrent des stratégies énergétiques radicalement différentes mais **complémentaires**. Leur intégration au marché européen permet de mutualiser les avantages de chaque modèle.

---

## 👥 Auteurs

- **Franck F.**
- **Charlotte M.**
- **Djourah O.**
- **Koffi A.**
- **Youssef S.**

---

## 📄 Licence

MIT

---

## 🔗 Ressources

- **Données** : [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- **Open Power System Data** : [OPSD](https://open-power-system-data.org/)
- **Documentation Streamlit** : [streamlit.io](https://streamlit.io/)
- **LightGBM** : [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/)
