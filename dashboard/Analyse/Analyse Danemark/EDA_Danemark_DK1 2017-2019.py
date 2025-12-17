# %% [markdown]
# # Analyse et Modélisation du Prix de l'Électricité - Danemark (DK1)
# *Exploratory Data Analysis (EDA), Visualisations  et Prédiction (LightGBM base , optimisé et SARIMAX)*

# %% [markdown]
# ---

# %% [markdown]
# ## 1. Configuration et Chargement des Données

# %%
import urllib.request
import os
import pandas as pd
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import display 
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# --- Téléchargement --- Vu que le fichier est lourd, on n'a pas mis sur le repo GitHub
os.makedirs('../data/raw', exist_ok=True)
url = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
destination = "../data/raw/time_series_60min2015-2020.csv"

if not os.path.exists(destination):
    print(" Téléchargement du dataset...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(" Dataset téléchargé !")
    except:
        print(" Impossible de télécharger, lecture directe depuis l'URL...")
        destination = url
else:
    print(" Dataset local trouvé.")

# --- Chargement ---
print("Lecture des données...")
df = pd.read_csv(destination, parse_dates=['utc_timestamp'], low_memory=False)
df = df.set_index('utc_timestamp')
print("Chargement terminé.")

# %% [markdown]
# ## 2. Préparation et Nettoyage (Zone DK1)

# %%
# Sélection des colonnes pour DK1 (Ouest Danemark)
cols_mapping = {
    'DK_1_price_day_ahead': 'price',
    'DK_1_load_actual_entsoe_transparency': 'load_actual',
    'DK_1_load_forecast_entsoe_transparency': 'load_forecast',
    'DK_1_solar_generation_actual': 'solar_generation',
    'DK_1_wind_generation_actual': 'wind_generation'
}

df_dk = df[list(cols_mapping.keys())].rename(columns=cols_mapping)

# --- Analyse de la Qualité (Choix de la période) ---
# On compte les données valides par an
yearly_counts = df_dk.groupby(df_dk.index.year).count()

fig = px.bar(
    yearly_counts, 
    barmode='group',
    title="Qualité des Données : Nombre d'observations valides par an",
    labels={"index": "Année", "value": "Heures valides", "variable": "Variable"},
    template="plotly_white"
)
fig.add_hline(y=8760, line_dash="dash", line_color="red", annotation_text="Année Complète")
fig.show()



# %%
# -> Décision : On garde 2017-2019 car ce sont les années complètes( plus ou moins) et récentes
df_dk = df_dk.loc['2017-01-01':'2019-12-31']
df_dk = df_dk.interpolate(method='linear').dropna()

print(f"Données filtrées (2017-2019) : {df_dk.shape[0]} heures.")

# %%
# Affichage des pourcentages de remplissage pour validation numérique
full_year_hours = 8760
completeness_pct = (yearly_counts / full_year_hours * 100).round(1)
print("Pourcentage de données disponibles par année (basé sur 8760h) :")
display(completeness_pct.style.background_gradient(cmap='RdYlGn', vmin=90, vmax=100))

# %% [markdown]
# ## 3. Feature Engineering (Variables Temporelles)

# %%
# Ajout des informations calendaires
df_dk['hour'] = df_dk.index.hour
df_dk['day_of_week'] = df_dk.index.dayofweek
df_dk['day_name'] = df_dk.index.day_name()
df_dk['month'] = df_dk.index.month
df_dk['month_name'] = df_dk.index.month_name()

# Saisons
def get_season(month):
    if month in [12, 1, 2]: return 'Hiver'
    elif month in [3, 4, 5]: return 'Printemps'
    elif month in [6, 7, 8]: return 'Été'
    else: return 'Automne'
df_dk['season'] = df_dk['month'].apply(get_season)

# Week-end vs Semaine
df_dk['day_type'] = df_dk['day_of_week'].apply(lambda x: 'Week-end' if x >= 5 else 'Semaine')

display(df_dk.head())

# %% [markdown]
# ## 4. Analyse Statistique Globale

# %%
# 1. Tableau de KPI
desc = df_dk['price'].describe()
nb_neg = df_dk[df_dk['price'] < 0].shape[0]
pct_neg = (nb_neg / len(df_dk)) * 100

stats_df = pd.DataFrame({
    'KPI': ['Prix Moyen', 'Médiane', 'Max', 'Min', 'Volatilité (Std)', 'Heures Négatives', '% Temps Négatif'],
    'Valeur': [
        f"{desc['mean']:.2f} €", f"{desc['50%']:.2f} €", f"{desc['max']:.2f} €", 
        f"{desc['min']:.2f} €", f"{desc['std']:.2f}", f"{nb_neg} h", f"{pct_neg:.2f} %"
    ]
})
display(stats_df)

# %% [markdown]
# ## 5. Analyse Temporelle (Cycles & Saisonnalité)

# %%
# A. Saisonnalité (Boxplots par Mois)
df_monthly = df_dk.groupby(['month_name', 'season'])['price'].agg(['mean', 'std']).reset_index()

# On définit l'ordre des mois
months_order = ["January", "February", "March", "April", "May", "June", 
                "July", "August", "September", "October", "November", "December"]

# 2. CRÉATION DU GRAPHIQUE EN BARRES
fig = px.bar(
    df_monthly, 
    x="month_name", 
    y="mean", 
    color="season",
    category_orders={"month_name": months_order},
    title="Saisonnalité : Prix Moyen par Mois (avec Volatilité)",
    labels={"mean": "Prix Moyen (€/MWh)", "month_name": "Mois", "season": "Saison"},
    template="plotly_white",
    text_auto=".0f" 
)

fig.update_layout(
    yaxis_title="Prix Moyen (€/MWh)",
    showlegend=True
)

fig.show()


# %% [markdown]
# Les mois de mars à mai affichent les prix moyens les plus bas (autour de 29–30 €/MWh), période où la demande se détend après l’hiver et où la production renouvelable (vent + premiers apports solaires) reste abondante.​
# 
# À partir de juin, les prix remontent progressivement pour atteindre un maximum en août‑novembre (jusqu’à 43 €/MWh), sous l’effet combiné d’une demande encore soutenue, de tensions possibles sur le parc thermique et d’événements de marché plus fréquents, avant une légère détente en décembre.​

# %%
# B. Heatmap Hebdomadaire (Le rythme de la semaine)
heatmap_data = df_dk.groupby(['day_name', 'hour'])['price'].mean().reset_index()
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig = px.imshow(
    heatmap_data.pivot(index='day_name', columns='hour', values='price').reindex(days_order),
    title="Heatmap : Profil Hebdomadaire des Prix",
    color_continuous_scale="RdYlGn_r", aspect="auto"
)
fig.show()



# %%
# C. Impact Week-end vs Semaine (Profil Horaire)
weekend_hourly = df_dk.groupby(['day_type', 'hour'])['price'].mean().reset_index()
fig = px.line(
    weekend_hourly, x="hour", y="price", color="day_type",
    title="Comparaison Semaine vs Week-end (Profil Horaire)",
    color_discrete_map={"Semaine": "royalblue", "Week-end": "orange"},
    template="plotly_white"
)
fig.show()



# %%
# D. Saisonnalité Annuelle : Distribution Mensuelle ---
fig1 = px.box(
    df_dk, 
    x="month_name", 
    y="price", 
    color="month_name",
    category_orders={"month_name": months_order}, # Force l'ordr
    title="Saisonnalité Annuelle : Distribution des Prix par Mois",
    labels={"price": "Prix (€/MWh)", "month_name": "Mois"},
    template="plotly_white"
)
fig1.update_layout(showlegend=False) 
fig1.show()

# %%
# E. Saisonnalité Hebdomadaire : Distribution par Jour ---
fig2 = px.box(
    df_dk, 
    x="day_name", 
    y="price", 
    color="day_name",
    category_orders={"day_name": days_order}, 
    title="Saisonnalité Hebdomadaire : Distribution par Jour",
    labels={"price": "Prix (€/MWh)", "day_name": "Jour"},
    template="plotly_white"
)
fig2.update_layout(showlegend=False)
fig2.show()

# %%
# F. Saisonnalité Quotidienne : Distribution par Heure ---
fig3 = px.box(
    df_dk, 
    x="hour", 
    y="price", 
    title="3. Saisonnalité Quotidienne : Profil Horaire (Distribution)",
    labels={"price": "Prix (€/MWh)", "hour": "Heure de la journée"},
    template="plotly_white",
    color_discrete_sequence=["#a30b0b"] 
)
fig3.show()

# %%
# G. Comparatif Semaine vs Week-end ---
# On compare les distributions globales
fig4 = px.box(
    df_dk, 
    x="day_type", 
    y="price", 
    color="day_type",
    title="4. Semaine vs Week-end : Impact sur les Prix",
    labels={"price": "Prix (€/MWh)", "day_type": "Type de Jour"},
    template="plotly_white",
    color_discrete_map={"Semaine": "royalblue", "Week-end": "orange"}
)
fig4.show()

# %%
# H. Analyse par Saison 

#  1. PRÉPARATION DES DONNÉES 
# Fonction pour définir la saison (si pas déjà fait)
def get_season(month):
    if month in [12, 1, 2]: return 'Hiver'
    elif month in [3, 4, 5]: return 'Printemps'
    elif month in [6, 7, 8]: return 'Été'
    else: return 'Automne'

# On applique la fonction
df_dk['season'] = df_dk.index.month.map(get_season)

# Ordre d'affichage logique
season_order = ["Hiver", "Printemps", "Été", "Automne"]

#  GRAPHIQUE A : Histogramme des Prix Moyens par Saison 
# On calcule la Moyenne (hauteur) et l'Écart-type (volatilité/barre d'erreur)
season_stats = df_dk.groupby('season')['price'].agg(['mean', 'std']).reset_index()

fig_bar = px.bar(
    season_stats, 
    x="season", 
    y="mean", 
    color="season",
    error_y="std", # La petite ligne noire qui montre la variation
    category_orders={"season": season_order},
    title="Saisonnalité : Prix Moyen et Volatilité par Saison",
    labels={"mean": "Prix Moyen (€/MWh)", "season": "Saison", "std": "Volatilité"},
    template="plotly_white",
    text_auto=".1f" # Affiche le prix avec 1 chiffre après la virgule
)
fig_bar.update_layout(showlegend=False) # Pas besoin de légende ici, l'axe X suffit
fig_bar.show()

#  GRAPHIQUE B : Profil Horaire (Lignes) 
# Note : Pour un profil horaire (24h), la ligne reste le meilleur choix visuel 
# par rapport aux barres (qui seraient trop serrées).
season_hourly = df_dk.groupby(['season', 'hour'])['price'].mean().reset_index()

fig_line = px.line(
    season_hourly, 
    x="hour", 
    y="price", 
    color="season",
    category_orders={"season": season_order},
    title="Profil Horaire Moyen : À quelle heure consomme-t-on selon la saison ?",
    labels={"price": "Prix Moyen (€/MWh)", "hour": "Heure de la journée"},
    template="plotly_white",
    markers=True # Ajoute les points sur la ligne
)

# On force l'axe X à afficher toutes les heures paires pour la lisibilité
fig_line.update_xaxes(tickmode='linear', dtick=2)

fig_line.show()

# %% [markdown]
# ## 6. Analyse Physique (Marché & Fondamentaux)

# %%
# A. Matrice de Corrélation
corr = df_dk[['price', 'load_actual', 'wind_generation', 'solar_generation']].corr()
fig = px.imshow(corr, text_auto=".2f", title="Matrice de Corrélation", color_continuous_scale="RdBu_r", aspect="auto")
fig.show()



# %% [markdown]
# Le prix est modérément corrélé positivement à la demande réelle (load_actual, ≈ 0,38) : quand la consommation augmente, les prix ont tendance à monter, ce qui est conforme à la logique de marché électricité.​
# 
# Le prix est modérément corrélé négativement à la production éolienne (wind_generation, ≈ −0,35), traduisant l’effet de l’éolien qui pousse le coût marginal moyen vers le bas et fait baisser les prix de marché lorsque le vent est abondant.​

# %%
# B. Corrélation Globale : Heatmap pour DK1
# On identifie les colonnes clés pour le Danemark Ouest (DK1)
# 1. Le Prix
col_price = [c for c in df.columns if "DK_1" in c and "price" in c]

# 2. La Consommation (Load)
col_load = [c for c in df.columns if "DK_1" in c and "load" in c and "actual" in c]

# 3. Toutes les Productions (Generation Actual)
# On cherche tout ce qui est "DK_1" + "generation" + "actual"
col_gen = [c for c in df.columns if "DK_1" in c and "generation_actual" in c]

# On combine tout
cols_to_corr = col_price + col_load + col_gen

#  2. NETTOYAGE ET RENOMMAGE (Pour que ce soit lisible) 
# On crée un sous-dataframe
df_dk1 = df[cols_to_corr].copy()

# On remplace les NaN par 0 uniquement pour la production (pas pour le prix !)
df_dk1[col_gen] = df_dk1[col_gen].fillna(0)

# On nettoie les noms des colonnes pour l'affichage
# Ex: "DK_1_wind_offshore_generation_actual" -> "Wind Offshore"
clean_names = {}
for col in df_dk1.columns:
    new_name = col.replace("DK_1_", "").replace("_generation_actual", "").replace("_entsoe_transparency", "")
    new_name = new_name.replace("price_day_ahead", "PRIX Spot").replace("load_actual", "CONSOMMATION")
    new_name = new_name.replace("_", " ").title() 
    clean_names[col] = new_name

df_dk1 = df_dk1.rename(columns=clean_names)

#  3. CALCUL DE LA CORRÉLATION 
# On supprime les colonnes qui sont toutes à 0 (sources d'énergie non présentes dans DK1)
df_dk1 = df_dk1.loc[:, (df_dk1 != 0).any(axis=0)]

# Matrice de corrélation
corr_matrix = df_dk1.corr()

#  4. AFFICHAGE DE LA HEATMAP GLOBALE 
fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r", 
    zmin=-1, zmax=1,
    title="Heatmap de Corrélation Globale : Écosystème DK1"
)

fig.update_layout(
    width=1000, height=800, # Un peu plus grand car il y a beaucoup de variables
    title_font_size=20,
    template="plotly_white"
)

# On tourne les labels de l'axe X pour que ce soit lisible
fig.update_xaxes(tickangle=-45)

fig.show()

# %% [markdown]
# Cette heatmap de corrélation pour DK1 montre clairement le rôle central de la demande et du vent (onshore + offshore) dans l’écosystème de prix.
# 
# Le prix spot est modérément corrélé positivement à la consommation (0,38), ce qui traduit le fait que les heures de forte demande tirent les prix vers le haut.​
# 
# Le prix est modérément corrélé négativement au vent total (−0,34) et aux composantes onshore/offshore (entre −0,31 et −0,33) : plus la production éolienne est élevée, plus le coût marginal baisse et plus le prix spot tend à diminuer.​
# 
# Les productions éoliennes onshore et offshore sont très fortement corrélées entre elles (0,98) et au vent total (0,88–0,98), ce qui reflète des conditions météo communes en mer du Nord et sur la côte ouest danoise.​
# 
# Le solaire n’est que faiblement corrélé au prix (0,02) et au vent (−0,07 à −0,16), mais un peu plus à la consommation (0,26), ce qui suggère qu’il reste marginal dans le mix DK1 tout en étant plutôt aligné sur des heures de forte demande diurne.​

# %%
# D. Analyse par Tranche : Impact de la Production Éolienne sur le Prix
#  1. PRÉPARATION DES DONNÉES (Création des tranches) 
# On crée des tranches (bins) de production éolienne de 500 MW (0-500, 500-1000, etc.)
step = 500
df_dk['Wind_Bin'] = (df_dk['wind_generation'] // step) * step

# On calcule la moyenne du prix et le nombre d'heures pour chaque tranche
df_agg = df_dk.groupby('Wind_Bin').agg(
    Prix_Moyen=('price', 'mean'),
    Nombre_Heures=('price', 'count')
).reset_index()

# On filtre les tranches vides ou aberrantes (si besoin)
df_agg = df_agg[df_agg['Wind_Bin'] >= 0]

#  2. CRÉATION DU GRAPHIQUE COMBINÉ 
# On utilise make_subplots pour avoir 2 axes Y (un pour le volume, un pour le prix)
fig = make_subplots(specs=[[{"secondary_y": True}]])

# A. L'HISTOGRAMME (Combien d'heures ça arrive ?)
fig.add_trace(
    go.Bar(
        x=df_agg['Wind_Bin'],
        y=df_agg['Nombre_Heures'],
        name="Fréquence (Heures)",
        marker_color='lightblue',
        opacity=0.6
    ),
    secondary_y=False, # Axe de gauche
)

# B. LA LIGNE (Quel est le prix moyen ?)
fig.add_trace(
    go.Scatter(
        x=df_agg['Wind_Bin'],
        y=df_agg['Prix_Moyen'],
        name="Prix Moyen (€/MWh)",
        mode='lines+markers',
        line=dict(color='firebrick', width=3),
        marker=dict(size=8)
    ),
    secondary_y=True, # Axe de droite
)

#  3. MISE EN FORME 
fig.update_layout(
    title="Analyse par Tranche : Impact de la Production Éolienne sur le Prix",
    title_font_size=18,
    xaxis_title=f"Production Éolienne (Tranches de {step} MW)",
    template="plotly_white",
    hovermode="x unified", # Affiche les deux infos au survol
    legend=dict(x=0.7, y=0.9)
)

# Titres des axes Y
fig.update_yaxes(title_text="Fréquence (Nombre d'heures)", secondary_y=False)
fig.update_yaxes(title_text="Prix Moyen (€/MWh)", secondary_y=True)

fig.show()

# %% [markdown]
# Ceci montre que plus il y'a la production éolienne, p;us le prix baisse. Ainsi de suite dans le sens inverse

# %%
#  E. Analyse des Prix Négatifs (Fréquence Mensuelle) 
# Insight : Quand les prix négatifs surviennent-ils le plus souvent ?
neg_df = df_dk[df_dk['price'] < 0].copy()
# Ajouter une colonne 'year' basée sur l'index datetime (il n'y a pas de colonne 'year' dans df_dk)
neg_df['year'] = neg_df.index.year
# 'month' existe déjà mais on peut la recalculer pour être sûr
neg_df['month'] = neg_df.index.month

negative_prices = neg_df.groupby(['year', 'month']).size().reset_index(name='count')

# Création d'une date fictive pour l'axe X
negative_prices['date'] = pd.to_datetime(negative_prices[['year', 'month']].assign(DAY=1))

fig = px.bar(
    negative_prices,
    x='date',
    y='count',
    title="Fréquence des Prix Négatifs (Nombre d'heures par mois)",
    labels={'count': "Nombre d'heures < 0€", 'date': 'Date'},
    color='count',
    color_continuous_scale='Reds'
)
fig.show()

# %% [markdown]
# ## 7. Modélisation Prédictive 

# %%
#  1. Préparation pour le ML 

df_ml = df_dk.copy()
df_ml['price_lag_24h'] = df_ml['price'].shift(24)
df_ml['wind_forecast_lag_24h'] = df_ml['wind_generation'].shift(24)
df_ml = df_ml.dropna()

features = ['hour', 'day_of_week', 'month', 'price_lag_24h', 'load_forecast', 'wind_forecast_lag_24h', 'solar_generation']
target = 'price'

# Split Train/Test (Juin 2019)
split_date = '2019-06-01'
X_train = df_ml.loc[df_ml.index < split_date, features]
y_train = df_ml.loc[df_ml.index < split_date, target]
X_test = df_ml.loc[df_ml.index >= split_date, features]
y_test = df_ml.loc[df_ml.index >= split_date, target]



# %%
#  2a.  LightGBM de Base 
print(" 1. Entraînement LightGBM (Base) ")
model_base = lgb.LGBMRegressor(n_estimators=500, random_state=42, verbose=-1)
model_base.fit(X_train, y_train)
y_pred_base = model_base.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
print(f"RMSE Base : {rmse_base:.2f} €/MWh")


# %%
#  2b. LightGBM Optimisé (GridSearch + TimeSeriesSplit) 
print("\n 2. Entraînement LightGBM (Optimisé) cele peut prendre un peu de temps donc ne vous inQUIètez pas normalment 3min 30 max  ")

# On définit une grille de paramètres à tester
param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [-1, 10, 20],
    'n_estimators': [500, 1000]
}

# Important : TimeSeriesSplit pour ne pas mélanger le futur et le passé dans la validation croisé
tscv = TimeSeriesSplit(n_splits=3)

lgb_opt = lgb.LGBMRegressor(random_state=42, verbose=-1)

# Recherche des meilleurs hyperparamètres (peut prendre 1à2 min)
grid_search = GridSearchCV(estimator=lgb_opt, param_grid=param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Meilleur modèle
best_model = grid_search.best_estimator_
print(f"Meilleurs params : {grid_search.best_params_}")

y_pred_opt = best_model.predict(X_test)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
print(f"RMSE Optimisé : {rmse_opt:.2f} €/MWh")




# %%
#  2c. SARIMAX (Modèle Statistique) 
print("\n 3. Entraînement SARIMAX ")
# NOTE : SARIMAX est très lent sur des milliers de données horaires. 
# Astuce : On utilise 'exog' (features) qui capturent déjà la saisonnalité, 
# donc on garde un ordre ARIMA simple (1,0,1) pour que ça tourne vite.

# Pour l'exemple, on réduit le train set aux 2 derniers mois pour éviter que ça plante/soit trop long
# Si tu as un gros serveur, tu peux utiliser tout X_train/y_train
train_size_limit = 24 * 60  # ~2 mois d'heures
X_train_sarima = X_train.iloc[-train_size_limit:]
y_train_sarima = y_train.iloc[-train_size_limit:]

# Définition du modèle (ARIMAX ici car on utilise des variables exogènes)
# order=(p,d,q). Ici (1,0,1) est standard. On peut ajouter seasonal_order=(...) mais c'est très lourd en calcul.
sarima_model = SARIMAX(y_train_sarima, exog=X_train_sarima, order=(1, 0, 1), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)

# Prédiction
# Attention : Pour SARIMAX, il faut fournir les exogènes du futur (X_test)
y_pred_sarima = sarima_result.predict(start=X_test.index[0], end=X_test.index[-1], exog=X_test)

rmse_sarima = np.sqrt(mean_squared_error(y_test, y_pred_sarima))
print(f"RMSE SARIMAX : {rmse_sarima:.2f} €/MWh")




# %%
#  VISUALISATION FINALE (Tableau Pandas) 

# 1. On rassemble les données
resultats = []

# Fonction pour ajouter une ligne proprement
def ajouter_modele(nom, y_true, y_pred):
    if y_pred is not None:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        resultats.append({
            "Modèle": nom,
            "RMSE (€)": round(rmse, 2),
            "MAE (€)": round(mae, 2),
            "Score R²": f"{r2:.2%}" # Format pourcentage propre
        })

# 2. Ajout des modèles (On utilise les noms de variables définis lors de l'entraînement)
# Note : J'utilise 'y_pred_base' etc. car c'est comme ça qu'on les a nommés plus haut
ajouter_modele("LightGBM (Base)", y_test, y_pred_base)
ajouter_modele("LightGBM (Optimisé)", y_test, y_pred_opt)

# On vérifie si SARIMAX existe avant de l'ajouter
if 'y_pred_sarima' in locals() and y_pred_sarima is not None:
    ajouter_modele("SARIMAX", y_test, y_pred_sarima)

# 3. Affichage du Tableau Simple
df_final = pd.DataFrame(resultats)

# Mise en forme (Optionnel : indexer par le nom du modèle pour un look plus propre)
df_final = df_final.set_index("Modèle")

print(" RÉSULTATS COMPARATIFS ")
# Si tu es dans Jupyter, 'display' rendra un joli tableau HTML
display(df_final)

# %%
#  3. Visualisation Interactive Comparée 
print("\n Génération du Graphique ")
fig = go.Figure()

# Réalité
fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name='Réel', 
                         line=dict(color='red', width=2)))

# LGBM Base (Ton modèle)
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred_base, name=f'LGBM Base (RMSE: {rmse_base:.2f})', 
                         line=dict(color='royalblue', dash='dot', width=1.5)))

# LGBM Optimisé
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred_opt, name=f'LGBM Opti (RMSE: {rmse_opt:.2f})', 
                         line=dict(color='green', width=1.5)))

# SARIMAX
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred_sarima, name=f'SARIMAX (RMSE: {rmse_sarima:.2f})', 
                         line=dict(color='orange', width=1.5, dash='dash')))

fig.update_layout(
    title="LGBM Base vs LGBM Optimisé vs SARIMAX",
    xaxis_title="Date", 
    yaxis_title="Prix (€/MWh)",
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.update_xaxes(rangeslider_visible=True)
# Zoom initial sur 1 semaine pour bien voir les détails
fig.update_xaxes(range=[y_test.index[0], y_test.index[168]]) 

fig.show()

# %%
# 4. Feature Importance (SHAP)
# Explainer sur le meilleur modèle
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap_sum = np.abs(shap_values).mean(axis=0)
imp_df = pd.DataFrame({
    'feature': features,
    'importance': shap_sum
}).sort_values('importance', ascending=True)

fig = px.bar(
    imp_df,
    x='importance',
    y='feature',
    orientation='h',
    color='importance',
    title="Importance des Variables (SHAP)",
    template="plotly_white"
)
fig.show()



