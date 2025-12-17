# %% [markdown]
# # Analyse Structurelle du Marché Électrique Danois (DK1 vs DK2)
# **Période d'étude : 2020 - 2025**
# 
# Le Danemark est divisé en deux zones de prix distinctes, reflétant une fracture géographique et énergétique :
# * **DK1 (Ouest - Jylland)** : Connectée à l'Allemagne, zone dominée par l'éolien, sujette à une forte volatilité.
# * **DK2 (Est - Sjælland/Copenhague)** : Connectée à la Suède, plus urbaine, avec une inertie thermique plus forte.
# 

# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Chargement des données 2020-2025
df = pd.read_csv("../data/raw/time_series_60min_fr_dk_20-25_ENRICHIE_FULL.csv", parse_dates=['utc_timestamp'])
df = df.set_index('utc_timestamp')

# Configuration des couleurs
COLORS = {'DK1': '#962b2b', 'DK2': "#2861FD"}

# %% [markdown]
# ## 1. Audit de la Qualité des Données
# Avant toute analyse visuelle, nous vérifions l'intégrité temporelle des séries (doublons, trous).
# 

# %%
# --- Vérification et Nettoyage des Données ---

print(f"Taille avant nettoyage : {df.shape}")

# 1. Traitement des doublons temporels
# Si des doublons existent, on garde uniquement la première occurrence
if df.index.duplicated().any():
    print(f"Suppression de {df.index.duplicated().sum()} doublons...")
    df = df[~df.index.duplicated(keep='first')]

# 2. Suppression des valeurs manquantes
# On cible les colonnes de prix (DK1 et DK2). Si l'un des deux manque, on supprime la ligne.
cols_to_check = ['DK_1_price_day_ahead', 'DK_2_price_day_ahead']

# Avant suppression
missing_count = df[cols_to_check].isnull().sum().sum()
print(f"Valeurs manquantes détectées (Prix) : {missing_count}")

# SUPPRESSION EFFECTIVE
df = df.dropna(subset=cols_to_check)

# 3. Vérification finale
print(f"\nTaille après nettoyage : {df.shape}")
print("Reste-t-il des NaN ?")
print(df[cols_to_check].isnull().sum())

# 4. Aperçu des statistiques descriptives (sur données propres)
print("\nStatistiques après nettoyage :")
print(df[cols_to_check].describe().round(2))

# %% [markdown]
# ## 2. Analyses Visuelles
# Les graphiques ci-dessous présentent la distribution et l'évolution des prix.
# 

# %% [markdown]
# ### Distribution des Prix : La queue des prix négatifs

# %%
fig1 = px.histogram(
    df, 
    x=["DK_1_price_day_ahead", "DK_2_price_day_ahead"],
    barmode="overlay", # Superposition
    nbins=150,
    title="Distribution des Prix : DK1 (Volatil) vs DK2 (Stable)",
    labels={"value": "Prix (€/MWh)", "variable": "Zone"},
    template="plotly_white",
    opacity=0.6,
    color_discrete_map={"DK_1_price_day_ahead": COLORS['DK1'], "DK_2_price_day_ahead": COLORS['DK2']}
)
fig1.update_layout(xaxis_range=[-100, 400]) # Zoom utile
fig1.show()

# %% [markdown]
# La majorité des prix pour les deux zones se situe dans une plage modérée (en gros entre 0 et 150 €/MWh), ce qui indique un marché généralement concentré autour de niveaux de prix “normaux”.​
# 
# DK1 semble présenter des queues plus marquées : on observe plus d’occurrences de prix très élevés (au-delà de 200 €/MWh, jusqu’à près de 400 €/MWh) et quelques valeurs négatives, alors que DK2 est plus “ramassée” autour du cœur de la distribution. Cela confirme un comportement plus volatil de DK1 et une plus grande stabilité de DK2.​
# 
# Analytquement, cela signifie que les risques de prix extrêmes (pics de prix ou prix négatifs) sont plus importants dans DK1 que dans DK2, ce qui a des implications en termes de gestion des risques, de couverture (hedging) et de stratégie d’offre sur la zone DK1.

# %% [markdown]
# ## Évolution Temporelle
# 

# %%
# Moyenne hebdomadaire pour lisibilité
df_weekly = df[['DK_1_price_day_ahead', 'DK_2_price_day_ahead']].resample('W').mean()

fig2 = px.line(
    df_weekly,
    title="Chronologie des Prix (Moyenne Hebdo)",
    labels={"value": "Prix Moyen (€/MWh)", "variable": "Zone"},
    template="plotly_white",
    color_discrete_map={"DK_1_price_day_ahead": COLORS['DK1'], "DK_2_price_day_ahead": COLORS['DK2']}
)
fig2.show()

# %% [markdown]
# On observe un choc majeur des prix autour de 2022, avec des pics dépassant 500 €/MWh pour DK2 et plus de 300 €/MWh pour DK1, ce qui correspond à une période de tension extrême sur le marché (crise énergétique, volatilité des combustibles, etc.).​
# 
# Après 2022, les prix se normalisent mais restent plus volatils et plus élevés qu’en 2020, ce qui suggère un nouveau régime de prix plus incertain. DK2 semble légèrement plus sujette à des pics ponctuels (amplitudes un peu plus fortes), alors que DK1 suit de près mais avec des extrêmes un peu plus contenus.​
# 
# Analytquement, cela indique que le risque de prix extrêmes est concentré sur une période historique (2022), tandis que sur le reste de l’horizon les deux zones sont fortement corrélées et évoluent dans des niveaux de prix comparables, ce qui est crucial pour la planification, la couverture et la modélisation des séries temporelles.

# %% [markdown]
# ##  Relation Prix vs Consommation (Charge)

# %%
fig3a = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("<b>DK1 (Ouest)</b> : Charge vs Prix", "<b>DK2 (Est)</b> : Charge vs Prix")
)

# partie DK1 
fig3a.add_trace(go.Scatter(
    x=df.sample(3000)["DK_1_load_actual_entsoe_transparency"], 
    y=df.sample(3000)["DK_1_price_day_ahead"], 
    mode='markers', name='DK1', 
    marker=dict(color=COLORS['DK1'], opacity=0.4)
), row=1, col=1)

# DK2 
fig3a.add_trace(go.Scatter(
    x=df.sample(3000)["DK_2_load_actual_entsoe_transparency"], 
    y=df.sample(3000)["DK_2_price_day_ahead"], 
    mode='markers', name='DK2', 
    marker=dict(color=COLORS['DK2'], opacity=0.4)
), row=1, col=2)

fig3a.update_xaxes(title_text="Conso (MW)", row=1, col=1)
fig3a.update_xaxes(title_text="Conso (MW)", row=1, col=2)
fig3a.update_yaxes(title_text="Prix (€/MWh)", row=1, col=1)
fig3a.update_layout(template="plotly_white", showlegend=False, title_text="Impact de la Demande : Comparatif")
fig3a.show()

# %% [markdown]
# Dans les deux zones, la relation “Conso → Prix” n’est pas linéaire : la hausse de la demande s’accompagne d’une tendance à des prix plus volatils, mais le nuage reste très large, ce qui indique que d’autres facteurs (production, interconnexions, combustibles) jouent fortement sur les prix.​
# 
# DK1 semble avoir une plage de consommation plus élevée et davantage de points extrêmes (prix très hauts ou très bas), ce qui renforce l’idée d’un système plus tendu et plus sujet aux chocs de prix quand la demande est forte. DK2, avec une demande plus modérée, montre un nuage légèrement plus resserré, suggérant une sensibilité un peu moindre de ses prix à la demande pure.​

# %% [markdown]
# ##  Relation Prix vs Température (Absence de lien)
# 

# %%
fig3b = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("<b>DK1 (Ouest)</b> : Température vs Prix", "<b>DK2 (Est)</b> : Température vs Prix")
)

# DK1 
fig3b.add_trace(go.Scatter(
    x=df.sample(3000)["temperature_denmark"], 
    y=df.sample(3000)["DK_1_price_day_ahead"], 
    mode='markers', name='DK1', 
    marker=dict(color=COLORS['DK1'], opacity=0.4)
), row=1, col=1)

# DK2 
fig3b.add_trace(go.Scatter(
    x=df.sample(3000)["temperature_denmark"], 
    y=df.sample(3000)["DK_2_price_day_ahead"], 
    mode='markers', name='DK2', 
    marker=dict(color=COLORS['DK2'], opacity=0.4)
), row=1, col=2)

fig3b.update_xaxes(title_text="Température (°C)", row=1, col=1)
fig3b.update_xaxes(title_text="Température (°C)", row=1, col=2)
fig3b.update_yaxes(title_text="Prix (€/MWh)", row=1, col=1)
fig3b.update_layout(template="plotly_white", showlegend=False, title_text="Impact de la Température (Nul)")
fig3b.show()

# %% [markdown]
# Visuellement, il n’y a pas de relation linéaire simple entre température et prix : le nuage reste très large pour toutes les températures, ce qui suggère une corrélation faible ou non monotone. On observe toutefois que les prix extrêmes apparaissent surtout dans une plage de températures modérées (autour de 0–15 °C), ce qui peut correspondre à des périodes de forte demande chauffage/éclairage ou de tensions système.​
# 
# DK2 semble présenter une dispersion verticale un peu plus marquée (plus de points très chers pour des températures comparables), ce qui peut indiquer une sensibilité légèrement plus forte du marché de l’Est aux conditions de demande/production sous-jacentes, même si la température seule n’explique pas les niveaux de prix.​

# %% [markdown]
# ## . Le Facteur Clé : La Vitesse du Vent
# 

# %%
fig3c = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("<b>DK1 (Ouest)</b> : Vent vs Prix", "<b>DK2 (Est)</b> : Vent vs Prix")
)

# DK1 (
fig3c.add_trace(go.Scatter(
    x=df.sample(3000)["wind_speed_denmark"], 
    y=df.sample(3000)["DK_1_price_day_ahead"], 
    mode='markers', name='DK1', 
    marker=dict(color=COLORS['DK1'], opacity=0.4)
), row=1, col=1)

# DK2 
fig3c.add_trace(go.Scatter(
    x=df.sample(3000)["wind_speed_denmark"], 
    y=df.sample(3000)["DK_2_price_day_ahead"], 
    mode='markers', name='DK2', 
    marker=dict(color=COLORS['DK2'], opacity=0.4)
), row=1, col=2)

fig3c.update_xaxes(title_text="Vent (m/s)", row=1, col=1)
fig3c.update_xaxes(title_text="Vent (m/s)", row=1, col=2)
fig3c.update_yaxes(title_text="Prix (€/MWh)", row=1, col=1)
fig3c.update_layout(template="plotly_white", showlegend=False, title_text="Le Roi Vent : Corrélation Inverse")
fig3c.show()

# %% [markdown]
# On devine une corrélation plutôt inverse : les prix élevés semblent plus fréquents lorsque la vitesse du vent est faible à modérée, alors qu’aux vitesses de vent plus élevées, les prix ont tendance à se rapprocher de la zone 0–150 €/MWh. Cela reflète l’effet classique de la production éolienne abondante qui fait baisser les prix de marché.​
# 
# L’effet est présent dans les deux zones, mais DK2 semble montrer une dispersion légèrement plus forte des prix pour des vitesses de vent similaires, ce qui peut indiquer que le vent n’est pas le seul facteur dominant (structure de production différente, interconnexions, contraintes réseau). Globalement, toutefois, le graphique confirme que le “roi vent” joue un rôle amortisseur sur les prix, surtout quand la vitesse dépasse certains seuils.​

# %% [markdown]
# ## PRIX vs Conso

# %%


# On s'assure d'avoir les colonnes globales 
if 'DK_price_day_ahead' not in df.columns:
    # Moyenne simple si pas de colonne globale
    df['DK_price_day_ahead'] = (df['DK_1_price_day_ahead'] + df['DK_2_price_day_ahead']) / 2

if 'DK_load_actual_entsoe_transparency' not in df.columns:
    df['DK_load_actual_entsoe_transparency'] = df['DK_1_load_actual_entsoe_transparency'] + df['DK_2_load_actual_entsoe_transparency']

# 3. Création des "Tranches de Consommation" (Binning)

df['Conso_Bin'] = (df['DK_load_actual_entsoe_transparency'] // 100) * 100

# 4. Calcul du Prix Moyen par Tranche
df_trend = df.groupby('Conso_Bin')['DK_price_day_ahead'].mean().reset_index()

# On filtre les valeurs extrêmes (erreurs de mesure ou cas rares) pour avoir une belle courbe
df_trend = df_trend[df_trend['Conso_Bin'] > 1000] 

# 5. Visualisation
fig = px.line(
    df_trend, 
    x='Conso_Bin', 
    y='DK_price_day_ahead',
    markers=True,
    title="Tendance : Prix Moyen selon le Niveau de Consommation (Danemark)",
    labels={
        'Conso_Bin': 'Niveau de Consommation (MW)', 
        'DK_price_day_ahead': 'Prix Moyen de l\'Électricité (€/MWh)'
    }
)

# Ajout d'une ligne de tendance (Régression) pour voir la pente
fig_trendline = px.scatter(df_trend, x='Conso_Bin', y='DK_price_day_ahead', trendline="ols")
fig.add_trace(fig_trendline.data[1])
fig.update_layout(
    template="plotly",
    xaxis_title="Si le Danemark consomme... (MW)",
    yaxis_title="Alors le prix moyen est de... (€/MWh)"
)

fig.show()

# %% [markdown]
# ##  Bilan : Moyennes et Extrêmes
# * **Moyenne :** Les prix moyens sont proches, mais DK1 est souvent légèrement moins cher grâce au vent.
# 

# %%
# 1. Préparation des données (Moyennes Annuelles)E
df_annual = df[['DK_1_price_day_ahead', 'DK_2_price_day_ahead']].resample('YE').mean()

# 2. Transformation en format "Long" (nécessaire pour personnaliser le texte par barre)
# On transforme le tableau pour avoir une colonne "Zone", une "Prix" et une "Année"
df_melted = df_annual.reset_index().melt(id_vars=df_annual.index.name, var_name='Zone', value_name='Prix')
# On s'assure que l'année est bien juste l'année (et pas la date complète 31-12-202X)
df_melted['Année'] = df_melted[df_annual.index.name].dt.year

# 3. Calcul du pourcentage d'évolution par groupe (Zone)
# On trie pour être sûr que les calculs se font dans l'ordre chronologique
df_melted = df_melted.sort_values(['Zone', 'Année'])
df_melted['Pct_Change'] = df_melted.groupby('Zone')['Prix'].pct_change() * 100

# 4. Création du texte personnalisé (Prix + Variation)
def create_label(row):
    price = f"{row['Prix']:.1f} €"
    
    # Si c'est la première année (NaN), on affiche juste le prix
    if pd.isna(row['Pct_Change']): 
        return price
    
    pct = row['Pct_Change']
    signe = "+" if pct > 0 else ""
    
    # Choix de la couleur et de la flèche
    if pct > 0:
        color = "green"  # Ou un code hexadécimal comme '#00aa00'
        fleche = "▲"
    else:
        color = "red"    # Ou un code hexadécimal comme '#cc0000'
        fleche = "▼"
    
    # On utilise HTML <span style='color:...'> pour colorer juste le pourcentage
    return f"{price}<br><span style='color:{color}'>({fleche} {signe}{pct:.1f}%)</span>"

df_melted['Label_Text'] = df_melted.apply(create_label, axis=1)

# 5. Le Graphique
fig4 = px.bar(
    df_melted, 
    x='Année', 
    y='Prix',
    color='Zone',
    barmode='group',
    text='Label_Text', 
    title="Prix Moyen Annuel et Évolution",
    labels={"Prix": "Prix Moyen (€/MWh)", "Année": "Année"},
    template="plotly",
    color_discrete_map={"DK_1_price_day_ahead": COLORS['DK1'], "DK_2_price_day_ahead": COLORS['DK2']}
)

# Petite retouche pour que le texte soit bien visible
fig4.update_traces(textposition='inside') 
fig4.show()

# %% [markdown]
# En 2022, les prix moyens annuels explosent pour DK1 et DK2 (environ 219 €/MWh et 210 €/MWh), très au‑dessus des niveaux de 2020–2021. Ce saut correspond à la crise gazière provoquée par l’invasion de l’Ukraine : réduction des flux russes, tensions sur l’approvisionnement, recours massif au GNL plus cher.
# 
# Comme le gaz sert souvent de technologie marginale pour fixer le prix de gros de l’électricité, la flambée du gaz s’est transmise presque mécaniquement aux prix électriques sur l’ensemble de l’Europe du Nord, y compris au Danemark (DK1/DK2), d’où ce pic exceptionnel.
# 
# À partir de 2023, la baisse relative des prix (tout en restant plus hauts que 2020) reflète la combinaison de mesures de sobriété, de diversification des sources d’énergie, de remplissage des stocks de gaz et de nouvelles capacités renouvelables, qui ont atténué, sans l’effacer totalement, l’impact du choc initial lié à la guerre.

# %%
# --- Graphique Box (Outliers) ---
fig5 = px.box(
    df, 
    y=["DK_1_price_day_ahead", "DK_2_price_day_ahead"],
    title="5. Volatilité et Prix Négatifs",
    labels={"value": "Prix (€/MWh)", "variable": "Zone"},
    template="plotly",
    color_discrete_map={"DK_1_price_day_ahead": COLORS['DK1'], "DK_2_price_day_ahead": COLORS['DK2']}
)
fig5.update_yaxes(range=[-100, 400])
fig5.show()

# %% [markdown]
# Les boîtes de DK1 et DK2 sont très proches : médiane et quartiles quasi identiques, ce qui suggère des profils de prix “typique” similaires entre l’Ouest et l’Est. La différence entre les zones se joue donc surtout dans les extrêmes plutôt que dans le cœur de la distribution.​
# 
# La présence de moustaches longues vers le haut et de points négatifs illustre une forte volatilité structurelle : périodes de surproduction (prix négatifs) et de tension (pics > 300–400 €/MWh). Pour un acteur de marché, cela implique un risque important aux deux extrémités, rendant cruciale la gestion de flexibilité, de stockage et de couverture pour absorber ces chocs.​

# %% [markdown]
# ## 4. Analyse de la Saisonnalité
# Comparaison des profils de Prix et de Consommation.
# 

# %%

# Création des features temporelles
df['Year'] = df.index.year
df['Month'] = df.index.month
df['DayOfWeek'] = df.index.dayofweek # 0=Lundi, 6=Dimanche
df['Hour'] = df.index.hour

# Fonction pour les Saisons
def get_season(month):
    if month in [12, 1, 2]: return 'Hiver'
    elif month in [3, 4, 5]: return 'Printemps'
    elif month in [6, 7, 8]: return 'Eté'
    else: return 'Automne'

df['Season'] = df['Month'].apply(get_season)

# Couleurs
C_DK1 = '#E31A1C' # Rouge (Ouest/Vent)
C_DK2 = '#1F78B4' # Bleu (Est/Ville)
COLORS = {'DK1': C_DK1, 'DK2': C_DK2}

# %% [markdown]
# 

# %%
#   Évolution Mensuelle COMPLETE (Prix vs Conso / DK1 vs DK2) 

# 1. On calcule la moyenne par mois pour les 4 colonnes
cols_monthly = [
    'DK_1_price_day_ahead', 'DK_2_price_day_ahead',
    'DK_1_load_actual_entsoe_transparency', 'DK_2_load_actual_entsoe_transparency'
]
df_monthly = df.groupby('Month')[cols_monthly].mean()

# 2. On crée le graphique à double axe Y
fig_season = make_subplots(specs=[[{"secondary_y": True}]])

#  ZONE DK1 (OUEST - ROUGE) 
# Prix (Ligne pleine) -> Axe Gauche
fig_season.add_trace(go.Scatter(
    x=df_monthly.index, y=df_monthly['DK_1_price_day_ahead'],
    mode='lines+markers', name='Prix DK1 (Ouest)',
    line=dict(color='#E31A1C', width=3)
), secondary_y=False)

# Conso (Ligne Pointillée) -> Axe Droit
fig_season.add_trace(go.Scatter(
    x=df_monthly.index, y=df_monthly['DK_1_load_actual_entsoe_transparency'],
    mode='lines', name='Conso DK1',
    line=dict(color='#E31A1C', dash='dot', width=2)
), secondary_y=True)

#  ZONE DK2 (EST - BLEU) 
# Prix (Ligne pleine) -> Axe Gauche
fig_season.add_trace(go.Scatter(
    x=df_monthly.index, y=df_monthly['DK_2_price_day_ahead'],
    mode='lines+markers', name='Prix DK2 (Est)',
    line=dict(color='#1F78B4', width=3)
), secondary_y=False)

# Conso (Ligne Pointillée) -> Axe Droit
fig_season.add_trace(go.Scatter(
    x=df_monthly.index, y=df_monthly['DK_2_load_actual_entsoe_transparency'],
    mode='lines', name='Conso DK2',
    line=dict(color='#1F78B4', dash='dot', width=2)
), secondary_y=True)

# 3. Mise en forme
fig_season.update_layout(
    title="Saisonnalité Mensuelle Comparée : Prix vs Consommation (DK1 vs DK2)",
    xaxis=dict(
        tickmode='array', 
        tickvals=list(range(1, 13)), 
        ticktext=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    ),
    template="plotly_white",
    legend=dict(orientation="h", y=-0.2) # Légende en bas pour ne pas cacher les courbes
)

# Titres des axes
fig_season.update_yaxes(title_text="Prix (€/MWh)", secondary_y=False)
fig_season.update_yaxes(title_text="Consommation (MW)", secondary_y=True)

fig_season.show()

# %% [markdown]
# On observe une nette saisonnalité mensuelle : la consommation est maximale en hiver (janvier‑février puis novembre‑décembre) et minimale en été, ce qui est cohérent avec les besoins de chauffage et de lumière dans un climat nordique.​
# 
# Les prix de gros suivent globalement cette dynamique, avec des hausses marquées lors des mois froids et parfois en août, période où la demande reste élevée alors que certaines capacités de production peuvent être en maintenance, ce qui tend à tendre le marché.​
# 
# DK1 et DK2 présentent des profils très proches, signe d’un marché largement intégré ; les écarts ponctuels de prix reflètent surtout des congestions de réseau et un mix de production légèrement différent entre l’ouest et l’est du Danemark.

# %% [markdown]
# 

# %%
season_order = ['Hiver', 'Printemps', 'Eté', 'Automne']

# Calcul des moyennes par saison
cols_metrics = [
    'DK_1_price_day_ahead', 'DK_2_price_day_ahead',
    'DK_1_load_actual_entsoe_transparency', 'DK_2_load_actual_entsoe_transparency'
]
df_season = df.groupby('Season')[cols_metrics].mean()

# On force l'ordre Hiver -> Automne
df_season = df_season.reindex(season_order)

#  CRÉATION DU GRAPHIQUE 
fig_saison_clim = make_subplots(specs=[[{"secondary_y": True}]])

#  ZONE DK1 (OUEST - ROUGE) 
fig_saison_clim.add_trace(go.Scatter(
    x=df_season.index, y=df_season['DK_1_price_day_ahead'],
    mode='lines+markers', name='Prix DK1 (Ouest)',
    line=dict(color='#E31A1C', width=3)
), secondary_y=False)

fig_saison_clim.add_trace(go.Scatter(
    x=df_season.index, y=df_season['DK_1_load_actual_entsoe_transparency'],
    mode='lines', name='Conso DK1',
    line=dict(color='#E31A1C', dash='dot', width=2)
), secondary_y=True)

#  ZONE DK2 (EST - BLEU) 
fig_saison_clim.add_trace(go.Scatter(
    x=df_season.index, y=df_season['DK_2_price_day_ahead'],
    mode='lines+markers', name='Prix DK2 (Est)',
    line=dict(color='#1F78B4', width=3)
), secondary_y=False)

fig_saison_clim.add_trace(go.Scatter(
    x=df_season.index, y=df_season['DK_2_load_actual_entsoe_transparency'],
    mode='lines', name='Conso DK2',
    line=dict(color='#1F78B4', dash='dot', width=2)
), secondary_y=True)

#  MISE EN FORME 
fig_saison_clim.update_layout(
    title="Saisonnalité Climatique : Prix vs Conso (Impact Hivernal)",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.2)
)
fig_saison_clim.update_yaxes(title_text="Prix (€/MWh)", secondary_y=False)
fig_saison_clim.update_yaxes(title_text="Consommation (MW)", secondary_y=True)

fig_saison_clim.show()

# %% [markdown]
# les prix et la consommation augmentent nettement en hiver, signe d’un impact hivernal marqué sur la demande électrique. Les deux zones danoises DK1 (Ouest) et DK2 (Est) présentent des profils similaires, mais DK1 enregistre des prix légèrement plus élevés en été, probablement liés à des contraintes de réseau et à la structure locale de la demande.

# %%
#  PRÉPARATION (JOURS DE LA SEMAINE) 
# On s'assure d'avoir la colonne DayOfWeek (0=Lundi, 6=Dimanche)
df['DayOfWeek'] = df.index.dayofweek

# Calcul des moyennes par Jour
df_weekly = df.groupby('DayOfWeek')[cols_metrics].mean()

#  CRÉATION DU GRAPHIQUE 
fig_weekly = make_subplots(specs=[[{"secondary_y": True}]])

#  ZONE DK1 (OUEST - ROUGE) 
fig_weekly.add_trace(go.Scatter(
    x=df_weekly.index, y=df_weekly['DK_1_price_day_ahead'],
    mode='lines+markers', name='Prix DK1 (Ouest)',
    line=dict(color='#E31A1C', width=3)
), secondary_y=False)

fig_weekly.add_trace(go.Scatter(
    x=df_weekly.index, y=df_weekly['DK_1_load_actual_entsoe_transparency'],
    mode='lines', name='Conso DK1',
    line=dict(color='#E31A1C', dash='dot', width=2)
), secondary_y=True)

#  ZONE DK2 (EST - BLEU) 
fig_weekly.add_trace(go.Scatter(
    x=df_weekly.index, y=df_weekly['DK_2_price_day_ahead'],
    mode='lines+markers', name='Prix DK2 (Est)',
    line=dict(color='#1F78B4', width=3)
), secondary_y=False)

fig_weekly.add_trace(go.Scatter(
    x=df_weekly.index, y=df_weekly['DK_2_load_actual_entsoe_transparency'],
    mode='lines', name='Conso DK2',
    line=dict(color='#1F78B4', dash='dot', width=2)
), secondary_y=True)

#  MISE EN FORME 
fig_weekly.update_layout(
    title="Profil Hebdomadaire : L'effet Week-end",
    xaxis=dict(
        tickmode='array', 
        tickvals=[0, 1, 2, 3, 4, 5, 6], 
        ticktext=['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    ),
    template="plotly_white",
    legend=dict(orientation="h", y=-0.2)
)
fig_weekly.update_yaxes(title_text="Prix (€/MWh)", secondary_y=False)
fig_weekly.update_yaxes(title_text="Consommation (MW)", secondary_y=True)

fig_weekly.show()

# %% [markdown]
# Le profil hebdomadaire montre une demande et des prix de l’électricité relativement stables du lundi au jeudi, avec un léger repli le vendredi, puis une forte baisse le week‑end. Cette dynamique reflète surtout la baisse de l’activité industrielle et tertiaire le samedi et le dimanche, où le système est davantage porté par la consommation résidentielle.​
# 
# Les zones DK1 (Ouest) et DK2 (Est) présentent des trajectoires presque parallèles, confirmant que les signaux de prix Nord Pool restent cohérents au niveau national, avec des écarts dus principalement aux congestions de réseau ou aux différences locales de mix de production

# %%


#  On trie les prix du plus grand au plus petit pour chaque zone
dk1_sorted = df['DK_1_price_day_ahead'].sort_values(ascending=False).values
dk2_sorted = df['DK_2_price_day_ahead'].sort_values(ascending=False).values

#  On crée l'axe X (Pourcentage du temps : 0% à 100%)
x_axis = np.linspace(0, 100, len(dk1_sorted))

# Graphique
fig_mono = go.Figure()

# DK1 (Rouge)
fig_mono.add_trace(go.Scatter(
    x=x_axis, y=dk1_sorted,
    mode='lines', name='DK1 (Ouest)',
    line=dict(color='#E31A1C', width=3)
))

# DK2 (Bleu)
fig_mono.add_trace(go.Scatter(
    x=x_axis, y=dk2_sorted,
    mode='lines', name='DK2 (Est)',
    line=dict(color='#1F78B4', width=2)
))

# Ligne zéro (pour bien voir les prix négatifs)
fig_mono.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="0 €/MWh")

fig_mono.update_layout(
    title="Courbe Monotone des Prix (Price Duration Curve)",
    xaxis_title="Pourcentage de l'année (%)",
    yaxis_title="Prix (€/MWh)",
    template="plotly_white",
    hovermode="x unified"
)
fig_mono.show()

# %% [markdown]
# La courbe de durée des prix montre que, pendant une très faible fraction de l’année, les prix de gros peuvent dépasser plusieurs centaines d’euros par MWh, traduisant des situations de tension extrême sur le système (forte demande ou faible disponibilité de production).

# %%
from IPython.display import display, Markdown 
#  FONCTION GÉNÉRATRICE DE RAPPORT 
def generer_rapport_zone(df, zone_name, col_prix, col_conso, col_vent, col_temp):
    
    # Couleur thématique
    couleur_theme = 'Reds' if "DK1" in zone_name else 'Blues'
    couleur_chart = '#E31A1C' if "DK1" in zone_name else '#1F78B4'
    
    display(Markdown(f"# RAPPORT GLOBAL : {zone_name}"))
    

    #   HEATMAP (Mois x Heure) 
    pivot = df.groupby(['Month', 'Hour'])[col_prix].mean().unstack()
    fig_heat = px.imshow(
        pivot,
        labels=dict(x="Heure", y="Mois", color="Prix (€)"),
        x=pivot.columns,
        y=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'],
        title=f"Signature Horaire & Mensuelle ({zone_name})",
        color_continuous_scale='RdBu_r', origin='upper'
    )
    fig_heat.show()

    #  CORRÉLATIONS (Matrice + Top ) 
    # Sélection des colonnes
    cols = [col_prix, col_conso, col_vent, col_temp]
    renames = {col_prix: 'Prix', col_conso: 'Conso', col_vent: 'Vent (Mix)', col_temp: 'Température'}
    
    corr_matrix = df[cols].rename(columns=renames).corr()
    
    # Heatmap de corrélation
    fig_corr = px.imshow(
        corr_matrix, text_auto='.2f', aspect="auto",
        color_continuous_scale='RdBu_r',
        title=f"Matrice de Facteurs d'Influence ({zone_name})"
    )
    fig_corr.show()

    
    # On vire les doublons et la diagonale
    corr_unstacked = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().reset_index()
    corr_unstacked.columns = ['Variable A', 'Variable B', 'Corrélation']

    # Top Positif
    top_pos = corr_unstacked.sort_values(by='Corrélation', ascending=False).head(5)
    display(Markdown(f" TOP CORRÉLATIONS POSITIVES ({zone_name})"))
    display(top_pos.style.background_gradient(cmap='Greens', axis=0))

    # Top Négatif
    top_neg = corr_unstacked.sort_values(by='Corrélation', ascending=True).head(5)
    display(Markdown(f" TOP CORRÉLATIONS NÉGATIVES ({zone_name})"))
    display(top_neg.style.background_gradient(cmap='Reds_r', axis=0))
    
    print("\n" + "="*80 + "\n")

#  EXÉCUTION DU RAPPORT 

# ZONE DK1 (OUEST)
generer_rapport_zone(
    df, "DK1 (Ouest - Vent)", 
    'DK_1_price_day_ahead', 'DK_1_load_actual_entsoe_transparency', 
    'wind_speed_denmark', 'temperature_denmark'
)

# ZONE DK2 (EST)
generer_rapport_zone(
    df, "DK2 (Est - Urbain)", 
    'DK_2_price_day_ahead', 'DK_2_load_actual_entsoe_transparency', 
    'wind_speed_denmark', 'temperature_denmark'
)

# %% [markdown]
# ## Matrice de Corrélation : Prix, Conso & Production

# %%
# --- 1. Calcul des colonnes de production ---
def sum_production(zone_prefix, df_source):
    # Toutes les sources (Charbon, Gaz, Vent, etc.) pour la zone
    all_prod_cols = [c for c in df_source.columns if zone_prefix in c and "Actual Aggregated" in c]
    
    # Juste le Vent et Solaire
    renew_cols = [c for c in all_prod_cols if "Wind" in c or "Solar" in c]
    
    # On somme (en remplaçant les NaN par 0)
    total_prod = df_source[all_prod_cols].fillna(0).sum(axis=1)
    renew_prod = df_source[renew_cols].fillna(0).sum(axis=1)
    
    return total_prod, renew_prod

# Calcul pour DK1 et DK2
# (On suppose que 'df' est déjà chargé dans ton environnement)
df['Prod_Totale_DK1'], df['Vent_Solaire_DK1'] = sum_production("DK1_", df)
df['Prod_Totale_DK2'], df['Vent_Solaire_DK2'] = sum_production("DK2_", df)

# --- 2. Sélection et Matrice de Corrélation ---
cols_heatmap = {
    'DK_1_price_day_ahead': 'Prix DK1',
    'DK_1_load_actual_entsoe_transparency': 'Conso DK1',
    'Prod_Totale_DK1': 'Prod. Totale DK1',
    'Vent_Solaire_DK1': 'Vent+Solaire DK1',
    
    'DK_2_price_day_ahead': 'Prix DK2',
    'DK_2_load_actual_entsoe_transparency': 'Conso DK2',
    'Prod_Totale_DK2': 'Prod. Totale DK2',
    'Vent_Solaire_DK2': 'Vent+Solaire DK2'
}

# Calcul de la matrice
df_corr = df[list(cols_heatmap.keys())].rename(columns=cols_heatmap).corr()

# --- 3. Affichage de la Heatmap (Plotly) ---
fig = px.imshow(
    df_corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Matrice de Corrélation : Prix, Conso & Production"
)

fig.update_layout(
    width=800, height=700,
    title_font_size=18,
    template="plotly_white"
)

fig.show()

# --- 4. Extraction des Tops 5 Positifs et Négatifs ---

def get_top_correlations(corr_matrix, n=5):
    # On crée un masque pour ignorer la diagonale (toujours 1.0) 
    # et le triangle supérieur (pour éviter les doublons A-B et B-A)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # On applique le masque et on "déplie" la matrice en liste
    corr_unstacked = corr_matrix.mask(mask).unstack().dropna()
    
    # On trie les valeurs
    sorted_corr = corr_unstacked.sort_values(ascending=False)
    
    return sorted_corr.head(n), sorted_corr.tail(n)

top_pos, top_neg = get_top_correlations(df_corr, n=5)



# %% [markdown]
# ## Coorélation : Météo -> Production -> Prix

# %%
if 'DK_Total_Wind_Onshore' not in df.columns:
    # On cherche les colonnes wind/solar
    wind_cols = [c for c in df.columns if 'Wind' in c and 'Actual' in c]
    solar_cols = [c for c in df.columns if 'Solar' in c and 'Actual' in c]
    df['DK_Total_Wind'] = df[wind_cols].sum(axis=1)
    df['DK_Total_Solar'] = df[solar_cols].sum(axis=1)
else:
    # Si elles existent déjà (via mon script précédent)
    df['DK_Total_Wind'] = df['DK_Total_Wind_Onshore'] + df.get('DK_Total_Wind_Offshore', 0)
    # DK_Total_Solar existe normalement déjà

# Pour l'analyse solaire, on ne garde que le JOUR (quand le soleil peut briller)
# Sinon la nuit (Nuages=100%, Solaire=0) fausse la corrélation
df_day = df[df['DK_Total_Solar'] > 10].copy()

#  CALCUL DES CORRÉLATIONS ---
corr_cloud = df_day['cloud_cover_denmark'].corr(df_day['DK_Total_Solar'])
corr_wind = df['wind_speed_denmark'].corr(df['DK_Total_Wind'])
corr_price = df['DK_Total_Wind'].corr(df['DK_1_price_day_ahead'])

print(f" RÉSULTATS STATISTIQUES :")
print(f"1. Nuages vs Solaire : {corr_cloud:.2f} ")
print(f"2. Vent vs Production : {corr_wind:.2f} ")
print(f"3. Vent vs Prix : {corr_price:.2f} ")

#  VISUALISATION (3 Graphiques côte à côte) ---

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        f"Solaire vs Nuages (Corr: {corr_cloud:.2f})", 
        f"Éolien vs Vitesse Vent (Corr: {corr_wind:.2f})", 
        f"Impact Prix vs Vent (Corr: {corr_price:.2f})"
    ),
    horizontal_spacing=0.1
)

# GRAPHIQUE 1 : Nuages vs Solaire (Scatter)
# On prend un échantillon de 2000 points pour ne pas alourdir le graphique
sample_day = df_day.sample(n=700, random_state=42)

fig.add_trace(go.Scatter(
    x=sample_day['cloud_cover_denmark'], 
    y=sample_day['DK_Total_Solar'],
    mode='markers',
    name='Solaire',
    marker=dict(size=3, color='orange', opacity=0.5)
), row=1, col=1)

# GRAPHIQUE 2 : Vent vs Production (Scatter)
sample_all = df.sample(n=800, random_state=42)

fig.add_trace(go.Scatter(
    x=sample_all['wind_speed_denmark'], 
    y=sample_all['DK_Total_Wind'],
    mode='markers',
    name='Éolien',
    marker=dict(size=3, color='blue', opacity=0.5)
), row=1, col=2)

# grahpique 3 : Production Vent vs PRIX (Scatter)
sample_all = df.sample(n=800, random_state=42)
fig.add_trace(go.Scatter(
    x=sample_all['DK_Total_Wind'], 
    y=sample_all['DK_1_price_day_ahead'],
    mode='markers',
    name='Prix Spot',
    marker=dict(size=3, color='green', opacity=0.5)
), row=1, col=3)

# Mise en forme
fig.update_xaxes(title_text="Couverture Nuageuse (%)", row=1, col=1)
fig.update_yaxes(title_text="Production Solaire (MW)", row=1, col=1)

fig.update_xaxes(title_text="Vitesse du Vent (m/s)", row=1, col=2)
fig.update_yaxes(title_text="Production Éolienne (MW)", row=1, col=2)

fig.update_xaxes(title_text="Production Vent (MW)", row=1, col=3)
fig.update_yaxes(title_text="Prix de l'Électricité (€/MWh)", row=1, col=3)

fig.update_layout(
    title_text="Coorélation : Météo -> Production -> Prix",
    height=500,
    showlegend=True,
    template="plotly_white"
)

fig.show()

# %% [markdown]
# * Météo → solaire
# Le nuage de points « Solaire vs Nuages » montre une corrélation négative modérée (−0,30) : plus la couverture nuageuse augmente, plus la production solaire tend à diminuer.​
# La dispersion importante indique que d’autres facteurs (saison, angle du soleil, température) jouent aussi un rôle, mais la tendance globale reste à la baisse avec les nuages.
# 
# * Vent → éolien
# Le graphe « Éolien vs Vitesse de vent » présente une forte corrélation positive (0,85) : quand la vitesse du vent augmente, la production éolienne croît presque linéairement jusqu’aux vitesses proches du nominal des turbines.​
# Cela illustre le fait que l’énergie éolienne est très sensible aux variations de vent, ce qui en fait le moteur principal de la variabilité de production au Danemark.​
# 
# * Éolien → prix
# « Impact Prix vs Vent » montre une corrélation négative (−0,26) entre la production éolienne et le prix spot : plus il y a de vent, plus les prix ont tendance à baisser.​
# 

# %% [markdown]
# ## Analyse de la Ressource Éolienne (Proxy du Mix)
# Le Danemark dépend du vent. Voici son cycle annuel.

# %%
#   Évolution de la Ressource Vent (Proxy Production) 
df_wind_monthly = df.groupby('Month')['wind_speed_denmark'].agg(['mean', 'std']).reset_index()

fig_wind = go.Figure()

# Moyenne du vent
fig_wind.add_trace(go.Scatter(
    x=df_wind_monthly['Month'], 
    y=df_wind_monthly['mean'],
    mode='lines+markers',
    name='Vitesse Moyenne (m/s)',
    line=dict(color='green', width=3)
))

# Zone de variabilité (Écart-type)
fig_wind.add_trace(go.Scatter(
    x=df_wind_monthly['Month'], y=df_wind_monthly['mean'] + df_wind_monthly['std'],
    mode='lines', line=dict(width=0), showlegend=False
))
fig_wind.add_trace(go.Scatter(
    x=df_wind_monthly['Month'], y=df_wind_monthly['mean'] - df_wind_monthly['std'],
    mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)',
    name='Variabilité (Std)'
))

fig_wind.update_layout(
    title="Cycle Annuel du Vent (Moteur du Mix Danois)",
    xaxis=dict(tickmode='array', tickvals=list(range(1,13)), ticktext=['Jan','Fev','Mar','Avr','Mai','Juin','Juil','Aou','Sep','Oct','Nov','Dec']),
    yaxis_title="Vitesse du Vent (m/s)",
    template="plotly_white"
)
fig_wind.show()

# %% [markdown]
# * On voit clairement que **l'Hiver est la saison de la production maximale** (vent fort).
# * L'été, le vent tombe, obligeant à importer ou utiliser la biomasse.

# %% [markdown]
# ## Analyse de Sensibilité : Relation Vent vs Prix de Marché

# %%

#  PRÉPARATION DES DONNÉES 
# On arrondit la vitesse du vent pour créer des "catégories" (ex: 4.2 m/s devient 4 m/s)
# Cela permet de regrouper les données pour le graphique
df['Wind_Round'] = df['wind_speed_denmark'].round(0)

#  On filtre les valeurs extrêmes de vent (très rares au-dessus de 20 m/s) pour garder le graphe lisible
df_wind_clean = df[df['Wind_Round'] <= 20]

#  CRÉATION DU GRAPHIQUE (BOXPLOT) 
fig_impact = px.box(
    df_wind_clean, 
    x="Wind_Round", 
    y="DK_1_price_day_ahead",
    color_discrete_sequence=['#E31A1C'], 
    title="Analyse de Sensibilité : Relation Vent vs Prix de Marché",
    template="plotly_white"
)

#  AJOUT DE LA LIGNE DE TENDANCE MOYENNE 
# On calcule la moyenne pour utiliser la ligne  par dessus les boîtes
df_trend = df_wind_clean.groupby('Wind_Round')['DK_1_price_day_ahead'].mean().reset_index()

fig_impact.add_trace(go.Scatter(
    x=df_trend['Wind_Round'], 
    y=df_trend['DK_1_price_day_ahead'],
    mode='lines',
    name='Prix Moyen',
    line=dict(color='green', width=3)
))

# Mise en forme
fig_impact.update_layout(
    xaxis_title="Vitesse du Vent (m/s)",
    yaxis_title="Prix de l'Électricité (€/MWh)",
    xaxis=dict(tickmode='linear', dtick=2), # Un trait tous les 2 m/s
    showlegend=True
)

# On fixe l'échelle Y pour bien voir les prix négatifs (si besoin)
fig_impact.update_yaxes(range=[-100, 300])

fig_impact.show()

# %% [markdown]
# Au‑delà de 11–12 m/s, la courbe se « tasse » puis remonte légèrement, ce qui reflète le fait que les turbines approchent de leur puissance nominale et que, à très haute vitesse, certaines peuvent se mettre en sécurité ou être contraintes par le réseau, ce qui limite l’effet de baisse supplémentaire sur les prix.​
# 
# Globalement, la sensibilité prix‑vent est clairement négative sur la plage principale de fonctionnement : le vent est un driver majeur de la baisse des prix et de l’apparition d’épisodes de prix très bas, voire négatifs, dans le marché danois.​

# %%
def somme_flexible(df, mots_cles_energie):
    marqueurs_pays = ['dk', 'denmark']

    
    # Etape A : On cherche d'abord avec le filtre Pays (le plus sûr)
    cols_pays = [c for c in df.columns if any(tag in c.lower() for tag in marqueurs_pays)]
    cols_finales = [c for c in cols_pays if any(mot in c.lower() for mot in mots_cles_energie)]
    
    if not cols_finales:
        cols_finales = [c for c in df.columns if any(mot in c.lower() for mot in mots_cles_energie)]
        # On exclut les colonnes parlant d'autres pays (FR, DE, SE, NO)
        exclus = ['fr', 'de', 'se', 'no', 'france', 'germany', 'sweden', 'norway']
        cols_finales = [c for c in cols_finales if not any(ex in c.lower() for ex in exclus)]

    if cols_finales:
        # print(f"   Trouvé pour {mots_cles_energie[0]} : {cols_finales}") # Décommenter pour debug
        return df[cols_finales].sum().sum()
    return 0


config_mix = {
    'Éolien (Wind)':     ['wind', 'eolien'],
    'Solaire (Solar)':   ['solar', 'solaire', 'pv', 'sun'],
    'Biomasse':          ['biomass', 'bio'],
    'Fossile (Gaz/Charbon)': ['gas', 'coal', 'oil', 'fossil', 'charbon', 'lignite'],
    'Hydro':             ['hydro', 'water'],
    'Nucléaire':         ['nuclear', 'nucleaire']
}

#  Calcul 
data_mix = []

for nom_energie, mots_cles in config_mix.items():
    total = somme_flexible(df, mots_cles)
    if total > 0: 
        data_mix.append({'Source': nom_energie, 'Production': total})

df_mix = pd.DataFrame(data_mix)

#  Affichage 
if df_mix.empty:
    print(" TOUJOURS RIEN. C'est anormal putttttttaaaaaaaiiiiiiiinnnnnn.")
    print(" Du coup il faut exécuter : print(list(df.columns))")
else:
    fig_mix = px.pie(
        df_mix, 
        values='Production', 
        names='Source', 
        title='<b>Mix Énergétique Danemark 2020-2025</b>',
        hole=0.4,
        template="plotly_white",
        color_discrete_map={
            'Éolien (Wind)': '#2ca02c', 
            'Solaire (Solar)': '#ff7f0e', 
            'Biomasse': '#8c564b', 
            'Fossile (Gaz/Charbon)': '#7f7f7f',
            'Hydro': '#1f77b4', 
            'Nucléaire': 'red' 
        }
    )

    # Taille + légende en bas
    fig_mix.update_layout(
        width=900,
        height=700,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,      
            xanchor="center",
            x=0.5
        )
    )

    fig_mix.update_traces(textposition='outside', textinfo='percent+label')
    fig_mix.show()


# %% [markdown]
# ## Modelisations

# %% [markdown]
# ## DK1

# %%
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %%
# ===== Préparation features DK1 =====
target_col = "DK_1_price_day_ahead"

data = df.copy().sort_index()

# Variables calendaires
data["hour"] = data.index.hour
data["dayofweek"] = data.index.dayofweek
data["month"] = data.index.month

# Lags de prix
data[f"{target_col}_lag1"] = data[target_col].shift(1)
data[f"{target_col}_lag24"] = data[target_col].shift(24)

# Moyenne mobile 24h
data[f"{target_col}_roll24"] = data[target_col].rolling(24).mean()

features = [
    "DK_1_load_actual_entsoe_transparency",
    "wind_speed_denmark",
    "temperature_denmark",
    "hour", "dayofweek", "month",
    f"{target_col}_lag1",
    f"{target_col}_lag24",
    f"{target_col}_roll24"
]

data = data.dropna(subset=[target_col] + features)

X = data[features]
y = data[target_col]

split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# %%
# ===== LightGBM baseline =====
print("LightGBM baseline...")

lgb_baseline = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
lgb_baseline.fit(X_train, y_train)
y_pred_base = lgb_baseline.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print(f"Baseline -> RMSE={rmse_base:.2f} | MAE={mae_base:.2f} | R²={r2_base:.3f}")

# ===== LightGBM optimisé =====
print("\nLightGBM optimisé (GridSearch)...")

param_grid = {
    "num_leaves": [31, 50, 70],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [-1, 10, 20],
    "n_estimators": [300, 600]
}

tscv = TimeSeriesSplit(n_splits=3)

lgb_opt = lgb.LGBMRegressor(random_state=42, verbose=-1)

grid_search = GridSearchCV(
    estimator=lgb_opt,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_lgb = grid_search.best_estimator_
print("Meilleurs hyperparamètres :", grid_search.best_params_)

y_pred_opt = best_lgb.predict(X_test)

rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
mae_opt = mean_absolute_error(y_test, y_pred_opt)
r2_opt = r2_score(y_test, y_pred_opt)

print(f"Optimisé -> RMSE={rmse_opt:.2f} | MAE={mae_opt:.2f} | R²={r2_opt:.3f}")


# %%

# SARIMAX DK1 sur les 2 derniers mois (avec conso en exogène)


target_col = "DK_1_price_day_ahead"
exog_vars = ["DK_1_load_actual_entsoe_transparency"]

# 1) Fenêtre : 2 derniers mois
end_date = df.index.max()
start_date = end_date - pd.DateOffset(months=2)

df_win = df.loc[start_date:end_date, [target_col] + exog_vars].copy()
print("Taille brute 2 mois :", df_win.shape)

# 2) Nettoyage NaN / inf
df_win = df_win.replace([np.inf, -np.inf], np.nan)
df_win = df_win.interpolate().dropna()
print("Après nettoyage/interpolation :", df_win.shape)

if len(df_win) < 100:
    raise ValueError("Trop peu d'observations propres sur les 2 derniers mois pour SARIMAX.")

y_series = df_win[target_col]
exog = df_win[exog_vars]

# 3) Split 80/20 temporel
split_idx = int(len(df_win) * 0.8)
y_train_sarimax, y_test_sarimax = y_series.iloc[:split_idx], y_series.iloc[split_idx:]
exog_train, exog_test = exog.iloc[:split_idx], exog.iloc[split_idx:]

print("Train size :", len(y_train_sarimax), " | Test size :", len(y_test_sarimax))

# 4) Paramètres SARIMAX
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 24)

print("Entraînement SARIMAX DK1 (2 derniers mois, avec conso en exogène)...")

model = sm.tsa.statespace.SARIMAX(
    y_train_sarimax,
    order=order,
    seasonal_order=seasonal_order,
    exog=exog_train,
    enforce_stationarity=False,
    enforce_invertibility=False
)
res = model.fit(disp=False)

# 5) Prévisions
y_pred_sarimax = res.predict(
    start=y_test_sarimax.index[0],
    end=y_test_sarimax.index[-1],
    exog=exog_test
)

# 6) Métriques
rmse = np.sqrt(mean_squared_error(y_test_sarimax, y_pred_sarimax))
mae = mean_absolute_error(y_test_sarimax, y_pred_sarimax)
r2 = r2_score(y_test_sarimax, y_pred_sarimax)

print(f"SARIMAX DK1 (2 derniers mois) -> RMSE = {rmse:.2f} €/MWh | "
      f"MAE = {mae:.2f} €/MWh | R² = {r2:.3f}")


# %%
# Résumé des modèles DK1
print("\nRésumé des performances DK1 :")
results_dk1 = pd.DataFrame([
    {"Modèle": "LightGBM baseline", "RMSE": rmse_base, "MAE": mae_base, "R²": r2_base},
    {"Modèle": "LightGBM optimisé", "RMSE": rmse_opt, "MAE": mae_opt, "R²": r2_opt},
    {"Modèle": "SARIMAX (2 derniers mois)", "RMSE": rmse_sarimax, "MAE": mae_sarimax, "R²": r2_sarimax},
])

display(results_dk1)


# %%
# 3. Visualisation Interactive Comparée 
print("\nGénération du Graphique")
fig = go.Figure()

# 1) Série réelle
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test.values,
    name="Réel",
    line=dict(color="red", width=2)
))

# 2) LightGBM baseline
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_pred_base,
    name=f"LGBM Base (RMSE: {rmse_base:.2f})",
    line=dict(color="royalblue", dash="dot", width=1.5)
))

# 3) LightGBM optimisé
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_pred_opt,
    name=f"LGBM Opti (RMSE: {rmse_opt:.2f})",
    line=dict(color="green", width=1.5)
))

# 4) SARIMAX / SARIMAX
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_pred_sarimax,   # ou y_pred_sarima selon ton nom de variable
    name=f"SARIMAX (RMSE: {rmse_sarimax:.2f})",
    line=dict(color="orange", width=1.5, dash="dash")
))

fig.update_layout(
    title="LGBM Base vs LGBM Optimisé vs SARIMAX",
    xaxis_title="Date",
    yaxis_title="Prix (€/MWh)",
    template="plotly_white",
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.01
    )
)

# Range slider + zoom initial sur 1 semaine
fig.update_xaxes(rangeslider_visible=True)
if len(y_test) > 168:
    fig.update_xaxes(range=[y_test.index[0], y_test.index[168]])

fig.show()


# %% [markdown]
# ## DK2

# %%
# ===== Préparation features DK2 =====
target_col = "DK_2_price_day_ahead"

data = df.copy().sort_index()

# Variables calendaires
data["hour"] = data.index.hour
data["dayofweek"] = data.index.dayofweek
data["month"] = data.index.month

# Lags de prix
data[f"{target_col}_lag1"] = data[target_col].shift(1)
data[f"{target_col}_lag24"] = data[target_col].shift(24)

# Moyenne mobile 24h
data[f"{target_col}_roll24"] = data[target_col].rolling(24).mean()

features = [
    "DK_2_load_actual_entsoe_transparency",
    "wind_speed_denmark",
    "temperature_denmark",
    "hour", "dayofweek", "month",
    f"{target_col}_lag1",
    f"{target_col}_lag24",
    f"{target_col}_roll24"
]

data = data.dropna(subset=[target_col] + features)

X = data[features]
y = data[target_col]

split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# %%
# ===== LightGBM baseline =====
print("LightGBM baseline...")

lgb_baseline = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
lgb_baseline.fit(X_train, y_train)
y_pred_base = lgb_baseline.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print(f"Baseline -> RMSE={rmse_base:.2f} | MAE={mae_base:.2f} | R²={r2_base:.3f}")

# ===== LightGBM optimisé =====
print("\nLightGBM optimisé (GridSearch)...")

param_grid = {
    "num_leaves": [31, 50, 70],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [-1, 10, 20],
    "n_estimators": [300, 600]
}

tscv = TimeSeriesSplit(n_splits=3)

lgb_opt = lgb.LGBMRegressor(random_state=42, verbose=-1)

grid_search = GridSearchCV(
    estimator=lgb_opt,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_lgb = grid_search.best_estimator_
print("Meilleurs hyperparamètres :", grid_search.best_params_)

y_pred_opt = best_lgb.predict(X_test)

rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
mae_opt = mean_absolute_error(y_test, y_pred_opt)
r2_opt = r2_score(y_test, y_pred_opt)

print(f"Optimisé -> RMSE={rmse_opt:.2f} | MAE={mae_opt:.2f} | R²={r2_opt:.3f}")


# %%

# SARIMAX DK2 sur les 2 derniers mois (avec conso en exogène)


target_col = "DK_2_price_day_ahead"
exog_vars = ["DK_2_load_actual_entsoe_transparency"]

# 1) Fenêtre : 2 derniers mois
end_date = df.index.max()
start_date = end_date - pd.DateOffset(months=2)

df_win = df.loc[start_date:end_date, [target_col] + exog_vars].copy()
print("Taille brute 2 mois :", df_win.shape)

# 2) Nettoyage NaN / inf
df_win = df_win.replace([np.inf, -np.inf], np.nan)
df_win = df_win.interpolate().dropna()
print("Après nettoyage/interpolation :", df_win.shape)

if len(df_win) < 100:
    raise ValueError("Trop peu d'observations propres sur les 2 derniers mois pour SARIMAX.")

y_series = df_win[target_col]
exog = df_win[exog_vars]

# 3) Split 80/20 temporel
split_idx = int(len(df_win) * 0.8)
y_train_sarimax, y_test_sarimax = y_series.iloc[:split_idx], y_series.iloc[split_idx:]
exog_train, exog_test = exog.iloc[:split_idx], exog.iloc[split_idx:]

print("Train size :", len(y_train_sarimax), " | Test size :", len(y_test_sarimax))

# 4) Paramètres SARIMAX
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 24)

print("Entraînement SARIMAX DK2 (2 derniers mois, avec conso en exogène)...")

model = sm.tsa.statespace.SARIMAX(
    y_train_sarimax,
    order=order,
    seasonal_order=seasonal_order,
    exog=exog_train,
    enforce_stationarity=False,
    enforce_invertibility=False
)
res = model.fit(disp=False)

# 5) Prévisions
y_pred_sarimax = res.predict(
    start=y_test_sarimax.index[0],
    end=y_test_sarimax.index[-1],
    exog=exog_test
)

# 6) Métriques
rmse = np.sqrt(mean_squared_error(y_test_sarimax, y_pred_sarimax))
mae = mean_absolute_error(y_test_sarimax, y_pred_sarimax)
r2 = r2_score(y_test_sarimax, y_pred_sarimax)

print(f"SARIMAX DK2 (2 derniers mois) -> RMSE = {rmse:.2f} €/MWh | "
      f"MAE = {mae:.2f} €/MWh | R² = {r2:.3f}")


# %%
# Résumé des modèles DK2
print("\nRésumé des performances DK2 :")
results_dk2 = pd.DataFrame([
    {"Modèle": "LightGBM baseline", "RMSE": rmse_base, "MAE": mae_base, "R²": r2_base},
    {"Modèle": "LightGBM optimisé", "RMSE": rmse_opt, "MAE": mae_opt, "R²": r2_opt},
    {"Modèle": "SARIMAX (2 derniers mois)", "RMSE": rmse_sarimax, "MAE": mae_sarimax, "R²": r2_sarimax},
])

display(results_dk2)


# %%
# 3. Visualisation Interactive Comparée 
print("\nGénération du Graphique")
fig = go.Figure()

# 1) Série réelle
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test.values,
    name="Réel",
    line=dict(color="black", width=2)
))

# 2) LightGBM baseline
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_pred_base,
    name=f"LGBM Base (RMSE: {rmse_base:.2f})",
    line=dict(color="royalblue", dash="dot", width=1.5)
))

# 3) LightGBM optimisé
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_pred_opt,
    name=f"LGBM Opti (RMSE: {rmse_opt:.2f})",
    line=dict(color="green", width=1.5)
))

# 4) SARIMAX / SARIMAX
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_pred_sarimax,   # ou y_pred_sarima selon ton nom de variable
    name=f"SARIMAX (RMSE: {rmse_sarimax:.2f})",
    line=dict(color="orange", width=1.5, dash="dash")
))

fig.update_layout(
    title="LGBM Base vs LGBM Optimisé vs SARIMAX",
    xaxis_title="Date",
    yaxis_title="Prix (€/MWh)",
    template="plotly_white",
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.01
    )
)

# Range slider + zoom initial sur 1 semaine
fig.update_xaxes(rangeslider_visible=True)
if len(y_test) > 168:
    fig.update_xaxes(range=[y_test.index[0], y_test.index[168]])

fig.show()
fig_season = make_subplots(specs=[[{"secondary_y": True}]])


