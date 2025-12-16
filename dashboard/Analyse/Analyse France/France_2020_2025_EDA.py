#!/usr/bin/env python
# coding: utf-8

# # Prix de l'Électricité France (2020-2025)

# In[38]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ## Chargement et Préparation des Données

# In[39]:


# Chargement
df_raw = pd.read_csv('../data/raw/time_series_60min_fr_dk_2020_2025.csv', 
                    parse_dates=['utc_timestamp'], 
                    index_col='utc_timestamp')
df_raw.head()


# * **France Dataset**

# In[40]:


# Sélection colonnes France
cols_france = [
    'FR_Biomass_generation_actual', 'FR_Energy_storage_generation_actual',
    'FR_Fossil_Gas_generation_actual', 'FR_Fossil_Hard_coal_generation_actual',
    'FR_Fossil_Oil_generation_actual', 'FR_Hydro_Pumped_Storage_generation_actual',
    'FR_Hydro_Run-of-river_and_poundage_generation_actual',
    'FR_Hydro_Water_Reservoir_generation_actual', 'FR_Nuclear_generation_actual',
    'FR_Solar_generation_actual', 'FR_Waste_generation_actual',
    'FR_Wind_Offshore_generation_actual', 'FR_Wind_Onshore_generation_actual',
    'FR_price_day_ahead', 'temperature_france', 'cloud_cover_france',
    'wind_speed_france', 'FR_load_actual_entsoe_transparency',
    'FR_load_forecast_entsoe_transparency'
]

available_cols = [c for c in cols_france if c in df_raw.columns]
df = df_raw[available_cols].copy()

# In[41]:


# Renommage
rename_dict = {
    'FR_load_actual_entsoe_transparency': 'load',
    'FR_load_forecast_entsoe_transparency': 'load_forecast',
    'FR_Solar_generation_actual': 'solar',
    'FR_Wind_Onshore_generation_actual': 'wind_onshore',
    'FR_Wind_Offshore_generation_actual': 'wind_offshore',
    'FR_Nuclear_generation_actual': 'nuclear',
    'FR_Hydro_Pumped_Storage_generation_actual': 'hydro_pumped',
    'FR_Hydro_Run-of-river_and_poundage_generation_actual': 'hydro_river',
    'FR_Hydro_Water_Reservoir_generation_actual': 'hydro_reservoir',
    'FR_Fossil_Gas_generation_actual': 'gas',
    'FR_Fossil_Hard_coal_generation_actual': 'coal',
    'FR_Fossil_Oil_generation_actual': 'oil',
    'FR_Biomass_generation_actual': 'biomass',
    'FR_Waste_generation_actual': 'waste',
    'FR_Energy_storage_generation_actual': 'storage',
    'FR_price_day_ahead': 'price_day_ahead',
    'temperature_france': 'temperature',
    'cloud_cover_france': 'cloud_cover',
    'wind_speed_france': 'wind_speed'
}
df = df.rename(columns=rename_dict)
df.head()

# In[42]:


# Agrégations
if 'wind_onshore' in df.columns and 'wind_offshore' in df.columns:
    df['wind'] = df['wind_onshore'].fillna(0) + df['wind_offshore'].fillna(0)
elif 'wind_onshore' in df.columns:
    df['wind'] = df['wind_onshore']

hydro_cols = ['hydro_pumped', 'hydro_river', 'hydro_reservoir']
available_hydro = [c for c in hydro_cols if c in df.columns]
if available_hydro:
    df['hydro'] = df[available_hydro].fillna(0).sum(axis=1)

print(f"Colonnes chargées: {len(df.columns)}")

# * **Gestion doublons et valeurs manquantes**

# In[43]:


# Gestion des Doublons
duplicates = df.index.duplicated().sum()
if duplicates > 0:
    df = df[~df.index.duplicated(keep='first')]
    print(f"Doublons supprimés: {duplicates}")


# In[44]:


# Gestion des Valeurs Manquantes
print("\n--- Analyse des valeurs manquantes ---")
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

if len(missing_pct) > 0:
    print(f"Colonnes avec valeurs manquantes: {len(missing_pct)}")
    for col, pct in missing_pct.items():
        print(f"  {col}: {pct:.1f}%")
    
    # Supprimer colonnes >50%
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    if cols_to_drop:
        print(f"\n Suppression de {len(cols_to_drop)} colonnes (>50% de valeurs manquantes):")
        for col in cols_to_drop:
            print(f"  - {col}")
        df = df.drop(columns=cols_to_drop)
    
    # Imputation
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"\n Imputation pour {remaining_missing:,} valeurs...")
        df = df.interpolate(method='time', limit_direction='both')
        df = df.fillna(method='ffill').fillna(method='bfill')
        print(" Imputation terminée")

final_missing = df.isnull().sum().sum()
if final_missing > 0:
    df = df.dropna()

print(f"\nDataset final: {df.shape}")

# ## Statistiques Descriptives

# In[45]:


display(df.describe())

# ## Détection et Traitement des Outliers (Price_day_ahead)

# In[46]:


# Calculate IQR for outlier detection
Q1 = df['price_day_ahead'].quantile(0.25)
Q3 = df['price_day_ahead'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create DataFrame without outliers
df_no_outliers = df[(df['price_day_ahead'] >= lower_bound) & (df['price_day_ahead'] <= upper_bound)]

fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Avec Outliers</b>", "<b>Sans Outliers</b>"))

# Graph with outliers
fig.add_trace(
    go.Scatter(x=df.index, y=df['price_day_ahead'], line=dict(color='#1976D2', width=1), name='Prix'),
    row=1, col=1
)

# Graph without outliers
fig.add_trace(
    go.Scatter(x=df_no_outliers.index, y=df_no_outliers['price_day_ahead'], line=dict(color='#28A745', width=1), name='Prix (sans outliers)'),
    row=1, col=2
)

fig.update_layout(
    title="<b>Évolution Temporelle du Prix avec et sans Outliers</b>",
    height=600,
    font=dict(size=12),
    plot_bgcolor='white',
    showlegend=False
)

fig.update_xaxes(title_text='Date', row=1, col=1)
fig.update_yaxes(title_text='Prix (€/MWh)', row=1, col=1)
fig.update_xaxes(title_text='Date', row=1, col=2)
fig.update_yaxes(title_text='Prix (€/MWh)', row=1, col=2)

fig.show()

# In[47]:


print("Détection des Outliers (IQR)\n")
Q1 = df['price_day_ahead'].quantile(0.25)
Q3 = df['price_day_ahead'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Bornes: [{lower_bound:.2f}, {upper_bound:.2f}] €/MWh")

outliers = df[(df['price_day_ahead'] < lower_bound) | (df['price_day_ahead'] > upper_bound)]
print(f"Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")


# In[48]:


# Visualisation
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['price_day_ahead'], mode='lines', line=dict(color='#1976D2', width=1)))
fig.add_trace(go.Scatter(x=outliers.index, y=outliers['price_day_ahead'], mode='markers',
                         marker=dict(color='#D32F2F', size=5, symbol='x'), name='Outliers'))
fig.add_hline(y=lower_bound, line_dash="dash", line_color="orange")
fig.add_hline(y=upper_bound, line_dash="dash", line_color="orange")
fig.update_layout(title="<b>Détection des Outliers</b>", height=600, font=dict(size=14))
fig.show()

# In[49]:


# Traitement (Winsorization)
print("\n Traitement (Winsorization) ")
df['price_raw'] = df['price_day_ahead'].copy()
df['price_day_ahead'] = df['price_day_ahead'].clip(lower=lower_bound, upper=upper_bound)
print(f"Valeurs modifiées: {(df['price_day_ahead'] != df['price_raw']).sum():,}")


# In[50]:


# Comparaison
fig = make_subplots(rows=1, cols=2, subplot_titles=('Avant', 'Après'))
fig.add_trace(go.Histogram(x=df['price_raw'], nbinsx=100, marker_color='#1976D2'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['price_day_ahead'], nbinsx=100, marker_color='#2E7D32'), row=1, col=2)
fig.update_layout(title="<b>Distribution Avant/Après</b>", height=500, showlegend=False, font=dict(size=14))
fig.show()

# ## EDA

# * **Distribution du Prix**

# In[51]:


fig = make_subplots(rows=1, cols=2, subplot_titles=('Distribution', 'Boxplot'))
fig.add_trace(go.Histogram(x=df['price_day_ahead'], nbinsx=50), row=1, col=1)
fig.add_trace(go.Box(y=df['price_day_ahead']), row=1, col=2)
fig.update_xaxes(title_text='Prix (€/MWh)', row=1, col=1)
fig.update_yaxes(title_text='fréquence', row=1, col=1)
fig.update_layout(height=500, title_text="<b>Distribution des Prix</b>", showlegend=False, font=dict(size=14))
fig.show()

# * **Évolution Temporelle**

# In[52]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['price_day_ahead'], line=dict(color='#1976D2', width=1)))
fig.update_layout(
    title="<b>Évolution Temporelle du Prix</b>",
    xaxis_title='Date', yaxis_title='Prix (€/MWh)',
    height=600, font=dict(size=12), plot_bgcolor='white'
)
fig.show()

# #### Évolution annuelle du prix

# In[53]:


# Évolution annuelle du prix
df['year'] = df.index.year
annual_price = df.groupby('year')['price_day_ahead'].mean().reset_index()

# Calcul du pourcentage d'augmentation par rapport à l'année précédente
annual_price['pct_change'] = annual_price['price_day_ahead'].pct_change() * 100

# Création du graphique
fig = go.Figure()

# Barres pour le prix moyen
fig.add_trace(go.Bar(
    x=annual_price['year'],
    y=annual_price['price_day_ahead'],
    name='Prix Moyen',
    marker_color='#1f77b4',
    text=annual_price['price_day_ahead'].round(2), # Affiche le prix sur la barre
    textposition='auto'
))

# Ajout des annotations de pourcentage
# On commence à l'index 1 car la première année n'a pas de variation
for i in range(1, len(annual_price)):
    change = annual_price.loc[i, 'pct_change']
    year = annual_price.loc[i, 'year']
    price = annual_price.loc[i, 'price_day_ahead']
    
    # Couleur : Rouge si augmentation, Vert si baisse
    color = "red" if change > 0 else "green"
    symbol = "▲" if change > 0 else "▼"
    
    fig.add_annotation(
        x=year,
        y=price + 5, # Un peu au-dessus de la barre
        text=f"{symbol} {abs(change):.1f}%",
        showarrow=False,
        font=dict(color=color, size=12, family="Arial Black")
    )

fig.update_layout(
    title="<b>Évolution du Prix Moyen Annuel (% Variation)</b>",
    xaxis_title='Année',
    yaxis_title='Prix Moyen (€/MWh)',
    height=500,
    plot_bgcolor='white',
    xaxis=dict(tickmode='linear'), # Affiche toutes les années
    showlegend=False
)

fig.show()

# Affichage tableau
print("Variation annuelle du prix :")
print(annual_price[['year', 'price_day_ahead', 'pct_change']].to_string(index=False))

# * **Prix vs Load** 

# In[54]:


# On découpe la charge en 20 tranches pour voir la tendance
df['load_bin'] = pd.cut(df['load'], bins=20)
# Calcul du prix moyen par tranche
df_trend = df.groupby('load_bin')['price_day_ahead'].mean().reset_index()
# Pour l'affichage, on prend le milieu de l'intervalle
df_trend['load_center'] = df_trend['load_bin'].apply(lambda x: x.mid).astype(int)

fig = go.Figure()

# Barplot (Tendance moyenne)
fig.add_trace(go.Bar(
    x=df_trend['load_center'],
    y=df_trend['price_day_ahead'],
    name='Prix Moyen',
    marker_color='indianred'
))

fig.update_layout(
    title="Tendance : Prix Moyen par Niveau de Consommation",
    xaxis_title='Consommation (MW)',
    yaxis_title='Prix Moyen (€/MWh)',
    height=600,
    font=dict(size=14),
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray')
)
fig.show()


# * **Prix vs production nucléaire**

# In[55]:


#   Prix moyen par tranche de production nucléaire ---
# Création de tranches de 2000 MW
df['nuclear_bin'] = (df['nuclear'] // 2000 * 2000).astype(int)
# Calcul du prix moyen par tranche
df_bar = df.groupby('nuclear_bin')['price_day_ahead'].mean().reset_index()

fig_bar = px.bar(
    df_bar, 
    x='nuclear_bin', 
    y='price_day_ahead',
    title="<b> Prix Moyen Vs production nucléaire</b>",
    labels={'nuclear_bin': 'Production Nucléaire (MW)', 'price_day_ahead': 'Prix Moyen (€)'}
)
fig_bar.show()


# **Saisonnalités**

# In[56]:


# Ajout colonnes temporelles
df['year'] = df.index.year
df['month'] = df.index.month
df['day_name'] = df.index.day_name()
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = df.index.dayofweek >= 5
def get_season(m): 
    return 'Hiver' if m in [12,1,2] else 'Printemps' if m in [3,4,5] else 'Eté' if m in [6,7,8] else 'Automne'
df['season_lbl'] = df['month'].apply(get_season)
# Jours fériés français (approximation - principaux jours fixes)
french_holidays = [
    (1, 1),   # Nouvel An
    (5, 1),   # Fête du Travail
    (5, 8),   # Victoire 1945
    (7, 14),  # Fête Nationale
    (8, 15),  # Assomption
    (11, 1),  # Toussaint
    (11, 11), # Armistice
    (12, 25), # Noël
]
df['is_holiday'] = df.index.to_series().apply(lambda x: (x.month, x.day) in french_holidays)


# In[57]:


# Prix moyen annuel
annual = df.groupby('year')['price_day_ahead'].mean().reset_index()
fig = go.Figure()
fig.add_trace(go.Bar(x=annual['year'], y=annual['price_day_ahead'], marker_color='#1976D2'))
fig.update_layout(title="<b>Prix Moyen Annuel</b>", height=500, font=dict(size=14))
fig.show()

# In[58]:


# Prix moyen mensuel
monthly = df.groupby('month')['price_day_ahead'].mean().reset_index()
month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
fig = go.Figure()
fig.add_trace(go.Bar(x=month_names, y=monthly['price_day_ahead'], marker_color='#2E7D32'))
fig.update_layout(title="<b>Prix Moyen Mensuel</b>", height=500, font=dict(size=14))
fig.show()

# In[59]:


#Saisonnalité par Saison
fig = px.box(df, x='season_lbl', y='price_day_ahead', 
             title='<b>Distribution des Prix par Saison</b>',
             category_orders={'season_lbl': ['Hiver', 'Printemps', 'Eté', 'Automne']},
             color='season_lbl',
             color_discrete_map={'Hiver': '#1E88E5', 'Printemps': '#43A047', 'Eté': '#FDD835', 'Automne': '#FB8C00'})
fig.update_layout(height=500, font=dict(size=14), showlegend=False)
fig.update_xaxes(title_text="Saison")
fig.update_yaxes(title_text="Prix (€/MWh)")
fig.show()


# In[60]:


#Saisonnalité par Saison
fig = px.box(df, x='season_lbl', y='load', 
             title='<b>Distribution des Consommations par Saison</b>',
             category_orders={'season_lbl': ['Hiver', 'Printemps', 'Eté', 'Automne']},
             color='season_lbl',
             color_discrete_map={'Hiver': '#1E88E5', 'Printemps': '#43A047', 'Eté': '#FDD835', 'Automne': '#FB8C00'})
fig.update_layout(height=500, font=dict(size=14), showlegend=False)
fig.update_xaxes(title_text="Saison")
fig.update_yaxes(title_text="Consommation (MW)")
fig.show()


# In[61]:


#Saisonnalité Hebdomadaire (Jour de la semaine)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_names_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
weekly = df.groupby('day_name')['price_day_ahead'].mean().reindex(day_order).reset_index()
weekly['day_name_fr'] = day_names_fr
fig = go.Figure()
fig.add_trace(go.Bar(x=weekly['day_name_fr'], y=weekly['price_day_ahead'], marker_color='#D32F2F'))
fig.update_layout(
    title=dict(text="<b>Prix Moyen par Jour de la Semaine</b>", font=dict(size=20)),
    xaxis_title='Jour',
    yaxis_title='Prix Moyen (€/MWh)',
    height=500,
    font=dict(size=14),
    plot_bgcolor='white'
)
fig.show()


# In[62]:


#Weekend vs Semaine
weekend_comparison = df.groupby('is_weekend')['price_day_ahead'].mean()
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Semaine', 'Weekend'],
    y=[weekend_comparison[False], weekend_comparison[True]],
    marker_color=['#1976D2', '#FFA726']
))
fig.update_layout(
    title=dict(text="<b>Prix Moyen : Semaine vs Weekend</b>", font=dict(size=20)),
    yaxis_title='Prix Moyen (€/MWh)',
    height=500,
    font=dict(size=14),
    plot_bgcolor='white'
)
fig.show()


# In[63]:


#Jours Fériés vs Jours Normaux
holiday_comparison = df.groupby('is_holiday')['price_day_ahead'].mean()
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Jours Normaux', 'Jours Fériés'],
    y=[holiday_comparison[False], holiday_comparison[True]],
    marker_color=['#1976D2', '#E53935']
))
fig.update_layout(
    title=dict(text="<b>Prix Moyen : Jours Normaux vs Jours Fériés</b>", font=dict(size=20)),
    yaxis_title='Prix Moyen (€/MWh)',
    height=500,
    font=dict(size=14),
    plot_bgcolor='white'
)
fig.show()


# In[64]:


# Profil horaire
hourly_week = df[~df['is_weekend']].groupby('hour')['price_day_ahead'].mean()
hourly_weekend = df[df['is_weekend']].groupby('hour')['price_day_ahead'].mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=hourly_week.index, y=hourly_week, name='Semaine', line=dict(width=3)))
fig.add_trace(go.Scatter(x=hourly_weekend.index, y=hourly_weekend, name='Weekend', line=dict(width=3)))
fig.update_layout(title="<b>Profil Horaire</b>", height=600, font=dict(size=14))
fig.show()

# #### Mix Énergétique (Répartition des Sources)

# In[65]:


# Calculer la production totale par source d'énergie
print("Mix Énergétique France (2020-2025)\n")

# Identifier les colonnes de génération disponibles
generation_cols = {
    'nuclear': 'Nucléaire',
    'hydro': 'Hydraulique',
    'wind': 'Éolien',
    'solar': 'Solaire',
    'gas': 'Gaz',
    'coal': 'Charbon',
    'biomass': 'Biomasse',
    'oil': 'Fioul'
}

# Calculer la production totale pour chaque source
energy_mix = {}
for col, label in generation_cols.items():
    if col in df.columns:
        total = df[col].sum()
        energy_mix[label] = total
        print(f"{label}: {total:,.0f} MWh")

# Trier par ordre décroissant
energy_mix = dict(sorted(energy_mix.items(), key=lambda x: x[1], reverse=True))

# In[66]:


# Graphique en camembert
import plotly.express as px

fig = px.pie(
    values=list(energy_mix.values()),
    names=list(energy_mix.keys()),
    title='<b>Mix Énergétique France (2020-2025)</b><br>Répartition de la Production Totale',
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig.update_traces(
    textposition='inside',
    textinfo='percent+label',
    textfont_size=14,
    marker=dict(line=dict(color='white', width=2))
)

fig.update_layout(
    height=600,
    font=dict(size=14),
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.05,
        font=dict(size=12)
    )
)

fig.show()

# In[67]:


# Tableau récapitulatif
mix_df = pd.DataFrame({
    'Source': list(energy_mix.keys()),
    'Production (MWh)': list(energy_mix.values())
})

mix_df['Part (%)'] = (mix_df['Production (MWh)'] / mix_df['Production (MWh)'].sum() * 100).round(2)
mix_df['Production (TWh)'] = (mix_df['Production (MWh)'] / 1_000_000).round(2)

print("Tableau Récapitulatif")
display(mix_df)

print(f"\nProduction totale: {mix_df['Production (TWh)'].sum():.2f} TWh")

# In[68]:


# Évolution mensuelle du mix énergétique
df_monthly = df.copy()
df_monthly['year_month'] = df_monthly.index.to_period('M')

# Agréger par mois
monthly_mix = {}
for col, label in generation_cols.items():
    if col in df.columns:
        monthly_mix[label] = df_monthly.groupby('year_month')[col].sum()

# Création d'un DataFrame
monthly_df = pd.DataFrame(monthly_mix)
monthly_df.index = monthly_df.index.to_timestamp()


# #### Évolution Temporelle du Mix Énergétique

# In[69]:



# Graphique en aires empilées
fig = go.Figure()

colors = {
    'Nucléaire': '#FF6B6B',
    'Hydraulique': '#4ECDC4',
    'Éolien': '#95E1D3',
    'Solaire': '#FFD93D',
    'Gaz': '#F38181',
    'Charbon': '#6C5B7B',
    'Biomasse': '#C8E6C9',
    'Fioul': '#B39DDB'
}

for source in monthly_df.columns:
    fig.add_trace(go.Scatter(
        x=monthly_df.index,
        y=monthly_df[source],
        name=source,
        mode='lines',
        stackgroup='one',
        fillcolor=colors.get(source, '#CCCCCC'),
        line=dict(width=0.5, color=colors.get(source, '#CCCCCC'))
    ))

fig.update_layout(
    title='<b>Évolution Mensuelle du Mix Énergétique</b>',
    xaxis_title='Date',
    yaxis_title='Production (MWh)',
    height=600,
    font=dict(size=14),
    hovermode='x unified',
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray')
)

fig.show()

# ## Corrélations

# In[70]:


numeric_cols = df.select_dtypes(include=[np.number]).columns
cols_for_corr = [c for c in numeric_cols if c not in ['year', 'month', 'hour']]
corr_matrix = df[cols_for_corr].corr()

fig = px.imshow(corr_matrix, text_auto='.2f', title='<b>Heatmap de Corrélation</b>',
                color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
fig.update_layout(height=1000, width=1200, font=dict(size=12))
fig.show()

price_corr = corr_matrix['price_day_ahead'].drop('price_day_ahead').sort_values(ascending=False)
print("\nTop 10 Corrélations Positives:")
print(price_corr.head(10))
print("\nTop 10 Corrélations Négatives:")
print(price_corr.tail(10))

# #### Analyse des Corrélations (Load, Nucléaire, Prix, Fossiles, Hydro)

# In[71]:


# Sélection des variables pour la corrélation
corr_vars = ['load', 'price_day_ahead', 'nuclear', 'gas', 'coal', 'hydro', 'oil', 'biomass']
# Filtrer celles qui existent
corr_vars = [c for c in corr_vars if c in df.columns]

# Calcul de la matrice
corr_mx = df[corr_vars].corr()

# Heatmap
fig = px.imshow(
    corr_mx, 
    text_auto='.2f',
    aspect='auto',
    title='<b>Corrélation : Consommation, Production et Prix</b>',
    color_continuous_scale='RdBu_r', 
    zmin=-1, zmax=1
)
fig.update_layout(height=600, width=800, font=dict(size=12))
fig.show()


# ## Sauvegarde du Dataset Nettoyé

# In[72]:


# Sauvegarder pour le notebook suivant
df.to_csv('../data/processed/df_features_france_2020_2025.csv')
print(" Dataset sauvegardé: ../data/processed/df_features_france_2020_2025.csv")
print(f"Shape final: {df.shape}")
