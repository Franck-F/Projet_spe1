# Lexique exact fourni par l'utilisateur (SANS MODIFICATION)
lexique_features = {
    'price_day_ahead_lag_1': """Prix observé 1 heure avant l'instant courant, 
p
r
i
c
e
t
−
1
price 
t−1
 .
Capture la dépendance très court terme.​""",
    
    'price_day_ahead_lag_3': """Prix observé 3 heures avant, 
p
r
i
c
e
t
−
3
price 
t−3
 .
Représente l'inertie du prix à horizon intra‑quotidien.​""",

    'price_day_ahead_lag_6': """Prix observé 6 heures avant, 
p
r
i
c
e
t
−
6
price 
t−6
 .
Permet de capter les patterns sur un quart de journée.​""",

    'price_day_ahead_lag_12': """Prix observé 12 heures avant, 
p
r
i
c
e
t
−
12
price 
t−12
 .
Cible des effets matin/soir ou jour/nuit.​""",

    'price_day_ahead_lag_24': """Prix observé 24 heures avant, 
p
r
i
c
e
t
−
24
price 
t−24
 .
Représente la saisonnalité quotidienne (même heure la veille).​""",

    'price_day_ahead_lag_168': """Prix observé 168 heures avant, 
p
r
i
c
e
t
−
168
price 
t−168
  (7 jours).
Capture la saisonnalité hebdomadaire (même heure une semaine avant).​""",

    'rolling_mean_6': """Moyenne des prix sur les 6, 24 ou 168 dernières heures.
Représente le niveau moyen de prix à court terme (6h), journalier (24h) ou hebdomadaire (168h).​""",
    
    'rolling_mean_24': """Moyenne des prix sur les 6, 24 ou 168 dernières heures.
Représente le niveau moyen de prix à court terme (6h), journalier (24h) ou hebdomadaire (168h).​""",
    
    'rolling_mean_168': """Moyenne des prix sur les 6, 24 ou 168 dernières heures.
Représente le niveau moyen de prix à court terme (6h), journalier (24h) ou hebdomadaire (168h).​""",

    'rolling_std_6': """Écart‑type des prix sur les 6, 24 ou 168 dernières heures.
Mesure la volatilité récente à ces trois horizons.​""",
    
    'rolling_std_24': """Écart‑type des prix sur les 6, 24 ou 168 dernières heures.
Mesure la volatilité récente à ces trois horizons.​""",
    
    'rolling_std_168': """Écart‑type des prix sur les 6, 24 ou 168 dernières heures.
Mesure la volatilité récente à ces trois horizons.​""",

    'rolling_min_6': """Minimum des prix observés sur les 6, 24 ou 168 dernières heures.
Donne le « plancher » de prix récent (intra‑jour, jour, semaine).​""",
    
    'rolling_min_24': """Minimum des prix observés sur les 6, 24 ou 168 dernières heures.
Donne le « plancher » de prix récent (intra‑jour, jour, semaine).​""",
    
    'rolling_min_168': """Minimum des prix observés sur les 6, 24 ou 168 dernières heures.
Donne le « plancher » de prix récent (intra‑jour, jour, semaine).​""",

    'rolling_max_6': """Maximum des prix observés sur les 6, 24 ou 168 dernières heures.
Donne le « plafond » de prix récent à ces horizons.​""",
    
    'rolling_max_24': """Maximum des prix observés sur les 6, 24 ou 168 dernières heures.
Donne le « plafond » de prix récent à ces horizons.​""",
    
    'rolling_max_168': """Maximum des prix observés sur les 6, 24 ou 168 dernières heures.
Donne le « plafond » de prix récent à ces horizons.​""",

    'price_delta': """Variation absolue horaire du prix : 
p
r
i
c
e
_
d
e
l
t
a
t
=
p
r
i
c
e
t
−
p
r
i
c
e
t
−
1
price_delta 
t
 =price 
t
 −price 
t−1
 .
Indique la magnitude de hausse/baisse entre deux heures consécutives.​""",

    'price_delta_pct': """Variation relative horaire du prix : 
p
r
i
c
e
_
d
e
l
t
a
_
p
c
t
t
=
p
r
i
c
e
t
−
p
r
i
c
e
t
−
1
∣
p
r
i
c
e
t
−
1
∣
price_delta_pct 
t
 = 
∣price 
t−1
 ∣
price 
t
 −price 
t−1
 
 .
Normalise la variation pour la rendre comparable à différents niveaux de prix.​""",

    'renewable_generation': """Puissance/énergie totale issue des renouvelables variables (souvent éolien + solaire, éventuellement hydro) à l'instant 
t
t.
Représente la part de production à coût marginal faible et fortement dépendante de la météo.​""",

    'total_generation': """Production électrique totale à l'instant 
t
t, toutes filières confondues.
Sert de référence pour calculer des ratios (mix, pénetration des ENR).​""",

    'renewable_ratio': """Part relative des renouvelables dans la production totale :
r
e
n
e
w
a
b
l
e
_
r
a
t
i
o
t
=
r
e
n
e
w
a
b
l
e
_
generationt_total/generation(t)
renewable_ratio(t) = total_generation(t)
 
renewable_generation(t).
Mesure la pénétration instantanée des ENR dans le mix.​""",

    'nuclear': """Production électrique d'origine nucléaire à l'instant (t)
t.
Source pilotable et peu émissive, clé pour le niveau de prix en France.​""",

    'nuclear_bin': """Version binaire de l'information nucléaire, par exemple :
1 si la production nucléaire est au‑dessus/dessous d'un seuil (ou « disponible »), 0 sinon.
Sert à encoder des régimes de fonctionnement (période de forte dispo vs maintenance).​""",

    'residual_load': """Charge résiduelle : demande totale moins génération renouvelable variable, typiquement
residual_load(t)=load(t)−renewable_generation(t)
residual_load(t) =load(t) −renewable_generation(t) .
Mesure la demande à couvrir par les moyens pilotables (nucléaire, thermique, stockage) et est fortement liée aux prix de marché.​""",

    'hydro_pumped': """Puissance liée aux stations de pompage-turbinage (pumped hydro) à l'instant t
t, souvent distinguée en mode pompage (consommation) et turbinage (production).
Reflète l'utilisation du stockage hydraulique, important pour lisser les prix et absorber les excès d'ENR.​""",
}
