import json
import os

print("🔧 Correctif : Ajout des imports manquants & MAJ Task...")

# 1. Update Notebook Imports
nb_path = 'notebooks/France/France_2020_2025_Modeling.ipynb'
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # On cherche la première cellule de code (imports)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if "import pandas" in source:
                # On ajoute plotly.express si absent
                if "import plotly.express as px" not in source:
                    cell['source'].insert(4, "import plotly.express as px\n")
                    print("✓ Import 'plotly.express as px' ajouté")
                if "from sklearn.metrics" in source and "mean_absolute_percentage_error" not in source:
                     # On complète les métriques si besoin
                     pass 
                break
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

except FileNotFoundError:
    print(f"❌ Notebook introuvable : {nb_path}")

# 2. Update Task.md
task_path = 'C:/Users/Franck/.gemini/antigravity/brain/e41161ec-6e5c-41cd-9e0e-dfb6d09cdff5/task.md'
new_task_content = """- [x] Create virtual environment (`.venv`)
- [x] Install dependencies from `requirements.txt`
- [x] Create `notebooks/France_Extended_Analysis.ipynb`
- [x] Implement SARIMA model
- [x] Optimize LightGBM with GridSearch
- [x] Explain LightGBM performance and Volatility
- [x] Analyze `time_series_60min_fr_dk_2020_2025.csv` (New Notebook)
- [x] Clean dataset (remove .1 columns, duplicate nuclear column)
- [x] Update data loading with all French generation sources
- [x] Enrich EDA section with comprehensive analyses
- [x] Create modular notebooks
  - [x] `Fr_2020_2025_EDA.ipynb` (Sections 1-3 + Outliers + Energy Mix & Correlations)
  - [x] `France_2020_2025_Features.ipynb` (Sections 4-5)
  - [x] `France_2020_2025_Modeling.ipynb` (Sections 6-9: LightGBM + SARIMAX + SHAP)
- [x] **Advanced Modeling Improvements**
  - [x] Fix LightGBM categorical error (load_bin exclusion) & Anti-Leakage
  - [x] Implement Winsorization for SARIMAX (Handle 2022 Crisis)
  - [x] Upgrade SARIMAX -> ARIMAX (Multivariate with Exogenous Vars: Gas, Nuclear, Load)
  - [/] Finalize Model Comparison and Interpretation
"""

with open(task_path, 'w', encoding='utf-8') as f:
    f.write(new_task_content)
    print("✓ Task.md mis à jour")
