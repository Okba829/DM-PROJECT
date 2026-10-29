# Détection d'attaques réseau par fouille de données

Projet de Data Mining — 2026/2027

Chaîne complète de détection d'intrusion sur le jeu **NSL-KDD** : exploration, prétraitement, sélection d'attributs, comparaison de classifieurs (supervisé), détection d'anomalies (non supervisé), test sur attaques inconnues.

## Structure

```
.
├── README.md
├── requirements.txt
├── data/                  # NSL-KDD à placer ici (voir data/README.md)
├── docs/rapport.md        # trame du rapport (à compléter avec vos résultats)
├── scripts/make_synthetic_data.py   # données SYNTHÉTIQUES pour tester le code
├── src/
│   ├── config.py          # chemins, colonnes, taxonomie des attaques
│   ├── data_loader.py     # lecture + étiquettes (cls multiclasse, bin binaire)
│   ├── preprocessing.py   # one-hot, standardisation, SelectKBest
│   ├── models.py          # classifieurs + pipelines (option SMOTE)
│   ├── evaluate.py        # précision, rappel, F1, DR, FAR, matrices de confusion
│   ├── train.py           # comparaison des modèles supervisés
│   ├── anomaly.py         # Isolation Forest (appris sur le normal uniquement)
│   ├── unseen_attacks.py  # test de généralisation (famille exclue du train)
│   └── eda.py             # analyse exploratoire
├── tests/test_pipeline.py # tests automatiques (pytest)
├── results/               # figures et CSV générés
└── models/                # meilleur modèle sauvegardé (joblib)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Données

Télécharger **NSL-KDD** (UNB Canadian Institute for Cybersecurity) et placer `KDDTrain+.txt` et `KDDTest+.txt` dans `data/`. Détails dans `data/README.md`.

## Utilisation

```bash
python -m src.eda                                  # exploration
python -m src.train --target cls                   # 5 classes (normal, DoS, Probe, R2L, U2R)
python -m src.train --target bin                   # normal / attaque
python -m src.train --target cls --select-k 30     # avec sélection d'attributs
python -m src.train --target cls --smote           # avec SMOTE
python -m src.train --models RandomForest kNN      # sous-ensemble de modèles
python -m src.anomaly                              # Isolation Forest
python -m src.unseen_attacks                       # attaques inconnues (H3)
pytest -q                                          # tests
```

Sans les vrais fichiers, pour vérifier que tout fonctionne :

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python -m src.train --train data/synthetic/KDDTrain+.txt --test data/synthetic/KDDTest+.txt
```

Les résultats synthétiques ne doivent **jamais** figurer dans le rapport.

## Sorties (dans `results/`)

- `metrics_<cible>.csv` : tableau comparatif (accuracy, précision/rappel/F1 macro, taux de détection, taux de fausses alarmes, temps)
- `cm_<cible>_<modèle>.png` : matrices de confusion
- `report_<cible>_<modèle>.txt` : rapports de classification détaillés
- `importance_<cible>_RandomForest.png` : attributs les plus importants
- `metrics_isolation_forest.csv`, `unseen_attacks.csv`, `eda_*.png`

## Guide Git (commits à faire au fil du travail)

Initialisation et premier envoi :

```bash
git init
git branch -M main
git remote add origin https://github.com/<votre-compte>/<depot>.git
```

Faire des commits **réguliers et petits**, au rythme de votre avancement, par exemple :

```bash
git add README.md requirements.txt .gitignore
git commit -m "Initialisation du projet : structure et dépendances"

git add src/config.py src/data_loader.py
git commit -m "Ajout du chargement NSL-KDD et de la taxonomie des attaques"

git add src/eda.py
git commit -m "Ajout de l'analyse exploratoire"

git add src/preprocessing.py src/models.py
git commit -m "Prétraitement et pipelines de classification"

git add src/evaluate.py src/train.py
git commit -m "Entraînement et évaluation des modèles supervisés"

git add src/anomaly.py src/unseen_attacks.py
git commit -m "Détection d'anomalies et test sur attaques inconnues"

git add tests scripts
git commit -m "Tests automatiques et générateur de données synthétiques"

git add docs
git commit -m "Trame du rapport"

git push -u origin main
```

Pour un projet de groupe : `git pull` avant chaque `git push`, et chaque membre commit depuis son propre compte.
