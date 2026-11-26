# Détection d'attaques réseau par fouille de données

Projet de Data Mining — 2026/2027

Chaîne complète de détection d'intrusion sur le jeu **NSL-KDD** : exploration, prétraitement, sélection d'attributs, comparaison de classifieurs (supervisé), méthodes d'ensemble, détection d'anomalies (non supervisé), test sur attaques inconnues, benchmark (latence, taille, mémoire) et **application interactive Streamlit**.

![Application Streamlit : test manuel d'une connexion réseau](docs/figures/app_streamlit.png)

## Démarrage rapide

```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

L'application s'ouvre sur `http://localhost:8501`. Sans les fichiers NSL-KDD, elle démarre en **mode démo** (données synthétiques). Détails dans la section [Application interactive](#application-interactive-streamlit) ; en cas de problème d'installation, voir [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

## Structure

```
.
├── README.md
├── requirements.txt
├── app/streamlit_app.py     # application Streamlit (test manuel, simulation temps réel, benchmark)
├── data/                    # NSL-KDD à placer ici (voir data/README.md)
├── docs/
│   ├── rapport.md           # rapport complet (figures, discussion)
│   ├── Rapport_Projet.pdf   # version PDF du rapport
│   ├── UTILISATION.md       # guide d'utilisation
│   ├── INSTALLATION.md      # installation et dépannage
│   └── figures/             # figures du rapport
├── scripts/make_synthetic_data.py   # données SYNTHÉTIQUES pour tester le code
├── src/
│   ├── config.py            # chemins, colonnes, taxonomie des attaques
│   ├── data_loader.py       # lecture + étiquettes (cls multiclasse, bin binaire)
│   ├── preprocessing.py     # one-hot, standardisation, SelectKBest
│   ├── models.py            # classifieurs + pipelines (option SMOTE)
│   ├── ensembles.py         # méthodes d'ensemble : RF, AdaBoost, XGBoost, LightGBM, Voting
│   ├── evaluate.py          # précision, rappel, F1, DR, FAR, matrices de confusion
│   ├── train.py             # comparaison des modèles supervisés
│   ├── benchmark.py         # benchmark multi-familles (latence, taille, mémoire)
│   ├── anomaly.py           # Isolation Forest (appris sur le normal uniquement)
│   ├── unseen_attacks.py    # test de généralisation (famille exclue du train)
│   ├── eda.py               # analyse exploratoire
│   ├── inference.py         # logique d'inférence de l'application (chargement du modèle, mode démo)
│   └── synthetic.py         # générateur de données synthétiques (mode démo et tests)
├── tests/                   # tests automatiques (pytest)
├── results/                 # figures et CSV générés
└── models/                  # modèles sauvegardés (joblib)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate           # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
```

Sur une connexion lente ou un disque presque plein, suivre [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

## Application interactive (Streamlit)

```bash
python -m streamlit run app/streamlit_app.py
```

L'application propose trois onglets :

- **Test manuel** : saisir ou charger une connexion réseau (exemples DoS, Probe, R2L, U2R, normal) et obtenir la classe prédite, le score d'attaque, les probabilités par classe et la latence de prédiction ;
- **Simulation temps réel** : rejouer un flux d'événements et observer les détections ;
- **Benchmark des modèles** : comparer les modèles évalués dans le projet.

Le modèle de l'application (LightGBM) est chargé depuis `models/app_model.joblib`. Il est entraîné automatiquement s'il est absent, illisible ou sauvegardé avec une autre version de scikit-learn.

**Mode démo.** Si `KDDTrain+.txt` est absent de `data/`, le modèle est entraîné sur des données synthétiques et un bandeau l'indique en haut de la page. Deux variables d'environnement permettent d'ajuster ce comportement :

| Variable | Effet |
|---|---|
| `IDS_DEMO=1` | force le mode démo, même si les vrais fichiers sont présents |
| `IDS_MODEL_PATH` | change l'emplacement du fichier du modèle |

Exemple : `set IDS_DEMO=1` (invite de commandes Windows) ou `$env:IDS_DEMO="1"` (PowerShell), puis lancer l'application.

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
python -m pytest tests/ -q                         # tests
```

Le benchmark et l'application sont décrits dans [`docs/UTILISATION.md`](docs/UTILISATION.md).

Sans les vrais fichiers, pour vérifier que tout fonctionne :

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python -m src.train --train data/synthetic/KDDTrain+.txt --test data/synthetic/KDDTest+.txt
```

Les résultats synthétiques ne doivent **jamais** figurer dans le rapport.

## Tests

```bash
python -m pytest tests/ -q
```

Les tests utilisent uniquement des données synthétiques (fixture partagée dans `tests/conftest.py`). Ils couvrent le chargement et la taxonomie (`test_data_loader.py`), le prétraitement (`test_preprocessing.py`), les métriques (`test_evaluate.py`), l'inférence (`test_inference.py`), l'analyse exploratoire (`test_eda.py`), la détection d'anomalies (`test_anomaly.py`), les attaques absentes de l'entraînement (`test_unseen_attacks.py`), les méthodes d'ensemble et l'application (`test_ensembles.py`) et la chaîne d'entraînement complète (`test_pipeline.py`). Sur un ordinateur peu puissant, la suite complète peut durer quelques minutes ; pour un seul fichier : `python -m pytest tests/test_evaluate.py -q`.

## Sorties (dans `results/`)

- `metrics_<cible>.csv` : tableau comparatif (accuracy, précision/rappel/F1 macro, taux de détection, taux de fausses alarmes, temps)
- `cm_<cible>_<modèle>.png` : matrices de confusion
- `report_<cible>_<modèle>.txt` : rapports de classification détaillés
- `importance_<cible>_RandomForest.png` : attributs les plus importants
- `benchmark_<cible>.csv` et `benchmark_<cible>.png` : benchmark multi-familles
- `metrics_isolation_forest.csv`, `unseen_attacks.csv`, `eda_*.png`

## Résultats principaux (NSL-KDD, jeu de test KDDTest+)

- **5 classes** : meilleur F1 macro = régression logistique (0,606, exactitude 0,791).
- **Binaire** : kNN, arbre et Random Forest, F1 macro de 0,77 à 0,78, fausses alarmes de 2,7 à 3,7 %.
- **Isolation Forest** (sans étiquettes d'attaque) : F1 macro 0,781, AUC 0,939.
- **Attaques inconnues** : taux de détection de 57 % (DoS), 37 % (Probe), 6 % (U2R), 0,1 % (R2L).

Rapport complet avec figures et discussion : [`docs/rapport.md`](docs/rapport.md) (PDF : [`docs/Rapport_Projet.pdf`](docs/Rapport_Projet.pdf)). Comparaison des configurations (base, sélection, SMOTE) : `results/comparaison_configurations_cls.csv`.

## Travail en équipe (Git)

Faire des commits **réguliers et petits**, avec un message clair. Pour un projet de groupe : `git pull` avant chaque `git push`, et chaque membre commit depuis son propre compte (vérifier `git config user.email` : l'adresse doit être celle du compte GitHub).
