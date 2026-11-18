# Utilisation des modules ajoutés

## Installation des nouvelles dépendances
```bash
python -m pip install -r requirements.txt     # ajoute xgboost, lightgbm, streamlit
```

## Benchmark multi-familles (avec temps d'inférence, taille et mémoire)
```bash
python -m src.benchmark --target bin          # normal / attaque
python -m src.benchmark --target cls          # 5 classes
python -m src.benchmark --target bin --models DecisionTree XGBoost LightGBM Voting
python -m src.benchmark --target bin --max-latency-ms 10 --max-size-mb 5
```
Sorties : `results/benchmark_<cible>.csv` et `results/benchmark_<cible>.png`.
Modèles : Naïve Bayes, régression logistique, SVM linéaire, arbre de décision, kNN,
Random Forest, AdaBoost, XGBoost, LightGBM et VotingClassifier (régression logistique +
Random Forest + LightGBM, vote pondéré sur les probabilités).
Colonnes de contraintes IoT : `latency_ms_single` (une prédiction isolée, 1 thread),
`model_size_kb` (modèle sérialisé), `peak_mem_mb` (pic mémoire Python à l'inférence),
`compatible_passerelle` (latence ≤ 10 ms et taille ≤ 5 Mo par défaut).

## Application Streamlit
```bash
streamlit run app/streamlit_app.py
```
- Onglet **Test manuel** : saisie des attributs d'une connexion ou chargement d'un exemple de chaque classe.
- Onglet **Simulation temps réel** : rejeu d'événements du jeu de test un par un, avec alertes et indicateurs.
- Onglet **Benchmark** : affiche les fichiers `results/benchmark_*`.

Au premier lancement, le modèle LightGBM est entraîné (10 à 30 s) et sauvegardé dans `models/app_model.joblib`
(ignoré par Git). Sans fichiers NSL-KDD dans `data/`, l'application passe en mode démo sur données synthétiques.
Variables d'environnement : `IDS_DEMO=1` force le mode démo, `IDS_MODEL_PATH` change l'emplacement du modèle.

## Tests
```bash
python -m pytest -q
```

## À ajouter dans le README.md du dépôt
```markdown
## Nouveautés
- Benchmark de 10 modèles (dont AdaBoost, XGBoost, LightGBM, Random Forest, VotingClassifier) avec latence, taille et mémoire : `python -m src.benchmark`
- Application Streamlit (test manuel + simulation temps réel) : `streamlit run app/streamlit_app.py`
- Rapport final : [`docs/Rapport_Projet.pdf`](docs/Rapport_Projet.pdf)
```
