"""Hypothèse H3 : que devient le taux de détection sur une famille d'attaques
ABSENTE de l'entraînement ?

On retire une famille (DoS, Probe, R2L ou U2R) du train, on apprend un détecteur
binaire (Random Forest), puis on mesure le taux de détection sur cette famille dans le test.

    python -m src.unseen_attacks
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .config import FEATURE_COLS, RANDOM_STATE, RESULTS_DIR, TEST_FILE, TRAIN_FILE
from .data_loader import load_nsl_kdd
from .evaluate import detection_rate
from .preprocessing import build_preprocessor


def run(train_path=TRAIN_FILE, test_path=TEST_FILE, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    train, test = load_nsl_kdd(train_path), load_nsl_kdd(test_path)
    rows = []
    for family in ["DoS", "Probe", "R2L", "U2R"]:
        reduced = train[train["cls"] != family]
        pipe = Pipeline([
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                           n_jobs=-1, random_state=RANDOM_STATE)),
        ]).fit(reduced[FEATURE_COLS], reduced["bin"])
        subset = test[test["cls"] == family]
        if subset.empty:
            continue
        pred = pipe.predict(subset[FEATURE_COLS])
        rows.append({"famille_exclue": family, "n_test": len(subset),
                     "taux_detection": round(detection_rate(subset["bin"], pred, 0), 4)})
        print(rows[-1])
    out = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(results_dir / "unseen_attacks.csv", index=False)
    return out


if __name__ == "__main__":
    run()
