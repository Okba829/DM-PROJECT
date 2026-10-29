"""Détection d'anomalies : Isolation Forest appris sur le trafic NORMAL uniquement.

    python -m src.anomaly
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

from .config import FEATURE_COLS, RANDOM_STATE, RESULTS_DIR, TEST_FILE, TRAIN_FILE
from .data_loader import load_nsl_kdd
from .evaluate import compute_metrics, plot_confusion_matrix
from .preprocessing import build_preprocessor


def run(train_path=TRAIN_FILE, test_path=TEST_FILE, contamination=0.05, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    train, test = load_nsl_kdd(train_path), load_nsl_kdd(test_path)
    prep = build_preprocessor()
    X_norm = prep.fit_transform(train.loc[train["bin"] == 0, FEATURE_COLS])
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=RANDOM_STATE, n_jobs=-1).fit(X_norm)

    X_test = prep.transform(test[FEATURE_COLS])
    pred = (iso.predict(X_test) == -1).astype(int)       # -1 = anomalie
    auc = roc_auc_score(test["bin"], -iso.score_samples(X_test))
    m = compute_metrics(test["bin"], pred, 0)
    m["auc"] = auc
    print(pd.Series(m).round(4).to_string())
    print(classification_report(test["bin"], pred, digits=4, zero_division=0))
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(m).round(4).to_csv(results_dir / "metrics_isolation_forest.csv", header=["valeur"])
    plot_confusion_matrix(test["bin"], pred, [0, 1], "Isolation Forest",
                          results_dir / "cm_isolation_forest.png")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contamination", type=float, default=0.05)
    run(contamination=ap.parse_args().contamination)
