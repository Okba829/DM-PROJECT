"""Entraînement et comparaison des classifieurs supervisés.

Exemples :
    python -m src.train --target cls
    python -m src.train --target bin --models RandomForest DecisionTree --select-k 30
    python -m src.train --target cls --smote --sample 20000
"""
import argparse
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

from .config import (FEATURE_COLS, MODELS_DIR, NORMAL_LABEL, RANDOM_STATE,
                     RESULTS_DIR, TEST_FILE, TRAIN_FILE)
from .data_loader import load_nsl_kdd
from .evaluate import compute_metrics, plot_confusion_matrix
from .models import build_pipeline, get_classifiers


def plot_feature_importance(pipe, out_path, top=20):
    """Importance des attributs pour un modèle à base d'arbres."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    names = pipe.named_steps["prep"].get_feature_names_out()
    if "select" in pipe.named_steps:
        names = names[pipe.named_steps["select"].get_support()]
    imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)[:top]
    fig, ax = plt.subplots(figsize=(7, 6))
    imp[::-1].plot.barh(ax=ax)
    ax.set_title("Top attributs (importance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_experiment(train_path, test_path, target="cls", model_names=None, select_k=None,
                   use_smote=False, sample=None, results_dir=RESULTS_DIR,
                   models_dir=MODELS_DIR, save_models=True):
    results_dir, models_dir = Path(results_dir), Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train = load_nsl_kdd(train_path)
    test = load_nsl_kdd(test_path)
    if sample and sample < len(train):
        train = train.sample(sample, random_state=RANDOM_STATE)

    X_train, y_train = train[FEATURE_COLS], train[target]
    X_test, y_test = test[FEATURE_COLS], test[target]
    normal = NORMAL_LABEL[target]
    labels = sorted(y_train.unique(), key=str)

    classifiers = get_classifiers()
    names = model_names or list(classifiers)
    rows, best = [], (None, -1.0, None)

    for name in names:
        pipe = build_pipeline(classifiers[name], select_k=select_k, use_smote=use_smote)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0
        t0 = time.time()
        pred = pipe.predict(X_test)
        pred_time = time.time() - t0

        m = compute_metrics(y_test, pred, normal)
        m.update(model=name, fit_time_s=round(fit_time, 2), predict_time_s=round(pred_time, 2))
        rows.append(m)
        print(f"[{name}] F1 macro={m['f1_macro']:.4f}  DR={m['detection_rate']:.4f}  "
              f"FAR={m['false_alarm_rate']:.4f}  ({fit_time:.1f}s)")

        (results_dir / f"report_{target}_{name}.txt").write_text(
            classification_report(y_test, pred, digits=4, zero_division=0))
        plot_confusion_matrix(y_test, pred, labels, f"{name} ({target})",
                              results_dir / f"cm_{target}_{name}.png")
        if name == "RandomForest":
            plot_feature_importance(pipe, results_dir / f"importance_{target}_{name}.png")
        if m["f1_macro"] > best[1]:
            best = (name, m["f1_macro"], pipe)

    df = pd.DataFrame(rows).set_index("model").round(4)
    df.to_csv(results_dir / f"metrics_{target}.csv")
    print("\n", df.to_string())
    if save_models and best[0]:
        joblib.dump(best[2], models_dir / f"best_{target}.joblib")
        print(f"\nMeilleur modèle : {best[0]} (F1 macro = {best[1]:.4f})")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=str(TRAIN_FILE))
    p.add_argument("--test", default=str(TEST_FILE))
    p.add_argument("--target", choices=["cls", "bin"], default="cls")
    p.add_argument("--models", nargs="+", choices=list(get_classifiers()))
    p.add_argument("--select-k", type=int, default=None, help="nb d'attributs à garder")
    p.add_argument("--smote", action="store_true", help="sur-échantillonnage SMOTE")
    p.add_argument("--sample", type=int, default=None, help="sous-échantillonner le train")
    a = p.parse_args()
    run_experiment(a.train, a.test, a.target, a.models, a.select_k, a.smote, a.sample)


if __name__ == "__main__":
    main()
