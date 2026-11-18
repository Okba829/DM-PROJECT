"""Benchmark complet : plusieurs familles d'algorithmes + méthodes d'ensemble,
avec les mesures utiles pour un déploiement sur passerelle IoT contrainte :
temps d'inférence (par échantillon), taille du modèle, pic de mémoire à l'inférence.

    python -m src.benchmark --target bin
    python -m src.benchmark --target cls --models DecisionTree XGBoost LightGBM Voting
    python -m src.benchmark --target bin --sample 30000 --n-latency 100
"""
import argparse
import pickle
import time
import tracemalloc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

from .config import FEATURE_COLS, RANDOM_STATE, RESULTS_DIR, TEST_FILE, TRAIN_FILE
from .data_loader import load_nsl_kdd
from .ensembles import get_ensemble_classifiers
from .evaluate import compute_metrics
from .models import build_pipeline, get_classifiers

FAMILIES = {
    "NaiveBayes": "Probabiliste",
    "LogisticRegression": "Linéaire",
    "LinearSVM": "Linéaire",
    "DecisionTree": "Arbre",
    "kNN": "À base d'instances",
    "RandomForest": "Ensemble (bagging)",
    "AdaBoost": "Ensemble (boosting)",
    "XGBoost": "Ensemble (boosting)",
    "LightGBM": "Ensemble (boosting)",
    "Voting": "Ensemble (vote)",
}


def all_classifiers():
    clfs = get_classifiers()
    clfs.update(get_ensemble_classifiers())
    return clfs


def set_single_thread(clf):
    """À l'inférence, une passerelle IoT n'a pas de multi-cœur : on force 1 thread."""
    if hasattr(clf, "n_jobs"):
        clf.n_jobs = 1
    if isinstance(clf, VotingClassifier):
        for est in clf.estimators_:
            set_single_thread(est)


def latency_ms(pipe, X, n):
    """Latence moyenne d'UNE prédiction isolée (cas d'usage temps réel), en ms."""
    rows = [X.iloc[[i]] for i in range(min(n, len(X)))]
    pipe.predict(rows[0])                       # échauffement
    t0 = time.perf_counter()
    for r in rows:
        pipe.predict(r)
    return (time.perf_counter() - t0) / len(rows) * 1000


def size_kb(pipe):
    return len(pickle.dumps(pipe, protocol=pickle.HIGHEST_PROTOCOL)) / 1024


def peak_mem_mb(pipe, X, n=1000):
    """Pic de mémoire Python alloué pendant l'inférence d'un lot (tracemalloc).
    Limite : les allocations faites par les bibliothèques C (LightGBM, XGBoost) ne sont
    pas comptées ; la taille du modèle sérialisé complète cette mesure."""
    tracemalloc.start()
    pipe.predict(X.iloc[:n])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def run(train_path=TRAIN_FILE, test_path=TEST_FILE, target="bin", model_names=None,
        sample=None, n_latency=200, max_latency_ms=10.0, max_size_mb=5.0,
        results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_nsl_kdd(train_path), load_nsl_kdd(test_path)
    if sample and sample < len(train):
        train = train.sample(sample, random_state=RANDOM_STATE)

    le = LabelEncoder().fit(train[target])
    y_train, y_test = le.transform(train[target]), le.transform(test[target])
    normal = int(le.transform([0 if target == "bin" else "normal"])[0])
    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]

    clfs = all_classifiers()
    names = model_names or list(clfs)
    rows = []
    for name in names:
        pipe = build_pipeline(clfs[name])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        set_single_thread(pipe.named_steps["clf"])

        t0 = time.perf_counter()
        pred = pipe.predict(X_test)
        batch_ms = (time.perf_counter() - t0) / len(X_test) * 1000

        m = compute_metrics(y_test, pred, normal)
        m.update(
            model=name, famille=FAMILIES.get(name, ""), fit_time_s=round(fit_s, 2),
            batch_ms_per_sample=round(batch_ms, 4),
            latency_ms_single=round(latency_ms(pipe, X_test, n_latency), 3),
            model_size_kb=round(size_kb(pipe), 1),
            peak_mem_mb=round(peak_mem_mb(pipe, X_test), 2),
        )
        m["compatible_passerelle"] = bool(
            m["latency_ms_single"] <= max_latency_ms and m["model_size_kb"] / 1024 <= max_size_mb)
        rows.append(m)
        print(f"[{name}] F1 macro={m['f1_macro']:.4f} DR={m['detection_rate']:.4f} "
              f"FAR={m['false_alarm_rate']:.4f} latence={m['latency_ms_single']:.2f} ms "
              f"taille={m['model_size_kb']/1024:.2f} Mo")

    cols = ["famille", "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "detection_rate", "false_alarm_rate", "fit_time_s", "batch_ms_per_sample",
            "latency_ms_single", "model_size_kb", "peak_mem_mb", "compatible_passerelle"]
    df = pd.DataFrame(rows).set_index("model")[cols].round(4)
    df.to_csv(results_dir / f"benchmark_{target}.csv")
    plot(df, target, results_dir / f"benchmark_{target}.png", max_latency_ms)
    print("\n", df.to_string())
    return df


def plot(df, target, out_path, max_latency_ms):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = df.sort_values("f1_macro")
    axes[0].barh(order.index, order["f1_macro"], color="#2a6fb0")
    axes[0].set_xlabel("F1 macro")
    axes[0].set_title(f"Qualité de détection ({target})")
    for i, v in enumerate(order["f1_macro"]):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    axes[1].scatter(df["latency_ms_single"], df["f1_macro"], s=60, color="#c0392b")
    for name, r in df.iterrows():
        axes[1].annotate(name, (r["latency_ms_single"], r["f1_macro"]),
                         textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[1].axvline(max_latency_ms, ls="--", color="gray")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Latence d'une prédiction isolée (ms, échelle log)")
    axes[1].set_ylabel("F1 macro")
    axes[1].set_title("Compromis précision / temps réel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(TRAIN_FILE))
    ap.add_argument("--test", default=str(TEST_FILE))
    ap.add_argument("--target", choices=["cls", "bin"], default="bin")
    ap.add_argument("--models", nargs="+", choices=list(all_classifiers()))
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--n-latency", type=int, default=200)
    ap.add_argument("--max-latency-ms", type=float, default=10.0)
    ap.add_argument("--max-size-mb", type=float, default=5.0)
    a = ap.parse_args()
    run(a.train, a.test, a.target, a.models, a.sample, a.n_latency,
        a.max_latency_ms, a.max_size_mb)
