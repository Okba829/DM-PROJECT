"""Logique d'inférence partagée par l'application Streamlit et les tests.

- ``load_or_train`` charge le modèle de l'application, ou l'entraîne s'il est absent ou
  s'il a été sauvegardé avec une autre version de scikit-learn.
- Sans fichiers NSL-KDD, l'application bascule en MODE DÉMO sur données synthétiques.
- ``IDS_DEMO=1`` force le mode démo ; ``IDS_MODEL_PATH`` change l'emplacement du modèle.
"""
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from .config import (CAT_COLS, FEATURE_COLS, MODELS_DIR, NUM_COLS, RANDOM_STATE,
                     TEST_FILE, TRAIN_FILE)
from .data_loader import load_nsl_kdd
from .preprocessing import build_preprocessor
from .synthetic import make_synthetic

MODEL_PATH = MODELS_DIR / "app_model.joblib"


def demo_forced():
    return os.environ.get("IDS_DEMO") == "1"


def model_path():
    return Path(os.environ.get("IDS_MODEL_PATH", MODEL_PATH))


def real_data_available(train_path=TRAIN_FILE):
    return (not demo_forced()) and Path(train_path).exists()


def build_model():
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31,
                               class_weight="balanced", random_state=RANDOM_STATE,
                               n_jobs=1, verbose=-1)),
    ])


def build_meta(df, source):
    return {
        "source": source,
        "classes": sorted(df["cls"].unique()),
        "choices": {c: sorted(df[c].unique()) for c in CAT_COLS},
        "defaults": {**{c: float(df[c].median()) for c in NUM_COLS},
                     **{c: df[c].mode().iloc[0] for c in CAT_COLS}},
        "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in NUM_COLS},
    }


def load_or_train(path=None, train_path=TRAIN_FILE, force=False):
    path = Path(path) if path else model_path()
    if path.exists() and not force:
        try:
            obj = joblib.load(path)
            if obj.get("sklearn_version") == sklearn.__version__:
                return obj["model"], obj["meta"]
        except Exception:
            pass                                  # modèle illisible : on réentraîne
    if real_data_available(train_path):
        df, source = load_nsl_kdd(train_path), "NSL-KDD"
    else:
        df, source = make_synthetic(6000, seed=0), "synthétique (mode démo)"
    model = build_model().fit(df[FEATURE_COLS], df["cls"])
    meta = build_meta(df, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta, "sklearn_version": sklearn.__version__},
                path, compress=3)
    return model, meta


def load_stream_source(test_path=TEST_FILE):
    """Événements qui seront « rejoués » en temps réel dans l'application."""
    if (not demo_forced()) and Path(test_path).exists():
        return load_nsl_kdd(test_path)
    return make_synthetic(2000, seed=1, shift=0.3)


def predict_one(model, meta, values):
    """Prédit UN événement. ``values`` complète/écrase les valeurs par défaut."""
    row = {c: values.get(c, meta["defaults"][c]) for c in FEATURE_COLS}
    X = pd.DataFrame([row], columns=FEATURE_COLS)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c])
    t0 = time.perf_counter()
    proba = model.predict_proba(X)[0]
    latency = (time.perf_counter() - t0) * 1000
    probs = dict(zip(model.classes_, proba))
    pred = max(probs, key=probs.get)
    return {
        "prediction": pred,
        "is_attack": pred != "normal",
        "attack_score": float(1 - probs.get("normal", 0.0)),
        "probabilities": probs,
        "latency_ms": latency,
    }
