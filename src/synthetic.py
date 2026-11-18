"""Données SYNTHÉTIQUES au format NSL-KDD (pour la démo de l'application et les tests).
Ne jamais utiliser pour les résultats du rapport."""
import numpy as np
import pandas as pd

from .config import ATTACK_CATEGORIES, ATTACK_TO_CLASS, COLS, NUM_COLS

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "private", "domain_u", "ecr_i"]
FLAGS = ["SF", "S0", "REJ", "RSTO"]
CLASS_PROBS = {"normal": 0.50, "DoS": 0.30, "Probe": 0.12, "R2L": 0.06, "U2R": 0.02}


def make_synthetic(n, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    classes = rng.choice(list(CLASS_PROBS), size=n, p=list(CLASS_PROBS.values()))
    X = rng.normal(size=(n, len(NUM_COLS)))
    for i, c in enumerate(CLASS_PROBS):
        X[classes == c, i * 3:i * 3 + 3] += 2.5 + shift
    df = pd.DataFrame(X, columns=NUM_COLS)
    df["protocol_type"] = rng.choice(PROTOCOLS, n)
    df["service"] = rng.choice(SERVICES, n)
    df["flag"] = rng.choice(FLAGS, n)
    df["label"] = [
        "normal" if c == "normal" else rng.choice(ATTACK_CATEGORIES[c]) for c in classes]
    df = df[[c for c in COLS if c != "difficulty"]]
    df["cls"] = df["label"].map(ATTACK_TO_CLASS)
    df["bin"] = (df["cls"] != "normal").astype(int)
    return df
