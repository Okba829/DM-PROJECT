"""Génère de petits fichiers AU FORMAT NSL-KDD (données SYNTHÉTIQUES) pour tester le code
sans télécharger le vrai jeu de données. Ne PAS utiliser pour les résultats du rapport.

    python scripts/make_synthetic_data.py --out data/synthetic
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ATTACK_CATEGORIES, CAT_COLS, COLS, NUM_COLS  # noqa: E402

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "private", "domain_u", "ecr_i"]
FLAGS = ["SF", "S0", "REJ", "RSTO"]
CLASS_PROBS = {"normal": 0.50, "DoS": 0.30, "Probe": 0.12, "R2L": 0.06, "U2R": 0.02}


def make(n, rng, shift_unseen=False):
    classes = rng.choice(list(CLASS_PROBS), size=n, p=list(CLASS_PROBS.values()))
    cls_index = {c: i for i, c in enumerate(CLASS_PROBS)}
    X = rng.normal(size=(n, len(NUM_COLS)))
    for c, i in cls_index.items():                      # décalage propre à chaque classe
        mask = classes == c
        X[mask, i * 3:(i * 3) + 3] += 2.5 + (0.5 if shift_unseen else 0)
    df = pd.DataFrame(X, columns=NUM_COLS)
    df["protocol_type"] = rng.choice(PROTOCOLS, n)
    df["service"] = rng.choice(SERVICES, n)
    df["flag"] = rng.choice(FLAGS, n)
    df["label"] = [
        "normal" if c == "normal" else rng.choice(ATTACK_CATEGORIES[c]) for c in classes
    ]
    df["difficulty"] = rng.integers(1, 21, n)
    return df[COLS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    make(a.n_train, rng).to_csv(out / "KDDTrain+.txt", header=False, index=False)
    make(a.n_test, rng, shift_unseen=True).to_csv(out / "KDDTest+.txt", header=False, index=False)
    print("Fichiers synthétiques écrits dans", out)
