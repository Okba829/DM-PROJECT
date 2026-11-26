"""Tests unitaires du pre-traitement (src/preprocessing.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.feature_selection import SelectKBest  # noqa: E402

from src.config import CAT_COLS, FEATURE_COLS, NUM_COLS  # noqa: E402
from src.preprocessing import build_preprocessor, build_selector  # noqa: E402


def _donnees(n=30, seed=0, protocoles=("tcp", "udp", "icmp"), decalage=0.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(NUM_COLS))) + decalage, columns=NUM_COLS)
    df["protocol_type"] = rng.choice(protocoles, n)
    df["service"] = rng.choice(["http", "ftp"], n)
    df["flag"] = rng.choice(["SF", "S0"], n)
    return df[FEATURE_COLS]


def test_build_selector_desactive():
    assert build_selector(None) is None
    assert build_selector(0) is None


def test_build_selector_k_attributs():
    sel = build_selector(7)
    assert isinstance(sel, SelectKBest)
    assert sel.k == 7


def test_preprocesseur_dimensions_sans_valeur_manquante():
    df = _donnees()
    X = build_preprocessor().fit_transform(df)
    n_one_hot = sum(df[c].nunique() for c in CAT_COLS)
    assert X.shape == (len(df), n_one_hot + len(NUM_COLS))
    assert not np.isnan(X).any()


def test_categorie_inconnue_ignoree_a_la_prediction():
    prep = build_preprocessor().fit(_donnees(protocoles=("tcp", "udp")))
    X = prep.transform(_donnees(n=5, seed=1, protocoles=("icmp",)))
    n_attendu = prep.transform(_donnees(n=5, seed=2, protocoles=("tcp",))).shape[1]
    assert X.shape[1] == n_attendu            # pas d'erreur, memes colonnes
    assert not np.isnan(X).any()


def test_standardisation_apprise_sur_le_train_uniquement():
    prep = build_preprocessor().fit(_donnees(seed=3))
    n_cat = len(prep.named_transformers_["cat"].get_feature_names_out())
    train_num = prep.transform(_donnees(seed=3))[:, n_cat:]
    autre_num = prep.transform(_donnees(seed=4, decalage=5.0))[:, n_cat:]
    assert abs(train_num.mean()) < 1e-6        # centre sur le train
    assert autre_num.mean() > 1.0              # pas re-centre sur les nouvelles donnees
