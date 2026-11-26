"""Tests unitaires de la logique d'inference (src/inference.py), sans entrainer LightGBM."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import inference  # noqa: E402
from src.config import CAT_COLS, FEATURE_COLS, NUM_COLS  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402


# ----------------------------------------------------------------- outils
def _petit_df():
    df = pd.DataFrame({c: [0.0, 2.0, 10.0] for c in NUM_COLS})
    df["protocol_type"] = ["udp", "tcp", "tcp"]
    df["service"] = ["http", "http", "ftp"]
    df["flag"] = ["SF", "SF", "S0"]
    df["cls"] = ["DoS", "normal", "DoS"]
    return df


class _FauxModele:
    """Modele factice : renvoie des probabilites fixes et memorise l'entree recue."""

    def __init__(self, classes, proba):
        self.classes_ = np.array(classes)
        self._proba = np.array([proba])
        self.entree = None

    def predict_proba(self, X):
        self.entree = X
        return self._proba


def _modele_rapide():
    return Pipeline([("prep", build_preprocessor()),
                     ("clf", DecisionTreeClassifier(max_depth=3, random_state=0))])


# ----------------------------------------------------------------- variables d'environnement
def test_demo_forcee_par_variable(monkeypatch):
    monkeypatch.delenv("IDS_DEMO", raising=False)
    assert inference.demo_forced() is False
    monkeypatch.setenv("IDS_DEMO", "0")
    assert inference.demo_forced() is False
    monkeypatch.setenv("IDS_DEMO", "1")
    assert inference.demo_forced() is True


def test_chemin_du_modele_modifiable(monkeypatch, tmp_path):
    monkeypatch.delenv("IDS_MODEL_PATH", raising=False)
    assert inference.model_path() == inference.MODEL_PATH
    monkeypatch.setenv("IDS_MODEL_PATH", str(tmp_path / "autre.joblib"))
    assert inference.model_path() == tmp_path / "autre.joblib"


def test_donnees_reelles_disponibles(monkeypatch, tmp_path):
    fichier = tmp_path / "KDDTrain+.txt"
    monkeypatch.delenv("IDS_DEMO", raising=False)
    assert inference.real_data_available(fichier) is False      # fichier absent
    fichier.write_text("x", encoding="utf-8")
    assert inference.real_data_available(fichier) is True       # fichier present
    monkeypatch.setenv("IDS_DEMO", "1")
    assert inference.real_data_available(fichier) is False      # mode demo force


# ----------------------------------------------------------------- build_meta
def test_build_meta_contenu():
    meta = inference.build_meta(_petit_df(), "test")
    assert meta["source"] == "test"
    assert meta["classes"] == ["DoS", "normal"]
    assert meta["choices"]["protocol_type"] == ["tcp", "udp"]
    assert meta["defaults"]["duration"] == 2.0               # mediane
    assert meta["defaults"]["protocol_type"] == "tcp"        # mode
    assert meta["ranges"]["duration"] == (0.0, 10.0)
    assert set(meta["defaults"]) == set(FEATURE_COLS)


# ----------------------------------------------------------------- predict_one
def test_predict_one_attaque():
    meta = inference.build_meta(_petit_df(), "test")
    res = inference.predict_one(_FauxModele(["DoS", "normal"], [0.8, 0.2]), meta, {})
    assert res["prediction"] == "DoS"
    assert res["is_attack"]
    assert abs(res["attack_score"] - 0.8) < 1e-9
    assert res["latency_ms"] >= 0
    assert set(res["probabilities"]) == {"DoS", "normal"}


def test_predict_one_normal():
    meta = inference.build_meta(_petit_df(), "test")
    res = inference.predict_one(_FauxModele(["DoS", "normal"], [0.1, 0.9]), meta, {})
    assert res["prediction"] == "normal"
    assert not res["is_attack"]
    assert abs(res["attack_score"] - 0.1) < 1e-9


def test_predict_one_valeurs_par_defaut_et_surcharge():
    meta = inference.build_meta(_petit_df(), "test")
    modele = _FauxModele(["DoS", "normal"], [0.5, 0.5])
    inference.predict_one(modele, meta, {"protocol_type": "udp", "duration": "5"})
    X = modele.entree
    assert list(X.columns) == FEATURE_COLS                   # ordre des colonnes respecte
    assert X.loc[0, "protocol_type"] == "udp"                # valeur fournie
    assert X.loc[0, "duration"] == 5.0                       # chaine convertie en nombre
    assert X.loc[0, "service"] == "http"                     # valeur par defaut (mode)
    assert X.loc[0, "src_bytes"] == 2.0                      # valeur par defaut (mediane)


# ----------------------------------------------------------------- flux et rechargement
def test_flux_en_mode_demo(monkeypatch, tmp_path):
    monkeypatch.setenv("IDS_DEMO", "1")
    df = inference.load_stream_source(tmp_path / "absent.txt")
    assert len(df) > 0
    assert set(FEATURE_COLS) <= set(df.columns)


def test_modele_valide_recharge_sans_reentrainer(monkeypatch, tmp_path):
    monkeypatch.setenv("IDS_DEMO", "1")
    chemin = tmp_path / "m.joblib"
    meta = {"source": "sauvegarde"}
    joblib.dump({"model": "SAUVEGARDE", "meta": meta,
                 "sklearn_version": sklearn.__version__}, chemin)
    assert inference.load_or_train(path=chemin) == ("SAUVEGARDE", meta)


def test_modele_illisible_ou_autre_version_est_reentraine(monkeypatch, tmp_path):
    monkeypatch.setenv("IDS_DEMO", "1")
    monkeypatch.setattr(inference, "build_model", _modele_rapide)   # evite LightGBM (lent)
    # 1) fichier corrompu
    chemin = tmp_path / "corrompu.joblib"
    chemin.write_bytes(b"ceci n'est pas un modele")
    model, meta = inference.load_or_train(path=chemin)
    assert meta["source"].startswith("synth")
    assert joblib.load(chemin)["sklearn_version"] == sklearn.__version__
    # 2) modele sauvegarde avec une autre version de scikit-learn
    autre = tmp_path / "ancien.joblib"
    joblib.dump({"model": "ANCIEN", "meta": {}, "sklearn_version": "0.0.1"}, autre)
    model, _ = inference.load_or_train(path=autre)
    assert model != "ANCIEN"
    # 3) force=True ignore le fichier existant
    joblib.dump({"model": "SAUVEGARDE", "meta": {},
                 "sklearn_version": sklearn.__version__}, autre)
    model, _ = inference.load_or_train(path=autre, force=True)
    assert model != "SAUVEGARDE"
