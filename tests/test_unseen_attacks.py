"""Test de l'experience H3 : attaques absentes de l'entrainement (src/unseen_attacks.py)."""
import pandas as pd

from src import unseen_attacks


def test_unseen_attacks_tableau_et_csv(donnees_nsl_kdd_synthetiques, tmp_path):
    train, test = donnees_nsl_kdd_synthetiques
    out = unseen_attacks.run(train, test, results_dir=tmp_path)
    assert list(out.columns) == ["famille_exclue", "n_test", "taux_detection"]
    assert set(out["famille_exclue"]) <= {"DoS", "Probe", "R2L", "U2R"}
    assert len(out) >= 1
    assert out["n_test"].gt(0).all()
    assert out["taux_detection"].between(0.0, 1.0).all()
    ecrit = pd.read_csv(tmp_path / "unseen_attacks.csv")
    assert list(ecrit["famille_exclue"]) == list(out["famille_exclue"])
