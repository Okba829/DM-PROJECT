"""Test de l'analyse exploratoire (src/eda.py) sur donnees synthetiques."""
import pandas as pd

from src import eda


def test_eda_produit_les_fichiers_attendus(donnees_nsl_kdd_synthetiques, tmp_path):
    train, _ = donnees_nsl_kdd_synthetiques
    eda.run(train_path=train, results_dir=tmp_path / "res")
    res = tmp_path / "res"
    for nom in ("eda_distribution_classes.csv", "eda_classes_protocoles.png",
                "eda_correlation.png"):
        assert (res / nom).exists() and (res / nom).stat().st_size > 0, nom


def test_eda_distribution_coherente(donnees_nsl_kdd_synthetiques, tmp_path):
    train, _ = donnees_nsl_kdd_synthetiques
    eda.run(train_path=train, results_dir=tmp_path)
    dist = pd.read_csv(tmp_path / "eda_distribution_classes.csv", index_col=0)
    assert {"normal", "DoS"} <= set(dist.index)
    assert abs(dist["pourcentage"].sum() - 100.0) < 0.1
    assert dist["effectif"].sum() <= 1500        # les doublons sont retires au chargement
